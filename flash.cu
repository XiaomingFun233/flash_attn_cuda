#include "../coding/common.h"
#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>  
#include <time.h>
#include <math.h>
float* h_attention(const float* Q, const float* K, const float* V,
                 int n, int m, int d_k, int d_v) {
    // 分配注意力分数矩阵内存 (n x m)
    float* scores = (float*)malloc(n * m * sizeof(float));
    if (!scores) return NULL;

    // 计算QK^T并存储到scores矩阵中
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < d_k; ++k) {
                sum += Q[i * d_k + k] * K[j * d_k + k];
            }
            scores[i * m + j] = sum;
        }
    }

    // 缩放操作：除以sqrt(d_k)
    float scale = sqrtf((float)d_k);
    for (int i = 0; i < n * m; ++i) {
        scores[i] /= scale;
    }

    // 对每行进行softmax归一化
    for (int i = 0; i < n; ++i) {
        // 找到当前行的最大值
        float max_val = scores[i * m];
        for (int j = 1; j < m; ++j) {
            if (scores[i * m + j] > max_val) {
                max_val = scores[i * m + j];
            }
        }

        // 计算指数和
        float sum_exp = 0.0f;
        for (int j = 0; j < m; ++j) {
            float exp_val = expf(scores[i * m + j] - max_val);
            scores[i * m + j] = exp_val;
            sum_exp += exp_val;
        }

        // 归一化处理
        for (int j = 0; j < m; ++j) {
            scores[i * m + j] /= sum_exp;
        }
    }

    // 分配输出矩阵内存 (n x d_v)
    float* output = (float*)malloc(n * d_v * sizeof(float));
    if (!output) {
        free(scores);
        return NULL;
    }

    // 计算注意力加权和：scores * V
    for (int i = 0; i < n; ++i) {
        for (int k = 0; k < d_v; ++k) {
            float sum = 0.0f;
            for (int j = 0; j < m; ++j) {
                sum += scores[i * m + j] * V[j * d_v + k];
            }
            output[i * d_v + k] = sum;
        }
    }

    // 释放中间矩阵内存
    free(scores);

    return output;
}
//当前困难是处理bm 和 bn 不等的情况我怎么把线程块映射到数据上面
template <unsigned int blockSize>
__device__ __forceinline__ float warpReduceSum(float sum) {
    if (blockSize >= 32)sum += __shfl_down_sync(0xffffffff, sum, 16); // 0-16, 1-17, 2-18, etc.
    if (blockSize >= 16)sum += __shfl_down_sync(0xffffffff, sum, 8);// 0-8, 1-9, 2-10, etc.
    if (blockSize >= 8)sum += __shfl_down_sync(0xffffffff, sum, 4);// 0-4, 1-5, 2-6, etc.
    if (blockSize >= 4)sum += __shfl_down_sync(0xffffffff, sum, 2);// 0-2, 1-3, 4-6, 5-7, etc.
    if (blockSize >= 2)sum += __shfl_down_sync(0xffffffff, sum, 1);// 0-1, 2-3, 4-5, etc.
    return sum;
}
/*、
    int stride_qm,int stride_qk,
    int stride_kk,int stride_kn,
    int stride_vm,int stride_vn
*/
template<const int num_threads_x,
        const int num_threads_y,
        const int d_model,
        const int d_q,
        const int block_m,
        const int block_n
>
__global__ void fa(float* Q,float* K,float* V,float* O,float* L,float* M){
    /*
    block_m is b_r
    block_n is b_c
    N is d_model

    m in d_model n in d_q k in d_v

    o_size = d_model x d_q

    q_size = d_model x d_q
    k_size = d_model x d_q
    v_size = d_model x d_q
    */

    __shared__ float q[block_m][d_q];
    __shared__ float k[block_n][d_q];
    __shared__ float v[block_n][d_q];
    __shared__ float o[block_m][d_q];
    __shared__ float s[block_m][block_n];

    __shared__ float l[block_m];
    __shared__ float m[block_m];
    int thread_num_inblock = num_threads_x*num_threads_y; 
    int T_r =  (int)(d_model / block_m);
    int T_c =  (int)(d_model / block_n);
    //global idx
    unsigned int g_idx = threadIdx.x + threadIdx.y*blockDim.x + thread_num_inblock*blockIdx.x;//grid must be a one dimensional vector like <<<1 ,(256,32) >>> 
    unsigned int b_idx = threadIdx.x + threadIdx.y*blockDim.x;
    unsigned int warpId = b_idx / 32;
    unsigned int laneId = b_idx % 32;
    //assumption that is one data for one thread and block_m == block_n
    
    //load gmem to smem
    if(g_idx < d_q*d_model){
        int data_num_block_q = d_q * block_m;//for q and o 
        int data_num_block_k = d_q * block_n;//for k v
        int row_kv = ((g_idx / d_q)%block_n);
        int row_qo = ((g_idx / d_q)%block_m);
        v[row_kv][g_idx % d_q] = V[g_idx];
        k[row_kv][g_idx % d_q] = K[g_idx];
        for(int iter = 0;iter < T_r;iter++){
            q[row_qo][g_idx%d_q] = Q[g_idx % data_num_block_q + iter*block_m*d_q];
            o[row_kv][g_idx%d_q] = O[g_idx % data_num_block_q + iter*block_m*d_q];
            if(b_idx < block_m){
                l[b_idx % block_m] = L[b_idx % block_m + iter*block_m];
                m[b_idx % block_m] = M[b_idx % block_m + iter*block_m];
            }
            //mat multiply
            //semm
            int tx = threadIdx.x;
            int ty = threadIdx.y;
            if(b_idx < d_q*block_m){
                __shared__ float tmp_sum[block_m][d_q];
                for(int z = 0;z < block_n;z++){
                    tmp_sum[ty][tx] = q[ty][tx]*k[z][tx];
                    
                    if(tx == 0){
                        float sum = 0;
                        for(int i=0;i < d_q;i++){
                            sum += tmp_sum[ty][i];
                        }
                        s[ty][z] = sum;//s_ij
                        
                        
                    }
                    __syncthreads();
                    tmp_sum[ty][tx] = 0;
                }
            }
            
            //step 10
            //calculate the max
            __shared__ float m_up[block_m];
            __shared__ float l_up[block_m];
            if(tx == 0)
            {
                float big = INT_MIN;
                for(int z=0;z < block_n;z++){
                    big = max(big,s[ty][z]);
                }
                m_up[ty] = big;
            }
            __syncthreads();
            //calculate the p_ij
            if(b_idx < block_m * block_n){
                int row = b_idx / block_n;
                int col = b_idx % block_n;
                s[row][col] = __expf(s[row][col]-m_up[row]);//p_ij
            }
            __syncthreads();
            //calculate the l_up_ij
            if(tx == 0)
            {
                float sum = 0;
                for(int z=0;z < block_n;z++){
                    sum += s[ty][z];//l_up_ij
                }
                l_up[ty] = sum;
            }
            __syncthreads();
            //step 11
            __shared__ float m_new[block_m];
            __shared__ float l_new[block_m];
            if(b_idx == 0){
                for(int i=0;i < block_m;i++){
                    m_new[i] = max(m[i],m_up[i]);
                    l_new[i] = __expf(m[i]-m_new[i])*l[i] + l_up[i]*__expf(m_up[i]-m_new[i]);
                }
                
            }
            __syncthreads();
            //calculate o_i
            __shared__ float pv[block_m][d_q];
            int row_s = b_idx / block_n;
            int col_s = b_idx % block_n;
            __shared__ float acc[block_m][block_n];
            //mat mutliply
            for(int z = 0;z < d_q;z++){
                if(b_idx < block_m*block_n){
                    acc[row_s][col_s] = s[row_s][col_s]*v[col_s][z];//p_ij*v_ij
                }
                
                //__syncthreads();
                //reduce it or just sum it up
                if(b_idx < block_m*block_n && col_s == 0){
                    float sum = 0;
                    for(int i=0;i < block_n;i++){
                        sum += acc[row_s][i];
                    }
                    pv[row_s][z] = sum;
                }
                __syncthreads();
                if(b_idx < block_m*block_n){
                    acc[row_s][col_s] = 0;
                }
            }
            o[row_qo][g_idx%d_q] = l[row_qo]*__expf(m[row_qo]-m_new[row_qo])*o[row_qo][g_idx%d_q] + __expf(m_up[row_qo]-m_new[row_qo])*pv[row_qo][g_idx%d_q];
            //write back L and M
            if(b_idx < block_m){
                L[b_idx % block_m + iter*block_m] = l_new[b_idx];
                M[b_idx % block_m + iter*block_m] = m_new[b_idx];
            }
            
            //write back to o
            O[g_idx % data_num_block_q + iter*block_m*d_q] = o[row_qo][g_idx%d_q];
            

        }
    }
    
    //debug 思路 只保留少数线程进行计算即可
}//线程分配的困难，因为不是一个块无法共享内存，无法使用sram进行通信，所以划分块和数据的索引变成了最大的困难，目前的方法每个块里面在一些计算阶段都会有很多闲置线程
int main(int argc,char** argv){
    int dev = 0;
	cudaDeviceProp deviceProp;
	CHECK(cudaGetDeviceProperties(&deviceProp,dev));
	CHECK(cudaSetDevice(dev));
    //set datasize
    constexpr int d_q = 512;
    constexpr int d_model = 2048;
    //init host data Q K V
    float* h_q;float* h_k;float* h_v;
    h_q = (float*)malloc(d_q*d_model*sizeof(float));
    h_k = (float*)malloc(d_q*d_model*sizeof(float));
    h_v = (float*)malloc(d_q*d_model*sizeof(float));
    for(int i=0;i < d_model*d_q;i++){
        h_q[i] = rand() % d_model;
    }
    for(int i=0;i < d_model*d_q;i++){
        h_k[i] = rand() % d_model;
    }
    for(int i=0;i < d_model*d_q;i++){
        h_v[i] = rand() % d_model;
    }
    //init host data O L M
    float* h_O;float*h_L;float*h_M;
    h_O = (float*)malloc(d_q*d_model*sizeof(float));
    h_L = (float*)malloc(d_model*sizeof(float));
    h_M = (float*)malloc(d_model*sizeof(float));
    for(int i=0;i < d_model*d_q;i++){
        h_O[i] = rand() % d_model;
    }
    for(int i=0;i < d_model;i++){
        h_L[i] = rand() % d_model;
    }
    for(int i=0;i < d_model;i++){
        h_M[i] = rand() % d_model;
    }
    //init device data Q K V
    float* dev_q;float* dev_k;float* dev_v;
    CHECK(cudaMalloc((float**)&dev_q,d_q*d_model*sizeof(float)));
    CHECK(cudaMalloc((float**)&dev_k,d_q*d_model*sizeof(float)));
    CHECK(cudaMalloc((float**)&dev_v,d_q*d_model*sizeof(float)));
    float* dev_O;float* dev_L;float* dev_M;
    CHECK(cudaMalloc((float**)&dev_O,d_q*d_model*sizeof(float)));
    CHECK(cudaMalloc((float**)&dev_L,d_model*sizeof(float)));
    CHECK(cudaMalloc((float**)&dev_M,d_model*sizeof(float)));
    //transfer data to device
    CHECK(cudaMemcpy(dev_q,h_q,d_q*d_model*sizeof(float),cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dev_k,h_k,d_q*d_model*sizeof(float),cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dev_v,h_v,d_q*d_model*sizeof(float),cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dev_O,h_O,d_q*d_model*sizeof(float),cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dev_L,h_L,d_model*sizeof(float),cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dev_M,h_M,d_model*sizeof(float),cudaMemcpyHostToDevice));
    //calculate b_r and b_c
    constexpr int M = 4096;//size of sram 
    constexpr int b_c = 2;//(int)M / (4*d_q);
    constexpr int b_r = 2;//(int)min(d_q,b_c);
    dim3 block(512,2);
    fa<512,2,d_model,d_q,b_r,b_c><<<256,block>>>(dev_q,dev_k,dev_v,dev_O,dev_L,dev_M);
    CHECK(cudaMemcpy(h_O,dev_O,d_q*d_model*sizeof(float),cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_L,dev_L,d_model*sizeof(float),cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_M,dev_M,d_model*sizeof(float),cudaMemcpyDeviceToHost));//now got the answer
    //calculate on host
    float* ans_h = (float*)malloc(d_model*d_q*sizeof(float));
    ans_h = h_attention(h_q,h_k,h_v,d_model,d_q,d_q,d_q);
    for(int i=0;i < 32;i++){
        printf("gpu g[%d] ans is :%f\n",i,h_O[i]);
        printf("cpu c[%d] ans is :%f\n",i,ans_h[i]);
    }
    free(h_q);
    free(h_k);
    free(h_v);
    free(h_O);
    free(h_L);
    free(h_M);
    free(ans_h);
    cudaFree(dev_q);
    cudaFree(dev_k);
    cudaFree(dev_v);
    cudaFree(dev_O);
    cudaFree(dev_L);
    cudaFree(dev_M);
    return 0;
}