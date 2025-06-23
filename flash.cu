#include "common/common.h"
#include <cooperative_groups.h>
#include <cfloat>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>


float* h_attention(const float* Q, const float* K, const float* V,
                   int n, int m, int d_k, int d_v) {
    // Allocate memory for attention score matrix (n x m)
    float* scores = (float*)malloc(n * m * sizeof(float));
    if (!scores) return NULL;

    // Calculate QK^T and store in scores matrix
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < d_k; ++k) {
                sum += Q[i * d_k + k] * K[j * d_k + k];
            }
            scores[i * m + j] = sum;
        }
    }

    // Scaling operation: divide by sqrt(d_k)
    float scale = sqrtf((float)d_k);
    for (int i = 0; i < n * m; ++i) {
        scores[i] /= scale;
    }

    // Apply softmax normalization to each row
    for (int i = 0; i < n; ++i) {
        // Find the maximum value in the current row
        float max_val = scores[i * m];
        for (int j = 1; j < m; ++j) {
            if (scores[i * m + j] > max_val) {
                max_val = scores[i * m + j];
            }
        }

        // Calculate exponential sum
        float sum_exp = 0.0f;
        for (int j = 0; j < m; ++j) {
            float exp_val = expf(scores[i * m + j] - max_val);
            scores[i * m + j] = exp_val;
            sum_exp += exp_val;
        }

        // Normalization process
        for (int j = 0; j < m; ++j) {
            scores[i * m + j] /= sum_exp;
        }
    }

    // Allocate memory for output matrix (n x d_v)
    float* output = (float*)malloc(n * d_v * sizeof(float));
    if (!output) {
        free(scores);
        return NULL;
    }

    // Calculate attention weighted sum: scores * V
    for (int i = 0; i < n; ++i) {
        for (int k = 0; k < d_v; ++k) {
            float sum = 0.0f;
            for (int j = 0; j < m; ++j) {
                sum += scores[i * m + j] * V[j * d_v + k];
            }
            output[i * d_v + k] = sum;
        }
    }

    // Free intermediate matrix memory
    free(scores);

    return output;
}

template<const int num_threads_x,
        const int num_threads_y,
        const int N,
        const int d,
        const int b_r,
        const int b_c
>
__global__ void fa(float* Q,float* K,float* V,float* O,float* L,float* M){
    /*
    parameter: name base on paper

    m in N
    n in d_k
    k in d_v

    o_size = N x d

    q_size = N x d
    k_size = N x d
    v_size = N x d
    */

    //SRAM allocation step 3 in paper
    __shared__ float q[b_r][d];
    __shared__ float k[b_c][d];
    __shared__ float v[b_c][d];
    int T_r =  (int)(N / b_r); // divide operations are slow in GPU find a way to replace it
    int T_c =  (int)(N / b_c);
    //SRAM allocation step 4 in paper
    __shared__ float o[b_r][d];
    __shared__ float s[b_r][b_c];
    __shared__ float l[b_r];
    __shared__ float m[b_r];
    float scale = 1.0f / sqrtf((float)d);
    //calculate the thread num in block and the thread num in row and col
    int thread_num_inblock = num_threads_x*num_threads_y;
    unsigned int b_idx = threadIdx.x + threadIdx.y*blockDim.x;// 2D matrix but 1D idx
    int data_num_block_q = d * b_r;//for q and o
    int data_num_block_k = d * b_c;//for k v
    //assumption that is one data for one thread and b_r == b_c
   
    //load gmem to smem
    for(int j = 0;j < T_c;j++){//step 5
        for(int idx = b_idx;idx < data_num_block_k;idx += blockDim.x * blockDim.y){
            
            v[idx / d][idx % d] = V[idx  + j * data_num_block_k];//each block load its own gmem to smem
            k[idx / d][idx % d] = K[idx  + j * data_num_block_k];//in paper it is step 6
        }
        __syncthreads();
        for(int iter = 0;iter < T_r;iter++){//step 7
           for (int i = b_idx; i < b_r * d; i += blockDim.x * blockDim.y) {
                int row = i / d;
                int col = i % d;
                q[row][col] = Q[iter * b_r * d + i];
                o[row][col] = O[iter * b_r * d + i];
            }

            for(int idx = b_idx;idx < b_r;idx += blockDim.x * blockDim.y){
                l[idx] = L[idx + iter*b_r];
                m[idx] = M[idx + iter*b_r];
            }
            __syncthreads();
            //mat multiply
            //semm step 9
            int tx = threadIdx.x;
            int ty = threadIdx.y;
           
            for (int row = ty; row < b_r; row += blockDim.y) {
                for (int col = tx; col < b_c; col += blockDim.x) {
                    float sum = 0.0f;
                    for (int k_dim = 0; k_dim < d; k_dim++) {
                        sum += q[row][k_dim] * k[col][k_dim];
                    }
                    // *** BUG FIX 1: Added scaling step ***
                    s[row][col] = sum * scale;
                }
            }
            __syncthreads(); // Essential: All threads must finish computing their part of 's'
           
            //step 10
            //calculate the max
            __shared__ float m_up[b_r];
            __shared__ float l_up[b_r];
            if (tx == 0) {
                for (int row = ty; row < b_r; row += blockDim.y) {
                    // 1. Find maximum value m_up
                    float row_max = -FLT_MAX;
                    for (int col = 0; col < b_c; col++) {
                        if (s[row][col] > row_max) {
                            row_max = s[row][col];
                        }
                    }
                    m_up[row] = row_max;

                    // 2. Calculate P_ij and sum to get l_up
                    float row_sum_exp = 0.0f;
                    for (int col = 0; col < b_c; col++) {
                        float p_val = __expf(s[row][col] - row_max);
                        s[row][col] = p_val; // Update s matrix in-place to become P matrix
                        row_sum_exp += p_val;
                    }
                    l_up[row] = row_sum_exp;
                }
            }
            __syncthreads();

            //step 11
            __shared__ float m_new[b_r];
            __shared__ float l_new[b_r];
            if (tx == 0) {
                for (int row = ty; row < b_r; row += blockDim.y) {
                    m_new[row] = fmaxf(m[row], m_up[row]);
                    l_new[row] = __expf(m[row] - m_new[row]) * l[row] + __expf(m_up[row] - m_new[row]) * l_up[row];
                }
            }
            __syncthreads();
            //calculate o_i
            __shared__ float pv[b_r][d];
            // Each thread (ty, tx) will be responsible for computing one element pv[ty][tx].
            // This requires iterating through the inner dimension 'b_c'.

            for (int row = ty; row < b_r; row += blockDim.y) {
                for (int col = tx; col < d; col += blockDim.x) {
                    float pv_sum = 0.0f;
                    for (int k_dim = 0; k_dim < b_c; k_dim++) {
                        pv_sum += s[row][k_dim] * v[k_dim][col];
                    }
                    pv[row][col] = pv_sum;
                }
            }
            __syncthreads();
            for (int row = ty; row < b_r; row += blockDim.y) {
                float m_old = m[row];
                float l_old = l[row];
                float m_new_val = m_new[row];
                float l_new_val = l_new[row];

                for (int col = tx; col < d; col += blockDim.x) {
                    float o_old = o[row][col];
                    float pv_val = pv[row][col];
                    // Update formula
                    o[row][col] = (l_old * __expf(m_old - m_new_val) * o_old + __expf(m_up[row] - m_new_val) * pv_val) / l_new_val;
                }
            }
             __syncthreads();
             if (tx == 0) {
                 for (int row = ty; row < b_r; row += blockDim.y) {
                    l[row] = l_new[row];
                    m[row] = m_new[row];
                 }
            }
            __syncthreads();

            for (int i = b_idx; i < b_r * d; i += blockDim.x * blockDim.y) {
                O[iter * b_r * d + i] = o[i / d][i % d];
            }
            for (int i = b_idx; i < b_r; i += blockDim.x * blockDim.y) {
                L[iter * b_r + i] = l[i];
                M[iter * b_r + i] = m[i];
            }
            __syncthreads();
        }
        
    }
   
   
}//thread block must bigger than max(b_r,b_c) * d

int main(int argc,char** argv){
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp,dev));
    CHECK(cudaSetDevice(dev));
    //Set data size
    constexpr int d = 512;
    constexpr int N = 2048;
    //Initialize host data Q K V
    float* h_q;float* h_k;float* h_v;
    h_q = (float*)malloc(d*N*sizeof(float));
    h_k = (float*)malloc(d*N*sizeof(float));
    h_v = (float*)malloc(d*N*sizeof(float));
    for(int i=0;i < N*d;i++){
        h_q[i] = rand() % N;
        h_q[i] /= d;
    }
    for(int i=0;i < N*d;i++){
        h_k[i] = rand() % N;
        h_k[i] /= d;    
    }
    for(int i=0;i < N*d;i++){
        h_v[i] = rand() % N;
        h_v[i] /= d;
    }
    //Initialize host data O L M
    float* h_O;float*h_L;float*h_M;
    h_O = (float*)malloc(d*N*sizeof(float));
    h_L = (float*)malloc(N*sizeof(float));
    h_M = (float*)malloc(N*sizeof(float));
    for(int i=0;i < N*d;i++){
        h_O[i] = 0;
    }
    for(int i=0;i < N;i++){
        h_L[i] = 0;
    }
    for(int i=0;i < N;i++){
        h_M[i] = -FLT_MAX;
    }
    //Initialize device data Q K V
    float* dev_q;float* dev_k;float* dev_v;
    CHECK(cudaMalloc((float**)&dev_q,d*N*sizeof(float)));
    CHECK(cudaMalloc((float**)&dev_k,d*N*sizeof(float)));
    CHECK(cudaMalloc((float**)&dev_v,d*N*sizeof(float)));
    float* dev_O;float* dev_L;float* dev_M;
    CHECK(cudaMalloc((float**)&dev_O,d*N*sizeof(float)));
    CHECK(cudaMalloc((float**)&dev_L,N*sizeof(float)));
    CHECK(cudaMalloc((float**)&dev_M,N*sizeof(float)));
    //Transfer data to device
    CHECK(cudaMemcpy(dev_q,h_q,d*N*sizeof(float),cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dev_k,h_k,d*N*sizeof(float),cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dev_v,h_v,d*N*sizeof(float),cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dev_O,h_O,d*N*sizeof(float),cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dev_L,h_L,N*sizeof(float),cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dev_M,h_M,N*sizeof(float),cudaMemcpyHostToDevice));
    //Calculate b_r and b_c
    constexpr int M = 10000;//size of SRAM
    constexpr int b_c = 4;//(int)M / (4*d);
    constexpr int b_r = 4;//(int)min(d,b_c);
    dim3 block(512,2);
    fa<512,2,N,d,b_r,b_c><<<1,block>>>(dev_q,dev_k,dev_v,dev_O,dev_L,dev_M);
    CHECK(cudaMemcpy(h_O,dev_O,d*N*sizeof(float),cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_L,dev_L,N*sizeof(float),cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_M,dev_M,N*sizeof(float),cudaMemcpyDeviceToHost));//now we have the answer
    //Calculate on host
    float* ans_h = (float*)malloc(N*d*sizeof(float));
    ans_h = h_attention(h_q,h_k,h_v,N,N,d,d);
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
