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

template<
    const int N,
    const int d,
    const int b_r,
    const int b_c
>
__global__ void fa_v2(float* Q, float* K, float* V, float* O, float* L, float* M) {
    // 基于FlashAttention论文的参数命名
    // N: 序列长度
    // d: 特征维度 (d_k == d_v == d)
    // b_r: Q/O 的行块大小
    // b_c: K/V 的列块大小

    // 步骤 3: 共享内存（SRAM）分配
    __shared__ float q[b_r][d];
    __shared__ float k[b_c][d];
    __shared__ float v[b_c][d];
    __shared__ float s[b_r][b_c];

    // 步骤 4: 分配 o, l, m 的共享内存
    __shared__ float o[b_r][d];
    __shared__ float l[b_r];
    __shared__ float m[b_r];
    __shared__ float m_old[b_r];  // 存储旧的m值

    // 线程索引
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int b_idx = ty * blockDim.x + tx; // 1D 线程索引

    // T_r 和 T_c 是沿序列长度的块数
    int T_r = N / b_r;
    int T_c = N / b_c;
    float scale = 1.0f / sqrtf((float)d); // 预计算缩放因子

    // 步骤 7: 外层循环遍历 Q 的块 (iter -> i in paper)
    for (int iter = 0; iter < T_r; iter++) {

        // --- 加载 Q 块 ---
        for (int i = b_idx; i < b_r * d; i += blockDim.x * blockDim.y) {
            q[i / d][i % d] = Q[iter * b_r * d + i];
        }
        __syncthreads();

        // --- 加载 O, l, m 的当前状态 ---
        for (int i = b_idx; i < b_r * d; i += blockDim.x * blockDim.y) {
            o[i / d][i % d] = 0.0f;
        }
        for (int i = b_idx; i < b_r; i += blockDim.x * blockDim.y) {
            l[i] = 0.0f;
            m[i] = -FLT_MAX;
        }
        __syncthreads();

        

        // 步骤 6: 内层循环遍历 K, V 的块 (j in paper)
        for (int j = 0; j < T_c; j++) {
            // --- 步骤 7: 加载 K, V 块 ---
            for (int i = b_idx; i < b_c * d; i += blockDim.x * blockDim.y) {
                k[i / d][i % d] = K[j * b_c * d + i];
                v[i / d][i % d] = V[j * b_c * d + i];
            }
            __syncthreads();

            // --- 步骤 8: 计算 S = QK^T ---
            // 使用循环让所有线程参与计算，而不是依赖于线程块的形状
            for (int row = ty; row < b_r; row += blockDim.y) {
                for (int col = tx; col < b_c; col += blockDim.x) {
                    float sum = 0.0f;
                    for (int k_dim = 0; k_dim < d; k_dim++) {
                        sum += q[row][k_dim] * k[col][k_dim];
                    }
                    s[row][col] = sum * scale;
                }
            }
            __syncthreads();

            // --- 步骤 9: 计算新的 m 和 l 值 ---
            // 使用一个线程（例如 tx=0 的线程）来为每一行计算
            
            
            
            if (tx == 0) {
                for (int row = ty; row < b_r; row += blockDim.y) {
                    // 保存旧的m值
                    m_old[row] = m[row];
                    
                    // 1. 找最大值作为新的m值
                    float row_max = -FLT_MAX;
                    for (int col = 0; col < b_c; col++) {
                        if (s[row][col] > row_max) {
                            row_max = s[row][col];
                        }
                    }
                    
                    // 更新m值，与旧的m取最大值 // 存在一个写入的竞争问题 但是因为一行一个 这个问题不存在
                    float m_cur = fmaxf(m_old[row], row_max);
                    m[row] = m_cur;

                    // 2. 计算 P_ij 并求和得到 l_new
                    float row_sum_exp = 0.0f;
                    for (int col = 0; col < b_c; col++) {
                        float p_val = __expf(s[row][col] - m[row]);
                        s[row][col] = p_val; // 将 s 矩阵原地更新为 P 矩阵
                        row_sum_exp += p_val;
                    }
                    
                    // 更新l值
                    float l_old = l[row];
                    l[row] = __expf(m_old[row] - m_cur) * l_old + row_sum_exp;
                }
            }
            __syncthreads();

            // --- 步骤 10: 更新 O ---
           
            
            // --- 计算 P_ij * V_jk ---
            __shared__ float pv[b_r][d];
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

            // --- 更新 O ---
            // 所有线程参与更新自己的 o 元素，按照公式 O_i^(j) = diag(e^(m_i^(j-1)-m_i^(j)))^(-1) * O_i^(j-1) + P_i^(j) * V_j
            for (int row = ty; row < b_r; row += blockDim.y) {
                for (int col = tx; col < d; col += blockDim.x) {
                     o[row][col] = o[row][col] * __expf(m_old[row] - m[row]) + pv[row][col];
                }
            }
            __syncthreads();
        } // 内层循环结束 (j)
        //step 12 update o_i^(T_r)
        for (int row = ty; row < b_r; row += blockDim.y) {
                for (int col = tx; col < d; col += blockDim.x) {
                    float o_old = o[row][col];
                    o[row][col] = o_old / l[row];
                    //printf("outside o[%d][%d] = %f\n",row,col,o[row][col]);
                }
            }
        __syncthreads();
        //step 13 compute L_i = m_i^(T_c) + log(l_i^(T_c))
        for (int row = ty; row < b_r; row += blockDim.y) {
            l[row] = m[row] + logf(l[row]);
        }
        __syncthreads();
        // --- 将更新后的 O, l, m 写回全局内存 ---
        for (int i = b_idx; i < b_r * d; i += blockDim.x * blockDim.y) {
            O[iter * b_r * d + i] = o[i / d][i % d];
        }
        for (int i = b_idx; i < b_r; i += blockDim.x * blockDim.y) {
            L[iter * b_r + i] = l[i];
        }
        __syncthreads();
    } // 外层循环结束 (iter)
}

int main(int argc,char** argv){
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp,dev));
    CHECK(cudaSetDevice(dev));
    //set datasize
    constexpr int d = 512;
    constexpr int N = 2048;
    //init host data Q K V
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
    //init host data O L M
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
    //init device data Q K V
    float* dev_q;float* dev_k;float* dev_v;
    CHECK(cudaMalloc((float**)&dev_q,d*N*sizeof(float)));
    CHECK(cudaMalloc((float**)&dev_k,d*N*sizeof(float)));
    CHECK(cudaMalloc((float**)&dev_v,d*N*sizeof(float)));
    float* dev_O;float* dev_L;float* dev_M;
    CHECK(cudaMalloc((float**)&dev_O,d*N*sizeof(float)));
    CHECK(cudaMalloc((float**)&dev_L,N*sizeof(float)));
    CHECK(cudaMalloc((float**)&dev_M,N*sizeof(float)));
    //transfer data to device
    CHECK(cudaMemcpy(dev_q,h_q,d*N*sizeof(float),cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dev_k,h_k,d*N*sizeof(float),cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dev_v,h_v,d*N*sizeof(float),cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dev_O,h_O,d*N*sizeof(float),cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dev_L,h_L,N*sizeof(float),cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dev_M,h_M,N*sizeof(float),cudaMemcpyHostToDevice));
    //calculate b_r and b_c
    constexpr int M = 10000;//size of sram
    constexpr int b_c = 4;//(int)M / (4*d);
    constexpr int b_r = 4;//(int)min(d,b_c);
    dim3 block(512,2);
    printf("Launching Kernel: N=%d, d=%d, b_r=%d, b_c=%d, Block=(%d,%d)\n", N, d, b_r, b_c, block.x, block.y);
    fa_v2<N,d,b_r,b_c><<<32,block>>>(dev_q,dev_k,dev_v,dev_O,dev_L,dev_M);
    CHECK(cudaGetLastError()); // Check for errors during kernel launch
    CHECK(cudaDeviceSynchronize()); // Wait for kernel to complete

    CHECK(cudaMemcpy(h_O,dev_O,d*N*sizeof(float),cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_L,dev_L,N*sizeof(float),cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_M,dev_M,N*sizeof(float),cudaMemcpyDeviceToHost));//now got the answer
    //calculate on host
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
