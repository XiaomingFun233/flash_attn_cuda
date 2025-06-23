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

template<
    const int N,
    const int d,
    const int b_r,
    const int b_c
>
__global__ void fa_v2(float* Q, float* K, float* V, float* O, float* L, float* M) {
    // Parameter naming based on FlashAttention paper
    // N: Sequence length
    // d: Feature dimension (d_k == d_v == d)
    // b_r: Row block size for Q/O
    // b_c: Column block size for K/V

    // Step 3: Shared memory (SRAM) allocation
    __shared__ float q[b_r][d];
    __shared__ float k[b_c][d];
    __shared__ float v[b_c][d];
    __shared__ float s[b_r][b_c];

    // Step 4: Allocate shared memory for o, l, m
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
    float scale = 1.0f / sqrtf((float)d); // Pre-compute scaling factor

    // Step 7: Outer loop iterates over Q blocks (iter -> i in paper)
    for (int iter = 0; iter < T_r; iter++) {

        // --- Load Q block ---
        for (int i = b_idx; i < b_r * d; i += blockDim.x * blockDim.y) {
            q[i / d][i % d] = Q[iter * b_r * d + i];
        }
        __syncthreads();

        // --- Load current state of O, l, m ---
        for (int i = b_idx; i < b_r * d; i += blockDim.x * blockDim.y) {
            o[i / d][i % d] = 0.0f;
        }
        for (int i = b_idx; i < b_r; i += blockDim.x * blockDim.y) {
            l[i] = 0.0f;
            m[i] = -FLT_MAX;
        }
        __syncthreads();

        

        // Step 6: Inner loop iterates over K, V blocks (j in paper)
        for (int j = 0; j < T_c; j++) {
            // --- Step 7: Load K, V blocks ---
            for (int i = b_idx; i < b_c * d; i += blockDim.x * blockDim.y) {
                k[i / d][i % d] = K[j * b_c * d + i];
                v[i / d][i % d] = V[j * b_c * d + i];
            }
            __syncthreads();

            // --- Step 8: Calculate S = QK^T ---
                                        // Use loops to involve all threads in computation, instead of relying on thread block shape
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

            // --- Step 9: Calculate new m and l values ---
            // Use one thread (e.g., thread with tx=0) to compute for each row
            
            
            
            if (tx == 0) {
                for (int row = ty; row < b_r; row += blockDim.y) {
                    // Save old m value
                    m_old[row] = m[row];
                    
                    // 1. Find maximum value as new m value
                    float row_max = -FLT_MAX;
                    for (int col = 0; col < b_c; col++) {
                        if (s[row][col] > row_max) {
                            row_max = s[row][col];
                        }
                    }
                    
                    // Update m value, take the maximum of old m // A write contention issue exists but it's not a problem because each row has one thread
                    float m_cur = fmaxf(m_old[row], row_max);
                    m[row] = m_cur;

                    // 2. Calculate P_ij and sum to get l_new
                    float row_sum_exp = 0.0f;
                    for (int col = 0; col < b_c; col++) {
                        float p_val = __expf(s[row][col] - m[row]);
                        s[row][col] = p_val; // Update s matrix in-place to become P matrix
                        row_sum_exp += p_val;
                    }
                    
                    // Update l value
                    float l_old = l[row];
                    l[row] = __expf(m_old[row] - m_cur) * l_old + row_sum_exp;
                }
            }
            __syncthreads();

            // --- Step 10: Update O ---
           
            
            // --- Calculate P_ij * V_jk ---
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

            // --- Update O ---
            // All threads participate in updating their o elements, according to formula O_i^(j) = diag(e^(m_i^(j-1)-m_i^(j)))^(-1) * O_i^(j-1) + P_i^(j) * V_j
            for (int row = ty; row < b_r; row += blockDim.y) {
                for (int col = tx; col < d; col += blockDim.x) {
                     o[row][col] = o[row][col] * __expf(m_old[row] - m[row]) + pv[row][col];
                }
            }
            __syncthreads();
        } // Inner loop ends (j)
        //Step 12: Update o_i^(T_r)
        for (int row = ty; row < b_r; row += blockDim.y) {
                for (int col = tx; col < d; col += blockDim.x) {
                    float o_old = o[row][col];
                    o[row][col] = o_old / l[row];
                    //printf("outside o[%d][%d] = %f\n",row,col,o[row][col]);
                }
            }
        __syncthreads();
        //Step 13: Compute L_i = m_i^(T_c) + log(l_i^(T_c))
        for (int row = ty; row < b_r; row += blockDim.y) {
            l[row] = m[row] + logf(l[row]);
        }
        __syncthreads();
        // --- Write updated O, l, m back to global memory ---
        for (int i = b_idx; i < b_r * d; i += blockDim.x * blockDim.y) {
            O[iter * b_r * d + i] = o[i / d][i % d];
        }
        for (int i = b_idx; i < b_r; i += blockDim.x * blockDim.y) {
            L[iter * b_r + i] = l[i];
        }
        __syncthreads();
    } // Outer loop ends (iter)
}

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
    printf("Launching Kernel: N=%d, d=%d, b_r=%d, b_c=%d, Block=(%d,%d)\n", N, d, b_r, b_c, block.x, block.y);
    fa_v2<N,d,b_r,b_c><<<32,block>>>(dev_q,dev_k,dev_v,dev_O,dev_L,dev_M);
    CHECK(cudaGetLastError()); // Check for errors during kernel launch
    CHECK(cudaDeviceSynchronize()); // Wait for kernel to complete

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
