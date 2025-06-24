#include "common/common.h"
#include <cooperative_groups.h>
#include <cfloat>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>


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
    __shared__ float m_old[b_r];  // Store old m values

    // Thread indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int b_idx = ty * blockDim.x + tx; // 1D thread index

    // T_r and T_c are the number of blocks along sequence length
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

// Explicit template instantiation - ensures implementations are available at link time
template __global__ void fa<512, 2, 2048, 512, 4, 4>(float*, float*, float*, float*, float*, float*);
template __global__ void fa_v2<2048, 512, 4, 4>(float*, float*, float*, float*, float*, float*); 