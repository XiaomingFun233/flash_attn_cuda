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
template<const int num_threads_x,
         const int num_threads_y,
         const int N,
         const int d,
         const int b_r,
         const int b_c>
__global__ void fa_op1(float* Q, float* K, float* V, float* O, float* L, float* M) {
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

    // Shared memory (SRAM) allocation
    __shared__ float q[b_r][d];
    __shared__ float k[b_c][d];
    __shared__ float v[b_c][d];
    __shared__ float o[b_r][d];
    __shared__ float s[b_r][b_c];
    __shared__ float l[b_r];
    __shared__ float m[b_r];
    __shared__ float m_up[b_r];
    __shared__ float l_up[b_r];
    __shared__ float m_new[b_r];
    __shared__ float l_new[b_r];
    __shared__ float pv[b_r][d];


    const int T_r = N / b_r;
    const int T_c = N / b_c;
    const float scale = 1.0f / sqrtf((float)d);

    // 1D thread index within the 2D thread block
    const unsigned int b_idx = threadIdx.x + threadIdx.y * blockDim.x;
    const unsigned int block_size = blockDim.x * blockDim.y;

    // Outer loop over blocks of K and V
    for (int j = 0; j < T_c; j++) { // step 5
        // --- COALESCED MEMORY READ for K and V ---
        // Each thread block loads a contiguous block of K and V from global to shared memory.
        const int gmem_block_offset_kv = j * b_c * d;
        for (int i = b_idx; i < b_c * d; i += block_size) {
            const int row = i / d;
            const int col = i % d;
            // Consecutive threads read consecutive memory locations
            k[row][col] = K[gmem_block_offset_kv + i]; // step 6
            v[row][col] = V[gmem_block_offset_kv + i];
        }
        __syncthreads();

        // Inner loop over blocks of Q
        for (int iter = 0; iter < T_r; iter++) { // step 7
            // --- COALESCED MEMORY READ for Q, O, L, M ---
            const int gmem_block_offset_q = iter * b_r * d;
            const int gmem_block_offset_lm = iter * b_r;

            // Fused loading of Q and O
            for (int i = b_idx; i < b_r * d; i += block_size) {
                const int row = i / d;
                const int col = i % d;
                q[row][col] = Q[gmem_block_offset_q + i];
                o[row][col] = O[gmem_block_offset_q + i];
            }

            // Fused loading of L and M (only first warp needs to do this)
            // This is a common optimization.
            if (threadIdx.y == 0) {
                 for(int i = threadIdx.x; i < b_r; i += blockDim.x){
                    l[i] = L[gmem_block_offset_lm + i];
                    m[i] = M[gmem_block_offset_lm + i];
                }
            }
            __syncthreads();


            // S = Q * K^T (step 9)
            for (int row = threadIdx.y; row < b_r; row += blockDim.y) {
                for (int col = threadIdx.x; col < b_c; col += blockDim.x) {
                    float sum = 0.0f;
                    for (int k_dim = 0; k_dim < d; k_dim++) {
                        sum += q[row][k_dim] * k[col][k_dim];
                    }
                    s[row][col] = sum * scale;
                }
            }
            __syncthreads();

            // Row-wise statistics (m_up, l_up) and update S to P (step 10)
            if (threadIdx.x == 0) {
                for (int row = threadIdx.y; row < b_r; row += blockDim.y) {
                    float row_max = -FLT_MAX;
                    for (int col = 0; col < b_c; col++) {
                        row_max = fmaxf(row_max, s[row][col]);
                    }
                    m_up[row] = row_max;

                    float row_sum_exp = 0.0f;
                    for (int col = 0; col < b_c; col++) {
                        float p_val = __expf(s[row][col] - row_max);
                        s[row][col] = p_val; // s becomes P
                        row_sum_exp += p_val;
                    }
                    l_up[row] = row_sum_exp;
                }
            }
            __syncthreads();

            // Update statistics m_new, l_new (step 11)
            if (threadIdx.x == 0) {
                for (int row = threadIdx.y; row < b_r; row += blockDim.y) {
                    m_new[row] = fmaxf(m[row], m_up[row]);
                    l_new[row] = __expf(m[row] - m_new[row]) * l[row] + __expf(m_up[row] - m_new[row]) * l_up[row];
                }
            }
            __syncthreads();

            // P_ij * V_jk (step 12)
            for (int row = threadIdx.y; row < b_r; row += blockDim.y) {
                for (int col = threadIdx.x; col < d; col += blockDim.x) {
                    float pv_sum = 0.0f;
                    for (int k_dim = 0; k_dim < b_c; k_dim++) {
                        pv_sum += s[row][k_dim] * v[k_dim][col];
                    }
                    pv[row][col] = pv_sum;
                }
            }
            __syncthreads();

            // Update O, L, M
            for (int row = threadIdx.y; row < b_r; row += blockDim.y) {
                float m_old_val = m[row];
                float l_old_val = l[row];
                float m_new_val = m_new[row];
                float l_new_val = l_new[row];
                float m_up_val = m_up[row];

                for (int col = threadIdx.x; col < d; col += blockDim.x) {
                     o[row][col] = (l_old_val * __expf(m_old_val - m_new_val) * o[row][col] + __expf(m_up_val - m_new_val) * pv[row][col]) / l_new_val;
                }
            }
             if (threadIdx.x == 0) {
                for (int row = threadIdx.y; row < b_r; row += blockDim.y) {
                    l[row] = l_new[row];
                    m[row] = m_new[row];
                }
            }
            __syncthreads();


            // --- COALESCED MEMORY WRITE for O, L, M ---
            // Fused writing of O
            for (int i = b_idx; i < b_r * d; i += block_size) {
                 O[gmem_block_offset_q + i] = o[i / d][i % d];
            }
            // Fused writing of L and M
            if (threadIdx.y == 0) {
                 for(int i = threadIdx.x; i < b_r; i += blockDim.x){
                    L[gmem_block_offset_lm + i] = l[i];
                    M[gmem_block_offset_lm + i] = m[i];
                }
            }
            __syncthreads();
        }
    }
}
template<const int num_threads_x,
         const int num_threads_y,
         const int N,
         const int d,
         const int b_r,
         const int b_c>
__global__ void fa_op2(float* Q, float* K, float* V, float* O, float* L, float* M) {
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

    // Shared memory (SRAM) allocation
    __shared__ float q[b_r][d];
    __shared__ float k[b_c][d];
    __shared__ float v[b_c][d];
    __shared__ float o[b_r][d];
    __shared__ float s[b_r][b_c];
    __shared__ float l[b_r];
    __shared__ float m[b_r];
    __shared__ float m_up[b_r];
    __shared__ float l_up[b_r];
    __shared__ float m_new[b_r];
    __shared__ float l_new[b_r];
    __shared__ float pv[b_r][d];


    const int T_r = N / b_r;
    const int T_c = N / b_c;
    const float scale = 1.0f / sqrtf((float)d);

    // 1D thread index within the 2D thread block
    const unsigned int b_idx = threadIdx.x + threadIdx.y * blockDim.x;
    const unsigned int block_size = blockDim.x * blockDim.y;

    // Outer loop over blocks of K and V
    for (int j = 0; j < T_c; j++) { // step 5
        // --- COALESCED MEMORY READ for K and V ---
        // Each thread block loads a contiguous block of K and V from global to shared memory.
        const int gmem_block_offset_kv = j * b_c * d;
        for (int i = b_idx; i < b_c * d; i += block_size) {
            const int row = i / d;
            const int col = i % d;
            // Consecutive threads read consecutive memory locations
            k[row][col] = K[gmem_block_offset_kv + i]; // step 6
            v[row][col] = V[gmem_block_offset_kv + i];
        }
        __syncthreads();

        // Inner loop over blocks of Q
        for (int iter = 0; iter < T_r; iter++) { // step 7
            // --- COALESCED MEMORY READ for Q, O, L, M ---
            const int gmem_block_offset_q = iter * b_r * d;
            const int gmem_block_offset_lm = iter * b_r;

            // Fused loading of Q and O
            for (int i = b_idx; i < b_r * d; i += block_size) {
                const int row = i / d;
                const int col = i % d;
                q[row][col] = Q[gmem_block_offset_q + i];
                o[row][col] = O[gmem_block_offset_q + i];
            }

            // Fused loading of L and M (only first warp needs to do this)
            // This is a common optimization.
            if (threadIdx.y == 0) {
                 for(int i = threadIdx.x; i < b_r; i += blockDim.x){
                    l[i] = L[gmem_block_offset_lm + i];
                    m[i] = M[gmem_block_offset_lm + i];
                }
            }
            __syncthreads();


            // --- OPTIMIZED Tiled Matrix Multiplication for S = Q * K^T ---

            // Define the size of the tiles. This is often the same as the thread block's dimension.
            // For this to work best, num_threads_x and num_threads_y should be equal (e.g., 16 or 32)
            // and 'd' should be a multiple of TILE_DIM.
            constexpr int TILE_DIM = num_threads_x;

            // Temporary tile storage in shared memory. One tile for Q, one for K.
            __shared__ float q_tile[TILE_DIM][TILE_DIM];
            __shared__ float k_tile[TILE_DIM][TILE_DIM]; // For K^T, we load K and access it transposed

            // Each thread will compute one element of the output matrix S.
            // We identify the target row and column for this thread.
            int target_row = threadIdx.y;
            int target_col = threadIdx.x;

            // Accumulator for the result, stored in a register for each thread.
            float sum = 0.0f;

            // Loop over the tiles in the 'd' dimension
            for (int tile_idx = 0; tile_idx < (d / TILE_DIM); ++tile_idx) {
                // --- Step 1: Cooperatively load tiles from shared memory ---
                // Each thread loads one element of q_tile and one element of k_tile.
                
                // Load element for q_tile
                q_tile[target_row][target_col] = q[target_row][tile_idx * TILE_DIM + target_col];

                // Load element for k_tile. We need to multiply by K^T.
                // This means s[r][c] = sum(q[r][k] * k[c][k]).
                // So thread (r,c) needs q[r][...] and k[c][...].
                k_tile[target_row][target_col] = k[target_col][tile_idx * TILE_DIM + target_row];

                // Synchronize to make sure both tiles are fully loaded before computation.
                __syncthreads();

                // --- Step 2: Compute matrix multiplication for the loaded tiles ---
                // Each thread computes a partial sum by iterating through the tile's dimension.
                // This loop is very fast as it only uses registers and shared memory.
                for (int i = 0; i < TILE_DIM; ++i) {
                    sum += q_tile[target_row][i] * k_tile[i][target_col];
                }

                // Synchronize after computation to ensure all threads are done with the
                // current tiles before loading the next ones.
                __syncthreads();
            }

            // --- Step 3: Write the final result to the 's' matrix ---
            // (Assuming b_r and b_c are equal to TILE_DIM)
            s[target_row][target_col] = sum * scale;

            // Final sync to ensure 's' is fully written before the next kernel steps.
            __syncthreads();

            // Row-wise statistics (m_up, l_up) and update S to P (step 10)
            if (threadIdx.x == 0) {
                for (int row = threadIdx.y; row < b_r; row += blockDim.y) {
                    float row_max = -FLT_MAX;
                    for (int col = 0; col < b_c; col++) {
                        row_max = fmaxf(row_max, s[row][col]);
                    }
                    m_up[row] = row_max;

                    float row_sum_exp = 0.0f;
                    for (int col = 0; col < b_c; col++) {
                        float p_val = __expf(s[row][col] - row_max);
                        s[row][col] = p_val; // s becomes P
                        row_sum_exp += p_val;
                    }
                    l_up[row] = row_sum_exp;
                }
            }
            __syncthreads();

            // Update statistics m_new, l_new (step 11)
            if (threadIdx.x == 0) {
                for (int row = threadIdx.y; row < b_r; row += blockDim.y) {
                    m_new[row] = fmaxf(m[row], m_up[row]);
                    l_new[row] = __expf(m[row] - m_new[row]) * l[row] + __expf(m_up[row] - m_new[row]) * l_up[row];
                }
            }
            __syncthreads();

            // --- OPTIMIZED Tiled Matrix Multiplication: PV = s * v ---

            // Assume TILE_DIM is already defined, typically matching the thread block dimension (e.g., 16 or 32),
            // and that b_r, d, and b_c are all integer multiples of TILE_DIM.
            // constexpr int TILE_DIM = num_threads_x;

            // Temporary shared memory for the tiles.
            __shared__ float p_tile[TILE_DIM][TILE_DIM];
            __shared__ float v_tile[TILE_DIM][TILE_DIM];

            // Each thread is responsible for one element of the output matrix pv.
            target_row = threadIdx.y;
            target_col = threadIdx.x;

            // Use a register to store the accumulator for the result.
            float pv_sum = 0.0f;

            // Loop over the tiles along the common dimension b_c.
            for (int tile_idx = 0; tile_idx < (b_c / TILE_DIM); ++tile_idx) {
                // --- Step 1: Cooperatively load tiles from shared memory ---
                // Each thread loads one element from s and one from v into the tile arrays.
                
                // Load element for p_tile (from the 's' matrix)
                p_tile[target_row][target_col] = s[target_row][tile_idx * TILE_DIM + target_col];

                // Load element for v_tile
                v_tile[target_row][target_col] = v[tile_idx * TILE_DIM + target_row][target_col];

                // Synchronize to ensure both tiles are fully loaded before proceeding.
                __syncthreads();

                // --- Step 2: Compute matrix multiplication on the tiles ---
                // Each thread calculates a partial sum by iterating through the tile's dimension.
                // This is extremely fast as it only uses registers and shared memory.
                for (int i = 0; i < TILE_DIM; ++i) {
                    pv_sum += p_tile[target_row][i] * v_tile[i][target_col];
                }

                // Synchronize to ensure all threads have finished with the current tile
                // before loading the next one.
                __syncthreads();
            }

            // --- Step 3: Write the final result to the pv matrix ---
            // This assumes b_r and d are equal to TILE_DIM.
            pv[target_row][target_col] = pv_sum;

            // A final sync to ensure pv is fully written before subsequent steps.
            __syncthreads();

            // Update O, L, M
            for (int row = threadIdx.y; row < b_r; row += blockDim.y) {
                float m_old_val = m[row];
                float l_old_val = l[row];
                float m_new_val = m_new[row];
                float l_new_val = l_new[row];
                float m_up_val = m_up[row];

                for (int col = threadIdx.x; col < d; col += blockDim.x) {
                     o[row][col] = (l_old_val * __expf(m_old_val - m_new_val) * o[row][col] + __expf(m_up_val - m_new_val) * pv[row][col]) / l_new_val;
                }
            }
             if (threadIdx.x == 0) {
                for (int row = threadIdx.y; row < b_r; row += blockDim.y) {
                    l[row] = l_new[row];
                    m[row] = m_new[row];
                }
            }
            __syncthreads();


            // --- COALESCED MEMORY WRITE for O, L, M ---
            // Fused writing of O
            for (int i = b_idx; i < b_r * d; i += block_size) {
                 O[gmem_block_offset_q + i] = o[i / d][i % d];
            }
            // Fused writing of L and M
            if (threadIdx.y == 0) {
                 for(int i = threadIdx.x; i < b_r; i += blockDim.x){
                    L[gmem_block_offset_lm + i] = l[i];
                    M[gmem_block_offset_lm + i] = m[i];
                }
            }
            __syncthreads();
        }
    }
}

template<const int num_threads_x,
         const int num_threads_y,
         const int N,
         const int d,
         const int b_r,
         const int b_c>
__global__ void fa_op3(float* Q, float* K, float* V, float* O, float* L, float* M) {

    // --- SHARED MEMORY: MINIMAL ALLOCATION ---
    // We only need shared memory for the tiles and the intermediate results (S, l, m).
    constexpr int TILE_DIM_X = num_threads_x;
    constexpr int TILE_DIM_Y = num_threads_y;

    // Tiles for Q*K^T multiplication
    __shared__ float q_tile[TILE_DIM_Y][d]; // Holds a tile of Q
    __shared__ float k_tile[TILE_DIM_X][d]; // Holds a tile of K

    // Tiles for P*V multiplication
    __shared__ float v_tile[TILE_DIM_X][d]; // Holds a tile of V
    
    // Intermediate attention score matrix S (and reused for P)
    __shared__ float s[TILE_DIM_Y][TILE_DIM_X];

    // Statistics for the current row block
    __shared__ float l_s[TILE_DIM_Y];
    __shared__ float m_s[TILE_DIM_Y];


    const int T_r = N / b_r;
    const int T_c = N / b_c;
    const float scale = 1.0f / sqrtf((float)d);

    // This thread block is responsible for the `blockIdx.x`-th block of rows of the output.
    const int iter = blockIdx.x; 

    // --- Initialize Ouput, L, and M from Global Memory ---
    // Each thread loads its portion of the final output row into registers.
    // This assumes d is a multiple of TILE_DIM_X. Each thread handles d / TILE_DIM_X values.
    constexpr int D_PER_THREAD = d / TILE_DIM_X;
    float o_acc[D_PER_THREAD];
    
    const int row_idx = iter * b_r + threadIdx.y;
    const int gmem_offset_o = row_idx * d + threadIdx.x * D_PER_THREAD;
    const int gmem_offset_lm = row_idx;

    for (int i = 0; i < D_PER_THREAD; ++i) {
        o_acc[i] = O[gmem_offset_o + i];
    }
    float l_acc = L[gmem_offset_lm];
    float m_acc = M[gmem_offset_lm];

    // Outer loop over blocks of K and V
    for (int j = 0; j < T_c; j++) {
        // --- Load a block of Q, K, V directly into shared memory tiles ---
        // This is a coalesced load. Each thread loads D_PER_THREAD elements.
        int gmem_offset_q = row_idx * d + threadIdx.x * D_PER_THREAD;
        int gmem_offset_k = (j * b_c + threadIdx.y) * d + threadIdx.x * D_PER_THREAD;
        int gmem_offset_v = (j * b_c + threadIdx.y) * d + threadIdx.x * D_PER_THREAD;

        for (int i = 0; i < D_PER_THREAD; ++i) {
            q_tile[threadIdx.y][threadIdx.x * D_PER_THREAD + i] = Q[gmem_offset_q + i];
            k_tile[threadIdx.y][threadIdx.x * D_PER_THREAD + i] = K[gmem_offset_k + i];
            v_tile[threadIdx.y][threadIdx.x * D_PER_THREAD + i] = V[gmem_offset_v + i];
        }
        __syncthreads();

        // --- 1. Compute S = Q_tile * K_tile^T ---
        float s_sum = 0.0f;
        for (int k_dim = 0; k_dim < d; ++k_dim) {
            s_sum += q_tile[threadIdx.y][k_dim] * k_tile[threadIdx.x][k_dim];
        }
        s[threadIdx.y][threadIdx.x] = s_sum * scale;
        __syncthreads();

        // --- 2. Compute Row-wise Statistics (m_up, l_up) ---
        float row_max = -FLT_MAX;
        for (int c = 0; c < TILE_DIM_X; c++) {
            row_max = fmaxf(row_max, s[threadIdx.y][c]);
        }
        
        float row_sum_exp = 0.0f;
        for (int c = 0; c < TILE_DIM_X; c++) {
            float p_val = __expf(s[threadIdx.y][c] - row_max);
            s[threadIdx.y][c] = p_val; // Update s in-place to become P
            row_sum_exp += p_val;
        }
        __syncthreads();


        // --- 3. Update Global Statistics L and M ---
        float m_new = fmaxf(m_acc, row_max);
        float l_new = __expf(m_acc - m_new) * l_acc + __expf(row_max - m_new) * row_sum_exp;

        // --- 4. Fused P*V and Output Update ---
        // Each thread computes its D_PER_THREAD slice of the new output.
        for (int i = 0; i < D_PER_THREAD; ++i) {
            float pv_sum = 0.0f;
            for (int k_dim = 0; k_dim < TILE_DIM_X; ++k_dim) {
                pv_sum += s[threadIdx.y][k_dim] * v_tile[k_dim][threadIdx.x * D_PER_THREAD + i];
            }
            // Update the output accumulator in-place
            o_acc[i] = (l_acc * __expf(m_acc - m_new) * o_acc[i] + __expf(row_max - m_new) * pv_sum) / l_new;
        }
        
        // Update l and m accumulators for the next iteration
        l_acc = l_new;
        m_acc = m_new;
    }

    // --- Write final results back to Global Memory ---
    for (int i = 0; i < D_PER_THREAD; ++i) {
        O[gmem_offset_o + i] = o_acc[i];
    }
    L[gmem_offset_lm] = l_acc;
    M[gmem_offset_lm] = m_acc;
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
template __global__ void fa_op1<512, 2, 2048, 512, 4, 4>(float*, float*, float*, float*, float*, float*);
template __global__ void fa_op2<512, 2, 2048, 512, 4, 4>(float*, float*, float*, float*, float*, float*);
template __global__ void fa_op3<512, 2, 2048, 512, 4, 4>(float*, float*, float*, float*, float*, float*); 