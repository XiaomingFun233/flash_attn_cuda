#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cfloat>

template<const int num_threads_x,
        const int num_threads_y,
        const int N,
        const int d,
        const int b_r,
        const int b_c
>
__global__ void fa(float* Q, float* K, float* V, float* O, float* L, float* M){
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
                    // *** BUG FIX 1: 增加了缩放步骤 ***
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
                    // 1. 找最大值 m_up
                    float row_max = -FLT_MAX;
                    for (int col = 0; col < b_c; col++) {
                        if (s[row][col] > row_max) {
                            row_max = s[row][col];
                        }
                    }
                    m_up[row] = row_max;

                    // 2. 计算 P_ij 并求和得到 l_up
                    float row_sum_exp = 0.0f;
                    for (int col = 0; col < b_c; col++) {
                        float p_val = __expf(s[row][col] - row_max);
                        s[row][col] = p_val; // 将 s 矩阵原地更新为 P 矩阵
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
                    // 更新公式
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

torch::Tensor forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    // 确定内核参数
    const int num_threads_x = 512;
    const int num_threads_y = 2;
    const int b_r = 4;
    const int b_c = 4;
    
    // 获取输入尺寸
    const int B = Q.size(0); const int nh = Q.size(1);
    const int N = Q.size(2); const int d = Q.size(3);
    
    // 初始化输出和中间张量
    auto O = torch::zeros_like(Q);
    auto L = torch::zeros({B, nh, N});
    auto M = torch::full({B, nh, N}, -INFINITY);
    torch::Device device(torch::kCUDA);
    L = L.to(device); M = M.to(device);
    
    // 打印SRAM使用信息
    const int sram_size = (b_r * d * sizeof(float)) + (b_c * d * sizeof(float)) + 
                         (b_c * d * sizeof(float)) + (b_r * d * sizeof(float)) + 
                         (b_r * b_c * sizeof(float)) + (b_r * sizeof(float)) + 
                         (b_r * sizeof(float));
    int max_sram_size;
    cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    printf("最大共享内存: %d字节, 请求共享内存: %d字节\n", max_sram_size, sram_size);
    
    // 设置CUDA内核启动参数
    dim3 grid_dim(B, nh);  // batch_size x num_heads
    dim3 block_dim(num_threads_x, num_threads_y);
    
    // 为每个批次和头部启动内核
    for (int b = 0; b < B; b++) {
        for (int h = 0; h < nh; h++) {
            // 获取当前批次和头部的数据指针
            float* q_ptr = Q.index({b, h}).data_ptr<float>();
            float* k_ptr = K.index({b, h}).data_ptr<float>();
            float* v_ptr = V.index({b, h}).data_ptr<float>();
            float* o_ptr = O.index({b, h}).data_ptr<float>();
            float* l_ptr = L.index({b, h}).data_ptr<float>();
            float* m_ptr = M.index({b, h}).data_ptr<float>();
            
            // 启动fa内核
            fa<num_threads_x, num_threads_y, 2048, 512, b_r, b_c><<<1, block_dim>>>(
                q_ptr, k_ptr, v_ptr, o_ptr, l_ptr, m_ptr
            );
        }
    }
    
    // 检查CUDA错误
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA错误: %s\n", cudaGetErrorString(error));
    }
    
    return O;
} 