#include "common/common.h"
#include <cooperative_groups.h>
#include <cfloat>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

// Declare fa kernel function
template<const int num_threads_x,
        const int num_threads_y,
        const int N,
        const int d,
        const int b_r,
        const int b_c
>
__global__ void fa(float* Q, float* K, float* V, float* O, float* L, float* M);

// Declare fa_v2 kernel function
template<
    const int N,
    const int d,
    const int b_r,
    const int b_c
>
__global__ void fa_v2(float* Q, float* K, float* V, float* O, float* L, float* M);

// Calculate FLOPS for attention computation
// Each attention operation involves:
// 1. QK^T: N * N * d multiplications and additions (2 * N^2 * d)
// 2. Softmax: 3 * N^2 operations (exp, sum, division)
// 3. Softmax * V: N^2 * d multiplications and additions (2 * N^2 * d)
// Total: 4 * N^2 * d + 3 * N^2 operations
double calculate_attention_flops(int n, int d) {
    double qk_flops = 2.0 * n * n * d;        // QK^T operations
    double softmax_flops = 3.0 * n * n;       // Softmax operations
    double output_flops = 2.0 * n * n * d;    // Softmax * V operations
    return qk_flops + softmax_flops + output_flops;
}

// Calculate GFLOPS (Giga Floating Point Operations Per Second)
double calculate_gflops(double flops, double time_ms) {
    return (flops / (time_ms / 1000.0)) / 1e9;
}

// CPU reference implementation
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

// Check if results are correct
bool verify_results(float* gpu_result, float* cpu_result, int size, float tolerance = 1e-4) {
    int errors = 0;
    for (int i = 0; i < size; i++) {
        float diff = fabs(gpu_result[i] - cpu_result[i]);
        if (diff > tolerance) {
            errors++;
            if (errors < 10) { // Only print first 10 errors
                printf("Mismatch at index %d: GPU = %f, CPU = %f, diff = %f\n", 
                       i, gpu_result[i], cpu_result[i], diff);
            }
        }
    }
    
    if (errors > 0) {
        printf("Found %d errors out of %d elements!\n", errors, size);
        return false;
    }
    
    return true;
}

int main(int argc, char** argv) {
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    CHECK(cudaSetDevice(dev));
    
    printf("Device Name: %s\n", deviceProp.name);
    printf("Compute Capability: %d.%d\n", deviceProp.major, deviceProp.minor);
    printf("Shared Memory: %lu KB\n", deviceProp.sharedMemPerBlock / 1024);
    
    // Data size settings
    constexpr int d = 512;     // Feature dimension
    constexpr int N = 2048;    // Sequence length
    
    // Initialize host data Q K V
    float* h_q = (float*)malloc(d * N * sizeof(float));
    float* h_k = (float*)malloc(d * N * sizeof(float));
    float* h_v = (float*)malloc(d * N * sizeof(float));
    
    // Set random seed
    srand(time(NULL));
    
    // Randomly initialize input data
    for (int i = 0; i < N * d; i++) {
        h_q[i] = (rand() % N) / (float)d;
        h_k[i] = (rand() % N) / (float)d;
        h_v[i] = (rand() % N) / (float)d;
    }
    
    // Initialize host output data
    float* h_O1 = (float*)malloc(d * N * sizeof(float));  // fa output
    float* h_O2 = (float*)malloc(d * N * sizeof(float));  // fa_v2 output
    float* h_L1 = (float*)malloc(N * sizeof(float));
    float* h_L2 = (float*)malloc(N * sizeof(float));
    float* h_M1 = (float*)malloc(N * sizeof(float));
    float* h_M2 = (float*)malloc(N * sizeof(float));
    
    // Initialize to 0 and -FLT_MAX
    for (int i = 0; i < N * d; i++) {
        h_O1[i] = 0;
        h_O2[i] = 0;
    }
    for (int i = 0; i < N; i++) {
        h_L1[i] = 0;
        h_L2[i] = 0;
        h_M1[i] = -FLT_MAX;
        h_M2[i] = -FLT_MAX;
    }
    
    // Initialize device data
    float *dev_q, *dev_k, *dev_v;
    float *dev_O1, *dev_L1, *dev_M1;  // fa output
    float *dev_O2, *dev_L2, *dev_M2;  // fa_v2 output
    
    // Allocate device memory
    CHECK(cudaMalloc((float**)&dev_q, d * N * sizeof(float)));
    CHECK(cudaMalloc((float**)&dev_k, d * N * sizeof(float)));
    CHECK(cudaMalloc((float**)&dev_v, d * N * sizeof(float)));
    
    CHECK(cudaMalloc((float**)&dev_O1, d * N * sizeof(float)));
    CHECK(cudaMalloc((float**)&dev_L1, N * sizeof(float)));
    CHECK(cudaMalloc((float**)&dev_M1, N * sizeof(float)));
    
    CHECK(cudaMalloc((float**)&dev_O2, d * N * sizeof(float)));
    CHECK(cudaMalloc((float**)&dev_L2, N * sizeof(float)));
    CHECK(cudaMalloc((float**)&dev_M2, N * sizeof(float)));
    
    // Transfer data to device
    CHECK(cudaMemcpy(dev_q, h_q, d * N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dev_k, h_k, d * N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dev_v, h_v, d * N * sizeof(float), cudaMemcpyHostToDevice));
    
    CHECK(cudaMemcpy(dev_O1, h_O1, d * N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dev_L1, h_L1, N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dev_M1, h_M1, N * sizeof(float), cudaMemcpyHostToDevice));
    
    CHECK(cudaMemcpy(dev_O2, h_O2, d * N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dev_L2, h_L2, N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dev_M2, h_M2, N * sizeof(float), cudaMemcpyHostToDevice));
    
    // Calculate b_r and b_c
    constexpr int M = 10000;  // Shared memory size
    constexpr int b_c = 4;    // Column block size
    constexpr int b_r = 4;    // Row block size
    
    // Set thread block configuration
    dim3 block(512, 2);
    
    // Create CUDA events to measure execution time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float fa_time = 0;     // fa execution time
    float fa_v2_time = 0;  // fa_v2 execution time
    
    // === Execute fa kernel ===
    printf("\n==== Running fa kernel ====\n");
    cudaEventRecord(start);
    
    fa<512, 2, N, d, b_r, b_c><<<1, block>>>(dev_q, dev_k, dev_v, dev_O1, dev_L1, dev_M1);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&fa_time, start, stop);
    printf("fa kernel execution time: %.3f ms\n", fa_time);
    
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    
    // Copy results back to host
    CHECK(cudaMemcpy(h_O1, dev_O1, d * N * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_L1, dev_L1, N * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_M1, dev_M1, N * sizeof(float), cudaMemcpyDeviceToHost));
    
    // === Execute fa_v2 kernel ===
    printf("\n==== Running fa_v2 kernel ====\n");
    cudaEventRecord(start);
    
    fa_v2<N, d, b_r, b_c><<<32, block>>>(dev_q, dev_k, dev_v, dev_O2, dev_L2, dev_M2);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&fa_v2_time, start, stop);
    printf("fa_v2 kernel execution time: %.3f ms\n", fa_v2_time);
    
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    
    // Copy results back to host
    CHECK(cudaMemcpy(h_O2, dev_O2, d * N * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_L2, dev_L2, N * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_M2, dev_M2, N * sizeof(float), cudaMemcpyDeviceToHost));
    
    // === Calculate CPU reference result ===
    printf("\n==== Computing CPU reference result ====\n");
    clock_t cpu_start = clock();
    
    float* cpu_result = h_attention(h_q, h_k, h_v, N, N, d, d);
    
    clock_t cpu_end = clock();
    double cpu_time = (double)(cpu_end - cpu_start) / CLOCKS_PER_SEC * 1000.0;
    printf("CPU computation time: %.3f ms\n", cpu_time);
    
    // === Verify results ===
    printf("\n==== Result Verification ====\n");
    printf("Verifying fa results...\n");
    bool fa_correct = verify_results(h_O1, cpu_result, N * d);
    
    printf("Verifying fa_v2 results...\n");
    bool fa_v2_correct = verify_results(h_O2, cpu_result, N * d);
    
    // Calculate FLOPS
    double total_flops = calculate_attention_flops(N, d);
    double fa_gflops = calculate_gflops(total_flops, fa_time);
    double fa_v2_gflops = calculate_gflops(total_flops, fa_v2_time);
    double cpu_gflops = calculate_gflops(total_flops, cpu_time);
    
    // === Performance comparison ===
    printf("\n==== Performance Comparison ====\n");
    printf("Problem size: N=%d, d=%d\n", N, d);
    printf("Total floating-point operations: %.2f GFLOPs\n", total_flops / 1e9);
    printf("\n");
    printf("CPU computation time: %.3f ms (%.2f GFLOPS)\n", cpu_time, cpu_gflops);
    printf("fa kernel time: %.3f ms (%.2f GFLOPS, vs CPU speedup: %.2fx)\n", 
           fa_time, fa_gflops, cpu_time / fa_time);
    printf("fa_v2 kernel time: %.3f ms (%.2f GFLOPS, vs CPU speedup: %.2fx)\n", 
           fa_v2_time, fa_v2_gflops, cpu_time / fa_v2_time);
    printf("fa_v2 vs fa speedup: %.2fx\n", fa_time / fa_v2_time);
    
    // === Result sample comparison ===
    printf("\n==== Result Sample Comparison (First 10 elements) ====\n");
    for (int i = 0; i < 10; i++) {
        printf("idx %d: CPU=%.6f, fa=%.6f, fa_v2=%.6f\n", 
               i, cpu_result[i], h_O1[i], h_O2[i]);
    }
    
    // Free resources
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    free(h_q);
    free(h_k);
    free(h_v);
    free(h_O1);
    free(h_O2);
    free(h_L1);
    free(h_L2);
    free(h_M1);
    free(h_M2);
    free(cpu_result);
    
    cudaFree(dev_q);
    cudaFree(dev_k);
    cudaFree(dev_v);
    cudaFree(dev_O1);
    cudaFree(dev_L1);
    cudaFree(dev_M1);
    cudaFree(dev_O2);
    cudaFree(dev_L2);
    cudaFree(dev_M2);
    
    return 0;
} 