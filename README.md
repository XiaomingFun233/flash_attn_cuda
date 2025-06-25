# Flash Attention CUDA Implementation

Easy naive flash attention without optimization and Flash Attention V2 with optimizations.

## Compilation / 编译

### Main Program / 主程序

To compile and run the main program / 编译运行主程序:

```bash
nvcc -o flash flash.cu
```

To run the program / 运行程序:

```bash
./flash
```

### Benchmark Program / 基准测试程序

To compile the benchmark program that compares fa and fa_v2 / 编译比较fa和fa_v2性能的基准测试程序:

```bash
nvcc -o ben benchmark.cu flash_kernels.cu
```

To run the benchmark / 运行基准测试:

```bash
./ben
```

To save the benchmark results to a report file / 将基准测试结果保存到报告文件:

```bash
./ben > reports/benchmark_result.md
```

## Performance Analysis / 性能分析

### Nsight System / 系统性能分析

To analyze using Nsight System / 使用Nsight System进行分析:

```bash
nsys profile -o --stats=true /data/coding/flash_attn_cuda/reports ./flash
```

### Nsight Compute / 计算性能分析

To perform further analysis using Nsight Compute / 使用Nsight Compute进行进一步分析优化:

```bash
ncu --set detailed -o /your_own_path/flash_attn_cuda/reports/ncu_result ./flash
```

Note: You may encounter permission issues when using Nsight Compute in a Docker container / 注意：在Docker容器中使用Nsight Compute可能会遇到权限问题:

```
==PROF== Connected to process 1722 (/...../flash_attn_cuda/flash)
==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU Performance Counters on the target device 0. For instructions on enabling permissions and to get more information see [https://developer.nvidia.com/ERR_NVGPUCTRPERM]
```

## Benchmark Results / 基准测试结果

The benchmark compares three implementations:
基准测试比较了三种实现:

1. CPU reference implementation / CPU参考实现
2. fa (Flash Attention) CUDA implementation / fa (Flash Attention) CUDA实现
3. fa_v2 (Flash Attention V2) optimized CUDA implementation / fa_v2 (Flash Attention V2) 优化CUDA实现

The benchmark measures:
基准测试测量了:

- Execution time / 执行时间
- GFLOPS (Giga Floating Point Operations Per Second) / 每秒十亿浮点运算次数
- Accuracy compared to CPU reference / 与CPU参考实现的精度比较

Example results show that fa_v2 achieves approximately 16-18% better performance than fa, and both GPU implementations significantly outperform the CPU reference implementation.
示例结果显示，fa_v2比fa的性能提高约16-18%，两种GPU实现都远优于CPU参考实现。

```jsx
nvcc -o flash flash.cu
```

要运行判断正确的程序使用/To run the correct program use

```jsx
./your_path/flash
```

只进行了使用nsight system 的分析/Only analysis using nsight system was performed

```jsx
nsys profile -o --stats=true /data/coding/flash_attn_cuda/reports ./flash
```

因为在使用nsight compute 进行进一步分析优化/Use nsight compute for further analysis and optimization

```jsx
ncu --set detailed -o /your_own_path/flash_attn_cuda/reports/ncu_result ./flash
```

遇到了权限不足的问题docker 容器权限不支持/Encountered a problem of insufficient permissions. Docker container permissions are not supported.

```jsx
==PROF== Connected to process 1722 (/...../flash_attn_cuda/flash)
==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU Performance Counters on the target device 0. For instructions on enabling permissions and to get more information see [https://developer.nvidia.com/ERR_NVGPUCTRPERM](https://developer.nvidia.com/ERR_NVGPUCTRPERM)
```