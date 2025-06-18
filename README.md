easy naive flash attention without optimization
# flash attention

如果要编译运行主程序/If you want to compile and run the main program

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