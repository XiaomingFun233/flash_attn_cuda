Device Name: NVIDIA GeForce RTX 4070
Compute Capability: 8.9
Shared Memory: 48 KB

==== Running fa kernel ====
fa kernel execution time: 1407.701 ms

==== Running fa_v2 kernel ====
fa_v2 kernel execution time: 1190.436 ms

==== Computing CPU reference result ====
CPU computation time: 25399.229 ms

==== Result Verification ====
Verifying fa results...
Verifying fa_v2 results...

==== Performance Comparison ====
CPU computation time: 25399.229 ms
fa kernel time: 1407.701 ms (vs CPU speedup: 18.04x)
fa_v2 kernel time: 1190.436 ms (vs CPU speedup: 21.34x)
fa_v2 vs fa speedup: 1.18x

==== Result Sample Comparison (First 10 elements) ====
idx 0: CPU=2.097856, fa=2.097856, fa_v2=2.097856
idx 1: CPU=2.262712, fa=2.262715, fa_v2=2.262713
idx 2: CPU=2.439600, fa=2.439606, fa_v2=2.439607
idx 3: CPU=2.233348, fa=2.233350, fa_v2=2.233351
idx 4: CPU=2.369487, fa=2.369489, fa_v2=2.369488
idx 5: CPU=2.353722, fa=2.353722, fa_v2=2.353724
idx 6: CPU=2.299703, fa=2.299704, fa_v2=2.299704
idx 7: CPU=2.218644, fa=2.218646, fa_v2=2.218645
idx 8: CPU=1.758039, fa=1.758038, fa_v2=1.758038
idx 9: CPU=1.841797, fa=1.841796, fa_v2=1.841797