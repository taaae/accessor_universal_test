# 002: cuBLAS Sdot N=2^27 profile

This experiment profiles exactly one post-warmup `cublasSdot` call at
`N=134217728` on an NVIDIA H200 NVL using Nsight Compute 2025.4.1.

- Command: `sbatch --wait scripts/profile_cublas_sdot_h200.sbatch`
- Main kernel: 2112 blocks x 128 threads, 256.576 us, 4.1945 TB/s DRAM
- Final kernel: 1 block x 128 threads, 5.728 us
- Measured physical traffic: approximately 1.07623 GB
- Algorithmic input traffic: 1.07374 GB

The valid performance time remains the unprofiled 0.262695 ms from experiment
001. The 1599.05 ms value in this run's `_benchmark.csv` is contaminated by
profiler instrumentation and 12 replay passes per kernel; it must not be used
as a performance measurement. Future profile-mode runs emit `nan` timing.
