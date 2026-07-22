# 001: cuBLAS DOT baseline

This experiment establishes performance and accuracy references for
`cublasSdot` and `cublasDdot` before adding custom kernels or accessors.

- GPU: NVIDIA H200 NVL (`sm_90`)
- CUDA/cuBLAS: 13.1 / 13.2.1
- Performance data: signed deterministic vectors, `N=2^10` through `2^27`
- Accuracy data: positive and signed deterministic vectors through `N=2^20`
- Command: `./scripts/run_cublas_baseline.sh`

The primary FP32 reference is `N=2^27`: 0.262695 ms, 4.087 TB/s algorithmic
bandwidth, and 1.022 TFLOP/s. The large-vector cases are HBM-bandwidth bound.
The Slurm log records the full build and execution environment.
