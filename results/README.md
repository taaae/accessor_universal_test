# Experiment results

Results are grouped by experiment rather than by file type or execution date.
Each directory contains a short README describing what was tested and how its
artifacts should be interpreted.

| ID | Experiment | Purpose |
|---:|---|---|
| 001 | `001_cublas_baseline` | Establish cuBLAS DOT performance and accuracy baselines. |
| 002 | `002_cublas_sdot_n2p27_profile` | Profile the HBM-bound FP32 reference case with Nsight Compute. |
| 003 | `003_accessor_dot` | Compare raw and 1D accessor loads with FP32/FP64 arithmetic over FP32 storage. |
| 004 | `004_storage_formats` | Validate scalar, packed-2, and packed-4 FP64 decoders for the shortlisted storage formats. |
| 005 | `005_storage_kernels` | Validate generic FP64 DOT and row-major GEMV kernels for every storage format and x1/x2/x4 load path. |
| 006 | `006_accuracy_model` | Predict scalar, DOT, and fixed-`M` GEMV storage error for every format under U(0,1) and N(0,1). |
| 007 | `007_gpu_accuracy_simulation` | Validate analytical storage and kernel-error predictions with statistically replicated H200 DOT/GEMV simulations. |
| 008 | `008_storage_performance` | Separate decode, stream, DOT/GEMV timing and collect hardware/algorithmic roofline inputs for scalar and packed access. |
| 009 | `009_e2e3_decoder_optimization` | Compare the initial E2M5/E3M4 scalar and packed branchless/LUT decoders. |
| 010 | `010_e2e3_strategy_smoke` | Validate the expanded E2M5/E3M4 decoder strategy inventory and screen preliminary kernel timings. |
| 011 | `011_e2e3_strategy_performance` | Compare complete DOT/GEMV time for all E2M5/E3M4 decoder strategies against raw FP64. |

Timestamped repetitions of an unchanged experiment belong in the existing
directory. Create the next numbered directory when changing the kernel,
storage format, arithmetic type, or main experimental question. Keep build
artifacts outside `results/`.
