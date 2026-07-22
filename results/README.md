# Experiment results

Results are grouped by experiment rather than by file type or execution date.
Each directory contains a short README describing what was tested and how its
artifacts should be interpreted.

| ID | Experiment | Purpose |
|---:|---|---|
| 001 | `001_cublas_baseline` | Establish cuBLAS DOT performance and accuracy baselines. |
| 002 | `002_cublas_sdot_n2p27_profile` | Profile the HBM-bound FP32 reference case with Nsight Compute. |

Timestamped repetitions of an unchanged experiment belong in the existing
directory. Create the next numbered directory when changing the kernel,
storage format, arithmetic type, or main experimental question. Keep build
artifacts outside `results/`.
