# 011: E2M5/E3M4 decoder strategy performance

This experiment measures complete FP64-arithmetic DOT and GEMV kernel time for
every decoder strategy introduced in experiment 010. The expanded benchmark
contains 85 variants per case: raw FP64 storage, 42 E2M5 strategies, and 42
E3M4 strategies.

The default sweep uses deterministic U(0,1) and N(0,1) inputs:

- DOT: `N=2^12,2^16,2^20,2^24,2^27`;
- GEMV: fixed `M=1024`, with `N=2^8,2^10,2^12,2^14,2^16`.

Each variant receives 10 warmups followed by 3 rounds of 5 samples. A sample
repeats the complete kernel until its aggregate duration is approximately 15
ms. Variant order is reshuffled for each sample to reduce temperature, clock,
and fixed-order bias. DOT timings include both the main map/reduce kernel and
the final reduction kernel.

Before the full sweep, the script runs the experiment-010 decoder and kernel
validation on a small case. Inputs are generated directly on the GPU with
cuRAND and the identical FP64 source arrays are encoded into E2M5 and E3M4.

Submit from the repository root and select the GPU in the command:

```bash
sbatch --wait --nodelist=gpu-nvidia-h200-1-studvm-2 \
  scripts/run_e2e3_strategy_benchmark_h200.sbatch
```

Each run is stored in a timestamped directory here. `timing_samples.csv` is
the lossless timing dataset. `timing_summary.csv` contains medians, 5th/95th
percentiles, variation, FP64 speedup, and within-format rank. `case_winners.csv`
contains the fastest strategy for each format and problem size.
