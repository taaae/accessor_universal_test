# 015: all-format decoder strategy performance

This experiment measures complete FP64-arithmetic DOT and GEMV time for every
promising decoder strategy registered for the 14 formats added after the
dedicated E2M5/E3M4 study. E2M5 and E3M4 remain in experiment 013 and are not
remeasured here.

The default deterministic sweep uses U(0,1) and N(0,1):

- DOT: `N=2^12,2^16,2^20,2^24,2^27`;
- GEMV: fixed `M=1024`, with `N=2^8,2^10,2^12,2^14,2^16`.

Each candidate receives 10 warmups and 15 measurements (three rounds of five).
A measurement repeats the complete kernel until its aggregate time is about 15
ms. Candidate order is reshuffled for every sample. DOT time includes the main
map/reduce and final reduction kernels.

Formats are processed one at a time to bound device memory. A raw FP64 baseline
is remeasured and randomized alongside every format, so each future per-format
graph has a temporally local comparison rather than one baseline taken at the
start of a long job.

Before the full sweep, the runner performs exhaustive/sampled decoder smoke
validation and a one-size timing/schema validation for every format. The job
fails if any registered strategy, workload size, sample coordinate, or FP64
baseline is missing.

Submit from the repository root while choosing the GPU node explicitly:

```bash
sbatch --wait --nodelist=gpu-nvidia-h200-1-studvm-2 \
  scripts/run_all_format_strategy_benchmark_h200.sbatch
```

Timestamped runs are stored here. `timing_samples.csv` preserves every CUDA
event measurement. `timing_summary.csv` contains medians, p05/p95, variation,
bandwidth, throughput, FP64 speedup, and within-format rank.
`case_winners.csv` records the fastest strategy for every format and size;
`strategy_rankings.csv` aggregates performance across sizes for later report
and graph generation.
