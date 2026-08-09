# 013: Expanded E2M5/E3M4 strategy performance

This experiment is the full performance sweep for the strategy inventory
validated by experiment 012. Each case contains 85 variants: raw FP64 storage,
42 E2M5 strategies, and 42 E3M4 strategies. Arithmetic and accumulation remain
FP64 for every storage strategy.

The deterministic default grid is:

- distributions: U(0,1) and N(0,1), without rescaling;
- DOT: `N=2^12,2^16,2^20,2^24,2^27`;
- GEMV: fixed `M=1024`, with `N=2^8,2^10,2^12,2^14,2^16`.

Each variant receives 10 warmups and 15 measured samples, organized as three
rounds of five. Every sample repeats the complete kernel until its aggregate
duration is approximately 15 ms. Variant order is reshuffled for each sample.
DOT time includes the map/reduce and final-reduction kernels. Raw FP64 is timed
inside the same randomized sweep rather than copied from another experiment.

Before measuring, the job repeats the exhaustive 256-code decoder validation
and small checked DOT/GEMV cases. The summarizer then requires exactly all 85
variants at every requested size and distribution, so a partial result cannot
be accepted as a complete run.

This is a performance experiment. With no rescaling, sufficiently large
N(0,1) inputs can overflow E2M5; that expected range limitation does not alter
the timing methodology and will be treated separately in accuracy analysis.

Submit from the repository root and select the GPU in the command:

```bash
sbatch --wait --nodelist=gpu-nvidia-h200-1-studvm-2 \
  scripts/run_e2e3_expanded_strategy_benchmark_h200.sbatch
```

Each execution writes a timestamped run directory here. The main artifacts
are `timing_samples.csv`, `timing_summary.csv`, `case_winners.csv`,
`strategy_inventory.csv`, and `cuda_resource_usage.txt`.
