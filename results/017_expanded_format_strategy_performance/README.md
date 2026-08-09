# Experiment 017: expanded format strategy performance

This experiment performs the same complete FP64-arithmetic DOT and GEMV timing
sweep as experiment 015 for the 22 formats introduced by experiment 016.

The default deterministic sweep uses U(0,1) and N(0,1):

- DOT: `N=2^12,2^16,2^20,2^24,2^27`;
- GEMV: fixed `M=1024`, with `N=2^8,2^10,2^12,2^14,2^16`.

Every registered strategy receives 10 warmups and 15 randomized-order timing
samples. Each sample repeats the complete kernel until its aggregate duration
is approximately 15 ms. The raw FP64 baseline is remeasured alongside every
format. Two- and four-bit inputs remain genuinely bit-dense throughout DOT and
GEMV; reported main-array byte counts use their physical 2-/4-bit sizes.

Five bit-width-specific executables bound CUDA compilation memory. Their raw
samples are preserved separately and merged into the same `timing_samples.csv`,
`timing_summary.csv`, `case_winners.csv`, and `strategy_rankings.csv` schema as
experiment 015.

Submit from the repository root while selecting the GPU node explicitly:

```bash
sbatch --wait --nodelist=gpu-nvidia-h200-1-studvm-2 \
  scripts/run_expanded_format_strategy_benchmark_h200.sbatch
```
