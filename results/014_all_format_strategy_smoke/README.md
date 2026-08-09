# All-format decoder strategy smoke test

This cumulative experiment exhaustively or statistically validates every
promising exact FP64 decoder strategy added after the E2M5/E3M4 study. Each
format commit extends the same executable and CSV inventory.

The smoke run checks decoder correctness, packed lane order, and launches a
small fused DOT and GEMV for every registered strategy. Its timings, if any,
are not performance conclusions; full DOT/GEMV benchmarking follows after the
strategy inventory is stable.

Submit from the repository root while choosing the GPU node explicitly:

```bash
sbatch --wait --nodelist=gpu-nvidia-h200-1-studvm-2 \
  scripts/run_all_format_strategy_smoke_h200.sbatch
```

The batch script itself does not select a node. Timestamped artifacts are
written below this directory.
