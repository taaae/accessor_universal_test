# Experiment 022: LNS conversion and fused multiplication

This experiment benchmarks the LNS inventory documented in
`docs/lns_strategy_benchmark.md` with FP32 and FP64 arithmetic.  Ordinary
decode/decode/multiply and fused log-add/decode kernels are reported
separately.  Dense, padded, per-thread packet, and cooperative-shuffle access
remain independent strategy dimensions.

The job first validates and smoke-tests every compiled strategy, screens all
variants at representative sizes, and performs full N sweeps only for the
best strategies in each arithmetic/access group.  Raw FP32 and FP64 anchors
are remeasured throughout the same allocation.

The batch script does not select a GPU.  Submit from the repository root only
after checking availability and naming the approved H200 explicitly:

```bash
STOP_AFTER=smoke sbatch --wait --nodelist=<approved-h200> \
  scripts/run_lns_strategy_benchmark_h200.sbatch
```

The hard job limit is eight hours.  The first run should stop after smoke so
compile and correctness failures cannot consume a full benchmark allocation.
