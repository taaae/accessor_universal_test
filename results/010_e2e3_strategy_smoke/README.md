# 010: E2M5/E3M4 decoder strategy smoke test

This preliminary experiment checks that the expanded E2M5/E3M4 decoder set is
correct and does not contain an obviously pathological DOT or GEMV
implementation. Its short timing samples are useful for screening only; they
are not the final performance benchmark.

Each format has 27 strategies:

- generic FP64, branchless FP32, FP32 LUT, FP64 LUT, and FP64-prefix LUT at
  x1, x2, x4, and x8 source-load widths;
- branchless direct construction of FP64 bits at x4;
- a decomposed sign/exponent-prefix decoder at x4;
- shared-memory FP32, FP64, and prefix LUTs at x4;
- explicitly software-pipelined global FP32 and prefix LUTs at x4.

The E3M4 prefix table contains 256 16-bit E11M4 prefixes. E2M5 needs five
fraction bits, so its exact prefix table stores 17-bit E11M5 prefixes in
32-bit entries. All 256 encodings, including signed zero, subnormals,
infinities, and NaNs, are checked before timing.

The smoke workload runs deterministic U(0,1) and N(0,1) inputs, DOT at
`N=2^16,2^22`, and GEMV at `M=256`, `N=2^10,2^12`. Every kernel result is
checked against a long-double reference over the decoded storage values.

Submit from the repository root and select the GPU in the command:

```bash
sbatch --wait --nodelist=gpu-nvidia-h200-1-studvm-2 \
  scripts/run_e2e3_strategy_smoke_h200.sbatch
```

Each run is written to a timestamped directory here. The main outputs are
`decoder_validation.csv`, `kernel_validation.csv`, `strategy_smoke_samples.csv`,
`timing_summary.csv`, and `strategy_inventory.csv`.
