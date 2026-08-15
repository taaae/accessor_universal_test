# Experiment 021: unified same-GPU strategy performance

This experiment reruns the old and new DOT/GEMV conversion-strategy suites
inside one physical H200 allocation.  Its purpose is to remove GPU, driver,
clock-policy, and VM/node differences from cross-format comparisons.

The job is deliberately staged:

1. CUDA `synccheck` of the cooperative arbitrary-width loader;
2. smoke validation of every common-harness strategy;
3. representative-size screening of every common-harness strategy;
4. full N-sweeps for the best strategies in every access group;
5. full reruns of the historical specialized E2M5/E3M4 and power-of-two
   FP64 strategy inventories.

The common harness covers widths 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16,
17, 20, 24, 28, and 32 with both FP32 and FP64 arithmetic where the requested
target can represent the selected layout exactly.  Raw FP32 and FP64 anchors
are remeasured before every width in the screen and full stages.

Historical suites are retained as separate subdirectories because they
contain decoder-specific experiments that are not meaningful for every new
width.  Final cross-format rankings should use `unified_core`; the legacy
subdirectories answer whether a specialized historical implementation remains
competitive on the same GPU.

Submit only after checking availability and selecting the normal H200 node in
the command.  The batch file never chooses a node itself.  `STOP_AFTER` can be
set to `sanitizer`, `smoke`, `unified`, or `full` for bounded preflights.

```bash
cd /storage/home/timofeirusanov/accessor_universal_test
sbatch --wait --nodelist=gpu-nvidia-h200-3 \
  scripts/run_unified_strategy_benchmark_h200.sbatch
```

The default hard limit is eight hours.  No Nsight Compute profiling or SASS
dump is performed by this job.
