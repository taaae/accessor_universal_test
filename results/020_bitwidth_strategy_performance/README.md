# Experiment 020: arbitrary-width conversion and access strategies

This experiment measures DOT and GEMV with true FP32 and FP64 accumulation for
selected 2–32-bit storage formats.  Numeric layout, physical dense/padded
storage, scalar/thread-packet/cooperative access, and conversion decoder are
recorded independently.

The unified inventory includes the prior 8-/16-/32-bit cohorts and their CUDA
native FP8, FP16, BF16, and FP32 decoders.  Layouts outside FP32's exact direct
decoder range remain FP64-only.  Same-topology raw FP32→FP32 and FP64→FP64
x1/x2/x4/x8 anchors are interleaved before every width in the screen and full
stages to expose clock or thermal drift.

The job has three stages:

1. small smoke runs and cross-strategy numerical validation;
2. representative large-size screening of every candidate;
3. complete N sweeps for the best two candidates per format, arithmetic type,
   kernel, storage layout, and access group, plus mandatory scalar controls.

Set `STOP_AFTER=smoke` for the first CUDA build/preflight.  A normal rerun uses
the cached build and completes all stages; `STOP_AFTER=screen` is also available
for an intermediate strategy-inventory run.

Before reserving a GPU for the smoke stage, submit
`scripts/check_bitwidth_strategy_build.sbatch`.  It compiles every width shard
and runs the host-only tests without requesting a GPU.

The committed strategy rationale for each width is under
`docs/decoder_strategies/bitwidth_*.md`.  Generated runs are stored in
`run_<UTC timestamp>/smoke`, `screen`, and `full` subdirectories.
