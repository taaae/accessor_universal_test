# Experiment 020: arbitrary-width conversion and access strategies

This experiment measures DOT and GEMV with true FP32 and FP64 accumulation for
selected 2–28-bit storage formats.  Numeric layout, physical dense/padded
storage, scalar/thread-packet/cooperative access, and conversion decoder are
recorded independently.

The 2-bit shard also records same-topology raw FP32→FP32 and FP64→FP64 x1/x2/
x4/x8 baselines once per dataset.  These are shared baselines for all width
cohorts rather than being remeasured redundantly in every executable.

The job has three stages:

1. small smoke runs and cross-strategy numerical validation;
2. representative large-size screening of every candidate;
3. complete N sweeps for the best two candidates per format, arithmetic type,
   kernel, storage layout, and access group, plus mandatory scalar controls.

The committed strategy rationale for each width is under
`docs/decoder_strategies/bitwidth_*.md`.  Generated runs are stored in
`run_<UTC timestamp>/smoke`, `screen`, and `full` subdirectories.
