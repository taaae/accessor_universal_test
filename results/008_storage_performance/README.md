# 008: Storage-format performance and roofline inputs

This experiment separates four questions that should not be inferred from one
end-to-end kernel time:

1. `register_decode` measures conversion throughput after one storage packet is
   loaded into registers. Empty compiler barriers prevent decode hoisting. Its
   FP64 additions are required to keep results live, so it is a decode-plus-add
   microbenchmark, not a literal standalone instruction latency.
2. `stream_load` reads the encoded bytes and consumes their bit patterns without
   conversion. `stream_decode` reads the same bytes, converts to FP64, and sums
   them. Their paired curves expose conversion/instruction pressure once global
   loading is included. Their times are not subtracted: the kernels have
   different instruction streams and subtraction would amplify noise.
3. `dot` times both reduction launches; `gemv` times the complete one-block-per-
   row kernel. All use the same FP64 arithmetic and x1/x2/x4 implementations as
   the validated kernels in experiment 005.
4. Nsight Compute profiles representative large cases outside the valid event
   timing run. It records actual DRAM traffic, executed FP32/FP64 operations,
   SM/DRAM utilization, cache behavior, issue activity, occupancy, registers,
   and the profiler's roofline sections. Profiler replay time is explicitly
   marked contaminated and is never used as the performance time.

Both U(0,1) and N(0,1) are timed because custom decoders can have data-dependent
instruction paths. Each x1/x2/x4 group is interleaved in rotating order over
three rounds. The default has 10 warmups and 15 recorded samples per case; each
sample aggregates enough launches to target 15 ms. Raw samples are retained so
confidence and clock-drift checks remain possible.

## Planned figures

The recorded schema is designed for these report figures after the H200 run:

- **size-regime curves:** median time and encoded-payload GB/s versus N, showing
  launch-latency, cache, and HBM plateaus;
- **conversion ladder:** register decoded-values/s, stream-load GB/s, and
  stream-decode GB/s in aligned small multiples, without pretending that the
  microbenchmarks are additive phases;
- **packed speedup:** x2/x1 and x4/x1 time ratios versus N plus a plateau
  dumbbell plot for every format, separately for DOT and GEMV;
- **algorithmic roofline:** useful GFLOP/s from event timing against useful
  FLOPs per unique encoded byte, with memory and scalar-FP64 ceilings;
- **hardware roofline:** executed FLOPs per measured DRAM byte from Nsight
  Compute, kept distinct from the compression-aware algorithmic roofline;
- **bottleneck map:** DRAM-percent-of-peak versus SM-percent-of-peak, annotated
  with issue activity, occupancy, and registers to distinguish bandwidth,
  conversion/instruction throughput, dependencies, and insufficient
  parallelism;
- **distribution sensitivity:** paired U(0,1)/N(0,1) ratios, mainly to expose
  decoder control-flow effects rather than numerical error.

For GEMV, `unique_storage_bytes` counts the vector once and is the
compression-aware algorithmic traffic. `requested_storage_bytes` counts the
vector load for every row. Nsight's measured DRAM bytes determine how much of
that repeated vector traffic was actually served by cache.

## H200 command

Submit from the repository root. The script does not select a node:

```bash
sbatch --wait --nodelist=gpu-nvidia-h200-1-studvm-2 \
  scripts/run_storage_performance_h200.sbatch
```

The default run profiles all 17 formats at x1 and x4 for register decode,
stream decode, DOT, and GEMV. To do event timing first without Nsight Compute,
submit with `--export=ALL,PROFILE=0`.
