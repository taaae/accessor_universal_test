# Experiment 018 context: packing and mixed-precision bottlenecks

## Main question

This experiment tests the claim that processing several storage values per
thread is especially useful when storage and arithmetic types are decoupled.

For a compact type `A`, compare:

1. `A` storage with `A` arithmetic;
2. `A` storage with wider arithmetic, especially FP64;
3. FP64 storage with FP64 arithmetic.

Each case is tested with logical widths x1, x2, x4, and x8. The goal is not
merely to find the fastest width. It is to show where the mixed-precision
slowdown comes from and whether wider per-thread processing mitigates it.

## Working hypothesis

Compact storage reduces HBM bytes, but conversion to a wider arithmetic type
adds instructions and can expose load-issue, conversion-throughput, dependency,
and instruction-scheduling limits that the classical memory/compute roofline
does not describe well.

Processing x2/x4/x8 values per thread may help by:

- amortizing loop, indexing, address-generation, and control instructions;
- issuing fewer or wider source-load instructions when packet loads are used;
- creating independent conversion and FMA chains that improve latency hiding;
- doing more useful work per participating thread.

The expected effect is larger for compact-storage/wide-arithmetic kernels than
for same-storage/same-arithmetic kernels. However, thread coarsening can also
increase register pressure or reduce the number of active threads, so wider is
not guaranteed to be faster.

## Formats and arithmetic modes

The implemented matrix is:

| Storage | Arithmetic modes |
|---|---|
| E5M10 aka FP16 | FP16, FP64 |
| E8M7 aka BF16 | BF16, FP64 |
| E8M23 aka FP32 | FP32, FP64 |
| E4M3 aka FP8 E4M3 | FP16, FP32, FP64 |
| E5M2 aka FP8 E5M2 | FP16, FP32, FP64 |
| E2M1 aka FP4 | FP16, FP32, FP64 |
| E11M52 aka FP64 | FP64 |

FP8 and FP4 have no ordinary scalar same-type FMA/accumulator path on H200.
Their FP16 arithmetic cases are the narrowest comparisons, but they are still
mixed-precision cases rather than true `A`-arithmetic baselines.

## Controlled access families

The experiment deliberately separates effects that are often all called
"packing":

| Family | Meaning |
|---|---|
| `scalar_single` x1 | One scalar load, one accumulator; reference implementation. |
| `scalar_unrolled` x2/x4/x8 | Multiple scalar loads and independent accumulators; isolates thread coarsening and instruction-level parallelism. |
| `vector_packet` x2/x4/x8 | Aligned packet loading plus the same independent scalar arithmetic; the difference from `scalar_unrolled` estimates the load-packet effect. |
| `packed_arithmetic` x2/x4/x8 | Native `half2` or `bfloat162` load and arithmetic; kept separate because it changes both access and arithmetic execution. |

"Packed x4" means one thread processes four logical values. It does not mean
four threads cooperatively unpack one value. FP4 storage is bit-dense, with two
logical values per byte. FP64 x2/x4/x8 processing remains meaningful as thread
coarsening and independent accumulation, even though the hardware still needs
multiple 64-bit loads.

## Kernels and datasets

- DOT includes its final reduction in complete operation time.
- GEMV fixes `M=1024` and varies reduction length `N`.
- DOT uses `N = 2^12, 2^16, 2^20, 2^24, 2^27`.
- GEMV uses `N = 2^8, 2^10, 2^12, 2^14, 2^16`.
- Both U(0,1) and N(0,1) inputs are retained.
- Timing uses ten warmups, three randomized rounds, five samples per round, and
  launch batching targeting at least 15 ms per sample.

The selected arithmetic type is used for per-thread and per-block accumulation.
DOT finalizes the small array of block partials in FP64 for every mode so FP16
overflow does not corrupt the inter-block reduction and that fixed cost remains
comparable. GEMV reduces each row in the selected arithmetic type and writes an
FP64 result afterward.

## Data being collected

### Complete kernels

Raw CUDA-event samples record complete DOT/GEMV time, useful throughput,
requested storage bytes, modeled load-instruction count, and correctness data.

### Isolated diagnostic kernels

- `stream_load`: source traffic without numeric decoding;
- `stream_decode`: load plus conversion and accumulation;
- `register_decode`: conversion after source bits are already in registers;
- `arithmetic_chain`: x1/x2/x4/x8 independent arithmetic chains.

These timings are isolated diagnostics. Their summaries scale them to the
logical work of a complete kernel, but they are not mathematical lower bounds:
checksum/reduction overhead and different overlap can make a scaled component
larger than complete-kernel time. They must never be stacked or summed as if
they were disjoint time components.

### Nsight Compute

Representative large DOT and GEMV cases collect:

- measured DRAM bytes and bandwidth utilization;
- SM utilization and classical roofline information;
- FP32/FP64 arithmetic and total instruction counts;
- registers, occupancy, eligible warps, and scheduler behavior;
- L1/L2 behavior;
- long-scoreboard, math-pipe, and MIO-throttle stall indicators.

Nsight Compute replays and instruments kernels. Its recorded duration is
profiler-contaminated and is not a normal performance result.

## Intended graph groups

1. Complete kernel time and throughput versus `N`.
2. x1-normalized speedup versus x1/x2/x4/x8.
3. Mixed-precision slowdown relative to the narrowest arithmetic mode.
4. Classical roofline position and distance from the empirical roof.
5. Actual time overlaid against separate, non-stacked scaled
   load/decode/arithmetic diagnostics.
6. Nsight Compute heatmaps for memory/SM use, instruction density, occupancy,
   and stalls.
7. Direct `scalar_unrolled` versus `vector_packet` comparison. This is the key
   attribution graph: it distinguishes gains from thread coarsening/independent
   chains from gains caused specifically by wider packet loads.

## Interpretation rules

- A conventional roofline point alone cannot establish a conversion or
  load-issue bottleneck. Use it together with component calibrations and NCU.
- Do not describe all x2/x4/x8 gains as memory-bandwidth savings: logical source
  bytes are unchanged between access families of the same storage format.
- If scalar-unrolled and vector-packet improve equally, the benefit is primarily
  coarsening, independent accumulation, or instruction amortization—not the
  packet load itself.
- If vector-packet improves beyond scalar-unrolled, that incremental improvement
  is evidence for fewer/better load instructions or cheaper unpacking.
- Same-type FP16/BF16 packet results and native `half2`/`bfloat162` arithmetic
  answer different questions and must remain separate in plots.
- Native packed arithmetic is omitted from derived scalar arithmetic roofs until
  it has a matching packed-arithmetic throughput calibration.
- Increasing width can lose because of register pressure, occupancy reduction,
  tail work, or insufficient parallelism, especially for smaller `N`.

## Current execution state when this note was written

- Full H200 job: `442766`.
- Node: `gpu-nvidia-h200-1-studvm-2`.
- Hard limit: four hours.
- Expected runtime after cached compilation: approximately 45–90 minutes.
- Result location:
  `results/018_precision_packing_bottlenecks/run_<UTC timestamp>/`.
- Results have not yet been analyzed. Check the Slurm `.out` before pulling or
  interpreting generated files.
