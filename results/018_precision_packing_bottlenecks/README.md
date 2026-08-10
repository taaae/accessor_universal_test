# Experiment 018: precision/packing bottlenecks

This experiment separates three effects in DOT and GEMV:

1. compact storage;
2. conversion to a wider arithmetic type;
3. processing several logical values per thread.

The central question is whether x2/x4/x8 processing helps mixed-precision
kernels more than same-storage/same-arithmetic kernels, and, if it does,
whether the gain comes from vector loads, instruction amortization, independent
accumulators, or packed arithmetic.

## Experiment matrix

The H200 timing sweep uses these storage/arithmetic pairs:

| Storage | Arithmetic modes |
|---|---|
| FP16 E5M10 | FP16, FP64 |
| BF16 E8M7 | BF16, FP64 |
| FP32 E8M23 | FP32, FP64 |
| FP8 E4M3 | FP16, FP32, FP64 |
| FP8 E5M2 | FP16, FP32, FP64 |
| FP4 E2M1 | FP16, FP32, FP64 |
| FP64 E11M52 | FP64 |

FP8 and FP4 do not have an ordinary scalar same-type FMA/accumulator path on
Hopper. Their FP16 mode is the narrowest scalar arithmetic comparison; this is
still explicitly labelled as mixed precision. FP4 storage is bit-dense: two
logical values per byte.

Each valid pair is tested with logical widths x1, x2, x4, and x8. Widths above
x1 have two controlled implementations:

| Family | Loads | Accumulators | What it isolates |
|---|---|---|---|
| `scalar_single` | one scalar value at a time | one | x1 baseline |
| `scalar_unrolled` | xL scalar loads | L | thread coarsening, loop amortization, and independent chains |
| `vector_packet` | naturally aligned packet loads | L | additional effect of fewer/wider load instructions |
| `packed_arithmetic` | packet loads | packed FP16/BF16 lanes | native `half2`/`bfloat162` arithmetic; separate from load packing |

`packed_arithmetic` is only valid for same-type FP16 and BF16 arithmetic. It is
not mixed with the main `vector_packet` result.

DOT sizes are `2^12, 2^16, 2^20, 2^24, 2^27`. GEMV fixes `M=1024` and uses
`N=2^8, 2^10, 2^12, 2^14, 2^16`. Uniform U(0,1) and normal N(0,1) inputs are
both retained because value-dependent decoding can affect narrow formats.

## Raw timing data

`timing_samples.csv` records every CUDA-event sample, not only aggregates. Its
schema includes:

- GPU, distribution, kernel, storage and arithmetic type;
- implementation family and logical width;
- M, N, round, sample, batch iterations, and time;
- useful operations and logical/requested storage bytes;
- derived useful throughput and effective requested bandwidth;
- result value and an FP64-reference error check.

The run uses ten warmups, three randomized rounds, five samples per round, and
enough launches per sample to target at least 15 ms. The final reduction is
included in complete DOT time. Per-thread and per-block accumulation uses the
selected arithmetic type. DOT's small array of block partials is finalized in
FP64 for every mode, avoiding FP16 overflow in the final inter-block sum and
keeping that fixed overhead comparable. GEMV row reductions remain in the
selected arithmetic type and write FP64 only after the row is complete.

`timing_summary.csv` contains medians, dispersion, packet speedups, mixed
precision penalties, classical roof limits, and roof gaps. The measured HBM
calibration is used instead of the product-sheet bandwidth.

## Component calibration

`component_samples.csv` provides empirical resource floors:

- `stream_load`: source traffic without decoding;
- `stream_decode`: source traffic plus conversion, accumulated in the selected
  arithmetic type;
- `register_decode`: conversion with the source packet already in registers;
- `arithmetic_chain`: one versus x2/x4/x8 independent accumulator chains;
- `hbm_calibration`: a large streaming read used as the sustainable-bandwidth
  reference.

These are lower-bound diagnostics, not additive time components. GPU resources
overlap, so later plots must not stack their times.

## Nsight Compute data

The full timing sweep is profiler-free. A separate representative sweep runs
one launch under Nsight Compute for large DOT and GEMV cases. It collects:

- Speed of Light and roofline sections;
- memory and compute workload analyses;
- instruction, scheduler, warp-state, launch, and occupancy sections;
- DRAM bytes and executed FP32/FP64 arithmetic instructions;
- the `.ncu-rep`, details text, raw CSV, and a case-to-report manifest.

The representative set profiles x1 and x8 for all storage/arithmetic pairs,
plus x8 scalar-unrolled controls. FP16/BF16 same-type cases additionally profile
packed arithmetic. Profiling time is marked contaminated and is never used as
ordinary performance data.

## Intended plots

The collected data supports:

1. complete kernel time and logical throughput versus N;
2. x1-normalized speedup versus x1/x2/x4/x8;
3. classical roofline points and measured roof gap;
4. non-stacked resource-floor bars with actual time overlaid;
5. an Nsight Compute resource/stall heatmap;
6. scalar-unrolled versus vector-packet comparisons that attribute the gain.

All generated run data belongs in a timestamped `run_...` child directory.
Slurm output belongs directly in this experiment directory.
