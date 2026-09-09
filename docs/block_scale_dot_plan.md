# Experiment 033: one-level shared scales inside DOT

Implementation contract from Astra, 2026-09-09. Highest precedence for this task.
Read with block_scale_dot_context.md. This document supplies implementation
decisions requested by the user; do not inherit obsolete 032 sweep settings.

## 1. Scientific scope and exact inventory

Question: how much do scale loading and application cost when reading FP32
payloads into FP64 DOT, and how does that change with N and storage block size?
Also measure a DOT-specific path applying scales after each block reduction.
No claims about optimal accuracy, universal instruction cost, or all kernels.

Full N = [65536, 1048576, 16777216, 67108864, 268435456]
       = [2^16, 2^20, 2^24, 2^26, 2^28].
Block B = [16, 32, 128], same contiguous boundaries for both DOT operands.
Scale type S = [float, double]. Payload type ALWAYS float.
Path = [per_element, deferred].
Baseline = [raw_fp32, fp32_to_fp64, raw_fp64].

12 scaled variants + 3 baselines = 15/N, 75 cases total.
10 warmups/case, 50 measured samples/case = 3,750 initial timing rows.
Never prune cases, omit B128, omit largest N, extend N, or average across N.
There is no instruction-count sweep, GEMV, scale-LUT, superblock, tensor-wide
scale, profiler, tensor-core, x4/x8, or separate accuracy experiment here.

## 2. Common memory, arithmetic, and geometry

- Four structure-of-arrays fields per encoded pair: q_left[N], q_right[N]
  as FP32, and s_left[ceil(N/B)], s_right[ceil(N/B)] as the selected scale type.
- No headers embedded among values, no padding beyond allocation alignment,
  no shared-memory scale staging, no explicit cache hints/flush experiments.
- Allocate separate payloads for each B but SHARE those payloads between the
  FP32-scale and FP64-scale variants. The two scale representations store the
  SAME numerical values; FP64 scale is exactly double(FP32 scale).
- Both operands independently scaled. Never precompute/store s_left*s_right
  as a new cross-operand metadata field. Deferred computes it inside timing.
- Every payload/scale multiplication and every scaled-kernel accumulation is
  FP64. Avoid float(q*s) followed by conversion, an unintended extra rounding.
- First stage exactly 512 CTAs x 256 threads for EVERY main variant/N.
  One final reduction CTA x256, included in event timing, reduces 512 partials.
  Common shared-memory block_sum structure can be reused from experiment032.
  Raw FP32 uses float partials/accumulator/final result, all others double.
  Give final results separate buffers instead of aliasing input partial storage.
- Scalar x1 loads, no vector types, no per-case grid tuning, no multiple
  independent accumulation chains. Deferred naturally needs a temporary
  block-local subtotal plus an accumulated subtotal; that is required semantics.
- Block size B is a compile-time template. CUDA CTA size is NOT storage B.
- Inputs/generation/transfers/allocation/validation/output copies are outside
  event timing. Scale loads and every multiplication, subgroup reduction,
  CTA reduction, second kernel and inter-kernel gap are inside timing.
- Changes in reduction organization between the two paths are deliberate and
  must be disclosed. This measures whole approaches, not isolated multiply cost.

## 3. Reproducible source and encoded data

Use a common underlying logical source for all three B and the three baselines.
No scale fitting or block-maximum calculation in this first cost experiment.
We want varying nontrivial scales, not scale=1, zero, or power-of-two-only data.

Counter generator: use the established splitmix64(seed XOR index), high32 bits.
Seeds, explicit uint64 hex:
  value left  = 0x6bd87c012a53f9e1
  value right = 0xf5ef05b8551985f4
  scale left  = 0x243f6a8885a308d3
  scale right = 0x13198a2e03707344

Source for each index:
  r = high32(hash(value_seed XOR i)) >> 8; // 24 bits
  x = double(int32(r) - 8388608) * 0x1p-23;
This yields signed, bounded [-1,1) values, exactly representable in FP32.
Use independent seeds for left/right. Same prefix of source at every N.

Scale for block b, independent from payload seed:
  r = high32(hash(scale_seed XOR b)) >> 9; // 23 bits
  s32 = float(8388608u + 3u*r) * 0x1p-24f;
This produces FP32 scales in [0.5,2), with one integer-to-float rounding and
exact binary scaling. s64=double(s32). Same scale-sequence prefix at each N
and B; what changes with B is its association with input elements.

Encode separately for each B, outside timing:
  q_i = float(x_i / double(s32[i/B]));
Use CPU double division, nearest rounding, then float. Host generation can be
chunked and copied; no need for a GPU generation kernel or quantizer timing.
No scale stored as a kernel constant; arrays must be read in the real decoder.

Baselines: raw_fp64 stores x_i as double; raw_fp32 and fp32_to_fp64 share
float(x_i). Only encoded q has the extra quantization rounding. These are
common logical inputs, not a claim encoded output equals source bit-for-bit.
Correctness of the scaled kernels is against s*q, NOT unquantized source.
Document that chosen scales are cost-test inputs, not an accuracy-optimal policy.

Data preparation bounds:
At largest N, all three encoded payload pairs total 6 GiB, baseline FP32 pair
2 GiB, baseline FP64 pair 4 GiB, all scale arrays about 0.61 GiB. Peak device
storage about 12.7 GiB plus small workspace. Query available device memory.
Process N sequentially and free previous allocations. Generate/copy source,
payload and scales in bounded CPU chunks (e.g. <=1M elements), keeping host
RSS comfortably inside 16 GiB. Do not retain huge host mirrors of all variants.
No unified-memory oversubscription, streaming H2D inside timing, or silent N skip.

## 4. Generic per-element implementation

template<int B, class S>
kernel per_element(qL, qR, sL, sR, N, partial):
  tid = blockIdx.x*256 + threadIdx.x
  stride = 512*256
  sum = FP64(0)
  for i=tid; i<N; i+=stride:        // #pragma unroll 1
    b = i/B                       // compiler turns power-of-two into shift
    a = FP64(qL[i]); sa = FP64(sL[b])
    c = FP64(qR[i]); sc = FP64(sR[b])
    xa = __dmul_rn(a,sa)
    xc = __dmul_rn(c,sc)
    sum = __fma_rn(xa,xc,sum)
  total = block_sum(sum)           // all CTA threads participate
  if threadIdx.x==0: partial[blockIdx.x]=total

No regrouping into sa*sc, no fusing scale into final accumulation, no explicit
lane-leader scale load/broadcast in this generic path. Ordinary identical
addresses within a warp may naturally benefit from cache/coalescing, which is
part of the measured behavior. One metadata value per block in storage does
NOT imply only one physical instruction/load occurs per block.
Explicit __dmul_rn + SASS inspection must retain the two reconstruction
multiplications before the DOT FMA. Do not use volatile memory as an artificial
barrier. Keep native code if correct; no assembly hacks to prescribe scheduling.

## 5. Deferred block application implementation

Use subwarps for B16, one warp/block for B32 or B128. B128 takes four scalar
coalesced rounds. Do NOT create one CUDA CTA for each 16-value storage block.

template<int B, class S>
kernel deferred(qL, qR, sL, sR, N, partial):
  L = min(B,32)                    // compile-time subgroup width
  V = B/L                         // 1,1,4 scalar rounds
  groupsPerCTA = 256/L
  group = threadIdx.x/L
  lane = threadIdx.x%L
  nblocks = ceil_div(N,B)
  acc = FP64(0)
  for tile = blockIdx.x*groupsPerCTA;
      tile < nblocks;
      tile += 512*groupsPerCTA:    // CTA-uniform loop; #pragma unroll 1
    b = tile+group
    blockLaneSum = FP64(0)
    for j=0; j<V; ++j:             // scalar, #pragma unroll 1
      i = b*B + j*L + lane
      if b<nblocks and i<N:
        blockLaneSum = __fma_rn(FP64(qL[i]), FP64(qR[i]), blockLaneSum)
    for offset=L/2; offset>0; offset/=2:
      other = __shfl_down_sync(0xffffffff, blockLaneSum, offset, L)
      blockLaneSum = __dadd_rn(blockLaneSum,other)
    if lane==0 and b<nblocks:
      sb = __dmul_rn(FP64(sL[b]),FP64(sR[b]))
      acc = __fma_rn(sb,blockLaneSum,acc)
  total=block_sum(acc)
  if threadIdx.x==0: partial[blockIdx.x]=total

Critical: ALL 32 lanes execute all shuffles, even a subgroup beyond nblocks
or invalid tail elements. Such lanes contribute zero. NO early return/break
or conditional around shuffle or block_sum. At a tail, bounds guards cover
loads only and the leader's scale access. The outer loop is CTA-uniform.
The subgroup leaders are the only threads accumulating scaled block totals.
All other acc values stay zero; final common CTA reduction is intentional.

Width16 gives two independent storage blocks per warp. Width32/B128 gathers
four scalar chunks into each lane's subtotal. This is x1 scalar access, NOT
the rejected vectorized x4 strategy. Do not preload four values or introduce
four independent accumulator chains. The extra shuffle/reduction overhead
is an inherent cost in this chosen straightforward deferred implementation.

Mathematical target: sum_b (sL_b*sR_b) * sum_{i in b}(qL_i*qR_i).
This is NOT one global final multiplication. FP rounding differs from the
generic per-element reconstruction, so do not require bitwise equality between
these two algorithms. Identical-scale-width versions SHOULD match bitwise
within each path because stored numerical scale values are identical.

## 6. Baselines and common reduction

Raw FP32: FP32 load, __fmaf_rn(a,b,sum), same x1 512x256 grid stride, float
CTA partial/final sum.
FP32->FP64: FP32 load, native widen, __fma_rn(a,b,sum), double partial/reduce.
Raw FP64: FP64 load, __fma_rn(a,b,sum), double partial/reduce.
Final reduction always 1x256 over 512 partials, timed. Use the same helper
and summation tree as experiment032 where applicable. No asymmetrical extra
launch between variants. All main cases launch exactly two timed kernels.

Three baseline rows/N are enough; do NOT repeat them per B/S/path, inflate
inventory, or substitute historical timings. All baselines measured this job.

## 7. Local tests, independent review, compiled-code gates

Suggested owned files:
  include/block_scale_core.hpp
  include/block_scale_kernels.cuh
  src/block_scale_dot_bench.cu
  tests/block_scale_core_test.cpp
  tools/check_block_scale_codegen.py
  tools/validate_block_scale_results.py  (stdlib, works on cluster)
  tools/analyze_block_scale_dot.py
  analysis/tests/test_block_scale_*.py
  scripts/check_block_scale_build.sbatch
  scripts/run_block_scale_smoke_h200.sbatch
  scripts/run_block_scale_full_h200.sbatch
  docs/block_scale_dot_progress.md, docs/block_scale_dot_review.md
  minimal CMake target additions only

Host tests: exact generator fixtures, range/finite/rounding and scale-width
identity; independent small decoder/DOT reference; storage size/count/tail
calculation; generated arrays are independent. C++ -O3 -DNDEBUG tests still
fail properly, no disabled assert-based validation. No fast math.

Python tests: exact 75-case inventory, duplicate/missing rows, wrong N/B/S/path,
wrong metadata, nonfinite time/result, invalid sample/round/order, mismatched
source/binary/audit digests, missing baseline, silent sample pruning. Explicit
exceptions even under python -O. Unit-test crossing logic if report uses it.

Independent code review BEFORE GPU reservation: contract plus actual code,
not a summary of why worker believes it correct. Cover deadlocks/subwarp masks,
tail bounds, group ownership, both operands' scales, FP64 multiplication vs
premature float rounding, two-kernel timing, reductions, CPU references,
provenance, CLI rejection of unknown cases, and job safety. Fix findings and
get follow-up review after material changes. Reuse existing reviewer if possible.

Zero-GPU build compiles all 12 scaled specializations, all 3 baseline kernels
and 2 final-reduction kernels, and host tests. Save SASS, ptxas logs/resources,
source commit and executable hash. Inspect actual TIMED kernels:
  generic: two FP32 payload loads and two correctly sized scale loads, native
  widening where needed, two FP64 reconstruction multiplies consumed by DFMA;
  deferred: payload scalar rounds, subgroup shuffles with correct width, scales
  applied to completed block totals, not to every input and not across blocks;
  baselines: expected FP32/FP64 loads and accumulation; no hidden format work.
No runtime B/path dispatch in hot kernels. No scale=constant replacement,
memory spills, device calls, vectorized payload loads, accidental per-element
divides. B power-of-two index division is a shift, not slow division.
Floating results/pairs/dataflow/predicates must be considered, not bare whole-
function DMUL/DFMA counts. Use a narrow checker plus saved explicit review of
real SASS where complex loops make automatic proof inappropriate. Unsupported
code must fail or be explicitly inspected, never silently marked verified.
Record registers/shared bytes/spills truthfully; add parser negative tests.
If compiler optimizes legally but alters intended placement, fix before GPU.

## 8. GPU correctness and smoke gates

Validation sizes: 0,1,15,16,17,31,32,33,127,128,129,257,4099, plus smoke N
which must exceed the generic grid stride and B128 deferred tile span.
Support N=0 validation by allocating at least one element while treating logical
length as zero; all 512 partials/final result must be initialized by kernels.
Validate every scaled variant and all baselines, including ragged final blocks.

Separate edge dataset actually copied to GPU: positive/negative/zero payloads,
including +/-1, representable neighbors, and nontrivial scales near 0.5,1,2.
Include random arrays with alternating scales between blocks so a wrong block
index or accidental cross-block sum fails. Repeat to span multiple full warps
and a ragged tail; include multiple outer-loop iterations at smoke sizes.

Host independent long-double sum of decoded products for small cases, with a
justified conservative absolute error bound based on sum(abs(products)) and
accumulation/reduction depth. Cancellation makes relative-to-result-only error
checks wrong. Raw FP32 needs its own tolerance; do not use its loose tolerance
for FP64. Check per-element decoded value bitwise where FP64 product exact.
Also use all-positive small fixtures where missing a block is easy to detect.

Compare FP32-scale and FP64-scale versions bitwise within each path. Compare
generic vs deferred using FP64 rounding bounds, not bitwise identity. Store
measured reference error/bound, not just a hard-coded all_passed line.
At full size, consume/check every scalar output, require repeat determinism;
no billion-element CPU reference per sample. Do not label mere finite checks
as a full-size high-precision accuracy proof. Small validation + assembly +
complete coverage of full size is sufficient for this cost benchmark.

Smoke timing N=[2^14,2^20], all15 cases each, 1warmup/3samples: exactly90rows.
Smoke flags must not silently override explicit dimensions into a wrong count.
Separate compute-sanitizer memcheck and synccheck validation runs must both
complete with zero errors and nonzero checked case inventory. Racecheck is
optional if reviewer suspects shared-memory issues, never used for timing.
Smoke/full use same hash-verified executable. Source change repeats appropriate
preflight and smoke. No validation failure can be demoted to a warning.

## 9. Timing protocol and result validation

Process N ascending. For each N allocate/generate ALL needed arrays before
timing. Warm all15 cases ten times. Then50rounds; shuffle all15 deterministically
each round, seed0x0335eed, recording exact order. CUDA events in one stream
surround first+final kernels and synchronize stop before elapsed time. No
generation/hostcopy/logging inside event bracket. No instrumentation.

Repeat-input, no-explicit-cache-flush protocol: small N can fit cache, large N
stream from memory. State this; don't call this a cold-cache benchmark. Raw
FP32 and widened FP32 share data, as do scale-width and path variants at each B.
Do not reorder all samples of one variant together or reuse warmup times.
Progress every round and preparation stage: N/case/sample, completed/expected,
wall time. Bound process externally to catch stalls.

At each N compare early/late25 baseline medians. If ANY of the three baselines
drifts >5%, rerun the COMPLETE15-case50-round block at that N once in the same
allocation; retain initial and rerun rows separately. Official=rerun, not
fastest. No further automatic repetitions. If still unstable, flag/report.
No N extensions. Initial rows exactly3750; each optional N rerun adds750.

CSV: mode,stage,N,B,scale_type,path,variant_id,payload_type,arithmetic,grid,
threads,round,order,time_ms,result,valid,job_id,node,seed-set-id. Baselines use
B=0, scale_type=none, path=baseline. IDs: per_element_b16_s32, etc, and the three
baseline IDs. Include source/provenance digests via manifest+CSV binding.

Validator must enforce all75 INITIAL cases and50rounds/case, exact15 order
positions in each(N,stage,round), all source/geometry/type identity fields,
no unknown stages/variants, correct full-vs-smoke inventory, no missing rerun
cases, valid positive finite times, finite results, fixed results per case,
FP32-/FP64-scale result identity. Check metadata against the actual manifest,
not just embed unvalidated JSON in HTML. Baseline ratios use same N/stage.
Checks must remain active with optimization. Reject malformed/truncated CSVs.

## 10. Cluster execution contract

First do a brief read-only connection/user-queue/H200-capacity check. If VPN
fails, finish local implementation/review and report the blocker, do not submit
blindly. Follow context's 1GPU/no-concurrent-user-job and bounded queue rules.

Preflight: zero GPU, compute partition1node1task8CPUs16GiB, hard15min.
Build new target ONLY and host test; SASS audit; expected1-5min, finite timeout.
Smoke: actualH2001GPU8CPUs16GiB, hard15min; validation and90timingrows, separate
sanitizers, each process bounded4min; expected1-5min.
Full: actualH2001GPU8CPUs16GiB, hard30min, process timeout20min; expected1-5min
including generation, potentially longer under host preparation. No matplot
on cluster. Stdlib validator before explicit GPU_MEASUREMENTS_COMPLETE marker.

Immediately before EACH submission verify no user job exists. Record resource
requests and ID. One-minute status/log check even for short jobs, 15min if
still running, then30min for longer waits. Distinguish pending from deadlock.
If SSH returns ambiguously after sbatch, check jobs before retry. No doubles.

Bind clean source commit, dirty-tree guard, binary hash, SASS hash, audit hash
at preflight and reuse checks in smoke/full. Save actual GPU name/UUID/driver,
node, CUDA/nvcc versions, CUDA_VISIBLE_DEVICES/allocation, limits. Query clocks
and power state if accessible without mutation, but do not change clocks/power.
Report model honestly (prior node reported H200 NVL). Nothing runs on 2 GPUs.

If a GPU job succeeds but local rendering fails, fix local rendering, not rerun
measurements. Preserve diagnostics and successfully completed raw results.
Collect via exact-path rsync or remote generated-results commit/push, verify
SHA256, then local analysis/graphs. No uncontrolled large input/binary files.

## 11. All-measurement report and screenshot presentation

Main graph contains ALL15 curves: 12scaled +3baselines, fiveN on X, actual
kernel time in milliseconds on Y. Log2 spacing for N; log Y as in approved
preview, labelled explicitly. Show all measured points/median and IQR; no fake
error bars. Baselines are curves, NOT horizontal lines when N varies.

Visual encoding: six color identities for B/scale combinations; solid lines
for per-element and dashed lines for deferred, markers distinct by path if
helpful. Three baselines neutral distinct line styles. Every curve gets a
right-side direct label with a dotted extension; no crowded legend box.
Labels must include B, scale storage, and path. Resolve collisions in screen
coordinates without moving actual data points or inventing ordering. Use a
larger canvas for15 labels. Do not silently hide overlaps or drop slow curves.

Deliver main dot_block_scale_all.png and .svg plus standalone report.html with
embedded plots/provenance and a clean screenshot.html with title, N list and
the large graph. The parent will decide later what to cut. Optional a single
same15-series ratio plot vs same-N FP32->FP64 may make overhead easier to see,
but it is supplementary and never replaces requested kernel milliseconds.
Do not generate numerous additional panels or install external rendering deps.

Report exact inventory, median/quartiles, same-N baseline ratios, storage
overhead formula 32+scale_bits/B per value, geometry, scale semantics, extra
deferred reductions, cache protocol, rounding differences, validation evidence,
drift, measured comparisons. No extrapolated accuracy/performance guarantees.
Unlike mockup, source of every plotted point must be collected full-run CSV.

View rendered PNG and desktop HTML; fix clipped/overlapping labels. Ensure
browser images can be displayed larger than chat thumbnails. All graph and
data files stay below100MB; no device arrays tracked.

## 12. Completion

Commit/push scoped implementation, docs, tests and successful results on
codex/block-scale-dot, without altering earlier experiments or merging main.
Report exact source/result commits, preflight/smoke/full IDs, actual GPU,
observed vs expected samples, review/correctness/SASS/sanitizer evidence, local
HTML/PNG paths and short numerical findings. Keep the same worker available
for future parent updates. Stop only after complete artifacts or a genuine
recorded external blocker under the stated queue/connectivity rules.
