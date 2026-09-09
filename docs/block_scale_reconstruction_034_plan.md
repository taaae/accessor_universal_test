# Experiment 034: reconstruction precision and thread-local deferred scales

Parent implementation contract and persistent-worker context update, 2026-09-09.
Read this entire document before implementation. This is the active task and
supersedes experiment 033's inventory and decoder semantics. Experiment 033 is
complete. Do not rerun or overwrite its results.

## Role, ownership, and permission

The user explicitly authorized the persistent GPT-5.6 Sol worker to implement,
independently review, run local tests, cluster preflight, GPU smoke and full
benchmark, fetch results, and produce graphs without asking approval for jobs.
Resume /root/sol_implementation, not a new user-facing task. Reuse the existing
independent reviewer for bounded read-only review; it must never submit jobs.
Report milestones and genuine blockers, not frequent unchanged status updates.

You are not alone in the repository. Preserve other edits. Continue in:
/Users/tae/Desktop/research/universal_types/accessor_universal_test_block_scale
branch codex/block-scale-dot, remote origin. Remote checkout remains:
/storage/home/timofeirusanov/accessor_universal_test_block_scale

Own new experiment-034 implementation, test, wrapper, analysis and result files,
plus minimal CMake additions. Prefer new block_scale_reconstruction_* files,
reusing unchanged 033 core generators/helpers where appropriate. Keep 033
reproducible. Do not silently replace its checker/validator with a 034 contract.
New outputs: results/034_block_scale_reconstruction/run_<UTC>_<job>/.
New progress/review docs: docs/block_scale_reconstruction_034_progress.md and
docs/block_scale_reconstruction_034_review.md. Record source/result commits,
job IDs, hashes, current stage and exact next action as you proceed.

Parent-owned untracked files currently present must NOT be deleted, included
in worker commits, or modified without coordination:
- tools/plot_block_scale_per_element.py
- results/033_block_scale_dot/run_20260909T163444Z_460445/report/
  dot_block_scale_per_element.{png,svg}
- the same directory's dot_block_scale_per_element_n28.{png,svg}
Read that plotting script for style if helpful.

## Context since your previous completion

033 completed through 54e7f6b, measured source 99410d5, full job 460445.
Parent independently verified the 4,500 rows, all 75 medians, host test, SASS
audit reproduction/hashes, and sanitizer/correctness logs. One N=2^20 baseline
retained 5.3% drift after a complete rerun; this was disclosed, not hidden.
At N=2^28, widened FP32 = 1.038368 ms; raw FP64 = 1.208768 ms. Per-element
B128/S64 = 1.054176 ms, B128/S32 = 1.085296 ms. These are historical context,
not inputs to the new measured graphs or substitutes for fresh baselines.

Discussion identified two distinct issues:
1. S32 reconstruction widened both payload and scale before FP64 multiplication.
   S64 avoids widening the scale. Multiplying q*s in FP32 then widening is a
   legitimate alternative with additional FP32 rounding. This was not tested.
2. Old deferred reduced an entire block with warp shuffles and used only its
   leader to scale/accumulate. Its five dependent FP64 shuffle/add stages for
   B128 cost more than the saved multiplications in the measured implementation.
   New deferred applies the scale to each thread's four-product subtotal,
   keeping all lanes' accumulators live and requiring NO per-block shuffles.
   Both deferred algorithms are mathematically legitimate, not interchangeable
   in floating-point rounding. Do not describe the old implementation as a bug.

User accepted 14 total cases per N, and deferred ONLY B128 with FP64 arithmetic.
No deferred S32 multiplication/reconstruction experiment, no B16/B32 deferred,
no old whole-block deferred, no new mappings or extra strategy search.
The exclusion of B16/B32 is planned: each lane only handles one value in the
existing mapping, so there is no thread-local multielement subtotal to amortize.

## Questions and fixed experiment inventory

Questions: Does accepting FP32 reconstruction reduce overhead for S32 scales?
How do scale storage width and B affect performance with identical scale values?
Does shuffle-free thread-local deferred scaling improve on per-element scaling?
This is a performance/rounding-contract test, not an application accuracy study.

DOT only, scalar x1. Full N = [65536,1048576,16777216,67108864,268435456].
All cases at every N, no pruning, no geometric means, no new N or GEMV.

Exact IDs and semantics:
- reconstruct64_b{16,32,128}_s32: 3 cases, S32, FP64 reconstruction.
- reconstruct32_b{16,32,128}_s32: 3 cases, S32, FP32 reconstruction then widen.
- reconstruct64_b{16,32,128}_s64: 3 cases, S64, FP64 reconstruction.
- deferred_local_b128_s{32,64}: 2 cases, FP64 subtotal and scale arithmetic.
- raw_fp32, fp32_to_fp64, raw_fp64: 3 cases.
Total 14/N, 70 initial groups, 3,500 initial samples at 50/group.
10 warmups/case. Each optional full-N drift rerun adds 700 samples.
Smoke N=[16384,1048576], 1 warmup and 3 rounds: exactly 84 rows/28 groups.

## Common data and timing contract

Preserve 033 data exactly: SplitMix64(seed XOR index), four established seeds,
source x in [-1,1) exactly FP32, random nontrivial S32 scales in [0.5,2),
S64=double(S32), q=float(double(x)/double(s32)). Read the exact formulas in
include/block_scale_core.hpp and sections 2-3 of docs/block_scale_dot_plan.md.
Use unchanged generators, same seed_set_id, and record explicit experiment ID
and reconstruction/subtotal modes separately. Do not accidentally hash twice.

Use both operands' independent scales with identical block boundaries. Separate
qL/qR arrays per B; share them across all strategies/scale widths for that B.
S64 is the exact widening of S32 to isolate storage/decoder cost, not improved
scale fitting. Baselines use original x arrays as before. Decoder rounding
changes outputs; no claim they equal the original source exactly.
No precomputed combined left/right scale, scale=1 specialization, power-of-two
only scales, global/shared table, shared scale staging, lane-leader broadcast,
explicit cache hints, input padding, vector loads, or Tensor Cores.

First kernel always 512 CTAs x256; final kernel always 1 CTA x256 over 512
partials. Same block_sum structure and separate FP32/FP64 result buffers.
Raw FP32 alone accumulates in FP32; ALL other DOT accumulation is FP64.
Two CUDA kernels and their gap within the same stream's CUDA event interval.
All allocations, data generation/copies, validation, result copies, logging
outside the event interval. Check/consume every timed output after timing.
Repeat-input/no explicit cache flushing, disclosed in report.
Sequential N allocations; host generation chunks <=1M values, no giant host
mirrors. Existing ~12.7 GiB peak device memory, host <16 GiB. No silent N skip.

## Per-element pseudocode

Compile-time B and decoder template, no hot runtime dispatch. Scalar grid stride
and #pragma unroll 1, one loop-carried FP64 sum, as in 033.

For reconstruct64 with scale type S:
  b = i/B
  a = __dmul_rn(double(qL[i]), double(sL[b]))
  c = __dmul_rn(double(qR[i]), double(sR[b]))
  sum = __fma_rn(a, c, sum)

For reconstruct32, ONLY S=float:
  b = i/B
  af = __fmul_rn(qL[i], sL[b])
  cf = __fmul_rn(qR[i], sR[b])
  sum = __fma_rn(double(af), double(cf), sum)

Use explicit __fmul_rn to require FP32 product rounding before widening. No
fast math, no FTZ shortcut, no replacing this with double operands. Require two
distinct FP32 reconstructions; never multiply qL*qR first or fuse a scale with
the DOT FMA. End with identical common CTA/final reductions for all paths.
No volatile memory or inline assembly hacks to pin arbitrary instruction counts.

## Thread-local deferred pseudocode (B128 only)

Each warp owns one storage block; every lane processes four scalar values
32 positions apart, with a single dependent FP64 temporary subtotal. These
are four scalar coalesced rounds, NOT vectorized x4 or four independent chains.

  constexpr L=32, V=4, groupsPerCTA=8
  group = threadIdx.x / 32; lane = threadIdx.x % 32
  nblocks = ceil_div(N,128)
  acc = double(0)
  for tile=blockIdx.x*8; tile<nblocks; tile+=512*8: // unroll 1, CTA-uniform
    b = tile+group
    partial = double(0)
    for j=0; j<4; ++j:                          // unroll 1
      i = b*128 + j*32 + lane
      if b<nblocks && i<N:
        partial = __fma_rn(double(qL[i]), double(qR[i]), partial)
    if b<nblocks:
      scale = __dmul_rn(double(sL[b]), double(sR[b]))
      acc = __fma_rn(scale, partial, acc)       // ALL lanes, no lane==0 guard
  total = block_sum(acc)                       // ALL CTA threads participate
  if threadIdx.x==0: output_partial[blockIdx.x]=total

No SHFL or intermediate DADD reduction inside the processing loop. The final
common block_sum is the only thread communication. All valid lanes multiply
their subtotal by the same scale product; ordinary repeated scale loads are
intentional. No early return or barrier under a lane-dependent guard.
Ragged blocks get zero contributions for absent items and guarded scale loads.
New semantics sum_b sum_lane ((sL_b*sR_b)*sum_j(qL*qR)); rounding differs from
033 whole-block deferred and generic FP64 reconstruction. Describe honestly.

## Correctness and test gates

Use explicit checks that remain active in Release/-DNDEBUG and Python -O.
Reuse and update good 033 tests, not only its passing marker/count strings.
17 datasets x14 variants =238 CPU/GPU correctness cases minimum:
- generated sizes 0,1,15,16,17,31,32,33,127,128,129,257,4099,1048576;
- existing 4099-element alternating-scale/edge and all-positive fixtures;
- new 257-element rounding-sensitive fixture containing q=s=1+2^-23, suitable
  positive RHS, distinct scale patterns across blocks and a ragged tail.
Construct the last fixture explicitly, not by encode() undoing its scales.
Its reconstruct32 and reconstruct64 reference outputs MUST differ, and GPU
must match each corresponding contract. Test exact per-value FP32 rounded
products on the host; add a bounded untimed GPU decoder check if necessary.

Independent CPU references:
- reconstruct64: high-precision sum of products of decoded q*s;
- reconstruct32: FIRST compute float-rounded q*s for each operand separately,
  THEN widen those values for the reference DOT. Do not use the unrounded q*s
  reference with a loose FP32 tolerance to paper over the missing rounding.
- deferred_local: high-precision target of the encoded products with a bound
  covering the actual FP64 subtotal, scale product, and final accumulation.
- raw32 gets its separate FP32 bound; native widened/raw64 get FP64 bounds.
Use long double on cluster with confirmed extra precision, record max errors
and bounds. Bound against sum of absolute ELEMENT product contributions, not
absolute block sums, which can hide cancellation. Derive conservative depth
for local-deferred outer accumulation (ceil(N/(512*8*128))) plus four inner
steps, scale multiplication, CTA and final reductions. Do not blindly reuse
generic grid-stride depth. Also all-positive fixtures catch dropped blocks.
S32/S64 results must match bitwise within reconstruct64 and deferred_local;
no such equality requirement between reconstruct32 and reconstruct64.
Full runs check finite deterministic outputs; do not claim full-N exact proof.

Review BEFORE GPU smoke: delegate contract + actual code to independent reviewer
without your success narrative. Check bounds/barriers, new rounding references,
all-lane deferred accumulation, original dataset reuse, timing, manifests,
sanitizer invocation and resource/queue safety. Fix material issues and re-review.

Zero-GPU preflight audits 16 TIMED kernels: 11 scaled +3 baselines +2 final
reducers. Untimed validation-only helpers are separately inventoried if added.
Machine-code evidence must follow operand dataflow, not just opcode totals:
- reconstruct32: FP32 q and S32 loads -> two FMUL -> two widens -> DOT DFMA;
- reconstruct64 S32: four widens -> two DMUL -> consuming DOT DFMA;
- reconstruct64 S64: two payload widens -> two DMUL -> consuming DOT DFMA;
- deferred_local: four scalar loop rounds -> per-lane subtotal -> scale DMUL
  and consuming DFMA, no per-block shuffles/reduction or leader-only predicate;
- actual native baselines and final reducers preserved.
Confirm no spills, device calls, hidden division/vector loads, constant scales.
Report resource use without confusing static opcode count with dynamic count.
Preserve parser negative tests; extend for wrong FMUL placement, missing widen,
premature scale application, ignored subtotal, and old leader/shuffle path.
033's checker had known gaps closed by manual inspection. Do not call a similar
034 checker a proof: get hash-bound explicit actual-SASS review where needed.
Smoke only after local tests, review, host preflight and exact SASS gate pass.
Run separate memcheck + synccheck against all validation fixtures; zero errors,
nonzero exact inventory. Same audited binary for smoke and full. Source/binary
changes require appropriate rebuilt preflight and repeated smoke.

## Measurement, manifests and reporting

Deterministically shuffle all14 variants for each of50 rounds, record order
0..13. Keep the existing shuffle seed 0x0335eed and document it. At each N warm
all variants before measured rounds. Progress at preparation and each round.
If any baseline early25/late25 medians drift >5%, rerun ALL14 cases at that N
once, retaining originals. Official = rerun, never fastest. Recheck and report
persistent drift. Do not extend repeats, adjust thresholds, or drop small N.

Stdlib validator on cluster must enforce exact modes, Ns,14 variants,70 full
initial groups/3500 rows,28 smoke groups/84 rows, samples/warmups/order/geometry,
reconstruction and scale types, round uniqueness, stage completeness, result
determinism, scale-width equality where applicable, finite positive timings,
same manifest source/binary/SASS/audit/job/node bindings and seed/dataset IDs.
New metadata should explicitly distinguish reconstruction arithmetic from DOT
accumulation so reconstruct32 is never mislabelled as FP32 DOT. Baselines have
B0/scale none. Include rounding/reference inventory and correctness evidence.
Unit-test missing/duplicate/unknown cases, wrong semantics/digests/dimensions,
malformed/truncated CSV, invalid order and incomplete reruns. Recompute all
summaries from measured CSV. No reused historical baseline rows.

Deliver locally rendered self-contained HTML report and large screenshot page,
PNG+SVG for TWO primary graphs:
1. All14 curves vs N, log2 X and explicitly labelled log milliseconds Y.
2. N=2^28 only, all14 long horizontal lines on LINEAR milliseconds Y, no X ticks.
   Use dotted extensions and collision-resolved right labels as parent plots.
Use readable labels distinguishing FP32/FP64 reconstruction, scale type, B and
local-deferred. Native baselines neutral. Do not hide or clip crowded labels,
move measured heights to space labels, or show invented values/error bands.
Median50/IQR bands. Baseline labels include FP32 accumulation exception.
Report same-N comparisons, error-contract distinction, actual GPU, drift and
limitations. Do not mix 033 into these primary plots. Explain no block-specific
accuracy-optimal scale fitting was attempted; this is a decoder-cost test.
Cluster Python has no Matplotlib: remote validate/mark success first, collect
data, then render locally using existing instruction-cost .venv read-only.
Inspect PNG and HTML at full useful size. Do not rerun successful GPU data to
fix a rendering bug. No extra profiler runs or unsolicited experiment expansion.

## Cluster execution and safety (binding)

Read CLUSTER_RULES.md and docs/block_scale_dot_context.md completely. Only
obsolete scientific inventory/paths are superseded; infrastructure rules stay.
Explicit user authorization here covers scoped submissions/retries, overriding
historical per-job approval prompts, not one-GPU or availability constraints.

- Exact SSH: ssh -o BatchMode=yes -o ConnectionAttempts=1 -o ConnectTimeout=15
  10.152.225.230. User timofeirusanov; existing key/config. Do not change auth,
  inspect secrets, or change VPN settings. Parent found macOS cache cleanup
  terminating eduVPN with disk nearly full; VPN may drop. Report if blocked.
- Read-only connection/user queue/actual-H200 precheck before work. Before EVERY
  submission squeue -u timofeirusanov must be empty, including CPU/pending jobs.
  Never more than ONE GPU or concurrent user jobs. After ambiguous sbatch SSH
  response inspect queue before retrying. Cancel only your exact failed job.
- Use only gpu-nvidia-h200-2 or gpu-nvidia-h200-3, never student VM. Inspect
  sinfo/scontrol, free GPU/CPU/memory capacity and allocations. Choose ONE
  eligible node; comma-listing both nodelists previously requested both nodes.
- If wait estimate exceeds1h stop/report. If unknown, own queued job can wait
  at most1h then cancel that exact pending job/report. No endless reservation.
- Push local reviewed source then separately verify remote pull --ff-only and
  exact commit. DO NOT put sbatch after a failed pull using a semicolon. Preserve
  conflicting evidence first, no reset/force-checkout or remote source edits.
- Schedule ZERO-GPU preflight on compute,1node/1task/8CPUs/16GiB/15min. Compile
  target only sm90 Release, no fast math, static cudart, not on login node.
- GPU smoke:1GPU/8CPUs/16GiB/15min, validation +84 rows + separate sanitizers,
  each process externally bounded4min. GPU full:1GPU/8CPUs/16GiB/30min, external
  timing process bound20min. --gres=gpu:nvidia:1. Same hash-checked executable.
- Shell set -eo pipefail; source /etc/profile; set -u; module load cuda/13.1.1;
  TMPDIR=/tmp. Record actual FQDN using hostname matching binary HOSTNAME to
  avoid033's short-name/FQDN validation failure. Record GPU name/UUID/driver.
- Check job and exact log after about1min, then15min if still running, then
 30min intervals for longer waits. Actual interruptible waits, no10second queue
  polling. Respect tool wait limits without issuing extra cluster queries.
  No automation. Completion requires success markers/validated artifacts, not
  just empty squeue. Preserve failures and all successful raw data.
- Local disk was nearly full. Check available space before collecting/building;
  do not create large new environments, copy arrays/binaries, delete unrelated
  files or run a broad cleanup. Existing plotting runtime is sufficient.
- Fetch exact artifacts, verify hashes, commit/push intended034 code/results
  only on codex/block-scale-dot, no main merge; files <100MB. Keep parent plots.

Finish with exact source/result commits, preflight/smoke/full job IDs, GPU and
row counts, review/test/SASS/sanitizer proof, both graph links and short findings.
If blocked, exhaust safe local progress and report precise blocker, no invented
measurements. Stay available for future context updates as the same Sol worker.
