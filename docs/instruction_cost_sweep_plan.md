# Experiment 032: dependent instruction cost inside DOT and GEMV

Implementation contract from Astra, 2026-09-09. Read together with
`implementation_worker_context.md`. The user authorized end-to-end execution
and automatic scoped fixes/retries. This is a new small experiment, not a rerun
of the older conversion-calibration regression project.

## 1. Research question, final approved choices, scope

Measure how kernel time changes as a 32-bit storage decoder performs more
dependent instructions before FP64 arithmetic. Compare directly with fresh
raw-FP32, FP32-to-FP64, and raw-FP64 baselines on the SAME H200 allocation.

Exactly five main curves:

1. 32-bit integer addition;
2. 32-bit XOR;
3. 32-bit rotation (NOT logical shift);
4. 32-bit integer multiplication (NOT integer multiply-add);
5. FP64 fused multiply-add.

Integer arithmetic is uint32_t modulo 2^32 for defined behavior. Display labels
may say "Int32 addition" etc. but report explicitly that arithmetic wraps
unsigned; no signed-overflow UB. All steps within one operand's decoder form a
single serial dependency chain. Independent left/right operand work and work
across threads are permitted and natural. No extra multi-chain decoder curve.

X = actual retained target-family machine instructions PER DECODED VALUE,
excluding input loads, common numerical UInt32-to-FP64 conversion, loop/index
overhead, kernel accumulation and reduction. Each dot product term decodes two
operands, thus nominally 2X extra target instructions per multiply-accumulate.
SASS, not source spelling, certifies X. One DFMA is one instruction, not two
FLOPs on this axis. A multiply implemented as IMAD with zero addend is still
one multiply instruction; report its actual opcode.

No LUTs, divide/exp functions, real-format sweep, mixed-operation curve,
multi-chain sweep, vectorized x4/x8, Tensor Core kernels, fitted prediction
model, clock-latency calibration, or accuracy tournament in this first stage.
No new N sweep. Do not silently reintroduce superseded research ideas.

## 2. Fixed geometry and input footprints

Full run:

| Item | Setting |
| --- | --- |
| DOT length | 67,108,864 = 2^26 |
| DOT first-stage grid | 512 blocks x 256 threads |
| DOT final reduction | one block x 256 threads, timed |
| GEMV shape | 4096 rows x 16384 columns = 2^26 matrix elements |
| Matrix layout | contiguous row-major, leading dimension 16384 |
| GEMV grid | one block per row, 256 threads |
| GEMV vector | encoded uint32_t, length 16384; decode inside each row kernel |
| Access | scalar x1, one accumulation register per thread |
| Arithmetic | FP64 everywhere except raw-FP32 baseline |
| Baselines | raw FP32, FP32 -> FP64, raw FP64 |
| Initial X | 0, 1, 2, 4, 8, 12, 16, 24, 32 |
| Warmups/samples | 10 warmups per variant, 50 measured launches per variant |
| Compilation | Release -O3, -lineinfo, ptxas verbose, sm_90, no fast math |

Keep the identical reduction algorithm, thread layout and indexing structure
across all compared variants of a kernel. Reuse simple shared-memory block_sum
from compander32 if practical. Shared scratch may differ only by arithmetic
element size for FP32 vs FP64. No return before a required block barrier. Use
`#pragma unroll 1` on grid-stride input loops so dynamic instruction accounting
is simple; unroll decoder chains at compile time. No per-format grid tuning,
register-count caps, or altered reduction tree to make one variant win.

DOT uint32 inputs total 512 MiB; raw-FP64 inputs total 1 GiB. GEMV uint32 matrix
is 256 MiB and vector 64 KiB. No need to retain unrelated prior datasets. Raw
input/reference generation and host/device transfers are outside event timing.

## 3. Deterministic finite data and common decoder

Use full-width uniformly distributed uint32 codes from a deterministic
counter-based generator, with independent streams for each operand. Prefer a
small SplitMix64 hash of index, take its high 32 bits; define exact generator
once for CPU/GPU and test identical results. Seeds:

- left/vector stream: 0x6bd87c012a53f9e1;
- right/matrix stream: seed XOR 0x9e3779b97f4a7c15.

For DOT, use distinct streams for left and right. For GEMV, matrix and vector
also use distinct streams. Do not generate either operand as a copy of the
other. Generate once per full run and reuse arrays for every X/family.

Common X=0 decoder: `double v = static_cast<double>(u);` with a verified native
UInt32-to-FP64 conversion. This preserves every uint32 value exactly as FP64.
Do NOT reinterpret raw codes as float bits (NaNs/Inf); do NOT bit-construct a
custom mantissa (different common decode cost). Define a single measured
`u32_base` case per kernel; use those SAME 50 samples as the X=0 anchor of all
five curves. It is not assumed equal to native FP32-to-FP64.

Raw baselines derive from this same source: raw64 = double(u), raw32 = float(u).
FP32-to-FP64 reuses the raw32 arrays. Raw32 has the expected rounding and uses
FP32 accumulation. These are performance baselines, not assertions of equal
numerical precision. All resulting values, products, sums remain finite.

Rationale: add, XOR, rotation and multiplication by an odd number are
permutations of uint32 codes, so uniform code distributions remain uniform
under every step. This avoids changing locality or data-dependent branches.
The affine FMA chain below keeps a comparable magnitude scale. This is a
uniform-code instruction experiment, not N(0,1) data or a normal-number format.

## 4. Exact operation recipes

For integer families, perform K selected instructions on u, then common
numerical UInt32-to-FP64 conversion. For FMA, common conversion first, then K
FP64 FMAs. Use the same scalar operands for both input streams.

| Family | Recurrence | Operand(s) |
| --- | --- | --- |
| add32 | u = u + a modulo 2^32 | a = 0x9e3779b9 |
| xor32 | u = u XOR mask | mask = 0xa5a5c3c3 |
| rot32 | u = rotate_left_32(u, s) | s = 7 |
| mul32 | u = u * m modulo 2^32 | m = 0x0019660d = 1664525, odd |
| fma64 | v = fma(v, a, b), round to nearest | a = 0x1.001p0, b = 0x1p-12 |

The fixed XOR operand makes even K return the original code; rotation repeats
at 32 steps. This is acceptable in an explicitly retained instruction-cost
test: values remain nondegenerate/uniform and SASS proves the work executes.
Do not add random masks, extra per-step hash arithmetic, per-step memory loads,
or branches solely to make the mathematical function more complicated.

Use runtime scalar kernel operands in registers, hoisted once OUTSIDE the
element loop, rather than many stage-specific constants. Use only required
operands for each kernel. Those setup costs are reported, not multiplied into
X. Runtime multiplicands also discourage replacing multiplication with shift
strength reduction. Assert no per-iteration coefficient loads arise.

Start from small inline PTX volatile helpers:

```
add.u32 out, previous, operand;
xor.b32 out, previous, operand;
shf.l.wrap.b32 out, previous, previous, shift;
mul.lo.u32 out, previous, operand;
fma.rn.f64 out, previous, a, b;
cvt.rn.f64.u32 out64, input32;   // common conversion
```

Repeat helpers using compile-time K and full unrolling. Use proper operand
constraints. No volatile input arrays, memory clobbers, dummy global writes,
inline clocks, sleeps, or synchronization inserted to force chain preservation.
PTX volatile alone is NOT proof; ptxas may still transform the instructions.
If the chain is folded, first adjust scalar operand placement/asm structure.
Do not silently replace the desired primitive with a mixed recipe or hide
preservation overhead. If exact chain semantics cannot be achieved safely,
send parent evidence before any full timing.

## 5. Assembly acceptance is a hard gate

Compile a statically identifiable entry point for every `(kernel,family,K)`.
Use readable generated extern-C wrapper names if that makes inspection easier.
All required full-run K values, plus optional extensions 48 and 64, must compile
in preflight. There must be no runtime K loop or family dispatch inside kernels.

Save `cuobjdump --dump-sass` and source/line correspondence where available.
Automated audit MUST inspect actual timed DOT/GEMV kernels, not only a separate
decoder probe. A probe can help understand opcodes, but cannot replace this.

Audit facts per specialization:

1. Locate input loop, its two scalar source loads and one arithmetic accumulation.
2. Trace source-value registers through target operations to common conversion
   and accumulation. Integer pointer increments are NOT decoder additions.
3. Each loaded operand follows exactly K target operations, each consuming the
   previous target result, ending in the value consumed by the accumulation.
4. In FMA kernels, separate K+K reconstruction DFMAs from the one DOT/GEMV DFMA.
   Reduction arithmetic and pointer/address math are excluded from X.
5. The hot loop executes one logical element per iteration as intended. If the
   compiler unrolls despite requested control, either prevent it or rigorously
   account for the replication in control-flow analysis; don't count a whole
   function and divide by two without evidence.
6. Track register pairs for FP64 instructions, compiler MOVs/renaming, predicates,
   uniform-register operands and actual instruction variants. Count semantically
   correct IADD3 or IMAD.IADD, LOP3 implementing XOR, SHF rotation, and IMAD with
   zero addend implementing low multiplication. Document actual lowering.
7. No folding into one instruction, independent chain substitution, skipped
   operations, input-independent result, strength-reduced multiply sequence,
   unexpected calls, local-memory spills, or coefficient memory traffic in loop.
8. Output `assembly_audit.csv/json` with symbol, family, K, both observed chain
   lengths, opcode histogram, status/reason, registers, local/shared bytes and
   setup/extra instructions. Keep SASS text and ptxas log alongside it.

A bare opcode count or total instruction delta from K=0 is insufficient because
compilers can change indexing/setup. A dataflow-based audit can be narrow to
the opcodes actually observed. Inspect any unhandled case instead of passing
it. Add meaningful negative tests: chain folded, one independent operation,
pointer-add false positive, missing operand chain, and FMA accumulation mistaken
for reconstruction. Fail closed on unsupported parsing.

Accept a scalar multiplication lowered to one IMAD with zero addend, but label
it clearly. If multiplication is multiple instructions, do not call K the number
of executed instructions; resolve with parent instead of corrupting x semantics.

Spills or unexplained extra work are grounds to fix code before benchmarking.
If a genuine register/occupancy change survives a clean implementation, record
it; do not force identical register counts or secretly tune per variant.

## 6. Correctness and smoke design

Host-only tests:

- independent straightforward uint32 reference recurrences for all families/K;
- edge codes 0, 1, 2, 0x7fffffff, 0x80000000, 0xfffffffe, 0xffffffff;
- deterministic random sample at least 4096 codes;
- rotation reference avoids shift by 32 and checks bit preservation;
- uint32 wrapping, odd multiplication and XOR periodicity checked explicitly;
- FMA reference uses `std::fma` or another true fused operation, not mul then add;
- finite result checks through K=64 and upper input bound;
- deterministic input generator agreement, manifest uniqueness and expected counts.

GPU smoke uses same binary and full specialization set, including K=48/64.
Small validation-only decoder kernels can return decoded values for exact CPU
comparison. Integer-to-double outputs should match exactly. FP64 chain values
should match host std::fma bitwise or have an explicitly justified portability
tolerance, never a tolerance large enough to hide a skipped operation.

Small full DOT/GEMV correctness validates the actual arithmetic kernels for
all cases against independently decoded host inputs and reductions. Use positive
inputs and a standard rounding-error bound based on sum of absolute products
and accumulation length, not one arbitrary tolerance across FP32 and FP64.
Include all three baseline arithmetic semantics separately.

Cover tiny and ragged shapes such as DOT 1,31,32,33,257,4099 and GEMV
3x33 and 17x257. Timed smoke shape can be DOT 2^20, GEMV 64x1024, 1 warmup and
3 samples. Full geometry is fixed as above; smoke outputs must carry smoke mode
and never enter full plots.

Run compute-sanitizer memcheck and synccheck on a compact validation mode covering
both kernels, all families, and representative K=0,1,32,64 with ragged tails.
Nonzero sanitizer errors fail smoke. Launch errors and cudaDeviceSynchronize
errors must propagate to process exit. Never allow an unrecognized --variant
selection to produce an empty success. Require full expected output inventories.

Every timed sample's result is consumed and checked finite. Full-run CPU reference
over all 2^26 elements at all K would be wasteful; use exhaustive small GPU/CPU
smoke correctness plus full-run output checks and sample validation. Do not add
hours of host reference computation to measure sub-millisecond kernels.

## 7. Timing and experimental integrity

Same arrays, same allocation, same GPU, fixed geometry, same stream. Use CUDA
events in a profiler-free run. DOT event covers first stage AND final reduction;
GEMV event covers its complete row-kernel launch. Exclude allocation, generation,
host reads, source hashing, validation, and report generation from timed regions.

Baseline cases per kernel: raw_fp32, fp32_to_fp64, raw_fp64, u32_base. The first
three are the horizontal references; u32_base anchors all curves at X=0.

For initial full grid: five families x eight nonzero K = 40 cases, plus four
baselines = 44 cases per kernel, 88 logical cases total. Exactly 4400 measured
rows at 50 samples, before optional extensions/refinement. Do not separately
measure five redundant K=0 cases and then pretend independent samples were
collected. Curve rendering may reuse the one u32_base record visibly.

Warm each case 10 times. Then 50 measurement rounds containing every selected
case exactly once per kernel, with deterministic shuffled/rotated order so a
family is not always first or last. Keep a reproducible order seed and record
round/index for every timing. Raw-FP32 and FP32-to-FP64 read the same float array;
interleaving has cache implications, so record the policy. Main footprints are
large; no artificial cache flush kernel in the main sweep.

Log progress with current kernel/family/K/round and completed/total cases; flush
at least every completed measurement round. Include elapsed wall time. Events
must be synchronized before reading durations, and errors never discarded.

Report median, Q1/Q3, sample count and ratios to fresh FP32-to-FP64 and raw FP64
anchors, separately for DOT and GEMV. No fabricated error bars or averaged
cross-kernel score. Save baseline drift by early/late subsets; if large >5%
relative drift, retry the complete timing block once in the same allocation,
retaining both blocks and clearly labelling rerun. Never select the fastest of
multiple runs as the official result. If instability persists, report it.

Optional bounded extension, preauthorized here:

- After initial grid, if a family/kernel median at K=32 remains below raw FP64,
  measure K=48 and 64 for that family/kernel, even if some other curve crossed.
- Interleave extension trials with fresh baseline trials and retain stage IDs.
  Fit no regression. Use stage-matched baseline ratios for threshold decisions.
- Main initial plot must still show all initial points. Any extension chart
  names its extra stage; do not pretend it shares identical measured baselines.
- If still faster at K=64, report "break-even not reached through 64".
- No automatic fine-grid crossing refinement for now. Report bracket between
  measured K points; this keeps the experiment small. A future request can refine.

5% of FP32-to-FP64 is a proposed useful-overhead screen, not a hardware fact.
Report both largest tested K within 5% and raw-FP64 crossing bracket. Timing can
be nonmonotonic: preserve all points, list reentries if needed, don't force a
monotonic curve or claim exact interpolated crossing counts. Overlapping IQR
near a crossing is descriptive uncertainty, not a formal significance test.

## 8. Proposed files and ownership

Own these new files/modules and minimal CMake additions:

- `include/instruction_cost_core.hpp`: enums, manifest, seeds, host references.
- `include/instruction_cost_kernels.cuh` or generated equivalent: inline PTX
  decoder templates, named kernels, shared reduction.
- `src/instruction_cost_bench.cu`: CLI, input generation, buffers, validation,
  timing, events, checked output and progress.
- `tests/instruction_cost_core_test.cpp`.
- `tools/check_instruction_cost_codegen.py` plus parser helpers if warranted.
- `tools/analyze_instruction_cost_sweep.py`.
- `analysis/tests/test_instruction_cost_codegen.py` and analysis validator tests.
- `scripts/check_instruction_cost_build.sbatch`: CPU-only CUDA build/audit.
- `scripts/run_instruction_cost_smoke_h200.sbatch`.
- `scripts/run_instruction_cost_full_h200.sbatch`.
- `scripts/run_instruction_cost_sweep.sh`: stage runner and manifests.
- `docs/instruction_cost_sweep_progress.md`.
- `results/032_instruction_cost_sweep/...` generated evidence/report.

Do not refactor unrelated sources. Use project numpy/matplotlib/standard Python
dependencies via uv. A small generator for named specialization wrappers is
acceptable; generated files and generation inputs must match deterministically.

## 9. Review, preflight and GPU job contracts

1. Implement and run relevant local C++/Python/shell checks. `git diff --check`.
2. Have a separate Sol reviewer, fresh context, inspect code plus this experiment
   contract, not your implementation reasoning. Assign read-only review of:
   deadlocks/barriers/tails, UB, finite values, SASS chain/count evidence, scalar
   access, output consumption, measurement boundaries, baseline consistency,
   completeness/CSV, and single-GPU submission/timeout safety. Fix findings and
   obtain a follow-up review if material changes. Save concise review evidence.
3. Commit/push; remote safe branch switch/pull; queue check.
4. CPU-only compile preflight: compute partition, 1 node/task, 8 CPUs, 16 GiB,
   ZERO GPU, hard limit 15 minutes. Build ALL new specializations with CUDA
   module, host test, and run complete SASS audit. Do not restrict this job to
   expensive H200 nodes unnecessarily. Record exact executable hash and source
   commit. No CUDA device access or GPU-dependent validation in preflight.
5. GPU smoke after preflight: selected actual H200, exactly 1 GPU, 8 CPUs,
   16 GiB RAM, hard limit 15 minutes. Reuse preflight binary/manifest if source
   hash matches; verify SASS hash matches loaded executable. Include sanitizer
   validation with bounded timeout and regular smoke outputs.
6. Full timing after smoke: selected actual H200, exactly 1 GPU, 8 CPUs,
   16 GiB RAM, hard limit 30 minutes, whole-process timeout <=20 minutes.
   Expect seconds to a few minutes of timed kernels plus setup; the limit is
   safety margin, not a reason to reserve resources idly. Use the tested binary.
7. Record binary/source hashes at EACH stage; never run an old binary after a
   new source fix. A source change affecting compiled code repeats preflight and
   appropriate smoke. Pure local report fixes need no GPU rerun.
8. Check job around one minute after submission/start, around fifteen minutes
   if still running, then 30-minute intervals for longer queue/workflow waits.
   Follow detailed user-queue, availability and timeout rules in context doc.

No approval pause for these described stages or scoped retries. Hardware/queue
failure can be a genuine blocker; implementation difficulty is not by itself
a reason to omit required checks or stop with only a plan.

## 10. Result data and report

Minimum artifacts per successful full run:

- `manifest.json`: commit, binary hash, CUDA/compiler/driver, node/GPU identity,
  geometry, source seeds, scalar operands, runtime resource metadata, initial
  and extension grids, repetitions, stage IDs and timing boundaries.
- `timing_samples.csv`: mode/stage, kernel, family, K, observed instructions per
  operand, dimensions, storage/arithmetic, seed, round, execution order, ms,
  result/checksum, valid flag, GPU/job identity.
- `timing_summary.csv`: groups, median, quartiles, count, stage-matched ratios.
- correctness/sanitizer/build logs and independent-review summary.
- compiled SASS and machine-readable audit for ALL plotted cases.
- `report.html`: self-contained HTML with inline charts and complete provenance.
- `dot_instruction_cost.png/svg`, `gemv_instruction_cost.png/svg`, and a
  combined two-panel PNG for showing directly in chat.
- optional separate extension panel if extension happened.

Main graph: X retained decoder target instructions, Y total kernel milliseconds.
Five solid colored curves with markers, IQR whiskers/bands; three dashed
horizontal baselines. Direct labels with dotted extensions and collision-free
spacing, matching the user's example. Show both kernels as separate panels.
Keep actual curve shapes and ordering; never copy the mockup's synthetic values.
Baseline bands can show measured IQR. Axis range must include all real points
and references. Clearly identify X=0 as shared UInt32-to-FP64 reference.

Example appearance only, parent-created file:
`/Users/tae/.codex/visualizations/2026/08/25/01a0389d-c086-7830-8b3f-3e325d6768db/five-operation-budget-example.png`
This file is NOT experimental data. Recreate plots from new CSV outputs.

Report text should answer how much each operation family costs in these kernels,
where it exceeds +5% native conversion and raw-FP64 time, explain pipeline/dependency
limitations cautiously, and identify compiler/register changes without inventing
causal proof. State this is retained-instruction synthetic decoding, not useful
compression by repeated addition/XOR and not a universal machine cost model.

Verify full row inventory, no duplicate/missing groups, positive finite timing,
exact sample indices, actual assembly pass for every plotted point, and separate
smoke/full data. View rendered graphs and the report at ordinary desktop size,
fix clipping/overlaps, and keep the original numerical observations intact.

## 11. Finish and persist

Collect successful artifacts locally, generate and validate final report, commit
and push intended source/docs/results on this branch. Never commit device input
arrays or build binaries. Keep the worktree clean except documented unrelated
preexisting changes. Update progress doc with exact paths and commit/job IDs.

Notify parent with completion evidence and a concise findings summary. Remain
available as the same Sol implementation agent for future context updates.
