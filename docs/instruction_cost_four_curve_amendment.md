# Approved four-curve amendment, 2026-09-09

The user approved exactly four curves: FP64 addition, FP64 multiplication,
FP64 FMA, and 32-bit rotation. Resume implementation and the previously
authorized complete execution workflow. This amendment supersedes conflicting
family/recipe/count requirements in instruction_cost_sweep_plan.md. Everything
else in that plan and implementation_worker_context.md remains in force.

## Exact decoder recipes

The stored input remains uint32_t. Each operand has its own serial chain.
Common X=0 is numerical UInt32-to-FP64 conversion, measured once per kernel.

| Family | Decoder for K steps | Runtime scalar operands |
| --- | --- | --- |
| add64 | v = double(u); repeat v = __dadd_rn(v, a) | a = 0.1 as binary64 |
| mul64 | v = double(u); repeat v = __dmul_rn(v, m) | m = 1.0001 as binary64 |
| fma64 | v = double(u); repeat v = __fma_rn(v, m, a) | same m and a |
| rot32 | repeat u = rotate_left_32(u, s); return double(u) | s = 7 |

Use the supported round-to-nearest CUDA intrinsics, or their equivalent
explicit-rounding PTX instructions. No fast math or reassociation that changes
rounding semantics. Coefficients are runtime kernel parameters, reused in
registers; no per-step loads. These coefficients keep all values finite through
K=64 for the existing input domain. Record their exact binary64 bits in manifests.

Remove integer add, XOR and low-product integer multiplication from current
timing cases, audit requirements, plots and expected inventories. Preserve their
old preflight evidence as historical failures, not successful timing results.
No multiply-high or FP32 arithmetic curves are requested.

## Counts, code generation and validation

Keep initial K=0,1,2,4,8,12,16,24,32; zero is the shared measured u32_base.
There are 4*8 nonzero curve cases plus four baseline cases per kernel, yielding
36 cases/kernel, 72 initial cases total and 3600 initial timing rows at 50
samples. Preserve optional K48/64 extension rules with stage-matched baselines.

Retain the exact fixed DOT/GEMV dimensions, scalar access, geometry, input
seeds, sampling order policy, baselines, drift checks, and output validation.
Each decoded operand must actually execute K dependent target operations in
the timed machine-code kernel. Trace FP64 register pairs from conversion into
the chain and through the final accumulation. Distinguish reconstruction from
the DOT/GEMV FMA and from reductions. Record actual DADD/DMUL/DFMA/SHF or
equivalent semantically correct lowering. If a requested operation lowers to a
single equivalent instruction, document it rather than assuming source spelling
equals the opcode. Do not accept folded chains, untracked overwritten registers,
or multi-instruction preservation tricks. The retained-instruction X axis still
requires honest assembly evidence; fail closed on unsupported cases.

Independent host references must execute separately rounded addition and
multiplication in binary64; use std::fma for the fused chain. Do not enable
reassociation/fast-math in host references. Validate actual kernel outputs and
all baselines in addition to standalone decoder probes, including ragged tests.

## Interpretation and worker next actions

The user accepts 'FP32 might allow approximately twice as many operations' as
a rough planning heuristic. It is NOT a measured finding, guaranteed bound,
or an estimate for integer instructions. If mentioned in the report, label it
as an unvalidated peak-throughput-based heuristic; do not plot invented FP32
curves or turn FP64 crossing brackets into measured FP32 thresholds. Report
the actual FP64 and rotation thresholds clearly first.

Reuse the persistent Sol worker and its independent reviewer. Resolve ALL
outstanding review findings, not just the earlier allocation bug. A successful
CUDA compilation is not an assembly/correctness pass. Update tests, parsers,
inventories and report labels for this amendment. Follow-up independent review
must inspect the current changes and remaining original findings.

Then proceed through zero-GPU scheduled compile/SASS preflight, actual H200
GPU smoke, full benchmark, collection and graphs. Previous GPU submission
authorization remains; one GPU and no concurrent user jobs remain hard rules.
Known separate remote worktree is now
/storage/home/timofeirusanov/accessor_universal_test_instruction_cost.
Use that checkout and preserve the previous repository's results. Maintain
accurate progress/commits/hash provenance and send meaningful milestones.
