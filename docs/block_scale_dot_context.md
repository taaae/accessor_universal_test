# Persistent Sol worker: block-scale DOT context update

2026-09-09. Read this file and block_scale_dot_plan.md completely before coding.
This is the NEW active experiment. Experiment 032 is finished, not to be resumed.

## Role and authorization

The user explicitly requests the existing GPT-5.6 Sol implementation subagent
to implement, independently review, smoke-test, run on the cluster, collect
results locally, and create graphs. Astra owns the design below; implement it
without inventing extra experiments. You may make normal implementation fixes,
but notify the parent before changing the scientific setup, counted cases,
semantics, timing, hardware, or resource policy. No additional user approval is
required for the described cluster stages or scoped retries.

You are not alone in these repositories. Do not revert other edits. Your new
ownership is the block-scale worktree and files assigned in the plan. Reuse
your existing independent reviewer with a bounded read-only assignment; it must
not submit jobs. One new review agent is permitted if that reviewer cannot be
resumed, using Sol and fresh context. Keep the parent free for discussion: send
milestones and blockers, not frequent unchanging status messages.

## What changed since experiment 032

The measured instruction-cost experiment is complete at 9911f0c. The parent
independently reran host tests, six audit tests, the 88-specialization SASS
audit, and the complete analyzer. All 3,600 rows and saved summaries matched.
The GPU was an H200 NVL on gpu-nvidia-h200-3. The full job was 460407, smoke
460404, preflight 460399. Those are historical IDs, NOT jobs to resubmit.

Parent changed the OLD report generator and graphs after your completion:
direct right-side baseline labels and measured crossing interval shading.
These changes are uncommitted in the instruction-cost worktree. Preserve them.
Its report/screenshot.html is also parent-owned. Read those files for style if
useful, but do not edit, commit, or push that worktree for this new task.

The research now examines block compression. Scope starts with FP32 payloads
and one scale per storage block. This is a cost experiment, not an accuracy
tournament. Both generic reconstruction at each access and a DOT-specific
deferred-scale optimization matter. The latter cannot establish a generic
accessor speedup for arbitrary kernels.

Approved: DOT only, scalar x1, FP64 compute except raw-FP32 baseline, five N,
block sizes 16/32/128, FP32/FP64 scale storage, per-element/deferred paths.
15 configurations at each N; 75 cases total; all must run. No screening of
larger N, no geometric averages. One main final graph shows ALL 15 curves.
The earlier seven-curve preview used invented values and excluded deferred
paths and B=128. NEVER reuse its numbers in any measured artifact.

## Worktree and source ownership

New local worktree already created from 9911f0c:
/Users/tae/Desktop/research/universal_types/accessor_universal_test_block_scale
Branch: codex/block-scale-dot
Experiment: 033_block_scale_dot
Remote: origin, https://github.com/taaae/accessor_universal_test.git

Use a separate remote worktree, following the established safe workflow:
/storage/home/timofeirusanov/accessor_universal_test_block_scale
Existing remote git repository may be used only to fetch/add that worktree;
preserve every existing checkout and its results. No force switching/reset.
The main checkout and all sibling experiments are outside your write scope.

New progress record: docs/block_scale_dot_progress.md. Record exact commits,
jobs, status, remote/local paths, evidence hashes, blockers, and next action.
The inherited docs/implementation_worker_context.md supplies historical
cluster lessons, but its experiment-032 task and paths are superseded here.

## Cluster rules, still binding

- SSH: ssh -o BatchMode=yes -o ConnectionAttempts=1 -o ConnectTimeout=15
  10.152.225.230. User timofeirusanov, login.int.coma-cluster.de.
- Existing SSH key/config handles authentication; no password is needed/known.
  Do not copy keys, print secrets, hunt credentials, or change VPN settings.
- Before EVERY submission run squeue -u timofeirusanov. If ANY user job is
  pending/running/completing, do not submit another. Inspect before retrying
  uncertain SSH responses, to avoid duplicate jobs. Never cancel unrelated jobs.
- At most ONE GPU at a time. CPU preflight uses ZERO GPUs, sequential with all
  other jobs. GPU smoke/full each request --gres=gpu:nvidia:1.
- Actual H200 nodes only: gpu-nvidia-h200-2, gpu-nvidia-h200-3. NO student VM.
  Check sinfo, scontrol nodes, CfgTRES/AllocTRES, CPU/memory, user jobs, relevant
  allocation end times. MIXED can have capacity. Never assume availability from
  prior messages. Prefer idle eligible node; otherwise free eligible capacity.
- Slurm compute partition. Pick ONE eligible node with --nodelist at submission.
  Do not comma-list both nodes: this cluster interprets it as requesting both.
- If estimated availability exceeds one hour, stop cluster work and report.
  If unknown, a pending job may wait at most one hour; cancel ONLY that exact
  own pending job on reaching the limit, save status, and report. Do not leave
  indefinite reservations or create a monitoring automation.
- Code locally, test, commit, push; safely fetch/pull --ff-only remotely.
  Never edit source on cluster. Compile via scheduled zero-GPU preflight,
  never expensive compilation on login. Reuse exact audited binary in smoke/full.
- Read CLUSTER_RULES.md. Current user approval explicitly covers these jobs and
  automatic scoped retries, superseding historical per-job approval requests.
  Do not modify that document or relax any resource/safety restriction.
- Scripts: set -eo pipefail; source /etc/profile; set -u;
  module load cuda/13.1.1. TMPDIR=/tmp. Repo=SLURM_SUBMIT_DIR.
  sm_90, Release, no fast math, static cudart without explicit shared cudart.
- Short SSH calls only. Around one minute after submission/start check state
  AND the exact log. If running at 15 minutes check progress, then at 30-minute
  intervals for longer waits. Use actual interruptible sleep, not frequent
  clock/queue polling. Respect tool per-call wait limits without converting
  them into extra cluster queries. Finish earlier jobs without artificial delay.
- Bound processes externally and all jobs with Slurm --time. Preserve all
  successful results; preserve failure diagnostics before exact failed-artifact
  cleanup. No broad deletion. Only own failed paths may be removed and reported.
- No Nsight in timed runs. Compute Sanitizer only in separate validation.
- An empty squeue is not success. Check explicit stage markers, return status,
  exact artifact inventories and contents. sacct may be unavailable.
- Commit/push intended source and collected results only, no big binaries or
  input arrays. Individual tracked files <100 MB. No main merge.

## Last experiment's lessons to apply now

1. The cluster Python lacks Matplotlib. DO NOT make a successful GPU timing job
   fail by trying to render there. Do stdlib CSV validation remotely and create
   graphs locally after collection. The existing instruction-cost .venv has
   Matplotlib/NumPy; use it read-only or create this worktree's own environment.
2. CUDA/SASS shows Hopper I2FP aliases and register reuse modifiers. Inspect
   actual machine code; do not pass on bare source counts or invented resources.
3. User requested all operations to remain where specified. Explicit rounding
   and source structure plus SASS verification prevent unintended factoring.
4. Use explicit failing checks, not assert that disappears in optimized tests.
   Validate manifest identity and numerical dimensions, not merely row counts.
5. GPU edge inputs must actually be in the GPU validation buffers; checking
   only finite host edge results is not a GPU boundary test.
6. Check every output and baseline against the proper semantic reference.
   Raw FP32 has FP32 accumulation; widened FP32 and raw FP64 do not.

Proceed through complete implementation -> independent review -> local tests ->
zero-GPU preflight -> GPU smoke -> GPU full -> local collection/analysis/plots.
When blocked on VPN, finish safe local work and report one concrete blocker;
do not busy-wait. Resume from recorded state when the user reconnects.
