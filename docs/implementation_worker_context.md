# Persistent implementation-worker context

Prepared by the Astra planning agent on 2026-09-09 for a persistent
`gpt-5.6-sol` worker. This file is the compact project memory; the adjacent
`instruction_cost_sweep_plan.md` is the current implementation contract.

## Roles and communication

- The user wants Astra to handle research/design and Sol to implement, validate,
  run experiments, collect results, and build reports autonomously.
- You are the persistent Sol implementation worker. Keep this agent available
  after completion for future tasks. Do not repeatedly create replacements for
  yourself. Future parent messages will describe context changes and new tasks.
- Read this file and the current task plan fully. Do not search the entire old
  conversation before starting. Ask the parent a precise question if genuinely
  missing research intent affects implementation. The parent retains history.
- Maintain `docs/instruction_cost_sweep_progress.md` with commits, evidence,
  current stage, exact job IDs and paths, failures/fixes, and next action. This
  file lets you resume after compaction or interruption without guessing.
- You are not alone in this repository. Preserve other agents' and users' edits.
  Your ownership is the new worktree and files assigned in the task plan.
- You may spawn one bounded independent code-review subagent as the user
  explicitly requested independent review in the experiment workflow. Use
  GPT-5.6 Sol, a fresh context, and a narrow review assignment. That reviewer
  must not submit jobs, change source, or interrupt another worker.
- Send the parent meaningful milestones or actual blockers, not continuous
  polling output. Parent need not approve routine changes or job submissions.
- Do not stop after creating code or queueing a job; pursue the complete outcome
  subject to the explicit queue-wait and resource constraints below.

## Research background and what matters

The project evaluates custom numeric storage through memory accessors. A kernel
loads compact encoded numbers, decodes into FP32 or FP64 registers, and performs
ordinary arithmetic. The primary interest is storage quality versus conversion
overhead, especially HPC DOT and GEMV on H200 GPUs.

Earlier experiments tested IEEE-like formats, LNS, posit, linear/log takum,
codebooks, split lookup tables, and DyadicNormal32. At 32 storage bits with FP64
arithmetic, native FP32 conversion and certain shifters were fast. Lookup
performance depended strongly on locality. More complicated decoders could
consume much of the speed advantage from halving storage versus raw FP64.

The recent compander experiment found that simple integer/polynomial decoding
can approach FP32-to-FP64 timing. It used fixed scalar x1 DOT, 512 blocks x 256
threads, and several N values. These results motivate a deliberately simple
instruction-budget sweep. We are now measuring controlled decoder cost, NOT
format accuracy, an optimal number density, or a universal GPU simulator.

User priorities:

- Make experimental decisions explicit before implementation, then follow them.
- No arbitrary screening that skips requested points or larger N values.
- No geometric mean over distributions, sizes, or kernels as a headline metric.
- Keep DOT and GEMV separate; scalar x1 only, no x4/x8 packing/shuffling tests.
- Show actual kernel milliseconds, all baselines freshly measured together.
- HTML reports and readable PNG/SVG plots, not only a Markdown narrative.
- Direct labels with dotted extensions; avoid overlapping tiny legends.
- All mockup timings are invented and must never be mixed into actual results.

## Current task and local ownership

- New worktree:
  `/Users/tae/Desktop/research/universal_types/accessor_universal_test_instruction_cost`
- Branch: `codex/instruction-cost-sweep`.
- Created from `9202a37`, the last compander32 report commit.
- Main shared checkout:
  `/Users/tae/Desktop/research/universal_types/accessor_universal_test`
  is DIRTY with user report and accuracy-screen changes. Do not edit/reset it.
- Your experiment: `032_instruction_cost_sweep`.
- Put runs under `results/032_instruction_cost_sweep/run_<UTC>_<jobid>/`.
- Git remote already configured: `origin`, GitHub repository
  `https://github.com/taaae/accessor_universal_test.git`.
- Use `apply_patch` for local code/doc edits. Generated artifacts may be written
  by the normal generators. Commit focused changes and push this branch.
- This task does not authorize merging to main or editing previous reports.

## Relevant reusable code (read selectively)

In your worktree:

- `src/compander32_bench.cu`: deterministic inputs, CUDA buffer/error helpers,
  scalar DOT, common reduction, baseline implementations, timing/CSV output.
- `tools/analyze_compander32_benchmark.py`: summary and standalone report style.
- `tools/check_compander32_codegen.py`: SASS section/parser examples. Its tests
  are weaker than this task's required chain audit; do not use it unchanged.
- `scripts/run_compander32_benchmark.sh` and `*_h200.sbatch`: environment,
  result manifest, compilation, stage timeouts, and run/report workflow.
- `src/e2e3_strategy_bench.cu`, `raw_fp64_gemv`: row-major scalar GEMV structure.
  Adapt to the SAME reduction helper used by your other variants and baselines.
- `CMakeLists.txt`, `pyproject.toml`, `uv.lock`: existing build/dependency setup.
- `results/031_compander32_conversion_cost/run_20260831T185459Z_455152/full/`:
  previous real report/data, for style and comparison of methodology only.

Read-only sibling if helpful:

`/Users/tae/Desktop/research/universal_types/accessor_universal_test_conversion_calibration/`

- `conversion_calibration/codegen.py`: inline-PTX helpers for add, logic,
  rotation, integer MAD, conversion, DFMA, and named-kernel generation.
- `conversion_calibration/sass.py` and related analysis tests: parsing and
  dependency-analysis ideas. Inspect before trusting; this task needs exact
  path counts, not a statistical approximation.
- DO NOT import its 112-case experiment, regression model, different N,
  multiple-chain mixes, coefficient LUTs, or nonfinite distributions.
- DO NOT copy its `finite_from_u32` bit construction: our common baseline is
  a numerical UInt32-to-FP64 conversion.
- DO NOT copy its integer multiply-ADD helper for the multiplication curve:
  the final user chose multiplication only.

Mac has no CUDA device/toolkit suitable for execution. Use standalone C++ host
tests locally and `uv run` for Python. Root CMake requires CUDA even for host
targets, so compile the new host-only test directly with clang++ if necessary.

## Cluster identity and authentication

- COMA SSH destination: `10.152.225.230`.
- User: `timofeirusanov`.
- Login host observed: `login.int.coma-cluster.de`.
- Exact working command:
  `ssh -o BatchMode=yes -o ConnectionAttempts=1 -o ConnectTimeout=15 10.152.225.230 ...`
- Authentication uses the workstation's existing SSH configuration/private key.
  The latest check authenticated with publickey. No password is needed or known
  in this handoff. Do not print/copy private keys or hunt for password files.
- Remote repo: `/storage/home/timofeirusanov/accessor_universal_test`.
- Actual eligible H200 nodes ONLY:
  `gpu-nvidia-h200-2`, `gpu-nvidia-h200-3`.
- Partition `compute`; GPU request `--gres=gpu:nvidia:1`.
- Student VMs and any other GPU nodes are excluded for this experiment.
- On 2026-09-09, most recent parent read found no user jobs, 4 unallocated GPUs
  on node 2 and 3 on node 3. This is stale immediately; recheck before submission.
- Nodes can be MIXED and still have unallocated GPUs. Compare CfgTRES and
  AllocTRES, state, CPU/memory availability, reservations, and pending reasons.
  Never cancel other users' jobs.

## Authorization and rules, including historical-rule conflict

Read `CLUSTER_RULES.md` fully. The current user explicitly authorizes this
worker to implement, test, run smoke, submit full GPU work, fix and retry,
collect results, and create graphs WITHOUT another approval question.

User instruction on 2026-09-09:
"No need to ask for my approval to run jobs, but it should use cluster rules
that I defined earlier (like only 1 gpu at a time, ocassionally check job
progress etc)"

This supersedes the older repo text demanding per-submission approval. It does
not relax resource limits or authorize unrelated jobs. Do not alter or delete
`CLUSTER_RULES.md` to hide the historical conflict. Treat the generic old
precision-packing build instruction as a requirement for a NEW-TARGET CPU-only
CUDA preflight, not an instruction to build unrelated format experiments.

Hard resource rules:

1. Never use more than ONE GPU at any time, across this workflow.
2. Before EACH submission, run `squeue -u timofeirusanov`. If there is any
   running/queued/completing user job, do not submit a second job or cancel the
   unrelated job. Wait/report. No simultaneous smoke/full/preflight jobs.
3. CPU-only compilation preflight requests ZERO GPUs. Never compile large CUDA
   targets on the login node. GPU smoke and full each request exactly one GPU.
4. Recheck `sinfo`, `scontrol show node` for both eligible nodes and user queue
   immediately before reserving a GPU.
5. Prefer an eligible idle node; otherwise use an eligible node with available
   GPU/CPU/memory, respecting the ONE GPU request. Choose node at submission.
6. This Slurm installation treats a comma-separated `--nodelist=node2,node3`
   as requiring BOTH nodes and rejects it with `--nodes=1`. DO NOT use that.
   Pick one observed eligible node. If neither is free, inspect expected end
   times/pending estimates. Earlier user rule: if wait is estimated over one
   hour, stop cluster work and notify parent with implemented state saved.
   Otherwise queue one job on the suitable earliest-available eligible node.
   If estimate is unknown, report uncertainty and use a bounded one-hour wait;
   do not leave an indefinite pending reservation. Only cancel your exact pending
   job to enforce that bound, not unrelated user jobs.
7. Use finite Slurm limits and shorter process timeouts. Log resource request,
   expected duration estimate, hard limit, selected node, and job ID.

## Cluster workflow

1. Develop only on the workstation. Run local tests/syntax checks, commit, push.
2. Use short SSH commands to check remote branch/status. Preserve all remote
   changes/results. No source edits on cluster and no forced checkout/reset.
   If remote is clean, fetch the new branch and switch/create its tracking branch
   non-destructively; then `git pull --ff-only`. Verify exact commit.
   If the remote worktree cannot switch safely, ask parent about a separate
   remote checkout, not the user for routine submission permission.
3. Submit CPU-only CUDA compile/SASS preflight for the NEW target.
4. After it passes, independently recheck queue and GPU availability; smoke.
5. Smoke must pass runtime correctness, sanitizer checks, and assembly gates.
6. After smoke finishes, independently recheck queue and availability; full run.
7. Initial status/log check about ONE MINUTE after submitting any job. If still
   pending, distinguish no output from failure and check again after it starts.
8. If running, progress/log check around FIFTEEN MINUTES from start, then every
   THIRTY MINUTES. Jobs that finish earlier need no artificial 15-minute wait.
   Each wait must be a real bounded sleep/timed wait, not a loop that queries
   clock/Slurm every few seconds. Use available interruptible waits. Do not keep
   an SSH session open while waiting. If a tool has a 60-second per-call limit,
   do not violate it; avoid turning those tool limits into repeated Slurm polls.
9. Progress messages must identify current case and completed case count and
   flush output. A stalled CUDA process must be bounded by an external timeout.
10. On failure inspect exact log, fix locally, retest, commit/push, fast-forward
    remote, remove only exact failed artifacts after preserving a compact
    diagnostic record, and retry the smallest relevant stage. No new approval
    needed. Preserve successful results. Report what was removed; normal rm
    removal is not recoverable.
11. After success inspect log/completion manifest, then collect results locally.
    Preferred existing repo workflow: commit ONLY intended generated results on
    cluster, push same branch, pull --ff-only locally, then generate local report.
    Keep local and remote history sequential to avoid divergent commits.
    If collection through rsync is simpler, that is an acceptable alternative:
    copy exact successful run, verify it, then commit/push results locally.
12. No normal performance timings under Nsight instrumentation. A sanitizer run
    is separate from timing and never contributes timings to plots.

## Cluster environment and previous failure lessons

- Batch initialization order:
  `set -eo pipefail`; `source /etc/profile`; `set -u`;
  `module load cuda/13.1.1`.
  Enabling nounset before /etc/profile has caused DEBUGINFOD_URLS failures.
- Set `TMPDIR=/tmp`. Previous scratch TMPDIR did not exist.
- H200 target `sm_90`, CMake `CMAKE_CUDA_ARCHITECTURES=90`.
- Slurm copies sbatch scripts; use `SLURM_SUBMIT_DIR` as repository root, not
  BASH_SOURCE in the spooled script. Submit from the repo root.
- Use static CUDA runtime consistently. Do not link explicit shared cudart
  alongside a nominal static-runtime setting. Verify build/run logs.
- `sacct` has been unavailable. Use `squeue`, `.out`, exit markers/manifest.
- Empty squeue is not proof of success. Confirm explicit stage-complete markers,
  exit status and complete output inventory.
- Earlier smoke missed partial-warp deadlocks. Cover full warps, multiple
  blocks, ragged input lengths, synccheck and memcheck before full timing.
- Earlier codegen checker rejected valid Hopper FSEL because it recognized
  only SEL. Audit actual SASS semantics; do not blindly hardcode older opcodes.
- Earlier CSV label `Posit<32,2>` was not quoted, corrupting analysis. Use a
  proper CSV writer/escaping; fail on malformed fields/missing rows.
- Earlier build failed due mismatched floating return type in a lambda. Compile
  every new specialization in CPU-only cluster preflight before GPU smoke.
- Individual files must stay below GitHub's 100 MB limit. Do not track large
  binaries, input arrays, or giant profiler captures. Keep raw timings and SASS.

## Completion response

Provide exact commits/branch, GPU/node/job IDs, test/review evidence, result and
HTML/PNG paths, number of expected/observed timing rows, instruction-audit
coverage, and honest threshold findings. No fabricated curves, no claiming
unmeasured points or a universal decoder complexity limit. Parent will use this
evidence to discuss scientific conclusions with the user.
