# Block-scale reconstruction progress

- Stage: local implementation and test review.
- Branch: `codex/block-scale-dot`.
- Active contract: `docs/block_scale_reconstruction_034_plan.md`.
- Experiment 033 results and the parent-owned untracked per-element plotting
  script and four graphs remain untouched.
- Implemented a separate experiment-034 benchmark, host references, SASS
  checker, strict result validator, analysis tool, tests, and Slurm wrappers.
- Current inventory is 14 cases per N: nine per-element reconstruction cases,
  two B128 thread-local deferred cases, and three native baselines.
- Local checks: Release `-O3 -DNDEBUG` host test passed, 32 Python tests passed,
  and Python compilation, shell syntax, synthetic 3,500-row rendering, and
  `git diff --check` passed. Both requested synthetic graph layouts were
  inspected at full resolution.
- Read-only cluster precheck on 2026-09-09 found an empty user queue. H200-2 had
  all eight GPUs allocated; H200-3 had seven of eight allocated. No job was
  submitted.
- Independent source review found no blocker after fixes. The mandatory next
  gate is a zero-GPU preflight followed by hash-bound manual inspection of the
  actual Hopper deferred loop and predicates.
- Source commit `be9c38a` was pushed. The remote fast-forward initially stopped
  because preserved 033 artifacts existed as untracked copies. Those exact
  paths were moved to
  `/storage/home/timofeirusanov/block_scale_033_remote_untracked_backup_20260909_034pull`;
  the remote worktree then fast-forwarded cleanly. No file was deleted.
- Zero-GPU job `460480` failed before its script started because the new Slurm
  output directory did not exist. Scheduler state recorded `JobLaunchFailure`;
  it used no GPU and produced no build artifact. The exact experiment directory
  was then created.
- Replacement zero-GPU job `460481` compiled all 16 timed kernels and passed the
  host test with zero spills. Its checker failed closed on compiler register
  reuse and commuted multiplication operands in both local-deferred kernels.
  The SASS and build log are preserved under
  `results/034_block_scale_reconstruction/preflight_460481/`.
- Inspection showed the intended static deferred shape: two payload loads, the
  correct FP32/FP64 scale loads, two DFMAs, one DMUL, zero SHFL, and a four-trip
  loop. The checker now handles phase-local last writers and commuted multiply
  operands; it passes all 16 actual kernels and all 32 local tests. Independent
  manual review of the hash-bound SASS is in progress.
- Next: finish actual-SASS review, commit the checker calibration and preserved
  preflight evidence, then run a replacement zero-GPU preflight. GPU smoke stays
  blocked until its SASS hash matches the reviewed cubin and every gate passes.
