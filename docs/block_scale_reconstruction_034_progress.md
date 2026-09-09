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
- Next: commit and push the reviewed source, fast-forward the separate remote
  worktree, and submit the authorized zero-GPU preflight after a fresh queue and
  node check.
