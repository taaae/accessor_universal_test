# Block-scale DOT progress

- Stage: initial independent-review fixes and local verification.
- Branch: `codex/block-scale-dot`.
- Contract commit: `9a6d603`.
- Cluster precheck on 2026-09-09 found an empty user queue. H200-2 had all
  eight GPUs allocated; H200-3 had seven of eight allocated. No job submitted.
- Release host tests and 13 Python validator/audit tests pass. Initial review
  confirmed the CUDA synchronization structure and found material assembly,
  large-validation, tolerance, drift and provenance gaps. These are addressed
  locally and await follow-up review. The actual compiled SASS remains a hard
  gate before GPU smoke.
- Follow-up review accepted the runtime, validation, drift, manifest and report
  fixes. It demonstrated four remaining false passes in the pre-SASS structural
  checker. The authorized zero-GPU preflight may collect the concrete Hopper
  assembly, but GPU smoke remains blocked until explicit saved dataflow and
  predicate review plus appropriate checker hardening pass.
- Next: commit and push, create the separate remote worktree, and run the
  authorized zero-GPU preflight to obtain actual machine code.
