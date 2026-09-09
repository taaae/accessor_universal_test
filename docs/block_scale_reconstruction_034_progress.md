# Block-scale reconstruction progress

- Stage: complete; measured artifacts rendered and validated locally.
- Branch: `codex/block-scale-dot`.
- Active contract: `docs/block_scale_reconstruction_034_plan.md`.
- Experiment 033 and the parent-owned untracked plotting script and four graphs
  remain untouched.
- Source/audit commit: `2c15934e478e315cb6d22236ee2aab21cefef5da`.
- Local gates passed: Release `-O3 -DNDEBUG` host test; 32 Python tests both
  normally and under `python -O`; Python compilation; shell syntax; synthetic
  3,500-row rendering; and `git diff --check`.
- Zero-GPU preflight `460480` failed before script startup because its Slurm
  output directory did not exist. It used no GPU. Preflight `460481` compiled
  successfully and preserved the first actual Hopper SASS, but its audit failed
  closed on register reuse and commuted multiply operands. Those compiler forms
  were inspected and the checker was hardened.
- Replacement zero-GPU preflight `460482` passed the host test and semantic
  audit for all 16 timed kernels with zero spills. Binary SHA-256 is
  `ae9e60226560cdf4e1b6d56d66cbc165d2b21579b1f669f3b57bbf296626c9fb`;
  SASS SHA-256 is
  `e01239c142bf3e5b59cb2b529e46ac2b5e57ad31de6b20a660908da8e0e51d32`;
  audit JSON SHA-256 is
  `0a6d508208722d651b1e12cb42ce9534ff3d1011c24e47ce29f37314a9c84dc8`.
  Evidence: `results/034_block_scale_reconstruction/preflight_460482/`.
- Independent review was bound to the identical SASS hash. It confirmed the
  16-symbol inventory, ABI roles, exact reconstruction chains, the four-round
  deferred loop and predicates, all-lane scaling, baselines, reducers, and
  forbidden-operation/resource gates. It found no material blocker.
- H200 smoke job `460483` ran on `gpu-nvidia-h200-3`. All 84 timing rows and 238
  correctness cases passed, the FP32 and FP64 reconstruction contracts were
  distinct, and Compute Sanitizer memcheck and synccheck each reported zero
  errors. Run: `run_20260909T180022Z_460483`.
- Full H200 job `460484` ran on the same node and exact audited binary. All 3,500
  initial samples passed strict validation: 5 sizes times 14 variants times 50
  samples. Maximum observed baseline drift was 0.126389%, below the complete-N
  rerun threshold, so no extension was required. Run:
  `run_20260909T180220Z_460484`.
- The full run was collected and revalidated locally. The environment digest,
  commit, binary, SASS, and audit provenance match the manifest. Analysis
  produced exactly 70 official summaries and the required two PNG/SVG figures
  plus self-contained HTML report; both PNGs were inspected at full resolution.
- Final action: commit the scoped experiment-034 evidence, reports, and this
  record, push, and report the result paths.
