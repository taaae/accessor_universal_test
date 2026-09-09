# Instruction cost sweep progress

- Stage: complete. The highest-precedence FP32 four-curve experiment ran on an
  H200 and the raw data, correctness evidence, audited SASS, summaries, plots,
  and standalone HTML report were collected locally.
- Branch: `codex/instruction-cost-sweep`.
- Experiment commit: `c6ed81f164e2b74ad42f6eec23f23b47d6a11821`.
- Approved curves: `add32f`, `mul32f`, `fma32f`, and `rot32`. Floating curves
  decode UInt32 to FP32, execute explicitly rounded FP32 operations, widen to
  FP64, and accumulate in FP64. Rotation precedes that same decoder. The shared
  X=0 anchor is therefore UInt32-to-FP32-to-FP64, not exact UInt32-to-FP64.
  Coefficient bit patterns were `a=0x44800333` and `m=0x3f800347`.
- Independent review was completed and its material findings were fixed before
  GPU timing. The follow-up review found no remaining CUDA correctness,
  compilation, smoke, or job-safety blocker. Its final report-only findings
  (assert-based failure and incomplete rerun/extension inventories) were fixed
  in `c6ed81f`.
- Local Release-mode gates pass: `instruction_cost_core_test`, six positive and
  negative assembly-audit tests, Python compilation, shell syntax, and
  `git diff --check`.
- Zero-GPU preflight job `460399` passed the host test and all 88 timed-kernel
  SASS audits (two kernels, four families, eleven K values), with exactly `2*K`
  dependent target operations in each hot loop and no spills. Evidence is in
  `results/032_instruction_cost_sweep/preflight_460399/`.
- Preflight binary SHA-256:
  `06352edc2abad2a832f2e257bd152bc1b2846d34ca17675296a01a4b21c25fbd`.
  SASS SHA-256:
  `5923dcd01523c128079109a330651637c50ce6bd9a812e6e1ce7d0e6ffb3a1aa`.
  Audit JSON SHA-256:
  `acea87ec75dff9fe24a6233d91d4c282b3f89d1983711562f5d2fd61bf7777d6`.
- One-GPU smoke job `460404` ran on `gpu-nvidia-h200-3`. Compute Sanitizer
  memcheck and synccheck both reported zero errors; exact decoder comparisons,
  ragged DOT/GEMV CPU-GPU comparisons for every family/K, and all baseline
  comparisons passed. It produced 216 timing rows over three rounds.
- One-GPU full job `460407` ran on `gpu-nvidia-h200-3` after a fresh empty-user-
  queue and node-capacity check. All correctness gates passed and it produced
  exactly 3,600 initial timing rows: 72 cases, 50 samples each. No curve was
  eligible for K=48/64 extension because every K=32 median was already at or
  above its stage-matched raw-FP64 median. Baseline drift was at most 0.25%, so
  no rerun was required. Raw timing CSV SHA-256:
  `bd8d8e85c00f9fc00cab412ec0c9e3e788ef1b8ed9f93b210df8a7203b4846c9`.
- The full job completed measurement but the cluster report step lacked
  Matplotlib. The fail-closed analyzer was run locally against the collected
  CSV, manifest, and audited-SASS JSON and passed: 3,600 rows and 72 exact
  groups. Report-only label spacing and X=0 decoder wording were corrected and
  the analyzer reran successfully.
- Final artifacts are under
  `results/032_instruction_cost_sweep/run_20260909T144906Z_460407/report/`:
  standalone `report.html`, separate DOT/GEMV PNG and SVG plots, a combined PNG,
  and `timing_summary.csv`.
