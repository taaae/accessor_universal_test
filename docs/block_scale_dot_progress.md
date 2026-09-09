# Block-scale DOT progress

- Stage: complete; measured artifacts and locally rendered report verified.
- Branch: `codex/block-scale-dot`.
- Contract commit: `9a6d603`.
- Implementation/SASS-hardening commits: `2ddfbb3`, `316aa6b`.
- Final measured source commit: `99410d5765f9e7e34b3dc8570f44c2d70fcfdcb0`.
- Local verification: Release host test passed; all 17 Python audit/result tests
  passed; Python compilation, shell syntax and `git diff --check` passed.
- Independent review: the implementation review findings were fixed. A final
  read-only inspection of all 17 actual Hopper kernels found no blocker to GPU
  execution. One documented checker limitation remains non-blocking because the
  saved cubin was also inspected manually for the missing deferred dataflow and
  predicate properties.

## Cluster execution

- Preflight `460435` at `2ddfbb3` compiled all specializations with no spills,
  but intentionally failed the first audit because Hopper emits two shuffles
  per FP64 value. The evidence is preserved for audit history.
- Job `460436` was canceled before execution after a remote fast-forward
  conflict left the old commit selected. No GPU was requested or used. The
  conflicting remote evidence was preserved separately rather than overwritten.
- Preflight `460437` at `316aa6b` passed the host test and 17-kernel SASS audit.
- Smoke `460442` was rejected before sanitizers: its CSV used the exact FQDN
  reported by the binary while the manifest used the short hostname. Its
  incomplete output is preserved and is not used as result input.
- Wrapper fix commit `99410d5` records the exact hostname. Replacement preflight
  `460443` passed. Binary SHA-256 is
  `ec4d92bccca4c2ebb8197b6daf4d59d89ca35fbbe6d9bdf700cf828079d3533b`;
  SASS SHA-256 is
  `2127db92b13fe61e799dffdb8388c4aaa626b3a0fb4b624ea5d47038a6e1dc91`;
  audit SHA-256 is
  `af181c9ec0eaefeaf3d4a52470e912119d604e7248dc654b07b404de31bcc337`.
- Replacement smoke `460444` ran on `gpu-nvidia-h200-3.int.coma-cluster.de`
  (NVIDIA H200 NVL). It passed the strict 90-row inventory, all 240 independent
  correctness cases, memcheck and synccheck with zero errors.
- Full job `460445` ran on the same node. It passed the strict 3,750-row initial
  inventory. The complete N=2^20 block was rerun after baseline drift, adding
  750 rows; the final 4,500-row/90-group file passed strict validation. No cases
  were pruned.

## Final artifacts

- Exact run: `results/033_block_scale_dot/run_20260909T163444Z_460445/`.
- Raw samples: `full/timing_samples.csv`; manifest: `manifest.json`;
  correctness: `full/correctness_checks.txt`.
- Report: `report/report.html`; standalone screenshot page:
  `report/screenshot.html`; all-15 plot: `report/dot_block_scale_all.png` and
  `.svg`; official 75-case summary: `report/timing_summary.csv`.
- Local analyzer revalidated all 4,500 rows, selected the complete drift rerun
  for N=2^20, produced exactly 75 official summaries, and verified the exact
  manifest/audit/correctness bindings. The PNG was inspected at full resolution;
  all 15 direct labels and dotted connectors are present and legible.

- Evidence/report commit: `19e6d07`; pushed to `origin/codex/block-scale-dot`.
- Next: none; experiment 033 is complete.
