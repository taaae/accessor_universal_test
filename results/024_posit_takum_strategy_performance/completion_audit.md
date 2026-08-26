# Completion audit

This audit checks the finished experiment against `docs/posit_takum_strategy_benchmark.md` and the approved cluster workflow.

| Requirement | Evidence | Status |
|---|---|---|
| Test posit, linear takum, and logarithmic takum at 8, 14, 16, and 32 bits | The full manifest lists all 12 alternative targets. `timing_samples.csv` contains every requested family, width, and FP32/FP64 arithmetic combination. | Pass |
| Use the fixed strategy matrix | The result matrix contains direct, shared full LUT, and global full LUT at 8 and 14 bits; direct and global full LUT at 16 bits; direct at 32 bits. | Pass |
| Retain the predeclared IEEE subset and scalar strategies | The full manifest lists all 28 IEEE target and arithmetic combinations from the specification. The exact validator found the complete expected matrix. | Pass |
| Use scalar x1 access only | Every timing row has `access_method=scalar` and `packet_values=1`. No x2, x4, x8, padding, shuffle, or loader-consumer result exists. | Pass |
| Use dense 8, 14, 16, and 32-bit storage | Every timing row has `storage_layout=dense`. Fourteen-bit inputs use the packed 14-bit stream and report physical packed bytes. | Pass |
| Keep the two main distributions separate | Every main case exists for `field_balanced_finite` and `paired_log_uniform_finite`. The analysis key retains the distribution. | Pass |
| Balance conversion fields and validate realized inputs | `histograms.csv` has 23,668 rows. The runner validates sign, regime, characteristic, exponent, subnormal, interval, and pair-admissible support rules before timing. Takum direction and regime marginals are cross-checked through all 16 direction-regime cells. | Pass |
| Run the paired LUT controls | Both scattered and concentrated traces exist at 8 and 14 bits for FP32 and FP64, shared and global placement, DOT and GEMV. The same table-only kernel and raw index stream are reused within each control. | Pass |
| Use only the largest DOT and fixed GEMV case | Full rows use DOT `N=134217728` and GEMV `M=1024`, `N=65536`. There is no smaller-N pruning or strategy screening stage. | Pass |
| Time kernel work only | The CUDA event region includes load, scalar unpack, conversion, arithmetic, reduction, and shared-table staging. Allocation, generation, LUT construction, copies, and validation occur outside the timed region. | Pass |
| Use 10 warm-ups and 30 interleaved samples | The run manifest records 10 warm-ups and 30 samples. Each of the 604 timed cases has 30 successful timing rows, for 18,120 total rows. | Pass |
| Validate decoders before timing | All 119 decoder-validation rows pass. Coverage is exhaustive through 16 bits and uses deterministic wide 32-bit samples and boundaries. Logarithmic takum is checked directly against the independently parsed paper formula with the specified ULP limits. | Pass |
| Require finite kernel outputs and complete source checks | Every strategy completed validation and timing. The exact validator reports zero infeasible cases, and each benchmark rejects nonfinite DOT or GEMV results. | Pass |
| Preserve raw samples, medians, intervals, validation, histograms, sizes, flags, device, launch data, and seeds | The full result directory contains raw and per-target timing, decoder checks, histograms, coverage, CTest output, standard output, manifest, and environment records. The analysis directory contains medians, winners, confidence intervals, and explicit infeasible rows. | Pass |
| Keep distributions, kernels, and arithmetic separate in analysis | `strategy_winners.csv` keys every result by format, arithmetic, distribution, and kernel. No total score or geometric mean is computed. | Pass |
| Use paired and independent bootstrap intervals correctly | Questions 1 to 3 use paired-round bootstrap intervals. Question 4 uses independent bootstrap intervals across format executables. All equivalence labels require the full interval inside `[0.97,1.03]`. | Pass |
| Answer all four questions case by case | `summary.md` includes complete case tables for every question, followed by measured answers. | Pass |
| Make no unsupported accuracy claim | The report states that the experiment measures storage conversion and ordinary FP32/FP64 arithmetic. It does not infer an application-independent accuracy advantage or accuracy-performance trade-off. | Pass |
| Obtain an independent context-free CUDA review | `code_review.md` records the separate review, fixes, repeated focused checks, and the final no-finding result for barriers, divergence, indexing, packing, shared memory, validation, coverage, and one-GPU compliance. | Pass |
| Pass the CPU-only all-target gate before GPU work | Slurm job 452825 compiled all 44 targets. `slurm-build-452825.out` records clean completion. | Pass |
| Run a one-GPU smoke test | Slurm job 452826 requested one `gpu:nvidia` resource and completed the 604-case smoke matrix. The smoke validator reports 1,208 timing rows, 119 decoder rows, 23,489 histogram rows, and zero infeasible cases. | Pass |
| Run the full job on one non-student H200 | Slurm job 452827 requested one `gpu:nvidia` resource and ran on one NVIDIA H200 NVL. `environment.txt` records one GPU UUID. The job finished cleanly in 421 seconds. | Pass |
| Enforce per-target timeouts | The full manifest records `target_timeout_seconds=5400`; the runner wraps each benchmark target separately. | Pass |
| Check jobs and availability before resource use | The user queue and H200 node state were checked before each approved submission. The experiment never overlapped another GPU job and submitted no further GPU work after completion. | Pass |
| Push focused source, pull fast-forward on cluster, and return results | Source changes were committed and pushed before cluster pulls. Generated smoke and full results returned in result commit `37882fb`, which is on local and remote `main`. Unrelated dirty report and accuracy-screen files were not staged. | Pass |

## Final run records

| Stage | Job | Result |
|---|---:|---|
| CPU-only all-target build | 452825 | 44 targets built |
| One-GPU smoke | 452826 | Complete, exact coverage validator passed |
| One-GPU full | 452827 | Complete in 421 seconds, exact coverage validator passed |

The full validator result is 604 timed cases, 18,120 timing rows, 119 decoder rows, 23,668 histogram rows, and zero infeasible cases.
