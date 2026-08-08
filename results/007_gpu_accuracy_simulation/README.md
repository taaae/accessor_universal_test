# 007: GPU Monte Carlo accuracy simulation

This experiment validates the analytical predictions from experiment 006 with
actual H200 simulations of the generic DOT and GEMV kernels.

Inputs are generated on the GPU with deterministic cuRAND Philox streams for
independent `U(0,1)` and `N(0,1)` operands. There is no rescaling. Every one of
the 17 storage formats is tested with scalar, packed-2, and packed-4 loads.
Arithmetic is FP64.

## Default statistical design

- DOT: `N = 2^10, 2^14, 2^18, 2^22`, with 8,192 independent DOT products per
  distribution, format, and size. Samples are divided into 32 statistical
  batches.
- GEMV: `M = 1024` and `N = 2^8, 2^10, 2^12, 2^14, 2^16`, with 16 independent
  matrices/vectors per configuration. This produces 16,384 row outputs while
  treating each matrix/vector replicate as the independent statistical
  cluster.

The cluster-level standard error of every MSE is recorded. The post-run checker
marks comparisons above a 5% relative standard-error target for review, so
randomness is measured rather than assumed negligible.

## References and recorded data

For every output, the harness calculates:

- an FP64-source reference using a GPU double-double product and reduction;
- a decoded-storage reference using the same double-double algorithm;
- actual FP64 CUDA results for x1, x2, and x4 access.

The double-double reference is checked against a compensated host `long double`
calculation before the experiment starts. The run saves:

- `simulation_summary.csv`: sufficient moments, class counts, error metrics,
  and cluster standard errors;
- `empirical_quantiles.csv`: 17 quantiles for signed, absolute, normalized,
  relative, and condition-number distributions;
- `batch_estimates.csv`: independent batch/replicate estimates used for
  confidence assessment;
- `encoding_stats.csv`: zero, infinity, NaN, and saturation counts;
- `generation_seeds.csv`: exact Philox seeds for every generated group;
- `convergence_report.csv`: automatic Monte Carlo uncertainty assessment;
- environment, manifest, self-test, validation, and stdout logs.

These compact tables contain everything needed for the planned error graphs;
raw multi-gigabyte element arrays are deterministic and are not stored.

## H200 command

Submit from the repository root and select the node at submission time:

```bash
sbatch --wait --nodelist=gpu-nvidia-h200-1-studvm-2 \
  scripts/run_accuracy_simulation_h200.sbatch
```

The batch script itself does not select a GPU node. Defaults can be overridden
through exported Slurm environment variables such as `DOT_SAMPLES`,
`GEMV_REPLICATES`, or `WORKSPACE_GIB`.
