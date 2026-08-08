# 006: Analytical storage-accuracy model

This experiment predicts quantization-induced DOT and GEMV error before CPU
simulation or GPU measurement. It covers all 17 implemented storage formats
under independent `U(0,1)` and `N(0,1)` inputs without rescaling.

The model calculates scalar quantization moments, propagates them into exact
DOT/GEMV bias and MSE formulas, adds explicitly labeled distributional
approximations, and reports non-finite/saturation probabilities. GEMV fixes
`M=1024` and sweeps reduction length `N`.

The complete derivation and exact/approximate classification are in
[`docs/accuracy_model.md`](../../docs/accuracy_model.md).

Generate the deterministic CSV files, SVG figures, and HTML report from the
repository root:

```bash
./scripts/build_accuracy_model.sh
```

Custom size ranges can be selected without editing code:

```bash
./scripts/build_accuracy_model.sh \
  --dot-min-power 0 --dot-max-power 24 \
  --gemv-min-power 4 --gemv-max-power 16 \
  --gemv-m 1024
```

Outputs are written to `results/006_accuracy_model/generated/`. This stage is
CPU-only and does not contain measured kernel results.
