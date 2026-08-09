# Implemented decoder strategy inventory

Every 2-, 4-, 8-, 16-, and 32-bit storage format now has a format-specific decision
note, exact portable decoder core coverage, registered CUDA packet strategies,
and cumulative decoder/DOT/GEMV smoke instantiations. These are candidate
implementations, not performance winners; experiment 015 measures the complete
kernel scaling needed to select the winners.

| Bits | Format | Decision note |
|---:|---|---|
| 2 | E0M1, E1M0 | [dense storage and endpoint strategies](expanded_formats.md#2-bit-formats) |
| 4 | E0M3, E1M2, FP4 E2M1, E3M0 | [dense storage, LUT, and native FP4 strategies](expanded_formats.md#4-bit-formats) |
| 8 | E0M7, E6M1, E7M0 | [fixed, IEEE, and exponent-only additions](expanded_formats.md#added-8-bit-formats) |
| 8 | E1M6 | [fixed-point, direct words, FP32, LUT, pair, packet variants](e1m6.md) |
| 8 | E2M5 | [existing 42 families plus pair-L2/PRMT](e2m5.md) |
| 8 | E3M4 | [existing 42 families plus pair-L2/PRMT](e3m4.md) |
| 8 | FP8 E4M3 | [native scalar/vector/half2, direct words, LUT families](fp8_e4m3.md) |
| 8 | FP8 E5M2 | [native scalar/vector/half2, direct words, LUT families](fp8_e5m2.md) |
| 16 | E1M14 | [fixed-point, direct words, FP32, L2/subnormal LUT](e1m14.md) |
| 16 | E2M13 | [direct words, FP32, prefix/subnormal/full LUT](e2m13.md) |
| 16 | E3M12 | [direct words, FP32, prefix/subnormal/full LUT](e3m12.md) |
| 16 | FP16 | [native half2, direct words, FP32, LUT controls](fp16_e5m10.md) |
| 16 | BF16 | [native bfloat162, raw FP32 lift, direct words, LUT controls](bf16_e8m7.md) |
| 16 | E11M4 | [direct FP64-prefix insertion](e11m4.md) |
| 16 | E0M15, E4M11, E6M9, E7M8, E9M6, E10M5 | [fixed, full/subnormal/prefix candidates](expanded_formats.md#added-16-bit-formats) |
| 32 | E1M30 | [fixed-point and direct two-word construction](e1m30.md) |
| 32 | E2M29 | [direct two-word and prefix strategies](e2m29.md) |
| 32 | E3M28 | [direct two-word and prefix strategies](e3m28.md) |
| 32 | FP32 | [native conversion, direct words, prefix helper](fp32_e8m23.md) |
| 32 | E11M20 | [direct FP64-high-word insertion](e11m20.md) |
| 32 | E0M31, E4M27, E5M26, E6M25, E7M24, E9M22, E10M21 | [fixed, two-word, and prefix candidates](expanded_formats.md#added-32-bit-formats) |

FP64 E11M52 remains the raw-load baseline and has no decoder strategy family.

Shared implementation files:

- [`decoder_strategy_core.hpp`](../../include/decoder_strategy_core.hpp): exact
  portable word construction and FP32/fixed-point helpers;
- [`format_decoder_strategies.cuh`](../../include/format_decoder_strategies.cuh):
  CUDA packet loads, native conversions, LUT placement, and fused DOT/GEMV;
- [`all_format_strategy_smoke.cu`](../../src/all_format_strategy_smoke.cu):
  cumulative exhaustive/sampled GPU validation.
- [`expanded_formats.md`](expanded_formats.md): endpoint semantics, dense
  sub-byte representation, per-format candidate rationale, and deliberate
  exclusions.
- [`all_format_strategy_bench.cu`](../../src/all_format_strategy_bench.cu):
  complete DOT/GEMV timing for the non-E2M5/E3M4 strategy families.
