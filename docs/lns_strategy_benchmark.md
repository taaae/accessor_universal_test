# LNS conversion and fused-multiply benchmark

This experiment compares Universal-style base-2 logarithmic storage with
linear FP32 and FP64 accumulation in DOT and GEMV.  `LNS<B,R>` uses one value
sign bit and a signed `(B-1)`-bit fixed-point logarithm with `R` fractional
bits.  The most-negative logarithm code is reserved for zero/NaN, matching
Universal's `lns` special encoding.

## Independent benchmark axes

- arithmetic: ordinary decode/decode/multiply or fused log-add/decode;
- physical layout: dense bitstream or padded byte/word;
- access: scalar, per-thread x2/x4/x8 packet, or cooperative shuffle;
- decoder: accurate exp2, approximate hardware ex2, full LUT, fractional LUT,
  warp-register LUT, split LUT plus polynomial, or small pair-product LUT.

Fused multiplication adds the two fixed-point log codes in a widened signed
integer.  It never rounds or clamps the product back to the storage format.

## Format inventory

| Target | Format | IEEE-like allocation | Role | Access plan |
|---|---|---|---|---|
| FP32+FP64 | LNS<4,1> | E2M1 | FP4-scale core | dense/padded x1/x2/x4/x8; 8-value shuffle |
| FP32+FP64 | LNS<6,2> | E3M2 | FP6-scale core | dense/padded x1/x2/x4/x8; 16-value shuffle |
| FP32+FP64 | LNS<8,2> | E5M2 | FP8 range match | natural x1/x2/x4/x8 |
| FP32+FP64 | LNS<8,3> | E4M3 | FP8 balance match | natural x1/x2/x4/x8 |
| FP32+FP64 | LNS<8,4> | E3M4 | custom IEEE match | natural x1/x2/x4/x8 |
| FP32+FP64 | LNS<8,5> | E2M5 | rescue screen | natural x1/x2/x4/x8 |
| FP32+FP64 | LNS<10,4> | E5M4 | dense 10-bit core | dense/padded x1/x2/x4/x8; 16-value shuffle |
| FP32+FP64 | LNS<12,6> | E5M6 | dense 12-bit core | dense/padded x1/x2/x4/x8; 8-value shuffle |
| FP32+FP64 | LNS<16,4> | E11M4 | wide-range/coarse | natural x1/x2/x4/x8 |
| FP32+FP64 | LNS<16,7> | E8M7 | BF16 allocation match | natural x1/x2/x4/x8 |
| FP32+FP64 | LNS<16,10> | E5M10 | FP16 allocation match | natural x1/x2/x4/x8 |
| FP32+FP64 | LNS<16,11> | E4M11 | rescue screen | natural x1/x2/x4/x8 |
| FP32+FP64 | LNS<16,12> | E3M12 | primary rescue screen | natural x1/x2/x4/x8 |
| FP64 | LNS<16,13> | E2M13 | limited-range diagnostic | natural x1/x2/x4/x8 |
| FP64 | LNS<32,20> | E11M20 | broad-range dominator | natural x1/x2/x4; x8 screen |
| FP64 | LNS<32,23> | E8M23 | FP32-like LNS | natural x1/x2/x4; x8 screen |
| FP64 | LNS<32,28> | E3M28 | awkward-IEEE stress test | natural x1/x2/x4; x8 screen |

Full code LUTs stop at 12 bits.  Fraction LUTs stop at 13 fractional
bits.  Larger formats use an 8-bit coarse LUT with linear, quadratic, or cubic
residual reconstruction.  `ex2.approx.f32` is always labelled approximate;
FP64 uses an FP32 approximation followed by promotion for this strategy.

## Staging

`--mode smoke` uses small DOT/GEMV cases.  `--mode screen` uses representative
large cases for every compiled variant.  Full N sweeps should be run only for
selected finalists, while retaining an x1 ordinary and x1 fused reference for
every format.
