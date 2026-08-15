# 17-bit formats

## Target-specific formats

- FP32: E2M14, E5M11, E8M8.
- FP64: E2M14, E5M11, E11M5.

FP32 retains layouts through FP32 exponent range.  FP64 replaces the E8 member
with an E11 prefix while retaining the precision-heavy and FP16-range points.

## Decoder candidates

Complete LUTs stop at 14 bits: a 17-bit final-value table adds 512 KiB of
mostly random lookup traffic and has no shared-memory case worth testing.
Candidates are therefore:

- direct final FP32-bit or FP64-word construction;
- sign+exponent prefix LUT plus direct mantissa placement;
- subnormal-only LUT where M<=12 (E5M11, E8M8, and E11M5);
- generic x1 correctness/control.

E2M14 deliberately omits its 64 KiB subnormal table; `clz`-based direct
normalization is the more scalable candidate.

## Physical access candidates

- exact 17-bit dense scalar versus 32-bit padded scalar;
- dense and padded x2/x4/x8 thread packets;
- dense cooperative access: 32 values = 544 bits = seventeen 32-bit loads
  across a full warp, one value per consumer.
