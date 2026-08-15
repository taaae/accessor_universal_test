# 3-bit formats

## Formats

Both arithmetic targets test all three possible layouts: E0M2, E1M1, and
E2M0.  At this width it is cheap to retain the fixed-point, finite E1, and
exponent-only endpoints as controls.

## Decoder candidates

| Layout | FP32 candidates | FP64 candidates |
|---|---|---|
| E0M2 | direct FP32 bits, fixed-integer scale, 8-entry final-FP32 LUT | direct FP64 words, fixed-integer scale, 8-entry FP64-high LUT |
| E1M1 | direct FP32 bits, E1 integer scale, 8-entry final-FP32 LUT | direct FP64 words, E1 integer scale, 8-entry FP64-high LUT |
| E2M0 | direct FP32 bits, exponent-only specialization, 8-entry final-FP32 LUT | direct FP64 words, exponent-only specialization, 8-entry FP64-high LUT |

The generic codec is retained only as a correctness/control path.  Full LUTs
are tested from read-only global/L1 and shared memory.  A prefix or subnormal
LUT is pointless here because the complete table is only 32 bytes.

## Physical access candidates

- dense scalar and padded-byte scalar;
- dense and padded-byte x2/x4/x8 thread packets;
- dense cooperative access: 32 values = 96 bits = three 32-bit loads,
  redistributed across 8 consumers, four decoded values per consumer.
