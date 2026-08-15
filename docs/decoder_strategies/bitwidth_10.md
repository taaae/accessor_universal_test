# 10-bit formats

## Target-specific formats

FP32 and FP64 both test E2M7, E5M4, and E8M1.  These layouts respectively
sample precision-heavy, FP16-range, and FP32-range storage.  Sharing the numeric
layouts is useful at this width, but FP32 builds final FP32 bits/tables while
FP64 independently builds final FP64 words/tables.

## Decoder candidates

- generic x1 control and direct target-bit construction;
- 1024-entry final-target LUT (4 KiB) in global/L1 and shared memory;
- sign+exponent prefix LUT plus direct mantissa placement;
- subnormal-only LUT plus direct normal/special decoding.

No endpoint specialization applies to these three layouts.  The full table is
still small enough to compete, while the hybrids test whether lower table
traffic wins once access becomes less cache-friendly.

## Physical access candidates

- exact 10-bit dense scalar versus 16-bit padded scalar;
- dense and padded x2/x4/x8 thread packets;
- dense cooperative access: 16 values = 160 bits = five 32-bit loads across
  8 consumers, two values per consumer.
