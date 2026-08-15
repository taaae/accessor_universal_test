# 9-bit formats

## Target-specific formats

- FP32: E0M8, E4M4, E8M0.
- FP64: E2M6, E5M3, E8M0.

FP32 samples fixed precision, a balanced layout, and exact FP32 exponent range.
FP64 instead keeps IEEE precision-heavy and FP16-range representatives before
the shared E8 endpoint.

## Decoder candidates

All layouts test direct target construction and the 512-entry (2 KiB)
final-target LUT in global/L1 and shared memory.  Non-endpoint formats also
test:

- sign+exponent prefix LUT followed by direct fraction placement;
- subnormal-only LUT followed by direct normal/special decoding.

E0M8 uses fixed-integer scaling and E8M0 uses exponent-only decoding.  Prefix
and subnormal tables become interesting here because they can be materially
smaller than the complete table.

## Physical access candidates

- exact 9-bit dense scalar versus 16-bit padded scalar;
- dense and padded x2/x4/x8 thread packets;
- dense cooperative access: 32 values = 288 bits = nine 32-bit loads across
  16 consumers, two values per consumer.
