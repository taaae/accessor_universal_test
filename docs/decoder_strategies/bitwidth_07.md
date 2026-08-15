# 7-bit formats

## Target-specific formats

- FP32: E0M6, E3M3, E5M1.
- FP64: E2M4, E3M3, E6M0.

FP32 retains a high-precision fixed endpoint and stops at E5 range.  FP64 uses
an IEEE precision-heavy endpoint, the shared balanced layout, and the E6 range
endpoint.  This makes the inventories answer different target questions rather
than mechanically sharing all layouts.

## Decoder candidates

Direct final-target construction and the generic x1 control apply to every
layout.  The complete 128-entry LUT occupies 512 bytes and is tested in
read-only global/L1 and shared memory.  E0M6 also tests fixed-integer scaling;
E6M0 tests exponent-only decoding.  Prefix and subnormal-only tables are larger
mechanisms than the already tiny complete LUT and are omitted.

## Physical access candidates

- exact 7-bit dense scalar versus one-byte padded scalar;
- dense and padded x2/x4/x8 thread packets;
- dense cooperative access: 32 values = 224 bits = seven 32-bit loads across
  8 consumers, four values per consumer.
