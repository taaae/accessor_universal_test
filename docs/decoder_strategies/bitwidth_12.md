# 12-bit formats

## Target-specific formats

- FP32: E0M11, E5M6, E8M3.
- FP64: E2M9, E5M6, E11M0.

FP32 spans fixed precision, FP16 range, and FP32 range.  FP64 spans a
precision-heavy IEEE layout, the shared FP16-range layout, and an exact
FP64-exponent prefix endpoint.  E11M0 is not sent through an FP32 decoder.

## Decoder candidates

The 4096-entry full LUT is 16 KiB and is tested in global/L1 and shared memory.
Layouts with mantissa bits also test sign+exponent prefix and subnormal-only
tables.  Direct FP32-bit and FP64-word construction remain mandatory baselines;
E0M11 adds fixed-integer scaling and E11M0 adds exponent-only/direct-prefix
construction.  Generic decoding is retained only for x1 validation/control.

## Physical access candidates

- exact 12-bit dense scalar versus 16-bit padded scalar;
- dense and padded x2/x4/x8 thread packets;
- dense cooperative access: 8 values = 96 bits = three 32-bit loads across
  four consumers, two values per consumer.
