# 14-bit formats

## Target-specific formats

- FP32: E2M11, E5M8, E8M5.
- FP64: E2M11, E5M8, E11M2.

The common layouts expose precision-heavy and FP16-range tradeoffs.  FP32 ends
at FP32 exponent range; FP64 substitutes an E11 prefix layout.

## Decoder candidates

This is the complete-LUT boundary.  A 16384-entry target-word table is 64 KiB,
so global/L1 and opt-in shared-memory placement are both measured.  All layouts
also test direct target construction, sign+exponent prefix LUTs, and
subnormal-only LUTs.  The latter two are expected to preserve occupancy and
reduce cache traffic relative to the full table.

Shared full-LUT launches explicitly request their dynamic shared-memory size;
launch failure is treated as a benchmark failure rather than dropping the
strategy.

## Physical access candidates

- exact 14-bit dense scalar versus 16-bit padded scalar;
- dense and padded x2/x4/x8 thread packets;
- dense cooperative access: 16 values = 224 bits = seven 32-bit loads across
  8 consumers, two values per consumer.
