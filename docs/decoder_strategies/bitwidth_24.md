# 24-bit formats

## Target-specific formats

- FP32: E0M23, E5M18, E8M15.
- FP64: E2M21, E5M18, E11M12.

E0M23 is the exact FP32 mantissa-width fixed endpoint.  FP64 substitutes a
precision-heavy IEEE layout and terminates at an E11 prefix.  E5M18 is shared
to anchor FP16-range comparisons.

## Decoder candidates

All layouts test direct target construction.  Floating layouts test compact
sign+exponent prefix tables; only E11M12 keeps a subnormal-only table at the
configured M<=12 limit.  E0M23 tests fixed-integer scaling.  Complete LUTs are
excluded (64 MiB of target words), and the generic codec is x1-only.

## Physical access candidates

- exact 24-bit dense scalar versus 32-bit padded scalar;
- dense and padded x2/x4/x8 thread packets;
- dense cooperative access: 4 values = 96 bits = three 32-bit loads across
  four consumers, one value per consumer.

The dense layout saves 25% of source traffic.  Because each consumer receives
one value, any win must repay extraction and shuffles directly from bandwidth.
