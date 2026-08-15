# 5-bit formats

## Target-specific formats

- FP32: E0M4, E2M2, E4M0.
- FP64: E0M4, E1M3, E2M2, E3M1, E4M0.

FP32 samples the precision endpoint, balanced middle, and range endpoint.  The
two intermediate layouts are retained for FP64 because all five possible
layouts are cheap to cover and remain meaningful below FP64 range.

## Decoder candidates

Every layout tests direct target-bit construction and a 32-entry final-value
LUT from read-only global/L1 and shared memory.  E0 uses the fixed-integer
specialization, E1 uses the integer-scale specialization, and M0 uses the
exponent-only specialization.  The generic codec is an x1 control only.  A
prefix or subnormal-only LUT cannot improve on the 128-byte complete table.

## Physical access candidates

- exact 5-bit dense scalar versus one-byte padded scalar;
- dense and padded x2/x4/x8 thread packets;
- dense cooperative access: 32 values = 160 bits = five 32-bit loads,
  redistributed across 8 consumers, four values per consumer.
