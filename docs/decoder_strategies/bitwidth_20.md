# 20-bit formats

## Target-specific formats

- FP32: E2M17, E5M14, E8M11.
- FP64: E2M17, E5M14, E11M8.

The common layouts preserve precision or FP16 range.  FP32 uses an E8 member;
FP64 uses an E11 prefix member.

## Decoder candidates

Direct target construction and sign+exponent prefix LUTs apply to every
layout.  Subnormal-only LUTs remain small enough only for E8M11 and E11M8;
E2M17 and E5M14 use direct `clz` normalization.  Complete LUTs are excluded
(4 MiB even with one 32-bit target word per code).  Generic decoding remains
an x1 control.

## Physical access candidates

- exact 20-bit dense scalar versus 32-bit padded scalar;
- dense and padded x2/x4/x8 thread packets;
- dense cooperative access: 8 values = 160 bits = five 32-bit loads across
  8 consumers, one value per consumer.

This cohort has a 37.5% dense-storage saving but no per-consumer coarsening in
the cooperative mapping, making it a clean redistribution-overhead test.
