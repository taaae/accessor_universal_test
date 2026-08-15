# 28-bit formats

## Target-specific formats

- FP32: E4M23, E5M22, E8M19.
- FP64: E2M25, E5M22, E11M16.

FP32 concentrates on layouts close to its own precision/range envelope.  FP64
retains a precision-heavy E2 layout, the shared E5 layout, and an E11 prefix.

## Decoder candidates

Direct target construction and sign+exponent prefix LUTs are the only scalable
optimized candidates.  Mantissas are too wide for configured subnormal tables,
and a complete LUT would require 1 GiB even with one word per code.  Generic
decoding is retained only as an x1 correctness/control path.

FP64 direct/prefix decoders construct both high and low FP64 register words for
M>20; FP32 decoders construct final FP32 bits directly and do not route through
FP64.

## Physical access candidates

- exact 28-bit dense scalar versus 32-bit padded scalar;
- dense and padded x2/x4/x8 thread packets;
- dense cooperative access: 8 values = 224 bits = seven 32-bit loads across
  8 consumers, one value per consumer.

Only 12.5% source traffic is saved, making this the negative-control density
threshold for cooperative/shuffled access.
