# 4-bit padded-control cohort

## Formats and arithmetic

FP32 and FP64 both test E0M3, E1M2, finite E2M1 (the NVIDIA FP4 numeric
encoding), and E3M0.  FP32 and FP64 construct their own final target values.

## Decoder candidates

Every layout tests direct target construction and a 16-entry final-target LUT
in global/L1 and shared memory.  E0, E1, and M0 add their endpoint-specific
integer/exponent decoders.  The generic x1 codec remains a control.  The CUDA
FP4 conversion implementation is already covered by the native-format strategy
experiment; this cohort isolates physical dense-versus-padded access for the
same E2M1 bits.

## Physical access candidates

- exact dense 4-bit scalar versus one-byte-per-value padded scalar;
- dense and padded x2/x4/x8 thread packets;
- dense cooperative access: 8 values in one 32-bit word, four consumers, two
  values per consumer.

The padded path doubles source traffic and provides the direct test requested
for deciding whether compact FP4 storage pays for its extraction machinery.
