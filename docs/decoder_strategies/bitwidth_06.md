# 6-bit formats

## Target-specific formats

FP32 and FP64 both test E0M5, E1M4, E2M3, E3M2, E4M1, and E5M0.  Full
coverage is worthwhile because six layouts are still cheap and E2M3/E3M2 are
the CUDA FP6 encodings.  FP32 conversion ends in `float`; FP64 conversion ends
in `double` and never reuses the FP32 direct-bit decoder.

## Decoder candidates

- generic x1 control;
- direct construction of final FP32 bits or FP64 high/low words;
- fixed E0, finite E1, and exponent-only M0 specializations;
- 64-entry final-target LUT (256 bytes) in read-only global/L1 or shared;
- for E2M3/E3M2, CUDA FP6 scalar conversion and CUDA FP6x2 conversion,
  with x4/x8 implemented as two/four independent x2 conversions.

CUDA FP6 conversion first produces half/half2.  FP32 consumes the resulting
float/float2; FP64 additionally widens those results.  CUDA documents native
FP6 benefit for CC 10.0a, so SM90 H200 executes the emulation path.  These are
candidates, not privileged baselines.

## Physical access candidates

- exact 6-bit dense scalar versus CUDA-compatible one-byte padded scalar;
- dense and padded x2/x4/x8 thread packets;
- dense cooperative access: 16 values = 96 bits = three 32-bit loads across
  four consumers, four decoded values per consumer.
