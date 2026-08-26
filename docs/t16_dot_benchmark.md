# T16 DOT performance experiment

This experiment measures conversion and memory-access cost. It does not run an
accuracy study.

| Format | Storage | Timed conversion strategy |
|---|---:|---|
| T16 normal codebook | 16-bit code | Scalar load from a 65,536-entry global FP32 LUT |
| IEEE FP16 E5M10 | 16-bit | Scalar CUDA `__half2float` conversion |
| Custom IEEE E6M9 | 16-bit | Scalar direct branchy decoder |
| Custom IEEE E8M15 | Dense 24-bit | Scalar dense load and direct FP32 field shift |
| Raw FP32 | 32-bit | Scalar load without conversion |

The benchmark uses one DOT size, `N = 2^27`, and FP32 multiplication and
accumulation. Both vectors contain independent samples from a normal
distribution with `sigma = 16,376`, truncated at four standard deviations.
The upper endpoint is therefore 65,504, the maximum finite FP16 value.

The runner creates one source pair and encodes it into all five formats before
timing. Each timed kernel uses scalar x1 access and the same 512-block,
256-thread reduction. Format order is shuffled in every timing round. The CSV
records the execution order so clock drift can be checked afterwards.
