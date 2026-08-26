# T16 DOT performance

The run directories contain raw per-sample timings, summaries, environment
metadata, and the run manifest for the five-format T16 DOT comparison.

## H200 result

The completed run used one NVIDIA H200 NVL, CUDA 13.1, `N = 2^27`, FP32
arithmetic, 10 warmups, and 30 randomized-order timing samples. Encoding and
packing were outside the timed kernel.

| Format | Strategy | Bits | Median (ms) | 5th--95th percentile (ms) | Speed vs raw FP32 |
|---|---|---:|---:|---:|---:|
| T16 | global LUT | 16 | 1.0112 | 1.0105--1.0117 | 0.514x |
| FP16 E5M10 | native scalar conversion | 16 | 0.4921 | 0.4918--0.4925 | 1.056x |
| E6M9 | branchy decoder | 16 | 0.8922 | 0.8917--0.8932 | 0.582x |
| E8M15 | dense direct shift | 24 | 0.5049 | 0.5045--0.5054 | 1.029x |
| Raw FP32 | direct pointer | 32 | 0.5197 | 0.5194--0.5202 | 1.000x |

The 256 KiB global T16 codebook does not behave like a cheap cached decoder
under this access distribution: T16 is 1.95x slower than raw FP32 and 2.05x
slower than native FP16. The direct-shift E8M15 control is slightly faster than
raw FP32 despite using 24-bit dense storage. The branchy E6M9 decoder also loses
most of its storage advantage to conversion work.
