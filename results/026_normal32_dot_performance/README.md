# Normal32 DOT performance

This experiment compares PWLNormal32 16/16, PWQNormal32 8/24, and QN32 with
native and custom IEEE storage controls. Every timed path uses scalar x1 loads,
decodes to FP64, and runs the same FP64 DOT reduction at one fixed size,
`N = 2^27`.

Inputs are independent samples from `N(0, sigma^2)`, resampled outside
`[-4 sigma, 4 sigma]`, with `sigma = max-finite-FP32 / 4`. Encoding and dense
packing happen before timing.

Run directories contain raw timing samples, checked summaries, environment
metadata, and the full run manifest.

## H200 result

The completed run used one NVIDIA H200 NVL, CUDA 13.1, commit `51a416b`, 10
warmups, and 30 randomized-order timing samples. The primary comparison is
median kernel time; throughput counts only the two stored input arrays.

| Format | Strategy | Bits | Median (ms) | 5th--95th percentile (ms) | Speed vs raw FP64 |
|---|---|---:|---:|---:|---:|
| PWLNormal32 16/16 | global coefficient table | 32 | 1.9646 | 1.9634--1.9673 | 0.306x |
| PWQNormal32 8/24 | shared coefficient table | 32 | 0.7438 | 0.7428--0.7450 | 0.808x |
| QN32 | direct quadratic | 32 | 0.5099 | 0.5095--0.5103 | 1.179x |
| FP32 | native conversion | 32 | 0.5254 | 0.5251--0.5261 | 1.144x |
| E11M20 | direct shift | 32 | 0.4928 | 0.4925--0.4934 | 1.220x |
| E9M22 | global prefix table | 32 | 0.9468 | 0.9463--0.9481 | 0.635x |
| E9M22 | branchy word decoder | 32 | 0.9248 | 0.9241--0.9261 | 0.650x |
| E8M29 | dense global-prefix decoder | 38 | 1.0732 | 1.0727--1.0747 | 0.560x |
| E8M30 | dense global-prefix decoder | 39 | 1.0740 | 1.0732--1.0753 | 0.560x |
| E11M36 | dense direct shift | 48 | 0.6069 | 0.6064--0.6075 | 0.991x |
| Raw FP64 | direct pointer | 64 | 0.6012 | 0.6008--0.6018 | 1.000x |

QN32 is the only tested Normal32 decoder that beats both native FP32-to-FP64
and raw FP64 in this kernel. It is 3.5% slower than the fastest 32-bit IEEE
control, E11M20, so its usefulness depends on whether its previously derived
accuracy advantage is sufficient for an application. PWQ remains viable but
costs 23.7% versus raw FP64 and 51.0% versus E11M20. PWL's two effectively
random accesses into a 1 MiB coefficient table dominate its cost; it is 3.27x
slower than raw FP64.

The accuracy ordering was not remeasured here by design. This experiment
answers only the decode-plus-DOT performance question for the fixed input
distribution and arithmetic type.
