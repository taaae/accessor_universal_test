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
