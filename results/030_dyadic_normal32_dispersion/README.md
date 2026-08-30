# DyadicNormal32 segment-dispersion benchmark

This experiment measures a single scalar x1 DOT kernel at `N=2^26` with
FP64 arithmetic and reduction. It compares 32-bit DyadicNormal32 storage with
raw FP64 in the same executable and GPU allocation.

DyadicNormal32 uses one sign bit and a 31-bit magnitude rank. The number of
leading one bits in the magnitude rank selects segment `h`. A zero delimiter
is followed by the segment payload `l`. Segments `h=0..30` reconstruct with

```text
fma(double(l), B[h], A[h])
```

using midpoint-spaced values between dyadic half-normal-CDF boundaries for an
`N(0,3)` code-point density. The all-ones magnitude rank is the terminal
`h=31,l=0` code and maps to the finite `2^-32` tail boundary. The 32 aligned
`{A,B}` FP64 pairs occupy 512 bytes and are staged into shared memory once per
timed first-stage block.

The experiment sweeps

```text
X = (mean unique h indices per warp - 1)
    / (expected unique h indices under uniform 32-way draws - 1)
```

at `0, 0.125, ..., 1`. A deterministic hot-segment-plus-uniform mixture is
solved separately for each target. Payloads and signs are independently
randomized, and the report plots measured X. Because X alone does not capture
shared-memory bank conflicts, the experiment also directly times the segment
probabilities induced by genuine `N(0,1)` values. It marks that point with a
diamond rather than inferring its performance from the artificial mixture.

Full runs use 512 first-stage blocks, 256 threads per block, 10 warmups, and 50
timing samples. Half the samples are collected in ascending-X order and half in
descending-X order, with five warmups before each half. The two genuine-source
halves sit at the center of that sequence. Raw FP64 samples and warmups are
split before and after the sweep.
