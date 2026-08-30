# LUT decomposition shapes

This experiment measures five scalar x1 DOT cases at `N=2^26` on one H200:

1. one 16-bit global FP64 LUT, with raw FP64 as the baseline;
2. one 8-bit shared FP32 LUT, with raw FP32 as the baseline;
3. one 8-bit shared FP64 LUT, with raw FP64 as the baseline;
4. two 8-bit shared FP32 LUTs versus four 4-bit shared FP32 LUTs;
5. two 16-bit global FP64 LUTs versus four 8-bit or eight 4-bit shared FP64 LUTs.

For split formats, the kernel looks up every component and adds the decoded
component values before multiplication. Every component field is generated
independently at the same target normalized lookup dispersion.

For a `K`-entry component table, the x-axis is

```text
X = (mean unique indices per warp and lookup instruction - 1)
    / (expected unique indices under a uniform K-way lookup - 1)
```

The generator mixes a fixed hot index with a uniform index. It solves the
mixture probability separately for 4-, 8-, and 16-bit component tables so the
expected `X` is one of `0, 0.125, ..., 1`. The report plots measured `X`.

Shared tables are cooperatively staged once per block inside the timed first
stage, followed by one uniform synchronization. Global diagnostics record
unique 32-byte LUT sectors per warp. Shared diagnostics record an estimated
maximum number of distinct addresses served by any bank per warp lookup.

Full runs use 10 warmups and 50 samples. The raw FP32 baseline comes from the
immutable experiment 027 follow-up on the same student H200 and exact DOT
geometry. Raw FP64 is measured in this run and reused across the three FP64
graphs.
