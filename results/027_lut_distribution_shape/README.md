# 16-bit LUT access-dispersion experiment

This experiment measures one instruction-identical FP32 DOT kernel with three
different 65,536-entry global FP32 tables:

- the T16 truncated-normal codebook;
- `posit<16,1>` decoded to FP32;
- `lns<16,11>` decoded to FP32.

The two inputs use natural `uint16_t` storage and scalar x1 loads. For every
dispersion point, all formats reuse the same left and right arrays byte for
byte. Only the table pointer changes. Posit NaR and LNS NaN table entries are
replaced with `0.0f`; numerical accuracy is deliberately outside this test.

For `q = 0, 1/8, ..., 1`, each code independently comes from one fixed
eight-entry LUT sector with probability `1-q`, or uniformly from all 65,536
codes with probability `q`. Inputs are independently generated without
sorting or repeated short pools.

The plotted X value is calculated from the generated arrays. For each operand
and warp, the benchmark counts unique 32-byte LUT sectors, averages over both
operands, and normalizes with

`X = (mean_unique - 1) / (U_uniform - 1)`,

where `U_uniform = 8192 * (1 - (1 - 1/8192)^32)`.

The full run uses `N=2^26`, 10 warmups, and 50 measured launches per format and
point. The generated run directory contains raw timing samples, access metrics,
a summary CSV, the single combined SVG graph, and an HTML wrapper.
