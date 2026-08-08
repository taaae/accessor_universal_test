# 009: E2/E3 decoder optimization

This focused experiment asks whether the slow custom 8-bit E2M5 and E3M4
formats can become useful by replacing four independent generic decodes with a
specialized x4 decoder or a 256-entry lookup table.

The timed strategies are:

- `fp64_e11m52/current_x1` and `current_x4`: width-matched FP64 baselines;
- `e2m5` and `e3m4` `current_x1`: the existing unpacked generic decoder;
- `current_x4`: the existing 32-bit packed load followed by four generic scalar
  decodes;
- `branchless_x4`: one 32-bit load followed by four exact FP32 decodes using an
  integer significand, a power-of-two scale, and a masked non-finite result;
- `lut_x1` and `lut_x4`: scalar or 32-bit packed source loads followed by a
  256-entry FP32 lookup and exact FP32-to-FP64 conversion.

Each format's lookup table is 1 KiB. It is read through CUDA's read-only/L1
path. This avoids divergent constant-memory serialization and avoids adding a
per-block shared-memory copy, synchronization, and bank-conflict behavior to
the first lookup experiment. Table construction and upload are excluded from
timing. The reported `source_bytes` count only the compressed array payload;
they deliberately do not pretend the four-byte cached table fetch for each
decoded value is HBM traffic. A later Nsight run can determine how much table
traffic reaches L2 or HBM if the lookup strategy wins on elapsed time.
`lookup_bytes_requested` records those logical table requests separately.

Before benchmarking, every strategy decodes all 256 byte patterns. Every finite
result must agree bit-for-bit with the generic FP64 decoder; infinity sign and
NaN classification must also agree.

The benchmark measures:

1. `register_decode`: decode throughput after the source packet is in registers
   (LUT variants still perform the table accesses that define their decoder);
2. `stream_load`: source bytes without conversion, at x1 and x4 widths;
3. `stream_decode`: source load, decode, and reduction;
4. complete two-launch DOT;
5. complete GEMV with M fixed at 1024.

Both U(0,1) and N(0,1) use the same deterministic Philox datasets for every
strategy. Variants are interleaved in rotating order, with 10 warmups and 15
recorded aggregate samples per case by default. Increasing N cannot amortize a
per-element decoder, so the sweep includes small, transition, and HBM-scale
sizes instead of extending beyond the established 2^27 DOT / 2^16 GEMV limits.

## Run on H200

From the repository root, choose the node in the submission command:

```bash
sbatch --wait --nodelist=gpu-nvidia-h200-1-studvm-2 \
  --export=ALL,TMPDIR=/tmp \
  scripts/run_decoder_optimization_h200.sbatch
```

The job creates a timestamped directory here containing raw samples, exactness
validation, median summaries, width-matched speedups, plateau rows, compiler
resource usage, and SASS.
