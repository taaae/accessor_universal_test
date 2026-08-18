# TODO — before the next GPU job

Collected while writing the IEEE/LNS summary.

Baseline for everything below:
`results/021_unified_strategy_performance/run_20260815T171642Z/unified_core/`.

## 0. Retracted: there is no strategy coverage gap

An earlier version of this file claimed `fixed_integer` and `e1_integer` were
missing `thread_packet` variants below 9 bits, and proposed implementing them.
That was wrong.  **Every one of those variants exists, compiles, and was
measured.**  They are absent from `full/timing_summary.csv` because of the
two-stage benchmark flow, not because of missing code:

- `scripts/run_bitwidth_strategy_benchmark.sh` runs a **screen** stage over the
  whole matrix, ranks variants, then runs the **full** stage with
  `--variant-file screen/finalists.txt`.
- Finalists are the top `FINALISTS_PER_GROUP` (default **2**) per
  (format, arithmetic, kernel, layout, packet_values) group.
- `fixed_integer` at ≤8 bits consistently ranks 3rd in its `thread_packet`
  group behind `full_lut_shared` and `full_lut_global`, so it is dropped.

Example — `e0m3` fp32 DOT, `padded/thread_packet/x8`, from
`screen/strategy_ranking.csv`:

| strategy | screen ms | selected |
|---|---:|---|
| `padded/thread_packet/x8/full_lut_shared` | 0.018974 | 1 |
| `padded/thread_packet/x8/full_lut_global` | 0.021884 | 1 |
| `padded/thread_packet/x8/fixed_integer`   | 0.024390 | **0** |

**Consequence:** absence from the full summary means "lost the screen", never
"not implemented".  Any analysis that treats a missing decoder as a coverage
gap is wrong, and any `best`-scope comparison built only from the full summary
can silently compare a packed winner against an unpacked also-ran.

## 1. Re-derive the E0 story from the screen ranking

The `best`-scope E0 tables built from `full/timing_summary.csv` reported gaps
up to 4.44x for `fixed_integer`.  Those were artifacts — a packed LUT measured
against an unpacked integer decoder.  Rebuilt from `screen/strategy_ranking.csv`
(best variant per decoder, all access methods):

| bits | DOT→FP32 | GEMV→FP32 | DOT→FP64 | GEMV→FP64 |
|---:|---|---|---|---|
| 2–7  | LUT by 1.23–1.29x | LUT by 1.23–1.29x | LUT by 1.38–1.39x | LUT by 1.40–1.41x |
| 8    | LUT by 1.10x | LUT by 1.06x | LUT by 1.41x | LUT by 1.41x |
| 9+   | `fixed_integer` wins | `fixed_integer` wins | `fixed_integer` wins (16+) | `fixed_integer` wins (16+) |

So `fixed_integer` really does lose below 9 bits, but by **1.1–1.4x**, not
2.0–4.4x.  The crossover at 9 bits (fp32) / 16 bits (fp64) is real.

Caveat: screen numbers are a single median at one size and distribution.  The
full stage aggregates a geometric mean over both distributions at the two
largest N.  Treat the screen figures as indicative, not equal quality.

**Action:** either (a) teach the report generators to fall back to the screen
ranking when a decoder is absent from the full summary, or (b) re-run the full
stage for E0/E1 with `FINALISTS_PER_GROUP` raised so the aggregate-quality
numbers exist.  (b) is a small job — it is a re-run of the full stage only,
with no new code.

## 2. Lookup tables above 14 bits

`src/bitwidth_strategy_bench.cu:758` gates both table decoders explicitly:

```cpp
if constexpr (Format::total_bits <= 14) {
  run_access_family<Format, Compute, bw::decoder_kind::full_lut_global>(runner);
  run_access_family<Format, Compute, bw::decoder_kind::full_lut_shared>(runner);
}
```

so 16-bit and wider formats have no LUT of any kind in this harness — global
included.  The table is `2^total_bits` entries at 4 bytes (only the high word
of the double is stored; the low word is provably zero for any format narrow
enough to tabulate):

| bits | entries | table | shared | global |
|---:|---:|---:|---|---|
| 12 | 4 096 | 16 KB | yes | yes |
| 14 | 16 384 | 64 KB | yes (needs the >48 KB opt-in) | yes |
| 16 | 65 536 | 256 KB | **no** | yes, L2-resident |
| 24 | 16 777 216 | 64 MB | no | exceeds L2 |
| 32 | 4 294 967 296 | 16 GB | no | no |

How a too-large shared table actually fails: the kernels use **dynamic** shared
memory (`extern __shared__`), so nothing about the size reaches `ptxas` and it
compiles cleanly.  The abort happens at
`cudaFuncSetAttribute(..., cudaFuncAttributeMaxDynamicSharedMemorySize, bytes)`,
which returns `cudaErrorInvalidValue` above the device ceiling and is wrapped in
`CUDA_CHECK`.  So it is a hard launch-time failure, not a slowdown.  (A *static*
`__shared__ uint32_t table[65536]` would be a compile error instead — that is
just not the shape this code uses.)

Worth building:

- **`full_lut_global` at 16 bits** (E0M15, both computes).  256 KB is
  L2-resident on H200; this is a genuine open question.
- **Skip 24 and 32 bits.**  64 MB exceeds L2; 16 GB is not a tuning question.
- **Optional sign-magnitude LUT**: index by magnitude, apply the sign after.
  That is `2^(bits-1)` entries, so 16 bits becomes 128 KB and fits under the
  ceiling at 1 block/SM.  Only worth it if the 16-bit global result looks
  competitive.

Do not hardcode the ceiling.  Read `cudaDevAttrMaxSharedMemoryPerBlockOptin`
at startup and gate the strategy on it, so the build adapts instead of aborting.

## 3. Run

Follow `CLUSTER_RULES.md`: check `sinfo`/`squeue` immediately before submitting,
finite `--time`, ask for approval before `sbatch`, check the job one minute
after submit.
