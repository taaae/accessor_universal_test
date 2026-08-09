# Decoder strategy research map

Status: preliminary design, 2026-08-09. This document decides what is worth
implementing and measuring; it does not claim winners before the H200 runs.
The machine-readable companion is
[`analysis/decoder_strategy_matrix.csv`](../analysis/decoder_strategy_matrix.csv).

## Bottom line

There is no single fair decoder for every format. The 16 storage formats fall
into five materially different groups:

1. **E1M6/E1M14/E1M30:** with the experiment's finite E1 semantics, these are
   signed fixed-point grids. Integer-to-FP64 conversion plus an exact power-of-two
   scale is a real candidate that does not apply to the other formats.
2. **E2/E3 custom formats:** values from U(0,1) and N(0,1) hit the source
   subnormal case frequently. A supposedly rare subnormal slow path is therefore
   not rare. Direct word construction and subnormal LUT hybrids matter.
3. **NVIDIA FP8, FP16, and BF16:** CUDA supplies scalar and/or packed conversion
   APIs. Those native routes must be tested, but "native type" does not imply
   that conversion to FP64 is a single fast instruction.
4. **E11M4 and E11M20:** these are literal prefixes of FP64. Decoding is just a
   shift or word insertion, with no numerical conversion and no exceptional
   slow path.
5. **32-bit E1/E2/E3 and FP32:** complete LUTs are impossible and FP32
   intermediates would lose information. The competition is between exact
   FP64-word construction, a few small prefix helpers, and native FP32-to-FP64.

The benchmark should find the fastest exact decoder independently for each
format and each kernel. A strategy can win DOT but lose GEMV because a shared
table is staged once per DOT block but once per GEMV row block.

## Required semantics and fairness rules

Every candidate must implement the same operation:

```text
stored source bits -> exact value represented by that format -> FP64 FMA
```

- No rescaling and no extra quantization are allowed.
- Intermediates may be FP32 only when every decoded source value is exactly
  representable in FP32, including special values under that format's policy.
- All candidates consume identical encoded arrays and use identical FP64
  accumulation and reduction code.
- Scalar/x2/x4/x8 describe load/decode width, not a different storage format.
- Packed and unpacked variants must agree bit-for-bit on decoded FP64 values.
- "Fastest format" means the best valid candidate for that format, not one
  generic decoder applied to all formats.
- Conversion-only results are diagnostic. Fused DOT/GEMV time selects winners.

The current generic E1/E2/E3 decoder is in
[`include/storage_formats.hpp`](../include/storage_formats.hpp), and the current
native CUDA routes are in
[`include/cuda_storage_formats.cuh`](../include/cuda_storage_formats.cuh).
The fused kernels currently support x1/x2/x4 through
[`include/storage_kernels.cuh`](../include/storage_kernels.cuh). E2M5/E3M4
already have the broader experimental strategy set in
[`include/e2e3_decoder_strategies.cuh`](../include/e2e3_decoder_strategies.cuh).

## Strategy vocabulary

These IDs are used below and in the CSV.

| ID | Exact operation |
|---|---|
| `REF` | Current generic codec, retained as the correctness/performance baseline. |
| `WORD-BR` | Construct FP64 high/low 32-bit words directly; branch for zero, source subnormal, and special. |
| `WORD-MASK` | Same exact construction with a branchless or predicated normal hot path. |
| `E1-CVT` | Treat E1 magnitude as an integer, convert it to FP64, then apply the exact binary scale and sign. |
| `FP32-BITS` | Construct exact FP32 bits, reinterpret as float, then execute FP32-to-FP64 conversion. |
| `PREFIX-WORD` | Insert stored E11 prefix directly into FP64's high word. |
| `NATIVE-DIRECT` | Ask the CUDA storage wrapper for `double` directly. Its SASS still has to be inspected. |
| `NATIVE-F32` | Native scalar storage-to-FP32 followed by FP32-to-FP64. |
| `NATIVE-F2/F4` | Native packed FP8 conversion to `float2`/`float4`, then component-wise FP64 conversion. |
| `NATIVE-H2` | Convert two FP8/FP16 values through `half2`, then `float2`, then FP64. |
| `NATIVE-B2` | Native packed BF16-to-`float2`, then component-wise FP64 conversion. |
| `L32` | Complete code-to-FP32 table, followed by FP32-to-FP64. |
| `L64` | Complete code-to-FP64 table. This doubles table traffic versus a high-word table. |
| `LUT-HI` / `LHW` | Complete table of FP64 high words; exact when source fraction bits are at most 20. |
| `LP` / `PREFIX-LUT` | Small sign+exponent table plus fraction bits shifted into place. |
| `SLUT` / `SUB` | Direct normal/special path plus a fraction-indexed table for source subnormals. |
| `PAIR-L2` | One lookup decodes a pair of 8-bit codes; the table is 512 KiB with two FP64 high words per entry. |
| `WARP-LUT` | Replicate a tiny table in registers across a warp and fetch with shuffles. Exploratory only. |
| `LUT-SWZ` | Duplicate or bank-swizzle a shared table so random lane indices produce fewer shared-memory conflicts. |
| `PACK-PRMT` | Extract 8-bit lanes with byte-permute instructions rather than independent shifts and masks. |

LUT suffixes are `-G` for cached global/read-only access and `-S` for a table
cooperatively staged into shared memory. `-S-DOT` means the staging cost is
credible for DOT but should not automatically be enabled for GEMV.

Priority levels:

- **P0:** required for a fair comparison.
- **P1:** credible challenger; implement after P0 correctness is established.
- **P2:** exploratory; retain only if a decoder microbenchmark or SASS gives a
  reason to run the full suite.

## Structural facts that control the search

### Source-subnormal frequency

For standard bias, the smallest normal magnitude is `2^(1-bias)`. The table
shows the probability of falling below it before rounding. The U column is for
U(0,1); the N column is for the magnitude of N(0,1).

| Exponent bits | Smallest normal | U fraction | N fraction |
|---:|---:|---:|---:|
| 1 | 2 | 100% | 95.4500% |
| 2 | 1 | 100% | 68.2689% |
| 3 | 0.25 | 25% | 19.7413% |
| 4 | 0.015625 | 1.5625% | 1.24664% |
| 5 | 0.0000610352 | 0.00610% | 0.00487% |
| 8 | about 1.18e-38 | negligible | negligible |
| 11 | about 2.23e-308 | negligible | negligible |

This is why a CLZ-based slow path is central for E1/E2 and secondary for
FP16/BF16. E1 is special: because the all-ones exponent is reclaimed, the
positive bit pattern is exactly an integer `K` times `2^(1-M)`.

### Exact LUT sizes

| Source width | Complete exact LUT | Practical placement | Conclusion |
|---:|---:|---|---|
| 8 bits | 1 KiB as FP64 high words, or 2 KiB as doubles | shared, L1/L2, constant experiment | Must test. |
| 16 bits | 256 KiB as FP64 high words | L2/global only | Worth testing as a cached control; too large for one Hopper block's shared memory. |
| 32 bits | 32 GiB because most values need both FP64 words | none | Do not implement. E11M20 would still be an unusable 16 GiB. |

All 8- and 16-bit formats have at most 20 fraction bits, so their exact FP64
low word is zero. The non-E11 32-bit formats have more than 20 fraction bits and
need both words.

Subnormal-only tables are smaller:

| Format | Exact subnormal table |
|---|---:|
| E1M6 / E2M5 / E3M4 | 256 B / 128 B / 64 B |
| FP8 E4M3 / E5M2 | 32 B / 16 B |
| E1M14 / E2M13 / E3M12 | 64 KiB / 32 KiB / 16 KiB |
| FP16 / BF16 | 4 KiB / 512 B |
| 32-bit formats | 64 MiB to 8 GiB | 

The full 16-bit table fits easily in Hopper's large L2, but an L2 hit is not
free and random lookups add a second memory stream. It is a useful experiment,
not the presumed winner. CUDA L2 access-policy windows are a P1 variant for
this table.

### Packet-load and table-layout variants

Packing does not reduce the compulsory source bytes: adjacent scalar loads are
already coalesced across a warp. It can still reduce load instructions, address
updates, loop overhead, and decode setup. Those effects must be separated from
the decoder family.

- For 8-bit x4, compare one 32-bit load followed by shifts/masks with
  byte-permute (`prmt`) extraction. For x8, compare two 32-bit loads with one
  aligned 64-bit load. x16 is a P2 screening point using four 32-bit loads and
  a bounded number of live decoded doubles.
- For 16-bit x2/x4/x8, use 32/64/128-bit aligned loads and inspect whether the
  compiler selects bit-field extraction. Hand-written `bfe` is tested only if
  generated SASS is clearly worse.
- For 32-bit x2/x4, use 64/128-bit loads. x8 is two x4 packets, not eight live
  FP64 values by default.
- Every shared 8-bit LUT gets plain, duplicated, and bank-swizzled layouts in
  decoder screening. FLUTE demonstrates that vectorization and table
  duplication can relieve shared-LUT bandwidth pressure, although its main
  target is 3/4-bit GEMM rather than our 8-bit FP64 DOT/GEMV.

## 8-bit formats

Every 8-bit format gets exhaustive validation over all 256 codes. Core load
widths are x1, x2, x4, and x8. x8 should use two aligned 32-bit loads initially;
a 64-bit load is a separate experiment because it can change instruction mix
and alignment requirements.

### E1M6

Difference: finite E1 semantics make the magnitude a fixed-point integer, and
almost the whole tested distribution uses the exponent-zero field.

- **P0:** `REF`; `E1-CVT`; exact `WORD-BR`; `LUT-HI-G`; `LUT-HI-S`, each at
  x1/x4/x8 where applicable.
- **P1:** `FP32-BITS`; shared subnormal LUT; `PAIR-L2`; compare x8 with four
  accumulators versus eight accumulators.
- **P2:** constant-memory full LUT and warp-register LUT.
- **Expected contenders:** shared high-word LUT x8 and fixed-point conversion
  x8. Which wins depends on LUT bank behavior versus integer-to-FP64 throughput.

### E2M5 and E3M4

These already have the broad 42-strategy study. Keep the existing generic,
branchless FP32, FP32/FP64/high-word/prefix LUT, direct-word, subnormal-LUT,
global/shared, and x1/x2/x4/x8 families as the reference implementation.

- **P0:** the already measured full family, especially `LP-S`, `LHW-S/G`,
  `L32-S/G`, direct branch/masked, and subnormal hybrids.
- **P1 new:** `PAIR-L2`, first as a decode microbenchmark and then fused only if
  competitive.
- **P2:** warp-register and constant full-LUT variants.
- **Current hypothesis:** `LP-S x8` remains the target to beat, but this must not
  be generalized to other formats without measurement.

### NVIDIA FP8 E4M3 and E5M2

Difference: CUDA exposes scalar, x2, and x4 storage wrappers. E4M3 also has a
different finite/special-value policy from the repository's generic IEEE-like
formats. Native results define the required bit semantics.

- **P0:** scalar CUDA conversion directly to double; scalar CUDA conversion via
  FP32; native x2-to-`float2`; native x4-to-`float4`; x8 composed from two x4
  operations; exact direct FP64-word construction; complete high-word LUT in
  global and shared memory. Test x1/x2/x4/x8.
- **P1:** x2 through `half2` and compositions of it; subnormal-only shared LUT;
  `PAIR-L2`; compare native vector decode with scalar decode after the same
  packed load.
- **P2:** warp-register and constant-memory LUT.
- **Expected difference between E4M3 and E5M2:** E5M2 has the same exponent
  width as FP16, while E4M3 has more source subnormals. Both fit exactly through
  FP16 and FP32, but the best native instruction sequence may differ.

CUDA's documented x4 conversion produces `float4`; there is no documented
x4-to-four-FP64 vector conversion. Consequently, native packing saves load and
unpack instructions but still leaves four FP32-to-FP64 conversions.

## 16-bit formats

Core load widths are x1/x2/x4/x8, corresponding to 16/32/64/128-bit loads.
A complete 256 KiB high-word LUT is tested only through cached global memory.
Shared-table candidates below are subnormal-only or prefix tables.

### E1M14

Difference: the same fixed-point identity as E1M6, but a 64 KiB subnormal LUT.

- **P0:** `REF`, `E1-CVT`, direct FP64-word construction, and exact
  `FP32-BITS`, at x1/x2/x4/x8.
- **P1:** subnormal high-word LUT from global memory; shared subnormal LUT for
  DOT only; complete high-word LUT in L2/global.
- **P2:** tiny exponent-prefix LUT. It saves too little arithmetic to be a
  presumed winner.
- **Expected contender:** `E1-CVT x8` or `FP32-BITS x8`; staging 64 KiB per GEMV
  row block is unlikely to pay.

### E2M13 and E3M12

Difference: FP32 can represent every finite decoded value exactly, while the
subnormal path occurs frequently for the chosen distributions.

- **P0:** `REF`, `WORD-BR`, `WORD-MASK`, `FP32-BITS`, subnormal LUT from global,
  and shared subnormal LUT for DOT, at x1/x2/x4/x8.
- **P1:** complete high-word LUT in L2/global and a small sign+exponent prefix
  LUT.
- **P2:** constant-memory prefix LUT. Random exponent addresses can serialize
  within a warp, so it requires evidence before full runs.
- **Expected difference:** E2M13's subnormal path is much hotter and its 32 KiB
  table is larger; E3M12 has a 16 KiB table and only 19.7% N(0,1) subnormals.

### Native FP16 E5M10

Difference: native `half2` conversion exists and source subnormals are almost
absent in these datasets.

- **P0:** scalar half-to-FP32-to-FP64; `half2`-to-`float2` at x2 and compositions
  at x4/x8; exact direct FP64-word construction; FP32-bit construction.
- **P1:** direct C++ half-to-double expression plus SASS inspection; 4 KiB
  subnormal LUT for DOT; complete high-word LUT in L2/global.
- **P2:** exponent-prefix LUT.
- **Expected contender:** native `half2` compositions. Direct word construction
  may still win because every FP32 intermediate needs `cvt.f64.f32`.

### Native BF16 E8M7

Difference: a normal BF16's bits are already the high 16 bits of FP32. The
FP32-bit path is one shift/reinterpretation before FP32-to-FP64, and source
subnormals are negligible here.

- **P0:** scalar BF16-to-FP32-to-FP64; packed BF16x2-to-`float2`; raw-bit lift
  into FP32 followed by FP64 conversion; direct FP64-word construction. Test
  x1/x2/x4/x8.
- **P1:** direct C++ BF16-to-double expression plus SASS inspection; full
  high-word LUT in L2/global; 512 B subnormal LUT as a control.
- **P2:** exponent-prefix LUT.
- **Expected contender:** raw-bit lift or native BF16x2, depending on whether
  packed native conversion saves enough work before `cvt.f64.f32`.

### E11M4

Difference: its 16 stored bits are exactly FP64's top 16 bits. The exact decoder
is conceptually:

```cpp
return __hiloint2double(int(uint32_t(code) << 16), 0);
```

- **P0:** `PREFIX-WORD` and `REF`, at x1/x2/x4/x8.
- **P1:** complete L2 LUT only as a negative/control result.
- **Do not test:** FP32 intermediate; E11's range cannot be represented exactly
  in FP32. No subnormal branch or special-value branch is needed.
- **Expected winner:** `PREFIX-WORD`, with the widest load that does not create
  excessive register pressure.

## 32-bit formats

Core widths are x1/x2/x4. x4 is already a 128-bit load. x8 is P1 and should be
implemented as two x4 packets while controlling accumulator count. No complete
or subnormal LUT is practical.

### E1M30

Difference: fixed-point identity still applies, but the 30-bit magnitude makes
an FP32 intermediate inexact.

- **P0:** `REF`, `E1-CVT`, and direct two-word FP64 construction at x1/x2/x4.
- **P1:** tiny sign+exponent prefix helper and x8 packet/unroll variants.
- **Do not test:** FP32 intermediate or any value-indexed LUT.
- **Expected contender:** direct word construction versus unsigned-integer to
  FP64 conversion plus exact scale.

### E2M29 and E3M28

Difference: decoded significands have more than FP32's 23 stored fraction bits,
and source subnormals remain common.

- **P0:** `REF`, `WORD-BR`, and `WORD-MASK` at x1/x2/x4. The direct path builds
  both FP64 words; subnormals use CLZ without a normalization loop.
- **P1:** tiny sign+exponent prefix helper and x8 packet/unroll variants.
- **Do not test:** FP32 intermediates, complete LUTs, or subnormal LUTs.
- **Expected difference:** E2M29 sends more values through CLZ than E3M28, so a
  predicated/masked path may rank differently between them.

### Native FP32 E8M23

Difference: CUDA has a genuine FP32-to-FP64 numerical conversion, and source
subnormals are negligible under the chosen distributions.

- **P0:** native scalar `cvt.f64.f32`, vectorized float2/float4 loads followed by
  scalar conversions, and exact direct FP64-word construction for comparison,
  at x1/x2/x4.
- **P1:** x8 packet/unroll variants.
- **P2:** sign+exponent prefix helper; it is unlikely to beat the very simple
  normal path but can expose conversion-throughput limits.
- **Expected contender:** native conversion on H200, where NVIDIA documents 16
  FP64-related conversion results per clock per SM, versus 64 32-bit shifts.

### E11M20

Difference: the stored 32-bit word is literally FP64's high word:

```cpp
return __hiloint2double(int(code), 0);
```

- **P0:** `PREFIX-WORD` and `REF`, at x1/x2/x4.
- **P1:** x8 packet/unroll variants.
- **Do not test:** FP32 intermediates and all LUTs.
- **Expected winner:** `PREFIX-WORD`; decoding should reduce to load plus bit
  reinterpretation.

## FP64 baseline

FP64 E11M52 is not decoded. Test raw aligned x1/x2/x4 loads using the same DOT
and GEMV geometry. It remains the performance and traffic baseline, not a
candidate strategy family.

## Kernel-specific choices

### DOT

- A block processes many packets for large N, so a small shared LUT prologue can
  be amortized.
- Wider packets create independent FP64 accumulators, which can hide dependency
  latency but consume two registers per live double.
- x8 must compare four versus eight accumulators; load width and unroll width
  should not be conflated.

### GEMV

- One block is launched per row. A shared table is therefore staged once per
  row, and the vector is reread by many row blocks.
- 16/32/64 KiB subnormal tables may lose to global/L2 lookup even if shared
  lookup latency is lower.
- The best decoder can vary with N because table setup is fixed per row while
  matrix traffic grows with N.

### Accessor design implication

Scalar direct, native, and global-LUT decoders fit a conventional
`accessor[index] -> double` API. Efficient packed loads require a packet method,
for example `load<8>(index)`. A shared LUT additionally requires a cooperative
block initialization/context step; that is a kernel policy or staged accessor,
not a self-contained scalar accessor operation.

## Benchmark program

### Stage 1: semantic validation

- Exhaust all 256 codes for every 8-bit format and all 65,536 codes for every
  16-bit format.
- For 32-bit formats, validate all boundary classes plus at least one million
  stratified/random codes against the host reference.
- Validate packed lane order and alignment separately for x2/x4/x8.
- Require exact FP64 bit agreement, including signed zero, infinity, and NaN
  class/payload policy where applicable.

### Stage 2: decoder screening

Measure both register-resident decode and load+decode. Record instruction mix,
registers/thread, occupancy limit, table traffic, and shared bank conflicts.
Keep P0. Promote P1 to fused kernels when it is within 15% of the best decoder
or has a kernel-level reason to improve after fusion.

### Stage 3: fused performance

Use the existing DOT and GEMV sizes and both U(0,1) and N(0,1), with identical
seeds and encoded arrays across strategies. Record all timing samples rather
than only medians. Compare:

1. time versus N;
2. speedup versus raw FP64;
3. packed width within one decoder;
4. decoder families within one format;
5. fastest exact candidate for every format.

### Stage 4: winner audit

Profile the fastest two strategies per format/kernel at the largest N. Inspect
SASS for FP32-to-FP64 conversions, integer shifts/CLZ, local-memory spills, load
width, and shared/L1/L2 behavior. A winner is accepted only after repeat runs
and a SASS/metrics explanation.

## Strategies deliberately excluded from the main search

- **Complete 32-bit LUTs:** 32 GiB and random accesses; structurally unusable.
- **Complete 16-bit shared LUTs:** 256 KiB exceeds Hopper's 227 KiB per-block
  limit before kernel scratch space.
- **FP32 intermediates for E11M4 or 32-bit custom formats:** not exact.
- **Pair LUTs above 8-bit codes:** table growth is prohibitive.
- **Constant memory for random complete LUTs:** a warp request is serialized for
  distinct addresses. Keep only tiny/coherent controls.
- **Texture fetch as a core family:** modern CUDA uses the unified L1/texture
  path and does not promise a general win over ordinary cached loads here.
- **Tensor Core kernels:** they would change FP64 arithmetic semantics, so they
  are a different experiment rather than a decoder optimization.
- **Standalone materialization of FP64 arrays:** it writes eight bytes per
  decoded value and measures output bandwidth, not fused DOT/GEMV behavior.

## Implementation order

1. Add a generic `decoder<Format, Strategy, Lanes>` interface and separate
   packet width from accumulator count.
2. Implement exact direct-word helpers shared by 8/16/32-bit formats.
3. Add E1 fixed-point, E11 prefix, and native CUDA families.
4. Generate rather than hand-write LUTs; upload once and validate every entry.
5. Add 8-bit full/pair LUTs and selected 16-bit subnormal/full-L2 tables.
6. Run validation and decoder screening before compiling the full Cartesian
   product of fused benchmarks.

This order first creates the reusable paths most likely to win and prevents the
benchmark binary from becoming an unmanageable set of low-value template
instantiations.

## Primary references

- [CUDA FP8 conversion and data movement](https://docs.nvidia.com/cuda/archive/13.0.2/cuda-math-api/cuda_math_api/group__CUDA__MATH__FP8__MISC.html)
- [CUDA FP16 conversion and data movement](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF__MISC.html)
- [CUDA BF16 conversion and data movement](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____BFLOAT16__MISC.html)
- [CUDA instruction-throughput table](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#throughput-of-native-arithmetic-instructions)
- [CUDA memory hierarchy and cache behavior](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html#memory-hierarchy)
- [CUDA L2 persistence controls](https://docs.nvidia.com/cuda/archive/12.2.0/cuda-c-programming-guide/index.html#device-memory-l2-access-management)
- [Hopper shared-memory and L2 limits](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html)
- [CUTLASS generic ExMy reference implementation](https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/exmy_base.h)
- [FLUTE: vectorized and duplicated LUT-quantized kernels](https://aclanthology.org/2024.findings-emnlp.724.pdf)
- [bitsandbytes NF4 table construction](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/main/bitsandbytes/functional.py)
