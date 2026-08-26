# Posit and takum storage-conversion benchmark specification

This document records the decisions made before implementation. The experiment
tests whether posit, linear takum, or logarithmic takum is useful as a compact
GPU storage type when the kernel converts every loaded value to FP32 or FP64
and performs ordinary floating-point arithmetic.

The experiment measures performance, not whether one number family is more
accurate than another. Accuracy depends on the application and input
distribution, so the preliminary accuracy-matching experiment does not decide
the formats or performance winners here.

## Questions the experiment must answer

| Question | Evidence used to answer it |
|---|---|
| 1. Is a full LUT the best strategy for low-bit posit and takum? | Compare the direct, shared full-LUT, and global full-LUT decoders for 8- and 14-bit formats in every main DOT and GEMV case. |
| 2. Does a posit or takum LUT have the same speed as an IEEE LUT? | Use the same raw index stream, scalar kernel, table placement, and output type. Change only the table contents. Compare both scattered and concentrated index traces. |
| 3. What is the best strategy once a shared full LUT is too large? | At 16 bits, compare a direct decoder with a full global LUT. At 32 bits, test the direct decoder because a full LUT is impractical. |
| 4. How do the large-bit strategies compare with IEEE at the same bit width? | Compare the best posit and takum result with the fastest retained same-width IEEE result for the same arithmetic type, distribution, kernel, and problem size. |

Each question is answered case by case. The report must not replace these
answers with one overall format or strategy score.

## Scope and non-goals

The storage value is decoded to FP32 or FP64 before multiplication. DOT and
GEMV then use ordinary FP32 or FP64 FMA and accumulation. The experiment does
not use a posit quire, posit arithmetic, or takum arithmetic.

The following work is outside this experiment:

- deciding whether posit or takum is generally more accurate than IEEE;
- finding an application-independent accuracy distribution or loss function;
- testing `N(0,1)` or `U(0,1)` as performance-selection distributions;
- searching packet widths, padded layouts, shuffle redistribution, or loader
  and consumer decompositions;
- dropping a strategy because it lost at a smaller problem size;
- combining distributions, kernels, or arithmetic types with a geometric mean;
- implementing many approximate antilogarithm variants for logarithmic takum.

## Alternative-format inventory

All alternative formats run with both FP32 and FP64 arithmetic.

| Bits | Posit | Linear takum | Logarithmic takum |
|---:|---|---|---|
| 8 | `posit<8,0>` | `takum<8>` | `takum_log<8>` |
| 14 | `posit<14,1>` | `takum<14>` | `takum_log<14>` |
| 16 | `posit<16,1>` | `takum<16>` | `takum_log<16>` |
| 32 | `posit<32,2>` | `takum<32>` | `takum_log<32>` |

Only one posit exponent-size configuration is tested at each width. The chosen
sequence is the conventional progression from `es=0` at 8 bits to `es=2` at
32 bits. This avoids turning the experiment into a posit-parameter search.

## Alternative-format strategies

| Bits | Strategies |
|---:|---|
| 8 | direct, shared full LUT, global full LUT |
| 14 | direct, shared full LUT, global full LUT |
| 16 | direct, global full LUT |
| 32 | direct |

The direct decoder is fixed per family.

### Posit direct decoder

1. Convert a negative posit encoding to its magnitude representation.
2. Count leading regime bits with `clz` or its leading-one equivalent.
3. Extract regime, exponent, and fraction with masks and shifts.
4. Assemble the FP32 or FP64 result directly.

There is no bit-by-bit regime loop and no `pow` call.

### Linear-takum direct decoder

1. Extract sign, direction, and the three regime bits.
2. Decode the characteristic with the bounded prefix and a tiny fixed table.
3. Extract the remaining fraction.
4. Assemble the FP32 or FP64 exponent and significand directly.

The prefix work remains bounded as the storage width grows.

### Logarithmic-takum direct decoder

1. Use the same bounded-prefix decode to recover the signed fixed-point
   logarithm.
2. Convert its magnitude with `exp2f` for FP32 or `exp2` for FP64.
3. Apply the decoded sign.

The main benchmark uses normal CUDA math, not `--use_fast_math`. A segmented
antilogarithm LUT or polynomial is not part of the first experiment. If
`exp2` dominates the result, that is already an answer about straightforward
logarithmic-takum storage conversion.

## IEEE comparison inventory and strategies

The IEEE subset contains native formats, direct target-shift formats, and two
custom allocations at each bit width. Integer-like E0 and E1 formats and
exponent-only M0 formats are excluded.

Every strategy in this table uses scalar `x1` access. `native_packed` and all
packet decoders are excluded.

| Bits | IEEE type | Arithmetic | Scalar strategies |
|---:|---|---|---|
| 8 | FP8 E4M3 | FP32, FP64 | `native_scalar`, `full_lut_shared`, `full_lut_global` |
| 8 | FP8 E5M2 | FP32, FP64 | `native_scalar`, `full_lut_shared`, `full_lut_global` |
| 8 | E3M4 | FP32, FP64 | `full_lut_shared`, `full_lut_global` |
| 8 | E6M1 | FP32, FP64 | `full_lut_shared`, `full_lut_global` |
| 14 | E8M5 | FP32 | `direct_branchy`, `full_lut_shared`, `full_lut_global` |
| 14 | E11M2 | FP64 | `direct_masked`, `full_lut_shared`, `full_lut_global` |
| 14 | E2M11 | FP32, FP64 | `direct_branchy`, `full_lut_shared`, `full_lut_global` |
| 14 | E5M8 | FP32, FP64 | `direct_branchy`, `full_lut_shared`, `full_lut_global` |
| 16 | FP16 E5M10 | FP32, FP64 | `native_scalar`, `full_lut_global` |
| 16 | BF16 E8M7 | FP32, FP64 | `native_scalar`, `full_lut_global` |
| 16 | E11M4 | FP64 | `direct_masked`, `full_lut_global` |
| 16 | E3M12 | FP32, FP64 | `subnormal_lut_global`, `full_lut_global` |
| 16 | E6M9 | FP32, FP64 | `direct_branchy`, `prefix_lut_global`, `full_lut_global` |
| 32 | FP32 E8M23 | FP32, FP64 | `native_scalar` |
| 32 | E11M20 | FP64 | `direct_masked` |
| 32 | E4M27 | FP64 | `direct_branchy`, `prefix_lut_global` |
| 32 | E10M21 | FP64 | `direct_branchy`, `prefix_lut_global` |

The reduced arithmetic combinations keep the IEEE source exactly representable
by the target arithmetic across the tested source format. Alternative formats
still run in both arithmetic modes because their safe input intervals are
defined separately for FP32 and FP64.

## Lookup-table sizes and contents

| Storage bits | FP32 table | FP64 table | Placement tested |
|---:|---:|---:|---|
| 8 | 1 KiB | 2 KiB | shared and global |
| 14 | 64 KiB | 128 KiB | shared and global |
| 16 | 256 KiB | 512 KiB | global only |
| 32 | 16 GiB | 32 GiB | not tested |

FP64 tables contain the complete 64-bit FP64 encoding. They must not store only
32 bits and reconstruct the rest.

Host-side LUT construction and the host-to-device copy happen before timing.
A global table lookup is part of the timed kernel. Copying a LUT from global
memory into per-block shared memory is also part of the timed kernel because
every block pays that cost.

A 14-bit FP64 shared LUT needs 128 KiB plus the kernel's other shared storage.
The implementation must request dynamic shared memory explicitly. If the
kernel cannot launch, the result is recorded as infeasible rather than silently
omitted or replaced.

## Storage and access

- Use one dense packed storage layout per width.
- Eight-, 16-, and 32-bit values use their natural byte or word representation.
- Fourteen-bit values occupy a true contiguous 14-bit stream. They are not
  padded to 16 bits.
- Each thread decodes one logical value at a time.
- Set `access_method=scalar` and `packet_values=1` everywhere.
- Do not compile or run x2, x4, x8, cooperative-shuffle, or split
  loader-consumer variants.
- Reuse the existing ordinary x1 DOT and GEMV launch structure. Do not search
  launch configurations as another experiment axis.

## Main distributions

Every main performance case uses both distributions below. Results remain
separate throughout analysis and reporting.

### `field_balanced_finite`

This is the primary conversion-path distribution. It prevents a narrow range
near one from making almost every posit regime short or every takum prefix the
same.

All families use a 50/50 sign split and exclude zero and special encodings.
Magnitudes stay inside the per-format interval listed below.

For posits:

- split samples equally between positive and negative regimes;
- give every reachable regime length equal weight;
- sample exponent bits uniformly;
- sample remaining fraction bits uniformly.

For linear and logarithmic takums:

- balance direction bit `D`;
- give all eight regime codes equal weight;
- sample available characteristic bits uniformly;
- sample remaining fraction or logarithmic-mantissa bits uniformly;
- reject encodings outside the allowed magnitude interval, then rebalance the
  regime buckets after rejection.

For IEEE formats, use equal normal and source-subnormal halves whenever the
source subnormals remain nonzero in the selected arithmetic type. In the normal
half, sample permitted exponent codes uniformly and sample fraction bits
uniformly. In the subnormal half, sample nonzero fraction fields uniformly.
Balance signs in both halves.

Operand magnitudes are paired anti-correlatively. Large values are paired with
small values, and the operands are swapped for half the pairs. This keeps
products and reductions finite while preserving the same balanced marginal
histogram in both input buffers. GEMV applies the same pairing by column so the
matrix column and its vector element have complementary magnitude ranges.

### `paired_log_uniform_finite`

For a format interval `[a,b]`, generate continuous exponents

\[
q_L \sim U(a,b), \qquad q_R = a+b-q_L,
\]

then generate

\[
x_L=(-1)^{s_L}2^{q_L}, \qquad x_R=(-1)^{s_R}2^{q_R}.
\]

The signs are independent and balanced. Continuous exponents generate ordinary
significands rather than only exact powers of two. Both operands have the same
log-uniform marginal distribution, while `q_L + q_R = a + b` bounds every
product.

For GEMV, choose `q_L` per column. Matrix elements in that column use the same
magnitude scale with independently generated signs and significands. The vector
uses the complementary exponent `q_R`.

### IEEE exponent intervals

The table contains only arithmetic combinations used in the main experiment.
Each interval describes `q = log2(abs(x))`.

| Bits | Format | FP32 arithmetic | FP64 arithmetic |
|---:|---|---:|---:|
| 8 | E4M3 FP8 | `[-8,8]` | `[-8,8]` |
| 8 | E5M2 FP8 | `[-15,15]` | `[-15,15]` |
| 8 | E3M4 | `[-5,3]` | `[-5,3]` |
| 8 | E6M1 | `[-28,28]` | `[-28,28]` |
| 14 | E8M5 | `[-120,120]` | not run |
| 14 | E11M2 | not run | `[-1000,1000]` |
| 14 | E2M11 | `[-10,1]` | `[-10,1]` |
| 14 | E5M8 | `[-20,14]` | `[-20,14]` |
| 16 | E5M10 FP16 | `[-20,14]` | `[-20,14]` |
| 16 | E8M7 BF16 | `[-120,120]` | `[-120,120]` |
| 16 | E11M4 | not run | `[-1000,1000]` |
| 16 | E3M12 | `[-12,3]` | `[-12,3]` |
| 16 | E6M9 | `[-36,28]` | `[-36,28]` |
| 32 | E8M23 FP32 | `[-120,120]` | `[-120,120]` |
| 32 | E11M20 | not run | `[-1000,1000]` |
| 32 | E4M27 | not run | `[-30,7]` |
| 32 | E10M21 | not run | `[-500,500]` |

The asymmetric E2M11, E3M12, and E4M27 intervals are intentional. These
formats have short exponent fields and a large subnormal region. A symmetric
interval would omit a material part of their finite range.

### Posit exponent intervals

| Bits | Format | FP32 arithmetic | FP64 arithmetic |
|---:|---|---:|---:|
| 8 | `posit<8,0>` | `[-5,5]` | `[-5,5]` |
| 14 | `posit<14,1>` | `[-22,22]` | `[-22,22]` |
| 16 | `posit<16,1>` | `[-26,26]` | `[-26,26]` |
| 32 | `posit<32,2>` | `[-112,112]` | `[-112,112]` |

These ranges stop inside `minpos` and `maxpos`. The endpoint values have almost
no fraction bits and should not receive ordinary-sample weight comparable to
the rest of the range.

### Linear-takum exponent intervals

Takum width changes precision much more than range, so all tested widths use
the same interval for a given arithmetic type.

| Bits | Format | FP32 arithmetic | FP64 arithmetic |
|---:|---|---:|---:|
| 8 | `takum<8>` | `[-140,127]` | `[-240,240]` |
| 14 | `takum<14>` | `[-140,127]` | `[-240,240]` |
| 16 | `takum<16>` | `[-140,127]` | `[-240,240]` |
| 32 | `takum<32>` | `[-140,127]` | `[-240,240]` |

The interval `[-140,127]` is uniform in the exponent `q`, not uniform in the
numeric value `x`. It reaches into FP32 subnormal magnitudes while keeping every
decoded input finite and nonzero.

### Logarithmic-takum exponent intervals

| Bits | Format | FP32 arithmetic | FP64 arithmetic |
|---:|---|---:|---:|
| 8 | `takum_log<8>` | `[-120,120]` | `[-170,170]` |
| 14 | `takum_log<14>` | `[-120,120]` | `[-170,170]` |
| 16 | `takum_log<16>` | `[-120,120]` | `[-170,170]` |
| 32 | `takum_log<32>` | `[-120,120]` | `[-170,170]` |

Field balancing for logarithmic takum uses encoded regime and characteristic
fields. It does not divide the base-2 exponent interval into equal buckets.

## Paired LUT controls

Question 2 needs a controlled comparison in addition to the two main numerical
distributions. Reuse one raw code-index trace while swapping only the LUT
contents among IEEE, posit, linear takum, and logarithmic takum.

Run this control at both 8 and 14 bits, for both FP32 and FP64 table entries.

Run two traces:

- scattered indices covering the complete table;
- concentrated indices using a small cached part of the table.

The input width, table location, table-entry width, scalar kernel, and launch
configuration must be identical within a paired comparison. These controls are
reported separately and do not act as main numerical distributions or enter
strategy selection.

Each width/arithmetic control is one executable. Its IEEE, posit, linear-takum,
and logarithmic-takum variants call the same compiled table-only kernel and are
rotated within every timing round. The runtime table pointer and table contents
are the only family-dependent kernel inputs.

## Kernels and problem sizes

Only the largest application cases select winners.

| Kernel | Dimensions | Meaning |
|---|---|---|
| DOT | `N = 2^27 = 134,217,728` | Two storage arrays are decoded and accumulated into one FP32 or FP64 result. |
| GEMV | `M = 1024`, `N = 65,536` | The matrix has 1024 rows and 65,536 columns. Each row is one reduction of length 65,536. |

The fixed GEMV value near one thousand is `M`, the number of rows. It is not
the reduction length. A row-per-block implementation has enough independent
rows at `M=1024` to keep the GPU occupied, so this experiment does not add a
second GEMV shape.

Large inputs are intentional. If memory stalls overlap conversion work and two
decoders have the same end-to-end time, then conversion has no measurable cost
for the intended HPC workload. A smaller DOT or GEMV is not used to expose
decoder cost. If raw decoder latency later needs explanation, that belongs in a
separate conversion microbenchmark.

Every listed strategy must run the full configured DOT and GEMV case. There is
no screening stage that can remove a strategy before these runs.

## Timed region and arithmetic

The timed region includes:

- loading the dense encoded inputs;
- extracting one logical value;
- direct conversion or LUT lookup;
- per-block shared-LUT staging when applicable;
- FP32 or FP64 multiplication, FMA, and reduction.

The timed region excludes:

- source-data generation and encoding;
- LUT construction;
- memory allocation;
- host-to-device and device-to-host copies.

The compiler must use the same numerical flags for every strategy. The main
build disables `--use_fast_math` and preserves subnormal behavior with
`--ftz=false`.

## Validation

Decoder validation happens before performance measurement.

- Exhaustively validate every encoding for 8, 14, and 16 bits.
- Validate special boundaries and a large deterministic random sample for
  32-bit formats.
- Use Universal's CPU conversion as the format reference.
- Require exact FP32 or FP64 output bits for algebraic direct decoders and LUTs
  when the reference conversion is exact.
- For direct logarithmic-takum conversion, allow at most 2 ULP for FP32 and
  1 ULP for FP64 against the reference.
- Verify that full LUTs contain the canonical output for every code.
- Require the DOT and every GEMV output to be finite.

Before timing each generated input buffer, verify:

- no source special encoding;
- no decoded NaN or infinity;
- no decoded zero;
- every realized `q` lies inside the promised interval;
- field-bucket counts differ by at most one after admissibility filtering;
- paired-log-uniform histograms cover the complete requested interval.

## Timing protocol

- Use CUDA events around the kernel only.
- Perform 10 untimed warm-up launches.
- Collect 30 timed samples.
- Interleave strategy order across timing rounds so temperature and clock drift
  do not consistently favor one strategy.
- Keep global LUTs warm after the warm-up stage. This models repeated use in an
  HPC application.
- Store every raw timing sample.
- Report the median and a confidence interval for paired timing ratios.

Use a paired round bootstrap when variants were interleaved in one executable.
Use an independent bootstrap for comparisons between separate format
executables, including the broad same-width alternative-versus-IEEE comparison.

For the statement that two implementations have the same speed, use a 3%
equivalence band. Declare equivalence only when the confidence interval for
their timing ratio lies completely inside `[0.97,1.03]`. If the interval crosses
a boundary, report the comparison as inconclusive rather than equal.

## Selection and comparison rules

The analysis key is

\[
(\text{format},\ \text{arithmetic},\ \text{distribution},\ \text{kernel}).
\]

There is only one configured problem size per kernel. Within each key, the
strategy with the lowest median kernel time is the case winner.

The analysis follows these rules:

- keep `field_balanced_finite` and `paired_log_uniform_finite` separate;
- keep DOT and GEMV separate;
- keep FP32 and FP64 arithmetic separate;
- do not calculate a geometric mean or one total performance score;
- do not choose a strategy using one distribution and silently apply it to the
  other distribution;
- report a distribution-dependent or kernel-dependent winner as such;
- compare full LUTs with the same placement directly for Question 2;
- compare the best alternative-format strategy against the fastest retained
  same-width IEEE strategy for Question 4.

## Required result records

The run must preserve enough information to reproduce every conclusion:

- the complete format, arithmetic, distribution, kernel, and strategy matrix;
- raw timing samples and medians;
- paired timing ratios and equivalence intervals;
- decoder and kernel validation results;
- realized sign, regime, characteristic, exponent, and subnormal histograms;
- physical input bytes and LUT bytes;
- compiler flags, CUDA version, GPU identity, launch configuration, and random
  seeds;
- explicit infeasible or failed rows rather than silent omissions.

The final report answers the four research questions in order. It shows the
case tables first and then explains where winners change with the family,
arithmetic type, distribution, or kernel. It does not claim that the experiment
establishes an application-independent accuracy advantage for any family.
