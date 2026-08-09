# Expanded E/M format decisions

This note records the decision made before registering each of the 22 formats
that complete the experiment. Together with the 16 formats already present,
the repository now covers every sign/exponent/mantissa split at 2, 4, 8, 16,
and 32 total bits subject to `E <= 11`.

All candidates decode exactly to FP64. A route through FP32 is registered only
when every finite source value is exactly representable as FP32. `xN` means one
thread loads and consumes `N` adjacent, densely stored source values.

## Shared strategy vocabulary

- `generic`: reference codec, retained as a correctness/control path.
- `word_branchy`, `word_masked`: construct FP64 high/low register words
  directly; the latter replaces value-class branches with masks.
- `fp32_bits`: construct exact FP32 bits, then use the native FP32-to-FP64
  conversion.
- `fixed_integer`: convert the E0 integer magnitude and multiply by an exact
  power-of-two scale.
- `exponent_only`: construct signed zero, a power of two, or infinity without
  fraction handling.
- `full_high`: look up FP64's 32-bit high word. This is complete for every
  format of at most 16 bits because the FP64 low word is zero.
- `subnormal`: look up only the subnormal fraction; construct normals directly.
- `prefix`: look up sign+exponent and insert the fraction directly.
- `pair`: one lookup returns the two FP64 high words for two source codes.
- `byte_quad`: one 8-bit lookup returns four decoded 2-bit values.
- `warp`: lanes 0..3 or 0..15 hold a 2-/4-bit full table in registers and
  distribute entries with warp shuffle; partial warps fall back to read-only
  cache.
- `native`: CUDA's FP4/FP8/FP16/BF16/FP32 conversion path.

## 2-bit formats

The physical representation is dense: four values per byte. x1/x2 extract a
field from a byte, x4 loads one byte, and x8 loads one 16-bit word.

### E0M1

Semantics: signed fixed point, `(-1)^s * m * 2^-1`; encodings are `+0`,
`+0.5`, `-0`, and `-0.5`.

Candidates: generic, branchy/masked words, exact FP32 bits, fixed integer,
16-byte full LUT in global/shared/warp placement, 128-byte pair LUT in
global/shared placement, and the 4-KiB byte-quad LUT in global/shared
placement. The byte-quad candidate trades a larger table for one lookup per
four values.

### E1M0

Semantics: IEEE-like exponent-only endpoint. Exponent zero is signed zero and
the all-ones exponent is infinity, so this deliberately has no finite nonzero
value and no NaN payload.

Candidates: generic, branchy/masked words, exact FP32 bits, exponent-only,
full/warp/pair/byte-quad LUTs, including shared placement for the pair and
byte-quad tables. Fixed-integer decoding is inapplicable.

## 4-bit formats

The physical representation is dense: two values per byte. x1 extracts a
nibble, x2 loads one byte, x4 loads 16 bits, and x8 loads 32 bits.

### E0M3

Semantics: signed fixed point `m * 2^-3`, with maximum magnitude `0.875`.

Candidates: generic, branchy/masked words, exact FP32 bits, fixed integer,
64-byte full LUT in global/shared/warp placement, and a 2-KiB pair LUT in
global/shared placement.

### E1M2

Semantics: the existing finite E1 uniform-grid policy generalized to four
bits. All exponent patterns are finite; decoding is an integer magnitude times
`2^(1-M)`.

Candidates: generic, branchy/masked words, exact FP32 bits, E1 integer,
full/warp/pair LUTs.

### FP4 E2M1

Semantics: NVIDIA E2M1, finite-only with values through magnitude 6; there are
no Inf/NaN encodings. CUDA 13.1 exposes scalar, x2, and x4 FP4 storage and
conversion intrinsics. On H200 (SM90) these are expected to use CUDA's
emulation path, which is still an important vendor baseline.

Candidates: generic, branchy/masked words, exact FP32 bits, full/warp/pair
LUTs, scalar CUDA FP4-to-half, and packed CUDA FP4x2-to-half2 applied across
x2/x4/x8 packets.

### E3M0

Semantics: exponent-only IEEE endpoint: signed zero, six finite powers of two,
and infinity; there is no NaN payload.

Candidates: generic, branchy/masked words, exact FP32 bits, exponent-only,
full/warp/pair LUTs.

## Added 8-bit formats

Every x2/x4/x8 candidate uses ordinary 16-/32-/64-bit coalesced packet loads.
All full tables are only 1 KiB; pair tables are 512 KiB and intentionally test
an L2-capacity tradeoff.

### E0M7

Signed fixed point with maximum magnitude `127/128`. Candidates: generic,
branchy/masked words, exact FP32 bits, fixed integer, full LUT
(global/shared/swizzled), and pair LUT. There is no exponent prefix or
subnormal class.

### E6M1

IEEE-like with a large exponent range and one fraction bit. Candidates:
generic, branchy/masked words, exact FP32 bits, full LUT
(global/shared/swizzled), two-entry subnormal LUT, sign+exponent prefix LUT,
and pair LUT.

### E7M0

IEEE exponent-only with signed zero, finite powers of two, and infinity.
Candidates: generic, branchy/masked words, exact FP32 bits, exponent-only,
full LUT (global/shared/swizzled), and pair LUT. A prefix LUT duplicates the
full table and is omitted.

## Added 16-bit formats

A full high-word table is 256 KiB and therefore an L2/read-only-cache
candidate, not a shared-memory candidate. Prefix and subnormal tables are much
smaller. Packet widths x1/x2/x4/x8 are all registered.

### E0M15

Signed fixed point with maximum magnitude `32767/32768`. Candidates: generic,
branchy/masked words, exact FP32 bits, fixed integer, and the 256-KiB full
table. Prefix/subnormal tables are inapplicable.

### E4M11

Candidates: generic, branchy/masked words, exact FP32 bits, full L2 table,
8-KiB subnormal table (global/shared), and 128-byte sign+exponent prefix table
(global/shared).

### E6M9

Candidates: the same direct and exact-FP32 routes, full L2 table, 2-KiB
subnormal table, and 512-byte prefix table.

### E7M8

Candidates: the same direct and exact-FP32 routes, full L2 table, 1-KiB
subnormal table, and 1-KiB prefix table.

### E9M6

Candidates: generic, branchy/masked words, full L2 table, 256-byte subnormal
table, and 4-KiB prefix table. FP32 is excluded because E9 values exceed the
FP32 exponent range.

### E10M5

Candidates: generic, branchy/masked words, full L2 table, 128-byte subnormal
table, and 8-KiB prefix table. FP32 is excluded because E10 values exceed the
FP32 exponent range.

## Added 32-bit formats

Full and subnormal tables are infeasible. Direct decoding constructs both
32-bit halves of FP64 as needed. Prefix tables remain small because they depend
only on sign+exponent. Testing x8 is intentional even though eight decoded
doubles can increase register pressure.

### E0M31

Signed fixed point with maximum magnitude just below one. Candidates: generic,
branchy/masked word construction and fixed-integer scaling. FP32 is inexact,
and exponent-prefix lookup does not model fixed-point normalization.

### E4M27

Candidates: generic, branchy/masked two-word construction, and a 128-byte
prefix table in global/shared placement. FP32 is inexact because M27 exceeds
FP32 precision.

### E5M26

Same families as E4M27; the prefix table is 256 bytes. FP32 remains inexact.

### E6M25

Same families; the prefix table is 512 bytes. FP32 remains inexact.

### E7M24

Same families; the prefix table is 1 KiB. M24 exceeds FP32's explicit
fraction capacity, so FP32 remains inexact.

### E9M22

Generic, branchy/masked two-word construction, and a 4-KiB prefix table.
Although M22 fits FP32 precision, E9 exceeds FP32 range, so FP32 is excluded.

### E10M21

Generic, branchy/masked two-word construction, and an 8-KiB prefix table.
FP32 is excluded by exponent range.

## Deliberate exclusions

- No approximate or lossy intermediate conversion is timed as an exact
  decoder.
- No full 32-bit LUT (`2^32` entries).
- No 16-bit full LUT in shared memory (256 KiB before any kernel state).
- No pair LUT above 8 bits (the index space grows as `2^(2B)`).
- No shuffled source-data loads for 2/4 bits: both divide bytes exactly, so
  ordinary dense packet loads are simpler. Warp shuffle is tested only as a
  LUT-distribution mechanism.
