# Arbitrary-width DOT/GEMV benchmark plan

This experiment compares number formats independently along four axes:

1. numeric layout: total bits, exponent bits, and mantissa bits;
2. physical storage: dense bitstream or next-container padded storage;
3. access: scalar, per-thread packet, or cooperative shuffled packet;
4. arithmetic and decoder: true FP32 or FP64 accumulation with a decoder that
   constructs that target type directly.

The FP32 path must never construct FP64 and cast it to FP32.  Timed DOT and
GEMV reductions also stay in their selected arithmetic type through the final
device result.  Conversion to double is allowed only after timing for report
validation.

## Access groups

| Group | Layout | Access |
|---|---|---|
| Dense scalar | exact B-bit bitstream | `arr[i]`-compatible scalar extraction |
| Padded scalar | `uint8/16/32` per value | ordinary `arr[i]` |
| Dense packet | exact B-bit bitstream | x2/x4/x8 values per thread |
| Padded packet | `uint8/16/32` per value | x2/x4/x8 values per thread |
| Dense cooperative | exact B-bit bitstream | aligned 32-bit loads redistributed with warp shuffles |

Dense and padded datasets are generated from identical raw codes.  Layout and
access are separate CSV fields; packetization is not described as a number
format.

## Width cohorts

The new widths are 3, 5, 6, 7, 9, 10, 12, 14, 17, 20, 24, and 28 bits.  They
sample large, medium, and small dense-storage savings within the 8-, 16-, and
32-bit padded-container tiers.  Existing 2- and 4-bit formats gain padded-byte
controls.  Natural 8-, 16-, and 32-bit formats need no separate padded layout.

FP32 and FP64 use separate format inventories and decoder registries.  FP32
prioritizes E<=8 and M<=23 layouts that retain useful FP32 range or precision;
FP64 additionally samples E11 prefix formats and wider mantissas.

## Measurement stages

1. Exhaustive/sampled decoder and kernel smoke validation.
2. Screening at representative large DOT/GEMV sizes for every candidate.
3. Selection of the best one or two strategies per format, arithmetic type,
   layout, and access group.
4. Full N sweeps for finalists plus mandatory scalar/padded/raw baselines.

Screening and full timing samples are retained separately.  The benchmark does
not use Nsight Compute in this first pass.
