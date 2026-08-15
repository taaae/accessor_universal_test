# 2-bit padded-control cohort

## Formats and arithmetic

FP32 and FP64 both test E0M1 and E1M0.  Each target has its own direct-bit/word
decoder and final-target LUT; no FP64 intermediate is used by FP32.

## Decoder candidates

E0M1 tests fixed-integer scaling, E1M0 tests exponent-only decoding, and both
test direct target construction plus a four-entry global/shared LUT.  The
generic codec is retained as an x1 control.

## Physical access candidates

- exact dense 2-bit scalar versus one-byte-per-value padded scalar;
- dense and padded x2/x4/x8 thread packets;
- dense cooperative access: 16 values in one 32-bit word, four consumers,
  four values per consumer.

The padded path intentionally spends four times as many source bytes.  It is a
baseline for deciding whether dense sub-byte storage is useful after extraction
and redistribution overhead, not an alternative numeric format.
