# Approved FP32 four-curve amendment, 2026-09-09

This document has highest precedence over the earlier instruction-cost plan and
FP64 four-curve amendment where they conflict.

The measured curves are FP32 addition (`add32f`), FP32 multiplication
(`mul32f`), FP32 fused multiply-add (`fma32f`), and UInt32 rotation (`rot32`).
The input remains a full-width deterministic UInt32 code. Floating curves first
convert it with round-to-nearest UInt32-to-FP32, execute K dependent operations,
then widen the final FP32 value to FP64 for FP64 DOT or GEMV accumulation.
Rotation executes K rotate-left-by-7 operations before the same
UInt32-to-FP32-to-FP64 conversion.

Runtime FP32 coefficients are `a = 1024.1f` and `m = 1.0001f`. FMA uses both.
The manifest records their actual binary32 bit patterns. Explicit-rounding CUDA
intrinsics are required and timed-kernel SASS must show K dependent FADD, FMUL,
FFMA, or SHF operations per decoded operand. No FP64 or integer add/multiply
curve remains in scope.

The shared X=0 `u32_base` is now UInt32-to-FP32-to-FP64, which is not exact for
all UInt32 values. Raw FP32, FP32-to-FP64, and raw FP64 baselines remain and
retain their distinct arithmetic precision. Initial inventory is 36 cases per
kernel, 72 cases total, and 3,600 samples. Conditional K=48/64 extension,
geometry, input seeds, timing policy, drift handling, correctness, audit,
reporting, provenance and cluster safety rules remain unchanged.
