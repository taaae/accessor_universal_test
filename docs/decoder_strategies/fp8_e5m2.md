# FP8 E5M2 strategy decision

NVIDIA E5M2 uses the IEEE-like exponent-31 Inf/NaN class and shares FP16's
five-bit exponent range. Every E5M2 value is exactly representable in FP16 and
FP32. Source subnormals occur with probability below 0.01% in both benchmark
distributions, so normal-path instruction count matters much more than CLZ.

Implemented candidates:

- native scalar conversion directly to FP64;
- native scalar conversion through FP32;
- native x2-to-`float2`, x4-to-`float4`, and composed x8;
- native x2-to-`half2`, composed for x4/x8;
- branchy and masked direct FP64 high-word construction;
- exact FP32-bit construction followed by FP64 conversion;
- complete 256-entry high-word LUT in global or shared memory;
- four-entry subnormal high-word LUT control;
- 65,536-entry global/L2 pair LUT;
- byte-permute unpacking and duplicated/bank-swizzled shared full tables;
- x1/x2/x4/x8 aligned packet loads.

The exhaustive local reference distinguishes infinity from NaN and verifies
all finite values bit-for-bit. NaN payloads are compared by class in the GPU
smoke test because CUDA may canonicalize them during native conversion.
