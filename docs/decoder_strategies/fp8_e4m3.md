# FP8 E4M3 strategy decision

NVIDIA E4M3 is not the generic IEEE E4M3 layout. Exponent 15 with fraction
0--6 remains finite, producing magnitudes through 448; fraction 7 is NaN. The
native CUDA conversion therefore defines the GPU reference semantics.

Implemented candidates:

- native scalar conversion directly to FP64;
- native scalar conversion to FP32 followed by FP64;
- native x2-to-`float2` and x4-to-`float4`, with x8 composed from two x4s;
- x2 `half2` conversion, composed for x4/x8;
- branchy and masked direct FP64 high-word construction;
- exact FP32-bit construction followed by FP64 conversion;
- complete 256-entry high-word LUT from global or shared memory;
- 8-entry subnormal high-word LUT hybrid;
- 65,536-entry pair LUT in global/L2, returning two high words per lookup;
- byte-permute unpacking and duplicated/bank-swizzled shared full tables;
- x1/x2/x4/x8 aligned packet loads.

The full scalar table is 1 KiB, the subnormal table is 32 B, and the pair table
is 512 KiB. The pair table is intentionally not staged into shared memory.
Local validation exhausts all codes against an independent E4M3FN formula;
NaNs are compared by class because payload canonicalization belongs to CUDA's
native conversion policy.
