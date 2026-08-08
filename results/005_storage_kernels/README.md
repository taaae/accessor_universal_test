# 005: Generic storage-format DOT and GEMV validation

This experiment integrates all 17 shortlisted storage formats into two FP64
arithmetic kernels:

- a two-stage DOT reduction;
- a one-block-per-row, row-major GEMV.

Every format is instantiated with scalar, x2, and x4 load/decode paths. Packed
variants operate on the same encoded scalar arrays, so they transfer the same
number of bytes and differ only in load, decode, indexing, and accumulation
structure. DOT uses an independent accumulator per lane. GEMV packs along the
column dimension and pads its physical leading dimension to x4 alignment while
also testing a non-multiple-of-four logical column count.

Inputs use deterministic U(-1,1) and N(0,1) distributions without rescaling.
All arithmetic and outputs are FP64. Correctness is checked against host
references computed from the encoded-and-decoded storage values; separate
metrics compare those storage references with the original FP64 inputs.
Matching non-finite results are accepted because strict E2 formats can overflow
on unscaled Gaussian tails.

This is a functional and generated-code experiment, not a performance result.
Each timestamped run contains DOT/GEMV CSVs, environment and manifest files,
CTest and validator logs, CUDA resource usage, and SASS.

## H200 command

Submit from the repository root:

```bash
sbatch --wait scripts/run_storage_kernels_h200.sbatch
```
