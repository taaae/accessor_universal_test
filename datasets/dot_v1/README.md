# DOT dataset v1

This reusable procedural dataset is designed for the FP32-storage DOT
experiments. Its implementation is `include/dot_dataset.hpp`; the manifest
records stable IDs and seeds. Do not change an existing generation rule after
results use it. Add `dot_v2` instead.

`validation_n1048576.csv` records reference statistics and 64-bit fingerprints
for both vectors. Future CPU or GPU generators can use those fingerprints to
confirm that they produced the same FP32 bit patterns.

The fingerprints and FP32 extrema are canonical. The last few printed digits
of the compensated `long double` reference may vary on hosts whose `long
double` format differs; this does not indicate a different dataset.

## Why procedural

The largest case contains two FP32 vectors of length `2^27`, or 1 GiB of input
storage. Committing every distribution at that size would be wasteful. The
generator creates identical values directly on the GPU outside timed regions,
and the CPU reference calls the same indexed generator for smaller accuracy
cases. No file I/O or host-to-device transfer is part of a timed DOT.

SplitMix64 supplies deterministic bits. The mappings use exactly representable
integer and power-of-two scaling. All defined cases contain only finite FP32
values and avoid subnormals. A rare exact zero is allowed in a signed-uniform
case and is not data-dependent control flow in the benchmark.

## Performance suite

The performance case is `signed_uniform_seed0`, with independent values
approximately uniform on `[-1, 1)`. It is evaluated at:

```text
N = 2^10, 2^16, 2^20, 2^23, 2^26, 2^27
```

`N=2^27` is the primary HBM case. Dataset size prevents cache residency, while
timing noise is controlled separately with warmups, repeated CUDA-event
samples, automatic batching, and rotating implementation order. The planned
accessor benchmark will use three timing rounds with a 50 ms target per sample,
placing an H200 run—including build, validation, and accuracy cases—within the
requested 1–5 minute budget.

## Accuracy suite

Accuracy uses every case in the manifest at:

```text
N = 2^10, 2^16, 2^20
```

At `N=2^20`, the validated DOT condition numbers are:

| Case | Condition number |
|---|---:|
| `positive_uniform` | 1 |
| `signed_uniform_seed0` | 824.84 |
| `signed_uniform_seed1` | 955.46 |
| `signed_uniform_seed2` | 2825.86 |
| `cancellation_1e2` | 127.00 |
| `cancellation_1e4` | 16383.01 |
| `cancellation_1e6` | 1048554.00 |
| `near_orthogonal` | 8387934.85 |
| `dynamic_range_40` | 39.37 |

The cancellation cases store matching random magnitudes in the two halves.
The second half has the opposite sign and a representable residual
`2^-parameter`. This makes the condition number deliberate rather than an
accidental property of a random sample. The near-orthogonal case is the most
severe member of the same construction.

`dynamic_range_40` chooses each input exponent uniformly from `[-20, 20]`, so
product magnitudes can span roughly 80 binary exponents without deliberately
introducing overflow or underflow.

References are compensated `long double` sums of exact FP32-to-`long double`
products. Every result records the realized condition number

```text
sum(abs(x[i] * y[i])) / abs(sum(x[i] * y[i]))
```

as well as absolute, relative, and normalized error.

## Validation

Build and inspect the definitions with:

```bash
cmake --build build-h200 --target dataset_inspect --parallel 4
./build-h200/bin/dataset_inspect
```

The inspector reports reference DOTs, condition numbers, extrema, zeros,
subnormals, and nonfinite values. The design follows the XBLAS practice of
combining ordinary random data with mathematically constructed cancellation,
and the ReproBLAS use of nearly orthogonal vectors to expose summation error.
