# cuBLAS DOT baseline

This repository benchmarks the vendor-optimized cuBLAS DOT implementations
before introducing custom storage accessors. The current executable contains no
custom DOT reduction kernel and has no Ginkgo, Universal, or external benchmark
dependency.

The baseline provides:

- `cublasSdot` and `cublasDdot`;
- CUDA-event timing with a device-resident scalar result;
- performance sweeps from launch-bound to HBM-bound vector sizes;
- deterministic finite input data;
- CPU `long double` accuracy references for smaller cases;
- CSV output with time, throughput, algorithmic bandwidth, arithmetic
  intensity, and error metrics;
- an optional Nsight Compute profiling mode.

## Build

CUDA 13.1 and the H100/H200 nodes both support the default SM90 build:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build build --parallel 4
```

No dependency is downloaded. CMake links only CUDA Runtime and cuBLAS from the
installed CUDA Toolkit.

## Run the baseline suite

Run this after obtaining one GPU allocation:

```bash
./scripts/run_cublas_baseline.sh
```

It writes three timestamped files under `results/`:

1. performance results for signed uniform inputs;
2. accuracy results for positive inputs;
3. accuracy results for signed uniform inputs.

The default performance sizes are `2^10`, `2^16`, `2^20`, `2^23`, `2^26`,
and `2^27`. At `2^27`, the FP32 inputs occupy 1 GiB in total and the FP64
inputs occupy 2 GiB.

Environment variables can shorten or specialize a run:

```bash
OPERATION=sdot PERF_POWERS=20,23,26 SAMPLES=10 \
  ./scripts/run_cublas_baseline.sh
```

## Profile one cuBLAS call

The profiling script warms up cuBLAS, then enables the profiler around exactly
one DOT operation. By default it profiles `cublasSdot` at `N=2^27` and collects
focused speed-of-light, roofline, memory-workload, launch, occupancy, and
scheduler sections:

```bash
./scripts/profile_cublas_baseline.sh
```

Select Ddot or another power-of-two size with environment variables:

```bash
OPERATION=ddot N=67108864 ./scripts/profile_cublas_baseline.sh
```

The script writes four timestamped artifacts: the `.ncu-rep` report, a readable
details dump, a raw profiler-metrics CSV, and the benchmark's profile metadata
CSV. Timing and derived throughput columns in that last CSV are `nan`: Nsight
Compute instrumentation and replay make profile-mode elapsed time unsuitable as
a performance measurement. Use the ordinary performance-suite CSV for timing.

On the configured cluster, submit the exact H200 FP32 reference case from the
repository root:

```bash
sbatch --wait scripts/profile_cublas_sdot_h200.sbatch
```

The batch script requests one GPU from `gpu-nvidia-h200-[1-3]`, profiles
`cublasSdot` at `N=134217728`, and releases the allocation automatically when
the command finishes. It must be submitted from the repository root because it
uses Slurm's submission directory to find the checkout; Slurm runs a copied
batch script from its spool directory. The Slurm log and all profiler outputs
are written under `results/`.

Generated result files are intentionally versioned. Before committing an
`.ncu-rep`, check its size with `du -h results/*.ncu-rep`; reports at or above
100 MiB require Git LFS instead of regular Git. If the cluster reports
`ERR_NVGPUCTRPERM`, ordinary benchmark timing still works, but the profiling
hardware counters are restricted by the cluster configuration.

## Direct CLI examples

```bash
# Large-vector FP32 performance
./build/bin/dot_bench \
  --mode performance --operation sdot --n 134217728 \
  --dataset signed --output results/sdot.csv

# Accuracy and DOT condition number
./build/bin/dot_bench \
  --mode accuracy --operation both --powers 10,16,20 \
  --dataset cancellation --output results/cancellation.csv
```

Use `./build/bin/dot_bench --help` for every option.

## Metric definitions

For a DOT of `N` real values, the harness counts `2N` algorithmic FLOPs: one
multiplication and one addition per pair. It counts the two input vectors as
algorithmic bytes and ignores the scalar result and cuBLAS's private reduction
workspace.

| Operation | Algorithmic bytes | Arithmetic intensity |
|---|---:|---:|
| Sdot | `8N` | 0.25 FLOP/byte |
| Ddot | `16N` | 0.125 FLOP/byte |

`algorithmic_gb_per_s` is therefore useful for comparing implementations that
perform the same operation. Nsight Compute's measured DRAM traffic remains the
authoritative view of physical traffic, including internal partial reductions.

Accuracy mode reports:

- absolute and relative error against a compensated `long double` reference;
- normalized error `abs(error) / sum(abs(x[i] * y[i]))`;
- DOT condition number `sum(abs(x[i] * y[i])) / abs(dot(x, y))`.

The normalized error is the most stable comparison when cancellation makes the
true DOT close to zero.
