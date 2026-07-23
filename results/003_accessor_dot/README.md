# 003: 1D reduced-storage accessor DOT

This experiment tests whether a Ginkgo-inspired one-dimensional memory
accessor is a zero-cost abstraction and whether FP64 arithmetic can improve DOT
accuracy while retaining FP32 storage bandwidth.

The controlled 2x2 variants all read the same two FP32 arrays:

| Variant | Storage | Arithmetic | Access |
|---|---|---|---|
| `raw_f32` | FP32 | FP32 | raw pointer |
| `raw_f64` | FP32 | FP64 | raw pointer plus conversion |
| `accessor_f32_f32` | FP32 | FP32 | `memory_accessor<float, float>` |
| `accessor_f64_f32` | FP32 | FP64 | `memory_accessor<double, float>` |

Every variant uses the same two-stage CUB `BlockReduce` implementation,
launch geometry, dataset, and device-resident result. The only intended
differences are access representation and arithmetic type.

## Outputs

Each run creates a timestamped `run_*` directory containing:

- `performance_samples.csv`: every unprofiled CUDA-event sample;
- `performance_summary.csv`: total DOT time, variation, bandwidth, FLOP/s,
  speedups, accessor overhead, and FP64 arithmetic cost;
- `accuracy.csv`: results and errors for every DOT v1 accuracy case;
- `accuracy_comparison.csv`: raw/accessor bit equality and FP64 error
  improvement;
- `environment.txt` and `run_manifest.txt`: reproducibility metadata;
- `profile_summary.csv`: compact per-kernel and whole-operation hardware
  metrics;
- `profile/<variant>/`: Nsight Compute reports, readable details, raw metric
  CSVs, and profiler-contaminated metadata;
- `sass.txt` and `cuda_resource_usage.txt`: compiled-code evidence for the
  zero-cost comparison.

`total_time_ms` and the derived performance values come only from the
unprofiled run. Nsight Compute replays kernels, so its enclosing application
time is never treated as benchmark timing.

The algorithmic workload is `2N` FLOPs and `8N` input bytes for all four
variants, giving 0.25 FLOP/byte. `profile_summary.csv` separately reports
physical DRAM traffic and executed FP32/FP64 operations.

## H200 command

Submit from the repository root:

```bash
sbatch --wait --nodelist=gpu-nvidia-h200-2 \
  scripts/run_accessor_dot_h200.sbatch
```

The default run performs three rotated timing rounds with 20 samples targeting
50 ms each, evaluates all nine accuracy distributions, and profiles one
`N=2^27` operation per variant.
