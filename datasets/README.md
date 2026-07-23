# Reusable datasets

Datasets in this repository are procedural and versioned. Large binary vectors
are not committed: each dataset directory defines stable case IDs, seeds, and
generation rules that produce the same FP32 inputs on the CPU and GPU.

| Dataset | Purpose |
|---|---|
| `dot_v1` | DOT performance and mixed-precision accuracy experiments |

Benchmark results belong under `results/`, while reusable input definitions
belong here.
