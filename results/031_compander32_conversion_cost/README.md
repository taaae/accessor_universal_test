# 32-bit compander conversion-cost experiment

This experiment compares low-cost 32-bit companders and established 32-bit
reference decoders in one scalar-x1 DOT harness. It uses fixed 512-by-256 launch
geometry and prefixes of one maximum-size deterministic clipped normal input.

The GPU jobs request one non-student H200. Node selection stays on the `sbatch`
command line so either `gpu-nvidia-h200-2` or `gpu-nvidia-h200-3` can run it.
