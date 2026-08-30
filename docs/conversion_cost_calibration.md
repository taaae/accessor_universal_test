# Fixed H200 conversion-cost calibration

This experiment predicts the complete two-stage scalar DOT time of a compiled
32-bit-storage-to-FP64 converter on one H200 NVL. It is intentionally not a
general GPU-performance model.

## Fixed target

- two independent `uint32_t` inputs generated once with Philox4x32-10;
- keys `0x6bd87c012a53f9e1` and
  `0x6bd87c012a53f9e1 XOR 0x9e3779b97f4a7c15`;
- uniform raw codes, including every format's genuine special encodings;
- scalar x1 FP64 DOT, `N=2^27`;
- 512 blocks × 256 threads, exactly 1,024 loop iterations per thread;
- one 256-thread final reduction, included in every event interval;
- CUDA 13.1.1, SM90, `-O3 -lineinfo -Xptxas=-v`, no fast math;
- ten warmups, per-case repetition calibration to about 20 ms, three rounds
  with ten samples, and deterministic shuffled case order;
- `u32 -> FP64` at the beginning and end of every round.

The complete machine-readable inventory is
`generated/conversion_calibration_manifest.json`. It contains exactly 112
synthetic training cases, 12 synthetic validation cases, five real development
cases, and six untouched real final cases.

## Static evidence and model

The generated CUDA header gives every converter a separately named compiled
kernel. `cuobjdump` SASS, rather than nominal source operations, supplies
pipeline counts, load/branch behavior, and an approximate register dependency
depth. `cudaFuncGetAttributes` supplies registers, local memory, and static
shared memory. Branch and LUT facts that cannot be soundly inferred are explicit
in the manifest.

Only the 112 training cases fit the nonnegative model:

`fixed + max(pipeline work × cost) + dependency + divergence + spill penalties`.

The model JSON is serialized and hashed before predictions for the final split
are evaluated. Development and synthetic-validation measurements never fit its
coefficients. The acceptance gates are:

- synthetic validation median APE ≤5% and p90 APE ≤10%;
- final real median APE ≤7.5%, maximum APE ≤15%, and Spearman ≥0.9.

Failure is a valid result and must be reported without final-split tuning.

## Job safety

The Slurm scripts request one GPU on the non-student `gpu-nvidia-h200-2` node.
Submission is external to the scripts and must occur only after `squeue -u
timofeirusanov` confirms the user's queue is empty. Smoke precedes full timing;
profiling is a later, separate one-GPU job. Every result directory refuses
overwrite.
