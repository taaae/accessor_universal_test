# Experiment 016: expanded format strategy smoke

This is a correctness and compilation gate for the 22 formats added to fill
every E/M split at 2, 4, 8, 16, and 32 total bits subject to `E <= 11`.

The test validates:

- exhaustive raw-code decoding for 2/4/8/16-bit formats and sampled decoding
  for 32-bit formats;
- genuinely bit-dense 2- and 4-bit device storage;
- scalar and x2/x4/x8 packet paths, including small LUT, warp-register LUT,
  byte-to-four lookup, and CUDA FP4 candidates;
- a small fused FP64-accumulating DOT and GEMV launch for every candidate.

This experiment is not a performance benchmark. Its output decides which
strategies are safe to put into the later timing experiment.
