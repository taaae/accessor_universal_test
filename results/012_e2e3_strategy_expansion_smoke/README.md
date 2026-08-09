# 012: E2M5/E3M4 strategy expansion smoke test

This experiment validates the selected high-priority decoder additions before
they are included in a full performance sweep. It is a correctness and gross
regression check, not a source of final timing conclusions.

The 15 new variants per format are:

- `direct_fp64_words_branchy_x4/x8`: construct the nonzero 32-bit word of the
  FP64 result directly, using a branch for zero/subnormal/normal/special cases;
- `direct_fp64_words_masked_x4/x8`: the same construction with masks instead
  of the outer case branch;
- `lut_subnormal_global_x4/x8` and `lut_subnormal_shared_x4/x8`: directly
  construct normal/special values and look up only subnormal high words in a
  128-byte E2M5 or 64-byte E3M4 table;
- `lut_high_word_global_x4/x8` and `lut_high_word_shared_x4/x8`: look up the
  complete FP64 high word in a 1 KiB table; the low word is always zero;
- `lut_high_word_swizzled_shared_x4/x8`: use four padded shared-memory copies
  of the high-word table so eight-lane warp groups use different bank maps;
- `lut_prefix_shared_x8`: extend the compact shared prefix decoder to an
  eight-value source packet.

All x8 paths load two aligned 32-bit source words and extract bytes with 32-bit
shifts. This avoids relying on a 64-bit shift sequence. Constant-memory,
texture, pair-LUT, and warp-register-table variants are intentionally omitted
from this pass because they were lower-priority candidates for random E2M5 and
E3M4 data.

The executable first checks all 256 encodings for both formats, including
signed zero, subnormals, infinities, and NaNs. It then checks DOT and GEMV over
deterministic U(0,1) and N(0,1) inputs against long-double references. Short
timing samples only detect obviously pathological implementations.

Submit from the repository root and select the GPU in the command:

```bash
sbatch --wait --nodelist=gpu-nvidia-h200-1-studvm-2 \
  scripts/run_e2e3_strategy_expansion_smoke_h200.sbatch
```

Each run is stored in a timestamped subdirectory. The decisive outputs are
`decoder_validation.csv`, `kernel_validation.csv`, and
`summary_validation.txt`.
