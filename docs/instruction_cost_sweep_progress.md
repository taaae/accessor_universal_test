# Instruction cost sweep progress

- Stage: blocked at assembly gate after real sm_90 compilation.
- Branch: `codex/instruction-cost-sweep`.
- Contract commit: `eedebba`.
- Implementation commit: `b69bef9` (pushed to origin).
- Current evidence: host C++ test passed; 4 negative SASS-audit tests passed;
  Python compilation, shell syntax, and `git diff --check` passed. Review found
  and prompted fixes for FP32 GEMV allocation, Release assertions, GPU decoder
  comparison, and conditional extensions.
- Cluster: VPN restored. A separate remote worktree at
  `/storage/home/timofeirusanov/accessor_universal_test_instruction_cost` keeps
  prior untracked result files untouched.
- Preflight job `460369` requested 0 GPUs, 8 CPUs, 16 GiB, and 15 minutes. It
  failed during CUDA compilation at `src/instruction_cost_bench.cu:111` due to
  a malformed compressed lambda body. No executable or SASS was produced.
- Preflight job `460371` compiled commit `e158c79` but the initial symbol parser
  did not recognize the actual Itanium enum mangling. The SASS was preserved.
- Preflight job `460373` compiled commit `cb3b465` after changing every helper
  to in-place inline-PTX constraints. The actual timed DOT K=2 SASS still folds:
  add32 is one `IMAD` per operand with multiplier 2, xor32 is one
  `LOP3.LUT ... 0xaa` identity per operand, and mul32 is one `IMAD` per operand
  with a composed multiplier. Rotation retains two SHFs and FMA retains two
  reconstruction DFMAs per operand plus accumulation. This fails the hard gate.
  Local evidence is under `results/032_instruction_cost_sweep/preflight_460373/`.
- No GPU job has been submitted. The parent research agent has the concrete
  SASS evidence and must choose whether the recipe may change. Full timing must
  not run with these folded curves.
