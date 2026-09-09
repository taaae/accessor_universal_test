# Instruction cost sweep progress

- Stage: implementing highest-precedence FP32 four-curve amendment.
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
- Current approved curves are `add32f`, `mul32f`, `fma32f`, and `rot32` under
  `instruction_cost_fp32_four_curve_amendment.md`. The FP64 amendment is
  historical and superseded. A new zero-GPU preflight is next after local tests.
- Zero-GPU preflight `460381` compiled commit `6797d4d`. Its timed K=2 SASS
  contains two dependent FADD, FMUL, FFMA, or SHF instructions per operand,
  followed by `F2F.F64.F32` and one DFMA accumulation. There were no spills.
  The audit correctly failed closed because Hopper spells UInt32-to-FP32 as
  `I2FP.F32.U32`, which the parser had not yet listed. That alias is now handled.
- Follow-up review found provenance, baseline-validation, metadata, drift and
  parser negative-test gaps. Commit `d323179` addressed them; subsequent local
  edits add baseline GEMV checks, direct binary/SASS/audit hash binding, exact
  inventories and two more audit counterexamples. A fresh preflight is required.
- FP32 implementation commit `6797d4d`; validation/provenance follow-up commit
  `d323179`. Local Release host test, six SASS-audit unit tests, Python compile,
  shell syntax and diff checks pass.
- Zero-GPU preflight `460381` was submitted from `6797d4d` with the user queue
  empty. VPN connectivity failed during every status check, so its outcome is
  not yet collected. Its 15-minute Slurm limit prevents an indefinite job.
- TCP connections to `10.152.225.230:22` continue to time out. When VPN access
  returns, inspect `460381`, fast-forward the remote to `d323179`, and submit a
  fresh zero-GPU preflight. Do not use the older binary for smoke or timing.
