# Instruction cost sweep progress

- Stage: local implementation corrected after independent review; preflight next.
- Branch: `codex/instruction-cost-sweep`.
- Contract commit: `eedebba`.
- Current evidence: host C++ test passed; 4 negative SASS-audit tests passed;
  Python compilation, shell syntax, and `git diff --check` passed. Review found
  and prompted fixes for FP32 GEMV allocation, Release assertions, GPU decoder
  comparison, and conditional extensions.
- Next action: pass local checks, commit and push, then submit zero-GPU preflight.
