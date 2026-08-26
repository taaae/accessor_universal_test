# Context-free CUDA review

Date: 2026-08-26

A separate agent received only the repository path, benchmark specification,
cluster rules, focused file list, and review criteria. It made no edits.

The initial review found seven issues:

1. alternative decoder validation needed an independent reference chain;
2. GEMV mutations needed final interval validation;
3. finiteness had to be checked for every feasible strategy;
4. 14-bit storage needed an aligned trailing guard word;
5. distribution histograms and their validation were incomplete;
6. LUT-control ranges needed realized and distinct-index evidence;
7. cross-executable confidence intervals needed an explicitly independent
   method rather than being described as paired.

The implementation was revised for all seven. Two focused re-reviews inspected
the live fixes. The final review reported no remaining CUDA synchronization,
packing, validation, timing, analysis, resource, or specification defect. It
also confirmed safe barriers, sufficient H200 dynamic shared memory, valid
buffer ownership, complete variant coverage, and compliance with the one-GPU
rule.

Local evidence at the final review gate:

- Universal and independent log-takum paper-formula cross-validation passed;
- exhaustive 8/14/16-bit and one-million-code 32-bit reference coverage passed;
- the synthetic complete-matrix validator test passed;
- the deterministic bootstrap-analysis test passed;
- shell syntax, Python compilation, and `git diff --check` passed.

After the first smoke runs exposed two host-side input-construction edge cases,
the reviewer performed one more context-free pass over the finished fixes. It
confirmed that column-indexed GEMV pools preserve the intended per-column
distribution and that bounded rejection of endpoint-rounding samples is
deterministic, symmetric between operands, sign-balanced, terminates safely,
and does not affect CUDA synchronization or timed regions. No new finding was
reported.

A subsequent smoke validation showed that endpoint coverage must account for
the support of the complementary pair, not each operand in isolation. The
reviewer reproduced the issue for FP32 `takum<8>` (individual support
`[-127,127]`, pair-admissible marginal support `[-127,119]`). After the check
was changed to intersect the quantization cells of both complementary targets,
the final review of commit `07256b8` reported no remaining finding. It verified
the midpoint tie handling, bounded inward endpoint nudges, 32-bit binary-search
safety, and the expected `takum<8>` support.
