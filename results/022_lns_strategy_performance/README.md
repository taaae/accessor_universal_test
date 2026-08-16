# Experiment 022: LNS conversion and fused multiplication

This experiment benchmarks the LNS inventory documented in
`docs/lns_strategy_benchmark.md` with FP32 and FP64 arithmetic.  Ordinary
decode/decode/multiply and fused log-add/decode kernels are reported
separately.  Dense, padded, per-thread packet, and cooperative-shuffle access
remain independent strategy dimensions.

The job runs `compute-sanitizer --tool synccheck` over the warp-shuffle
variants, validates and smoke-tests every compiled strategy, screens all
variants at representative sizes, and performs full N sweeps only for the
best strategies in each arithmetic/access group.  Raw FP32 and FP64 anchors
are remeasured throughout the same allocation.

The batch script does not select a GPU.  Submit from the repository root only
after checking availability and naming the approved H200 explicitly:

```bash
STOP_AFTER=sanitizer sbatch --wait --nodelist=<approved-h200> \
  scripts/run_lns_strategy_benchmark_h200.sbatch
```

The hard job limit is eight hours.  `STOP_AFTER` accepts `sanitizer`, `smoke`,
`screen`, or `full`.  Start at `sanitizer`: it builds only `lns4_r1` -- the one
target that exercises both the cooperative loader and the warp-register
fraction LUT -- and synchronisation-checks it in a few minutes, so a warp
deadlock cannot consume a full benchmark allocation.  Then `smoke`, so compile
and correctness failures are caught before an expensive sweep.

Every benchmark binary runs under a per-target wall clock: `SMOKE_TIMEOUT`
(600 s), `SCREEN_TIMEOUT` (1800 s), and `FULL_TIMEOUT` (5400 s).  If one
expires, the run stops and names the target; the last line of that target's
`stdout_<target>.txt` is the variant that hung.

## Run 20260815T194131Z (job 443870) -- failed

The first attempt deadlocked in
`lns4_r1/fp32/ordinary/dense/scalar/x1/fraction_lut_warp` six minutes into the
run and held the GPU idle until the eight-hour limit killed it.  `warp_lookup`
branched on a per-lane index before its `__shfl_sync`, so the lanes that took
the branch never reached the shuffle and the rest waited for them forever.
Decoder validation passed immediately beforehand because a four-bit domain is
only sixteen cases, which never produces the partial warp that triggers it.
The fix, the widened validation launch, the synccheck stage, and the per-target
timeouts all landed together.
