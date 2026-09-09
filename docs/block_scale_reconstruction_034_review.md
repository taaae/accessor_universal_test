# Independent review record

The read-only reviewer inspected the full experiment-034 contract and the
actual implementation before cluster work. It found no CUDA barrier, tail, or
ownership defect. The deferred outer loop is CTA-uniform, each block guard is
warp-uniform, lane-specific tail guards cover loads only, and all threads reach
the final CTA reduction. The B128 mapping retains one dependent four-product
subtotal for every lane and applies the scale product on every valid lane.

The first review found fail-open checks in the SASS parser and result validator.
Synthetic assembly could bypass the required FP32 multiply-to-widen chain,
substitute unrelated widens in FP64 reconstruction or local deferred code, use
FTZ opcodes, overwrite an intermediate register, or feed baselines and reducers
from unrelated registers. A set-only inventory also hid duplicate kernels.
Synthetic CSVs could use empty provenance, omit a required drift rerun, or add
an unqualified rerun. The report accepted substring correctness markers and did
not bind the GPU environment.

The implementation now checks last-writer chains through FMUL, F2F, DMUL and
DFMA where applicable. It checks baseline and reducer load lineage, rejects FTZ,
and requires exactly 16 unique timed symbols. The validator requires shaped,
nonempty commit and digest provenance and enforces drift if and only if a rerun
exists. The analyzer parses exact correctness records for all 17 datasets,
requires 238 cases, checks a unique 16-row audit, and binds the environment file
by SHA-256. Known FP32 product bits are tested for both operands. The benchmark
checks and records long-double precision by mantissa digits.

After these fixes, the reviewer reran every supplied counterexample. All failed
for the intended reason. It also reproduced 32 passing Python tests, the Release
`-O3 -DNDEBUG` host test, shell syntax checks, and a clean diff check. The source
review found no blocker before zero-GPU preflight.

The narrow checker cannot prove from a generic opcode sequence that the local
deferred body runs four rounds for every lane rather than once under a leader
predicate. The successful preflight must therefore retain a hash-bound manual
inspection of the actual Hopper loop and predicate dataflow. The checker alone
must not be described as proof of that property.

The reviewer then inspected the actual Hopper artifact independently. The
review is bound to SASS SHA-256
`e01239c142bf3e5b59cb2b529e46ac2b5e57ad31de6b20a660908da8e0e51d32`,
which both preflights `460481` and `460482` reproduced. It confirmed all 16
symbols, argument roles, scalar load pairing, FP32 and FP64 reconstruction
chains, baseline and reducer lineage, and the absence of spills and forbidden
operations. For both deferred kernels it traced the warp ownership and tail
predicates, the loop counter from zero through four dynamic rounds, the single
loop-carried subtotal, the warp-uniform invalid-block branch, and unpredicated
scale application by every valid lane. No leader predicate or processing-region
shuffle was present. The reviewer reran the hardened audit and all 32 tests,
normally and under `python -O`, and reported no material blocker.
