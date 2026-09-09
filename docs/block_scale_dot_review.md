# Independent review record

The read-only reviewer inspected the full experiment contract and the actual
implementation before cluster submission. It found the CUDA barrier and
subgroup structure sound: every lane reaches each shuffle and CTA barrier,
width-16 partitions a warp into two subgroups, tail loads and scale loads are
guarded, B128 uses four scalar rounds and one subtotal, and both operands use
independent scales with FP64 reconstruction.

The initial review found four material gaps. The first SASS checker counted
whole-function opcodes without following loaded values. Validation stopped
before a second B128 deferred tile. Its error allowance scaled with N rather
than reduction depth and lacked an all-positive fixture. Drift reruns were not
rechecked or reported. It also found incomplete manifest checks, audit binding,
and report detail, plus silent smoke timing overrides.

The implementation now validates N=2^20 against independent CPU references,
including a second deferred tile and multiple generic grid-stride iterations.
It adds an all-positive fixture, uses a gamma bound based on reduction depth,
rechecks rerun drift, rejects conflicting smoke dimensions, validates the full
manifest identity, binds the audit hash, checks the exact 17-kernel audit
inventory, and reports drift plus every median, quartile and baseline ratio.
The SASS checker now traces load provenance through widening, DMUL and DFMA and
audits both final-reduction kernels. Follow-up counterexamples showed that this
structural gate still cannot infer pointer roles, exclude an intervening
arithmetic change, prove all shuffles feed the deferred subtotal, or prove
baseline operand consumption. The zero-GPU preflight will collect actual Hopper
SASS. GPU smoke remains blocked until a saved specialization-by-specialization
dataflow and predicate inspection closes these points, with checker hardening
where the concrete instruction form permits it.

## Hopper SASS inspection after preflight 460435

The first zero-GPU compile produced all 17 timed kernels with no stack spills or
device calls. CUDA parameter slots map q-left, q-right, scale-left and
scale-right to constant offsets `0x210`, `0x218`, `0x220` and `0x228`. In every
per-element specialization, address calculations from those four slots feed
four scalar loads in that order. The machine code widens the values, computes
`DMUL(q-left, scale-left)` and `DMUL(q-right, scale-right)`, and immediately
feeds both products to the loop-carried DFMA. FP32-scale variants use four FP32
loads and four widens; FP64-scale variants use two FP32 payload loads, two FP64
scale loads and two payload widens.

Every deferred specialization loads the two payloads, widens them and feeds
them to the block-subtotal DFMA. Hopper represents each FP64 shuffle as two
32-bit `SHFL.DOWN` instructions. B16 has paired offsets 8, 4, 2 and 1 with
control `0x101f`; B32 and B128 have paired offsets 16, 8, 4, 2 and 1 with
control `0x1f`. The DADD chain consumes each shuffled pair. Scale-left and
scale-right loads, their widening when stored as FP32, the scale DMUL and the
final subtotal DFMA share the same subgroup-leader predicate. That final DFMA
consumes the completed DADD subtotal and scale-product registers. B128 retains
one loop-carried payload DFMA across four scalar rounds before the shuffle
sequence.

The raw-FP32 loop has two scalar FP32 loads consumed by one FFMA. The widened
FP32 loop has two scalar loads, two widens and one consuming DFMA. Raw FP64 has
two FP64 loads consumed by one DFMA. The two final reducers each load one
partial per loop iteration, accumulate with FADD or DADD, execute the common
shared-memory tree, and store a distinct final result. Across all timed kernels
the SASS contains no division, vectorized payload load, local-memory spill or
device call. The saved final audit will bind these findings to the successful
preflight SASS hash; job 460435 itself remains a failed checker calibration run.
