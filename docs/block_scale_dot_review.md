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
