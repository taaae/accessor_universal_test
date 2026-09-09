# Independent review

The fresh-context read-only Sol review ran before cluster submission. It found
five material issues in the first draft: an undersized FP32 GEMV output buffer,
Release-mode assertions that compiled away, no GPU result comparison, an
insufficient SASS dataflow audit, and no conditional K=48/64 extension stage.

The implementation worker fixed the buffer and host-test defects, added bitwise
GPU decoder comparisons for every family and K through 64, exercised timed DOT
and GEMV kernels under sanitizer validation, and added conditional extensions
with fresh stage baselines. The SASS audit remains a hard preflight gate and
will be adapted only from saved real Hopper SASS if its fail-closed parser
rejects an otherwise valid specialization.
