# Cold expert read queue

Private Flash-Next hot/current + cold/Q2 experiment. Manifest option
`cold_prefetch_k=0..8` defaults to zero (disabled). Requires `cold_mmap=1`
and at least two `cold_readers`. Owner: F35 performance lane. Review after
its first matched real-model screen; keep disabled unless that gate passes.

Predicted next-layer IDs are ranked by frequency across verify rows, then
summed top-k rank, then expert ID. Only cold, nonresident experts enter the
queue. This changes read scheduling, never actual router selections or weights.
Prediction runs only within the existing TQFN_PREDICT_N row limit.

The speculative queue holds at most twice k records, retains only adjacent
layers, and runs one speculative read at a time. Actual demands promote
queued work, join duplicate in-flight reads, and take precedence over starting
speculative work. An already-running speculative read is not interrupted.
The caller joins ordered futures; workers never wait on their own pool.
Payloads retain mapped ownership and undergo normal integrity checks.

This is host-read prefetch, not speculative GPU-slot replacement. Existing
pinned staging and CUDA slot-completion events remain authoritative.
TQLLM_DEBUG=mem emits COLD_QUEUE counters per decode. Prefetched, used,
and expired are distinct: a prediction is not a useful hit merely because
it was submitted, and counters can span the prefill/decode boundary.

Gate: expert_fetch_queue tests demand priority, promotion, duplicate sharing,
expiry, bounded admission, failure recovery and prediction ranking. Removing
demand priority must fail the test. Then run existing cache/lifetime/Q2 tests
and matched same-binary k=0/2 code/prose screens, checking exact output/counts,
physical I/O, queue utilization and warm/cold timings separately.

Separate opt-in `cold_overlap=1` (default zero) defers the host wait after the
last hot wave, so preparation of the separate cold bank can overlap it. Waits
between waves reusing the same bank and all final cold-wave waits stay intact.
No CUDA arithmetic order, assignment layout, or slot-release event changes.
This requires its own real-model exactness and same-binary timing gate; it does
not depend on enabling prediction. Owner: F35 performance lane; review after
first screen, keep disabled unless validated.

`cold_overlap=2` also queues the independent shared expert just before cold
prepare, reusing the already-converted FP16 token input. Its output buffer is
separate from routed experts. Final combination and each expert's arithmetic
are unchanged. Mode1 keeps shared-expert scheduling as before, so mode1/2
isolate this change. Mode2 requires its own exact output and speed screen.

`cold_streaming=1` (default0, requires mapped records and at least2 readers)
consumes ordered read futures incrementally for known row-layout staged banks.
The first ready record starts normal pinned staging/DMA while later checksums
run. Initial bank creation retains the synchronous validated source path.
Sorted victim selection and all integrity/reuse guards remain unchanged. A
later read failure drains partial DMA before rethrowing, so staging is safe for
the next request. This option has a separate unit and real-model gate.
