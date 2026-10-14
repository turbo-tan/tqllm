## What and why

<!-- One change per PR. Link the issue it addresses. -->

## Exactness

<!-- Same greedy tokens as before (name the test), or the measured quality cost (KLD vs BF16 and a task score). -->

## Measurements

<!-- For speed or memory changes: base commit and this branch, same GPU, same session.
GPU, model and drafter files, context, prompt type, and the results.jsonl lines from bench/relbench.py. -->

## Checklist

- [ ] Builds; `ctest` passes (CPU build at least, GPU build for kernel or engine changes)
- [ ] New behaviour has a test; a bug fix has a test that failed before the fix
- [ ] New environment flags are in `config/flags.yaml` (docs regenerated)
- [ ] Borrowed ideas credited in README; copied code listed in NOTICE
- [ ] AI assistance disclosed if it wrote a meaningful part of the change
