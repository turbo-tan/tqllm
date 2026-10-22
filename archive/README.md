# Archive: not built, do not restore

Files here are kept for history only. Nothing in the build, tests or scripts may reference them
(`tests/test_layering.py` fails if one does).

- `tools/tqfn_stage.cpp`: the Flash-Next bring-up monolith (fixture replays, chain KLD, offline decode). The engine moved to
  `src/models/qwen/qwen4exp/fn_*` and is served by tqserve; expert packing moved to `tools/tqfn_pack.cpp` (`tqfn-pack`).
  New Flash-Next work goes into the engine components, never back into this file.
