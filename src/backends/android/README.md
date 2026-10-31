# Android backend

Each platform lives in its own folder under `src/backends/` and plugs into the shared engine front:

- `src/engine/`: backend-neutral. `engine.hpp` is the interface every engine implements, `registry.hpp` picks the engine
  for a model path, and `model.cpp` is `tqllm::Model` (the server and tools only talk to it).
- `src/backends/cuda/`: `builtin_cuda.cpp` lists the CUDA engines; the CUDA loaders live beside it.
- `src/backends/metal/`: `builtin_mlx.cpp` lists the Metal engine (`engine_mlx.cpp`).
- `src/backends/android/` (this folder): the Android engine (Vulkan / OpenCL / CPU) goes here the same way: an `Engine`
  subclass, a `builtin_android.cpp` with `register_builtin_engines()`, and a CMake option that compiles this folder
  instead of the CUDA or Metal one. The Android work on the `feat/android-*` branches moves here when it merges.
