# CUDA kernels

C++/CUDA runtime kernels will be added after pinned Qwen reference fixtures exist.
No inherited Rust build.rs or mandatory TQ3_4S kernel port is part of this design.
Use cuBLAS/cuBLASLt initially where appropriate; custom code must pass numerical gates.
