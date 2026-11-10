#pragma once
// Tensor parallelism runs on NCCL (CUDA) / RCCL (HIP). The HIP build leaves TQLLM_NCCL_DIR unset in this
// iteration and links tp_comm_stub.cpp; including <nccl.h> under HIP is a deliberate build error until RCCL
// gets its own CMake hook (same pattern as the NCCL one: -DTQLLM_RCCL_DIR).
#error "nccl.h is CUDA-only in tqLLM: tensor parallelism on AMD needs RCCL (a later PR). Single-GPU builds do not include this header."
