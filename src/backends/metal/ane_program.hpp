// TQLLM's own private-API W8A8 Neural Engine FFN program. Borrows the MIL program structure, the int8 dequant-by-1/128
// convention, the tanh-SiLU, the intermediate Hadamard rotation + per-token requantize, and the private-client
// compile/load/evaluate flow from Splash (Apache-2.0: runtime/ops/AneFfn.cpp, runtime/ane/Program.mm), simplified to the
// one FFN shape. Step 1 drives it from the CPU (bind_weights / evaluate). Step 2 adds the GPU-side pack/join/weight
// requantize (into the program's IOSurfaces, zero-copy through MLX) and an asynchronous evaluate the GPU can overlap.
#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// Step-2 GPU methods take and return MLX arrays and read NaxAffine weights; forward-declared so step-1 users need no MLX.
namespace mlx::core {
class array;
}

namespace tqllm::mlx_backend {

struct NaxAffine;  // nax_affine.hpp

// One W8A8 FFN slice on the Neural Engine: y = down(silu(x Wg^T) * (x Wu^T)) over C intermediate channels, hidden H,
// for a fixed row count. The program (one MIL function) is compiled and loaded once through the private AppleNeuralEngine
// client; each layer's gate/up rows [C, H] and down columns [H, C] (FP16 host arrays) are quantized to int8 + per-row
// scales once at bind; evaluate() packs the FP16 [rows, H] input to per-token int8 on the CPU, runs one ANE evaluation,
// and joins the per-token-scaled FP32 result back to FP16 [rows, H]. Blocking.
class AneProgram {
public:
    // Build the program for (columns C, rows, hidden H) and reserve `layers` weight slots. H defaults to Qwen3's 5120.
    // Throws std::runtime_error if the Neural Engine private interface is unavailable or the program fails to compile/load.
    static std::unique_ptr<AneProgram> open(int columns, int rows, int hidden = 5120, int layers = 1);
    virtual ~AneProgram() = default;

    virtual int rows() const = 0;
    // The GPU path runs a slice of n tokens on the smallest compiled function with at least n rows. row_counts() lists the
    // function sizes (512 up to rows() in steps of 256); fit_rows(n) is that size, or -1 if n > rows(). pack_input_gpu
    // takes exactly one of these sizes (the caller pads), join_output_gpu returns that many rows.
    virtual std::vector<int> row_counts() const = 0;
    virtual int fit_rows(int n) const = 0;
    virtual int columns() const = 0;   // C intermediate channels
    virtual int hidden() const = 0;    // H
    virtual bool has(int layer) const = 0;

    // Quantize and store layer `layer`'s weights: gate/up rows row-major [C, H], down columns row-major [H, C], FP16.
    // Pointers need not stay alive after the call. Throws on a bad layer index or shape.
    virtual void bind_weights(int layer, const void* wg, const void* wu, const void* wd) = 0;

    // Blocking one-layer evaluation: x is row-major FP16 [rows, H], y is row-major FP16 [rows, H] (overwritten).
    // Buffers must stay alive for the call. Throws if the layer is unbound or the ANE evaluation fails.
    virtual void evaluate(int layer, const void* x, void* y) = 0;

    // The Neural Engine evaluation time of the last evaluate() in milliseconds (its ANE eval alone, without the CPU
    // input pack and output join). For the speed comparison against Splash's ANE time.
    virtual double last_eval_ms() const = 0;

    // =================================================================================================================
    // Step 2: GPU-side pack / join / weight requantize into the program's IOSurfaces (zero-copy through MLX) and an
    // asynchronous evaluate the GPU can overlap. The ANE reads and writes the same IOSurfaces the GPU kernels wrote.
    // All MLX arrays live on the GPU; the caller evaluates them on MLX's default stream.
    // =================================================================================================================

    // Weight slots for the GPU path, double-buffered so layer l+1 is requantized while layer l evaluates. `slots`
    // (>= 1; 2 for double buffering) is separate from the step-1 `layers`. Call once before requantize_weights_gpu.
    // No-op if already reserved with >= `slots`.
    virtual void reserve_slots(int slots) = 0;

    // Requantize one weight slot from NaxAffine on the GPU: gate/up are the last `share` rows of wg_full/wu_full
    // ([F, H] each, F >= share == columns()), down is the last `share` columns of wd_full ([H, F]); down is pre-rotated
    // by the intermediate Hadamard as in bind_weights. Dequant is nax_affine_dequant + a quantize kernel. The int8
    // codes and per-row scales are written into slot `slot`'s weight surfaces. Caller must mx::eval the returned array
    // (a zero-size handle that depends on the writes) before enqueue(slot). Numerics equal bind_weights on the same
    // FP16 weights. Throws on a bad slot or shape.
    virtual mlx::core::array requantize_weights_gpu(int slot, const NaxAffine& wg_full, const NaxAffine& wu_full,
                                                    const NaxAffine& wd_full, int share) = 0;

    // Pack a GPU FP16 input x16 [rows, H] (contiguous) to per-token int8 + per-token scale directly into the input
    // IOSurfaces, on the GPU. Returns a token-scale handle [rows] FP16 (an MLX array over the tx surface) that
    // join_output_gpu takes back. Caller must mx::eval the returned array before enqueue. Numerics equal the CPU
    // packInput bit for bit. Throws on a bad shape.
    virtual mlx::core::array pack_input_gpu(const mlx::core::array& x16, int parity = 0,
                                            const mlx::core::array* after = nullptr) = 0;

    // Join the program's output surface [H+1, rows] with the per-token scale from pack_input_gpu into y [rows, H]
    // FP32 (the ANE's partial, to be added to the GPU share's down output by the caller), on the GPU. `token_scale`
    // is the handle pack_input_gpu returned. Must be called after the ANE evaluation of this slice finished (wait()
    // or the shared-event ordering). Numerics equal the CPU joinOutput. Throws if the output is not ready.
    virtual mlx::core::array join_output_gpu(const mlx::core::array& token_scale, int parity = 0) = 0;

    // Start the ANE evaluation of weight slot `slot` asynchronously and return without blocking. The input surfaces
    // must already hold this slice's packed input (pack_input_gpu + mx::eval) and the slot's weights must be staged
    // (requantize_weights_gpu + mx::eval). `after_pack` is accepted for the step-3 interface but, because MLX 0.32.3
    // does not export mx::Event, the ANE cannot wait on a shared event MLX signals; it always starts at once, so the
    // caller must have evaluated the pack first. The overlap with the GPU's complementary share still holds (the ANE
    // is a separate engine and enqueue returns immediately). Pairs with wait(). Throws if the slot is unstaged.
    virtual void enqueue(int slot, bool after_pack, int parity = 0) = 0;

    // The intended GPU -> ANE handoff (the ANE waits on an MTLSharedEvent MLX signals on its stream after the pack) is
    // Step-3 GPU -> ANE handoff without a CPU wait: call after mx::async_eval of the pack_input_gpu handle (not yet
    // finished). MLX signals an event at the end of the command buffer that ran the pack; the next enqueue(slot, true,
    // parity) makes the ANE wait on it. Needs an MLX that exports mx::Event (built with MLX_EXPORT_ALL_SYMBOLS).
    virtual void ready_after(const mlx::core::array& evaluated_pack) = 0;

    // Block until the evaluation started by the last enqueue() completed. Throws if it failed. Sets last_eval_ms().
    virtual void wait(int parity = 0) = 0;
    // Makes MLX's GPU stream wait for the evaluation enqueued on `parity` (the ANE raises a shared event when done), so
    // GPU work encoded next (the join) runs after it without the CPU waiting. Call wait(parity) later, before the
    // parity is enqueued again, to collect the result and surface a failure.
    virtual void wait_gpu(int parity) = 0;

    // Test-only: copy the current tx (per-token scale) surface [rows] and the first x input segment surface
    // (int8 [segWidth, rows], row-padded) out to host vectors, for comparing the GPU pack to the CPU pack.
    virtual void debug_copy_pack(std::vector<uint16_t>& tx, std::vector<int8_t>& x0, int& x0_rows, int& x0_stride) = 0;

    // Test-only: dump staged weight scales (sg/sd) and first wg/wd code blocks from the CPU path (slot < 0, shared
    // surfaces) or a GPU slot, to compare GPU requantize to CPU bind.
    virtual void debug_copy_weights(int slot, std::vector<uint16_t>& sg, std::vector<uint16_t>& sd,
                                    std::vector<int8_t>& wg0, std::vector<int8_t>& wd0, int& wg0_rows, int& wg0_stride,
                                    int& wd0_rows, int& wd0_stride) = 0;
};

// test hook: the input rotation off compares the ANE share without it (on by default; read when a program is built)
void ane_program_set_input_rotate(bool on);
} // namespace tqllm::mlx_backend
