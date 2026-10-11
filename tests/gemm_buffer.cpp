// gemm_buffer_plan: a capped buffer must still hold every matrix that cannot be cut into row blocks (TQ3_4S on a
// 16 GB card: the fit set gemm_rows 2048 and every request failed with "capped prefill GEMM buffer needs a
// row-sliceable format").
#include "gemm_buffer.hpp"
#include <cstdio>

int main() {
    int bad = 0;
    auto expect = [&](const char * what, bool ok) { if (!ok) { std::fprintf(stderr, "FAIL %s\n", what); ++bad; } };
    auto holds_unsliceable = [](const tqllm::GemmBufferPlan & p, const std::vector<tqllm::GemmMat> & mats) {
        for (const auto & m : mats)
            if (!m.row_sliceable && m.rows * m.cols > p.elems) return false;
        return true;
    };
    // Qwen3.8-27B shapes: gate/up 17408x5120, down 5120x17408, q 12288x5120
    const std::vector<tqllm::GemmMat> tq3 = {{17408, 5120, false}, {5120, 17408, false}, {12288, 5120, false}};
    const std::vector<tqllm::GemmMat> q4k = {{17408, 5120, true}, {5120, 17408, true}, {12288, 5120, true}};
    const std::vector<tqllm::GemmMat> mixed = {{17408, 5120, true}, {5120, 17408, true}, {12288, 5120, false}};

    const auto uncapped = tqllm::gemm_buffer_plan(0, 17408, tq3);
    expect("no cap: whole largest matrix", uncapped.block_rows == 0 && uncapped.elems == size_t(17408) * 5120);
    expect("TQ3_4S with the fit's 2048-row cap still holds every matrix", holds_unsliceable(tqllm::gemm_buffer_plan(2048, 17408, tq3), tq3));
    expect("mixed formats: unsliceable matrix still fits", holds_unsliceable(tqllm::gemm_buffer_plan(2048, 17408, mixed), mixed));
    const auto k = tqllm::gemm_buffer_plan(2048, 17408, q4k);
    expect("row-sliceable formats are capped", k.block_rows == 2048 && k.elems == size_t(2048) * 17408);
    expect("cap is at least 256 rows", tqllm::gemm_buffer_plan(100, 17408, q4k).block_rows == 256);
    const auto all_whole = tqllm::gemm_buffer_plan(2048, 17408, tq3);
    expect("all unsliceable: no row blocks", all_whole.block_rows == 0 && all_whole.elems == size_t(17408) * 5120);
    const auto mix = tqllm::gemm_buffer_plan(2048, 17408, mixed);
    expect("mixed: sliceable matrices still blocked", mix.block_rows == 2048 && mix.elems == size_t(12288) * 5120);
    if (bad) return 1;
    std::puts("gemm_buffer: ok");
    return 0;
}
