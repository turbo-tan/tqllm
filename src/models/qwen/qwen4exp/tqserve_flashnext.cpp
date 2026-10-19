// Flash-Next behind the tqllm Engine interface (tqserve), registered as the .tqfn model type.
#include "fn_engine.hpp"

namespace {
using tqllm::flashnext::FlashNextEngine;
std::unique_ptr<tqllm::Engine> make_flashnext(const std::string & manifest, const std::string &, uint32_t context, int slots,
                                              const tqllm::EngineOptions & o) {
    if (o.prefill_chunk > 0) { g_tune.chunk = o.prefill_chunk; g_tune.chunk_set = true; }
    if (o.prefill_sub > 0) g_tune.sub = o.prefill_sub;
    if (o.prefill_stage > 0) { g_tune.stage = o.prefill_stage; g_tune.stage_set = true; }
    if (o.prefill_reserve_mb >= 0) g_tune.reserve_mb = o.prefill_reserve_mb;
    if (o.prefill_arena_mb >= 0) g_tune.arena_mb = o.prefill_arena_mb;
    if (o.dense_cache_mb >= 0) { g_tune.dense_cache_mb = o.dense_cache_mb; g_tune.dense_cache_set = true; }
    if (o.prefill_borrow == "all") g_tune.borrow = tqllm::BorrowMode::All;
    else if (!o.prefill_borrow.empty() && o.prefill_borrow != "auto") { g_tune.borrow = tqllm::BorrowMode::Count; g_tune.borrow_count = std::atoi(o.prefill_borrow.c_str()); }
    return std::make_unique<FlashNextEngine>(manifest, context, slots);
}
const bool registered_flashnext = (tqllm::register_flashnext_engine(&make_flashnext), true);
}  // namespace
