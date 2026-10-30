#pragma once
// Draft-ahead speculative decoding (greedy): each step submits verify(k) and, chained behind it on the GPU, the
// drafter's inject of verify(k)'s accepted feature rows and draft(k+1), all from device values (accepted count, next
// anchor, next position), then waits only for verify(k)'s choices. The host's accept / output / graph building for
// step k+1 runs while the GPU drafts, so the GPU does not idle between verify and the next draft. Exact: the same
// verify and accept as the synchronous loop; only the draft's timing moves.
#include "dflash.hpp"
#include "qwen35.hpp"
#include <mlx/mlx.h>
#include <chrono>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <vector>

namespace tqllm::mlx_backend {
class DraftAhead {
public:
    DraftAhead(Qwen35& engine, DFlash2& drafter, int width) : engine_(engine), drafter_(drafter), width_(width) {
        if (width < 1 || width > 15) throw std::invalid_argument("draft-ahead width must be 1..15");
    }
    // after the prefill and its host inject: the first draft (`width` tokens, 1..width) from the pending token at the
    // engine's position
    void start(uint32_t pending, int width = 0) {
        anchor_ = mlx::core::array(pending);
        position_ = mlx::core::array(engine_.position(), mlx::core::int32);
        drafted_width_ = clamp_width(width);
        drafted_ = drafter_.draft_device_at(*anchor_, *position_, drafted_width_);
        mlx::core::async_eval(*drafted_);
    }
    bool active() const {return drafted_.has_value();}
    // tokens in the draft waiting for the next step (the verify width cannot exceed it)
    int drafted_width() const {return drafted_width_;}
    void stop() {anchor_.reset(); position_.reset(); drafted_.reset();}
    struct Step {
        std::vector<uint32_t> proposed;  // the verified drafted tokens
        std::vector<uint32_t> choices;   // the target's argmax per verify row (verified + 1)
        int accepted = 0;                // drafted tokens equal to the target's choice, from the first
        // host phases of this step (ms): graph build of verify(k), submit (encode of verify + next draft), wait for
        // verify(k), post (readback, accept, settle). GPU idle shows as build + submit beyond the previous draft's run
        double build_ms = 0, submit_ms = 0, wait_ms = 0, post_ms = 0;
    };
    // verify the first `verify` (1..drafted_width) drafted tokens; the next draft is `next_width` tokens (1..width,
    // 0: width): a 7-token draft runs the drafter's native 8-row block, which drafts better than the head of a 16-draft
    Step step(int verify, int next_width = 0) {
        namespace mx = mlx::core;
        if (!active()) throw std::logic_error("draft-ahead step before start");
        if (verify < 1 || verify > drafted_width_) throw std::invalid_argument("draft-ahead verify width must be 1..drafted width");
        next_width = clamp_width(next_width);
        using Clock = std::chrono::steady_clock;
        auto ms_since = [](Clock::time_point t) {return std::chrono::duration<double, std::milli>(Clock::now() - t).count();};
        const auto t0 = Clock::now();
        const int pos = engine_.position();
        auto drafted = verify == drafted_width_ ? *drafted_ : mx::slice(*drafted_, {0}, {verify});
        auto block = mx::concatenate({mx::reshape(*anchor_, {1}), drafted}, 0);
        Qwen35::Verification ticket;
        engine_.defer_eval(true);
        mx::array verified = mx::array(0.0f);
        try {verified = engine_.verify_device(block, ticket);} catch (...) {engine_.defer_eval(false); throw;}
        engine_.defer_eval(false);
        auto choices = mx::astype(mx::argmax(verified, -1), mx::uint32);
        auto equal = mx::astype(mx::equal(mx::slice(choices, {0}, {verify}), drafted), mx::int32);
        auto accepted = mx::sum(mx::cumprod(equal, 0), false);
        auto next_anchor = mx::take(choices, accepted);
        auto rows = mx::add(accepted, mx::array(1, mx::int32));
        // two submissions: the wait below covers verify(k) only. One async_eval of {choices, next} signals once, after
        // the next draft too, so the host would wait for the draft and build verify(k+1) with the GPU idle (j771: +2.7-3.8 ms)
        const auto t1 = Clock::now();
        mx::async_eval({choices, accepted, next_anchor, rows});
        drafter_.inject_device(engine_.feature_states(), rows, *position_);
        auto next_position = mx::add(*position_, rows);
        auto next = drafter_.draft_device_at(next_anchor, next_position, next_width);
        mx::async_eval(next);
        const auto t2 = Clock::now();
        mx::eval({choices, accepted, drafted});
        const auto t3 = Clock::now();
        Step out;
        out.build_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        out.submit_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
        out.wait_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
        out.accepted = accepted.item<int32_t>();
        out.proposed.assign(drafted.data<uint32_t>(), drafted.data<uint32_t>() + verify);
        out.choices.assign(choices.data<uint32_t>(), choices.data<uint32_t>() + verify + 1);
        if (engine_.position() != pos + verify + 1) throw std::logic_error("draft-ahead: verify moved the position unexpectedly");
        engine_.accept_prefix(ticket, size_t(out.accepted + 1));
        drafter_.settle_device_inject(out.accepted + 1);
        anchor_ = next_anchor; position_ = next_position; drafted_ = next; drafted_width_ = next_width;
        out.post_ms = ms_since(t3);
        return out;
    }
private:
    int clamp_width(int w) const {return w < 1 || w > width_ ? width_ : w;}
    Qwen35& engine_;
    DFlash2& drafter_;
    int width_, drafted_width_ = 0;
    std::optional<mlx::core::array> anchor_, position_, drafted_;
};
}  // namespace tqllm::mlx_backend
