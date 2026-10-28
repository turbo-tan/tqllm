#pragma once
// Greedy speculative decoding for the native Android backend. Proposals come from earlier text (the tokens
// that followed the latest repeat of the last few tokens, free) or from a small draft model of the same
// vocabulary. The target checks a proposal in one batched pass over its weights and keeps the longest agreeing
// prefix plus its own next token. Batched logits are bit-identical to single-token logits, so the emitted tokens
// are exactly the target's greedy tokens.
//
// Each step picks plain decoding, a text proposal or a draft proposal, and its length, by expected tokens per
// millisecond: acceptance rates and the costs of a verify, each extra verified token and each draft token are
// measured while generating, so a proposal source that does not pay off (prose for a weak draft, a hot phone)
// switches itself off.
#include "android_model.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <vector>

namespace tqllm::android_backend {
struct SpeculativeStats { int verifies = 0, drafted = 0, accepted = 0, text_drafted = 0, text_accepted = 0; };

inline uint32_t greedy(const std::vector<float> & logits) {
    return uint32_t(std::max_element(logits.begin(), logits.end()) - logits.begin());
}

class Speculator {
public:
    // history: every token both models have consumed (the prompt so far); extended as tokens are accepted.
    Speculator(Model & target, Model * draft, int depth, std::vector<uint32_t> & history)
        : target_(target), draft_(draft), depth_(std::max(depth, 0)), history_(history), synced_(history.size()) {}

    // `next` is the target's chosen token, not yet consumed. emit(token) returns false to stop.
    // Returns "eos", "length" or "stopped".
    const char * run(uint32_t next, int budget, const std::function<bool(uint32_t)> & emit, SpeculativeStats & stats) {
        int produced = 0;
        for (;;) {
            if (next == target_.eos()) return "eos";
            if (produced >= budget) return "length";
            if (!emit(next)) return "stopped";
            if (++produced >= budget) return "length";
            const int room = std::min(target_.context() - target_.position() - 1, budget - produced);
            if (room < 1) return "length";
            const int limit = std::min(depth_, room - 1);
            // Candidate proposals and their expected rate; plain decoding is the baseline.
            std::vector<uint32_t> text = limit > 0 ? lookup(next, limit) : std::vector<uint32_t>{};
            int text_k = 0, draft_k = 0;
            double best = rate(0, 0.0, 0.0);
            for (int k = 1; k <= int(text.size()); ++k)
                if (rate(k, text_p(), 0.0) > best) { best = rate(k, text_p(), 0.0); text_k = k; }
            if (draft_ && limit > 0) {
                for (int k = 1; k <= limit; ++k)
                    if (rate(k, draft_p(), draft_cost()) > best) { best = rate(k, draft_p(), draft_cost()); draft_k = k; text_k = 0; }
                // Keep the draft estimate fresh: probe one token after a long plain or text-only stretch.
                if (!draft_k && ++idle_ > 24) draft_k = 1, text_k = 0;
            }
            std::vector<uint32_t> proposal;
            const bool from_draft = draft_k > 0;
            if (from_draft) {
                idle_ = 0;
                sync_draft();
                draft_->checkpoint();
                const auto before = Clock::now();
                uint32_t token = next;
                for (int i = 0; i < draft_k; ++i) {
                    token = greedy(draft_->forward(token));
                    proposal.push_back(token);
                    if (token == target_.eos()) break;
                }
                draft_ms_ = blend(draft_ms_, ms(before) / double(proposal.size()));
            } else {
                proposal.assign(text.begin(), text.begin() + text_k);
            }
            std::vector<uint32_t> batch{next};
            batch.insert(batch.end(), proposal.begin(), proposal.end());
            if (!proposal.empty()) target_.checkpoint();
            const auto before = Clock::now();
            const auto & logits = target_.forward_batch(batch);
            int accepted = 0;
            while (accepted < int(proposal.size()) && greedy(logits[accepted]) == proposal[accepted]) ++accepted;
            const uint32_t bonus = greedy(logits[accepted]);
            measure(int(batch.size()), ms(before));
            ++stats.verifies;
            if (!proposal.empty()) {
                target_.rollback(accepted + 1);
                const int tested = accepted + (accepted < int(proposal.size()) ? 1 : 0);
                if (from_draft) {
                    stats.drafted += int(proposal.size()); stats.accepted += accepted;
                    draft_hits_ = 0.9 * draft_hits_ + accepted; draft_tries_ = 0.9 * draft_tries_ + tested;
                    // The draft consumed next and proposal[0 .. size-2]; keep next and the accepted prefix.
                    const int consumed = int(proposal.size());
                    draft_->rollback(std::min(accepted + 1, consumed));
                    synced_ = history_.size() + std::min(accepted + 1, consumed);
                } else {
                    stats.text_drafted += int(proposal.size()); stats.text_accepted += accepted;
                    text_hits_ = 0.9 * text_hits_ + accepted; text_tries_ = 0.9 * text_tries_ + tested;
                }
            }
            history_.push_back(next);
            history_.insert(history_.end(), proposal.begin(), proposal.begin() + accepted);
            for (int i = 0; i < accepted; ++i) {
                if (proposal[i] == target_.eos()) return "eos";
                if (!emit(proposal[i])) return "stopped";
                if (++produced >= budget) return "length";
            }
            next = bonus;
        }
    }
    // Feed the draft every history token it has not seen (it is fed lazily, only when it is about to draft).
    void sync_draft() {
        if (!draft_) return;
        constexpr size_t chunk = 32;
        while (synced_ < history_.size()) {
            const size_t end = std::min(history_.size(), synced_ + chunk);
            draft_->forward_batch(std::vector<uint32_t>(history_.begin() + synced_, history_.begin() + end), false);
            synced_ = end;
        }
    }
    // The caller reset both models.
    void restart() { synced_ = history_.size(); }

private:
    using Clock = std::chrono::steady_clock;
    static double ms(Clock::time_point since) { return std::chrono::duration<double, std::milli>(Clock::now() - since).count(); }
    static double blend(double old, double sample) { return old < 0 ? sample : 0.8 * old + 0.2 * sample; }
    double draft_cost() const { return draft_ms_ >= 0 ? draft_ms_ : 0.25 * (single_ms_ > 0 ? single_ms_ : 100.0); }
    double text_p() const { return std::min(0.95, (text_hits_ + 1.0) / (text_tries_ + 2.0)); }
    double draft_p() const { return std::min(0.95, (draft_hits_ + 1.0) / (draft_tries_ + 2.0)); }
    // Expected tokens per millisecond for a proposal of k tokens accepted position by position with rate p.
    double rate(int k, double p, double per_draft) const {
        const double single = single_ms_ > 0 ? single_ms_ : 100.0;
        const double extra = extra_ms_ >= 0 ? extra_ms_ : 0.3 * single;
        const double tokens = k ? (1.0 - std::pow(p, k + 1)) / (1.0 - p) : 1.0;
        return tokens / (single + k * (extra + per_draft));
    }
    void measure(int size, double elapsed) {
        if (size == 1) single_ms_ = blend(single_ms_, elapsed);
        else if (single_ms_ > 0) extra_ms_ = blend(extra_ms_, std::max(0.0, (elapsed - single_ms_) / (size - 1)));
    }
    // Tokens that followed the latest earlier occurrence of the last 3 (else 2) tokens, ending with `next`.
    std::vector<uint32_t> lookup(uint32_t next, int limit) const {
        std::vector<uint32_t> tail(history_.end() - std::min<size_t>(history_.size(), 2), history_.end());
        tail.push_back(next);
        for (int n = std::min<int>(3, int(tail.size())); n >= 2; --n) {
            const uint32_t * key = tail.data() + tail.size() - n;
            // Search history + [next] for an earlier match whose continuation is known.
            for (int start = int(history_.size()) - n; start >= 0; --start) {
                if (!std::equal(key, key + n, history_.begin() + start)) continue;
                std::vector<uint32_t> out;
                for (int i = start + n; i < int(history_.size()) && int(out.size()) < limit; ++i) out.push_back(history_[i]);
                if (int(out.size()) < limit && start + n == int(history_.size())) out.push_back(next);
                if (!out.empty()) return out;
            }
        }
        return {};
    }
    Model & target_;
    Model * draft_;
    int depth_;
    std::vector<uint32_t> & history_;
    size_t synced_;
    double single_ms_ = -1, extra_ms_ = -1, draft_ms_ = -1;
    double text_hits_ = 0, text_tries_ = 0, draft_hits_ = 0, draft_tries_ = 0;
    int idle_ = 0;
};
}
