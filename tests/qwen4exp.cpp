#include "qwen4exp.hpp"
#include <cstdio>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>

using namespace tqllm;
static void require(bool ok, const char * why) { if (!ok) throw std::runtime_error(why); }
static void rejects(const std::function<void()> & call) {
    try { call(); } catch (const std::runtime_error &) { return; }
    throw std::runtime_error("invalid configuration did not throw");
}
static GgufFile fixture() {
    GgufFile f; f.kv["general.architecture"].str = "qwen4exp";
    for (const auto & [key, value] : std::vector<std::pair<std::string, int>>{
        {"block_count", 4}, {"embedding_length", 2560}, {"expert_count", 512}, {"expert_used_count", 10},
        {"expert_feed_forward_length", 640}, {"expert_shared_feed_forward_length", 640},
        {"hyper_connection.count", 4}, {"hyper_connection.low_rank", 320}, {"ple.conv_kernel", 4},
        {"attention.indexer.head_count", 4}, {"attention.indexer.key_length", 128}, {"attention.indexer.top_k", 2048},
        {"embedding_length_per_layer_input", 160}, {"ple.ngram_size", 3}, {"ple.heads_per_ngram", 2},
        {"ple.eos_token_id", 99}, {"ple.image_token_id", 88}, {"full_attention_interval", 4}})
        f.kv["qwen4exp." + key].num = value;
    f.kv["qwen4exp.ple.layers"].arr = {1};
    f.kv["qwen4exp.ple.layer_multipliers"].arr = {3, 5, 7};
    f.kv["qwen4exp.ple.head_offsets"].arr = {0, 11, 24, 41};
    f.kv["qwen4exp.ple.head_vocab_sizes"].arr = {11, 13, 17, 19};
    f.kv["qwen4exp.attention.compress_ratios"].arr = {0, 4, 0, 4};
    return f;
}
int main() {
    try {
        const auto valid = fixture(); const auto c = qwen4exp_config(valid);
        require(c.hidden == 2560 && c.hc == 4 && c.experts == 512 && c.topk == 10 &&
                c.index_heads == 4 && c.index_dim == 128 && c.index_topk == 2048 &&
                c.ple_layers == std::vector<int>{1},
                "valid architecture dimensions differ");
        for (const char * key : {"block_count", "expert_count", "hyper_connection.count", "hyper_connection.low_rank",
                                 "ple.conv_kernel", "attention.indexer.head_count", "attention.indexer.key_length",
                                 "attention.indexer.top_k"}) {
            auto bad = valid; bad.kv[std::string("qwen4exp.") + key].num = 0;
            rejects([&] { qwen4exp_config(bad); });
        }
        for (const char * key : {"ple.layer_multipliers", "ple.head_offsets", "ple.head_vocab_sizes", "attention.compress_ratios"}) {
            auto bad = valid; bad.kv[std::string("qwen4exp.") + key].arr.pop_back();
            rejects([&] { qwen4exp_config(bad); });
        }
        for (double value : {-1., 0., 1.5, std::numeric_limits<double>::infinity()}) {
            auto bad = valid; bad.kv["qwen4exp.ple.head_vocab_sizes"].arr[0] = value;
            rejects([&] { qwen4exp_config(bad); });
        }
        auto bad = valid; bad.kv["qwen4exp.ple.head_offsets"].arr[0] = 2147483647.;
        rejects([&] { qwen4exp_config(bad); });
        bad = valid; bad.kv["qwen4exp.ple.layers"].arr = {4}; rejects([&] { qwen4exp_config(bad); });
        bad = valid; bad.kv["qwen4exp.ple.layers"].arr = {3}; rejects([&] { qwen4exp_config(bad); });
        bad = valid; bad.kv["qwen4exp.attention.recurrent_layers"].arr = {1, 0, 1, 0};
        rejects([&] { qwen4exp_config(bad); });
        bad = valid; bad.kv["qwen4exp.expert_used_count"].num = 513.; rejects([&] { qwen4exp_config(bad); });
        bad = valid; bad.kv["qwen4exp.attention.compress_ratios"].arr[1] = 65.; rejects([&] { qwen4exp_config(bad); });
        bad = valid; bad.kv["qwen4exp.ple.eos_token_id"].num = 2147483648.; rejects([&] { qwen4exp_config(bad); });
        bad = valid; bad.kv["general.architecture"].str = "qwen35moe"; rejects([&] { qwen4exp_config(bad); });
        auto no_ple = valid; no_ple.kv.erase("qwen4exp.ple.layers");
        require(qwen4exp_config(no_ple).ple_layers.empty(), "optional PLE was invented");

        const std::vector<int32_t> tokens = {10, 20, 99, 30, 40, 50};
        // Independently generated arbitrary-precision, uint64-masked fixtures.
        const std::vector<int32_t> golden = {2, 14, 27, 41, 3, 12, 26, 56, 3, 19, 36, 42,
                                            8, 19, 27, 49, 7, 15, 32, 55, 6, 14, 28, 48};
        PleTokenHistory history(c.ple, 128);
        require(history.append_batch(tokens) == golden, "PLE hash differs from independent golden rows");
        auto big_hash = c.ple; big_hash.multipliers = {23703573157769, 20109073645365, 8052911324071};
        PleTokenHistory overflow(big_hash, 8);
        require(overflow.append(2147483647) == std::vector<int32_t>{5, 11, 25, 49},
                "PLE uint64 overflow differs from masked arbitrary-precision golden");
        history.reset();
        std::vector<int32_t> chunked;
        for (auto token : tokens) {
            const auto rows = history.append(token); chunked.insert(chunked.end(), rows.begin(), rows.end());
        }
        require(chunked == golden, "chunked PLE hash differs");
        history.rollback(2); history.append_batch({33, 44, 55}); history.rollback(2);
        require(history.append_batch({99, 30, 40, 50}) == std::vector<int32_t>(golden.begin() + 8, golden.end()),
                "rollback retained speculative history");
        history.reset(); history.append_batch({10, 99, 20, 99});
        const auto after_eos = history.append(30);
        history.reset(); require(history.append(30) == after_eos, "EOS failed to cut earlier context");
        history.reset(); const auto image_rows = history.append_image();
        history.reset(); require(history.append(88) == image_rows, "image token mapping differs");
        auto no_image = c.ple; no_image.image = 0;
        PleTokenHistory fallback(no_image, 8), eos(no_image, 8);
        require(fallback.append_image() == eos.append(99), "missing image metadata did not fall back to EOS");
        const auto before = history.size();
        rejects([&] { history.append_batch({12, -1, 14}); }); require(history.size() == before, "invalid batch changed history");
        rejects([&] { history.rollback(999); }); require(history.size() == before, "invalid rollback changed history");
        PleTokenHistory small(c.ple, 2); small.append_batch({1, 2});
        rejects([&] { small.append(3); }); require(small.size() == 2, "capacity failure changed history");
        auto invalid = c.ple; invalid.ngram = 1; rejects([&] { PleTokenHistory h(invalid, 8); });
        const auto full = qsa_select_visible({0.5f, 0.9f}, 8, 7, 4, 4);
        require(full.positions == std::vector<int32_t>{4, 5, 6, 7, 0, 1, 2} &&
                full.valid == std::vector<uint8_t>(7, 1), "QSA full-block selection differs");
        const auto tail = qsa_select_visible({0.5f, 0.9f}, 10, 9, 4, 4);
        require(tail.positions == std::vector<int32_t>{8, 9, 4, 5, 6, 7, 0} &&
                tail.valid == std::vector<uint8_t>(7, 1), "QSA incomplete tail was not forced visible");
        const auto prefill = qsa_select_visible({0.5f, 0.9f}, 8, 5, 4, 4);
        require(prefill.positions == std::vector<int32_t>{4, 5, 0, 1, 2, 3, 6} &&
                prefill.valid == std::vector<uint8_t>{1, 1, 1, 1, 1, 1, 0}, "QSA future cell was fetched");
        const auto tie = qsa_select_visible({1.f, 1.f}, 8, 7, 4, 4);
        require(tie.positions == std::vector<int32_t>{0, 1, 2, 3, 4, 5, 6}, "QSA tie order differs");
        std::vector<float> scores(513);
        for (int i = 0; i < 513; ++i) scores[i] = float(513 - i);
        const auto boundary = qsa_select_visible(scores, 2052, 2051, 4, 2048);
        require(boundary.positions.size() == 2051 && boundary.valid.size() == 2051 &&
                boundary.positions[2047] == 2047 && boundary.positions[2048] == 2048 &&
                boundary.positions[2050] == 2050, "QSA top-k plus tail capacity differs");
        rejects([&] { qsa_select_visible({}, 8, 7, 4, 4); });
        rejects([&] { qsa_select_visible({std::numeric_limits<float>::quiet_NaN(), 1.f}, 8, 7, 4, 4); });
        rejects([&] { qsa_select_visible({1.f}, 8, 8, 4, 4); });
        std::puts("qwen4exp metadata, PLE independent goldens, chunking, EOS, image, reset, rollback and capacity: PASS");
        return 0;
    } catch (const std::exception & e) { std::fprintf(stderr, "FAIL: %s\n", e.what()); return 1; }
}
