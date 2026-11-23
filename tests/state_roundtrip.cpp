// A kept conversation state through the blob (Engine::export_state / import_state, src/kv/conversation_state.hpp): turn 1
// on slot 0, its state exported and imported into slot 1, turn 2 there must reuse it and give the same tokens as turn 2
// on slot 0, which kept its own state. Model test: TQLLM_TEST_MODEL (a .tqfn manifest or a qwen35 GGUF),
// TQLLM_TEST_TOKENIZER (optional: the model file); skipped without them (exit 77).
// Across processes (nothing in the blob may point at the exporting process): `--save DIR` writes the blob and turn 2's
// tokens, `--load DIR` (a fresh process) imports the blob into slot 1 and checks turn 2 there.
#include "tqllm/model.hpp"
#include "tqllm/options.hpp"
#include "tqllm/tokenizer.hpp"
#include "kv/conversation_state.hpp"
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

static std::vector<uint32_t> run(tqllm::Model & m, int slot, const std::vector<uint32_t> & in, uint32_t count) {
    m.select_slot(slot);
    m.spec_begin(in, count, 3);
    while (m.spec_advance()) {}
    return m.spec_finish();
}

int main(int argc, char ** argv) {
    const char * model = std::getenv("TQLLM_TEST_MODEL");
    if (!model) { std::cout << "state_roundtrip: skipped (set TQLLM_TEST_MODEL)\n"; return 77; }
    const char * tokp = std::getenv("TQLLM_TEST_TOKENIZER");
    const std::string mode = argc > 2 ? argv[1] : "", dir = argc > 2 ? argv[2] : "";
    try {
        tqllm::Tokenizer tok;
        tok.load(tokp ? tokp : model);
        std::string notes;
        for (int i = 0; i < 60; ++i) notes += "Note " + std::to_string(i) + ": shipment " + std::to_string(i * 37 % 101) + " left depot " + std::to_string(i % 7) + ".\n";
        const std::string turn1 = "<|im_start|>user\nHere are notes:\n" + notes + "Which depot shipped the most? Answer in two sentences.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
        const auto p1 = tok.encode(turn1);
        const uint32_t count = 48;
        tqllm::EngineOptions o; o.prompt_cache = true;
        tqllm::Model m(model, tokp ? tokp : model, 8192, 2, o);
        auto turn2 = [&](const std::vector<uint32_t> & out1) {
            std::vector<uint32_t> p2 = p1;
            p2.insert(p2.end(), out1.begin(), out1.end());
            const auto more = tok.encode("<|im_end|>\n<|im_start|>user\nAnd the least?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
            p2.insert(p2.end(), more.begin(), more.end());
            return p2;
        };
        if (mode == "--load") {
            const auto blob = tqllm::read_state_blob(dir + "/state.bin");
            std::ifstream f(dir + "/turn2.txt"); std::vector<uint32_t> p2, ref; size_t n = 0; f >> n; p2.resize(n); for (auto & t : p2) f >> t; f >> n; ref.resize(n); for (auto & t : ref) f >> t;
            if (!f) throw std::runtime_error("cannot read " + dir + "/turn2.txt");
            if (const auto why = m.import_state(1, blob); !why.empty()) { std::cout << "state_roundtrip: FAIL: import: " << why << "\n"; return 1; }
            const uint32_t reused = m.prefix_reuse(1, p2);
            const auto got = run(m, 1, p2, count);
            std::cout << "state_roundtrip --load: slot 1 reused " << reused << " of " << p2.size() << " prompt tokens; turn 2 " << (got == ref ? "identical" : "DIFFERS") << "\n";
            return reused > 0 && got == ref ? 0 : 1;
        }
        const auto out1 = run(m, 0, p1, count);
        const auto blob = m.export_state(0);
        if (!blob) { std::cout << "state_roundtrip: skipped (this engine exports no state)\n"; return 77; }
        std::cout << "state_roundtrip: turn 1 " << p1.size() << " + " << out1.size() << " tokens; kept state " << blob->meta.positions << " positions, "
                  << blob->parts.size() << " parts, " << blob->bytes() / 1048576.0 << " MB\n";
        const auto p2 = turn2(out1);
        if (mode == "--save") tqllm::write_state_blob(dir + "/state.bin", *blob);
        // a blob whose layout is not this engine's is refused with the field named
        {
            tqllm::StateBlob bad = *blob; bad.meta.layout += "|extra=1";
            const auto why = m.import_state(1, bad);
            if (why.empty() || why.find("extra") == std::string::npos) { std::cout << "state_roundtrip: FAIL: a foreign layout was not refused by name: '" << why << "'\n"; return 1; }
        }
        if (const auto why = m.import_state(1, *blob); !why.empty()) { std::cout << "state_roundtrip: FAIL: import: " << why << "\n"; return 1; }
        const uint32_t reused = m.prefix_reuse(1, p2);
        const auto got = run(m, 1, p2, count);
        const auto ref = run(m, 0, p2, count);
        std::cout << "state_roundtrip: slot 1 (imported) reused " << reused << " of " << p2.size() << " prompt tokens; turn 2 "
                  << (got == ref ? "identical" : "DIFFERS") << " to slot 0 (kept its own)\n";
        if (mode == "--save") {
            std::ofstream f(dir + "/turn2.txt"); f << p2.size(); for (auto t : p2) f << ' ' << t; f << '\n' << ref.size(); for (auto t : ref) f << ' ' << t; f << '\n';
        }
        return reused > 0 && got == ref ? 0 : 1;
    } catch (const std::exception & e) {
        std::cout << "state_roundtrip: FAIL: " << e.what() << "\n";
        return 1;
    }
}
