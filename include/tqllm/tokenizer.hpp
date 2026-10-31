#pragma once
#include <cstdint>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>
namespace tqllm {
class Tokenizer {
public:
    void load(const std::string & path);  // tokenizer.tqtok, or a GGUF model file (its tokenizer.ggml.* metadata)
    std::vector<uint32_t> encode(const std::string & text) const;
    std::string decode(const std::vector<uint32_t> & ids) const;
    uint32_t eos_id() const { return eos_id_; }
    // end of generation: EOS, or the GGUF's end-of-turn / end-of-message ids when it names them (GLM ends a turn with
    // <|user|> / <|observation|>, not its EOS)
    bool is_eog(uint32_t id) const { return id == eos_id_ || id == eot_id_ || id == eom_id_; }
    // the id of a special token by its text, ~0u when the vocabulary has none
    uint32_t special_id(const std::string & text) const {
        const auto it = special_to_id_.find(text);
        return it == special_to_id_.end() ? ~0u : it->second;
    }
private:
    std::vector<std::string> id_to_piece_;
    std::unordered_map<std::string, uint32_t> piece_to_id_;
    std::unordered_map<std::string, uint32_t> special_to_id_;
    std::vector<std::pair<std::string, std::string>> merges_;
    std::unordered_map<std::string, uint32_t> merge_rank_;
    std::vector<std::tuple<uint32_t, uint32_t, uint8_t>> unicode_ranges_;
    uint32_t eos_id_ = 151645, eot_id_ = UINT32_MAX, eom_id_ = UINT32_MAX;
    bool is_letter(uint32_t cp) const;
    bool marks_in_letters_ = false;  // Qwen3.5 pre-tokenizer: \p{M} joins letter runs
    bool o200k_ = false;             // MiniMax-M3 pre-tokenizer: o200k-style case-aware letter runs
    bool digits3_ = false;           // GLM-4 / GLM-5 pre-tokenizer: digit runs of up to 3
    std::vector<std::tuple<uint32_t, uint32_t, uint8_t>> case_ranges_;  // 16 = Lu / Lt, 32 = Ll
    uint8_t letter_case(uint32_t cp) const;
    std::vector<std::string> pretokenize_o200k(const std::string & text) const;
    uint8_t category(uint32_t cp) const;
    void load_gguf(const std::string & path);
    std::vector<std::string> pretokenize(const std::string & text) const;
    std::vector<uint32_t> encode_piece(const std::string & piece) const;
};
}
