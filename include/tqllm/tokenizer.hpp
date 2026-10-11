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
private:
    std::vector<std::string> id_to_piece_;
    std::unordered_map<std::string, uint32_t> piece_to_id_;
    std::unordered_map<std::string, uint32_t> special_to_id_;
    std::vector<std::pair<std::string, std::string>> merges_;
    std::unordered_map<std::string, uint32_t> merge_rank_;
    std::vector<std::tuple<uint32_t, uint32_t, uint8_t>> unicode_ranges_;
    uint32_t eos_id_ = 151645;
    bool is_letter(uint32_t cp) const;
    bool marks_in_letters_ = false;  // Qwen3.5 pre-tokenizer: \p{M} joins letter runs
    uint8_t category(uint32_t cp) const;
    void load_gguf(const std::string & path);
    std::vector<std::string> pretokenize(const std::string & text) const;
    std::vector<uint32_t> encode_piece(const std::string & piece) const;
};
}
