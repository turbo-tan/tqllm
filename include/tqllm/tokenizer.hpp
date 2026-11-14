#pragma once
#include <cstdint>
#include <memory>
#include <mutex>
#include <shared_mutex>
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
    // Which added tokens are matched atomically in the text: all (encode() above), or the user-defined ones only, so
    // that control tokens written in the text are tokenized as text (HF split_special_tokens=True: embedding inputs).
    enum class Specials { all, user_only };
    // offsets (optional): per id, the byte range [begin, end) of the text it covers
    std::vector<uint32_t> encode(const std::string & text, Specials specials,
                                 std::vector<std::pair<uint32_t, uint32_t>> * offsets = nullptr) const;
    std::string decode(const std::vector<uint32_t> & ids) const;
    // fill the piece cache with every ordinary vocabulary piece (most words of real text are one), so the first request
    // does not pay the merges; returns the entries added. Parallel; seconds at load for a 250K vocabulary.
    size_t warm_cache(unsigned threads = 0) const;
    // piece cache entries now / the most it holds (above the vocabulary, so warming never fills it; an overfull cache drops
    // a quarter of its entries, not all of them)
    size_t piece_cache_size() const;
    size_t piece_cache_capacity() const;
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
    std::vector<std::string> user_specials_;  // GGUF token_type 4: still atomic under Specials::user_only
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
    std::vector<uint32_t> encode_piece(const std::string & piece, std::vector<uint32_t> * lens = nullptr) const;
    std::vector<uint32_t> merge_piece(const std::string & piece, std::vector<uint32_t> * lens) const;
    // BPE result per pre-token piece (ids and the bytes each covers): words repeat, the merge loop is the tokenizer's
    // cost. Shared by copies; readers in parallel (tqserve request threads).
    struct PieceCache {
        std::shared_mutex m;
        std::unordered_map<std::string, std::pair<std::vector<uint32_t>, std::vector<uint32_t>>> map;
    };
    std::shared_ptr<PieceCache> cache_ = std::make_shared<PieceCache>();
};
}
