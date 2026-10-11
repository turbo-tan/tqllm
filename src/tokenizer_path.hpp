#pragma once
#include <string>

namespace tqllm {
// --tokenizer when given; otherwise a GGUF model carries its own tokenizer; "" = none found (the tool reports usage)
inline std::string tokenizer_for(const std::string & model, const std::string & tokenizer) {
    if (!tokenizer.empty()) return tokenizer;
    const std::string ext = ".gguf";
    if (model.size() > ext.size() && model.compare(model.size() - ext.size(), ext.size(), ext) == 0) return model;
    return "";
}
}  // namespace tqllm
