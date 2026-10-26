#pragma once
// The architectures a Hugging Face folder can be opened as (hf_folder.hpp). A new architecture: one row in hf_arch.cpp
// plus its map function in arch_<name>.cpp, written with HfFolder's builders.
#include "hf_folder.hpp"

namespace tqllm {

struct HfArch {
    const char * hf;    // config.json architectures[0]
    const char * arch;  // the engine's general.architecture
    void (*map)(const HfFolder & h, GgufFile & f);
};

const HfArch * hf_arch(const std::string & hf_name);

// arch_minimax_m3.cpp, arch_minimax_m3_dspark.cpp
void hf_map_minimax_m3(const HfFolder & h, GgufFile & f);
void hf_map_minimax_m3_dspark(const HfFolder & h, GgufFile & f);

}  // namespace tqllm
