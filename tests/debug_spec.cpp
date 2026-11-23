// TQLLM_DEBUG parsing: a moe_dump:DIR@L item keeps its layer beside other items (',' separates items, so the layer
// cannot follow a ','; the old DIR,L form lost L and dumped every layer). No CUDA.
#include "src/core/debug.hpp"
#include <cstdio>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

int main() {
    const auto m = tqllm::parse_debug("memlog,moe_dump:/w/dump@20,steps");
    CHECK(m.count("memlog") && m.count("steps") && m.size() == 3);
    CHECK(m.count("moe_dump"));
    const tqllm::DirLayer dl = tqllm::debug_dir_layer(m.count("moe_dump") ? m.at("moe_dump") : std::string());
    CHECK(dl.dir == "/w/dump");
    CHECK(dl.layer == 20);
    const tqllm::DirLayer all = tqllm::debug_dir_layer("/w/dump");
    CHECK(all.dir == "/w/dump" && all.layer == -1);
    const tqllm::DirLayer l0 = tqllm::debug_dir_layer(tqllm::parse_debug("moe_dump:/a:b@0").at("moe_dump"));
    CHECK(l0.dir == "/a:b" && l0.layer == 0);
    CHECK(tqllm::parse_debug("tf_dump:/x.txt").at("tf_dump") == "/x.txt");
    if (fails) return 1;
    std::printf("debug_spec: ok\n");
    return 0;
}
