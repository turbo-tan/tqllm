// Weight-format plugins (ABI 2, src/formats/weight/plugin_format.hpp) with the example plugins of tests/plugins/:
//   new PATH      a new type loads, passes the contract, and serves matrices of its type
//   patch PATH    an exact I8H patch applies: Q8_0 matrices run on the patched format, byte-identical to the built-in
//   refuse PATH   a plugin with a wrong CPU reference / a false exact claim is refused at load, naming why
#include "formats/weight/plugin_format.hpp"
#include "formats/weight/weight_format.hpp"
#include "gguf.hpp"
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>

int main(int argc, char ** argv) {
    if (argc != 4) { std::fprintf(stderr, "usage: test-plugin-formats new|patch|refuse PLUGIN.so EXPECTED\n"); return 2; }
    const std::string mode = argv[1], path = argv[2], expect = argv[3];
    try {
        const auto * builtin_i8h = tqllm::weight_format(1000);
        const tqllm::PluginInfo info = tqllm::load_plugin(path);
        if (mode == "refuse") { std::printf("PLUGIN FAIL: %s was not refused\n", path.c_str()); return 1; }
        if (mode == "new") {
            const auto * f = tqllm::weight_format(2001);
            const bool ok = info.formats.size() == 1 && f && std::string(f->name()) == expect &&
                            tqllm::gguf_tensor_bytes(2001, {64, 4}, "w") == 4 * 2 * 34;
            std::printf("PLUGIN %s: new format %s\n", ok ? "PASS" : "FAIL", info.formats.empty() ? "-" : info.formats[0].c_str());
            return ok ? 0 : 1;
        }
        // patch: the Q8_0 loader hands out the patched I8H
        const auto * i8h = tqllm::weight_format(1000);
        std::vector<char> raw(size_t(4) * 34, 1);  // 4 x 32: no tuning launch, so the upload can stay on the host
        tqllm::Upload up = [](const void * h, size_t) { return h; };
        const tqllm::Mat m = tqllm::weight_format(8)->load(raw, 4, 32, up, nullptr);
        const bool ok = info.patches.size() == 1 && info.patches[0].find(expect) != std::string::npos && m.fmt == i8h && i8h != builtin_i8h;
        std::printf("PLUGIN %s: patch %s, Q8_0 runs on %s\n", ok ? "PASS" : "FAIL", info.patches.empty() ? "-" : info.patches[0].c_str(),
                    m.fmt == i8h ? "the patched I8H" : "another format");
        return ok ? 0 : 1;
    } catch (const std::exception & e) {
        const bool ok = mode == "refuse" && std::strstr(e.what(), expect.c_str());
        std::printf("PLUGIN %s: load refused: %s\n", ok ? "PASS" : "FAIL", e.what());
        return ok ? 0 : 1;
    }
}
