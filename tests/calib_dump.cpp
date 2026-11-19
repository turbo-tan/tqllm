// TQLLM_DEBUG=calib_dump:LAYERS:DIR (src/models/qwen/qwen35/calib_dump.hpp), the capture behind tools/convert/qwen35/
// tqv12_quant.py. Without the token nothing is written (debug_value is null, the engine hook returns at once). The value
// parses into ascending unique layers and the directory after the first ':' (a directory may contain ':'); malformed
// values are refused at the first capture instead of writing the wrong layers.
// `test-calib-dump --token` checks the value as the engine reads it from a TQLLM_DEBUG list (ctest sets the variable).
#include "calib_dump.hpp"
#include "debug.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

using namespace tqllm;
namespace {
int fails = 0;
void expect(bool ok, const char * what) {
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", what);
    fails += !ok;
}
bool refused(const char * v) {
    try { parse_calib_dump(v); } catch (const std::invalid_argument &) { return true; }
    return false;
}
}  // namespace

int main(int argc, char ** argv) {
    if (argc > 1 && std::strcmp(argv[1], "--token") == 0) {
        const std::string * v = debug_value("calib_dump");
        expect(v && *v == "0-1+5:/tmp/a:b", "value after the first ':' of the token, other diagnostics beside it");
        if (v) {
            const auto s = parse_calib_dump(*v);
            expect(s.layers == std::vector<int>{0, 1, 5} && s.dir == "/tmp/a:b", "list in a TQLLM_DEBUG list parses");
        }
        return fails ? EXIT_FAILURE : EXIT_SUCCESS;
    }
    if (const char * env = std::getenv("TQLLM_DEBUG"); !env || !std::strstr(env, "calib_dump"))
        expect(debug_value("calib_dump") == nullptr, "off by default: no calib_dump token, no capture");
    auto s = parse_calib_dump("12:/d");
    expect(s.layers == std::vector<int>{12} && s.dir == "/d", "one layer");
    s = parse_calib_dump("0-3+8+10-11:/x/y");
    expect(s.layers == std::vector<int>{0, 1, 2, 3, 8, 10, 11} && s.dir == "/x/y", "ranges and single layers joined by '+'");
    s = parse_calib_dump("4+4-5:d");
    expect(s.layers == std::vector<int>{4, 5}, "a repeated layer is written once");
    expect(refused(""), "empty value refused");
    expect(refused("0-3"), "no directory refused");
    expect(refused(":/d"), "no layers refused");
    expect(refused("0-3:"), "empty directory refused");
    expect(refused("a:/d"), "non-numeric layer refused");
    expect(refused("5-3:/d"), "empty range refused");
    expect(refused("8+4:/d"), "descending list refused");
    expect(refused("1++2:/d"), "empty list item refused");
    std::printf("calib_dump: %s\n", fails ? "FAIL" : "PASS");
    return fails ? EXIT_FAILURE : EXIT_SUCCESS;
}
