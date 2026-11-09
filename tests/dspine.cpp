#include "dspine.hpp"
#include <algorithm>
#include <cstdio>

using namespace tqllm::spec;
int main() {
    const DSpineConfig c{2, 2, 1, 1, 3, 16, 1e-6f};
    const float read[] = {1, 0, 0, 1}, write[] = {1, 0};
    const float book[] = {1, 0, 0, 1}, gate[] = {0, 0}, prev[] = {1, 0}, bias[] = {0};
    const DSpineWeights w{read, write, book, gate, gate, prev, bias};
    const std::vector<float> x{1, 1, 1, 1, 2, 1, 4, 3, 1, 2, 3, 4};
    const auto y = dspine_write_reference(c, w, x, {0, 1});
    int failed = 0;
    auto check = [&](bool ok, const char* name) { std::printf("%s %s\n", ok ? "PASS" : "FAIL", name); failed += !ok; };
    check(y[0] == x[0] && y[1] == x[1] && y[6] == x[6] && y[7] == x[7], "each slot's anchor is unchanged");
    const float expected = dspine_bf16(1.f + dspine_bf16(.5f * dspine_bf16(std::sqrt(2.f))));
    check(y[2] == expected && y[3] == 1.f, "first write uses the exact anchor code and a sigmoid gate");
    auto changed = x; changed[4] = 100; changed[5] = -200;
    const auto future = dspine_write_reference(c, w, changed, {0, 1});
    check(std::equal(y.begin(), y.begin() + 4, future.begin()), "future rows cannot change earlier writes");
    changed = x; std::fill(changed.begin() + 6, changed.end(), 120);
    const auto other = dspine_write_reference(c, w, changed, {0, 0});
    check(std::equal(y.begin(), y.begin() + 6, other.begin()), "writes do not cross slot boundaries");
    const auto single = dspine_write_reference(c, w, {x.begin() + 6, x.end()}, {1});
    check(std::equal(y.begin() + 6, y.end(), single.begin()), "batched and single-slot bridge outputs match");
    const auto zero = dspine_write_reference(c, w, std::vector<float>(6, 0), {0});
    check(std::all_of(zero.begin(), zero.end(), [](float v) { return std::isfinite(v); }), "zero features stay finite");
    return failed ? 1 : 0;
}
