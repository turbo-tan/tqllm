// Draft-ahead keeps the step's control values on the GPU (spec_pipeline.hpp / dflash.cpp inject_device). The device
// formulas must equal the host loop's integers: accepted = leading drafted tokens equal to the target's choices
// (sum of the cumulative product of the equality flags), the next anchor = choices[accepted], the inject rows =
// accepted + 1, and a 16-row inject scatters row j to ring row block + (head + j) % window while j < rows, else to
// scratch row j (distinct slots). Random cases incl. accepted = 0, width (all accepted) and ring wrap-around.
#include <mlx/mlx.h>
#include <cstdio>
#include <random>
#include <set>
#include <vector>

namespace mx = mlx::core;

int main() {
    std::mt19937 rng(11);
    int failures = 0, cases = 0;
    auto check = [&](bool ok, const char* what, int c) {if (!ok) {std::printf("FAIL case %d: %s\n", c, what); ++failures;}};
    const int width = 15, block = 16, window = 2048;
    for (int c = 0; c < 200; ++c, ++cases) {
        // choices: width + 1 target tokens; drafted: width tokens equal to the choices for the first `agree` rows
        const int agree = c % 4 == 0 ? 0 : c % 4 == 1 ? width : int(rng() % (width + 1));
        std::vector<uint32_t> choices(width + 1), drafted(width);
        for (auto& t : choices) t = rng() % 1000;
        for (int j = 0; j < width; ++j) drafted[j] = j < agree ? choices[j] : choices[j] + 1 + rng() % 5;
        auto ch = mx::array(choices.data(), {width + 1}, mx::uint32), dr = mx::array(drafted.data(), {width}, mx::uint32);
        auto equal = mx::astype(mx::equal(mx::slice(ch, {0}, {width}), dr), mx::int32);
        auto accepted = mx::sum(mx::cumprod(equal, 0), false);
        auto next_anchor = mx::take(ch, accepted);
        auto rows = mx::add(accepted, mx::array(1, mx::int32));
        mx::eval(accepted, next_anchor, rows);
        int host = 0;
        while (host < width && drafted[host] == choices[host]) ++host;
        check(accepted.item<int32_t>() == host, "accepted == host loop", c);
        check(next_anchor.item<uint32_t>() == choices[host], "next anchor == choices[accepted]", c);
        check(rows.item<int32_t>() == host + 1, "inject rows == accepted + 1", c);
        // ring scatter slots for this inject at a random head (incl. wrap)
        const int head = c % 3 == 0 ? window - 1 - int(rng() % block) : int(rng() % window);
        auto j = mx::arange(block, mx::int32);
        auto slot = mx::where(mx::less(j, rows), mx::add(mx::remainder(mx::add(mx::array(head, mx::int32), j), mx::array(window, mx::int32)),
                                                         mx::array(block, mx::int32)), j);
        mx::eval(slot);
        std::set<int> seen;
        bool slots_ok = true;
        for (int r = 0; r < block; ++r) {
            const int expect = r < host + 1 ? block + (head + r) % window : r;
            slots_ok = slots_ok && slot.data<int32_t>()[r] == expect && seen.insert(expect).second;
        }
        check(slots_ok, "ring / scratch slots and distinct", c);
    }
    std::printf("DRAFT_AHEAD_MATH %d cases, %d failures\n", cases, failures);
    std::puts(failures ? "DRAFT_AHEAD_MATH_FAIL" : "DRAFT_AHEAD_MATH_PASS");
    return failures ? 1 : 0;
}
