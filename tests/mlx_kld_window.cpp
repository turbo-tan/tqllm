#include "../tools/mlx_kld_window.hpp"
#include <iostream>
#include <stdexcept>
using namespace tqllm::mlx_backend;
int main() {
    if (kld_window_begin(0,100,51201) != 0 || kld_window_begin(7,10,51201) != 3584 ||
        kld_window_begin(99,1,51201) != 50688) return 1;
    // Adjacent batches cover distinct positions and keep exactly one lookahead token.
    for (int first = 0; first < 100; first += 10)
        if (kld_window_begin(first,10,51201) != size_t(first) * 512) return 2;
    for (auto bad : {"", "3.5", "10junk", "2147483648"}) {
        bool rejected = false;
        try {kld_integer(bad);} catch (const std::invalid_argument&) {rejected = true;}
        if (!rejected) return 3;
    }
    for (auto pair : {std::pair{-1,10},std::pair{100,1},std::pair{99,2},std::pair{0,101},std::pair{0,0}}) {
        bool rejected = false;
        try {kld_window_begin(pair.first,pair.second,51201);} catch (const std::invalid_argument&) {rejected = true;}
        if (!rejected) return 4;
    }
    bool rejected = false;
    try {kld_window_begin(7,10,8704);} catch (const std::invalid_argument&) {rejected = true;}
    if (!rejected || kld_integer("99") != 99) return 5;
    std::cout << "kld_window: PASS (bounds, integer parsing, non-overlap, lookahead)\n";
}
