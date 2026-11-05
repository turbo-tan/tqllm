// Weight-format contract (src/formats/weight/format_contract.hpp) over every registered GGUF weight format: GPU dequant_h
// against the CPU reference and matvec against a double dot, both passes. A registered format with no CPU reference is a
// contract gap.
#include "formats/weight/format_contract.hpp"
#include <cstdio>
#include <random>

int main() {
    std::mt19937 rng(11);
    int tested = 0, fails = 0;
    for (uint32_t type = 0; type < 64; ++type) {
        const auto * f = tqllm::weight_format(type);
        if (!f) continue;
        for (int pass = 0; pass < 2; ++pass) {
            const auto r = tqllm::contract_pass(type, *f, nullptr, pass, rng);
            if (!r.error.empty()) {
                std::printf("FAIL %-10s type %2u: %s\n", f->name(), type, r.error.c_str());
                ++fails; break;
            }
            std::printf("%s %-10s type %2u pass %c: dequant %zu values, %zu > 1 ulp (worst %d, max |diff| / group max %.2e); matvec rel-rms %.2e over %d rows\n",
                        r.gap ? "GAP " : r.ok ? "ok  " : "FAIL", f->name(), type, pass ? 'B' : 'A', r.compared, r.bad, r.worst_ulp, r.worst_rel,
                        r.matvec_rel, r.rows_used);
            fails += !r.ok;
        }
        ++tested;
    }
    std::printf("FORMAT_CONTRACT %s (%d formats)\n", fails ? "FAIL" : "PASS", tested);
    return fails ? 1 : 0;
}
