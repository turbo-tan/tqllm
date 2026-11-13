// QSA bookkeeping (QsaPlan): token-key coverage, block keys built only from committed positions, rollback after a
// rejected draft, rewrites of a reused prefix, and the dense / sparse / crossing mode of a verify block.
#include "qwen35_qsa_plan.hpp"
#include <cstdio>

using tqllm::QsaPlan;
using tqllm::qsa_split_point;
static int failures = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++failures; } } while (0)

int main() {
    constexpr int from = 4096;
    // CUDA default: dense unless --qsa-from N is given
    CHECK(tqllm::qsa_cuda_from(-1) == 0);
    CHECK(tqllm::qsa_cuda_from(0) == 0);
    CHECK(tqllm::qsa_cuda_from(49152) == 49152);
    // a file without indexers says so when QSA was asked for, and only then
    CHECK(tqllm::qsa_missing_note(49152, false) != nullptr);
    CHECK(tqllm::qsa_missing_note(49152, true) == nullptr);
    CHECK(tqllm::qsa_missing_note(-1, false) == nullptr);
    CHECK(tqllm::qsa_missing_note(0, false) == nullptr);
    {  // prefill chunks, then decode: dense below `from`, a crossing block, then sparse
        QsaPlan p;
        for (int pos = 0; pos < 4096; pos += 512) CHECK(p.write(pos, 512, true, false, from).mode == 0);
        CHECK(p.keyed == 4096 && p.made == 0);
        CHECK(p.write(4096, 8, true, true, from).mode == 0);  // split(4103) = 4064
        auto s = p.write(4104, 12, true, true, from);          // rows at 4104 .. 4115: split 4096 from row 4111
        CHECK(s.mode == 2 && s.pool_first == 0 && s.pool_count == 1024 && p.made == 1024);
        // 5 of the 12 rows accepted: the next block starts inside the rows just written
        s = p.write(4109, 12, true, true, from);
        CHECK(s.mode == 2 && s.pool_count == 0 && p.keyed == 4121);
        s = p.write(4160, 12, true, true, from);  // a gap (rows 4121 .. 4159 never keyed): dense
        CHECK(s.mode == 0 && p.keyed == 4121);
        s = p.write(4121, 16, true, true, from);
        CHECK(s.mode == 1 && s.pool_count == 0);
        s = p.write(4200, 4, true, true, from);  // gap again
        CHECK(s.mode == 0);
    }
    {  // every block key comes from positions below the block's first row
        QsaPlan p;
        p.write(0, 20000, true, false, from);
        for (int pos = 20000; pos < 20400; pos += 7) {
            const auto s = p.write(pos, 16, true, true, from);
            CHECK(s.mode == 1);
            CHECK(p.made * QsaPlan::kBlock <= pos);
            CHECK(p.made * QsaPlan::kBlock == qsa_split_point(pos + 15));
        }
    }
    {  // a new request keeps a shorter prefix: blocks past it are rebuilt, rows without keys stay dense
        QsaPlan p;
        p.write(0, 30000, true, false, from);
        CHECK(p.write(30000, 8, true, true, from).mode == 1 && p.made == 29984 / 4);
        p.write(10000, 512, true, false, from);
        CHECK(p.made == 2500 && p.keyed == 10512);
        const auto s = p.write(10512, 8, true, true, from);
        CHECK(s.mode == 1 && s.pool_first == 2500 && s.pool_count == qsa_split_point(10519) / 4 - 2500);
        p.unkeyed(6001);  // a path that writes rows without keys
        CHECK(p.keyed == 6001 && p.made == 1500);
        CHECK(p.write(10520, 8, true, true, from).mode == 0);
        CHECK(p.write(6001, 8, false, true, from).mode == 0 && p.keyed == 6001);
        CHECK(p.write(6001, 8, true, true, from).mode == 1 && p.keyed == 6009);
        p.unkeyed(0);
        CHECK(p.keyed == 0 && p.made == 0);
        CHECK(p.write(0, 512, true, false, from).mode == 0 && p.keyed == 512);
    }
    {  // images / oversize blocks are never sparse but keep the keys
        QsaPlan p;
        p.write(0, 9000, true, false, from);
        CHECK(p.write(9000, 8, true, false, from).mode == 0 && p.keyed == 9008);
        CHECK(p.write(9008, 8, true, true, from).mode == 1);
    }
    std::puts(failures ? "qsa27_plan FAIL" : "qsa27_plan PASS");
    return failures ? 1 : 0;
}
