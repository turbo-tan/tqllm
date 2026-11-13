#pragma once
// QSA bookkeeping of one sequence's cache (host side): which positions hold indexer token keys, which block keys are
// built, and how the next verify block attends. Block keys are pooled from committed positions only: a row picks among
// blocks below its split point, which lies at or below the block's first position.
#include <algorithm>

namespace tqllm {

// tq_split_point (src/core/common.cuh, device code); tests/qsa27.cu checks they agree
constexpr int qsa_split_point(int qpos) { return qpos >= 15 ? (qpos - 15) / 32 * 32 : 0; }

// CUDA: QSA is an approximate path, so it is opt-in. option = --qsa-from (-1 not given, 0 off, N keys): rows past N keys
// are sparse only when N is given; 0 = dense. (The file's qsa.sparse_from is not a request.)
constexpr int qsa_cuda_from(int option) { return option > 0 ? option : 0; }
// --qsa-from N asked for a file without QSA indexers: the start-up WARN (null: nothing to say)
constexpr const char * qsa_missing_note(int option, bool indexers) {
    return option > 0 && !indexers ? "--qsa-from is set, but the model file has no QSA indexers: attention stays dense "
                                     "(add them with tools/qsa_indexer_gguf.py + tools/gguf_add.py)"
                                   : nullptr;
}

struct QsaPlan {
    static constexpr int kBlock = 4;
    int keyed = 0;  // cache positions [0, keyed) hold token keys
    int made = 0;   // block keys [0, made) are built
    struct Step {
        int mode = 0;                 // 0 dense, 1 every row sparse, 2 the block crosses `from`
        int pool_first = 0, pool_count = 0;  // block keys to build before the launch
    };
    // positions from pos are (re)written without token keys
    void unkeyed(int pos) {
        keyed = std::min(keyed, pos);
        made = std::min(made, pos / kBlock);
    }
    // a verify block writes n rows at pos. keys: it writes their token keys; sparse_ok: its rows may be sparse (a
    // verify-sized block of text); rows whose split point is >= from are sparse
    Step write(int pos, int n, bool keys, bool sparse_ok, int from) {
        Step s;
        made = std::min(made, pos / kBlock);
        if (!keys || pos > keyed) {  // a gap without keys: dense until the cache is rewritten from a keyed position
            keyed = std::min(keyed, pos);
            return s;
        }
        keyed = pos + n;
        if (!sparse_ok) return s;
        const int lo = qsa_split_point(pos), hi = qsa_split_point(pos + n - 1);
        if (hi < from) return s;
        if (const int want = hi / kBlock; made < want) s.pool_first = made, s.pool_count = want - made, made = want;
        s.mode = lo >= from ? 1 : 2;
        return s;
    }
};

}  // namespace tqllm
