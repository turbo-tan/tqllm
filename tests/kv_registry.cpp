// Host-only contract test for the KV-cache format registry: every format is registered, reports sane sizes and
// provenance, and declares its verify-scratch need. No GPU needed.
#include "kv_format.hpp"

#include <cstdio>
#include <cstring>
#include <stdexcept>

using namespace tqllm;

static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { std::printf("FAIL %s:%d ", __FILE__, __LINE__); std::printf(__VA_ARGS__); std::printf("\n"); ++fails; } } while (0)

int main() {
    const auto names = kv_format_names();
    CHECK(names.size() >= 7, "expected the 7 built-in formats, got %zu", names.size());
    for (const char * must : {"bf16", "q8", "q8k-turbo3v", "t4k-turbo3v", "t4kv", "tq3", "tq3k-tq2v"}) {
        bool found = false;
        for (const auto & n : names) found = found || n == must;
        CHECK(found, "format %s not registered", must);
    }
    for (const auto & n : names) {
        const KvFormat * f = kv_format(n);
        CHECK(n == f->name(), "registered as %s but name() is %s", n.c_str(), f->name());
        const KvProvenance p = f->provenance();
        CHECK(p.summary && std::strlen(p.summary) > 0, "%s: empty provenance summary", n.c_str());
        CHECK(p.authors && std::strlen(p.authors) > 0, "%s: empty authors", n.c_str());
        CHECK(p.source && std::strlen(p.source) > 0, "%s: empty source", n.c_str());
        CHECK(p.licence && std::strlen(p.licence) > 0, "%s: empty licence", n.c_str());
        CHECK(p.borrowed && std::strlen(p.borrowed) > 0, "%s: empty borrowed", n.c_str());
        const size_t k1 = f->key_bytes(1, 4, 256), k2 = f->key_bytes(1024, 4, 256);
        const size_t v1 = f->value_bytes(1, 4, 256), v2 = f->value_bytes(1024, 4, 256);
        CHECK(k1 > 0 && v1 > 0, "%s: zero record size", n.c_str());
        CHECK(k2 >= k1 * 1024 && k2 <= k1 * 1024 + 4096, "%s: key bytes not linear in context (%zu vs %zu)", n.c_str(), k1, k2);
        CHECK(v2 >= v1 * 1024 && v2 <= v1 * 1024 + 4096, "%s: value bytes not linear in context (%zu vs %zu)", n.c_str(), v1, v2);
        CHECK(f->split_factor() >= 1, "%s: split_factor < 1", n.c_str());
        // KV storage (src/kv/kv_store.hpp) places position p at p * row inside a page: sizes must be exactly linear
        for (int64_t c : {int64_t(255), int64_t(256), int64_t(4096), int64_t(204800)}) {
            CHECK(f->key_bytes(c, 4, 256) == size_t(c) * k1, "%s: key bytes at %lld not c * row", n.c_str(), (long long)c);
            CHECK(f->value_bytes(c, 4, 256) == size_t(c) * v1, "%s: value bytes at %lld not c * row", n.c_str(), (long long)c);
        }
        CHECK(k1 % 16 == 0 && v1 % 16 == 0, "%s: rows must stay 16-byte aligned", n.c_str());
    }
    // the conservative default must stay on for formats not validated without the context-sized scratch
    for (const char * keep : {"bf16", "q8", "q8k-turbo3v", "t4k-turbo3v", "t4kv"})
        CHECK(kv_format(keep)->needs_context_scratch(), "%s must keep the conservative scratch", keep);
    CHECK(!kv_format("tq3")->needs_context_scratch(), "tq3 uses only the multi-query partials (measured 2026-10-05)");
    CHECK(!kv_format("tq3k-tq2v")->needs_context_scratch(), "tq3k-tq2v uses only the multi-query partials");
    // tq3 record: 112 B per head per token for K and for V
    CHECK(kv_format("tq3")->key_bytes(1, 1, 256) == 112, "tq3 record is 112 B");
    CHECK(kv_format("tq3k-tq2v")->key_bytes(1, 1, 256) == 112, "tq3k-tq2v K record is 112 B");
    CHECK(kv_format("tq3k-tq2v")->value_bytes(1, 1, 256) == 80, "tq3k-tq2v V record is 80 B");
    bool threw = false;
    try { kv_format("no-such-format"); } catch (const std::runtime_error &) { threw = true; }
    CHECK(threw, "unknown format must throw");
    const std::string listing = kv_format_listing(16, 4, 256);
    CHECK(listing.find("tq3") != std::string::npos && listing.find("tq3k-tq2v") != std::string::npos &&
              listing.find("80 B V") != std::string::npos && listing.find("multi-query only") != std::string::npos,
          "listing incomplete");
    if (fails) std::printf("kv_registry: %d failure(s)\n", fails);
    else std::printf("kv_registry: ok (%zu formats)\n", names.size());
    return fails ? 1 : 0;
}
