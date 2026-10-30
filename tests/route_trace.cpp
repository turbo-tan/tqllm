#include "models/qwen/qwen4exp/route_trace.hpp"
#include <array>
#include <vector>

using namespace tqllm::flashnext;
static void expect(bool ok, const char * what) { if (!ok) throw std::runtime_error(what); }

int main() {
    try {
        using File = RouteTraceFile;
        File out(std::tmpfile()); expect(bool(out), "temporary trace file");
        std::vector<int> ids(60), pred(60);
        for (int i = 0; i < 60; ++i) { ids[size_t(i)] = i; pred[size_t(i)] = 511 - i; }
        // A six-row verify must remain one record, not be mistaken for a new prompt.
        write_route_trace(out.get(), true, 7, 6, 10, ids.data(), ids.size(), pred.data(), pred.size());
        write_route_trace(out.get(), true, 8, 6, 10, ids.data(), ids.size());
        std::rewind(out.get());
        std::array<int32_t, 3> header{};
        std::vector<int> actual(60);
        expect(std::fread(header.data(), 4, 3, out.get()) == 3 && header == std::array<int32_t, 3>{7, 6, 60}, "prediction header");
        expect(std::fread(actual.data(), 4, 60, out.get()) == 60 && actual == ids, "original route order");
        expect(std::fread(actual.data(), 4, 60, out.get()) == 60 && actual == pred, "prediction order");
        expect(std::fread(header.data(), 4, 3, out.get()) == 3 && header == std::array<int32_t, 3>{8, 6, 0}, "mixed route without predictions");
        expect(std::fread(actual.data(), 4, 60, out.get()) == 60 && actual == ids && std::fgetc(out.get()) == EOF, "record boundaries");

        File plain(std::tmpfile()); expect(bool(plain), "plain trace file");
        write_route_trace(plain.get(), false, 2, 6, 10, ids.data(), ids.size());
        std::rewind(plain.get());
        std::array<unsigned char, 8> bytes{};
        expect(std::fread(bytes.data(), 1, 8, plain.get()) == 8 &&
               bytes == std::array<unsigned char, 8>{2, 0, 0, 0, 6, 0, 0, 0}, "legacy little-endian plain header");
        expect(std::fread(actual.data(), 4, 60, plain.get()) == 60 && actual == ids && std::fgetc(plain.get()) == EOF, "plain payload framing");

        for (int mode = 0; mode < 5; ++mode) {
            File bad(std::tmpfile()); expect(bool(bad), "invalid-shape file"); bool rejected = false;
            try {
                write_route_trace(bad.get(), true, 0, mode == 0 ? 0 : 6, mode == 1 ? 8 : 10,
                                  mode == 2 ? nullptr : ids.data(), mode == 3 ? 59 : 60,
                                  pred.data(), mode == 4 ? 59 : 60);
            } catch (const std::invalid_argument &) { rejected = true; }
            expect(rejected && std::ftell(bad.get()) == 0, "invalid shape must fail before writing a header");
        }
        File full(std::fopen("/dev/full", "wb"));
        if (full) {
            bool rejected = false;
            try { write_route_trace(full.get(), true, 0, 6, 10, ids.data(), ids.size()); }
            catch (const std::runtime_error &) { rejected = true; }
            expect(rejected, "disk-full trace must not silently pass");
        }
        std::puts("route trace: PASS");
    } catch (const std::exception & e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
