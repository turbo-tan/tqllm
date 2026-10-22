// tqfn-pack: writes the Flash-Next packed experts file (experts-packed.tqfnexp): every expert of every layer as one block
// in the decode cache's row layout, so a cache miss is a single copy. Optional: without it tqserve builds the same bytes
// in memory at start-up; with it (manifest experts=) decode runs faster. Usage: tqfn-pack OUT shard1 shard2 --gsq-rco
#include "fn_model.hpp"
#include <thread>

int main(int argc, char ** argv) {
    try {
        require(argc >= 3, "usage: tqfn-pack OUT.tqfnexp shard1.gguf [more shards] [--gsq-rco]");
        const std::string out = argv[1];
        std::vector<char *> args{argv[0], argv[0]}; for (int i = 2; i < argc; ++i) args.push_back(argv[i]);
        Checkpoint checkpoint(int(args.size()), args.data()); const auto & c = checkpoint.config;
        const auto pe = build_packed(checkpoint, std::max(4, std::min(32, int(std::thread::hardware_concurrency()))));
        FILE * f = std::fopen(out.c_str(), "wb"); require(f, "cannot create " + out);
        std::fwrite(pe.tiled ? "TQFNEXPT" : "TQFNEXP1", 1, 8, f); int32_t dims[2] = {c.layers, c.experts}; std::fwrite(dims, 4, 2, f);
        for (int l = 0; l < c.layers; ++l) std::fwrite(pe.sizes[l].data(), 8, 6, f);
        for (size_t done = 0; done < pe.bytes;) {
            const size_t n = std::min<size_t>(size_t(1) << 28, pe.bytes - done);
            require(std::fwrite(pe.map + done, 1, n, f) == n, "write failed: " + out); done += n;
        }
        require(std::fclose(f) == 0, "write failed: " + out);
        std::printf("PACK_DONE %s\n", out.c_str());
        return 0;
    } catch (const std::exception & e) { std::fprintf(stderr, "tqfn-pack: %s\n", e.what()); return 1; }
}
