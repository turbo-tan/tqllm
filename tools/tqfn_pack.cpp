// tqfn-pack: writes the Flash-Next packed experts file (experts-packed.tqfnexp): every expert of every layer as one block
// in the decode cache's tiled row layout, so a cache miss is a single copy.
#include "fn_model.hpp"

int main(int argc, char ** argv) {
    try {
        require(argc >= 3, "usage: tqfn-pack OUT.tqfnexp shard1.gguf [more shards] [--gsq-rco]");
        const std::string out = argv[1];
        std::vector<char *> args{argv[0], argv[0]}; for (int i = 2; i < argc; ++i) args.push_back(argv[i]);
        Checkpoint checkpoint(int(args.size()), args.data()); const auto & c = checkpoint.config;
        FILE * f = std::fopen(out.c_str(), "wb"); require(f, "cannot create " + out);
        g_tile_experts = true;
        std::fwrite("TQFNEXPT", 1, 8, f); int32_t dims[2] = {c.layers, c.experts}; std::fwrite(dims, 4, 2, f);
        std::vector<std::array<uint64_t, 6>> sizes(c.layers);
        for (int l = 0; l < c.layers; ++l) { const auto parts = repack_expert(checkpoint, l, 0); for (int k = 0; k < 6; ++k) sizes[l][k] = parts[k].size(); std::fwrite(sizes[l].data(), 8, 6, f); }
        for (int l = 0; l < c.layers; ++l) {
            for (int e = 0; e < c.experts; ++e) {
                const auto parts = repack_expert(checkpoint, l, e);
                for (int k = 0; k < 6; ++k) { require(parts[k].size() == sizes[l][k], "expert layout varies within a layer"); std::fwrite(parts[k].data(), 1, parts[k].size(), f); }
            }
            std::printf("packed layer %d\n", l); std::fflush(stdout);
        }
        require(std::fclose(f) == 0, "write failed: " + out);
        std::printf("PACK_DONE %s\n", out.c_str());
        return 0;
    } catch (const std::exception & e) { std::fprintf(stderr, "tqfn-pack: %s\n", e.what()); return 1; }
}
