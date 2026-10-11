#include "graph_capture.hpp"
#include <iostream>
#include <cstdint>
#include <stdexcept>

int main() {
    int failures = 0, destroyed = 0;
    auto check = [&](bool ok, const char * name) { if (!ok) { std::cerr << "FAIL: " << name << "\n"; ++failures; } };
    auto graph = reinterpret_cast<cudaGraph_t>(uintptr_t(1));
    auto expected = reinterpret_cast<cudaGraphExec_t>(uintptr_t(2));
    auto destroy = [&](cudaGraph_t g) { check(g == graph, "destroy correct graph"); ++destroyed; return cudaSuccess; };
    auto fail = [](cudaGraphExec_t *, cudaGraph_t) { return cudaErrorMemoryAllocation; };
    bool threw = false;
    try { tqllm::instantiate_graph(graph, fail, destroy); } catch (const std::runtime_error &) { threw = true; }
    check(threw && destroyed == 1, "failed instantiate releases captured graph");
    auto raise = [](cudaGraphExec_t *, cudaGraph_t) -> cudaError_t { throw std::runtime_error("injected failure"); };
    try { tqllm::instantiate_graph(graph, raise, destroy); } catch (const std::runtime_error &) {}
    check(destroyed == 2, "exception releases captured graph");
    auto success = [&](cudaGraphExec_t * out, cudaGraph_t g) { check(g == graph, "instantiate correct graph"); *out = expected; return cudaSuccess; };
    const auto exec = tqllm::instantiate_graph(graph, success, destroy);
    check(exec == expected && destroyed == 3, "next instantiate succeeds and releases graph once");
    if (failures) return 1;
    std::cout << "graph_ownership: 3 paths passed (host-only fault injection)\n";
}
