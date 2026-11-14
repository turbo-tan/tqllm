// tqserve's CUDA context check (tools/serve/cuda_health.hpp). A pending non-sticky error (an allocation out of memory) is
// cleared and serving goes on; a lost context (a kernel fault: sticky) is reported and the process exits with
// kContextLostStatus instead of answering every later request with "illegal memory access" (3090 27B, 11-14, Xid 31).
// Each case runs in its own child: a lost context cannot be recovered in-process. The parent never touches CUDA.
#include "../tools/serve/cuda_health.hpp"
#include <sys/wait.h>
#include <unistd.h>
#include <cstdio>
#include <string>

static int device_present() { int n = 0; return cudaGetDeviceCount(&n) == cudaSuccess && n > 0 ? 0 : 77; }
static int non_sticky() {
    void * p = nullptr;
    if (cudaMalloc(&p, size_t(1) << 50) == cudaSuccess) return 2;  // a 1 PiB allocation must fail
    if (!tqserve::check_cuda_context().empty() || tqserve::context_lost()) return 3;
    if (cudaMalloc(&p, 1 << 20) != cudaSuccess || cudaFree(p) != cudaSuccess) return 4;
    return 0;
}
static int sticky() {
    tqllm_cuda::debug_device_fault();
    if (tqserve::check_cuda_context().empty() || !tqserve::context_lost()) return 5;  // must report the lost context
    sleep(10);
    return 6;  // the exit after the grace period did not come
}
static int run(int (*f)()) {
    const pid_t pid = fork();
    if (pid == 0) { std::fflush(stdout); _exit(f()); }
    int st = 0; waitpid(pid, &st, 0);
    return WIFEXITED(st) ? WEXITSTATUS(st) : 128 + WTERMSIG(st);
}
int main() {
    if (run(device_present) == 77) { std::puts("no CUDA device: skipped"); return 77; }
    int bad = 0;
    const int a = run(non_sticky), b = run(sticky);
    if (a != 0) { std::printf("FAIL: an out-of-memory error was not cleared (child status %d)\n", a); ++bad; }
    if (b != tqserve::kContextLostStatus) { std::printf("FAIL: a lost context did not exit with %d (child status %d)\n", tqserve::kContextLostStatus, b); ++bad; }
    std::puts(bad ? "CUDA_HEALTH_FAILED" : "CUDA_HEALTH_OK");
    return bad ? 1 : 0;
}
