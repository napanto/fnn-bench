// Radeon: what does it cost to pass work from one HIP stream (one hardware queue) to another, without a
// profiler? N launches of a short kernel (a wall_clock64 spin of T ticks in every block):
//   one-stream          each kernel waits for the previous one (stream order, one queue)
//   two-streams-events  alternating over two streams, each kernel waiting for the previous by event:
//                       the same dependency chain, handed over between queues at every kernel
//   three-streams-events the chain round-robin over three streams, as cudann's three layers
//   two-streams-free    alternating over two streams with no dependency
// Each arm runs R times; one line per repeat (model,case,repeat,launches,host_loop_ns,total_ns).
// Run in many fresh processes: a bimodal hand-over cost across processes would show here untraced.
#include "common.hpp"
#include <hip/hip_runtime.h>
#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::fprintf(stderr, "%s\n", hipGetErrorString(e_)); std::exit(1); } } while (0)
__global__ void spin(long long ticks, int *sink) {
    const long long t0 = wall_clock64();
    while (wall_clock64() - t0 < ticks) {}
    if (ticks < 0 && threadIdx.x == 0) *sink = 1;
}
int main() {
    const int R = env_int("PROBE_R", 5), N = env_int("PROBE_N", 5000);
    hipStream_t s[3];
    for (auto &x : s) CK(hipStreamCreateWithFlags(&x, hipStreamNonBlocking));
    hipEvent_t e[3];
    for (auto &x : e) CK(hipEventCreateWithFlags(&x, hipEventDisableTiming));
    int *sink; CK(hipMalloc(&sink, sizeof(int)));
    auto sync = [&] { for (auto &x : s) CK(hipStreamSynchronize(x)); };
    for (long long ticks : {0LL, 200LL}) {  // wall_clock64 runs at 100 MHz on gfx11: 200 ticks = 2 us
        for (int blocks : {1, 96}) {
            for (int i = 0; i < 1000; ++i) spin<<<blocks, 256, 0, s[i % 3]>>>(ticks, sink);
            sync();
            char name[64];
            auto arm = [&](const char *tag, int streams, bool events) {
                for (int r = 0; r < R; ++r) {
                    const long long t0 = now_ns();
                    for (int i = 0; i < N; ++i) {
                        const int k = i % streams, prev = (i + streams - 1) % streams;
                        if (events && i) CK(hipStreamWaitEvent(s[k], e[prev], 0));
                        spin<<<blocks, 256, 0, s[k]>>>(ticks, sink);
                        if (events) CK(hipEventRecord(e[k], s[k]));
                    }
                    const long long t1 = now_ns();
                    sync();
                    std::snprintf(name, sizeof name, "%s-t%lld-b%d", tag, ticks, blocks);
                    emit("hip", name, r, N, t1 - t0, now_ns() - t0);
                }
            };
            arm("one-stream", 1, false);
            arm("two-streams-events", 2, true);
            arm("three-streams-events", 3, true);
            arm("two-streams-free", 2, false);
        }
    }
    CK(hipFree(sink));
    return 0;
}
