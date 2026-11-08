// What does the dependency between consecutive kernels cost on the GPU, and what do events cost on
// the host? N launches of a kernel of fixed length (a clock64 spin of C cycles in every block):
//   one-stream          each kernel waits for the previous one (stream order)
//   one-stream-events   the same, plus the event record and wait cudann issues per operation
//   two-streams-events  alternating over two streams, each kernel waiting for the previous by event:
//                       the same dependency chain, expressed across streams as cudann does
//   two-streams-free    alternating over two streams with no dependency
// With 1 block, kernels of different streams can run side by side; with FULL blocks one kernel
// fills the GPU, so two free streams can only hide the gap between kernels and the tails.
// Loop time = host submission of the N launches; total = until the last kernel has finished.
#include "common.hpp"
#include <cuda_runtime.h>
#define CK(x) do { cudaError_t e_ = (x); if (e_) { std::fprintf(stderr, "%s\n", cudaGetErrorString(e_)); std::exit(1); } } while (0)
__global__ void spin(long long cycles, int *sink) {
    const long long t0 = clock64();
    while (clock64() - t0 < cycles) {}
    if (cycles < 0 && threadIdx.x == 0) *sink = 1;
}
int main() {
    const int R = env_int("PROBE_R", 5), N = env_int("PROBE_N", 20000);
    int sms = 0; CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
    const int full = sms * 16;  // 16 blocks of 128 threads fill an SM (2048 threads)
    cudaStream_t s[2];
    for (auto &x : s) CK(cudaStreamCreateWithFlags(&x, cudaStreamNonBlocking));
    cudaEvent_t e[2];
    for (auto &x : e) CK(cudaEventCreateWithFlags(&x, cudaEventDisableTiming));
    int *sink; CK(cudaMalloc(&sink, sizeof(int)));
    auto sync = [&] { for (auto &x : s) CK(cudaStreamSynchronize(x)); };
    for (long long cycles : {0LL, 2000LL, 10000LL}) {
        for (int blocks : {1, full}) {
            for (int i = 0; i < 2000; ++i) spin<<<blocks, 128, 0, s[i & 1]>>>(cycles, sink);
            sync();
            char name[64];
            auto arm = [&](const char *tag, int streams, bool events) {
                for (int r = 0; r < R; ++r) {
                    const long long t0 = now_ns();
                    for (int i = 0; i < N; ++i) {
                        const int k = streams == 2 ? (i & 1) : 0;
                        if (events && i) CK(cudaStreamWaitEvent(s[k], e[streams == 2 ? 1 - k : 0], 0));
                        spin<<<blocks, 128, 0, s[k]>>>(cycles, sink);
                        if (events) CK(cudaEventRecord(e[k], s[k]));
                    }
                    const long long t1 = now_ns();
                    sync();
                    std::snprintf(name, sizeof name, "%s-c%lld-b%d", tag, cycles, blocks);
                    emit("cuda", name, r, N, t1 - t0, now_ns() - t0);
                }
            };
            arm("one-stream", 1, false);
            arm("one-stream-events", 1, true);
            arm("two-streams-events", 2, true);
            arm("two-streams-free", 2, false);
        }
    }
    CK(cudaFree(sink));
    return 0;
}
