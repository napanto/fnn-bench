// Do kernels on different streams overlap without a profiler? cuBLAS GEMMs of order M, in one stream
// and on separate streams, freely and with the shape of the backward pass: one product, then two
// products that depend only on it, then a join.
#include "common.hpp"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#define CK(x) do { cudaError_t e_ = (x); if (e_) { std::fprintf(stderr, "%s\n", cudaGetErrorString(e_)); std::exit(1); } } while (0)
#define CB(x) do { if ((x) != CUBLAS_STATUS_SUCCESS) { std::fprintf(stderr, "cublas\n"); std::exit(1); } } while (0)
struct Product {
    float *a, *b, *c;
    cublasHandle_t h;
    int m;
    void run() const {
        const float one = 1.f;
        CB(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, m, m, m, &one, a, m, b, m, &one, c, m));
    }
    void on(cudaStream_t s) const { CB(cublasSetStream(h, s)); }
};
int main() {
    const int R = env_int("PROBE_R", 9);
    cudaStream_t s[3];
    for (auto &x : s) CK(cudaStreamCreateWithFlags(&x, cudaStreamNonBlocking));
    cudaEvent_t e[3];
    for (auto &x : e) CK(cudaEventCreateWithFlags(&x, cudaEventDisableTiming));
    for (int m : {256, 512, 1024, 2048}) {
        const int N = m >= 2048 ? 40 : m >= 1024 ? 200 : 1000;
        Product p[3];
        for (auto &q : p) {
            q.m = m;
            CK(cudaMalloc(&q.a, sizeof(float) * m * m)); CK(cudaMalloc(&q.b, sizeof(float) * m * m)); CK(cudaMalloc(&q.c, sizeof(float) * m * m));
            CK(cudaMemset(q.a, 0, sizeof(float) * m * m)); CK(cudaMemset(q.b, 0, sizeof(float) * m * m)); CK(cudaMemset(q.c, 0, sizeof(float) * m * m));
            CB(cublasCreate(&q.h));
        }
        char name[48];
        auto sync = [&] { for (auto &x : s) CK(cudaStreamSynchronize(x)); };
        for (int k = 0; k < 3; ++k) p[k].on(s[0]);
        for (int w = 0; w < 20; ++w) for (auto &q : p) q.run();
        sync();
        for (int r = 0; r < R; ++r) {  // one product
            long long t0 = now_ns();
            for (int i = 0; i < N; ++i) p[0].run();
            long long t1 = now_ns();
            sync();
            std::snprintf(name, sizeof name, "one-m%d", m);
            emit("cuda", name, r, N, t1 - t0, now_ns() - t0);
        }
        for (int r = 0; r < R; ++r) {  // two products per step, one stream
            long long t0 = now_ns();
            for (int i = 0; i < N; ++i) { p[1].run(); p[2].run(); }
            long long t1 = now_ns();
            sync();
            std::snprintf(name, sizeof name, "pair-one-stream-m%d", m);
            emit("cuda", name, r, N, t1 - t0, now_ns() - t0);
        }
        p[1].on(s[1]); p[2].on(s[2]);
        for (int r = 0; r < R; ++r) {  // two products per step, two free streams
            long long t0 = now_ns();
            for (int i = 0; i < N; ++i) { p[1].run(); p[2].run(); }
            long long t1 = now_ns();
            sync();
            std::snprintf(name, sizeof name, "pair-two-streams-m%d", m);
            emit("cuda", name, r, N, t1 - t0, now_ns() - t0);
        }
        p[1].on(s[0]); p[2].on(s[0]);
        for (int r = 0; r < R; ++r) {  // backward shape, in order
            long long t0 = now_ns();
            for (int i = 0; i < N; ++i) { p[0].run(); p[1].run(); p[2].run(); }
            long long t1 = now_ns();
            sync();
            std::snprintf(name, sizeof name, "fork-one-stream-m%d", m);
            emit("cuda", name, r, N, t1 - t0, now_ns() - t0);
        }
        p[1].on(s[1]); p[2].on(s[2]);
        for (int r = 0; r < R; ++r) {  // backward shape, the two branches on their own streams, joined by events
            long long t0 = now_ns();
            for (int i = 0; i < N; ++i) {
                p[0].run();
                CK(cudaEventRecord(e[0], s[0]));
                CK(cudaStreamWaitEvent(s[1], e[0], 0)); p[1].run(); CK(cudaEventRecord(e[1], s[1]));
                CK(cudaStreamWaitEvent(s[2], e[0], 0)); p[2].run(); CK(cudaEventRecord(e[2], s[2]));
                CK(cudaStreamWaitEvent(s[0], e[1], 0)); CK(cudaStreamWaitEvent(s[0], e[2], 0));
            }
            long long t1 = now_ns();
            sync();
            std::snprintf(name, sizeof name, "fork-three-streams-m%d", m);
            emit("cuda", name, r, N, t1 - t0, now_ns() - t0);
        }
        for (auto &q : p) { CB(cublasDestroy(q.h)); CK(cudaFree(q.a)); CK(cudaFree(q.b)); CK(cudaFree(q.c)); }
    }
    return 0;
}
