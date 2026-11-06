// Cost of one kernel launch in CUDA, with a kernel that touches 16 floats.
#include "common.hpp"
#include <cuda_runtime.h>
#define CK(x) do { cudaError_t e_ = (x); if (e_) { std::fprintf(stderr, "%s\n", cudaGetErrorString(e_)); std::exit(1); } } while (0)
__global__ void touch(float *a, const float *b, float s, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] = s * b[i];
}
int main() {
    const int N = env_int("PROBE_N", 2000), R = env_int("PROBE_R", 9);
    float *a, *b;
    CK(cudaMalloc(&a, ARRAY_N * sizeof(float))); CK(cudaMalloc(&b, ARRAY_N * sizeof(float)));
    cudaStream_t s; CK(cudaStreamCreate(&s));
    std::vector<cudaEvent_t> ev(N);
    for (auto &e : ev) CK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
    for (int w = 0; w < 50; ++w) touch<<<1, 16, 0, s>>>(a, b, 2.f, 16);
    CK(cudaStreamSynchronize(s));
    for (int r = 0; r < R; ++r) {
        long long t0 = now_ns();
        for (int i = 0; i < N; ++i) touch<<<1, 16, 0, s>>>(a, b, 2.f, 16);
        long long t1 = now_ns();
        CK(cudaStreamSynchronize(s));
        emit("cuda", "launch", r, N, t1 - t0, now_ns() - t0);
    }
    for (int r = 0; r < R; ++r) {
        long long t0 = now_ns();
        for (int i = 0; i < N; ++i) { touch<<<1, 16, 0, s>>>(a, b, 2.f, 16); CK(cudaEventRecord(ev[i], s)); }
        long long t1 = now_ns();
        CK(cudaStreamSynchronize(s));
        emit("cuda", "launch+event", r, N, t1 - t0, now_ns() - t0);
    }
    for (int r = 0; r < R; ++r) {
        long long t0 = now_ns();
        for (int i = 0; i < N; ++i) { touch<<<1, 16, 0, s>>>(a, b, 2.f, 16); CK(cudaStreamSynchronize(s)); }
        long long t1 = now_ns();
        emit("cuda", "launch+wait", r, N, t1 - t0, t1 - t0);
    }
    return 0;
}
