// Submission cost of a small cuBLAS GEMM and of a reduction-like kernel in CUDA.
#include "common.hpp"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#define CK(x) do { auto e_ = (x); if ((int)e_) { std::fprintf(stderr, "error %d at line %d\n", (int)e_, __LINE__); std::exit(1); } } while (0)
__global__ void sum_k(const float *a, int n, float *out) {
    float s = 0;
    for (int i = 0; i < n; ++i) s += a[i];
    if (blockIdx.x == 0 && threadIdx.x == 0) *out = s;
}
int main() {
    const int N = env_int("PROBE_N", 2000), R = env_int("PROBE_R", 9), M = env_int("PROBE_M", 8);
    float *a, *b, *c, *o;
    CK(cudaMalloc(&a, M * M * sizeof(float))); CK(cudaMalloc(&b, M * M * sizeof(float))); CK(cudaMalloc(&c, M * M * sizeof(float))); CK(cudaMalloc(&o, 64));
    CK(cudaMemset(a, 0, M * M * sizeof(float))); CK(cudaMemset(b, 0, M * M * sizeof(float)));
    cudaStream_t s; CK(cudaStreamCreate(&s));
    cublasHandle_t h; CK(cublasCreate(&h)); CK(cublasSetStream(h, s));
    const float one = 1.f;
    for (int w = 0; w < 50; ++w) CK(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, M, M, M, &one, a, M, b, M, &one, c, M));
    CK(cudaStreamSynchronize(s));
    for (int r = 0; r < R; ++r) {
        long long t0 = now_ns();
        for (int i = 0; i < N; ++i) CK(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, M, M, M, &one, a, M, b, M, &one, c, M));
        long long t1 = now_ns();
        CK(cudaStreamSynchronize(s));
        emit("cuda", "cublas-gemm", r, N, t1 - t0, now_ns() - t0);
    }
    float host;
    for (int r = 0; r < R; ++r) {
        long long t0 = now_ns();
        for (int i = 0; i < N; ++i) { sum_k<<<1, 16, 0, s>>>(a, M, o); CK(cudaMemcpyAsync(&host, o, sizeof(float), cudaMemcpyDeviceToHost, s)); }
        long long t1 = now_ns();
        CK(cudaStreamSynchronize(s));
        emit("cuda", "sum-kernel+copyback", r, N, t1 - t0, now_ns() - t0);
    }
    return 0;
}
