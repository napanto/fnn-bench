// Launch cost against the number of pointer arguments, CUDA.
#include "common.hpp"
#include <cuda_runtime.h>
#define CK(x) do { auto e_ = (x); if ((int)e_) { std::fprintf(stderr, "error at line %d\n", __LINE__); std::exit(1); } } while (0)
__global__ void k12(float *a, const float *b, const float *c, const float *d, const float *e, const float *f, const float *g,
                    const float *h, const float *p, const float *q, const float *r, const float *s, float x, int n) {
    int i = threadIdx.x;
    if (i < n) a[i] = x * b[i] + c[i] + d[i] + e[i] + f[i] + g[i] + h[i] + p[i] + q[i] + r[i] + s[i];
}
int main() {
    const int N = env_int("PROBE_N", 2000), R = env_int("PROBE_R", 9);
    float *v[12];
    for (auto &p : v) CK(cudaMalloc(&p, ARRAY_N * sizeof(float)));
    cudaStream_t s; CK(cudaStreamCreate(&s));
    for (int r = 0; r < R; ++r) {
        long long t0 = now_ns();
        for (int i = 0; i < N; ++i) k12<<<1, 16, 0, s>>>(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9], v[10], v[11], 2.f, 16);
        long long t1 = now_ns();
        CK(cudaStreamSynchronize(s));
        emit("cuda", "kernel-12-args", r, N, t1 - t0, now_ns() - t0);
    }
    return 0;
}
