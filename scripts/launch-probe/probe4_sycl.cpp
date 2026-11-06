// Where the 15 microseconds of a oneMath GEMM go: cuBLAS called directly from a native command, with and
// without a handle lookup per call, against the empty native command.
#include "common.hpp"
#include <cublas_v2.h>
#include <cuda.h>
#include <sycl/sycl.hpp>
#include <sycl/backend/cuda.hpp>
int main() {
    const int N = env_int("PROBE_N", 2000), R = env_int("PROBE_R", 9), M = env_int("PROBE_M", 8);
    sycl::queue q{sycl::gpu_selector_v, {sycl::property::queue::in_order{}}};
    float *a = sycl::malloc_device<float>(M * M, q), *b = sycl::malloc_device<float>(M * M, q), *c = sycl::malloc_device<float>(M * M, q);
    q.fill(a, 0.f, M * M); q.fill(b, 0.f, M * M); q.wait();
    cublasHandle_t h = nullptr;
    const float one = 1.f;
    auto direct = [&](sycl::handler &cgh) {
        cgh.ext_codeplay_enqueue_native_command([=, &h](sycl::interop_handle ih) {
            if (!h) cublasCreate(&h);
            cublasSetStream(h, ih.get_native_queue<sycl::backend::ext_oneapi_cuda>());
            cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, M, M, M, &one, a, M, b, M, &one, c, M);
        });
    };
    for (int w = 0; w < 50; ++w) q.submit(direct);
    q.wait();
    for (int r = 0; r < R; ++r) {
        long long t0 = now_ns();
        for (int i = 0; i < N; ++i) q.submit(direct);
        long long t1 = now_ns();
        q.wait();
        emit("dpcpp", "native-command+cublas", r, N, t1 - t0, now_ns() - t0);
    }
    for (int r = 0; r < R; ++r) {
        long long t0 = now_ns();
        for (int i = 0; i < N; ++i) q.submit([&](sycl::handler &cgh) { cgh.ext_codeplay_enqueue_native_command([](sycl::interop_handle) {}); });
        long long t1 = now_ns();
        q.wait();
        emit("dpcpp", "native-command-empty", r, N, t1 - t0, now_ns() - t0);
    }
    return 0;
}
