// Submission cost of the oneMath GEMM path and of sycl::reduction in DPC++ on the CUDA backend.
#include "common.hpp"
#include <sycl/sycl.hpp>
#include <oneapi/math.hpp>
class Plain; class Deps; class Red; class Atom;
int main() {
    const int N = env_int("PROBE_N", 2000), R = env_int("PROBE_R", 9), M = env_int("PROBE_M", 8);
    sycl::queue q{sycl::gpu_selector_v, {sycl::property::queue::in_order{}}};
    float *a = sycl::malloc_device<float>(M * M, q), *b = sycl::malloc_device<float>(M * M, q), *c = sycl::malloc_device<float>(M * M, q);
    double *acc = sycl::malloc_host<double>(4, q);
    double *dev_acc = sycl::malloc_device<double>(4, q);
    q.fill(a, 0.f, M * M); q.fill(b, 0.f, M * M); q.wait();
    using namespace oneapi::math;
    auto gemm = [&](std::vector<sycl::event> deps) {
        return blas::column_major::gemm(q, transpose::nontrans, transpose::nontrans, M, M, M, 1.f, a, M, b, M, 1.f, c, M, deps);
    };
    for (int w = 0; w < 50; ++w) gemm({});
    q.wait();
    for (int r = 0; r < R; ++r) {
        long long t0 = now_ns();
        for (int i = 0; i < N; ++i) gemm({});
        long long t1 = now_ns();
        q.wait();
        emit("dpcpp", "onemath-gemm", r, N, t1 - t0, now_ns() - t0);
    }
    for (int r = 0; r < R; ++r) {
        long long t0 = now_ns();
        sycl::event last;
        for (int i = 0; i < N; ++i) last = gemm({last});
        long long t1 = now_ns();
        last.wait();
        emit("dpcpp", "onemath-gemm+dep", r, N, t1 - t0, now_ns() - t0);
    }
    for (int r = 0; r < R; ++r) {
        long long t0 = now_ns();
        sycl::event last;
        for (int i = 0; i < N; ++i)
            last = q.submit([&](sycl::handler &h) { h.depends_on(last); h.parallel_for<Deps>(sycl::nd_range<1>(16, 16), [=](sycl::nd_item<1> it) { auto j = it.get_global_id(0); if (j < 8) c[j] = a[j] + b[j]; }); });
        long long t1 = now_ns();
        q.wait();
        emit("dpcpp", "kernel+dep", r, N, t1 - t0, now_ns() - t0);
    }
    for (int r = 0; r < R; ++r) {
        long long t0 = now_ns();
        for (int i = 0; i < N; ++i)
            q.submit([&](sycl::handler &h) {
                h.parallel_for<Red>(sycl::range<1>(16), sycl::reduction(acc, sycl::plus<double>{}), [=](sycl::id<1> j, auto &sum) { sum += double(a[j[0] % 8]); });
            });
        long long t1 = now_ns();
        q.wait();
        emit("dpcpp", "reduction", r, N, t1 - t0, now_ns() - t0);
    }
    for (int r = 0; r < R; ++r) {
        long long t0 = now_ns();
        for (int i = 0; i < N; ++i) {
            q.submit([&](sycl::handler &h) {
                h.parallel_for<Atom>(sycl::nd_range<1>(16, 16), [=](sycl::nd_item<1> it) {
                    sycl::atomic_ref<double, sycl::memory_order::relaxed, sycl::memory_scope::device> s(dev_acc[0]);
                    s += a[it.get_global_id(0) % 8];
                });
            });
            q.memcpy(acc, dev_acc, sizeof(double));
        }
        long long t1 = now_ns();
        q.wait();
        emit("dpcpp", "atomic+copyback", r, N, t1 - t0, now_ns() - t0);
    }
    return 0;
}
