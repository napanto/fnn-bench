// Launch cost against the number of pointer arguments, DPC++ on the CUDA backend.
#include "common.hpp"
#include <sycl/sycl.hpp>
class K12; class K12D;
int main() {
    const int N = env_int("PROBE_N", 2000), R = env_int("PROBE_R", 9);
    sycl::queue q{sycl::gpu_selector_v, {sycl::property::queue::in_order{}}};
    float *v[12];
    for (auto &p : v) p = sycl::malloc_device<float>(ARRAY_N, q);
    float *a = v[0], *b = v[1], *c = v[2], *d = v[3], *e = v[4], *f = v[5], *g = v[6], *h = v[7], *p = v[8], *w = v[9], *x = v[10], *y = v[11];
    auto body = [=](sycl::nd_item<1> it) { auto i = it.get_global_id(0); a[i] = 2.f * b[i] + c[i] + d[i] + e[i] + f[i] + g[i] + h[i] + p[i] + w[i] + x[i] + y[i]; };
    for (int r = 0; r < R; ++r) {
        long long t0 = now_ns();
        for (int i = 0; i < N; ++i) q.submit([&](sycl::handler &hd) { hd.parallel_for<K12>(sycl::nd_range<1>(16, 16), body); });
        long long t1 = now_ns();
        q.wait();
        emit("dpcpp", "kernel-12-args", r, N, t1 - t0, now_ns() - t0);
    }
    for (int r = 0; r < R; ++r) {
        long long t0 = now_ns();
        std::vector<sycl::event> deps; sycl::event last;
        for (int i = 0; i < N; ++i) {
            last = q.submit([&](sycl::handler &hd) { hd.depends_on(deps); hd.parallel_for<K12D>(sycl::nd_range<1>(16, 16), body); });
            deps.assign(3, last);
        }
        long long t1 = now_ns();
        q.wait();
        emit("dpcpp", "kernel-12-args+3deps", r, N, t1 - t0, now_ns() - t0);
    }
    return 0;
}
