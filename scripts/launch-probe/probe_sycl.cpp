// Cost of one kernel launch in DPC++ on the CUDA backend, with the same kernel as probe_cuda.cu.
#include "common.hpp"
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/experimental/enqueue_functions.hpp>
namespace syclex = sycl::ext::oneapi::experimental;
template <class Q> static void kernel_cg(sycl::handler &h, float *a, const float *b, float s) {
    h.parallel_for<Q>(sycl::nd_range<1>(16, 16), [=](sycl::nd_item<1> it) { auto i = it.get_global_id(0); a[i] = s * b[i]; });
}
class K1; class K2; class K3; class K4; class K5; class K6; class K7;
int main() {
    const int N = env_int("PROBE_N", 2000), R = env_int("PROBE_R", 9);
    sycl::queue inorder{sycl::gpu_selector_v, {sycl::property::queue::in_order{}}};
    sycl::queue discard{inorder.get_context(), inorder.get_device(),
                        {sycl::property::queue::in_order{}, sycl::ext::oneapi::property::queue::discard_events{}}};
    sycl::queue ooo{inorder.get_context(), inorder.get_device()};
    float *a = sycl::malloc_device<float>(ARRAY_N, inorder), *b = sycl::malloc_device<float>(ARRAY_N, inorder);
    for (int w = 0; w < 50; ++w) {
        inorder.submit([&](sycl::handler &h) { kernel_cg<K1>(h, a, b, 2.f); });
        discard.submit([&](sycl::handler &h) { kernel_cg<K2>(h, a, b, 2.f); });
        ooo.submit([&](sycl::handler &h) { kernel_cg<K3>(h, a, b, 2.f); });
    }
    inorder.wait(); discard.wait(); ooo.wait();
    for (int r = 0; r < R; ++r) {
        long long t0 = now_ns();
        for (int i = 0; i < N; ++i) inorder.submit([&](sycl::handler &h) { kernel_cg<K1>(h, a, b, 2.f); });
        long long t1 = now_ns();
        inorder.wait();
        emit("dpcpp", "in-order", r, N, t1 - t0, now_ns() - t0);
    }
    for (int r = 0; r < R; ++r) {
        long long t0 = now_ns();
        for (int i = 0; i < N; ++i) discard.submit([&](sycl::handler &h) { kernel_cg<K2>(h, a, b, 2.f); });
        long long t1 = now_ns();
        discard.wait();
        emit("dpcpp", "in-order+discard_events", r, N, t1 - t0, now_ns() - t0);
    }
    for (int r = 0; r < R; ++r) {
        long long t0 = now_ns();
        for (int i = 0; i < N; ++i)
            syclex::submit(inorder, [&](sycl::handler &h) { kernel_cg<K4>(h, a, b, 2.f); });
        long long t1 = now_ns();
        inorder.wait();
        emit("dpcpp", "in-order+eventless", r, N, t1 - t0, now_ns() - t0);
    }
    for (int r = 0; r < R; ++r) {
        long long t0 = now_ns();
        sycl::event last;
        for (int i = 0; i < N; ++i)
            last = ooo.submit([&](sycl::handler &h) { h.depends_on(last); kernel_cg<K3>(h, a, b, 2.f); });
        long long t1 = now_ns();
        last.wait();
        emit("dpcpp", "out-of-order+dep", r, N, t1 - t0, now_ns() - t0);
    }
    for (int r = 0; r < R; ++r) {
        long long t0 = now_ns();
        for (int i = 0; i < N; ++i) inorder.submit([&](sycl::handler &h) { kernel_cg<K5>(h, a, b, 2.f); }).wait();
        long long t1 = now_ns();
        emit("dpcpp", "in-order+wait", r, N, t1 - t0, t1 - t0);
    }
    for (int r = 0; r < R; ++r) {
        long long t0 = now_ns();
        for (int i = 0; i < N; ++i)
            inorder.submit([&](sycl::handler &h) { h.ext_codeplay_enqueue_native_command([](sycl::interop_handle) {}); });
        long long t1 = now_ns();
        inorder.wait();
        emit("dpcpp", "native-command(empty)", r, N, t1 - t0, now_ns() - t0);
    }
    for (int r = 0; r < R; ++r) {
        long long t0 = now_ns();
        for (int i = 0; i < N; ++i)
            inorder.submit([&](sycl::handler &h) { h.host_task([] {}); });
        long long t1 = now_ns();
        inorder.wait();
        emit("dpcpp", "host_task(empty)", r, N, t1 - t0, now_ns() - t0);
    }
    return 0;
}
