// Cost of one target region, with the same work as probe_cuda.cu. Built by each OpenMP compiler.
#include "common.hpp"
#include <omp.h>
#ifndef COMPILER
#define COMPILER "omp"
#endif
int main() {
    const int N = env_int("PROBE_N", 500), R = env_int("PROBE_R", 9);
    static float a[ARRAY_N], b[ARRAY_N];
#pragma omp target enter data map(alloc : a[:ARRAY_N], b[:ARRAY_N])
    for (int w = 0; w < 50; ++w) {
#pragma omp target teams distribute parallel for
        for (int i = 0; i < 16; ++i) a[i] = 2.f * b[i];
    }
    for (int r = 0; r < R; ++r) {
        long long t0 = now_ns();
        for (int n = 0; n < N; ++n) {
#pragma omp target teams distribute parallel for
            for (int i = 0; i < 16; ++i) a[i] = 2.f * b[i];
        }
        long long t1 = now_ns();
        emit(COMPILER, "target-teams-parallel", r, N, t1 - t0, t1 - t0);
    }
    for (int r = 0; r < R; ++r) {
        long long t0 = now_ns();
        for (int n = 0; n < N; ++n) {
#pragma omp target teams num_teams(1) thread_limit(1)
            { a[0] = 2.f * b[0]; }
        }
        long long t1 = now_ns();
        emit(COMPILER, "target-single", r, N, t1 - t0, t1 - t0);
    }
#pragma omp parallel num_threads(2)
#pragma omp single
    for (int r = 0; r < R; ++r) {
        long long t0 = now_ns();
        for (int n = 0; n < N; ++n) {
#pragma omp target teams distribute parallel for nowait depend(inout : a[0])
            for (int i = 0; i < 16; ++i) a[i] = 2.f * b[i];
        }
        long long t1 = now_ns();
#pragma omp taskwait
        emit(COMPILER, "target-nowait-depend", r, N, t1 - t0, now_ns() - t0);
    }
    // the clauses of ompnn's kernels: thread_limit, simd, a small firstprivate functor
    struct Functor { float s[8]; } f{};
    for (int r = 0; r < R; ++r) {
        long long t0 = now_ns();
        for (int n = 0; n < N; ++n) {
#pragma omp target teams distribute parallel for simd thread_limit(256) firstprivate(f)
            for (int i = 0; i < 16; ++i) a[i] = f.s[0] * b[i];
        }
        long long t1 = now_ns();
        emit(COMPILER, "target-ompnn-clauses", r, N, t1 - t0, t1 - t0);
    }
    for (int r = 0; r < R; ++r) {
        double total = 0;
        long long t0 = now_ns();
        for (int n = 0; n < N; ++n) {
#pragma omp target teams distribute parallel for simd thread_limit(256) firstprivate(f) reduction(+ : total)
            for (int i = 0; i < 16; ++i) total += f.s[0] * b[i];
        }
        long long t1 = now_ns();
        emit(COMPILER, "target-ompnn-reduction", r, N, t1 - t0, t1 - t0);
    }
#pragma omp target exit data map(delete : a[:ARRAY_N], b[:ARRAY_N])
    return 0;
}
