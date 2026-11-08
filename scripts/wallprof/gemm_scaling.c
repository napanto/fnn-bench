// How far do the MNIST products of one batch scale with oneMKL threads on this host? The eight GEMMs
// of a training batch (784-512-256-10, batch 256: forward, weight gradients, hidden deltas), timed per
// batch at each thread count, as a ceiling for the GEMM part of the epoch.
//   gcc -O2 gemm_scaling.c -I$MKL/include -L$MKL/lib -l:libmkl_rt.so.3 -o gemm_scaling
#include <mkl.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
typedef struct { CBLAS_TRANSPOSE ta, tb; int m, n, k; } G;
int main(void) {
    const int B = 256;
    const G g[8] = {
        {CblasNoTrans, CblasNoTrans, 512, B, 784}, {CblasNoTrans, CblasNoTrans, 256, B, 512}, {CblasNoTrans, CblasNoTrans, 10, B, 256},
        {CblasNoTrans, CblasTrans, 512, 784, B}, {CblasNoTrans, CblasTrans, 256, 512, B}, {CblasNoTrans, CblasTrans, 10, 256, B},
        {CblasTrans, CblasNoTrans, 256, B, 10}, {CblasTrans, CblasNoTrans, 512, B, 256}};
    float *a = mkl_calloc(784 * 784, sizeof(float), 64), *b = mkl_calloc(784 * 784, sizeof(float), 64), *c = mkl_calloc(784 * 784, sizeof(float), 64);
    for (int i = 0; i < 784 * 784; ++i) { a[i] = 1e-3f * (i % 97); b[i] = 1e-3f * (i % 89); }
    const int threads[] = {1, 2, 4, 8, 16, 32, 64};
    for (int t = 0; t < 7; ++t) {
        mkl_set_num_threads(threads[t]);
        double best = 1e9;
        for (int rep = 0; rep < 5; ++rep) {
            int batches = threads[t] <= 2 ? 20 : 100;
            double t0 = now();
            for (int r = 0; r < batches; ++r)
                for (int j = 0; j < 8; ++j) {
                    const G *x = &g[j];
                    int lda = x->ta == CblasNoTrans ? x->m : x->k, ldb = x->tb == CblasNoTrans ? x->k : x->n;
                    cblas_sgemm(CblasColMajor, x->ta, x->tb, x->m, x->n, x->k, 1.f, a, lda, b, ldb, 0.f, c, x->m);
                }
            double per = (now() - t0) / batches;
            if (rep > 0 && per < best) best = per;
        }
        printf("%d,%.6f,%.3f\n", threads[t], 1e3 * best, 235e3 * best);
        fflush(stdout);
    }
    return 0;
}
