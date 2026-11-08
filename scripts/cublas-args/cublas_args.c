// LD_PRELOAD shim: counts cublasSgemm_v2 and cublasSgemv_v2 calls by their argument tuple
// (operations, sizes, leading dimensions, pointer mode) and writes the table at exit to $CUBLAS_ARGS_OUT.
// It forwards every call unchanged; it records arguments only, never data.
#define _GNU_SOURCE
#include <dlfcn.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
typedef int st_t; typedef void *h_t; typedef int op_t;
typedef st_t (*gemm_t)(h_t, op_t, op_t, int, int, int, const float *, const float *, int, const float *, int, const float *, float *, int);
typedef st_t (*gemv_t)(h_t, op_t, int, int, const float *, const float *, int, const float *, int, const float *, float *, int);
typedef st_t (*pm_t)(h_t, int *);
struct rec { char kind; int ta, tb, m, n, k, lda, ldb, ldc, pmode; long count; };
static void *real_sym(const char *name) {
    void *f = dlsym(RTLD_NEXT, name);  // cuBLAS in the global scope (oneMath)
    if (!f) { void *h = dlopen("libcublas.so.12", RTLD_LAZY | RTLD_NOLOAD); if (h) f = dlsym(h, name); }  // loaded locally (Python extension)
    if (!f) { fprintf(stderr, "cublas_args: %s not found\n", name); abort(); }
    return f;
}
static struct rec recs[512]; static int nrec; static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
static int pmode(h_t h) { static pm_t f; int m = -1; if (!f) f = (pm_t)real_sym("cublasGetPointerMode_v2"); if (f) f(h, &m); return m; }
static void add(char kind, int ta, int tb, int m, int n, int k, int lda, int ldb, int ldc, int pm) {
    pthread_mutex_lock(&mu);
    for (int i = 0; i < nrec; ++i) {
        struct rec *r = &recs[i];
        if (r->kind == kind && r->ta == ta && r->tb == tb && r->m == m && r->n == n && r->k == k && r->lda == lda && r->ldb == ldb && r->ldc == ldc && r->pmode == pm) { r->count++; pthread_mutex_unlock(&mu); return; }
    }
    if (nrec < 512) recs[nrec++] = (struct rec){kind, ta, tb, m, n, k, lda, ldb, ldc, pm, 1};
    pthread_mutex_unlock(&mu);
}
st_t cublasSgemm_v2(h_t h, op_t ta, op_t tb, int m, int n, int k, const float *a, const float *A, int lda, const float *B, int ldb, const float *b, float *C, int ldc) {
    static gemm_t real; if (!real) real = (gemm_t)real_sym("cublasSgemm_v2");
    add('G', ta, tb, m, n, k, lda, ldb, ldc, pmode(h));
    return real(h, ta, tb, m, n, k, a, A, lda, B, ldb, b, C, ldc);
}
st_t cublasSgemv_v2(h_t h, op_t t, int m, int n, const float *a, const float *A, int lda, const float *x, int incx, const float *b, float *y, int incy) {
    static gemv_t real; if (!real) real = (gemv_t)real_sym("cublasSgemv_v2");
    add('V', t, 0, m, n, 0, lda, incx, incy, pmode(h));
    return real(h, t, m, n, a, A, lda, x, incx, b, y, incy);
}
__attribute__((destructor)) static void dump(void) {
    const char *p = getenv("CUBLAS_ARGS_OUT"); if (!p) return;
    FILE *f = fopen(p, "w"); if (!f) return;
    fprintf(f, "kind,ta,tb,m,n,k,lda,ldb_or_incx,ldc_or_incy,pointer_mode,count\n");
    for (int i = 0; i < nrec; ++i) { struct rec *r = &recs[i]; fprintf(f, "%c,%d,%d,%d,%d,%d,%d,%d,%d,%d,%ld\n", r->kind, r->ta, r->tb, r->m, r->n, r->k, r->lda, r->ldb, r->ldc, r->pmode, r->count); }
    fclose(f);
}
