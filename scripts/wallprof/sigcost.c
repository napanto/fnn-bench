// What does one sampling signal cost the thread it interrupts? The same work with and without the
// sampler at several periods: time added per signal, against the time the handler measures itself.
// The work runs 40 calls deep, near the stack depth of the training runs, so backtrace() walks a
// similar number of frames. "compute" stays in registers; "cache" walks a 2 MiB array.
//   gcc -O2 sigcost.c wallprof.so -o sigcost
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
int wallprof_start(long, long, long);
long wallprof_stop(const char *);
uint64_t wallprof_handler_ns(void);

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
static volatile double sink;
static unsigned char *arr;
enum { BYTES = 2 << 20 };

__attribute__((noinline)) static double work(int kind, long units) {
    double acc = 0;
    if (kind == 0) {
        for (long u = 0; u < units; ++u)
            for (int i = 0; i < 1000; ++i) acc = acc * 1.0000001 + i;
    } else {
        for (long u = 0; u < units; ++u)
            for (int i = 0; i < BYTES; i += 64) acc += arr[i];
    }
    return acc;
}
__attribute__((noinline)) static double deep(int d, int kind, long units) {
    if (d == 0) return work(kind, units);
    double r = deep(d - 1, kind, units);
    sink = r;
    return r + d;
}
static double timed(int kind, long units) { double t0 = now(); sink = deep(40, kind, units); return now() - t0; }

int main(void) {
    arr = calloc(BYTES, 1);
    const char *names[] = {"compute", "cache"};
    const long periods[] = {2000, 400, 200, 100};
    for (int kind = 0; kind < 2; ++kind) {
        long units = 1000;
        while (timed(kind, units) < 1.5) units *= 2;
        double base[5];
        for (int r = 0; r < 5; ++r) base[r] = timed(kind, units);
        double b = base[0];
        for (int r = 1; r < 5; ++r) if (base[r] < b) b = base[r];
        for (int p = 0; p < 4; ++p) {
            for (int r = 0; r < 3; ++r) {
                if (wallprof_start(1, periods[p], 2000000)) return 1;
                double t = timed(kind, units);
                long n = wallprof_stop(NULL);
                double h = wallprof_handler_ns() / 1e9;
                printf("%s,%ld,%d,%.6f,%.6f,%ld,%.3f,%.3f\n", names[kind], periods[p], r, b, t, n,
                       1e6 * (t - b) / n, 1e6 * h / n);
                fflush(stdout);
            }
        }
    }
    return 0;
}
