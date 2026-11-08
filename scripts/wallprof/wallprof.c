// wallprof: wall-clock sampling of one thread by call stack, with no perf events and no privileges.
// A POSIX timer on CLOCK_MONOTONIC sends SIGPROF to the calling thread. Each interval is drawn
// uniformly in [0.5, 1.5] of the period, so the samples do not lock to the rhythm of the batches.
//   gcc -O2 -shared -fPIC wallprof.c -o wallprof.so
// wallprof_start(delay_ms, period_us, max_samples) from the thread under study, wallprof_stop(path)
// from the same thread. The output holds the process mappings and one line per sample. With a null
// path nothing is written: a short period then serves as a measured delay on the host thread, and
// wallprof_handler_ns() returns the time spent in the handler. With WALLPROF_PROCESS set, the timer
// runs on the process CPU-time clock and the signal goes to the process: samples then fall on the
// threads that consume CPU, in proportion to their CPU time, and each records its thread id.
#define _GNU_SOURCE
#include <errno.h>
#include <execinfo.h>
#include <inttypes.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <time.h>
#include <ucontext.h>
#include <unistd.h>

#define DEPTH 96
typedef struct {
    uint64_t wall_ns, cpu_ns, handler_ns, rip;
    uint32_t depth, tid;
    void *pc[DEPTH];
} sample;

static sample *buf;
static long cap, period_ns;
static volatile long n;
static volatile int running;
static uint64_t handler_total;
static timer_t timer;
static int process_mode;
static uint64_t rng = 88172645463325252ULL;

static uint64_t now(clockid_t id) {
    struct timespec t;
    clock_gettime(id, &t);
    return (uint64_t)t.tv_sec * 1000000000ULL + (uint64_t)t.tv_nsec;
}

static void arm(long ns) {
    struct itimerspec it = {{0, 0}, {ns / 1000000000L, ns % 1000000000L}};
    timer_settime(timer, 0, &it, NULL);
}

static void on_signal(int sig, siginfo_t *si, void *context) {
    (void)sig; (void)si;
    if (!running) return;
    int saved = errno;
    uint64_t begin = now(CLOCK_MONOTONIC);
    long slot = __atomic_fetch_add(&n, 1, __ATOMIC_RELAXED);
    if (slot < cap) {
        sample *s = &buf[slot];
        s->wall_ns = begin;
        s->cpu_ns = now(CLOCK_THREAD_CPUTIME_ID);
        s->rip = (uint64_t)((ucontext_t *)context)->uc_mcontext.gregs[REG_RIP];
        s->depth = (uint32_t)backtrace(s->pc, DEPTH);
        s->tid = (uint32_t)syscall(SYS_gettid);
        s->handler_ns = now(CLOCK_MONOTONIC) - begin;
        __atomic_fetch_add(&handler_total, s->handler_ns, __ATOMIC_RELAXED);
    } else {
        __atomic_fetch_sub(&n, 1, __ATOMIC_RELAXED);
    }
    rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
    arm(period_ns / 2 + (long)(rng % (uint64_t)period_ns));
    errno = saved;
}

int wallprof_start(long delay_ms, long period_us, long max_samples) {
    void *probe[4];
    backtrace(probe, 4);  // loads libgcc outside the handler
    cap = max_samples; n = 0; period_ns = period_us * 1000; handler_total = 0;
    buf = mmap(NULL, (size_t)cap * sizeof(sample), PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE, -1, 0);
    if (buf == MAP_FAILED) return -1;
    struct sigaction sa;
    memset(&sa, 0, sizeof sa);
    sa.sa_sigaction = on_signal;
    sa.sa_flags = SA_SIGINFO | SA_RESTART;
    sigemptyset(&sa.sa_mask);
    if (sigaction(SIGPROF, &sa, NULL)) return -2;
    struct sigevent ev;
    memset(&ev, 0, sizeof ev);
    process_mode = getenv("WALLPROF_PROCESS") != NULL;
    ev.sigev_signo = SIGPROF;
    if (process_mode) {
        ev.sigev_notify = SIGEV_SIGNAL;
        if (timer_create(CLOCK_PROCESS_CPUTIME_ID, &ev, &timer)) return -3;
    } else {
        ev.sigev_notify = SIGEV_THREAD_ID;
        ev._sigev_un._tid = (pid_t)syscall(SYS_gettid);
        if (timer_create(CLOCK_MONOTONIC, &ev, &timer)) return -3;
    }
    running = 1;
    arm(delay_ms * 1000000L);
    return 0;
}

uint64_t wallprof_handler_ns(void) { return handler_total; }

long wallprof_stop(const char *path) {
    running = 0;
    timer_delete(timer);
    if (!path) return n;
    FILE *out = fopen(path, "w");
    if (!out) return -1;
    FILE *maps = fopen("/proc/self/maps", "r");
    char line[4096];
    while (maps && fgets(line, sizeof line, maps)) fprintf(out, "M %s", line);
    if (maps) fclose(maps);
    long total = n < cap ? n : cap;
    for (long i = 0; i < total; ++i) {
        const sample *s = &buf[i];
        fprintf(out, "S %" PRIu64 " %" PRIu64 " %" PRIu64 " %" PRIx64, s->wall_ns, s->cpu_ns, s->handler_ns, s->rip);
        if (process_mode) fprintf(out, " t%u", s->tid);
        for (uint32_t d = 0; d < s->depth; ++d) fprintf(out, " %" PRIxPTR, (uintptr_t)s->pc[d]);
        fputc('\n', out);
    }
    fclose(out);
    munmap(buf, (size_t)cap * sizeof(sample));
    return total;
}
