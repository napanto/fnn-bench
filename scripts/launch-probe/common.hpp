// Shared timing helpers of the launch-cost probes. Every probe prints one CSV line per case and repeat:
// model,case,repeat,launches,host_loop_ns,total_ns
#pragma once
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>
inline long long now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
inline int env_int(const char *name, int fallback) {
    const char *v = std::getenv(name);
    return v ? std::atoi(v) : fallback;
}
// host_loop_ns: from the first submission to the return of the last one.
// total_ns: from the first submission to confirmed completion of all work.
inline void emit(const char *model, const char *name, int repeat, int n, long long loop, long long total) {
    std::printf("%s,%s,%d,%d,%lld,%lld\n", model, name, repeat, n, loop, total);
    std::fflush(stdout);
}
constexpr int ARRAY_N = 1024;
