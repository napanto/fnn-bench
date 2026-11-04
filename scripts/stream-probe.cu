// Device-resident bandwidth probe. Three 128 MiB arrays by default; decimal GB/s.
// CUDA: nvcc -O3 -arch=sm_80 -std=c++17 scripts/stream-probe.cu -o stream-probe
// HIP:  hipcc -O3 --offload-arch=gfx1100 -std=c++17 -x hip scripts/stream-probe.cu -o stream-probe
#include <algorithm>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#if defined(__HIPCC__)
#include <hip/hip_runtime.h>
#define GPU(name) hip##name
using DeviceProperties = hipDeviceProp_t;
#else
#include <cuda_runtime.h>
#define GPU(name) cuda##name
using DeviceProperties = cudaDeviceProp;
#endif

#define CHECK(expr) do { auto e = (expr); if (e != 0) { \
    std::cerr << #expr << ": " << GPU(GetErrorString)(e) << '\n'; std::exit(1); } } while (0)

__global__ void init(float *a, float *b, float *c, size_t n) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) {
        a[i] = 1.25f; b[i] = 2.5f; c[i] = 0.0f;
    }
}

template<int op>
__global__ void stream_op(const float *__restrict__ a, const float *__restrict__ b,
                          float *__restrict__ c, size_t n) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) {
        if constexpr (op == 0) c[i] = a[i];
        if constexpr (op == 1) c[i] = 3.0f * a[i];
        if constexpr (op == 2) c[i] = a[i] + b[i];
        if constexpr (op == 3) c[i] = a[i] + 3.0f * b[i];
    }
}

template<int op>
void measure(const char *name, float *a, float *b, float *c, size_t n, int reps, int blocks) {
    GPU(Event_t) start, stop;
    CHECK(GPU(EventCreate)(&start)); CHECK(GPU(EventCreate)(&stop));
    for (int i = 0; i < 5; ++i) stream_op<op><<<blocks, 256>>>(a, b, c, n);
    CHECK(GPU(GetLastError)()); CHECK(GPU(DeviceSynchronize)());
    std::vector<float> times;
    for (int i = 0; i < reps; ++i) {
        CHECK(GPU(EventRecord)(start));
        stream_op<op><<<blocks, 256>>>(a, b, c, n);
        CHECK(GPU(GetLastError)()); CHECK(GPU(EventRecord)(stop)); CHECK(GPU(EventSynchronize)(stop));
        float ms; CHECK(GPU(EventElapsedTime)(&ms, start, stop)); times.push_back(ms);
    }
    float sample;
    CHECK(GPU(Memcpy)(&sample, c + n - 1, sizeof(float), GPU(MemcpyDeviceToHost)));
    const float expected[] = {1.25f, 3.75f, 3.75f, 8.75f};
    if (sample != expected[op]) throw std::runtime_error("probe validation failed");
    std::sort(times.begin(), times.end());
    const double med = (times[(reps-1)/2] + times[reps/2]) / 2.0;
    const size_t bytes = n * sizeof(float) * (op < 2 ? 2 : 3);
    std::cout << name << ',' << n << ',' << n*3*sizeof(float) << ',' << bytes << ',' << reps << ','
              << times.front() << ',' << med << ',' << bytes/times.front()/1e6 << ',' << bytes/med/1e6 << '\n';
    CHECK(GPU(EventDestroy)(start)); CHECK(GPU(EventDestroy)(stop));
}

int main(int argc, char **argv) {
    const size_t n = argc > 1 ? std::stoull(argv[1]) : 32ull*1024*1024;
    const int reps = argc > 2 ? std::stoi(argv[2]) : 31;
    if (!n || reps < 3) throw std::runtime_error("need positive size and >=3 repetitions");
    DeviceProperties prop; CHECK(GPU(GetDeviceProperties)(&prop, 0));
    int runtime, driver;
    CHECK(GPU(RuntimeGetVersion)(&runtime)); CHECK(GPU(DriverGetVersion)(&driver));
    std::cout << "# device=" << prop.name << " runtime=" << runtime << " driver=" << driver << '\n';
#if defined(__HIPCC__)
    std::cout << "# architecture=" << prop.gcnArchName << '\n';
#else
    std::cout << "# architecture=sm_" << prop.major << prop.minor << '\n';
#endif
    float *a, *b, *c;
    CHECK(GPU(Malloc)((void**)&a, n*sizeof(float))); CHECK(GPU(Malloc)((void**)&b, n*sizeof(float)));
    CHECK(GPU(Malloc)((void**)&c, n*sizeof(float)));
    const int blocks = prop.multiProcessorCount * 32;
    init<<<blocks, 256>>>(a, b, c, n); CHECK(GPU(GetLastError)()); CHECK(GPU(DeviceSynchronize)());
    std::cout << "operation,elements,arrays_bytes,traffic_bytes,repetitions,best_ms,median_ms,best_GB_s,median_GB_s\n" << std::setprecision(9);
    measure<0>("copy", a, b, c, n, reps, blocks); measure<1>("scale", a, b, c, n, reps, blocks);
    measure<2>("add", a, b, c, n, reps, blocks); measure<3>("triad", a, b, c, n, reps, blocks);
    CHECK(GPU(Free)(a)); CHECK(GPU(Free)(b)); CHECK(GPU(Free)(c));
}
