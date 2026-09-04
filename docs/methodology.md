# Benchmark methodology

How the numbers in `results/` are produced, what they mean, and what they do not
mean. This is the reference for the "Methodology" chapter of the report.

## Workloads 

| id | network | data | batch | why |
|---|---|---|---|---|
| W1 `monk` | 17-4-1, tanh / sigmoid, Adam (lr 0.01), L2 1e-3 | MONK-1 train (124 samples) | 40 | the ML project's real configuration: launch-overhead regime |
| W2 `cup` | 12-64-64-3, tanh / linear, Adam (lr 1e-3), L2 1e-4 | ML-CUP-like synthetic regression (1000 samples) | 40 | realistic small regression (the CUP data itself is course-private) |
| W3 `mnist` / `mnist-512-256` | 784-1024-1024-10 / 784-512-256-10, ReLU / sigmoid, SGD + momentum 0.9 (lr 0.1) | MNIST (60 000 samples) | 64 / 256 / 1024 | recognisable, GEMM-bound from batch 256 |
| W4 `sweep-w{256,1024,4096}-d{2,4,8}-b{32,256,4096}` | 256-w×d-16, ReLU / linear, Adam | synthetic (65 536 samples; 16 384 on the CPU) | 32 / 256 / 4096 | scaling laws and roofline coverage |
| W5 `infer` | W1-W3 forward only | as above | 1 (latency) and 4096 (throughput) | the inference half of the study |

Every workload runs in `float` and `double`. Definitions: `testkit/fnn_testkit/workloads.py`.

## Protocol 

`fnnbench run` / `fnnbench sweep` (`bench/fnnbench/runner.py`):

1. the network is built from fixed initial weights (seeded NumPy RNG, uniform in
   ±0.1) through the same adapter the parity tests use;
2. **numerical check** before anything is timed: the untrained forward pass on
   256 samples must match the float64 oracle at the parity tolerance (2e-4 for
   float, 1e-9 for double, relative to the largest output) and the first-epoch loss must match the
   oracle's within 5e-2 (float; training is chaotic in the summation order) /
   1e-6 (double). A run that computes the wrong thing is stored with
   `check.ok = false` and never enters a figure;
3. one untimed warm-up call (JIT compilation, rocBLAS/cuBLAS kernel loading,
   workspace allocation);
4. `repeat` (5 by default; 3 for the MNIST and W4 sweeps) timed calls of
   `epochs` epochs each. One call = one C++ `train()`: dataset upload, all
   epochs, the per-epoch loss read-back. The reported per-epoch wall time is the
   call's wall time divided by `epochs`, so the dataset upload is amortised over
   `epochs` (rows with different `epochs` are not directly comparable);
5. median and IQR of the per-epoch wall times, samples/s, steps/s;
6. the library's per-phase profile of the timed calls only (`reset_profile()`
   after the warm-up), normalised per epoch;
7. the FLOP model (below) and the arithmetic intensity of the step;
8. the losses of every timed call, checked to be finite (weights keep training
   across repetitions; only timing is measured there).

Thread counts (`--threads`) run in a fresh interpreter with `OMP_NUM_THREADS`,
`MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `OMP_PROC_BIND=close`,
`OMP_PLACES=cores`, because the OpenMP and BLAS runtimes read them once.

Every JSONL row records: run id (hash of the effective configuration), the
backend's `build_info` (compiler, flags, offload targets, BLAS backends, git
revision), the device (name, vendor, driver, memory, compute units, compute
capability), the BLAS backend actually used, dtype, workload, layer sizes,
sample count, batch, epochs, repetitions, mode, thread count and the runtime's
`omp_get_max_threads`, the requested and effective `Options`, the results
above, and `sysinfo` (host, kernel, CPU model / ISA / caches / governor, NUMA
masks, GPU clocks / power / throttle reasons from `nvidia-smi` or `rocm-smi`,
container image, filtered environment, package versions, git SHA of the bench).
`fnnbench replay <file> [--id]` re-runs any row; `fnnbench collect` flattens
everything into one CSV.

## FLOP model 

For layer sizes n_0 … n_L and batch B (`bench/fnnbench/flops.py`):

* forward GEMMs 2·B·Σ n_l·n_{l+1}; hidden-delta GEMMs 2·B·Σ_{l≥1} n_l·n_{l+1};
  weight-gradient GEMMs 2·B·Σ n_l·n_{l+1} — together **GEMM FLOPs**;
* element-wise kernels ≈ 8·B·Σ_{l≥1} n_l and the update ≈ 12·(parameters) —
  added for **total FLOPs**;
* inference: 2·N·Σ n_l·n_{l+1} (+4·N·Σ n_l).

`gflops_gemm` = GEMM FLOPs per epoch / median epoch time. The arithmetic
intensity of a step is Σ 2MNK / Σ (MK+KN+MN)·sizeof(T) over the step's GEMMs
(forward-only for inference rows), used for the roofline placement against the
measured peaks.

## Peaks

`fnnbench peak` runs a one-layer identity network (one n×n×n GEMM plus one
bias/activation kernel) through each backend and reports the BLAS library's
achieved GFLOP/s and a streaming bandwidth from the activation kernel
(3·n²·sizeof(T) bytes). Measured on ws-amd (`results/ws-amd/peaks/`):

| device | backend | GEMM float | GEMM double | streaming |
|---|---|---|---|---|
| RX 7900 XTX (8192³) | syclnn / AdaptiveCpp + rocBLAS | 26.2 TFLOP/s | 1.12 TFLOP/s | 808 GB/s |
| RX 7900 XTX | cudann / HIP + hipBLAS | 26.2 TFLOP/s | 1.11 TFLOP/s | 892 GB/s |
| RX 7900 XTX | ompnn / amdclang++ + hipBLAS | 23.2 TFLOP/s | – | 600 GB/s |
| Threadripper 2950X (4096³) | syclnn / DPC++ opencl:cpu + MKLCPU | 369 GFLOP/s | 199 GFLOP/s | 30 GB/s |
| Threadripper 2950X | syclnn / DPC++ opencl:cpu + NETLIB (OpenBLAS) | 371 GFLOP/s | 201 GFLOP/s | 30 GB/s |
| Threadripper 2950X | syclnn / AdaptiveCpp OpenMP host + OpenBLAS | 561 GFLOP/s | – | 13 GB/s |
| Threadripper 2950X | ompnn / amdclang++ + OpenBLAS | 404 GFLOP/s | – | 9 GB/s |

The fp64 rate of the RX 7900 XTX is 1/23 of its fp32 rate (RDNA3 has no fast
FP64), which is the "float vs double" discussion point of the report.

### OpenMP-specific caveats

- gcc's offload model maps one OpenMP thread to a warp/wavefront and uses the
  lanes only inside `simd` loops; the two ablation kernels that cannot carry
  `simd` (the `omp atomic` loss of `loss_reduction=false` and the serial
  `bias_gemv=false` bias gradient) therefore run at one lane per warp on gcc
  builds. Those two ablation rows from gcc offload builds are reported but not
  compared with the SYCL/CUDA equivalents, which use one work-item per element.
- `ompnn` synchronises after every vendor BLAS call (`cudaDeviceSynchronize`
  / `hipDeviceSynchronize`): the OpenMP runtime's queues and the BLAS stream
  are not ordered otherwise. This is a genuine cost of the interop model and
  is part of what E4 measures, not an artefact.
- One-time costs (BLAS handle creation, lazy module loading, gcc's PTX JIT)
  land in the first batch; every row discards `--warmup` epochs before timing.

### Two per-epoch numbers

Every row carries two per-epoch times:

- `median_epoch_s`: wall time of one `train(X, Y, batch, epochs)` call divided
  by `epochs`, median over `repeat` calls. It includes the dataset staging and
  upload, the per-epoch loss read-back and every synchronisation: what a user
  of the library pays. With few epochs per call the upload is a sizeable share
  (about a third of the MNIST rows at 3 epochs on PCIe 4).
- `steady_epoch_s`: median of the library's own per-epoch wall clocks
  (`Profile.epoch_wall_ns`) over all timed calls, excluding the first epoch of
  each call. The first epoch carries the asynchronous dataset upload and lazy
  initialisation (BLAS handles, module loading, JIT); in CUDA-graph mode the
  graphs are captured once, by the warm-up call, and replayed by every later
  call of the same shape, so no capture is timed. This is the number used for
  the programming-model comparison (E3, E4, E6); the call-level number is
  reported next to it.

### Inference rows (W5)

`mode = infer` rows time `predict(X, batch)` over the whole set (`epochs`
passes, `repeat` calls, same warm-up/median protocol). Batch 1 on 2048
samples is the latency regime (one launch chain per sample: the number that
shows the launch overhead of each programming model); batch 4096 is the
throughput regime. The FLOP model is the forward GEMM only, 2B*sum(n_l n_{l+1}).

### GPU utilisation and power

Every GPU row carries `gpu_monitor`: mean/max utilisation, mean/max power,
mean clock and the energy of the timed region, sampled every 200 ms from the
amdgpu sysfs counters (`gpu_busy_percent`, hwmon `power1_average`,
`freq1_input`) or from `nvidia-smi --query-gpu ... -lms 200`. The sampling
thread costs nothing measurable. The CPU package energy (RAPL) needs root on
ws-amd and is not recorded.

### Interpretation caveats from code review

- `memory=shared` on the RX 7900 XTX (ROCm 7.2, no XNACK) is host-coherent
  memory, not page migration: the AMD "shared" rows measure zero-copy PCIe
  traffic, the NVIDIA ones will measure managed-memory migration (no explicit
  prefetch in either library; first-touch faults land in the warm-up epoch).
- The multi-stream default (`out_of_order`, 4 streams) costs up to 1.7x on the
  launch-bound workloads (monk, cup) under HIP: the event fork/join overhead
  dominates when kernels are microseconds long. The `queue=in_order` rows are
  reported next to the default in E3; the ratio is a result, not noise.
- `memory=host` with `loss_reduction` relies on 64-bit atomics to fine-grained
  host memory over PCIe (works on ws-amd, may silently fail elsewhere; the
  parity suite catches it).
- ompnn's `sumsq` accumulates in double (nrm2 squared in `T` elsewhere) and its
  host tiled GEMM reassociates the 16-term partial sums: the OpenMP rows are
  not bit-identical to the SYCL/CUDA ones, only within the check tolerances.
- E7 (`blas=tiled`): the three kernels share the tile size (16x16), the work
  decomposition (one work-item/thread per C element, op(A) tile stored
  transposed in local/shared memory, op(B) tile broadcast) and the sequential
  k accumulation in a register on the SYCL, CUDA and clang-OpenMP paths;
  gcc's OpenMP offload gives a team 16 wavefronts, so its variant strides over
  the tile elements; the CPU paths vectorise across rows. The reductions for
  asum/nrm2 differ in decomposition (SYCL `sycl::reduction`, one CUDA block,
  OpenMP `reduction` in double), affecting only the reported penalty term.
- E7 on gcc's OpenMP offload: a team gets 16 wavefronts, so the tiled kernel
  runs its strided variant with a private accumulator array; it is an order of
  magnitude slower than the one-thread-per-element kernels and the 4096-wide
  row is omitted for the gcc build (hours per row). The gcc rows are reported
  as what gcc's offload model delivers, not as the same kernel.
- CUDA-graph rows have no per-phase profile (the kernels are inside graph
  launches); their `other_ms` is the graph launch time.

## Profiler cross-validation 

`scripts/rocprof-crosscheck.sh` trains `mnist-512-256` (8 192 samples, batch
256, float) on the RX 7900 XTX under `rocprofv3 --kernel-trace` and compares
the second epoch's kernel time per family with the library's own profile of
that epoch (`results/ws-amd/rocprof/`). The three profilers measure different
things, and the difference is itself a result:

* **rocprofv3** counts kernel execution only (≈ 7-9 ms per epoch for all three
  backends: the GPU work is the same);
* **cudann** (`cudaEvent` pairs on the stream) and **syclnn** (SYCL event
  profiling) measure the interval between the event before and the event after
  a command on its queue: launch latency and idle gaps while the host is still
  submitting are included, so at this size (8 192 samples) they report 2-3×
  the kernel time — the launch-bound regime made visible;
* **ompnn** (host timers around synchronous regions) also includes the host
  side of every launch and the synchronisation after each vendor BLAS call
  (4× the kernel time).

Hence the report quotes device time from the library profilers as "phase time
as seen by the programming model" and uses the rocprof/nsys kernel sums as the
"pure kernel" reference; the difference is the synchronisation/launch cost of
each model.

CPU: `scripts/perf-crosscheck.sh` attaches `perf record -e cpu-clock` (user
space only, no root) to a profiled `fnnbench run` of syclnn (AdaptiveCpp
host) and ompnn (amdclang++ host) and aggregates the samples by DSO class
(BLAS library, the libraries' own kernels, OpenMP/SYCL runtime, Python);
the shares are compared with the profiler's GEMM / element-wise / runtime
split in `results/ws-amd/perf/`.

## Rigor checklist

* warm-up + ≥ 3 repetitions, median + IQR, per-row system state;
* CPU and GPU sweeps never overlap (`scripts/matrix-ws-amd.sh` runs the
  environments sequentially; the OpenBLAS/MKL threads would otherwise perturb
  the GPU launch path);
* AdaptiveCpp's persistent JIT cache is process-shared: never two AdaptiveCpp
  test sessions at once;
* every benchmarked configuration passes the parity suite first
  (`docs/toolchains.md`), and every row carries its own numerical check;
* GPU clocks are recorded (`rocm-smi` / `nvidia-smi`) but not locked (no root on
  the department machine); the CPU governor is recorded.
