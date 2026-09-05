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

| device | backend / compiler | BLAS | dtype | GEMM peak (GFLOP/s) | element-wise streaming (GB/s) |
|---|---|---|---|---|---|
| RX 7900 XTX | cudann / clang-22 | rocblas | double | 1120 | 323 |
| RX 7900 XTX | cudann / clang-22 | rocblas | float | 26578 | 932 |
| RX 7900 XTX | cudann / clang-22 | tiled | float | 3009 | 932 |
| RX 7900 XTX | ompnn / clang-22 | rocblas | float | 23312 | 606 |
| RX 7900 XTX | ompnn / gcc-14 | rocblas | float | 23038 | 22 |
| RX 7900 XTX | ompnn / clang-22 | tiled | float | 1858 | 665 |
| RX 7900 XTX | syclnn / acpp | auto | double | 1122 | 322 |
| RX 7900 XTX | syclnn / acpp | auto | float | 26728 | 925 |
| RX 7900 XTX | syclnn / acpp | tiled | float | 2957 | 930 |
| Ryzen 2950X 16-Core Processor | syclnn / clang-22 | mklcpu | double | 201 | 36 |
| Ryzen 2950X 16-Core Processor | syclnn / clang-22 | netlib | double | 201 | 36 |
| Ryzen 2950X 16-Core Processor | syclnn / clang-22 | generic | float | 12 | 35 |
| Ryzen 2950X 16-Core Processor | syclnn / clang-22 | mklcpu | float | 369 | 30 |
| Ryzen 2950X 16-Core Processor | syclnn / clang-22 | netlib | float | 374 | 30 |
| Ryzen 2950X 16-Core Processor (OpenMP host, 16 threads) | ompnn / gcc-14 | mkl | float | 597 | 29 |
| Ryzen 2950X 16-Core Processor (OpenMP host, 16 threads) | ompnn / gcc-14 | openblas | float | 514 | 31 |
| Ryzen 2950X 16-Core Processor (OpenMP host, 16 threads) | ompnn / clang-22 | openblas | float | 352 | 18 |
| Ryzen 2950X 16-Core Processor (OpenMP host, 16 threads) | ompnn / clang-18 | openblas | float | 340 | 16 |
| Ryzen 2950X 16-Core Processor (OpenMP host, 32 threads) | ompnn / clang-22 | openblas | float | 404 | 9 |
| AdaptiveCpp OpenMP host device | syclnn / acpp | auto | float | 561 | 13 |
| GTX 1080 Ti | cudann / nvcc | cublas | double | 408 | 355 |
| GTX 1080 Ti | cudann / nvcc | cublas | float | 9817 | 359 |
| GTX 1080 Ti | ompnn / gcc-14 | cublas | float | 9916 | 159 |
| GTX 1080 Ti | ompnn / clang-18 | cublas | float | 9585 | 243 |
| GTX 1080 Ti | syclnn / clang-22 | auto | double | 409 | 355 |
| GTX 1080 Ti | syclnn / clang-22 | auto | float | 9876 | 359 |
| GTX 1080 Ti | syclnn / clang-22 | tiled | float | 673 | 358 |

The fp64 rate of the RX 7900 XTX is 1/23 of its fp32 rate (RDNA3 has no fast
FP64), which is the "float vs double" discussion point of the report.

### OpenMP-specific caveats

- gcc's offload model maps one OpenMP thread to a warp/wavefront and uses the
  lanes only inside `simd` loops; the two ablation kernels that cannot carry
  `simd` (the `omp atomic` loss of `loss_reduction=false` and the serial
  `bias_gemv=false` bias gradient) therefore run at one lane per warp on gcc
  builds. Those two ablation rows from gcc offload builds are reported but not
  compared with the SYCL/CUDA equivalents, which use one work-item per element.
- `ompnn` executes every operation synchronously: target regions block the
  host until they finish and every vendor BLAS call is followed by a device
  synchronisation (the OpenMP runtime's streams and the BLAS stream are not
  ordered otherwise). This is the interop model's cost and part of what E4
  measures; to separate it from the compiler and runtime, cudann and syclnn
  have the `sync_ops=true` ablation, which waits after every launch, i.e.
  runs them in ompnn's execution model on the same hardware and library
  (E3 and E6 rows). Asynchronous OpenMP (`target nowait` with `depend`
  clauses and cuBLAS on an `omp interop` stream) was verified correct with
  clang-18 on the GTX 1080 Ti but gained nothing on a sequential chain
  (`scripts/../omp_async` feasibility test, 2026-09-05), so ompnn keeps the
  synchronous model and the report quantifies it instead.
- One-time costs (BLAS handle creation, lazy module loading, gcc's PTX JIT)
  land in the first batch; every row discards `--warmup` epochs before timing.

### Two per-epoch numbers

Every training row carries two per-epoch times (inference rows only the first):

- `median_epoch_s`: wall time of one `train(X, Y, batch, epochs)` call divided
  by `epochs`, median over `repeat` calls. It includes the dataset staging and
  upload, the per-epoch loss read-back and every synchronisation: what a user
  of the library pays. With few epochs per call the upload is a sizeable share
  (10-15 % of the call at batch 64, 20-35 % at 256 and 45-65 % at 1024 for the
  MNIST rows at 3 epochs on PCIe 4; smaller in double).
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
samples (E2 and E5 alike) is the latency regime (one launch chain per sample: the number that
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
  reported next to the default in E3; the ratio is a result, not noise. On the
  GTX 1080 Ti the sign flips: `in_order` is 2.2-2.8x *slower* than the four
  streams (monk 1.79 vs 0.81 ms, cup 15.5 vs 5.6 ms, mnist-512-256 107 vs
  79 ms), the CUDA driver overlapping the small kernels where HIP does not.
- `memory=host` with `loss_reduction` relies on 64-bit atomics to fine-grained
  host memory over PCIe (works on ws-amd, may silently fail elsewhere; the
  parity suite catches it).
- ompnn's `sumsq` accumulates in double (nrm2 squared in `T` elsewhere) and its
  host tiled GEMM reassociates the 16-term partial sums: the OpenMP rows are
  not bit-identical to the SYCL/CUDA ones, only within the check tolerances.
- E7 on the GTX 1080 Ti: the syclnn `blas=tiled` rows run with the out-of-order
  queue while the oneMath rows are in-order (the runtime defect described in
  `docs/toolchains.md`), so the tiled/library ratio there mixes the queue
  mode with the kernel; on the RX 7900 XTX both use the same queue.
- E7 (`blas=tiled`): the three kernels share the tile size (16x16), the work
  decomposition (one work-item/thread per C element, op(A) tile stored
  transposed in local/shared memory, op(B) tile broadcast) and the sequential
  k accumulation in a register on the SYCL, CUDA and clang-OpenMP paths;
  gcc's OpenMP offload gives a team 16 wavefronts, so its variant strides over
  the tile elements; the CPU paths vectorise across rows. Tiled/library epoch
  ratios at 4096 wide: syclnn 3.7x, cudann 4.5x, ompnn/amdclang++ 5.4x on the
  RX 7900 XTX; 8.6x / 7.1x / 7.0x (clang-18) on the GTX 1080 Ti. The reductions for
  asum/nrm2 differ in decomposition (SYCL `sycl::reduction`, one CUDA block,
  OpenMP `reduction` in double), affecting only the reported penalty term.
- E7 on gcc's OpenMP offload: a team gets 16 wavefronts, so the tiled kernel
  runs its strided variant with a private accumulator array; it is an order of
  magnitude slower than the one-thread-per-element kernels and the 4096-wide
  row is omitted for the gcc build (hours per row). The gcc rows are reported
  as what gcc's offload model delivers, not as the same kernel.
- `sync_every` (all three libraries accept it; only syclnn and cudann act on
  it): a queue/stream drain every N batches. syclnn sets it to 4 on CPU
  devices because the OpenCL CPU runtime's per-submission cost grows with the
  outstanding commands (see `docs/toolchains.md`); GPUs run unbounded. The
  effective value is in the `effective_options` of every row measured with a
  build that has the option (syclnn >= v0.2.0-5, cudann/ompnn >= the same day);
  earlier GPU rows behave as `sync_every = 0`.
- W4 on the CPU (`plans/w4_sweep_cpu.json`) is float only, widths 256 and
  1024, depths 2 and 4 on 16384 samples: the 4096-wide and 8-deep double
  configurations take hours per row on the CPU and add nothing to the scaling
  picture the GPU sweep gives.
- Provenance: the E4 GPU rows on the RX 7900 XTX carry ompnn v0.1.0-2 and the
  gcc-14 nvptx rows on the GTX 1080 Ti v0.1.0-4, later ompnn rows v0.1.0-8; the
  intervening commits changed only the hand-written tiled kernel and added
  the inert-on-GPU `sync_every` option, so those rows are comparable with the
  rest. Rows that predate the GPU sampler or run shorter than its 200 ms
  interval have no `gpu_monitor`.
- `blas_queue=dedicated` rows are timing rows only: with the profiler on, the
  event recorded for a BLAS call is the marker kernel's, not the library's
  (the breakdown plan never sets it). On the RX 7900 XTX and on the CPU the
  option measures the pure cost of the bracketing, since those backends do
  not need it.
- **The reference workloads are launch-bound on the GPUs, and the BLAS is not
  the difference.** rocBLAS reaches the same rate through oneMath/AdaptiveCpp
  and through hipBLAS/HIP on the RX 7900 XTX (`results/ws-amd/2026-09-05/peaks-sizes`:
  26.77 vs 26.80 TFLOP/s at 8192, 12.2 vs 12.5 at 2048; at 512 one call costs
  111 vs 82 us, the per-call runtime cost showing). mnist-512-256 (float)
  costs the same per batch whatever the batch size, 16x more work per batch
  from 64 to 1024 notwithstanding: cudann/HIP 251 / 279 / 355 us per batch at
  b64 / b256 / b1024, syclnn/AdaptiveCpp 873 / 972 / 1077 us (613 us with
  `queue=in_order`), ompnn/amdclang++ (sync after every op) about 780-930 us.
  A batch is about 30 operations, so the libraries differ in the per-operation
  cost of their runtimes (HIP streams ~10 us, AdaptiveCpp's scheduler ~30 us,
  a synchronous OpenMP target region ~25 us), not in kernel or GEMM speed;
  the E5 breakdown shows the same (device totals 82 / 173 / 191 ms against
  walls of 100 / 297 / 192 ms). The same on the GTX 1080 Ti with DPC++:
  cudann 181 / 240 / 497 us per batch, syclnn 611 / 620 / 690 (588 in-order).
  The W4 sweep is where the GEMMs get long enough for the compute rates to
  matter: the syclnn/cudann epoch ratio at width 4096 is 1.01-1.19x on the
  RX 7900 XTX and 0.99-1.02x on the GTX 1080 Ti (b256 / b4096, depths 2-8),
  against 2.3-3.1x at width 256. Report the two regimes separately: the
  reference workloads measure runtime overhead per operation, the wide sweep
  measures kernels and BLAS.
- CUDA-graph rows have no per-phase profile (the kernels are inside graph
  launches); their `other_ms` is the graph launch time.

### CPU rows: thread counts, affinity and the OpenBLAS variant (third-pass fix-ups)

Four defects in how the CPU rows were run were found while checking the
third pass for run-to-run consistency (`scripts/headline.py` lists every
visible configuration that was measured in more than one sweep); all four are
fixed and the affected rows re-measured (`scripts/ws-amd-cpu-fixups.sh` +
`ws-amd-cpu-fixups-resume.sh` for the thread rows, the AdaptiveCpp host and
the OpenMP host rows; `ws-amd-cpu-fixups-2.sh` for the DPC++ CPU blocks;
`ws-nvidia-fixups.sh` on ws-nvidia; old rows in `superseded.jsonl`):

- **The OpenMP thread-scaling rows ran on one core.** A `--threads N` row
  runs in a child process (the OpenMP and BLAS runtimes read their thread
  count at load). The parent had imported the backend, and libgomp under
  `OMP_PROC_BIND=close` pins the importing thread to its place at load time;
  the child inherited that single core (`cpus_allowed 0,16`, `affinity 2` in
  the superseded rows), so every E4 thread count ran on one core (gcc-14
  host, mnist-512-256 b256: 3.4-6.1 s per epoch for 1 to 32 threads, against
  0.75 s for the in-process default row). The harness now resets the child's
  affinity to the cgroup's effective cpuset (ad-hoc check in the container:
  2 threads 1.90 s, 16 threads 0.84 s; the re-measured E4 series is the
  record). The syclnn E1 children were not pinned (`affinity 32`); their
  series was flat for the next reason.
- **The DPC++ OpenCL CPU device ignores `OMP_NUM_THREADS`.** Its thread count
  is `DPCPP_CPU_NUM_CUS` (the harness sets it with the OpenMP variables now;
  the device keeps reporting 32 compute units, only the times follow). The
  E1 syclnn series had been flat (0.93-0.98 s at every "thread count"); the
  re-measured one scales, MKLCPU mnist-512-256 b256: 3.50 / 2.87 / 1.79 /
  1.01 / 0.79 / 0.94 s at 1 / 2 / 4 / 8 / 16 / 32. With no variable set the
  device uses all 32 hardware threads, 20 % slower than the 16 physical
  cores that every OpenMP row uses, so the DPC++ CPU blocks are re-measured
  with `DPCPP_CPU_NUM_CUS=16` (`ws-amd-cpu-fixups-2.sh`).
- **ompnn loaded the pthread OpenBLAS.** The build pointed at Ubuntu's
  OpenMP-threaded OpenBLAS, but the soname `libopenblas.so.0` was resolved at
  run time through the distro alternatives symlink to the *pthread* build; its
  pool and the pinned OpenMP team contend. Same wheel, mnist-512-256 b256
  float, 16 threads: pthread 0.89-0.94 s, OpenMP variant 0.61-0.63 s (1.5x).
  ompnn now carries an rpath to the chosen variant's directory, and rows
  written since the fix record the mapped libraries in `sysinfo.libs`. The
  opposite holds for the SYCL Netlib
  rows: oneMath's Netlib backend on the OpenCL CPU device runs at 1.02 s with
  the pthread OpenBLAS and 5.1 s with the OpenMP one (a libgomp team spinning
  next to the OpenCL runtime's TBB workers), so those keep the pthread build.
  Each runtime gets the OpenBLAS threading model that matches it; both are
  stock Ubuntu packages.
- **AdaptiveCpp's host device took 32 threads.** The AdaptiveCpp CPU sweeps
  run from the `fnn-rocm` distrobox venv, where nothing set `OMP_NUM_THREADS`
  (the container images set 16), so the host device ran on 32 threads: 1.67 s
  at mnist-512-256 b256 against 0.96 s for the same binary at 16 threads
  (found through the portability run, which had set 16). Re-measured at 16.

- **LLVM's libomp and the OpenMP OpenBLAS do not tolerate thread binding.**
  The clang-18/22 wheels use LLVM's `libomp`, which also answers the GOMP
  calls of the OpenMP-threaded OpenBLAS (one runtime, as intended); under
  `OMP_PROC_BIND=close OMP_PLACES=cores` that combination collapses (clang-22,
  mnist-512-256 b256 float, 16 threads: 4.04 s; `KMP_BLOCKTIME=0`,
  `OMP_WAIT_POLICY=passive` and preloading libomp change nothing) while
  unbound it runs at 0.81 s, faster than the pthread OpenBLAS (0.99-1.01 s
  bound or not). gcc's libgomp prefers the binding (0.63 s bound, 0.77 s
  unbound). The host rows therefore run each runtime in the better of the
  two settings, bound for gcc and MKL (GNU threading layer), unbound for the
  LLVM compilers, and the setting is in every row's `sysinfo.env`; the
  gcc/clang gap (about 1.3x) is partly this runtime-level difference.
- **ws-nvidia's OpenMP host rows ran 16 threads on 12 cores.** The container
  image sets `OMP_NUM_THREADS=16` (ws-amd's core count); ws-nvidia has two
  6-core Xeons (24 hardware threads), so its clang-22 host rows ran 16 threads
  on 12 places. Re-measured at 12 (`scripts/ws-nvidia-fixups.sh`; the matrix
  script sets it too now): 1.33 s at mnist-512-256 b256 against 3.73 s
  before (pthread OpenBLAS, 16 bound threads). The thread series on that
  machine (unbound) peaks at 16 threads (0.84 s) rather than at the 12
  physical cores (1.33-1.47 s), and collapses at 24 (3.5 s) and 32 (5.2 s):
  the Ivy Bridge Xeon gains from a few hyper-threads on these loops and loses
  with all of them, where the Zen+ desktop part loses from 16 on. The default
  rows keep the physical-core rule for comparability; the series (12 and 24
  added to the E4 plan's 1-32 series) is the record of the effect.

Run-to-run spread: the same visible configuration is measured in more than
one sweep of the third pass (E3 default vs E7 `blas=auto` vs the E6 baseline,
E4 vs E7 vendor rows). `scripts/headline.py` tabulates every such pair from
the raw rows (before de-duplication) and summarises the spread at the end of
`analysis/headline-<machine>-<date>.md`. On ws-amd's third pass: the
launch-bound monk/cup rows (epochs under 10 ms) differ by 21 % (median) and
up to 60 % between sweeps, the larger workloads by 2 % (median) with single
outliers around 50 %. Differences below the spread of the row family in
question are not results.

### The profiler is not free: timing rows run unprofiled

The per-launch event pairs of the in-library profiler cost, measured on the
same configuration with the profiler off and on (median epoch, 10 epochs per
call, second pass of 2026-09-04):

| device | library | cup b40 | mnist-512-256 b256 |
|---|---|---|---|
| RX 7900 XTX | cudann (HIP) | 6.5 -> 9.2 ms (+41 %) | 76 -> 106 ms (+39 %) |
| RX 7900 XTX | syclnn (AdaptiveCpp) | 18.1 -> 24.1 ms (+33 %) | 238 -> 281 ms (+18 %) |
| RX 7900 XTX | ompnn (amdclang++) | 14.6 -> 14.7 ms (0 %) | 194 -> 194 ms (0 %) |
| GTX 1080 Ti | cudann (nvcc) | 4.6 -> 6.0 ms (+29 %) | 76 -> 100 ms (+31 %) |
| GTX 1080 Ti | syclnn (DPC++) | 14.9 -> 19.5 ms (+31 %) | 167 -> 225 ms (+34 %) |

ompnn pays nothing because it synchronises after every operation anyway;
the two asynchronous libraries pay 20-40 %, and not equally. The first two
passes (2026-09-03/04) had the profiler on in every timed row, which biased
the OpenMP-vs-others ratios by that much. From the third pass on, every
timing plan runs unprofiled: the libraries always record the per-epoch wall
clock and the epoch/batch counters (one clock read per epoch), which is all
the steady-state metric needs, and the per-phase device times come from one
dedicated profiled plan (`plans/e5_breakdown.json`, the E5 breakdown
figures) that is never used for a timing comparison.

## Profiler cross-validation 

`scripts/rocprof-crosscheck.sh` trains `mnist-512-256` (8 192 samples, batch
256, float) on the RX 7900 XTX under `rocprofv3 --kernel-trace` and compares
the second epoch's kernel time per family with the library's own profile of
that epoch (`results/ws-amd/rocprof/`). The three profilers measure different
things, and the difference is itself a result:

* **rocprofv3** counts kernel execution only (7.6 ms (syclnn), 8.3 ms (cudann) and 10.2 ms (ompnn) per epoch
  backends: the GPU work is the same);
* **cudann** (`cudaEvent` pairs on the stream) and **syclnn** (SYCL event
  profiling) measure the interval between the event before and the event after
  a command on its queue: launch latency and idle gaps while the host is still
  submitting are included, so at this size (8 192 samples) they report 2-3×
  the kernel time — the launch-bound regime made visible;
* **ompnn** (host timers around synchronous regions) also includes the host
  side of every launch and the synchronisation after each vendor BLAS call
  (4.4x the kernel time).

Hence the report quotes device time from the library profilers as "phase time
as seen by the programming model" and uses the rocprof/nsys kernel sums as the
"pure kernel" reference; the difference is the synchronisation/launch cost of
each model.

CPU (`scripts/perf-crosscheck.sh`, `results/ws-amd/perf/README.md`):
`perf record -e cpu-clock` (user space, all threads) attached to a profiled
`fnnbench run` of syclnn (AdaptiveCpp host device) and ompnn (amdclang++
host), MNIST 512-256 at batch 256 on 24 pinned threads. The profiler
attributes 61-70 % of the epoch to the GEMM calls and about 30 % to the
element-wise kernels; perf finds only 11-13 % of the samples inside the
OpenBLAS kernels and 52-56 % inside the OpenMP runtime (fork/join and
spin-wait around GEMMs of 512x784x256 and kernels of a few hundred thousand
elements). Same ordering, different question: the profiler measures the wall
time of each call, perf where the cores are, and at these sizes the host
runtime, not the arithmetic, sets the CPU epoch time.

NVIDIA (`scripts/nsys-crosscheck.py`, `results/ws-nvidia/2026-09-04/nsys-crosscheck.md`):
the MNIST 512-256 float row at batch 256 on the GTX 1080 Ti, traced with
`nsys profile` and summarised with `nsys stats --report cuda_gpu_kern_sum`,
against the E3 row's profiler phases. Per batch the profiler reports 1.8x
(cudann) and 1.5x (syclnn) the pure kernel time, because an event pair
brackets the launch and its queueing, not only the kernel; the phase shares
agree within a few points (GEMM 57 % vs 70 % / 56 % vs 66 %, update 18 vs
15 / 12 vs 13, loss 1.5 vs 1.3 / 6 vs 7) except the bias-gradient GEMV
(11 vs 7 / 17 vs 6 %), the smallest kernels, whose launch overhead dominates
the bracketed interval. Both tools count the same 20 kernels per batch. The
same conclusion as rocprof on the RX 7900 XTX: the profiler is a faithful
breakdown of where the *launch stream* spends its time and over-reports
short kernels; absolute kernel efficiencies come from the peak probes and
the vendor profilers.

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
