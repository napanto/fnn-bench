# Benchmark methodology

How the rows in `results/` are produced, what each number means and under which
conditions it was measured. The analysis of the results is in the report.

## Workloads

| id | network | data | batch | why |
|---|---|---|---|---|
| W1 `monk` | 17-4-1, tanh / sigmoid, Adam (lr 0.01), L2 1e-3 | MONK-1 train (124 samples) | 40 | the ML project's real configuration: launch-overhead regime |
| W2 `cup` | 12-64-64-3, tanh / linear, Adam (lr 1e-3), L2 1e-4 | ML-CUP-like synthetic regression (1000 samples) | 40 | realistic small regression (the CUP data itself is course-private) |
| W3 `mnist` / `mnist-512-256` | 784-1024-1024-10 / 784-512-256-10, ReLU / sigmoid, SGD + momentum 0.9 (lr 0.1) | MNIST (60 000 samples) | 64 / 256 / 1024 | recognisable, GEMM-bound from batch 256 |
| W4 `sweep-w{256,1024,4096}-d{2,4,8}-b{32,256,4096}` | 256-w×d-16, ReLU / linear, Adam | synthetic (65 536 samples; 16 384 on the CPU) | 32 / 256 / 4096 | scaling laws and roofline coverage |
| W5 `infer` | W1-W3 forward only | as above | 1 (latency) and 4096 (throughput) | the inference half of the study |

Every workload runs in `float` and `double`. Definitions: `testkit/fnn_testkit/workloads.py`.

## Experiments

Each experiment is a JSON plan in `plans/` (the harness input: a list of
configurations, run with `fnnbench sweep --plan`); the ids are used throughout
the docs, the sweep directory names and the figures.

| id | plan | what it measures |
|---|---|---|
| E1 | `e1_cpu_blas.json`, `e1_cpu_generic.json`, `e1_cpu_threads.json` | SYCL on the CPU: the three oneMath BLAS backends (MKLCPU, Netlib/OpenBLAS, generic SYCL BLAS) and thread scaling |
| E2 | `e2_sycl_cpu_gpu.json` | SYCL on the CPU vs the GPU, training and inference |
| E3 | `e3_cuda_vs_sycl_gpu.json` | CUDA (cudann) vs SYCL (syclnn) on the same GPU, with the stream / graph / memory-mode ablations |
| E4 | `e4_omp_cpu.json`, `e4_omp_gpu.json` | OpenMP (ompnn) host and target-offload paths, one run per compiler |
| E5 | `e5_breakdown.json`, `e5_inference.json` | per-phase device-time breakdown (the only profiled plan) and the inference rows (W5) |
| E6 | `e6_sycl_ablations.json` | syclnn's optimisations switched off one at a time, down to the version-0.1 behaviour |
| E7 | `e7_tiled_gemm.json` | the hand-written 16x16 tiled GEMM against the vendor BLAS, same kernel in the three models |
| W4 | `w4_sweep_cpu.json`, `w4_sweep_gpu.json` | the width x depth x batch scaling sweep (compute-bound regime) |

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

Thread counts (`--threads`) run in a fresh interpreter with the thread variables
of the [CPU thread policy](#cpu-thread-policy), because the OpenMP and BLAS
runtimes read them once; the child's affinity is reset to the cgroup's effective
cpuset first.

Every JSONL row records: run id (hash of the effective configuration), the
backend's `build_info` (compiler, flags, offload targets, BLAS backends, git
revision), the device (name, vendor, driver, memory, compute units, compute
capability), the BLAS backend actually used, dtype, workload, layer sizes,
sample count, batch, epochs, repetitions, mode, thread count and the runtime's
`omp_get_max_threads`, the requested and effective `Options`, the results
above, and `sysinfo` (host, kernel, CPU model / ISA / caches / governor, NUMA
masks, GPU clocks / power / throttle reasons from `nvidia-smi` or `rocm-smi`,
container image, filtered environment, mapped libraries, package versions, git
SHA of the bench). `fnnbench replay <file> [--id]` re-runs any row;
`fnnbench collect` flattens everything into one CSV.

### Per-epoch times

Every training row carries two per-epoch times (inference rows only the first):

- `median_epoch_s`: wall time of one `train(X, Y, batch, epochs)` call divided
  by `epochs`, median over `repeat` calls. It includes the dataset staging and
  upload, the per-epoch loss read-back and every synchronisation: what a user
  of the library pays.
- `steady_epoch_s`: median of the library's own per-epoch wall clocks
  (`Profile.epoch_wall_ns`) over all timed calls, excluding the first epoch of
  each call. The first epoch carries the asynchronous dataset upload and lazy
  initialisation (BLAS handles, module loading, JIT); in CUDA-graph mode the
  graphs are captured once, by the warm-up call, and replayed by every later
  call of the same shape, so no capture is timed. This is the number used for
  the programming-model comparison (E3, E4, E6) and by the figures; the
  call-level number is reported next to it.

### Inference rows (W5)

`mode = infer` rows time `predict(X, batch)` over the whole set (`epochs`
passes, `repeat` calls, same warm-up/median protocol). Batch 1 on 2048
samples for cup and mnist, on its 124 training samples for monk (E2 and E5 alike), is the latency
regime (one launch chain per sample); batch 4096 is the throughput regime. The FLOP model is the
forward GEMM only, 2B·Σ n_l·n_{l+1}.

The bindings hand the (C-contiguous, same-dtype) NumPy buffer to raw-pointer
overloads of `train`/`predict`, and the library records `predict_wall_ns`, the
host wall clock of every `predict()` call. An inference row's `steady_epoch_s`
is the median of `predict_wall_ns` over the `repeat` x `epochs` timed calls
(the same boundary as the training rows' per-epoch clock: inside the library,
after the buffer hand-over); `median_epoch_s` is the Python-side call time,
which also includes the output array's construction. Rows measured with builds
that copied the input in the bindings (before syclnn 0.2.1, cudann 0.1.1,
ompnn 0.1.1) are in `superseded.jsonl`.

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
(3·n²·sizeof(T) bytes). The rows are in `results/<machine>/<date>/peaks*/`;
`scripts/peaks-table.py` prints them as one table (the denominators of every
%-of-peak figure).

## Library options that change what a row measures

- `pinned_host`: since syclnn 0.2.2 and cudann 0.1.2 the host data is copied
  directly from pageable memory (`false`, as in ompnn); `true` stages every
  `train()`/`predict()` call through pinned buffers allocated for that call and is
  measured as an ablation (E3, E6, pinned inference rows). Rows of the two
  libraries measured with the earlier default are in `superseded.jsonl`.
- `memory=shared` on the RX 7900 XTX (ROCm 7.2, no XNACK) is host-coherent
  memory, not page migration: the AMD "shared" rows measure zero-copy PCIe
  traffic, the NVIDIA ones managed-memory migration (no explicit prefetch in
  either library; first-touch faults land in the warm-up epoch).
- `memory=host` with `loss_reduction` relies on 64-bit atomics to fine-grained
  host memory over PCIe (works on ws-amd, may silently fail elsewhere; the
  parity suite catches it).
- `queue` / `streams`: the default is out-of-order with 4 streams (event
  fork/join); `queue=in_order` rows are reported next to it in E3. On the DPC++
  CUDA backend syclnn forces the in-order queue when a oneMath BLAS is used
  (`docs/toolchains.md`, known issues); the effective value is in each row's
  `effective_options`.
- `sync_every`: a queue/stream drain every N batches (all three libraries
  accept it; syclnn and cudann act on it). syclnn sets it to 4 on CPU devices,
  whose OpenCL runtime's per-submission cost grows with the outstanding commands;
  GPUs run unbounded.
- `sync_ops=true`: wait after every launch, i.e. run cudann or syclnn in ompnn's
  execution model on the same hardware and library (E3, E6). ompnn executes every
  operation synchronously: target regions block the host and every vendor BLAS
  call is followed by a device synchronisation (the OpenMP runtime's streams and
  the BLAS stream are not ordered otherwise).
- `blas_queue=dedicated` rows are timing rows only: with the profiler on, the
  event recorded for a BLAS call is the bracketing marker kernel's, not the
  library's (the breakdown plan never sets it).
- CUDA-graph rows have no per-phase profile (the kernels are inside graph
  launches); their `other_ms` is the graph launch time.

## OpenMP and hand-written kernel rows

- gcc's offload model maps one OpenMP thread to a warp/wavefront and uses the
  lanes only inside `simd` loops; the two ablation kernels that cannot carry
  `simd` (the `omp atomic` loss of `loss_reduction=false` and the serial
  `bias_gemv=false` bias gradient) run at one lane per warp on gcc builds and
  are not compared with the SYCL/CUDA equivalents.
- ompnn's `sumsq` accumulates in double and its host tiled GEMM reassociates the
  16-term partial sums: the OpenMP rows are not bit-identical to the SYCL/CUDA
  ones, only within the check tolerances.
- E7 (`blas=tiled`): the three kernels share the tile size (16x16), the work
  decomposition (one work-item/thread per C element, op(A) tile stored
  transposed in local/shared memory, op(B) tile broadcast) and the sequential k
  accumulation in a register on the SYCL, CUDA and clang-OpenMP paths. gcc's
  OpenMP offload gives a team 16 wavefronts, so its variant strides over the
  tile elements: those rows are what gcc's offload model delivers, not the same
  kernel. The reductions for asum/nrm2 differ in decomposition (SYCL
  `sycl::reduction`, one CUDA block, OpenMP `reduction` in double), affecting
  only the reported penalty term. On the GTX 1080 Ti the syclnn tiled rows run
  on the out-of-order queue while its oneMath rows are in-order (see `queue`
  above), so the tiled/library ratio there mixes the queue mode with the kernel.
- One-time costs (BLAS handle creation, lazy module loading, gcc's PTX JIT)
  land in the first batch; every row discards `--warmup` epochs before timing.

## CPU thread policy

- One thread per physical core: 16 on ws-amd, 12 on ws-nvidia, 32 on hpc-a30
  (`OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`; the hardware
  threads are not used). The thread-scaling series (E1, E4) add the other counts.
- The DPC++ OpenCL CPU device takes its thread count from `DPCPP_CPU_NUM_CUS`
  (it ignores `OMP_NUM_THREADS` and keeps reporting the hardware thread count as
  `compute_units`); the harness sets it with the OpenMP variables.
- Binding: gcc and MKL rows (GNU threading layer) run bound
  (`OMP_PROC_BIND=close`, `OMP_PLACES=cores`); the LLVM-built OpenMP wheels run
  unbound (LLVM's `libomp` with the OpenMP-threaded OpenBLAS collapses under
  binding); the DPC++ CPU rows keep the binding variables (the OpenCL runtime's
  TBB workers ignore them) and the AdaptiveCpp host rows set neither. The
  setting is in every row's `sysinfo.env`.
- OpenBLAS variant: ompnn links the OpenMP-threaded OpenBLAS through an rpath to
  its directory (Ubuntu's alternatives symlink would otherwise resolve the
  pthread build); oneMath's Netlib backend on the OpenCL CPU device uses the
  pthread build (no libgomp team next to the TBB workers). The mapped libraries
  are in `sysinfo.libs`.

## Measurement conditions per machine

- **ws-amd**: every row runs with the host clock fixed at 2.8 GHz and boost off
  (`cpupower frequency-set -u 2.8GHz`; the 2950X reaches its thermal limit under
  any 16-thread load at stock settings), with `scripts/thermal-guard.sh` stopping
  the sweeps at Tctl 92 C. CPU rows carry `results.cpu_monitor` (temperatures and
  the mean core clock sampled every 200 ms) and `sysinfo.cpu.scaling_max_khz` /
  `boost`; `scripts/ws-amd-cpu-fixed-clock.sh` refuses to start without the cap.
- **ws-nvidia**: the Xeon E5-2643 v2 runs at a constant 3.5 GHz (no frequency
  scaling, no temperature sensor). The GPU is not dedicated in any pass, so its rows are comparable with each other
  but carry that background load.
- **hpc-a30**: every block runs in an exclusive SLURM job on the node
  (`gpu-excl`, 30-minute limit; `scripts/run-hpc.sh` chains jobs and
  `fnnbench sweep` resumes where the previous job stopped), so no other work
  shares the cores or the GPU. Clocks are as set by the firmware (no cpufreq
  interface) and recorded by the monitors. nvc++ rows use the node's NVIDIA HPC
  SDK 26.5 bind-mounted into the fnn-cuda sandbox.
- GPU clocks are recorded but not locked anywhere (no root on the department
  node). Every GPU row carries `gpu_monitor`: mean/max utilisation, power, mean
  clock and the energy of the timed region, sampled every 200 ms from the amdgpu
  sysfs counters or `nvidia-smi --query-gpu ... -lms 200`.

## Restricted row sets

- W4 on the CPU (`plans/w4_sweep_cpu.json`) is float only, widths 256 and 1024,
  depths 2 and 4 on 16 384 samples: the 4096-wide and 8-deep double
  configurations take hours per row on the CPU.
- hpc-a30, DPC++ CPU device: the batch-1 mnist rows of E2, E5 inference and E5
  breakdown are float only (the double row does not finish inside one 30-minute
  job); the batch-1 monk, cup and mnist-512-256 rows and every batch-4096 row
  exist in both precisions.
- gcc-14 nvptx tiled GEMM (E7): restricted row set, no 4096-wide row; on the
  A30 its kernel aborts the process on some rows (`libgomp: cuLaunchKernel
  error: invalid resource handle`), so only the rows measured before an abort
  exist and its tiled GEMM peak was not attempted.

## Profiling

Timing rows run unprofiled: the per-launch event pairs of the in-library
profiler are not free for the asynchronous libraries, so the libraries always
record only the per-epoch wall clock and the epoch/batch counters, and the
per-phase device times come from the dedicated profiled plan
`plans/e5_breakdown.json` (the breakdown figures), never used for a timing
comparison. The vendor-profiler cross-checks are produced by
`scripts/rocprof-crosscheck.sh` (`results/ws-amd/rocprof/`),
`scripts/perf-crosscheck.sh` (`results/ws-amd/perf*/README.md`) and
`scripts/nsys-crosscheck.py` (`results/ws-nvidia/2026-09-04/nsys-crosscheck.md`,
`results/hpc-a30/2026-09-17/nsys-crosscheck.md`).

## Run-to-run spread and provenance

The same visible configuration is measured in more than one sweep (E3 default vs
E7 `blas=auto` vs the E6 baseline, E4 vs E7 vendor rows). `scripts/headline.py`
tabulates the worst such pairs from the raw rows and summarises all of them at the
end of `results/analysis/headline-<machine>-<date>.md`. Differences below the
spread of the row family in question are not results; the headline tables take
the median over the sweeps.

Rows replaced by a re-measurement are in `superseded.jsonl` next to the live
file and are skipped by every reader. `build_info.git_sha` is the
`git describe` of the library tree at build time; for rows built before the
repositories were published it counts from the same tags but names
pre-publication commits. A `-dirty` suffix marks a wheel built from a tree with
uncommitted changes; a row's `effective_options` and `build_info` are the record
of what ran.

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
