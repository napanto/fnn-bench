# Benchmark methodology

How the rows in `results/` are produced, what each number means and under which
conditions it was measured. The analysis of the results is in the report.

## Workloads

| id | network | data | batch | why |
|---|---|---|---|---|
| W1 `monk` | 17-4-1, tanh / sigmoid, Adam (lr 0.01), L2 1e-3 | MONK-1 train (124 samples) | 40 | the ML project's real configuration: launch-overhead regime |
| W2 `cup` | 12-64-64-3, tanh / linear, Adam (lr 1e-3), L2 1e-4 | ML-CUP-like synthetic regression (1000 samples) | 40 | realistic small regression (the CUP data itself is course-private) |
| W3 `mnist` / `mnist-512-256` | 784-1024-1024-10 / 784-512-256-10, ReLU / sigmoid, SGD + momentum 0.9 (lr 0.1) | MNIST (60 000 samples) | 64 / 256 / 1024 | recognisable; on the A30 the cost per batch stays the same from batch 64 to 256 (launch-bound) and grows at 1024, where the GEMMs start to dominate |
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

New captures use `FNN_MACHINE_ID` as the public host identifier (default `host`).
Workstation extra-experiment scripts set it to `ws-amd` or `ws-nvidia`.
CSV exports use the machine directory name when available. September raw rows
retain their original provenance fields.

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

The main inference metric in figures and headline tables is `median_epoch_s`:
the Python-side time of the whole `predict()` call, averaged over `epochs`
passes within each repeat and then median over repeats. It includes destruction
of call-local temporaries and construction of the returned array. The bindings
pass the C-contiguous, same-dtype NumPy input to raw-pointer overloads.
The library also records `predict_wall_ns`; its median is stored in the legacy
`steady_epoch_s` field and reported as **library internal predict time**, a
secondary metric. That timer stops before local temporaries are destroyed and
is therefore not the whole call. Rows measured with builds
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
  directly from pageable memory (`false`, as in ompnn); `true` stages the host
  data through pinned buffers and is measured as an ablation (E3, E6, pinned
  inference rows). syclnn allocates those buffers in every `train()`/`predict()`
  call; cudann allocates them per `predict()` call and keeps the training ones as
  members, re-allocated only when a call needs more room. Every syclnn and
  cudann row of the three passes was re-measured with the new default (the old
  rows are in `superseded.jsonl`), the portability rows included (they now use
  the 0.2.2 wheels; the hashes are in `results/ws-amd/2026-09-05/portable-*.SHA256`).
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

- One worker per physical core: 16 on ws-amd, 12 on ws-nvidia, 32 on hpc-a30
  (`OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `DPCPP_CPU_NUM_CUS`; the
  hardware threads are not used). The thread-scaling series (E1, E4) add the other counts. The
  settings of each row are in `sysinfo.env`.
- The DPC++ OpenCL CPU device takes its worker count from `DPCPP_CPU_NUM_CUS` (it ignores
  `OMP_NUM_THREADS` and keeps reporting the hardware thread count as `compute_units`); the
  harness sets it with the OpenMP variables.
- Binding: gcc and MKL rows (GNU threading layer) run bound (`OMP_PROC_BIND=close`,
  `OMP_PLACES=cores`); the LLVM-built OpenMP wheels run unbound (LLVM's `libomp` with the
  OpenMP-threaded OpenBLAS collapses under binding). DPC++ and AdaptiveCpp CPU rows keep the
  binding variables. Observed placement during an E1 row on ws-amd: libgomp binds the master
  thread to core 0 ({0,16}); the OpenCL CPU runtime pins its 15 busy workers one per CPU (1-15),
  and one more busy thread is unpinned. The placement controls of that runtime are
  `DPCPP_CPU_CU_AFFINITY` / `DPCPP_CPU_PLACES` (Intel, "Supported environment variables").
- OpenBLAS variant (read from `sysinfo.libs`): the ompnn rows load the OpenMP-threaded OpenBLAS
  on every machine. On hpc-a30 this needs `/usr/lib/x86_64-linux-gnu/openblas-openmp` first on
  `LD_LIBRARY_PATH` (`scripts/hpc-blocks.sh`, `ompblas`): inside apptainer `LD_LIBRARY_PATH` is
  searched before the wheel's `DT_RUNPATH`, and the alternatives symlink there is the pthread
  build, which the ompnn rows of the 2026-09-17 pass had loaded (they were re-measured). nvc++
  multicore keeps the pthread build, since the OpenMP variant would bring libgomp next to nvc++'s
  own OpenMP runtime. oneMath's Netlib backend (DPC++ OpenCL CPU device) and the AdaptiveCpp host
  rows use the pthread build on every machine (no libgomp team next to the runtime's workers).
  `threads_effective` is OpenMP's maximum thread count, not the size of a pthread BLAS pool.

## Memory placement

The default Linux policy places a page on the NUMA node of the thread that first touches it.
The data of a row is first touched mostly by the master thread, which is bound to core 0, so
the rows measured until 2026-11-02 had most of their pages on node 0 while half of the workers
ran elsewhere: on ws-amd, during an E1 row, 1118 of the process's 1302 MiB were on node 0
(`/proc/<pid>/numa_maps`).

| machine | NUMA nodes | GPU on |
|---|---|---|
| ws-amd | 2 (TR 2950X: CPUs 0-7,16-23 / 8-15,24-31) | node 0 |
| ws-nvidia | 2 guest nodes, each pinned 1:1 to one host socket with strict memory binding | node 1 (host socket 1) |
| hpc-a30 | 8 (2 x EPYC 7301, 4 cores per node; automatic NUMA balancing on) | node 4 |

A pilot replayed rows of each pass under both policies in two ABAB rounds
(`results/<machine>/numa-pilot-2026-11-03`, ws-nvidia after the VM fix in
`numa-pilot-2026-11-03b`; tables in `results/analysis/numa-pilot-<machine>.md`, from
`scripts/numa-pilot-compare.py`). Interleaving the CPU rows' memory over all nodes made ompnn
gcc-14 19-23 % faster on hpc-a30 (round-to-round noise 6-12 %) and AdaptiveCpp 10 % faster; DPC++
did not change. On ws-amd most rows moved by less than 3 % (exceptions: DPC++ inference at batch
4096, 14 % slower; AdaptiveCpp mnist at batch 64, 5 % faster). Binding a GPU row's process and
memory to the GPU's node changed nothing beyond the noise on hpc-a30 and ws-amd; on ws-nvidia
the pilot's binding reached only the memory (see below), and moving it between the nodes changed
nothing beyond the 6-50 % noise of that shared GPU.

Policy since 2026-11-03:

- **CPU rows**: `numactl --interleave=all` on every machine, with the thread binding above.
  Every CPU row of the three passes was re-measured so (ws-amd 2026-11-03; hpc-a30 2026-11-03/04;
  ws-nvidia 2026-11-03).
- **GPU rows**: on ws-amd the process is already on the GPU's node (the master thread is bound
  to core 0); on hpc-a30 it stays unbound (binding CPU and memory to the A30's node changed
  nothing beyond the noise); on ws-nvidia the memory is bound to the GPU's node 1 and the host
  threads are unbound. `numactl --cpunodebind=1 --membind=1` was applied outside `podman run`
  there: the memory policy reaches the container, but crun (1.28) resets the CPU affinity of the
  container process to all CPUs, so the rows record `membind: 1` with `cpus_allowed_list 0-23`.
  A CPU binding has to be applied inside the container (`podman run ... numactl ... cmd`), as the
  portability rows and the extra experiments do.
- Each row records its policy in `sysinfo.numa.numactl_show`.

## Measurement conditions per machine

- **ws-amd**: every row runs with the host clock capped at 2.8 GHz and boost off
  (`cpupower frequency-set -u 2.8GHz`; the 2950X reaches its thermal limit under any 16-thread
  load at stock settings), with `scripts/thermal-guard.sh` stopping the sweeps at Tctl 92 C. CPU
  rows carry `results.cpu_monitor` (temperatures and the mean core clock sampled every 200 ms)
  and `sysinfo.cpu.scaling_max_khz` / `boost`. The GPU rows were measured on 2026-09-18 with host
  kernel 7.2.3, the CPU rows on 2026-11-03 with kernel 7.3.0 (`sysinfo.kernel`); GPU rows measured
  on 7.3.0 were 20-50 % slower and noisier, so the GPU rows of 7.2.3 are kept.
- **ws-nvidia**: a VM (24 vCPUs, 2 sockets x 6 cores x 2 threads, host-passthrough CPU) on a
  2 x Xeon E5-2643 v2 host. Until 2026-11-02 the guest's hyperthread pairs were not real host
  siblings (12 unbound threads ran on about 9 physical cores) and the host ran at a constant
  3.6 GHz (all-core turbo). The pass was re-measured on 2026-11-03 after re-pinning the vCPUs (each
  guest pair is one physical core), with the host clock fixed at 3.5 GHz (turbo off),
  `numa_balancing` off in host and guest, guest memory on 4 KiB host pages, C-states limited to
  POLL/C1. The GPU is not dedicated, and the round-to-round spread of short
  GPU rows reaches 30-50 %; ws-nvidia shows trends, hpc-a30 is the reference.
- **hpc-a30**: every block runs in an exclusive SLURM job on the node (`gpu-excl`, 30-minute
  limit; `fnnbench sweep` resumes where the previous job stopped), so no other work shares the
  cores or the GPU. The node is shared with the students of the courses: blocks are submitted one
  job at a time. Clocks are as set by the firmware (no cpufreq interface) and recorded by the
  monitors. nvc++ rows use the node's NVIDIA HPC SDK 26.5 bind-mounted into the fnn-cuda sandbox.
- GPU clocks are recorded but not locked anywhere (no root on the department node). Every GPU row
  carries `gpu_monitor`: mean/max utilisation, power, mean clock and the energy of the timed
  region, sampled every 200 ms from the amdgpu sysfs counters or `nvidia-smi --query-gpu ... -lms 200`.

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
  exist and its tiled GEMM peak was not attempted. On ws-nvidia the same abort
  stops a sweep, so the 2026-11-03 rows ran one process per row: 15 of the 16
  rows exist; the tiled 1024-wide row exceeded its one-hour limit.

## Profiling

Timing rows run with phase profiling disabled. The libraries still record
per-epoch wall clocks and epoch/batch counters. The dedicated profiled plan
`plans/e5_breakdown.json` records **times attributed to phases by the internal
profiler**: CUDA brackets launches with stream events; SYCL uses the returned
event's `command_start` / `command_end` timestamps; OpenMP measures host time
around synchronous operations, including submission and waiting. These are
different measurement boundaries and profiling can itself change execution.

The stacked bars are not a partition of wall time. Phase sums can contain
overlap across streams; host waiting can overlap device work. The bars average
all epochs, including the first, so their wall reference must cover those same
epochs, not just the steady-state subset. OpenMP loss includes the per-batch
`map(tofrom: loss)` readback; a zero D2H attribution does not mean no readback.
Steady-state timing rows are kept separate from profiled comparisons. New rows
also retain `steady_epochs_s` and `steady_iqr_s`; older rows cannot supply an
IQR for this metric from their call-level IQR alone.

The vendor-profiler cross-checks are produced by
`scripts/rocprof-crosscheck.sh` (`results/ws-amd/rocprof/`),
`scripts/perf-crosscheck.sh` (`results/ws-amd/perf*/README.md`) and
`scripts/nsys-crosscheck.py` (`results/ws-nvidia/2026-09-04/nsys-crosscheck.md`,
`results/hpc-a30/2026-09-17/nsys-crosscheck.md`).
Those historical Nsight traces include check, warm-up and measurement. The
new `scripts/nsys_epoch.py` capture uses `cudaProfilerStart/Stop` around one
two-epoch train call after warm-up: upload, first epoch and one steady epoch.
Vendor busy-time unions and idle gaps must use that same captured range;
phase attribution is available for the whole call, not individually for its
second epoch.

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
