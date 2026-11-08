# Optional experiments

The four optional experiments of the study: what was run, how, and where the rows are. The results
are discussed in the report. Rows of the third pass are in `results/<machine>/2026-09-05/`
(hpc-a30: `results/hpc-a30/2026-09-17/`).

## 1. Hand-written tiled GEMM in the three models (E7)

`Options.blas = "tiled"` in all three libraries: the same 16x16 local/shared-memory tiled GEMM
(op(A) tile stored transposed, one work-item per C element, sequential k accumulation in a
register), a row-per-work-item GEMV and reductions for asum/nrm2. Plan `plans/e7_tiled_gemm.json`;
kernel-equivalence notes and the gcc-offload exception in `docs/methodology.md` (OpenMP and
hand-written kernel rows) and `docs/toolchains.md` (Hand-written BLAS). The tiled/vendor ratios per
device and toolchain are in `results/analysis/headline-<machine>-<date>.md` and the `tiled-*`
figures.

## 2. DPC++ vs AdaptiveCpp on the same device

| device | DPC++ | AdaptiveCpp | rows |
|---|---|---|---|
| GTX 1080 Ti | intel/llvm 7.1.0, oneMath cuBLAS | v25.10.0 generic SSCP, oneMath cuBLAS + Netlib (`containers/fnn-acpp-cuda.Containerfile`, `scripts/ws-nvidia-acpp.sh`) | `sycl-gpu*`, `sycl-gpu-acpp`, `peaks-acpp` |
| A30 | as above, `sm_80` wheel | as above | `sycl-gpu*`, `sycl-gpu-acpp` |
| RX 7900 XTX | not possible: no HIP adapter and no amdgcn libspirv in the intel/llvm release (`docs/toolchains.md`) | ROCm backend, oneMath rocBLAS | `sycl-gpu*` |
| CPUs (2950X, 2 x EPYC 7301) | OpenCL CPU device (`DPCPP_CPU_NUM_CUS` = cores) | OpenMP host device (`OMP_NUM_THREADS` = cores) | `sycl-cpu*`, `sycl-cpu-acpp` |

`scripts/acpp-vs-dpcpp.py` prints the side-by-side table (steady epochs of the float training
rows, vendor and tiled BLAS) from the rows.

## 3. Portability: one binary, four devices

The *same* `_syclnn` shared object, identified by SHA-256, run on every device it can reach; the
same `ompnn` source built by five compilers (the E4 matrix); `cudann` on NVIDIA only, plus its
HIPified build on AMD and the unmodified NVIDIA wheel on AMD through ZLUDA (4.).

| wheel (SHA-256 of `_syclnn*.so`) | built on / targets | devices | rows |
|---|---|---|---|
| AdaptiveCpp generic SSCP, `c698b8a0...0374657` | ws-amd, one generic IR (JIT per device) | RX 7900 XTX, TR 2950X (host device), GTX 1080 Ti, Xeon E5-2643 v2 (host device) | `results/ws-amd/2026-09-05/portable-acpp`, `results/ws-nvidia/2026-09-05/portable-acpp` (`scripts/ws-nvidia-portable-acpp.sh`) |
| DPC++ `spir64` + `nvidia_gpu_sm_61`, `2a41db17...6945e99` | ws-nvidia; `spir64` is SPIR-V compiled by JIT at runtime, `nvidia_gpu_sm_61` is the CUDA AOT target | TR 2950X (OpenCL CPU), GTX 1080 Ti; not the RX 7900 XTX (no HIP adapter) | `results/ws-amd/2026-09-05/portable-dpcpp`, `results/ws-nvidia/2026-09-05/sycl-gpu` |

The full hashes are in `results/ws-amd/2026-09-05/portable-{acpp,dpcpp}.SHA256`.

## 4. ZLUDA: the unmodified CUDA wheel on the RX 7900 XTX

`scripts/zluda-crosscheck.sh`: ZLUDA v7-preview.10 (`zluda-linux-9c8b43f2`), the `cudann` wheel
built on ws-nvidia with nvcc 12.9 for sm_61 + sm_80, CUDA 12.9's `libcudart.so.12` from the fnn-cuda
image, ZLUDA's own `libcuda.so.1` and `libcublas.so.12` in front on `LD_LIBRARY_PATH`, inside the
`fnn-rocm` environment (ROCm 7.2.4). Output: `results/ws-amd/zluda/README.md` (what runs, what does
not, speed against the HIPified build), rows in `results/ws-amd/zluda/rows/`.

## 5. Where the time and the variance go (2026-11-06 and 07)

Separate directories under `results/`, each with its scripts in `scripts/`. None of these rows enters the reported pass.

| Question | Directory | Script |
|---|---|---|
| GPU busy and idle time of an epoch, per path | `hpc-a30/nsys-gap-*` | `hpc-blocks.sh nsys-gap`, `nsys-gap-table.py` |
| Cost of one operation in each runtime | `hpc-a30/launch-probe-2026-11-06` | `scripts/launch-probe/` |
| Host CPU time by library | `ws-nvidia/perf-host-2026-11-06` | `ws-nvidia-perf-night.sh` |
| NUMA node and pinning of the submitting thread | `hpc-a30/gpu-placement-2026-11-06` | `hpc-blocks.sh gpu-placement` |
| GPU clock against recent load | `hpc-a30/clock-state-2026-11-06` | `hpc-blocks.sh clock-state`, `clock_ramp.py` |
| BLAS on the EPYC host, with AMD BLIS | `hpc-a30/cpu-blas-amd-2026-11-06` | `hpc-blocks.sh cpu-blas-amd` |
| Variance of CUP on the ws-amd CPU, real-time priority, SMT siblings | `ws-amd/cup-threads-*`, `ws-amd/cup-interference-*` | `ws-amd-cup-threads.sh`, `ws-amd-cup-interference.sh`, `ws-amd-gpu-fifo.sh`, `cup_threads.py` |
| Wall time of the submitting thread by call stack, with no perf events | `hpc-a30/wallprof-2026-11-07`, `ws-nvidia/wallprof-2026-11-07` | `scripts/wallprof/`, `hpc-blocks.sh wallprof` |
| Epoch against a measured host delay; overlap of kernels across streams without a profiler; gcc epoch against `thread_limit` | `hpc-a30/overlap-probe-2026-11-07` | `hpc-blocks.sh overlap-probe`, `scripts/launch-probe/probe5_cuda.cu` |
| The 2 ms sampling across batch, CUP and width; the stream probe with its arms reversed | `hpc-a30/wallprof-sweep-2026-11-07` | `hpc-blocks.sh wallprof-sweep` |
| Discarded events, on a scratch copy of syclnn | `hpc-a30/discard-probe-2026-11-07`, `ws-nvidia/discard-2026-11-07` | `scripts/syclnn-discard-events.py`, `hpc-blocks.sh build-sycl-discard`, `hpc-blocks.sh discard-probe` |
| Checked rows of the options the wall profile suggests | `hpc-a30/tuned-2026-11-07` | `hpc-blocks.sh tuned-rows` |
| Whether those options change the trained network, ten epochs, loss in double from the predictions | `ws-nvidia/compare-final-2026-11-07` | `scripts/wallprof/compare_final.py` |
| Each path's fastest tested configuration against the suggested SYCL options, one job (MNIST at three batches, CUP, AdaptiveCpp) | `hpc-a30/best-2026-11-08` | `hpc-blocks.sh best-rows` |
| The suggested options on the GTX 1080 Ti | `ws-nvidia/tuned-2026-11-08` | `scripts/ws-nvidia-tuned.sh` |
| Hardware counters inside cuBLAS, one-stream CUDA against DPC++ | `ws-nvidia/cublas-counters-2026-11-08` | `scripts/ws-nvidia-cublas-counters.sh` |
| User-space counters of the sampling signal (inconclusive: noisy, whole-process counts) | `ws-nvidia/signal-counters-2026-11-08` | `scripts/ws-nvidia-signal-counters.sh` |
| The stream probe on the GTX 1080 Ti | `ws-nvidia/overlap-probe-2026-11-08` | `scripts/ws-nvidia-overlap-probe.sh` |
| Cost of one signal outside the handler; GPU traces with the options; AdaptiveCpp and tuned DPC++ stacks | `hpc-a30/explain-2026-11-08` | `hpc-blocks.sh explain-a`, `scripts/wallprof/sigcost.c` |
| Whole-process CPU sampling (inconclusive: the runtimes' worker threads block the signal) | `hpc-a30/explain-cpu-2026-11-08` | `hpc-blocks.sh explain-b` |
| oneMKL scaling of one batch's GEMMs; main-thread stacks of the CPU paths at 1 to 64 workers | `hpc-a30/explain-cpu2-2026-11-08` | `hpc-blocks.sh explain-c`, `scripts/wallprof/gemm_scaling.c` |
| GPU traces of the hand-written tile against cuBLAS | `hpc-a30/explain-tile-2026-11-08` | `hpc-blocks.sh explain-d` |
| Radeon modes against core-complex confinement, with clock levels | `ws-amd/cup-interference-2026-11-08/gpu` | `scripts/ws-amd-gpu-fifo.sh` (variants `ccx0`, `ccx1`, `ccx0core`, `DPMLOG=1`) |
| Radeon MNIST modes: wall stacks of fast and slow processes; one or two HIP hardware queues | `ws-amd/radeon-wallprof-2026-11-08`, `ws-amd/cup-interference-2026-11-08/gpu/mnist-hwq*` | `scripts/ws-amd-radeon-wallprof.sh`, `scripts/ws-amd-gpu-fifo.sh` (variants `hwq1`, `hwq2`) |
| What runs on CPU 0 while the main thread runs on its sibling; idle-state residency; main thread on the siblings of quiet cores | `ws-amd/cup-threads-sibcensus-2026-11-08`, `ws-amd/cup-threads-2026-11-08-sib*` | `scripts/cpu0-census.py`, `scripts/ws-amd-sibling-census.sh`, `scripts/ws-amd-cup-threads.sh` |
| DPC++ CPU under FIFO, round robin, no OpenMP spin, 12 workers | `ws-amd/dpc-fifo-wallprof-2026-11-08`, `-08b`, `-08c` | `scripts/ws-amd-dpc-fifo-wallprof.sh` |
| DPC++ CPU under FIFO: run-queue waits, migrations and switches per thread (bpftrace, run as root), and FIFO against default with one hardware thread per core | `ws-amd/dpc-fifo-sched-2026-11-08`, `-08b` | `scripts/ws-amd-dpc-fifo-sched.sh`, `scripts/sched/runq.bt`, `scripts/sched/runq-table.py` |
| Radeon MNIST with library streams, one stream, and kernel traces of the default: gaps before a kernel on the same or another HSA queue | `ws-amd/radeon-modes-2026-11-08` (raw traces kept local) | `scripts/ws-amd-radeon-modes.sh`, `scripts/radeon-queue-gaps.py` |
| A30 CUDA with one to four streams at batch 64, 256, 1024 and on CUP; event cost and dependent-kernel gap probe; GPU metrics of one call | `hpc-a30/stream-explain-2026-11-08` | `hpc-blocks.sh stream-explain`, `scripts/launch-probe/probe6_cuda.cu`, `scripts/gpu-metrics-summary.py` |
| A30 CUDA graph replay captured from one or four streams; cuBLAS entry points in the stacks of CUDA and of DPC++ with and without less runtime work | `hpc-a30/explain-e-2026-11-08` | `hpc-blocks.sh explain-e`, `scripts/wallprof/cublas_entries.py` |
| Radeon: cost of handing a dependent kernel to another HIP stream, untraced, alternated with HIP MNIST | `ws-amd/radeon-handover-2026-11-08` | `scripts/ws-amd-radeon-handover.sh`, `scripts/launch-probe/probe7_hip.cpp` |
| DPC++ CPU under FIFO: waits including preemptions, and run time beside a benchmark thread on the SMT sibling (bpftrace, run as root) | `ws-amd/dpc-fifo-sched-2026-11-08c` | `scripts/sched/runq2.bt`, `scripts/sched/runq2-table.py` |
| Radeon MNIST under power-management arms: DPM high against auto; COMPUTE profile and GFXOFF disabled; raised minimum graphics clock (root helper applies each arm) | `ws-amd/radeon-dpm-2026-11-08`, `radeon-pm-2026-11-08`, `radeon-od-2026-11-08` | `scripts/ws-amd-radeon-dpm.sh` |
| Radeon MNIST: hardware placement of each process's compute queues from KFD debugfs snapshots (root helper; raw snapshots kept local) | `ws-amd/radeon-queues-2026-11-08` | `scripts/ws-amd-radeon-queues.sh`, `scripts/radeon-queue-slots.py` |
| cuBLAS argument tuples of CUDA and DPC++ (LD_PRELOAD shim, GTX host) | `ws-nvidia/cublas-args-2026-11-08` | `scripts/ws-nvidia-cublas-args.sh`, `scripts/cublas-args/cublas_args.c` |
| DPC++ CPU with every thread pinned: main thread on a free core or beside a worker, default and FIFO; two workers on one core | `ws-amd/dpc-fifo-pin-2026-11-08b`, `-08d` (`-08` and `-08c` are discarded runs, see their README) | `scripts/ws-amd-dpc-fifo-pin.sh` |
| Radeon MNIST: ASLR off, HSA_ENABLE_INTERRUPT=0, HIP active-wait timeout, CPU idle state C2 on and off (root helper) | `ws-amd/radeon-modes2-2026-11-08`, `-08b`, `radeon-cstate-2026-11-08` | `scripts/ws-amd-radeon-modes2.sh`, `scripts/ws-amd-radeon-dpm.sh` |
| Radeon MNIST: wake-up count and latency of every runtime thread (bpftrace, run as root) | `ws-amd/radeon-wake-2026-11-08` | `scripts/ws-amd-radeon-wake.sh`, `scripts/sched/wake3.bt`, `scripts/sched/wake3-table.py` |
| GTX host: cycles, iTLB and L1i misses inside cuBLAS for CUDA, DPC++ and DPC++ with less runtime work | `ws-nvidia/cublas-counters-2026-11-08b` | `scripts/ws-nvidia-cublas-counters.sh` (LW=1), `scripts/cublas-counters-table.py` |
| DPC++ CPU: every thread pinned one to one, with and without busy SMT pairs; sixteen-CPU sets; FIFO on 32 CPUs with C2 on and off (root helper) | `ws-amd/dpc-fifo-pin1-2026-11-08`, `dpc-fifo-sched-2026-11-08e`, `dpc-fifo-c2-2026-11-08` | `scripts/ws-amd-dpc-fifo-pin1.sh`, `scripts/ws-amd-dpc-fifo-sched.sh`, `scripts/ws-amd-dpc-fifo-c2.sh` |
| Radeon MNIST confined to the core complex of the interrupt CPU or to node 1; DPC++ CPU FIFO on 24 CPUs and on the isolated node | `ws-amd/radeon-modes2-2026-11-08c`, `dpc-fifo-sched-2026-11-08f` | `scripts/ws-amd-radeon-modes2.sh`, `scripts/ws-amd-dpc-fifo-sched.sh` |
| Radeon MNIST with the amdgpu interrupt moved to CPU 12 (root helper, `irq.log`): near it and on CPUs 4-7, 20-23 | `ws-amd/radeon-modes2-2026-11-08d` | `scripts/ws-amd-radeon-modes2.sh` |
| Wait policy of clang's OpenMP runtime | `hpc-a30/clang-spin-2026-11-06`, `hpc-a30/clang-wait-2026-11-06`, `ws-nvidia/clang-spin-2026-11-06` | `hpc-blocks.sh clang-spin`, `hpc-blocks.sh clang-wait`, `ws-nvidia-clang-spin.sh` |

The last row motivated `Options::wait` in ompnn 0.1.2. Its default keeps the runtime's policy, so the reported rows keep their meaning.

`wallprof` needs no privileges: a POSIX timer on the monotonic clock sends a signal to the thread under study, and the handler records its call stack. Each interval is drawn at random around the period, so samples do not lock to the batches. `wallprof_report.py` symbolises the stacks inside the image that ran them, and `wallprof_layers.py` assigns each sample to one layer. The sample dumps stay local; the symbolised stacks and `layers.json` are the record.
