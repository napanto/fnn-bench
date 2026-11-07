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
| Wait policy of clang's OpenMP runtime | `hpc-a30/clang-spin-2026-11-06`, `hpc-a30/clang-wait-2026-11-06`, `ws-nvidia/clang-spin-2026-11-06` | `hpc-blocks.sh clang-spin`, `hpc-blocks.sh clang-wait`, `ws-nvidia-clang-spin.sh` |

The last row motivated `Options::wait` in ompnn 0.1.2. Its default keeps the runtime's policy, so the reported rows keep their meaning.

`wallprof` needs no privileges: a POSIX timer on the monotonic clock sends a signal to the thread under study, and the handler records its call stack. Each interval is drawn at random around the period, so samples do not lock to the batches. `wallprof_report.py` symbolises the stacks inside the image that ran them, and `wallprof_layers.py` assigns each sample to one layer. The sample dumps stay local; the symbolised stacks and `layers.json` are the record.
