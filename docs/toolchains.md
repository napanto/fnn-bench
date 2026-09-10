# Toolchain matrix

Status of every compiler × device combination used by the study. "OK" means the
shared parity suite (`fnn-testkit`, 117 tests × float/double) passes; "build"
means it compiles but no device to run on; "-" not applicable.

Last update: 2026-09-06.

## Images and environments

| Environment | Base | Contents | Built |
|---|---|---|---|
| `localhost/fnn-cuda:dev` (→ `ghcr.io/napanto/fnn-cuda`) | ubuntu 24.04 | CUDA 12.9 (nvcc 12.9.86, cuBLAS, NVTX, CUPTI, Nsight Systems 2025.1.3), gcc-13 (nvcc host), gcc-14 + `gcc-14-offload-nvptx` + `gcc-14-offload-amdgcn`, clang-22 (apt.llvm.org, host OpenMP only), clang-18 (Ubuntu) with `libomptarget` nvptx/amdgpu device runtimes, OpenBLAS 0.3.26 (openmp + pthread), Intel oneMKL 2026.1 from PyPI (`mkl`, `mkl-devel`, `onemkl-sycl-blas`, `onemkl-sycl-include`, `tbb-devel`), Python 3.12 venv (numpy 2.5.2, pytest 9.1, pybind11 3.1.0, scikit-build-core 1.0.3) | 2026-09-03 on ws-amd, 7.6 GB |
| `localhost/fnn-sycl:dev` (→ `ghcr.io/napanto/fnn-sycl`) | fnn-cuda | intel/llvm **v7.1.0** `sycl_linux.tar.gz` (clang 22.1, libsycl 9; adapters: opencl, level_zero, **cuda**), Intel OpenCL CPU runtime **oclcpuexp 2026-WW28** (`sycl-ls` → `[opencl:cpu] AMD Ryzen Threadripper 2950X`), oneMath **v0.9** in `/opt/onemath` (MKLCPU + NETLIB/OpenBLAS + cuBLAS); `CC=clang CXX=clang++` | 2026-09-03 |
| `localhost/fnn-sycl-generic:dev` | fnn-sycl | + oneMath v0.9 with the generic SYCL BLAS backend in `/opt/onemath-generic` (INTEL_CPU tuning = `spir64_x86_64` AOT, ~2 h at -j8) | 2026-09-03 |
| `fnn-rocm` distrobox (ws-amd) | Ubuntu 24.04 + ROCm 7.2.4 (a distrobox container on the host) | hipcc, amdclang++ 22 (ROCm), rocBLAS 5.2 / hipBLAS 3.2, rocprofv3, gcc-13/14 + `gcc-14-offload-amdgcn`, clang-18 + libomp-18 (`libomptarget-amdgpu-gfx1100.bc`), OpenBLAS; `~/.local/opt/fnn-rocm/`: **AdaptiveCpp 25.10.0** (LLVM 18, SSCP `generic` target, ROCm + OpenMP backends), oneMath v0.9 (rocBLAS + NETLIB), Python venv | 2026-09-03, `scripts/rocm-toolchain.sh` |

Build-safety: every image build runs with `podman build --memory=20g`, every
compile with `-j8` (`CMAKE_BUILD_PARALLEL_LEVEL=8`).

## syclnn (SYCL)

| Compiler | Device | BLAS | Status | Notes |
|---|---|---|---|---|
| AdaptiveCpp 25.10 (generic/SSCP) | ws-amd CPU (OpenMP host device) | NETLIB (OpenBLAS-openmp) | **OK** double + float, all 15 ablation switches | `blas=netlib` explicit selection fails on AdaptiveCpp (JIT kernel library load error in oneMath's compile-time dispatch path); `blas=auto` resolves to NETLIB anyway |
| AdaptiveCpp 25.10 (generic/SSCP) | ws-amd RX 7900 XTX (gfx1100, HIP) | rocBLAS 5.2 | **OK** double + float, ablations OK | `memory=host` (zero-copy host USM): 20/118 tests fail with stale reads → not supported on this platform, kept as a result |
| intel/llvm v7.1.0 (DPC++, `-fsycl-targets=spir64,nvidia_gpu_sm_61,nvidia_gpu_sm_80`) | ws-amd CPU (`opencl:cpu`, oclcpuexp 2026-WW28) | MKLCPU 2026.1 (SYCL API, PyPI) | **OK** double + float (`blas=mklcpu` and `auto`) | MKLCPU works with the open-source DPC++ (libsycl 9 ABI = oneAPI 2026.1) |
| intel/llvm v7.1.0 (DPC++) | ws-amd CPU (`opencl:cpu`) | NETLIB (OpenBLAS 0.3.26 openmp) | **OK** double + float (`blas=netlib`) | |
| intel/llvm v7.1.0 (DPC++) | ws-amd CPU (`opencl:cpu`) | generic SYCL BLAS (oneMath `/opt/onemath-generic`, INTEL_CPU tuning) | **OK** double + float (`blas=generic` and `auto`) | separate wheel/venv: the backend is exclusive in oneMath |
| intel/llvm v7.1.0 (DPC++) | ws-nvidia GTX 1080 Ti (`cuda:gpu`, sm_61, CUDA 12.9, driver 580) | cuBLAS (oneMath run-time loader) | **runs** (2026-09-04): parity suites green in every mode, E2/E6/W4/E3/E7/E5 measured | wheel built for `spir64;nvidia_gpu_sm_61` only; in-order queue forced with oneMath (event race, see below) |
| AdaptiveCpp 25.10 (generic/SSCP, `containers/fnn-acpp-cuda.Containerfile`: clang-18 host, CUDA backend, oneMath cuBLAS + Netlib built with acpp) | ws-nvidia GTX 1080 Ti (`cuda:gpu`) | cuBLAS (oneMath, `AdaptiveCpp_enqueue_custom_operation`) | **runs** (2026-09-05): parity double 117 passed, float 111 passed, `queue=in_order` / `streams=1` / `blas_queue=dedicated` / `memory=shared` 221 passed each, `blas=tiled` 111 passed; `memory=host` 3 failed (zero-copy host USM, as on the RX 7900 XTX); E2/E3/E7/E5/E6 rows and peaks in `results/ws-nvidia/2026-09-05/sycl-gpu-acpp`, `peaks-acpp`, `sycl-gpu-acpp/parity.txt` | the out-of-order queue with cuBLAS is correct here (the DPC++ defect below is DPC++'s); mnist-512-256 b256 float: 209-224 ms against 146 ms with DPC++ on the same GPU, tiled 192 against 244 ms (`docs/optional.md`) |
| AdaptiveCpp wheel built on ws-amd (generic IR, one binary) | ws-nvidia GTX 1080 Ti and Xeon E5-2643 v2 host device | cuBLAS / Netlib | **runs unchanged** (2026-09-05): 111 passed, 7 skipped on the GPU; same speed as the wheel built on ws-nvidia (210 ms) | `scripts/ws-nvidia-portable-acpp.sh`, `results/ws-nvidia/2026-09-05/portable-acpp`, SHA-256 in `docs/optional.md` |
| intel/llvm v7.1.0 (DPC++) | dept. A30 (sm_80) | cuBLAS | the department runs | |

Known AdaptiveCpp note: its persistent JIT cache (`~/.local/share/acpp/apps/<app>/jit-cache`)
is shared by every process of the same executable; running two test sessions
concurrently corrupted it ("could not load shared kernel library"). Run
sequentially, or give each run its own `ACPP_APPDB_DIR`.

## cudann (CUDA)

| Compiler | Device | Status | Notes |
|---|---|---|---|
| nvcc 12.9 | ws-nvidia 1080 Ti (sm_61) | **runs** (2026-09-04): parity green in every mode (graphs, streams, managed/host memory, tiled), E3/W4/E7/E5 measured, nsys trace | |
| nvcc 12.9 / 13.x | dept. A30 (sm_80) | the department runs | |
| hipcc 7.2.4 (`CUDANN_HIP=ON`: hipify-perl + hipBLAS 3.2) | ws-amd 7900 XTX (gfx1100) | **OK** double + float; `memory=device/shared/host`, `queue=in_order/out_of_order/graph`, `streams=1/8`, every kernel ablation | graph mode captures from one stream on HIP (multi-stream fork/join capture segfaults in ROCm 7.2); `memory=host` (mapped zero-copy) works here, unlike SYCL host USM through AdaptiveCpp |

## ompnn (OpenMP)

One wheel per compiler; `build_info()` records compiler, flags, target and CBLAS.
Host path = `parallel for simd` + CBLAS; target path = `target teams distribute
parallel for simd` on `omp_target_alloc` memory + vendor BLAS interop
(hipBLAS/cuBLAS on the device pointers, synchronised after each call);
`blas=omp` = hand-written OpenMP GEMM on both paths.

| Compiler | Target | BLAS | Status | Notes |
|---|---|---|---|---|
| gcc 14.2 (`-foffload=disable`) | ws-amd CPU | OpenBLAS 0.3.26, OpenMP build (the pthread build was silently loaded until 2026-09-05, see below) | **OK** double + float | one intermittent double failure seen once in the fnn-cuda image (explained by the oneMath cuBLAS event race below) |
| gcc 14.2 | ws-amd CPU | oneMKL 2026.1 (`mkl_rt`) | **OK** double + float | `OMPNN_BLAS=mkl OMPNN_BLAS_ROOT=/opt/venv` |
| clang 22.1 (apt.llvm.org) | ws-amd CPU | OpenBLAS | **OK** double + float | host-only build (no offload runtime shipped) |
| clang 18.1 (Ubuntu) | ws-amd CPU | OpenBLAS | **OK** double + float | |
| amdclang++ 22 (ROCm 7.2.4) | ws-amd CPU (host path of the AMD build) | OpenBLAS | **OK** double + float, all ablations, `blas=omp` | |
| amdclang++ 22 (`-fopenmp-targets=amdgcn-amd-amdhsa --offload-arch=gfx1100`) | ws-amd RX 7900 XTX | hipBLAS 3.2 interop; `blas=omp` | **OK** double + float, ablations, `blas=omp` | mnist-512-256 b256 float (third pass, host at 2.8 GHz, steady epoch): ompnn/amdclang++ 210 ms, cudann/HIP 73 ms, syclnn/AdaptiveCpp 240 ms |
| gcc 14.2 (`-foffload=amdgcn-amdhsa -march=gfx1100`, needs `-fcf-protection=none`) | ws-amd RX 7900 XTX | hipBLAS interop | **OK** double + float | 7.2x slower than amdclang++ at batch 256 (third pass, host at 2.8 GHz): mnist-512-256 1.52 s/epoch (steady) against 0.21 s |
| clang 18.1 (`--offload-arch=gfx1100`) | ws-amd RX 7900 XTX | - | **fails to build**: `/opt/rocm/amdgcn/bitcode/ocml.bc: Unknown attribute kind (Producer LLVM 22, Reader LLVM 18)` | ROCm 7.2's device libs are LLVM-22 bitcode; clang-18 would need older device libs |
| clang 18.1 (`-fopenmp-targets=nvptx64-nvidia-cuda --offload-arch=sm_61 --cuda-path=/usr/local/cuda-12.9`) | ws-nvidia 1080 Ti / A30 | cuBLAS interop | **runs** (2026-09-04): E4/E7/E5 measured (mnist-512-256 float: 779 GFLOP/s GEMM rate at batch 1024; the 1024-wide `mnist` reaches 1.84 TFLOP/s (third pass)) | needs `clang-tools-18` (`clang-offload-packager`) and the host offload runtime `libomptarget.so.18.1` + `libomptarget.rtl.cuda.so`, which Ubuntu's `libomp5-18` does not ship: extracted from apt.llvm.org's `llvm-toolchain-noble-18` `libomp5-18` into `/usr/lib/llvm-18/lib` (both machines' `fnn-cuda:dev` were patched in place on 2026-09-04 and the Containerfile now does the same); ompnn rpaths the compiler's own lib dir |
| gcc 14.2 (`-foffload=nvptx-none -foffload-options=nvptx-none=-misa=sm_53 -fcf-protection=none -fno-stack-protector`) | ws-nvidia 1080 Ti / A30 | cuBLAS interop | **runs** (2026-09-04): parity green, E4/E7/E5 measured; the tiled kernel is 50-526x slower than cuBLAS on the GEMM-bound rows (third pass: mnist-512-256 float 142x, double 96x, w256 50x, w1024 526x) and 1.8-2.3x on monk/cup (one thread per warp), and its PTX fails to load for one of the two precisions depending on the build (`libgomp: cuLaunchKernel error: invalid resource handle`, double on 2026-09-04, float on 2026-09-05 after the team-memory C tile): the gcc-offload tiled rows on NVIDIA are reported as unreliable | gcc's nvptx back end knows sm_30/35/53/70/75/80 only; sm_53 PTX runs on Pascal through the driver JIT; Ubuntu's `-fstack-protector-strong` default breaks ptxas (`__stack_chk_guard`) |
| nvc++ 26.5 (`-mp=gpu -gpu=cc61|cc80`) | ws-nvidia 1080 Ti / A30 | cuBLAS interop | the department runs / ws-nvidia | V100+ officially |

Findings: gcc compiles every `omp target` region for all installed accelerator
back ends unless told `-foffload=disable`; Ubuntu's default `-fcf-protection=full`
and `-fstack-protector-strong` break gcc's nvptx/amdgcn offload step; LLVM's `omp_target_*` API lives in
`libomptarget`, which clang only links when the `-fopenmp` driver flag is on the
link line (host-only builds compile the calls out instead); the OpenMP runtime
of amdclang++ lives under `/opt/rocm/lib/llvm/lib` (rpath from
`-print-file-name=libomp.so`). Ubuntu ships OpenBLAS twice
(`openblas-pthread/`, `openblas-openmp/`) behind one alternatives symlink for
`libopenblas.so.0`; linking against the OpenMP build's directory is not
enough, the loader follows the symlink to the pthread build unless the module
carries an rpath to the variant's directory (ompnn does since 2026-09-05;
rows written since then record the mapped libraries in `sysinfo.libs`). The OpenMP build is 1.5x
faster under ompnn (one thread pool), the pthread build 5x faster under
oneMath's Netlib backend on the OpenCL CPU device (no libgomp team next to
the TBB workers): `docs/methodology.md`, CPU fix-ups.

## Hand-written BLAS (E7, `blas=tiled`)

All three libraries carry the same 16x16 tiled GEMM (SYCL `nd_range` + local
memory, CUDA shared memory, OpenMP `target teams` with team-shared tiles), a
one-work-item-per-row GEMV and reductions for asum/nrm2, selectable with
`Options.blas = "tiled"`; ompnn also keeps the naive `blas="omp"` loop.
Validated on 2026-09-03 with the parity suites: syclnn (AdaptiveCpp on the
RX 7900 XTX and on the CPU, DPC++ on `opencl:cpu`), cudann (hipcc/gfx1100,
also with `queue=graph`, `streams=4`, `memory=shared`), ompnn (amdclang++ and
gcc-14 amdgcn on the GPU; gcc-14, clang-18, clang-22, gcc-14+MKL, amdclang++
on the host). gcc's offload gives a team 16 wavefronts, not 256 threads, so
the OpenMP kernel has a strided variant for `omp_get_num_threads() < 256`.
Code review then found the op(A) tile indexed with the fast thread index as
the slow dimension (stride-16 local-memory bank conflicts in all three); the
tile is now stored transposed (`AsT[kk][li]`). Host suites re-validated
(DPC++ opencl:cpu, gcc-14, clang-22), the GPU suites re-run by
`results/ws-amd/2026-09-03/tiled-parity.txt` (second pass).

## Facts worth remembering

- **ws-amd's 2950X is at its thermal limit under any 16-thread load** (Tctl
  95-99 C within 20 s, `k10temp`) and hard-froze once (2026-09-05 18:22,
  nothing in the journal). every ws-amd sweep, CPU and GPU rows alike, runs with
  `cpupower frequency-set -u 2.8GHz` and boost off (volatile: re-apply after
  a reboot; `scripts/ws-amd-cpu-fixed-clock.sh` refuses to start otherwise)
  and `scripts/thermal-guard.sh` running; every row carries
  `results.cpu_monitor`.

- **libgomp pins the importing thread at load time** under `OMP_PROC_BIND`:
  a process that imported an OpenMP-linked module and then spawns a child
  hands it a one-core affinity mask (`cpus_allowed 0,16`), and the child's
  OpenMP runtime builds its places from that mask, so every thread count runs
  on one core. `fnnbench`'s per-thread-count children reset their affinity to
  the cgroup's effective cpuset (2026-09-05; the ompnn E4 thread rows before
  that were measured on one core and are superseded).
- **The DPC++ OpenCL CPU device sizes itself with `DPCPP_CPU_NUM_CUS`**
  (`OMP_NUM_THREADS` is ignored). The device keeps reporting 32
  `compute_units` whatever the variable says, only the times follow it; a
  `--cpuset-cpus` limit is honoured and is reflected in `compute_units`.
  Unset, it uses all 32 hardware threads of the 2950X while the OpenMP rows
  use the 16 cores; 16 units are faster than 32 (9 % at the fixed clock, 20 % in the pre-cap probe; E1).

- **Intel OpenCL CPU runtime and deep queues**: the per-submission cost of
  `opencl:cpu` (oclcpuexp 2026-WW28 under DPC++) grows with the number of
  outstanding commands. A mnist-512-256 epoch of 938 batches (about 12 commands
  each) took 157 s at batch 64 while a 64-batch epoch extrapolated to 20 s,
  and 16384 samples took 26-31 s versus 1.4 s for 4096 (out-of-order and
  in-order queues alike, profiling on or off). syclnn therefore waits for the
  queue every `Options.sync_every` batches, automatically 4 on CPU devices
  (never on GPUs; 2-4 measured best, 8 costs 10 %, 32 three times): the same
  epoch takes 2.3 s. AdaptiveCpp's OpenMP CPU
  backend does not show the effect (3.7 s unbounded; the 1024-wide `mnist`
  takes 6.3-6.4 s with DPC++ and 6.6 s with AdaptiveCpp at the fixed 2.8 GHz; 9.9 s in the pre-cap 32-thread run). The 16384-sample and
  depth-sweep numbers come from ad-hoc runs, not from result rows. Found 2026-09-04; the
  DPC++ CPU rows measured before it were superseded and re-measured.
- **oneMath on the DPC++ CUDA backend does not compose with out-of-order
  queues**: with `-fsycl-targets=nvidia_gpu_sm_61` (intel/llvm 7.1.0 nightly
  2026-09-01, oneMath v0.9 built with the native-command enqueue) the parity
  suite fails and MNIST trains to 16-59 % accuracy whenever the queue is
  out-of-order and a oneMath BLAS is used; in-order passes. Ruled out on
  2026-09-05: the cuBLAS workspace (`CUBLAS_WORKSPACE_CONFIG=:0:0` and
  `:4096:8` still fail) and the shared handle (a dedicated in-order BLAS
  queue alone still fails). What fails is the runtime's handling of the
  native command itself: neither its cross-stream input dependencies nor its
  completion event are honoured. Bracketing every call with two empty kernels
  on the dedicated in-order queue (input dependencies carried by a kernel,
  the event of a kernel returned) is correct, so syclnn offers it as
  `blas_queue=dedicated`; it is 40-60 % slower than plain in-order on the
  GTX 1080 Ti (third pass: cup 23.0 vs 14.3 ms, MNIST 512-256 216 vs 138 ms), so the default
  on the CUDA backend is the in-order queue. AdaptiveCpp + rocBLAS on the RX
  7900 XTX is unaffected (out-of-order passes). Same conclusion for the E6
  rows on NVIDIA: `queue=in_order` equals the default, `blas_queue=dedicated`
  is the out-of-order variant.
- **clang-18 OpenMP 5.1 `interop` works on nvptx**: `#pragma omp interop
  init(targetsync: obj) depend(in: x) nowait`, `omp_get_interop_ptr(obj,
  omp_ipr_targetsync)` as a `cudaStream_t` for cuBLAS, `interop destroy(obj)
  depend(out: y) nowait` orders consumers correctly (feasibility test
  2026-09-05, 2000-step chain, results exact). Per-step latency on a purely
  sequential chain was the same as the synchronous form within the noise of
  the 1080 Ti's clock ramps (0.11-0.22 ms per k1+GEMM+k2 step), so the
  benefit would come only from branch-level overlap.
- **The intel/llvm release tarball has no AMD target**: `sycl_linux.tar.gz`
  (v7.1.0) ships libspirv for nvptx64 only (`lib/clang/22/lib/libclc/`) and
  the CUDA, Level Zero and OpenCL adapters, no `libur_adapter_hip.so`; a
  syclnn build with `amd_gpu_gfx1100` fails at link time (missing
  `remangled-l64-signed_char.libspirv-amdgcn-amd-amdhsa.bc`). DPC++ on the
  RX 7900 XTX would need intel/llvm built from source with `--hip`
  (`scripts/dpcpp-hip-toolchain.sh` is kept for that; not done). The
  SYCL implementation comparison is therefore made on the GTX 1080 Ti
  (DPC++ vs AdaptiveCpp with the CUDA backend, `fnn-acpp-cuda`) and on the
  CPU (DPC++ `opencl:cpu` vs AdaptiveCpp OpenMP host, E1/E2), and the
  one-binary portability run uses AdaptiveCpp's generic SSCP binary on the
  CPU, the RX 7900 XTX and the GTX 1080 Ti, next to the DPC++
  `spir64 + nvidia_gpu_sm_61` wheel on x86 and Pascal.
- **DPC++ fat binaries with two CUDA device images**: a wheel built with
  `-fsycl-targets=spir64,nvidia_gpu_sm_61,nvidia_gpu_sm_80` fails on the
  GTX 1080 Ti with "The program was built for 1 devices" and an empty build
  log (the CUDA adapter of intel/llvm 7.1.0 nightly 2026-09-01 picks the
  wrong image); `spir64,nvidia_gpu_sm_61` alone works. One CUDA target per
  wheel, so ws-nvidia and the A30 get separate builds (checked 2026-09-04).

- oneMath v0.9's NETLIB backend redefines `cblas_i?amin` with an `int` return
  type; OpenBLAS's `cblas.h` declares them with `CBLAS_INDEX` → compile error.
  `containers/patches/onemath-netlib-openblas.patch` guards the definitions with
  `#ifndef OPENBLAS_VERSION`.
- oneMath's MKLCPU backend uses the *SYCL* oneMKL API (`libmkl_sycl_blas`), so the
  PyPI packages `onemkl-sycl-blas` + `onemkl-sycl-include` + `mkl-devel`
  (+ `tbb-devel` for `MKLConfig.cmake`) replace the multi-GB apt toolkit.
- oneMath's `FindcuBLAS.cmake` uses the legacy `find_package(CUDA)` and needs
  `-DCUDA_CUDA_LIBRARY=$CUDA_HOME/lib64/stubs/libcuda.so` when no driver is present.
- oneMath's exported CMake targets link `ONEMATH::SYCL::SYCL`, which its
  `FindCompiler.cmake` only defines for DPC++; AdaptiveCpp consumers must define
  it themselves (syclnn's CMakeLists does).
- apt.llvm.org's `libomp-22-dev` conflicts with Ubuntu's `libomp-18-dev`
  (`libomp-x.y-dev`), although all files live under versioned directories;
  `fnn-cuda` unpacks the 18 packages with `dpkg -x` into `/usr/lib/llvm-18`.
- AdaptiveCpp needs `libclang-rt-18-dev` (`libclang_rt.builtins-x86_64.a`) for
  its HIP backend library.
- Ubuntu's `libomp-18-dev` ships `libomptarget.rtl.{cuda,amdgpu}.so` and device
  bitcode for `sm_*` and `gfx*` (incl. gfx1100): clang-18 offloads out of the box.
