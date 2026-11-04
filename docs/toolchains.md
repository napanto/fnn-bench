# Toolchain matrix

Status of every compiler × device combination used by the study. "OK" means the
shared parity suite (`fnn-testkit`, 118 tests per precision: 117 run in double,
111 in float, the rest skipped by precision) passes; "build"
means it compiles but no device to run on; "-" not applicable.

Last update: 2026-11-03.

## Images and environments

| Environment | Base | Contents | Built |
|---|---|---|---|
| `localhost/fnn-cuda:dev` (→ `ghcr.io/napanto/fnn-cuda`) | ubuntu 24.04 | CUDA 12.9 (nvcc 12.9.86, cuBLAS, NVTX, CUPTI, Nsight Systems 2025.1.3), gcc-13 (nvcc host), gcc-14 + `gcc-14-offload-nvptx` + `gcc-14-offload-amdgcn`, clang-22 (apt.llvm.org, host OpenMP only), clang-18 (Ubuntu) with `libomptarget` nvptx/amdgpu device runtimes, OpenBLAS 0.3.26 (openmp + pthread), Intel oneMKL 2026.1 from PyPI (`mkl`, `mkl-devel`, `onemkl-sycl-blas`, `onemkl-sycl-include`, `tbb-devel`), Python 3.12 venv (numpy 2.5.2, pytest 9.1, pybind11 3.1.0, scikit-build-core 1.0.3) | 2026-09-03 on ws-amd, 7.6 GB |
| `localhost/fnn-sycl:dev` (→ `ghcr.io/napanto/fnn-sycl`) | fnn-cuda | intel/llvm **v7.1.0** `sycl_linux.tar.gz` (clang 22.1, libsycl 9; adapters: opencl, level_zero, **cuda**), Intel OpenCL CPU runtime **oclcpuexp 2026-WW28** (`sycl-ls` → `[opencl:cpu] AMD Ryzen Threadripper 2950X`), oneMath **v0.9** in `/opt/onemath` (MKLCPU + NETLIB/OpenBLAS + cuBLAS); `CC=clang CXX=clang++` | 2026-09-03 |
| `localhost/fnn-sycl-generic:dev` | fnn-sycl | + oneMath v0.9 with the generic SYCL BLAS backend in `/opt/onemath-generic` (INTEL_CPU tuning = `spir64_x86_64` AOT, ~2 h at -j8) | 2026-09-03 |
| `localhost/fnn-rocm:dev` (→ `ghcr.io/napanto/fnn-rocm`; `containers/fnn-rocm.Containerfile`) | `rocm/dev-ubuntu-24.04:7.2.4-complete` | the AMD stack as a pullable image: ROCm 7.2.4 (hipcc, amdclang++ 22, rocBLAS 5.2 / hipBLAS 3.2, rocprofv3), gcc-14 + `gcc-14-offload-amdgcn`, clang-18 host compiler, both OpenBLAS variants, AdaptiveCpp 25.10 (generic SSCP, ROCm + OpenMP backends), oneMath v0.9 rocBLAS + NETLIB compiled by acpp, venvs for the ompnn compiler matrix; 24.7 GB. Verified on ws-nvidia (no AMD GPU): syclnn builds and passes the parity suite on the host device (117 double / 111 float / 111 tiled), cudann's HIP build and ompnn's amdclang++ and gcc-14 amdgcn builds compile | 2026-09-10 |
| `fnn-rocm` distrobox (ws-amd) | Ubuntu 24.04 + ROCm 7.2.4 (a distrobox container on the host; the same recipe, `scripts/rocm-toolchain.sh`, as the image above: this is where the ws-amd rows were measured) | hipcc, amdclang++ 22 (ROCm), rocBLAS 5.2 / hipBLAS 3.2, rocprofv3, gcc-13/14 + `gcc-14-offload-amdgcn`, clang-18 + libomp-18 (`libomptarget-amdgpu-gfx1100.bc`), OpenBLAS; `~/.local/opt/fnn-rocm/`: **AdaptiveCpp 25.10.0** (LLVM 18, SSCP `generic` target, ROCm + OpenMP backends), oneMath v0.9 (rocBLAS + NETLIB), Python venv | 2026-09-03, `scripts/rocm-toolchain.sh` |
| hpc-a30 sandboxes (`~/sandboxes/<image>`, `scripts/hpc-blocks.sh`) | the `ghcr.io/napanto/fnn-{cuda,sycl,sycl-generic,acpp-cuda}:2026-09` images converted on the front-end with `singularity build --sandbox` (the node has no container runtime) and run on the node with an unprivileged apptainer 1.5.3 (`apptainer exec --nv`, `.deb` extracted into the home; user namespaces, no setuid) | node: Ubuntu 22.04, driver 610.43.02 (CUDA 13.3 user-mode driver runs the CUDA 12.9 images), NVIDIA HPC SDK 26.5 at `/opt/nvidia` bind-mounted for the nvc++ variants | 2026-09-17 |

Build-safety: every image build runs with `podman build --memory=20g`, every
compile with `-j8` (`CMAKE_BUILD_PARALLEL_LEVEL=8`).

### Published images (2026-09-11)

The images above are on the GitHub container registry, tags `latest` and `2026-09` (same content);
the `FNN_IMAGE` / `FNN_IMAGE_BUILT` variables inside them identify the build a row ran in.

| image | manifest digest | compressed size |
|---|---|---|
| `ghcr.io/napanto/fnn-cuda:2026-09` | `sha256:d111a865add84b0b8fd673357596b0644904a9e738d97f9698c7d4eaee8a4e78` | 3452 MB |
| `ghcr.io/napanto/fnn-sycl:2026-09` | `sha256:eaf7debc84fc906f75059e87ad964bb166ea04c7e0adc0046b61da72508f5a56` | 3631 MB |
| `ghcr.io/napanto/fnn-sycl-generic:2026-09` | `sha256:12ca315d547ec498dcb5b6b0caf4b53f8ebbc7364f3ef49bc9ef13cc5df633c0` | 3659 MB |
| `ghcr.io/napanto/fnn-acpp-cuda:2026-09` | `sha256:676b0c3b6d9b9797dc5f40a5017cc85079d29429133950e94c844c21ac612a91` | 3471 MB |
| `ghcr.io/napanto/fnn-rocm:2026-09` | `sha256:151c6abcb9ba0b13113cd9db4cfd6311e2dc3fbdb26373afa6ef5c405a4a458a` | 8476 MB |

Library images (the library installed into a toolchain image; `latest` tags built by each
repository's CI on every push, AMD tags by `scripts/publish-amd-variants.sh` on a workstation):

| image | base | manifest digest | compressed size |
|---|---|---|---|
| `ghcr.io/napanto/syclnn:latest` | fnn-sycl | `sha256:a592ad6c78b6e7a551685fd2aa0a465741739a3b3fc95eedbbcb3c9fa822d892` | 3646 MB |
| `ghcr.io/napanto/cudann:latest` | fnn-cuda | `sha256:4505c2f6c0d6e44973e769903dd146ed6ca9b8ceef846ab582a3d7d17ae437f4` | 3458 MB |
| `ghcr.io/napanto/ompnn:latest` | fnn-cuda | `sha256:6a4da2f8d86b87131cb53c0bcb9fd89ec0f30aefcc95101d46506f24dc2f3d4b` | 3548 MB |
| `ghcr.io/napanto/syclnn:acpp-rocm` | fnn-rocm | `sha256:53dc24d88a637e9c9e919f88e85cc43ca68cde4cf3ef67ebe467aee17faaeee7` | 8485 MB |
| `ghcr.io/napanto/cudann:hip` | fnn-rocm | `sha256:d12a051b11bd46a82eb52900517b99d6539d7296b277a515607e7a9344bf807a` | 8478 MB |
| `ghcr.io/napanto/ompnn:rocm` | fnn-rocm | `sha256:28f179a0b36db0d8d5b2239b7de95dee856a65859fe60077d9203627250f00a1` | 8480 MB |

The AMD tags passed their build-time checks only (syclnn's device enumeration on the host device,
cudann's and ompnn's build info): a run from the registry on the RX 7900 XTX, with the parity
suite, is still to be done and is listed as an open item in the README.

`fnn-cuda` carries two layers added after its 2026-09-04 build: the `FNN_IMAGE` stamp and the
`libomptarget` runtime-lookup fix (`containers/fnn-cuda.Containerfile`); the wheels measured in the
study are unaffected by either.

## syclnn (SYCL)

| Compiler | Device | BLAS | Status | Notes |
|---|---|---|---|---|
| AdaptiveCpp 25.10 (generic/SSCP) | ws-amd CPU (OpenMP host device) | NETLIB (OpenBLAS-openmp) | **OK** double + float, all 15 ablation switches | `blas=netlib` explicit selection fails on AdaptiveCpp (JIT kernel library load error in oneMath's compile-time dispatch path); `blas=auto` resolves to NETLIB anyway |
| AdaptiveCpp 25.10 (generic/SSCP) | ws-amd RX 7900 XTX (gfx1100, HIP) | rocBLAS 5.2 | **OK** double + float, ablations OK | `memory=host` (zero-copy host USM): 20/118 tests fail with stale reads → not supported on this platform, kept as a result |
| intel/llvm v7.1.0 (DPC++, `-fsycl-targets=spir64,nvidia_gpu_sm_61,nvidia_gpu_sm_80`) | ws-amd CPU (`opencl:cpu`, oclcpuexp 2026-WW28) | MKLCPU 2026.1 (SYCL API, PyPI) | **OK** double + float (`blas=mklcpu` and `auto`) | MKLCPU works with the open-source DPC++ (libsycl 9 ABI = oneAPI 2026.1) |
| intel/llvm v7.1.0 (DPC++) | ws-amd CPU (`opencl:cpu`) | NETLIB (OpenBLAS 0.3.26 openmp) | **OK** double + float (`blas=netlib`) | |
| intel/llvm v7.1.0 (DPC++) | ws-amd CPU (`opencl:cpu`) | generic SYCL BLAS (oneMath `/opt/onemath-generic`, INTEL_CPU tuning) | **OK** double + float (`blas=generic` and `auto`) | separate wheel/venv: the backend is exclusive in oneMath |
| intel/llvm v7.1.0 (DPC++) | ws-nvidia GTX 1080 Ti (`cuda:gpu`, sm_61, CUDA 12.9, driver 580) | cuBLAS (oneMath run-time loader) | **runs** (2026-09-04): parity suites green in every mode, E2/E6/W4/E3/E7/E5 measured | wheel built for `spir64;nvidia_gpu_sm_61` only; in-order queue forced with oneMath (event race, see below) |
| AdaptiveCpp 25.10 (generic/SSCP, `containers/fnn-acpp-cuda.Containerfile`: clang-18 host, CUDA backend, oneMath cuBLAS + Netlib built with acpp) | ws-nvidia GTX 1080 Ti (`cuda:gpu`) | cuBLAS (oneMath, `AdaptiveCpp_enqueue_custom_operation`) | **runs** (2026-09-05): parity double 117 passed, float 111 passed, `queue=in_order` / `streams=1` / `blas_queue=dedicated` / `memory=shared` 221 passed each, `blas=tiled` 111 passed; `memory=host` 3 failed (zero-copy host USM, as on the RX 7900 XTX); E2/E3/E7/E5/E6 rows and peaks in `results/ws-nvidia/2026-09-05/sycl-gpu-acpp`, `peaks-acpp`, `sycl-gpu-acpp/parity.txt` | the out-of-order queue with cuBLAS is correct here (the DPC++ defect below is DPC++'s) |
| AdaptiveCpp wheel built on ws-amd (generic IR, one binary) | ws-nvidia GTX 1080 Ti and Xeon E5-2643 v2 host device | cuBLAS / Netlib | **runs unchanged** (2026-09-05): 111 passed, 7 skipped on the GPU | `scripts/ws-nvidia-portable-acpp.sh`, `results/ws-nvidia/2026-09-05/portable-acpp`, SHA-256 in `docs/optional.md` |
| intel/llvm v7.1.0 (DPC++, `-fsycl-targets=spir64,nvidia_gpu_sm_80`) | hpc-a30 A30 (`cuda:gpu`, sm_80, CUDA 13.3 driver, 12.9 image) and its 2 x EPYC 7301 (`opencl:cpu`) | cuBLAS / MKLCPU, NETLIB, generic | **runs** (2026-09-17): parity green on the A30 in every mode (double, float, memory=shared, in_order, fine_deps=False, tiled, mnist) and on the CPU device (double, float, netlib, tiled, generic double + float); the hpc-a30 matrix measured with it | one wheel per CUDA architecture (see the 1080 Ti note); run through the fnn-sycl sandbox with an unprivileged apptainer |

Known AdaptiveCpp note: its persistent JIT cache (`~/.local/share/acpp/apps/<app>/jit-cache`)
is shared by every process of the same executable; running two test sessions
concurrently corrupted it ("could not load shared kernel library"). Run
sequentially, or give each run its own `ACPP_APPDB_DIR`.

## cudann (CUDA)

| Compiler | Device | Status | Notes |
|---|---|---|---|
| nvcc 12.9 | ws-nvidia 1080 Ti (sm_61) | **runs** (2026-09-04): parity green in every mode (graphs, streams, managed/host memory, tiled), E3/W4/E7/E5 measured, nsys trace | |
| nvcc 12.9 (`CUDANN_CUDA_ARCHS=80`) | hpc-a30 A30 (sm_80) | **runs** (2026-09-17): parity green in every mode (graphs, streams, managed/host memory, tiled, mnist); E3/W4/E7/E5 measured | the CUDA 12.9 image on the node's 13.3 driver; the node's own nvcc 13.3 compiles the smoke tests too |
| hipcc 7.2.4 (`CUDANN_HIP=ON`: hipify-perl + hipBLAS 3.2) | ws-amd 7900 XTX (gfx1100) | **OK** double + float; `memory=device/shared/host`, `queue=in_order/out_of_order/graph`, `streams=1/8`, every kernel ablation | graph mode captures from one stream on HIP (multi-stream fork/join capture segfaults in ROCm 7.2); `memory=host` (mapped zero-copy) works here, unlike SYCL host USM through AdaptiveCpp |

## ompnn (OpenMP)

One wheel per compiler; `build_info()` records compiler, flags, target and CBLAS.
Host path = `parallel for simd` + CBLAS; target path = `target teams distribute
parallel for simd` on `omp_target_alloc` memory + vendor BLAS interop
(hipBLAS/cuBLAS on the device pointers, synchronised after each call);
`blas=omp` = hand-written OpenMP GEMM on both paths.

| Compiler | Target | BLAS | Status | Notes |
|---|---|---|---|---|
| gcc 14.2 (`-foffload=disable`) | ws-amd CPU | OpenBLAS 0.3.26, OpenMP build (rpath, see below) | **OK** double + float | one intermittent double failure seen once in the fnn-cuda image (explained by the oneMath cuBLAS event race below) |
| gcc 14.2 | ws-amd CPU | oneMKL 2026.1 (`mkl_rt`) | **OK** double + float | `OMPNN_BLAS=mkl OMPNN_BLAS_ROOT=/opt/venv` |
| clang 22.1 (apt.llvm.org) | ws-amd CPU | OpenBLAS | **OK** double + float | host-only build (no offload runtime shipped) |
| clang 18.1 (Ubuntu) | ws-amd CPU | OpenBLAS | **OK** double + float | |
| amdclang++ 22 (ROCm 7.2.4) | ws-amd CPU (host path of the AMD build) | OpenBLAS | **OK** double + float, all ablations, `blas=omp` | |
| amdclang++ 22 (`-fopenmp-targets=amdgcn-amd-amdhsa --offload-arch=gfx1100`) | ws-amd RX 7900 XTX | hipBLAS 3.2 interop; `blas=omp` | **OK** double + float, ablations, `blas=omp` | |
| gcc 14.2 (`-foffload=amdgcn-amdhsa -march=gfx1100`, needs `-fcf-protection=none`) | ws-amd RX 7900 XTX | hipBLAS interop | **OK** double + float | |
| clang 18.1 (`--offload-arch=gfx1100`) | ws-amd RX 7900 XTX | - | **fails to build**: `/opt/rocm/amdgcn/bitcode/ocml.bc: Unknown attribute kind (Producer LLVM 22, Reader LLVM 18)` | ROCm 7.2's device libs are LLVM-22 bitcode; clang-18 would need older device libs |
| clang 18.1 (`-fopenmp-targets=nvptx64-nvidia-cuda --offload-arch=sm_61 --cuda-path=/usr/local/cuda-12.9`) | ws-nvidia 1080 Ti / hpc-a30 A30 (`--offload-arch=sm_80`) | cuBLAS interop | **runs** (2026-09-04; A30 2026-09-17: parity green incl. tiled and `blas=omp`): E4/E7/E5 measured | needs `clang-tools-18` (`clang-offload-packager`) and the host offload runtime `libomptarget.so.18.1` + `libomptarget.rtl.cuda.so`, which Ubuntu's `libomp5-18` does not ship: extracted from apt.llvm.org's `llvm-toolchain-noble-18` `libomp5-18` into `/usr/lib/llvm-18/lib` (both machines' `fnn-cuda:dev` were patched in place on 2026-09-04 and the Containerfile now does the same); ompnn rpaths the compiler's own lib dir |
| gcc 14.2 (`-foffload=nvptx-none -foffload-options=nvptx-none=-misa=sm_53 -fcf-protection=none -fno-stack-protector`) | ws-nvidia 1080 Ti / hpc-a30 A30 (`-misa=sm_80`) | cuBLAS interop | **runs** (2026-09-04; A30 2026-09-17: the library-BLAS suites green, the tiled and `blas=omp` suites abort without a summary, the unstable kernel below): parity green, E4/E7/E5 measured; the tiled kernel's PTX fails to load for one of the two precisions depending on the build (`libgomp: cuLaunchKernel error: invalid resource handle`, double on 2026-09-04, float on 2026-09-05 after the team-memory C tile): the gcc-offload tiled rows on NVIDIA are reported as unreliable | gcc's nvptx back end knows sm_30/35/53/70/75/80 only; sm_53 PTX runs on Pascal through the driver JIT; Ubuntu's `-fstack-protector-strong` default breaks ptxas (`__stack_chk_guard`) |
| nvc++ 26.5 (`-mp=gpu -gpu=cc80`, the node's HPC SDK bind-mounted into the fnn-cuda sandbox, `makelocalrc` for gcc-13) | hpc-a30 A30 | cuBLAS 13.4 interop (the SDK's `math_libs/13.2`, on `LD_LIBRARY_PATH` at run time) | **runs** (2026-09-17): parity green (double, float, tiled double + float, `blas=omp`); E4/E7/E5 measured | V100+ officially; nvc++ rejects `allocate(omp_pteam_mem_alloc : ...)` but places team-private arrays in CUDA shared memory by itself (ompnn 0.1.1); not run on the 1080 Ti (Pascal) |
| nvc++ 26.5 (`-mp=multicore`) | hpc-a30 CPU (2 x EPYC 7301) | OpenBLAS 0.3.26 (OpenMP build) | **runs** (2026-09-17): parity green (double, float, tiled); E4 cpu measured | the fifth host compiler of E4 |

Findings: gcc compiles every `omp target` region for all installed accelerator
back ends unless told `-foffload=disable`; Ubuntu's default `-fcf-protection=full`
and `-fstack-protector-strong` break gcc's nvptx/amdgcn offload step; LLVM's `omp_target_*` API lives in
`libomptarget`, which clang only links when the `-fopenmp` driver flag is on the
link line (host-only builds compile the calls out instead); the OpenMP runtime
of amdclang++ lives under `/opt/rocm/lib/llvm/lib` (rpath from
`-print-file-name=libomp.so`). Ubuntu ships OpenBLAS twice
(`openblas-pthread/`, `openblas-openmp/`) behind one alternatives symlink for
`libopenblas.so.0`; linking against the OpenMP build's directory is not
enough. The module carries the variant's directory in its runtime search path,
but `DT_RUNPATH` is searched after `LD_LIBRARY_PATH`. On hpc-a30 the latter
includes `/lib/x86_64-linux-gnu`, whose alternatives link selects pthread
OpenBLAS despite an OpenMP build-time link. The reported workstation ompnn
rows map OpenMP OpenBLAS; the original A30 CPU rows map pthread OpenBLAS.
The cluster script now explicitly selects the OpenMP directory at runtime for
GNU/LLVM CPU wheels, and prepares `DT_RPATH` for their next build with
`-Wl,--disable-new-dtags`. nvc++ multicore retains pthread OpenBLAS to avoid
introducing libgomp alongside its own runtime. Read `sysinfo.libs` for the
observed variant and version; thread limits do not measure active BLAS workers.
See `docs/methodology.md`, CPU thread policy, and the generated conditions tables.

## Hand-written BLAS (E7, `blas=tiled`)

All three libraries carry the same 16x16 tiled GEMM (SYCL `nd_range` + local
memory, CUDA shared memory, OpenMP `target teams` with team-shared tiles), a
one-work-item-per-row GEMV and reductions for asum/nrm2, selectable with
`Options.blas = "tiled"`; ompnn also keeps the naive `blas="omp"` loop.
Validated on 2026-09-03 with the parity suites: syclnn (AdaptiveCpp on the
RX 7900 XTX and on the CPU, DPC++ on `opencl:cpu`), cudann (hipcc/gfx1100,
also with `queue=graph`, `streams=4`, `memory=shared`), ompnn (amdclang++ and
gcc-14 amdgcn on the GPU; gcc-14, clang-18, clang-22, gcc-14+MKL, amdclang++
on the host; `results/ws-amd/2026-09-03/tiled-parity.txt`), then on NVIDIA
(GTX 1080 Ti, A30). gcc's offload gives a team 16 wavefronts, not 256 threads,
so the OpenMP kernel has a strided variant for `omp_get_num_threads() < 256`.
The op(A) tile is stored transposed (`AsT[kk][li]`), so that consecutive
work-items read consecutive local-memory banks.

## Known issues and build notes

- **oneMath on the DPC++ CUDA backend does not compose with out-of-order
  queues** (intel/llvm 7.1.0, oneMath v0.9 built with the native-command
  enqueue): with an out-of-order queue and a oneMath BLAS call the parity suite
  fails and MNIST trains to 16-59 % accuracy; in-order passes. The runtime honours
  neither the native command's cross-stream input dependencies nor its completion
  event (the cuBLAS workspace and the shared handle were ruled out). syclnn
  therefore uses the in-order queue on the CUDA backend unless `blas=tiled`;
  `blas_queue=dedicated` (every BLAS call bracketed by two empty kernels on a
  dedicated in-order queue) is the correct out-of-order variant, kept as an
  ablation. AdaptiveCpp with cuBLAS or rocBLAS is unaffected.
- **DPC++ fat binaries with two CUDA device images**: a wheel built with
  `-fsycl-targets=spir64,nvidia_gpu_sm_61,nvidia_gpu_sm_80` fails on the
  GTX 1080 Ti with "The program was built for 1 devices" and an empty build log
  (the CUDA adapter picks the wrong image); `spir64,nvidia_gpu_sm_61` alone
  works. One CUDA target per wheel, so ws-nvidia and the A30 get separate builds.
- **The intel/llvm release tarball has no AMD target**: `sycl_linux.tar.gz`
  (v7.1.0) ships libspirv for nvptx64 only and the CUDA, Level Zero and OpenCL
  adapters, no `libur_adapter_hip.so`; a syclnn build with `amd_gpu_gfx1100`
  fails at link time. DPC++ on the RX 7900 XTX would need intel/llvm built from
  source with `--hip` (`scripts/dpcpp-hip-toolchain.sh` records the attempt).
- **The Intel OpenCL CPU runtime and deep queues**: the per-submission cost of
  `opencl:cpu` (oclcpuexp 2026-WW28 under DPC++) grows with the number of
  outstanding commands, with in-order and out-of-order queues alike. syclnn waits
  for the queue every `Options.sync_every` batches, 4 by default on CPU devices
  (never on GPUs). AdaptiveCpp's OpenMP host device does not show the effect.
- **The DPC++ OpenCL CPU device sizes itself with `DPCPP_CPU_NUM_CUS`**
  (`OMP_NUM_THREADS` is ignored). The device keeps reporting the hardware thread
  count as `compute_units` whatever the variable says, only the times follow it;
  a `--cpuset-cpus` limit is honoured and is reflected in `compute_units`.
- **libgomp pins the importing thread at load time** under `OMP_PROC_BIND`: a
  process that imported an OpenMP-linked module and then spawns a child hands it
  a one-core affinity mask, and the child's OpenMP runtime builds its places from
  that mask. `fnnbench`'s per-thread-count children reset their affinity to the
  cgroup's effective cpuset.
- **gcc's GCN offload plugin aborts without an AMD GPU**: importing a gcc-14
  amdgcn build of ompnn on a host without `/dev/kfd` ends in `libgomp: GCN fatal
  error: Run-time could not be initialized` at the first OpenMP runtime call,
  where clang's offload runtime falls back to the host. Compile-only checks of
  that build are fine on any machine; running it needs the GPU.
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
