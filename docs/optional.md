# Optional experiments (optional experiments)

The four optional experiments of the study, what was run and what came out. Numbers
are steady epochs (median per-epoch wall time excluding the first epoch of
each call, unprofiled, third pass 2026-09-05 unless stated); the rows are in
`results/<machine>/2026-09-05/` and `results/ws-amd/zluda/`.

## 1. Hand-written tiled GEMM in the three models (E7)

Done as `Options.blas = "tiled"` in all three libraries: the same 16x16
local/shared-memory tiled GEMM (op(A) tile stored transposed, one work-item
per C element, sequential k accumulation in a register), a row-per-work-item
GEMV and reductions for asum/nrm2. Protocol, kernel-equivalence notes and
the gcc-offload exception are in `docs/methodology.md` (E7 caveats) and
`docs/toolchains.md` (Hand-written BLAS). The tiled/vendor ratios per device
and toolchain are in `analysis/headline-<machine>-<date>.md` and the
`tiled-*.png` figures.

## 2. DPC++ vs AdaptiveCpp on the same GPU

- **RX 7900 XTX**: AdaptiveCpp only. DPC++ on AMD needs the HIP adapter and
  the amdgcn libspirv, which the intel/llvm 7.1.0 release tarball does not
  ship and which no longer builds from source against ROCm 7.2's LLVM-22
  device libraries (`scripts/dpcpp-hip-toolchain.sh`, kept as the
  record of the attempt; `docs/toolchains.md`).
- **GTX 1080 Ti**: both. The AdaptiveCpp wheel (`containers/fnn-acpp-cuda.Containerfile`:
  AdaptiveCpp v25.10.0 generic SSCP, oneMath cuBLAS + Netlib) runs the same
  E2/E3/E7 rows as the DPC++ wheel (`scripts/ws-nvidia-acpp.sh`, sweep
  `sycl-gpu-acpp`). Results: see the "DPC++ vs AdaptiveCpp" table below
  (filled from `analysis/headline-ws-nvidia-2026-09-05.md`).
- **CPU**: both, on ws-amd's Threadripper 2950X: DPC++ through the Intel
  OpenCL CPU runtime (`sycl-cpu*`), AdaptiveCpp through its OpenMP host
  device (`sycl-cpu-acpp`, 16 threads). See the same table.

<!-- DPCPP-VS-ACPP-TABLE -->

## 3. Portability: one binary, four devices

The *same* `_syclnn` shared object, identified by SHA-256, run on every device
it can reach; the *same* `ompnn` source built by four compilers (the E4
matrix); `cudann` on NVIDIA only, plus its HIPified build on AMD and, below,
the unmodified NVIDIA wheel on AMD through ZLUDA.

| wheel (SHA-256 of `_syclnn*.so`) | built on / targets | device | mnist-512-256 b256 float | monk b40 | cup b40 | sweep dir |
|---|---|---|---|---|---|---|
| AdaptiveCpp generic SSCP, `c698b8a0...0374657` | ws-amd, one generic IR (JIT per device) | RX 7900 XTX (HIP) | 215 ms (tiled 102 ms) | 1.9 ms (tiled 1.0) | 17.5 ms (tiled 8.2) | `results/ws-amd/2026-09-05/portable-acpp` |
| same | | TR 2950X (OpenMP host device, 16 threads) | 956 ms (tiled 2288 ms) | 0.9 ms (tiled 1.6) | 7.0 ms (tiled 8.4) | same |
| same | | GTX 1080 Ti (CUDA) | see `results/ws-nvidia/2026-09-05/portable-acpp` | | | `scripts/ws-nvidia-portable-acpp.sh` |
| DPC++ `spir64` + `nvidia_gpu_sm_61`, `2a41db17...6945e99` | ws-nvidia, two AOT targets | TR 2950X (OpenCL CPU, 16 compute units) | 998 ms (tiled 2381 ms) | 1.9 ms (tiled 1.4) | 13.8 ms (tiled 11.4) | `results/ws-amd/2026-09-05/portable-dpcpp` |
| same | | GTX 1080 Ti | the `sycl-gpu` rows of ws-nvidia (same wheel) | | | `results/ws-nvidia/2026-09-05/sycl-gpu` |
| same | | RX 7900 XTX | **cannot**: no HIP adapter in the DPC++ release; a fat binary with `amd_gpu_gfx1100` does not build (see 2.) | | | |

What the table shows: the AdaptiveCpp binary carries one generic IR and runs
on the CPU, on RDNA3 and on Pascal unchanged; the DPC++ binary carries one
SPIR-V and one PTX/SASS image per target listed at build time (a second
NVIDIA target, `sm_80`, made the wheel fail to load on Pascal: one CUDA
target per wheel, `docs/toolchains.md`) and reaches exactly those. ompnn's
source builds with gcc-14, clang-18, clang-22, amdclang++ and (host path)
gcc+MKL; the offload variants differ by an order of magnitude in what they
deliver (gcc: one thread per warp/wavefront). cudann needs `hipify-perl` and a
second build for AMD.

## 4. ZLUDA: the unmodified CUDA wheel on the RX 7900 XTX

`scripts/zluda-crosscheck.sh`; ZLUDA v7-preview.10 (`zluda-linux-9c8b43f2`),
the `cudann` wheel built on ws-nvidia with nvcc 12.9 for sm_61+sm_80, CUDA 12.9's
`libcudart.so.12` from the fnn-cuda image, ZLUDA's own `libcuda.so.1` and
`libcublas.so.12` in front on `LD_LIBRARY_PATH`, inside the `fnn-rocm`
distrobox (ROCm 7.2.4). Output: `results/ws-amd/zluda/README.md`, rows in
`results/ws-amd/zluda/rows/`.

Does it run:

| feature | ZLUDA v7-preview.10 on gfx1100 |
|---|---|
| module load, device enumeration | yes: `AMD Radeon RX 7900 XTX [ZLUDA]`, "CUDA runtime 12.9", reported compute capability 8.5 |
| the library's own kernels (element-wise, fused Adam, reductions, the tiled GEMM/GEMV) | yes, numerically correct (`check=ok` on every row with `blas=tiled`) |
| cuBLAS `gemm` | yes (`bias_gemv=False` rows: the only cuBLAS calls left are GEMMs) |
| cuBLAS `gemv` | **no**: `cublasSgemv` returns status 15 (`CUBLAS_STATUS_NOT_SUPPORTED`), so every default-configuration row fails at the bias-gradient GEMV |
| cuBLAS `nrm2` | **no**: `cublasSnrm2` status 15 (the monk/cup rows with a weight-decay penalty) |
| streams, events, the fork/join default (4 streams, out-of-order) | yes (the tiled rows use the default stream mode) |
| CUDA graphs, managed/host memory, pinned host | not reachable: the parity suite constructs default-option networks first and stops at the GEMV |

Speed (steady epoch, float; native rows from the same pass on the same GPU):

| workload | cudann on ZLUDA, `blas=tiled` | cudann/HIP (hipify), `blas=tiled` | cudann on ZLUDA, cuBLAS gemm (`bias_gemv=False`) | cudann/HIP, rocBLAS | syclnn/AdaptiveCpp, `blas=tiled` | syclnn/AdaptiveCpp, rocBLAS |
|---|---|---|---|---|---|---|
| monk b40 | 0.69 ms | 0.50 ms | fails (nrm2) | 0.85 ms | 1.01 ms | 2.35 ms |
| cup b40 | 4.00 ms | 4.86 ms | fails (nrm2) | 4.55 ms | 8.18 ms | 20.7 ms |
| mnist-512-256 b256 | 79.2 ms | 85.5 ms | 74.5 ms | 65.5 ms | 102 ms | 228 ms |
| mnist-512-256 b1024 | 49.2 ms | - | 28.6 ms | 21.0 ms | - | 63.5 ms |

Reading: the CUDA binary's own kernels run on RDNA3 through ZLUDA at the
speed of the HIPified source build (79 vs 85 ms with the same tiled GEMM; the
PTX is JIT-compiled through ZLUDA's own back end), and ZLUDA's cuBLAS
translation reaches 74.5 ms against 65.5 ms for rocBLAS called natively
(14 % slower at b256, 36 % at b1024). What breaks is coverage of the BLAS
API: two level-1/level-2 routines the library uses are not implemented in
this preview, and a real application would hit them at once. Binary-level
CUDA-on-AMD is therefore a demonstration, not a deployment path, in this
version; source-level HIPify (one script, everything runs) and write-once
SYCL (one binary, everything runs, 1.5-3x slower than the CUDA/HIP one with
the vendor BLAS and on par with the tiled one) are the two working routes.
