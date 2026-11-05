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
