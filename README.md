# fnn-bench

The umbrella repository of the *SYCL vs CUDA vs OpenMP* study built on
[syclnn](https://github.com/napanto/syclnn), [cudann](https://github.com/napanto/cudann)
and [ompnn](https://github.com/napanto/ompnn) (Accelerated Computing course,
University of Pisa, 2026): the NumPy oracle and the shared parity test-suite,
the benchmark harness, the container images, the experiment plans, the raw
results and the analysis.

## Layout

| Directory | Contents |
|---|---|
| `testkit/` | `fnn-testkit`: the float64 NumPy reference implementation and the pytest parity suite run against every backend |
| `bench/` | `fnnbench`, the benchmark harness (`run`, `sweep`, `collect`, `replay`, `peak`, `plot`): FLOP model, system capture, one JSONL row per measurement |
| `plans/` | the experiments E1-E7 and the W4 sweep as JSON plans for `fnnbench sweep` (defined in `docs/methodology.md`) |
| `containers/` | the toolchain images: `fnn-cuda`, `fnn-sycl`, `fnn-sycl-generic`, `fnn-acpp-cuda` (AdaptiveCpp on CUDA), `fnn-rocm` (the AMD stack) |
| `scripts/` | the measurement drivers of each machine, toolchain set-up, table generators and profiler cross-checks (see [Scripts](#scripts)) |
| `docs/` | `methodology.md` (protocol, metrics, caveats), `toolchains.md` (compiler × device matrix), `optional.md` (optional experiments) |
| `results/` | submodule → [fnn-results](https://github.com/napanto/fnn-results): the raw rows per machine and pass, and `results/analysis/` (CSVs, figures, headline tables) |

## Machines

| id | CPU | GPU | role |
|---|---|---|---|
| `ws-amd` | AMD Ryzen Threadripper 2950X (16 cores, Zen+) | AMD Radeon RX 7900 XTX (RDNA3, gfx1100), ROCm 7.2.4 | CPU rows (DPC++ OpenCL, AdaptiveCpp host, OpenMP host), AMD GPU rows (AdaptiveCpp, HIP, OpenMP offload) |
| `ws-nvidia` | Intel Xeon E5-2643 v2 (2 x 6 cores, Ivy Bridge) | NVIDIA GeForce GTX 1080 Ti (Pascal, sm_61), CUDA 12.9 | NVIDIA GPU rows (DPC++, AdaptiveCpp, CUDA, OpenMP offload) |
| `hpc-a30` | 2 x AMD EPYC 7301 (2 x 16 cores, Zen, 8 NUMA nodes) | NVIDIA A30 (Ampere, sm_80), driver 610 (CUDA 13.3 host, CUDA 12.9 images) | every row family on one exclusive SLURM node (30-minute jobs): CPU rows (DPC++ OpenCL, AdaptiveCpp host, OpenMP host incl. nvc++ multicore), NVIDIA GPU rows (DPC++, AdaptiveCpp, CUDA, OpenMP offload incl. nvc++) |

Result directories, analysis files and driver scripts are named after these ids.

## Reproduce

The measurements live in the [fnn-results](https://github.com/napanto/fnn-results)
repository, checked out as the `results/` submodule: clone with
`--recurse-submodules` (or `git submodule update --init`) to get the rows and
the figures; the drivers below write new rows into it.

```sh
# images (or pull ghcr.io/napanto/fnn-{cuda,sycl})
podman build --memory=20g -f containers/fnn-cuda.Containerfile -t fnn-cuda:dev containers/
podman build --memory=20g -f containers/fnn-sycl.Containerfile -t fnn-sycl:dev containers/

# oracle + tests
python -m venv .venv && .venv/bin/pip install -e testkit -e bench matplotlib
.venv/bin/pytest --pyargs fnn_testkit                          # the oracle checks itself
# a backend (inside its image / venv):
pytest --pyargs fnn_testkit --backend syclnn --device cpu --blas mklcpu

# one measurement, a sweep, the whole local matrix
fnnbench run --backend cudann --device gpu --workload mnist --batch 256 --dtype float --epochs 5 --repeat 5 --option profile=True --out results/ws-nvidia/$(date +%F)
fnnbench sweep --plan plans/e3_cuda_vs_sycl_gpu.json --out results/ws-nvidia/$(date +%F)/gpu
scripts/matrix-ws-amd.sh                                        # ws-amd: all environments, sequentially
scripts/run-ws-nvidia.sh all                                    # ws-nvidia: sync, build, parity, matrix, fetch (NVIDIA_HOST, NVIDIA_DIR)
HPC_HOST=<front-end> scripts/run-hpc.sh all                     # hpc-a30: sync, wheels, parity jobs, matrix job chains (then status / fetch)
scripts/analyze.sh                                              # the reported passes -> results/analysis/<machine>-<date>.csv, results/analysis/figures/<machine>-<date>/
PYTHONPATH=bench python scripts/headline.py results/ws-amd/2026-09-05   # Markdown: defaults per toolchain, tiled/vendor ratios, run-to-run spread (needs numpy)
```

The rows the report uses are `results/ws-amd/2026-09-05`, `results/ws-nvidia/2026-09-05` and
`results/hpc-a30/2026-09-17` in fnn-results (timing rows unprofiled, oracle check on a
separate instance, the measurement conditions in `docs/methodology.md`). Rows replaced by a
re-measurement live in `superseded.jsonl` next to the live file and are
skipped by every reader.

See `docs/methodology.md` for what a row means (two per-epoch numbers, the
bounded oracle check, the FLOP model), `docs/toolchains.md` for which
compiler/device combinations are validated and `docs/optional.md` for the
optional experiments (hand-written GEMM, DPC++ vs AdaptiveCpp, portability,
ZLUDA).

## Scripts

| Purpose | Scripts |
|---|---|
| Measurement matrix, one driver per machine | `matrix-ws-amd.sh`; `run-ws-nvidia.sh` (over ssh + podman CDI); `run-hpc.sh` + `hpc-blocks.sh` (SLURM job chains, sandbox images run with apptainer) |
| ws-amd at a fixed host clock | `ws-amd-cpu-fixed-clock.sh`, `ws-amd-gpu-fixed-clock.sh`, `thermal-guard.sh` |
| Re-measurements | `ws-amd-infer-rerun.sh`, `run-ws-nvidia.sh infer-rerun`; `supersede-stale.py`, `supersede-backends.py` (move replaced rows to `superseded.jsonl`) |
| AdaptiveCpp on CUDA, one-binary portability | `ws-nvidia-acpp.sh`, `ws-nvidia-portable-acpp.sh` |
| Toolchains and images | `rocm-toolchain.sh` (AdaptiveCpp + oneMath for the RX 7900 XTX), `dpcpp-hip-toolchain.sh` (the DPC++-on-AMD attempt, kept with `containers/fnn-sycl-portable.Containerfile`), `publish-amd-variants.sh` (the AMD tags of the library images) |
| Tables and figures | `analyze.sh`, `headline.py`, `peaks-table.py`, `acpp-vs-dpcpp.py` |
| Profiler and runtime cross-checks | `nsys-crosscheck.py`, `rocprof-crosscheck.sh`, `perf-crosscheck.sh`, `zluda-crosscheck.sh` |

## Open items

- Run the AMD tags of the library images (`syclnn:acpp-rocm`, `cudann:hip`, `ompnn:rocm`) from the
  registry on the RX 7900 XTX and pass the parity suite there: they have only been checked at build
  time (`docs/toolchains.md`, published images).
