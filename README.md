# fnn-bench

The umbrella repository of the *SYCL vs CUDA vs OpenMP* study built on
[syclnn](https://github.com/napanto/syclnn), [cudann](https://github.com/napanto/cudann)
and [ompnn](https://github.com/napanto/ompnn) (Accelerated Computing course,
University of Pisa, 2026): the NumPy oracle and the shared parity test-suite,
the benchmark harness, the container images, the experiment plans, the raw
results and the analysis.

```
testkit/      fnn-testkit: float64 reference implementation + pytest suite run against every backend
bench/        fnnbench: run / sweep / collect / replay / peak / plot, FLOP model, system capture, JSONL rows
containers/   fnn-cuda, fnn-sycl, fnn-sycl-generic, fnn-acpp-cuda (AdaptiveCpp on CUDA) Containerfiles, fnn-sycl-portable (kept as the record of the DPC++-on-AMD attempt, with scripts/dpcpp-hip-toolchain.sh)
plans/        the experiments as JSON plans for `fnnbench sweep` (E1-E7 and the W4 sweep, defined in docs/methodology.md; e5_breakdown is the only profiled plan)
scripts/      rocm-toolchain.sh (AdaptiveCpp + oneMath for the RX 7900 XTX), matrix-ws-amd.sh (the
              ws-amd matrix), run-ws-nvidia.sh (the ws-nvidia matrix over ssh + podman CDI),
              ws-nvidia-acpp.sh / ws-nvidia-portable-acpp.sh (AdaptiveCpp on CUDA, one-binary portability),
              ws-amd-{cpu,gpu}-fixed-clock.sh + thermal-guard.sh (the ws-amd rows at a fixed host clock),
              analyze.sh, headline.py, peaks-table.py, acpp-vs-dpcpp.py (tables from the rows),
              supersede-stale.py, zluda-crosscheck.sh, rocprof-crosscheck.sh, perf-crosscheck.sh,
              nsys-crosscheck.py, dpcpp-hip-toolchain.sh (the DPC++-on-AMD attempt)
docs/         toolchains.md (compiler x device matrix and the facts learned), methodology.md
              (protocol, metrics, caveats, the profiler and CPU fix-ups), optional.md (optional experiments:
              tiled GEMM, DPC++ vs AdaptiveCpp, one-binary portability, ZLUDA)
results/      submodule -> fnn-results: the raw JSONL rows per machine/date (+ peaks, rocprof/perf/nsys
              cross-checks; superseded.jsonl = replaced rows) and results/analysis/ (CSVs, figures, headline tables)
```

## Machines

| id | CPU | GPU | role |
|---|---|---|---|
| `ws-amd` | AMD Ryzen Threadripper 2950X (16 cores, Zen+) | AMD Radeon RX 7900 XTX (RDNA3, gfx1100), ROCm 7.2.4 | CPU rows (DPC++ OpenCL, AdaptiveCpp host, OpenMP host), AMD GPU rows (AdaptiveCpp, HIP, OpenMP offload) |
| `ws-nvidia` | Intel Xeon E5-2643 v2 (2 x 6 cores, Ivy Bridge) | NVIDIA GeForce GTX 1080 Ti (Pascal, sm_61), CUDA 12.9 | NVIDIA GPU rows (DPC++, AdaptiveCpp, CUDA, OpenMP offload) |

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
scripts/analyze.sh                                              # both machines -> results/analysis/<machine>-<date>.csv, results/analysis/figures/<machine>-<date>/
PYTHONPATH=bench python scripts/headline.py results/ws-amd/2026-09-05   # Markdown: defaults per toolchain, tiled/vendor ratios, run-to-run spread (needs numpy)
```

The rows the report uses are the third pass, `results/*/2026-09-05` in fnn-results (timing
rows unprofiled, oracle check on a separate instance, every confounder found
on the way fixed or isolated: `docs/methodology.md`). Rows replaced by a
re-measurement live in `superseded.jsonl` next to the live file and are
skipped by every reader.

See `docs/methodology.md` for what a row means (two per-epoch numbers, the
bounded oracle check, the FLOP model), `docs/toolchains.md` for which
compiler/device combinations are validated and `docs/optional.md` for the
optional experiments (hand-written GEMM, DPC++ vs AdaptiveCpp, portability,
ZLUDA).
