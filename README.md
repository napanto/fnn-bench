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
containers/   fnn-cuda, fnn-sycl, fnn-sycl-generic Containerfiles (+ the oneMath/OpenBLAS patch)
plans/        the experiment matrix as JSON plans (E1-E7, W4 sweeps)
scripts/      rocm-toolchain.sh (AdaptiveCpp + oneMath for the RX 7900 XTX), matrix-ws-amd.sh,
              ws-amd-pass2.sh, run-ws-nvidia.sh (NVIDIA runs over ssh + podman CDI), analyze.sh,
              supersede-stale.py, rocprof-crosscheck.sh, perf-crosscheck.sh, nsys-crosscheck.py
docs/         toolchains.md (compiler x device matrix and the facts learned), methodology.md
results/      raw JSONL per machine/date (+ peaks, rocprof/perf/nsys cross-checks); superseded.jsonl = replaced rows
analysis/     figures produced by `fnnbench plot`
report/, slides/   the write-up
```

## Reproduce

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
scripts/run-ws-nvidia.sh all                                       # ws-nvidia: sync, build, parity, matrix, fetch
scripts/analyze.sh                                              # both machines -> analysis/<machine>-<date>.csv, analysis/figures/<machine>-<date>/
```

See `docs/methodology.md` for what a row means (two per-epoch numbers, the
bounded oracle check, the FLOP model) and `docs/toolchains.md` for which
compiler/device combinations are validated.
