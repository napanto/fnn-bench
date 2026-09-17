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
containers/   fnn-cuda, fnn-sycl, fnn-sycl-generic, fnn-acpp-cuda (AdaptiveCpp on CUDA), fnn-rocm (the AMD stack) Containerfiles;
              fnn-sycl-portable is kept as the record of the DPC++-on-AMD attempt (with scripts/dpcpp-hip-toolchain.sh);
              every image carries FNN_IMAGE / FNN_IMAGE_BUILT, which fnnbench records in each row
plans/        the experiments as JSON plans for `fnnbench sweep` (E1-E7 and the W4 sweep, defined in docs/methodology.md; e5_breakdown is the only profiled plan)
scripts/      rocm-toolchain.sh (AdaptiveCpp + oneMath for the RX 7900 XTX), matrix-ws-amd.sh (the
              ws-amd matrix), run-ws-nvidia.sh (the ws-nvidia matrix over ssh + podman CDI),
              ws-nvidia-acpp.sh / ws-nvidia-portable-acpp.sh (AdaptiveCpp on CUDA, one-binary portability),
              ws-amd-{cpu,gpu}-fixed-clock.sh + thermal-guard.sh (the ws-amd rows at a fixed host clock),
              run-hpc.sh + hpc-blocks.sh (the hpc-a30 matrix: SLURM job chains, sandbox images + apptainer),
              ws-amd-infer-rerun.sh / run-ws-nvidia.sh infer-rerun (the inference rows re-measured with the zero-copy bindings),
              analyze.sh, headline.py, peaks-table.py, acpp-vs-dpcpp.py (tables from the rows),
              supersede-stale.py, zluda-crosscheck.sh, rocprof-crosscheck.sh, perf-crosscheck.sh,
              nsys-crosscheck.py, dpcpp-hip-toolchain.sh (the DPC++-on-AMD attempt),
              publish-amd-variants.sh (the AMD tags of the library images, built on the fnn-rocm base)
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

## Open items

- ~~Fix the inference timing boundary and re-run the inference rows~~ **Done on 2026-09-17**
  (syclnn 0.2.1, cudann 0.1.1, ompnn 0.1.1): the bindings pass the NumPy buffers in place
  (raw-pointer overloads of `train`/`predict`), the library records `predict_wall_ns` per
  call and the harness takes the inference `steady_epoch_s` from it, so both modes share the
  measurement boundary. The pre-fix rows had timed a host copy of the input inside `predict`
  (mnist throughput rows up to 2x in float; latency rows and monk/cup within 3 %); they are in
  `superseded.jsonl`, re-measured on 2026-09-17 on both workstations, and hpc-a30 was measured
  with the fix from the start (`docs/methodology.md`, Inference rows).
- Run the AMD tags of the library images (`syclnn:acpp-rocm`, `cudann:hip`, `ompnn:rocm`) from the
  registry on the RX 7900 XTX and pass the parity suite there: they have only been checked at build
  time (`docs/toolchains.md`, published images).
