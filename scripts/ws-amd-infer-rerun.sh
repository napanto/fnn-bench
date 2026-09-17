#!/usr/bin/env bash
# Re-measure every inference row of ws-amd's third pass (mode=infer: E2 cpu/gpu, E5 inference and
# breakdown, the AdaptiveCpp CPU/GPU rows, the OpenMP host and offload rows) with the zero-copy
# bindings (syclnn 0.2.1, cudann 0.1.1, ompnn 0.1.1): the old rows timed a host copy of the input
# inside the predict call (docs/methodology.md, Inference rows). Old rows -> superseded.jsonl.
# Same conditions as the third pass: CPU at the fixed 2.8 GHz (boost off), thermal guard running,
# every CPU runtime on the 16 physical cores. The fnn-rocm venvs must already hold the fixed wheels
# (syclnn/AdaptiveCpp + cudann/HIP in venv, ompnn in venv-amdclang and venv-gcc14amd); the DPC++
# and OpenMP host wheels are rebuilt here inside the images.
#   fnn-bench/scripts/ws-amd-infer-rerun.sh [date]
set -uo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
AC=$(cd "$HERE/.." && pwd)
DATE=${1:-2026-09-05}
RES=results/ws-amd/$DATE
PY=$HOME/.local/opt/fnn-rocm/venv/bin/python
PODMAN="podman run --rm --memory=20g --security-opt label=disable -v $AC/syclnn:/work/syclnn -v $AC/ompnn:/work/ompnn -v $HERE:/work/fnn-bench"
log() { echo "==== [$(date +%T)] $*"; }
cd "$HERE"

cap=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq); boost=$(cat /sys/devices/system/cpu/cpufreq/boost 2>/dev/null || echo ?)
if [ "$cap" != 2800000 ] || [ "$boost" != 0 ]; then echo "CPU cap not in force (max $cap kHz, boost $boost): refusing to run"; exit 1; fi
pgrep -f 'thermal-guar[d]\.sh' >/dev/null || { echo "thermal-guard.sh is not running: refusing to run"; exit 1; }

log "supersede the inference rows of $RES"
$PY - <<PYEOF
import json
from pathlib import Path
root = Path("$RES"); total = 0
for f in sorted(root.rglob("*.jsonl")):
    if f.name in ("superseded.jsonl", "peak.jsonl"): continue
    rows = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    drop = [r for r in rows if r.get("mode") == "infer"]; keep = [r for r in rows if r.get("mode") != "infer"]
    if drop:
        with (f.parent / "superseded.jsonl").open("a") as fh:
            for r in drop: fh.write(json.dumps(r) + "\n")
        f.write_text("".join(json.dumps(r) + "\n" for r in keep)); total += len(drop)
        print(f"{f}: superseded {len(drop)} inference rows, kept {len(keep)}")
print("superseded", total, "rows")
PYEOF

log "1. RX 7900 XTX: syclnn/AdaptiveCpp + cudann/HIP (fnn-rocm venv)"
distrobox enter fnn-rocm -- bash -c "
    P=\$HOME/.local/opt/fnn-rocm; export PATH=\$P/venv/bin:\$PATH FNN_REF_CACHE=$HERE/.cache/ref OMP_NUM_THREADS=16; cd $HERE
    pip install -q --no-deps -e testkit -e bench
    fnnbench sweep --plan plans/e2_sycl_cpu_gpu.json --select device=gpu --select mode=infer --out $RES/sycl-gpu-acpp 2>&1 | grep -v 'AdaptiveCpp Warning'
    for b in syclnn cudann; do
        fnnbench sweep --plan plans/e5_inference.json --select device=gpu --select backend=\$b --out $RES/e5-infer-gpu 2>&1 | grep -v 'AdaptiveCpp Warning'
        fnnbench sweep --plan plans/e5_breakdown.json --select device=gpu --select backend=\$b --select mode=infer --out $RES/e5-breakdown-gpu 2>&1 | grep -v 'AdaptiveCpp Warning'
    done"

log "2. ompnn offload on the RX 7900 XTX: amdclang++, gcc-14 amdgcn"
distrobox enter fnn-rocm -- bash -c "
    P=\$HOME/.local/opt/fnn-rocm; export FNN_REF_CACHE=$HERE/.cache/ref; cd $HERE
    for v in venv-amdclang venv-gcc14amd; do
        export PATH=\$P/\$v/bin:\$PATH; pip install -q --no-deps -e testkit -e bench
        fnnbench sweep --plan plans/e5_inference.json --select device=gpu --select backend=ompnn --out $RES/e5-infer-omp-gpu-\${v#venv-}
        fnnbench sweep --plan plans/e5_breakdown.json --select device=gpu --select backend=ompnn --select mode=infer --out $RES/e5-breakdown-omp-gpu-\${v#venv-}
    done"

log "3. AdaptiveCpp host device (E2 cpu, E5 breakdown), fnn-rocm venv, 16 cores"
distrobox enter fnn-rocm -- bash -c "
    P=\$HOME/.local/opt/fnn-rocm; export PATH=\$P/venv/bin:\$PATH FNN_REF_CACHE=$HERE/.cache/ref OMP_NUM_THREADS=16; cd $HERE
    fnnbench sweep --plan plans/e2_sycl_cpu_gpu.json --select device=cpu --select dtype=float --select mode=infer --out $RES/sycl-cpu-acpp 2>&1 | grep -v 'AdaptiveCpp Warning'
    fnnbench sweep --plan plans/e5_breakdown.json --select device=cpu --select backend=syclnn --select dtype=float --select mode=infer --out $RES/e5-breakdown-cpu-acpp 2>&1 | grep -v 'AdaptiveCpp Warning'"

log "4. DPC++ opencl:cpu (fnn-sycl image, wheel rebuilt): E2 cpu, E5 inference + breakdown"
$PODMAN -w /work/syclnn localhost/fnn-sycl:dev bash -c "
    export CC=clang CXX=clang++ SYCLNN_TARGETS='spir64;nvidia_gpu_sm_61;nvidia_gpu_sm_80' SYCLNN_ONEMATH_ROOT=/opt/onemath CMAKE_BUILD_PARALLEL_LEVEL=8
    rm -rf /work/fnn-bench/.wheels/sycl-dpcpp
    pip install -q --no-deps --target /work/fnn-bench/.wheels/sycl-dpcpp --config-settings=build-dir=/tmp/syclnn-build . 2>&1 | grep -E 'error:' || true
    export FNN_REF_CACHE=/work/fnn-bench/.cache/ref; pip install -q --no-deps -e /work/fnn-bench/testkit -e /work/fnn-bench/bench
    export PYTHONPATH=/work/fnn-bench/.wheels/sycl-dpcpp OMP_PROC_BIND=close OMP_PLACES=cores DPCPP_CPU_NUM_CUS=16
    cd /work/fnn-bench
    fnnbench sweep --plan plans/e2_sycl_cpu_gpu.json --select device=cpu --select mode=infer --out $RES/sycl-cpu
    fnnbench sweep --plan plans/e5_inference.json --select device=cpu --select backend=syclnn --out $RES/e5-infer-cpu-dpcpp
    fnnbench sweep --plan plans/e5_breakdown.json --select device=cpu --select backend=syclnn --select mode=infer --out $RES/e5-breakdown-cpu-dpcpp"

log "5. ompnn host path, gcc-14 (fnn-cuda image, wheel rebuilt): E5 inference + breakdown"
$PODMAN -w /work/ompnn localhost/fnn-cuda:dev bash -c "
    export FNN_REF_CACHE=/work/fnn-bench/.cache/ref; pip install -q --no-deps -e /work/fnn-bench/testkit -e /work/fnn-bench/bench
    export CXX=g++-14 OMPNN_TARGET=cpu OMPNN_BLAS=openblas OMPNN_BLAS_ROOT=/opt/openblas-openmp CMAKE_BUILD_PARALLEL_LEVEL=8
    rm -rf /work/fnn-bench/.wheels/omp-gcc14
    pip install -q --no-deps --target /work/fnn-bench/.wheels/omp-gcc14 --config-settings=build-dir=/tmp/ompnn-gcc14 . 2>&1 | grep -E 'error:' || true
    export OMP_PROC_BIND=close OMP_PLACES=cores PYTHONPATH=/work/fnn-bench/.wheels/omp-gcc14; cd /work/fnn-bench
    fnnbench sweep --plan plans/e5_inference.json --select device=cpu --select backend=ompnn --out $RES/e5-infer-omp-cpu-gcc14
    fnnbench sweep --plan plans/e5_breakdown.json --select device=cpu --select backend=ompnn --select mode=infer --out $RES/e5-breakdown-omp-cpu-gcc14"

log "inference re-run done: $(find $RES -name '*.jsonl' -newer $HERE/scripts/ws-amd-infer-rerun.sh | wc -l) result files touched"
