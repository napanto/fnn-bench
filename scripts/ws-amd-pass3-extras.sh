#!/usr/bin/env bash
# The optional experiments on ws-amd, after the pass-3 chain (nothing else may be measuring):
#   1. DPC++ with the HIP adapter on the RX 7900 XTX (SYCL implementation comparison on AMD):
#      toolchain + syclnn wheel (scripts/dpcpp-hip-toolchain.sh), parity, E2/E3/E7 rows
#   2. one-wheel portability: fnn-sycl-portable image, a wheel with spir64 + nvidia_gpu_sm_61 +
#      amd_gpu_gfx1100, run on the CPU and the 7900 XTX here and copied to ws-nvidia's 1080 Ti
#   3. ZLUDA: the nvcc-built cudann wheel on the 7900 XTX (scripts/zluda-crosscheck.sh)
set -uo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
AC=$(cd "$HERE/.." && pwd)
DATE=${DATE:-2026-09-05}
P=$HOME/.local/opt/fnn-rocm
log() { echo "==== [$(date +%T)] $*"; }
export FNN_REF_CACHE=$HERE/.cache/ref

log "1. DPC++ + HIP adapter toolchain (fnn-rocm distrobox)"
distrobox enter fnn-rocm -- bash -lc "bash $HERE/scripts/dpcpp-hip-toolchain.sh" 2>&1 | grep -vE '^\s*$' | tail -15
log "1. parity + rows: syclnn/DPC++ on the RX 7900 XTX"
distrobox enter fnn-rocm -- bash -lc "
    P=$P; export PATH=\$P/dpcpp/bin:\$P/venv-dpcpp-hip/bin:\$PATH LD_LIBRARY_PATH=\$P/dpcpp/lib:\${LD_LIBRARY_PATH:-} FNN_REF_CACHE=$HERE/.cache/ref OMP_NUM_THREADS=8
    cd $AC/syclnn
    for o in '--dtype double' '--dtype float' '--option queue=in_order' '--option blas_queue=dedicated' '--blas tiled --dtype float'; do printf 'syclnn dpcpp hip:gpu %-30s ' \"\$o\"; python -m pytest -q --device gpu -p no:cacheprovider \$o 2>&1 | tail -1; done
    python -m pytest -q --device gpu --run-slow -k mnist -p no:cacheprovider 2>&1 | tail -1
    cd $HERE
    fnnbench sweep --plan plans/e2_sycl_cpu_gpu.json --select device=gpu --out results/ws-amd/$DATE/sycl-gpu-dpcpp
    fnnbench sweep --plan plans/e3_cuda_vs_sycl_gpu.json --select backend=syclnn --out results/ws-amd/$DATE/sycl-gpu-dpcpp
    fnnbench sweep --plan plans/e7_tiled_gemm.json --select device=gpu --select backend=syclnn --out results/ws-amd/$DATE/sycl-gpu-dpcpp
    fnnbench sweep --plan plans/e5_breakdown.json --select device=gpu --select backend=syclnn --out results/ws-amd/$DATE/sycl-gpu-dpcpp
    for dt in float double; do fnnbench peak --backend syclnn --device gpu --dtype \$dt --size 8192 --out results/ws-amd/$DATE/peaks-dpcpp-hip; done
" 2>&1 | grep -vE 'Warning|^\s*$' | tail -12

log "2. portable wheel image + wheel (spir64 + nvidia_gpu_sm_61 + amd_gpu_gfx1100)"
podman image exists localhost/fnn-sycl-portable:dev || podman build --memory=20g -f "$HERE/containers/fnn-sycl-portable.Containerfile" -t fnn-sycl-portable:dev "$HERE/containers/" 2>&1 | tail -3
PODMAN="podman run --rm --memory=20g --security-opt label=disable --device /dev/kfd --device /dev/dri --group-add keep-groups -v $AC/syclnn:/work/syclnn -v $HERE:/work/fnn-bench"
$PODMAN -w /work/syclnn localhost/fnn-sycl-portable:dev bash -c '
    export CC=clang CXX=clang++ SYCLNN_TARGETS="spir64;nvidia_gpu_sm_61;amd_gpu_gfx1100" SYCLNN_ONEMATH_ROOT=/opt/onemath SYCLNN_CT_BACKENDS=netlib CMAKE_BUILD_PARALLEL_LEVEL=8
    rm -rf /work/fnn-bench/.wheels/sycl-portable; pip install -q --no-deps --target /work/fnn-bench/.wheels/sycl-portable --config-settings=build-dir=/tmp/b-portable . 2>&1 | grep -E "error" -A2 | head -8
    sha256sum /work/fnn-bench/.wheels/sycl-portable/syclnn/_syclnn*.so | tee /work/fnn-bench/.wheels/sycl-portable/SHA256
    pip install -q --no-deps -e /work/fnn-bench/testkit -e /work/fnn-bench/bench; export PYTHONPATH=/work/fnn-bench/.wheels/sycl-portable FNN_REF_CACHE=/work/fnn-bench/.cache/ref
    python -c "import syclnn; print(syclnn.build_info()[\"sycl_targets\"]); print([d[\"name\"] for d in syclnn.devices()])"
    cd /work/fnn-bench; for dev in cpu gpu; do for b in tiled auto; do for wl in "monk 40" "cup 40" "mnist-512-256 256"; do set -- $wl; fnnbench run --backend syclnn --device $dev --workload $1 --batch $2 --dtype float --epochs 5 --repeat 3 --warmup 1 --option blas=$b --out results/ws-amd/'"$DATE"'/portable --tag portable-ws-amd-$dev 2>&1 | tail -1; done; done; done' 2>&1 | grep -vE '^\s*$' | tail -16
log "2. the same wheel on ws-nvidia's GTX 1080 Ti"
rsync -a "$HERE/.wheels/sycl-portable/" ws-nvidia:$HOME/fnn/fnn-bench/.wheels/sycl-portable/
ssh ws-nvidia 'R=$HOME/fnn; podman run --rm --memory=40g --device nvidia.com/gpu=all --security-opt=label=disable -v $R/syclnn:/work/syclnn -v $R/fnn-bench:/work/fnn-bench -w /work/fnn-bench localhost/fnn-sycl:dev bash -c "
    pip install -q --no-deps -e /work/fnn-bench/testkit -e /work/fnn-bench/bench; export PYTHONPATH=/work/fnn-bench/.wheels/sycl-portable FNN_REF_CACHE=/work/fnn-bench/.cache/ref
    sha256sum /work/fnn-bench/.wheels/sycl-portable/syclnn/_syclnn*.so; python -c \"import syclnn; print([d[\\\"name\\\"] for d in syclnn.devices()])\"
    cd /work/syclnn && pytest -q --device gpu --dtype float -p no:cacheprovider 2>&1 | tail -1; cd /work/fnn-bench
    for dev in cpu gpu; do for b in tiled auto; do for wl in \"monk 40\" \"cup 40\" \"mnist-512-256 256\"; do set -- \$wl; fnnbench run --backend syclnn --device \$dev --workload \$1 --batch \$2 --dtype float --epochs 5 --repeat 3 --warmup 1 --option blas=\$b --out results/ws-nvidia/'"$DATE"'/portable --tag portable-ws-nvidia-\$dev 2>&1 | tail -1; done; done; done"' 2>&1 | grep -vE '^\s*$' | tail -16
rsync -a ws-nvidia:$HOME/fnn/fnn-bench/results/ws-nvidia/ "$HERE/results/ws-nvidia/"

log "3. ZLUDA"
distrobox enter fnn-rocm -- bash -lc "bash $HERE/scripts/zluda-crosscheck.sh" 2>&1 | grep -vE '^\s*$' | tail -25

log "analyze"
distrobox enter fnn-rocm -- bash -lc "export PATH=$P/venv/bin:\$PATH; bash $HERE/scripts/analyze.sh" 2>&1 | grep -vE 'Warning' | tail -6
log "pass3 extras done"
