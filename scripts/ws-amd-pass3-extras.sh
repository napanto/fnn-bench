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

log "1. DPC++ on the RX 7900 XTX: not possible with the intel/llvm release tarball (no amdgcn libspirv, no HIP adapter); skipped"

log "2a. portability, AdaptiveCpp generic binary: ws-amd's acpp wheel on CPU + 7900 XTX here, then on ws-nvidia's 1080 Ti"
ACPP_WHEEL=$P/venv/lib/python3.12/site-packages
sha256sum "$ACPP_WHEEL"/syclnn/_syclnn*.so | tee "$HERE/results/ws-amd/$DATE/portable-acpp.SHA256" 2>/dev/null || true
distrobox enter fnn-rocm -- bash -lc "
    P=$P; export PATH=\$P/venv/bin:\$PATH FNN_REF_CACHE=$HERE/.cache/ref OMP_NUM_THREADS=16; cd $HERE
    for dev in cpu gpu; do for b in tiled auto; do for wl in 'monk 40' 'cup 40' 'mnist-512-256 256'; do set -- \$wl
        fnnbench run --backend syclnn --device \$dev --workload \$1 --batch \$2 --dtype float --epochs 5 --repeat 3 --warmup 1 --option blas=\$b --out results/ws-amd/$DATE/portable-acpp --tag portable-acpp-ws-amd-\$dev 2>&1 | tail -1
    done; done; done
" 2>&1 | grep -vE 'Warning|^\s*$' | tail -13
rsync -a "$ACPP_WHEEL/syclnn/" ws-nvidia:$HOME/fnn/fnn-bench/.wheels/portable-acpp/syclnn/
ssh ws-nvidia 'R=$HOME/fnn; podman image exists localhost/fnn-acpp-cuda:dev && podman run --rm --memory=40g --device nvidia.com/gpu=all --security-opt=label=disable -v $R/syclnn:/work/syclnn -v $R/fnn-bench:/work/fnn-bench -w /work/fnn-bench localhost/fnn-acpp-cuda:dev bash -c "
    pip install -q --no-deps -e /work/fnn-bench/testkit -e /work/fnn-bench/bench; export PYTHONPATH=/work/fnn-bench/.wheels/portable-acpp LD_LIBRARY_PATH=/opt/onemath-acpp/lib:/opt/acpp/lib:/opt/openblas-openmp/lib:\${LD_LIBRARY_PATH:-} FNN_REF_CACHE=/work/fnn-bench/.cache/ref ACPP_TARGETS=generic
    sha256sum /work/fnn-bench/.wheels/portable-acpp/syclnn/_syclnn*.so; python -c \"import syclnn; print([d[\\\"name\\\"] for d in syclnn.devices()])\"
    cd /work/syclnn && pytest -q --device gpu --dtype float -p no:cacheprovider 2>&1 | tail -1; cd /work/fnn-bench
    for dev in cpu gpu; do for b in tiled auto; do for wl in \"monk 40\" \"cup 40\" \"mnist-512-256 256\"; do set -- \$wl; fnnbench run --backend syclnn --device \$dev --workload \$1 --batch \$2 --dtype float --epochs 5 --repeat 3 --warmup 1 --option blas=\$b --out results/ws-nvidia/'"$DATE"'/portable-acpp --tag portable-acpp-ws-nvidia-\$dev 2>&1 | tail -1; done; done; done" || echo "fnn-acpp-cuda image not built yet on ws-nvidia"' 2>&1 | grep -vE 'Warning|^\s*$|sign_and_send' | tail -16

log "2b. portability, DPC++ wheel (spir64 + nvidia_gpu_sm_61, built on ws-nvidia): on ws-amd's CPU"
podman run --rm --memory=20g --security-opt label=disable -v "$AC/syclnn:/work/syclnn" -v "$HERE:/work/fnn-bench" -w /work/fnn-bench localhost/fnn-sycl:dev bash -c '
    pip install -q --no-deps -e /work/fnn-bench/testkit -e /work/fnn-bench/bench; export PYTHONPATH=/work/fnn-bench/.wheels/ws-nvidia-sycl FNN_REF_CACHE=/work/fnn-bench/.cache/ref OMP_NUM_THREADS=16
    sha256sum /work/fnn-bench/.wheels/ws-nvidia-sycl/syclnn/_syclnn*.so | tee /work/fnn-bench/results/ws-amd/'"$DATE"'/portable-dpcpp.SHA256; python -c "import syclnn; print([d[\"name\"] for d in syclnn.devices()])"
    for b in tiled auto; do for wl in "monk 40" "cup 40" "mnist-512-256 256"; do set -- $wl; fnnbench run --backend syclnn --device cpu --workload $1 --batch $2 --dtype float --epochs 5 --repeat 3 --warmup 1 --option blas=$b --out results/ws-amd/'"$DATE"'/portable-dpcpp --tag portable-dpcpp-ws-amd-cpu 2>&1 | tail -1; done; done' 2>&1 | grep -vE '^\s*$' | tail -8
rsync -a ws-nvidia:$HOME/fnn/fnn-bench/results/ws-nvidia/ "$HERE/results/ws-nvidia/" 2>/dev/null

log "3. ZLUDA"
distrobox enter fnn-rocm -- bash -lc "bash $HERE/scripts/zluda-crosscheck.sh" 2>&1 | grep -vE '^\s*$' | tail -25

log "analyze"
distrobox enter fnn-rocm -- bash -lc "export PATH=$P/venv/bin:\$PATH; bash $HERE/scripts/analyze.sh" 2>&1 | grep -vE 'Warning' | tail -6
log "pass3 extras done"
