#!/usr/bin/env bash
# Portability (optional experiments): ws-amd's AdaptiveCpp generic-binary syclnn wheel (built for the RX 7900 XTX and
# the CPU) run unchanged on ws-nvidia's GTX 1080 Ti inside fnn-acpp-cuda (acpp with the CUDA backend,
# oneMath cuBLAS + NETLIB built with acpp). Rows -> results/ws-nvidia/<date>/portable-acpp.
#   DATE=2026-09-05 fnn-bench/scripts/ws-nvidia-portable-acpp.sh
set -uo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
HOST=${NVIDIA_HOST:-ws-nvidia}
REMOTE=${NVIDIA_DIR:-$HOME/fnn}
DATE=${DATE:-$(date +%F)}
P=$HOME/.local/opt/fnn-rocm
ACPP_WHEEL=$P/venv/lib/python3.12/site-packages
ssh "$HOST" "mkdir -p $REMOTE/fnn-bench/.wheels/portable-acpp"
rsync -a "$ACPP_WHEEL/syclnn/" "$HOST:$REMOTE/fnn-bench/.wheels/portable-acpp/syclnn/"
ssh "$HOST" "R=$REMOTE; podman run --rm --memory=40g --device nvidia.com/gpu=all --security-opt=label=disable -v \$R/syclnn:/work/syclnn -v \$R/fnn-bench:/work/fnn-bench -w /work/fnn-bench localhost/fnn-acpp-cuda:dev bash -c '
    pip install -q --no-deps -e /work/fnn-bench/testkit -e /work/fnn-bench/bench; export PYTHONPATH=/work/fnn-bench/.wheels/portable-acpp LD_LIBRARY_PATH=/opt/onemath-acpp/lib:/opt/acpp/lib:/opt/openblas-openmp/lib FNN_REF_CACHE=/work/fnn-bench/.cache/ref ACPP_TARGETS=generic
    sha256sum /work/fnn-bench/.wheels/portable-acpp/syclnn/_syclnn*.so; python -c \"import syclnn; print(syclnn.build_info()[\\\"git_sha\\\"], [d[\\\"name\\\"] for d in syclnn.devices()])\" 2>&1 | tail -2
    cd /work/syclnn && pytest -q --device gpu --dtype float -p no:cacheprovider 2>&1 | tail -1; cd /work/fnn-bench
    for dev in cpu gpu; do for b in tiled auto; do for wl in \"monk 40\" \"cup 40\" \"mnist-512-256 256\"; do set -- \$wl; fnnbench run --backend syclnn --device \$dev --workload \$1 --batch \$2 --dtype float --epochs 5 --repeat 3 --warmup 1 --option blas=\$b --out results/ws-nvidia/$DATE/portable-acpp --tag portable-acpp-ws-nvidia-\$dev 2>&1 | tail -1; done; done; done'" 2>&1 | grep -vE 'Warning|^\s*$'
rsync -a "$HOST:$REMOTE/fnn-bench/results/ws-nvidia/" "$HERE/results/ws-nvidia/"
