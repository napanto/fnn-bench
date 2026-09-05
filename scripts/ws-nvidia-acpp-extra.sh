#!/usr/bin/env bash
# AdaptiveCpp wheel on ws-nvidia's 1080 Ti, second part: the parity suites (saved to
# results/ws-nvidia/<date>/sycl-gpu-acpp/parity.txt this time) and the E6 ablation rows, so the
# DPC++ vs AdaptiveCpp comparison has the queue/stream/BLAS-queue switches on both.
#   DATE=2026-09-05 fnn-bench/scripts/ws-nvidia-acpp-extra.sh [pid-to-wait-for]
set -uo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
HOST=${NVIDIA_HOST:-ws-nvidia}
REMOTE=${NVIDIA_DIR:-$HOME/fnn}
DATE=${DATE:-2026-09-05}
WAIT=${1:-}
RUN="podman run --rm --memory=40g --device nvidia.com/gpu=all --security-opt=label=disable -v $REMOTE/syclnn:/work/syclnn -v $REMOTE/fnn-bench:/work/fnn-bench"
log() { echo "==== [$(date +%T)] $*"; }
while [ -n "$WAIT" ] && kill -0 "$WAIT" 2>/dev/null; do sleep 60; done
log "ws-nvidia: AdaptiveCpp parity suites + E6 ablations on the 1080 Ti"
ssh "$HOST" "$RUN -w /work/syclnn localhost/fnn-acpp-cuda:dev bash -c '
    export FNN_REF_CACHE=/work/fnn-bench/.cache/ref; pip install -q --no-deps -e /work/fnn-bench/testkit -e /work/fnn-bench/bench
    export PYTHONPATH=/work/fnn-bench/.wheels/ws-nvidia-sycl-acpp
    P=/work/fnn-bench/results/ws-nvidia/$DATE/sycl-gpu-acpp/parity.txt; : > \$P
    for o in \"--dtype double\" \"--dtype float\" \"--option queue=in_order\" \"--option streams=1\" \"--option blas_queue=dedicated\" \"--option memory=shared\" \"--option memory=host\" \"--option blas=tiled --dtype float\"; do
        printf \"syclnn AdaptiveCpp cuda:gpu %-34s \" \"\$o\" | tee -a \$P; pytest -q --device gpu -p no:cacheprovider \$o 2>&1 | grep -E \"passed|failed|error\" | tail -1 | tee -a \$P
    done
    printf \"syclnn AdaptiveCpp cuda:gpu %-34s \" \"--run-slow -k mnist\" | tee -a \$P; pytest -q --device gpu --run-slow -k mnist -p no:cacheprovider 2>&1 | grep -E \"passed|failed|error\" | tail -1 | tee -a \$P
    cd /work/fnn-bench
    fnnbench sweep --plan plans/e6_sycl_ablations.json --select device=gpu --out results/ws-nvidia/$DATE/sycl-gpu-acpp
'" 2>&1 | grep -v 'AdaptiveCpp Warning'
log "ws-nvidia acpp extra done"
rsync -a "$HOST:$REMOTE/fnn-bench/results/ws-nvidia/" "$HERE/results/ws-nvidia/"
