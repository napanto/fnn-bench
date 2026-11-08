#!/usr/bin/env bash
# ws-amd, RX 7900 XTX, HIP MNIST b256 with library streams: snapshot of the KFD queue tables (hardware queue
# registers per pipe and queue, queue descriptors per process) taken by a root helper 3 s into every fresh
# process, to compare the queue placement of fast and slow processes. Submitting process on node 0, nice 0.
#   scripts/ws-amd-radeon-queues.sh <ctl-dir> [rounds] [date]
set -uo pipefail
if [ "$(numactl --show 2>/dev/null | grep -c 'cpubind: 0 $')" = 0 ]; then exec numactl --cpunodebind=0 --membind=0 bash "$0" "$@"; fi
HERE=$(cd "$(dirname "$0")/.." && pwd); CTL=$1; ROUNDS=${2:-10}; DATE=${3:-$(date +%F)}
RES=results/ws-amd/radeon-queues-$DATE; cd "$HERE"; mkdir -p "$RES"
[ "$(nice)" = 0 ] || { echo "not at nice 0"; exit 1; }
RUN="python -m fnnbench.cli run --backend cudann --device gpu --dtype float --workload mnist-512-256 --batch 256 --epochs 3 --repeat 3 --warmup 1 --seed 1 --samples 60000"
P=$HOME/.local/opt/fnn-rocm
for k in $(seq 1 "$ROUNDS"); do
    out=$RES/round-$k; mkdir -p "$out"
    timeout -s KILL 240 distrobox enter fnn-rocm -- bash -c "export PATH=$P/venv/bin:\$PATH FNN_REF_CACHE=$HERE/.cache/ref OMP_NUM_THREADS=16; cd $HERE; $RUN --out $out" > "$out/run.log" 2>&1 &
    pid=$!; sleep 3.5
    echo "dump-$k" > "$CTL/want"
    for i in $(seq 1 50); do [ "$(cat "$CTL/have" 2>/dev/null)" = "dump-$k" ] && break; sleep 0.1; done
    wait $pid; echo $? > "$out/exit.txt"
done
echo stop > "$CTL/want"
echo "==== [$(date +%T)] done" >> "$RES/run.log"
