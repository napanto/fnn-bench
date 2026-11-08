#!/usr/bin/env bash
# ws-amd, RX 7900 XTX: the cost of handing a dependent kernel to another HIP stream, untraced, in fresh
# processes (launch-probe/probe7_hip.cpp), alternated with fresh HIP MNIST b256 processes (library streams)
# so that probe and network can be compared in time. Submitting process on node 0, nice 0.
#   scripts/ws-amd-radeon-handover.sh [rounds] [date]
set -uo pipefail
if [ "$(numactl --show 2>/dev/null | grep -c 'cpubind: 0 $')" = 0 ]; then exec numactl --cpunodebind=0 --membind=0 bash "$0" "$@"; fi
HERE=$(cd "$(dirname "$0")/.." && pwd); ROUNDS=${1:-10}; DATE=${2:-$(date +%F)}
RES=results/ws-amd/radeon-handover-$DATE; cd "$HERE"; mkdir -p "$RES"
[ "$(nice)" = 0 ] || { echo "not at nice 0"; exit 1; }
[ -x "$RES/probe7-hip" ] || { echo "build $RES/probe7-hip first (hipcc -O3 --offload-arch=gfx1100)"; exit 1; }
RUN="python -m fnnbench.cli run --backend cudann --device gpu --dtype float --workload mnist-512-256 --batch 256 --epochs 3 --repeat 3 --warmup 1 --seed 1 --samples 60000"
P=$HOME/.local/opt/fnn-rocm
for k in $(seq 1 "$ROUNDS"); do
    { date +%s.%N; timeout -s KILL 120 distrobox enter fnn-rocm -- bash -c "cd $HERE; PROBE_R=5 PROBE_N=5000 $RES/probe7-hip"; date +%s.%N; } > "$RES/probe-r$k.csv" 2> "$RES/probe-r$k.err"
    out=$RES/mnist/round-$k; mkdir -p "$out"; date +%s.%N > "$out/start.txt"
    timeout -s KILL 240 distrobox enter fnn-rocm -- bash -c "export PATH=$P/venv/bin:\$PATH FNN_REF_CACHE=$HERE/.cache/ref OMP_NUM_THREADS=16; cd $HERE; $RUN --out $out" > "$out/run.log" 2>&1
    echo $? > "$out/exit.txt"
done
echo "==== [$(date +%T)] done" >> "$RES/run.log"
