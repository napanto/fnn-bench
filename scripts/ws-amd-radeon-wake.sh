#!/usr/bin/env bash
# ws-amd, RX 7900 XTX, HIP MNIST b256 with library streams, default runtime settings: fresh processes whose
# python runs as v-mn-r<round> (a symlink in the venv), so that scripts/sched/wake3.bt (run as root meanwhile)
# records the wake-up latency of each of their threads. Submitting process on node 0, nice 0.
#   scripts/ws-amd-radeon-wake.sh [rounds] [date]
set -uo pipefail
if [ "$(numactl --show 2>/dev/null | grep -c 'cpubind: 0 $')" = 0 ]; then exec numactl --cpunodebind=0 --membind=0 bash "$0" "$@"; fi
HERE=$(cd "$(dirname "$0")/.." && pwd); ROUNDS=${1:-12}; DATE=${2:-$(date +%F)}
RES=results/ws-amd/radeon-wake-$DATE; cd "$HERE"; mkdir -p "$RES"
[ "$(nice)" = 0 ] || { echo "not at nice 0"; exit 1; }
P=$HOME/.local/opt/fnn-rocm
for k in $(seq 1 "$ROUNDS"); do
    out=$RES/round-$k; ls "$out"/*.jsonl >/dev/null 2>&1 && continue; mkdir -p "$out"; tag=v-mn-r$k
    ln -sf python "$P/venv/bin/$tag"
    timeout -s KILL 240 distrobox enter fnn-rocm -- bash -c "export PATH=$P/venv/bin:\$PATH FNN_REF_CACHE=$HERE/.cache/ref OMP_NUM_THREADS=16; cd $HERE; $tag -m fnnbench.cli run --backend cudann --device gpu --dtype float --workload mnist-512-256 --batch 256 --epochs 3 --repeat 3 --warmup 1 --seed 1 --samples 60000 --out $out" > "$out/run.log" 2>&1
    echo $? > "$out/exit.txt"; rm -f "$P/venv/bin/$tag"
done
echo "==== [$(date +%T)] done" >> "$RES/run.log"
