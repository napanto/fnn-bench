#!/usr/bin/env bash
# ws-amd, RX 7900 XTX, HIP MNIST b256: are the fast and slow modes in the kernels or between them?
# Fresh processes, alternated: default (library streams), inorder (one stream), trace (default under
# rocprofv3 --kernel-trace: duration, queue and timestamps of every dispatch). Same conditions as
# ws-amd-gpu-fifo.sh: submitting process on node 0, nice 0.
#   scripts/ws-amd-radeon-modes.sh [rounds] [date]
set -uo pipefail
if [ "$(numactl --show 2>/dev/null | grep -c 'cpubind: 0 $')" = 0 ]; then exec numactl --cpunodebind=0 --membind=0 bash "$0" "$@"; fi
HERE=$(cd "$(dirname "$0")/.." && pwd); ROUNDS=${1:-6}; DATE=${2:-$(date +%F)}
RES=results/ws-amd/radeon-modes-$DATE; TRACES=$HERE/../.logs/results-private/radeon-modes-$DATE; cd "$HERE"; mkdir -p "$RES" "$TRACES"
[ "$(nice)" = 0 ] || { echo "not at nice 0"; exit 1; }
RUN="python -m fnnbench.cli run --backend cudann --device gpu --dtype float --workload mnist-512-256 --batch 256 --epochs 3 --repeat 3 --warmup 1 --seed 1 --samples 60000"
P=$HOME/.local/opt/fnn-rocm
one() { local v=$1 k=$2; local out=$RES/$v/round-$k; ls "$out"/*.jsonl >/dev/null 2>&1 && return 0; mkdir -p "$out"
    local pre="" opt=""
    case $v in inorder) opt="--option queue=in_order" ;; trace) pre="rocprofv3 --kernel-trace --output-format csv -d $TRACES/round-$k -o run -- " ;; esac
    { date +%T; grep -H '\*' /sys/class/drm/card*/device/pp_dpm_sclk /sys/class/drm/card*/device/pp_dpm_mclk 2>/dev/null; } > "$out/dpm-before.txt"
    timeout -s KILL 240 distrobox enter fnn-rocm -- bash -c "export PATH=$P/venv/bin:\$PATH FNN_REF_CACHE=$HERE/.cache/ref OMP_NUM_THREADS=16; cd $HERE; $pre$RUN $opt --out $out" > "$out/run.log" 2>&1
    echo $? > "$out/exit.txt"; }
for k in $(seq 1 "$ROUNDS"); do
    list="default inorder trace"; [ $((k % 2)) = 0 ] && list="trace inorder default"
    for v in $list; do one "$v" "$k"; done
done
echo "==== [$(date +%T)] done" >> "$RES/run.log"
