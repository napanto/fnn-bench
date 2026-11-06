#!/usr/bin/env bash
# ws-amd, RX 7900 XTX, launch-bound CUP b40 float on HIP: SCHED_FIFO 1 against the default policy, one fresh process per row.
#   scripts/ws-amd-gpu-fifo.sh [rounds] [date]
set -uo pipefail
if [ "$(numactl --show 2>/dev/null | grep -c 'cpubind: 0 $')" = 0 ]; then exec numactl --cpunodebind=0 --membind=0 bash "$0" "$@"; fi
HERE=$(cd "$(dirname "$0")/.." && pwd); ROUNDS=${1:-8}; DATE=${2:-$(date +%F)}
RES=results/ws-amd/cup-interference-$DATE/gpu; cd "$HERE"; mkdir -p "$RES"
[ "$(nice)" = 0 ] || { echo "not at nice 0"; exit 1; }
WL=${WL:-cup}; TAG=${TAG:-$WL}
case $WL in
cup) RUN="python -m fnnbench.cli run --backend cudann --device gpu --dtype float --workload cup --batch 40 --epochs 5 --repeat 5 --warmup 1 --seed 1 --samples 1000" ;;
mnist) RUN="python -m fnnbench.cli run --backend cudann --device gpu --dtype float --workload mnist-512-256 --batch 256 --epochs 3 --repeat 3 --warmup 1 --seed 1 --samples 60000" ;;
esac
P=$HOME/.local/opt/fnn-rocm
one() { local v=$1 k=$2; local out=$RES/$TAG/$v/round-$k; mkdir -p "$out"; local pre=""; case $v in fifo) pre="chrt -f 1 " ;; fifo51) pre="chrt -f 51 " ;; fifo91) pre="chrt -f 91 " ;; node1) pre="numactl --cpunodebind=1 --membind=0 " ;; slack) pre="python scripts/slack_exec.py " ;; pin1) pre="taskset -c 3 " ;; pin2) pre="taskset -c 3,4 " ;; esac
    { date +%T; grep -H '\*' /sys/class/drm/card*/device/pp_dpm_sclk /sys/class/drm/card*/device/pp_dpm_mclk 2>/dev/null; } > "$out/dpm-before.txt"
    timeout -s KILL 120 distrobox enter fnn-rocm -- bash -c "export PATH=$P/venv/bin:\$PATH FNN_REF_CACHE=$HERE/.cache/ref OMP_NUM_THREADS=16; cd $HERE; $pre$RUN --out $out" > "$out/run.log" 2>&1
    echo $? > "$out/exit.txt"; grep -H '\*' /sys/class/drm/card*/device/pp_dpm_sclk /sys/class/drm/card*/device/pp_dpm_mclk 2>/dev/null > "$out/dpm-after.txt"; }
VARS=${VARS:-"default fifo"}
for k in $(seq 1 "$ROUNDS"); do
    list=$VARS; [ $((k % 2)) = 0 ] && list=$(echo "$VARS" | tr ' ' '\n' | tac | tr '\n' ' ')
    for v in $list; do one "$v" "$k"; done
done
echo "==== [$(date +%T)] done"
