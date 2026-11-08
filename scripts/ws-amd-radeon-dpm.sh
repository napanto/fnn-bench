#!/usr/bin/env bash
# ws-amd, RX 7900 XTX, HIP MNIST b256 with library streams: do the two speeds depend on power management?
# Fresh processes alternate between power-management arms (VARS): high = power_dpm_force_performance_level
# high (every clock fixed at its top level), auto, compute = manual level with the COMPUTE power profile,
# nogfxoff = auto with GFXOFF disabled through debugfs. A root helper applies the level written to <ctl>/want and confirms it in <ctl>/have. During
# each process the current graphics, fabric (fclk) and SoC clock levels are logged every 50 ms.
#   scripts/ws-amd-radeon-dpm.sh <ctl-dir> [rounds] [date]
set -uo pipefail
if [ "$(numactl --show 2>/dev/null | grep -c 'cpubind: 0 $')" = 0 ]; then exec numactl --cpunodebind=0 --membind=0 bash "$0" "$@"; fi
HERE=$(cd "$(dirname "$0")/.." && pwd); CTL=$1; ROUNDS=${2:-10}; DATE=${3:-$(date +%F)}
RES=results/ws-amd/radeon-${TAG:-dpm}-$DATE; cd "$HERE"; mkdir -p "$RES"
[ "$(nice)" = 0 ] || { echo "not at nice 0"; exit 1; }
DEV=/sys/class/drm/card1/device
RUN="python -m fnnbench.cli run --backend cudann --device gpu --dtype float --workload mnist-512-256 --batch 256 --epochs 3 --repeat 3 --warmup 1 --seed 1 --samples 60000"
P=$HOME/.local/opt/fnn-rocm
level() { echo "$1" > "$CTL/want"; for i in $(seq 1 100); do [ "$(cat "$CTL/have" 2>/dev/null)" = "$1" ] && return 0; sleep 0.1; done; echo "level $1 not confirmed"; exit 1; }
for k in $(seq 1 "$ROUNDS"); do
    list=${VARS:-"high auto"}; [ $((k % 2)) = 0 ] && list=$(echo "$list" | tr ' ' '\n' | tac | tr '\n' ' ')
    for v in $list; do
        level "$v"; sleep 2
        out=$RES/$v/round-$k; mkdir -p "$out"
        ( while :; do printf '%s %s %s %s %s\n' "$(date +%s.%N)" "$(cat $DEV/power_dpm_force_performance_level)" "$(grep -h '\*' $DEV/pp_dpm_sclk | head -1 | tr -d ' ')" "$(grep -h '\*' $DEV/pp_dpm_fclk | head -1 | tr -d ' ')" "$(grep -h '\*' $DEV/pp_dpm_socclk | head -1 | tr -d ' ')"; sleep 0.05; done ) > "$out/clocks.log" 2>/dev/null & lp=$!
        timeout -s KILL 240 distrobox enter fnn-rocm -- bash -c "export PATH=$P/venv/bin:\$PATH FNN_REF_CACHE=$HERE/.cache/ref OMP_NUM_THREADS=16; cd $HERE; $RUN --out $out" > "$out/run.log" 2>&1
        echo $? > "$out/exit.txt"; kill $lp 2>/dev/null
    done
done
echo stop > "$CTL/want"
echo "==== [$(date +%T)] done" >> "$RES/run.log"
