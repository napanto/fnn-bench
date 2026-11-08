#!/usr/bin/env bash
# ws-amd, RX 7900 XTX, HIP MNIST b256 with library streams: is the per-process mode set by the address
# layout or by interrupt-driven signal waits? Fresh processes, alternated: default, noaslr (setarch -R:
# the same address layout in every process), nointr (HSA_ENABLE_INTERRUPT=0: the runtime polls signals),
# aw100/aw1000 (ROC_ACTIVE_WAIT_TIMEOUT=100 or 1000: the HIP runtime waits actively that many microseconds
# before blocking on an interrupt), irqccx (taskset 4-7,20-23: the core complex of CPU 6, which takes every
# amdgpu interrupt), node1 (taskset 8-15,24-31: the other die, isolated nohz_full cores), near12 (taskset
# 12-15,28-31: the core complex of CPU 12, used while a root helper moves the interrupt there).
# Submitting process on node 0, nice 0.
#   scripts/ws-amd-radeon-modes2.sh [rounds] [date]
set -uo pipefail
if [ "$(numactl --show 2>/dev/null | grep -c 'cpubind: 0 $')" = 0 ]; then exec numactl --cpunodebind=0 --membind=0 bash "$0" "$@"; fi
HERE=$(cd "$(dirname "$0")/.." && pwd); ROUNDS=${1:-8}; DATE=${2:-$(date +%F)}
RES=results/ws-amd/radeon-modes2-$DATE${SUFFIX:-}; cd "$HERE"; mkdir -p "$RES"
[ "$(nice)" = 0 ] || { echo "not at nice 0"; exit 1; }
RUN="python -m fnnbench.cli run --backend cudann --device gpu --dtype float --workload mnist-512-256 --batch 256 --epochs 3 --repeat 3 --warmup 1 --seed 1 --samples 60000"
P=$HOME/.local/opt/fnn-rocm
for k in $(seq 1 "$ROUNDS"); do
    list=${VARS:-"default noaslr nointr"}; [ $((k % 2)) = 0 ] && list=$(echo "$list" | tr ' ' '\n' | tac | tr '\n' ' ')
    for v in $list; do
        out=$RES/$v/round-$k; ls "$out"/*.jsonl >/dev/null 2>&1 && continue; mkdir -p "$out"
        pre=""; case $v in noaslr) pre="setarch $(uname -m) -R " ;; nointr) pre="env HSA_ENABLE_INTERRUPT=0 " ;; aw100) pre="env ROC_ACTIVE_WAIT_TIMEOUT=100 " ;; aw1000) pre="env ROC_ACTIVE_WAIT_TIMEOUT=1000 " ;; irqccx) pre="taskset -c 4-7,20-23 " ;; node1) pre="taskset -c 8-15,24-31 " ;; near12) pre="taskset -c 12-15,28-31 " ;; esac
        timeout -s KILL 240 distrobox enter fnn-rocm -- bash -c "export PATH=$P/venv/bin:\$PATH FNN_REF_CACHE=$HERE/.cache/ref OMP_NUM_THREADS=16; cd $HERE; $pre$RUN --out $out" > "$out/run.log" 2>&1
        echo $? > "$out/exit.txt"
    done
done
echo "==== [$(date +%T)] done" >> "$RES/run.log"
