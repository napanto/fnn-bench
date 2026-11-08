#!/usr/bin/env bash
# ws-amd, RX 7900 XTX, HIP, MNIST batch 256: wall stacks of the submitting thread in fresh processes,
# to see whether the slow modes spend their extra time waiting for the GPU or working on the host.
#   scripts/ws-amd-radeon-wallprof.sh [rounds] [date]
set -uo pipefail
if [ "$(numactl --show 2>/dev/null | grep -c 'cpubind: 0 $')" = 0 ]; then exec numactl --cpunodebind=0 --membind=0 bash "$0" "$@"; fi
HERE=$(cd "$(dirname "$0")/.." && pwd); ROUNDS=${1:-8}; DATE=${2:-$(date +%F)}
RES=results/ws-amd/radeon-wallprof-$DATE; cd "$HERE"; mkdir -p "$RES"
[ "$(nice)" = 0 ] || { echo "not at nice 0"; exit 1; }
P=$HOME/.local/opt/fnn-rocm
distrobox enter fnn-rocm -- bash -c "cd $HERE && gcc -O2 -shared -fPIC scripts/wallprof/wallprof.c -o /tmp/wallprof-rocm.so" || exit 1
for k in $(seq 1 "$ROUNDS"); do
    out=$RES/mnist-r$k
    timeout -s KILL 150 distrobox enter fnn-rocm -- bash -c "export PATH=$P/venv/bin:\$PATH FNN_REF_CACHE=$HERE/.cache/ref OMP_NUM_THREADS=16 PYTHONPATH=$HERE/bench:$HERE/testkit; cd $HERE; python scripts/wallprof/wallprof_run.py --backend cudann --epochs 30 --lib /tmp/wallprof-rocm.so --out $out.dump && python scripts/wallprof/wallprof_report.py $out.dump" >> "$RES/run.log" 2>&1
done
echo "==== [$(date +%T)] done" >> "$RES/run.log"
