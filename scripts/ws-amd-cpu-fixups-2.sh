#!/usr/bin/env bash
# Second CPU fix-up on ws-amd (2026-09-05): the DPC++ OpenCL CPU device on the 16 physical
# cores (DPCPP_CPU_NUM_CUS=16) like every other CPU runtime of the study; unset it used the
# 32 hardware threads, 20 % slower at mnist-512-256 b256 (E1 re-measured: 0.79 s at 16,
# 0.94 s at 32). Re-measures the whole DPC++ CPU block (E1 BLAS backends, E2, E6, W4, E7,
# E5 inference/breakdown, CPU peaks) and the DPC++ portability rows. Old rows -> superseded.jsonl.
# Waits for ws-amd-cpu-fixups.sh to finish first.
set -uo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
AC=$(cd "$HERE/.." && pwd)
DATE=${1:-2026-09-05}
RES=results/ws-amd/$DATE
PY=$HOME/.local/opt/fnn-rocm/venv/bin/python
log() { echo "==== [$(date +%T)] $*"; }
cd "$HERE"

while pgrep -f 'ws-amd-cpu-fixup[s]\.sh 2026' >/dev/null; do sleep 60; done

log "cpu fix-ups 2: supersede the DPC++ CPU rows (32 compute units)"
for d in sycl-cpu sycl-cpu-ablations sycl-cpu-generic sycl-cpu-w4 e7-tiled-cpu-dpcpp e5-infer-cpu-dpcpp e5-breakdown-cpu-dpcpp portable-dpcpp; do
    [ -d $RES/$d ] && $PY scripts/supersede-stale.py $RES/$d --all
done
$PY - <<EOF
import json
from pathlib import Path
p = Path("$RES/peaks/peak.jsonl"); rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
cpu = [r for r in rows if (r.get("device") or {}).get("type") == "cpu" and r.get("backend") == "syclnn"]
keep = [r for r in rows if r not in cpu]
if cpu:
    with (p.parent / "superseded.jsonl").open("a") as fh:
        for r in cpu: fh.write(json.dumps(r) + "\n")
    p.write_text("".join(json.dumps(r) + "\n" for r in keep))
print(f"peaks: kept {len(keep)}, superseded {len(cpu)} syclnn CPU peak rows")
EOF

log "1. DPC++ CPU blocks at 16 compute units (MKLCPU/Netlib, then the generic SYCL BLAS image)"
bash scripts/matrix-ws-amd.sh $DATE sycl-cpu sycl-generic

log "2. portability rows, DPC++ wheel built on ws-nvidia, on ws-amd's CPU (16 compute units)"
podman run --rm --memory=20g --security-opt label=disable -v "$AC/syclnn:/work/syclnn" -v "$HERE:/work/fnn-bench" -w /work/fnn-bench localhost/fnn-sycl:dev bash -c '
    pip install -q --no-deps -e /work/fnn-bench/testkit -e /work/fnn-bench/bench; export PYTHONPATH=/work/fnn-bench/.wheels/ws-nvidia-sycl FNN_REF_CACHE=/work/fnn-bench/.cache/ref OMP_NUM_THREADS=16 DPCPP_CPU_NUM_CUS=16 OMP_PROC_BIND=close OMP_PLACES=cores
    sha256sum /work/fnn-bench/.wheels/ws-nvidia-sycl/syclnn/_syclnn*.so | tee /work/fnn-bench/'"$RES"'/portable-dpcpp.SHA256; python -c "import syclnn; print([(d[\"name\"], d[\"compute_units\"]) for d in syclnn.devices()])"
    for b in tiled auto; do for wl in "monk 40" "cup 40" "mnist-512-256 256"; do set -- $wl; fnnbench run --backend syclnn --device cpu --workload $1 --batch $2 --dtype float --epochs 5 --repeat 3 --warmup 1 --option blas=$b --out '"$RES"'/portable-dpcpp 2>&1 | grep -E "epoch|rror" | tail -1 | cut -c1-150; done; done
'

log "3. supersede older duplicates, analyze, headline"
$PY scripts/supersede-stale.py $RES --dupes
bash scripts/analyze.sh 2>&1 | tail -3
$PY scripts/headline.py $RES > analysis/headline-ws-amd-$DATE.md && head -20 analysis/headline-ws-amd-$DATE.md
log "cpu fix-ups 2 done"
