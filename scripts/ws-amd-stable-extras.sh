#!/usr/bin/env bash
# Worker of amd-profile-session.py. GPU profiling state is already active.
set -euo pipefail
B=$(cd "$(dirname "$0")/.." && pwd); ACTION=$1; DATE=$2
if [ "${FNN_EXTRA_STABLE_INSIDE:-0}" != 1 ]; then
    exec nice -n 19 distrobox enter fnn-rocm -- env FNN_EXTRA_STABLE_INSIDE=1 numactl --cpunodebind=0 --membind=0 nice -n 19 bash "$B/scripts/ws-amd-stable-extras.sh" "$ACTION" "$DATE"
fi
cd "$B"
P=$HOME/.local/opt/fnn-rocm;R=results/ws-amd/rocprof-stable-$DATE;OLD=results/ws-amd/rocprof-tiled-$DATE
TOOLS=results/ws-amd/extra-tools-$DATE
mkdir -p "$R"
export PATH=$P/venv/bin:$PATH PYTHONPATH=$B/bench:$B/testkit PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=16
export FNN_MACHINE_ID=ws-amd FNN_DATA_DIR=$B/$TOOLS/data FNN_REF_CACHE=$B/$TOOLS/ref
unset OMP_PROC_BIND OMP_PLACES
if [ "$ACTION" = lds ]; then
    for cfg in 'cudann hipcc venv tiled' 'syclnn adaptivecpp venv tiled' 'ompnn amdclang venv-amdclang tiled' 'cudann hipcc venv auto'; do
        read -r be label venv blas <<< "$cfg"
        export PATH=$P/$venv/bin:/opt/rocm/bin:/usr/local/bin:/usr/bin:/bin
        python scripts/rocprof-counters.py --backend "$be" --label "$label" --blas "$blas" --out "$R" --groups lds-instructions-base --timeout 60
    done
    exit 0
fi
if [ "$ACTION" = peak ]; then
    python -m fnnbench.cli peak --backend cudann --device gpu --dtype float --size 4096 --out "$R/peaks" > "$R/peak-4096.log" 2>&1
    exit 0
fi
if [ "$ACTION" = cache ]; then
    S=results/ws-amd/stream-cache-stable-$DATE;mkdir -p "$S"
    python scripts/extra_conditions.py "$S" --machine ws-amd --binding 'numactl --cpunodebind=0 --membind=0; GPU profile_standard'
    cp "results/ws-amd/stream-$DATE/versions.txt" "$S/versions.txt"
    printf '%s\n' "results/ws-amd/stream-$DATE/stream-probe 6291456 31 (72 MiB total; profile_standard)" > "$S/commands.txt"
    "results/ws-amd/stream-$DATE/stream-probe" 6291456 31 > "$S/stream.csv"
    for metric in FETCH_SIZE WRITE_SIZE; do
        cmd=(rocprofv3 --kernel-trace --pmc "$metric" --kernel-include-regex '.*stream_op<3>.*' --output-format csv -d "$S/$metric" -o run -- "results/ws-amd/stream-$DATE/stream-probe" 6291456 3)
        printf '%q ' "${cmd[@]}" >> "$S/commands.txt"; printf '\n' >> "$S/commands.txt"
        "${cmd[@]}" > "$S/$metric.log" 2>&1
    done
    python scripts/stream-table.py "$S"
    python scripts/stream-interface.py "$S"
    python scripts/extra-plots.py stream "$S"
    exit 0
fi
cp "$OLD/list-avail.txt" "$OLD/versions.txt" "$OLD/counter-units.json" "$OLD/counter_defs.yaml" "$R/"
python scripts/extra_conditions.py "$R" --machine ws-amd --binding 'numactl --cpunodebind=0 --membind=0; GPU profile_standard'
printf '%s\n' 'python3 scripts/amd-profile-session.py (ordinary DRM context; profile_standard; release in finally)' "worker phase: $ACTION" >> "$R/commands.txt"
if [ "$ACTION" = probe ]; then
    python scripts/rocprof-counters.py --backend cudann --label hipcc --blas tiled --out "$R" --groups lds-instructions valu-instructions memory-read --size mnist --timeout 45
    python scripts/rocprof-table.py "$R"
    exit 0
fi
for cfg in 'cudann hipcc venv tiled' 'syclnn adaptivecpp venv tiled' 'ompnn amdclang venv-amdclang tiled' 'cudann hipcc venv auto'; do
    read -r be label venv blas <<< "$cfg"
    export PATH=$P/$venv/bin:/opt/rocm/bin:/usr/local/bin:/usr/bin:/bin
    python scripts/rocprof-counters.py --backend "$be" --label "$label" --blas "$blas" --out "$R" --timeout 60
 done
export PATH=$P/venv/bin:/opt/rocm/bin:/usr/local/bin:/usr/bin:/bin
python scripts/rocprof-table.py "$R"
S=results/ws-amd/stream-stable-$DATE;mkdir -p "$S"
python scripts/extra_conditions.py "$S" --machine ws-amd --binding 'numactl --cpunodebind=0 --membind=0; GPU profile_standard'
cp "results/ws-amd/stream-$DATE/versions.txt" "$S/versions.txt"
printf '%s\n' "results/ws-amd/stream-$DATE/stream-probe (same binary; GPU profile_standard)" > "$S/commands.txt"
"results/ws-amd/stream-$DATE/stream-probe" > "$S/stream.csv"
python scripts/stream-table.py "$S"
python -m fnnbench.cli peak --backend cudann --device gpu --dtype float --size 8192 --out "$R/peaks" > "$R/peak.log" 2>&1
python scripts/extra-plots.py counters "$R"
python scripts/extra-plots.py stream "$S"
C=results/ws-amd/stream-cache-stable-$DATE
if [ -s "$C/calibration.json" ]; then
    python scripts/extra-plots.py roofline "$R" --stream "$C"
else
    echo 'Interface-matched roofline waits for the cache calibration phase.'
fi
