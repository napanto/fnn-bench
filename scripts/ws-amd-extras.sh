#!/usr/bin/env bash
# Workstation-only counters and HIP bandwidth. No pass row or host setting changes.
# Invoke after the local measurement gate clears:
#   nice -n 19 bash scripts/ws-amd-extras.sh check|counters|stream [YYYY-MM-DD]
set -euo pipefail
B=$(cd "$(dirname "$0")/.." && pwd)
ACTION=${1:-check}; DATE=${2:-$(date +%F)}
[[ $DATE =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]] || exit 2
if [ "${FNN_EXTRA_INSIDE:-0}" != 1 ]; then
    if pgrep -f '[w]s-amd-cpu-numa' >/dev/null; then
        echo 'WAIT: ws-amd CPU remeasurement is active'; exit 75
    fi
    if pgrep -x fnnbench >/dev/null; then
        echo 'WAIT: a local fnnbench process is active'; exit 75
    fi
    [ "$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq)" = 2800000 ] || { echo 'CPU cap changed; no experiment started'; exit 1; }
    [ "$(cat /sys/devices/system/cpu/cpufreq/boost)" = 0 ] || { echo 'Boost is not off; no experiment started'; exit 1; }
    exec nice -n 19 distrobox enter fnn-rocm -- env FNN_EXTRA_INSIDE=1 numactl --cpunodebind=0 --membind=0 nice -n 19 bash "$B/scripts/ws-amd-extras.sh" "$ACTION" "$DATE"
fi
cd "$B"
P=$HOME/.local/opt/fnn-rocm
TOOLS=results/ws-amd/extra-tools-$DATE
mkdir -p "$TOOLS/data" "$TOOLS/ref"
export PATH=$P/venv/bin:$PATH PYTHONPATH=$B/bench:$B/testkit PYTHONDONTWRITEBYTECODE=1
export FNN_DATA_DIR=$B/$TOOLS/data FNN_REF_CACHE=$B/$TOOLS/ref OMP_NUM_THREADS=16
unset OMP_PROC_BIND OMP_PLACES
R=results/ws-amd/rocprof-tiled-$DATE
mkdir -p "$R"
case "$ACTION" in
check)
    rocprofv3 --list-avail > "$R/list-avail.txt" 2>&1
    { rocprofv3 --version; hipcc --version; python --version; uname -r; } > "$R/versions.txt" 2>&1
    python scripts/extra_conditions.py "$R" --machine ws-amd --binding 'numactl --cpunodebind=0 --membind=0'
    find /opt/rocm -name counter_defs.yaml -o -name 'lib*roctx*.so*' > "$R/sdk-files.txt"
    printf '%s\n' 'nice -n 19 bash scripts/ws-amd-extras.sh check' 'rocprofv3 --list-avail' > "$R/commands.txt"
    ;;
counters)
    [ -s "$R/list-avail.txt" ] || { echo 'Run check and inspect available counters first'; exit 1; }
    for cfg in 'cudann hipcc venv tiled' 'syclnn adaptivecpp venv tiled' 'ompnn amdclang venv-amdclang tiled' 'cudann hipcc venv auto'; do
        read -r be label venv blas <<< "$cfg"
        export PATH=$P/$venv/bin:/opt/rocm/bin:/usr/local/bin:/usr/bin:/bin
        cmd=(python scripts/rocprof-counters.py --backend "$be" --label "$label" --blas "$blas" --out "$R")
        printf '%q ' "${cmd[@]}" >> "$R/commands.txt"; printf '\n' >> "$R/commands.txt"
        "${cmd[@]}" || echo "$label/$blas: failed; details retained"
    done
    python scripts/rocprof-table.py "$R"
    ;;
stream)
    R=results/ws-amd/stream-$DATE; mkdir -p "$R"
    python scripts/extra_conditions.py "$R" --machine ws-amd --binding 'numactl --cpunodebind=0 --membind=0'
    { hipcc --version; python --version; uname -r; } > "$R/versions.txt" 2>&1
    printf '%s\n' 'hipcc -O3 -std=c++17 --offload-arch=gfx1100 -x hip scripts/stream-probe.cu -o stream-probe' './stream-probe (three 128 MiB arrays, 31 repetitions)' > "$R/commands.txt"
    hipcc -O3 -std=c++17 --offload-arch=gfx1100 -x hip scripts/stream-probe.cu -o "$R/stream-probe"
    "$R/stream-probe" > "$R/stream.csv"
    python scripts/stream-table.py "$R"
    ;;
*) echo 'expected check, counters or stream'; exit 2 ;;
esac
