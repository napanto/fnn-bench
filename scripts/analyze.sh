#!/usr/bin/env bash
# Collect and plot the results of every machine into results/analysis/ (run inside an environment
# with fnnbench + matplotlib, e.g. the fnn-rocm distrobox venv):
#   distrobox enter fnn-rocm -- bash -lc 'export PATH=$HOME/.local/opt/fnn-rocm/venv/bin:$PATH; fnn-bench/scripts/analyze.sh'
set -euo pipefail
command -v fnnbench >/dev/null 2>&1 || export PATH=$HOME/.local/opt/fnn-rocm/venv/bin:$PATH   # the harness lives in the fnn-rocm venv when run from the host
HERE=$(cd "$(dirname "$0")/.." && pwd)
cd "$HERE"
# The passes the study reports; ANALYZE_ALL=1 also redoes the earlier passes (2026-09-03/04, measured
# with the per-launch profiler on), whose raw rows stay in results/ without an analysis.
REPORTED="ws-amd/2026-09-05 ws-nvidia/2026-09-05 hpc-a30/2026-09-17"
for m in ws-amd ws-nvidia hpc-a30; do
    for d in results/$m/*/; do
        date=$(basename "$d")
        case $date in peaks*|perf*|rocprof*|*smoke*|zluda*) continue ;; esac
        [ "${ANALYZE_ALL:-0}" = 1 ] || [[ " $REPORTED " == *" $m/$date "* ]] || continue
        [ -n "$(find "$d" -name '*.jsonl' -not -name superseded.jsonl | head -1)" ] || continue
        echo "== $m $date"
        fnnbench collect "$d" -o "results/analysis/$m-$date.csv"
        peaks=(); for pk in "results/$m/peaks" "$d"/peaks*; do [ -d "$pk" ] && peaks+=("$pk"); done
        { fnnbench plot "$d" "${peaks[@]}" -o "results/analysis/figures/$m-$date" 2>&1 | grep -c png | sed 's/$/ figures/'; } || true
    done
done
echo "done: results/analysis/*.csv, results/analysis/figures/<machine>-<date>/"
