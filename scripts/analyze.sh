#!/usr/bin/env bash
# Collect and plot the results of both machines into analysis/ (run inside an environment
# with fnnbench + matplotlib, e.g. the fnn-rocm distrobox venv):
#   distrobox enter fnn-rocm -- bash -lc 'export PATH=$HOME/.local/opt/fnn-rocm/venv/bin:$PATH; fnn-bench/scripts/analyze.sh'
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
cd "$HERE"
for m in ws-amd ws-nvidia; do
    for d in results/$m/*/; do
        date=$(basename "$d")
        case $date in peaks*|perf*|rocprof*|*smoke*) continue ;; esac
        [ -n "$(find "$d" -name '*.jsonl' -not -name superseded.jsonl | head -1)" ] || continue
        echo "== $m $date"
        fnnbench collect "$d" -o "analysis/$m-$date.csv"
        peaks=(); for pk in "results/$m/peaks" "$d"/peaks*; do [ -d "$pk" ] && peaks+=("$pk"); done
        fnnbench plot "$d" "${peaks[@]}" -o "analysis/figures/$m-$date" 2>&1 | grep -c png | sed 's/$/ figures/'
    done
done
echo "done: analysis/*.csv, analysis/figures/<machine>-<date>/"
