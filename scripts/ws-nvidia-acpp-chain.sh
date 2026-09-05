#!/usr/bin/env bash
# After ws-nvidia's CPU fix-up: build fnn-acpp-cuda:dev (oneMath needs AdaptiveCpp_DIR), run the
# AdaptiveCpp rows (ws-nvidia-acpp.sh), the portability rows (ws-nvidia-portable-acpp.sh), fetch,
# analyze and write the ws-nvidia headline. Waits for the pid given as $2 (the fix-up wrapper).
set -uo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
AC=$(cd "$HERE/.." && pwd)
DATE=${1:-2026-09-05}
WAIT=${2:-}
PY=$HOME/.local/opt/fnn-rocm/venv/bin/python
log() { echo "==== [$(date +%T)] $*"; }
cd "$AC"
while [ -n "$WAIT" ] && kill -0 "$WAIT" 2>/dev/null; do sleep 60; done

log "ws-nvidia: fnn-acpp-cuda image (oneMath: AdaptiveCpp_DIR), acpp rows, portability"
rsync -a fnn-bench/containers/ ws-nvidia:$HOME/fnn/fnn-bench/containers/
ssh ws-nvidia 'cd $HOME/fnn/fnn-bench && podman build --memory=40g --build-arg JOBS=16 -f containers/fnn-acpp-cuda.Containerfile -t fnn-acpp-cuda:dev containers/ > $HOME/fnn/.logs/acpp-build.log 2>&1; tail -n 2 $HOME/fnn/.logs/acpp-build.log; podman image exists localhost/fnn-acpp-cuda:dev && echo IMAGE-OK || echo IMAGE-FAILED'
log "ws-nvidia: image step done"
DATE=$DATE bash fnn-bench/scripts/ws-nvidia-acpp.sh 2>&1 | grep -vE '^\s*$' | tail -40
log "ws-nvidia acpp rows done"
DATE=$DATE bash fnn-bench/scripts/ws-nvidia-portable-acpp.sh 2>&1 | tail -20
log "ws-nvidia portable-acpp done"
rsync -a ws-nvidia:$HOME/fnn/fnn-bench/results/ws-nvidia/ fnn-bench/results/ws-nvidia/
cd fnn-bench && bash scripts/analyze.sh 2>&1 | tail -3
$PY scripts/headline.py results/ws-nvidia/$DATE > analysis/headline-ws-nvidia-$DATE.md
log "ws-nvidia acpp chain complete"
