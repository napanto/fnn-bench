#!/usr/bin/env bash
# ZLUDA (CUDA-on-AMD, binary level): run the unmodified cudann CUDA wheel (built on ws-nvidia
# with nvcc, sm_61 + sm_80 SASS and PTX) on the RX 7900 XTX through ZLUDA's libcuda, next to
# the source-level HIPify port and syclnn. Deliverable: a does-it-run table + anecdotal speed.
#   distrobox enter fnn-rocm -- bash fnn-bench/scripts/zluda-crosscheck.sh
# Needs: fnn-bench/.wheels/ws-nvidia-cuda (rsync from ws-nvidia), the CUDA 12 runtime libraries the
# wheel links (libcudart.so.12, libcublas.so.12, libcublasLt.so.12) extracted from the
# fnn-cuda image into ~/.local/opt/cuda12-libs (done here with podman if missing).
set -uo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
P=$HOME/.local/opt
ZLUDA_URL=${ZLUDA_URL:-https://github.com/vosen/ZLUDA/releases/download/v7-preview.10/zluda-linux-9c8b43f2.tar.gz}
OUT=$HERE/results/ws-amd/zluda
mkdir -p "$OUT" "$P/zluda" "$P/cuda12-libs"
log() { echo "==== [$(date +%T)] $*"; }

if [ ! -e "$P/zluda/libcuda.so.1" ] && [ ! -e "$P/zluda/libcuda.so" ]; then
    log "ZLUDA download"
    curl -fsSL "$ZLUDA_URL" | tar -xz -C "$P/zluda" --strip-components=1 || curl -fsSL "$ZLUDA_URL" | tar -xz -C "$P/zluda"
fi
ls "$P/zluda" | head -20
if [ ! -e "$P/cuda12-libs/libcudart.so.12" ]; then
    log "CUDA 12 runtime libraries from the fnn-cuda image (host podman)"
    # run from the host: distrobox exposes the host's podman through distrobox-host-exec
    distrobox-host-exec podman run --rm -v "$P/cuda12-libs:/out" localhost/fnn-cuda:dev bash -c \
        'cp -L /usr/local/cuda-12.9/lib64/libcudart.so.12 /usr/local/cuda-12.9/lib64/libcublas.so.12 /usr/local/cuda-12.9/lib64/libcublasLt.so.12 /out/ 2>/dev/null; ls /out'
fi
WHEEL=$HERE/.wheels/ws-nvidia-cuda
[ -d "$WHEEL" ] || { echo "missing $WHEEL (rsync it from ws-nvidia:$HOME/fnn/fnn-bench/.wheels/ws-nvidia-cuda)"; exit 1; }
export PYTHONPATH=$WHEEL
export LD_LIBRARY_PATH=$P/zluda:$P/cuda12-libs:${LD_LIBRARY_PATH:-}
export ZLUDA_LOG=${ZLUDA_LOG:-}
PY=$P/fnn-rocm/venv/bin/python
{
echo "# ZLUDA cross-check, $(date -u +%FT%TZ)"
echo "ZLUDA: $(basename "$ZLUDA_URL"); wheel: $WHEEL; libcuda: $(readlink -f "$P/zluda/libcuda.so.1" 2>/dev/null || ls "$P/zluda"/libcuda* | head -1)"
echo
echo "## import / device enumeration"
$PY -c "import cudann; print(cudann.build_info()); print(cudann.devices())" 2>&1 | tail -5
echo
echo "## parity suite per option (does it run / does it compute the right thing)"
cd "$HERE/../cudann"
for o in "--dtype float" "--dtype double" "--option queue=in_order" "--option streams=4" "--option queue=graph" "--option memory=shared" "--option memory=host" "--option pinned_host=False" "--option blas=tiled --dtype float"; do
    printf "%-40s " "$o"; timeout 900 $PY -m pytest -q --device gpu -p no:cacheprovider $o 2>&1 | grep -E "passed|failed|rror" | head -1
done
echo
echo "## anecdotal speed (steady epoch, unprofiled): cudann-on-ZLUDA vs cudann/HIP vs syclnn/AdaptiveCpp"
cd "$HERE"
for wl in "monk 40" "cup 40" "mnist-512-256 256" "mnist-512-256 1024"; do set -- $wl
    printf "cudann/ZLUDA %-14s b=%-5s " $1 $2; timeout 1800 $PY -m fnnbench run --backend cudann --device gpu --workload $1 --batch $2 --dtype float --epochs 5 --repeat 3 --warmup 1 --out "$OUT/rows" --tag zluda 2>&1 | grep -oE "train epoch +[0-9.]+ ms.*samples/s" | head -1
done
} | tee "$OUT/README.md"
log "written $OUT/README.md"
