#!/usr/bin/env bash
# Build and push the AMD variants of the library images on the fnn-rocm base:
#   ghcr.io/napanto/syclnn:acpp-rocm   AdaptiveCpp (generic SSCP), oneMath rocBLAS + Netlib
#   ghcr.io/napanto/cudann:hip         the HIPify build, gfx1100
#   ghcr.io/napanto/ompnn:rocm         amdclang++ and gcc-14 amdgcn offload (/opt/venvs/amdclang, /opt/venvs/gcc14amd)
# GitHub's runners cannot build these (the base is 25 GB), so this runs on a workstation with podman and a
# registry login; no GPU is needed at build time (the gcc amdgcn import check is skipped without /dev/kfd).
#   scripts/publish-amd-variants.sh [git ref of the library repos, default main]
set -euo pipefail
REF=${1:-main}
BASE=${BASE:-ghcr.io/napanto/fnn-rocm:latest}
DATE=$(date -u +%F)
W=$(mktemp -d)
trap 'rm -rf "$W"' EXIT
log() { echo "==== [$(date +%T)] $*"; }
for lib in syclnn cudann ompnn; do git clone -q --depth 1 -b "$REF" "https://github.com/napanto/$lib" "$W/$lib"; done
build() {
    local lib=$1 tag=$2; shift 2
    log "build $lib:$tag"
    podman build --memory=40g --build-arg BASE="$BASE" --build-arg IMAGE_NAME="ghcr.io/napanto/$lib:$tag" \
        --build-arg IMAGE_BUILT="$DATE" "$@" -t "ghcr.io/napanto/$lib:$tag" -f "$W/$lib/Containerfile" "$W/$lib"
    log "push $lib:$tag"
    podman push "ghcr.io/napanto/$lib:$tag"
    log "$lib:$tag done: $(podman image inspect "ghcr.io/napanto/$lib:$tag" --format '{{.Digest}}')"
}
build syclnn acpp-rocm --build-arg BUILD_CC=clang-18 --build-arg BUILD_CXX=acpp --build-arg ONEMATH_ROOT=/opt/fnn-rocm/onemath --build-arg CT_BACKENDS=netlib
build cudann hip --build-arg BUILD_CXX=hipcc --build-arg HIP=ON --build-arg HIP_ARCHS=gfx1100
build ompnn rocm --build-arg BUILDS="amdclang amdclang++ amd gfx1100|gcc14amd g++-14 amd gfx1100" --build-arg BLAS_ROOT=/opt/fnn-rocm/openblas-openmp
log "AMD variants published"
