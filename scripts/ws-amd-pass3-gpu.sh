#!/usr/bin/env bash
# Third pass on ws-amd, GPU/OpenMP half (run after the CPU phases): rebuild the venv wheels at
# the current library revisions, run the GPU, OpenMP-GPU and OpenMP-host phases, the perf
# cross-check, then collect and plot both machines.
set -uo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
AC=$(cd "$HERE/.." && pwd)
DATE=${DATE:-2026-09-05}
P=$HOME/.local/opt/fnn-rocm
log() { echo "==== [$(date +%T)] $*"; }

log "rebuild venv wheels (syclnn acpp, cudann hip, ompnn amdclang++/gcc-14 amdgcn)"
distrobox enter fnn-rocm -- bash -lc "
    P=$P; export PATH=\$P/acpp/bin:\$P/venv/bin:\$PATH CMAKE_BUILD_PARALLEL_LEVEL=8
    cd $AC/syclnn && CXX=acpp CC=clang-18 SYCLNN_SYCL_IMPL=adaptivecpp SYCLNN_ONEMATH_ROOT=\$P/onemath CMAKE_PREFIX_PATH=\$P/acpp:\$P/onemath pip install -q --no-deps --force-reinstall --config-settings=build-dir=/tmp/syclnn-build-acpp . && python -c 'import syclnn; print(\"syclnn\", syclnn.build_info()[\"git_sha\"])'
    cd $AC/cudann && CXX=hipcc CC=hipcc CUDANN_HIP=ON CUDANN_HIP_ARCHS=gfx1100 pip install -q --no-deps --force-reinstall --config-settings=build-dir=/tmp/cudann-build-hip . && python -c 'import cudann; print(\"cudann\", cudann.build_info()[\"git_sha\"])'
    cd $AC/ompnn
    for cfg in 'amdclang amdclang++ venv-amdclang' 'gcc14amd g++-14 venv-gcc14amd'; do set -- \$cfg
        PATH=\$P/\$3/bin:\$PATH CXX=\$2 OMPNN_TARGET=amd OMPNN_OFFLOAD_ARCH=gfx1100 OMPNN_BLAS=openblas OMPNN_BLAS_ROOT=\$P/openblas-openmp pip install -q --no-deps --force-reinstall --config-settings=build-dir=/tmp/ompnn-build-\$1-venv . && \$P/\$3/bin/python -c 'import ompnn; print(\"ompnn\", ompnn.build_info()[\"git_sha\"], ompnn.build_info()[\"compiler\"][:30])'
    done
" 2>&1 | grep -v 'AdaptiveCpp Warning'

log "parity of the rebuilt wheels (default + sync_ops) on the RX 7900 XTX"
distrobox enter fnn-rocm -- bash -lc "
    P=$P; export PATH=\$P/venv/bin:\$PATH OMP_NUM_THREADS=8
    cd $AC/syclnn; for a in '--dtype float' '--option sync_ops=True --dtype float'; do printf 'syclnn acpp gpu %-34s ' \"\$a\"; python -m pytest -q --device gpu -p no:cacheprovider \$a 2>&1 | grep -E 'passed|failed' | head -1; done
    cd $AC/cudann; for a in '--dtype float' '--option sync_ops=True --dtype float'; do printf 'cudann hip gpu %-34s ' \"\$a\"; python -m pytest -q --device gpu -p no:cacheprovider \$a 2>&1 | grep -E 'passed|failed' | head -1; done
    cd $AC/ompnn; for v in venv-amdclang venv-gcc14amd; do printf 'ompnn %-14s gpu ' \$v; \$P/\$v/bin/python -m pytest -q --device gpu --dtype float -p no:cacheprovider 2>&1 | grep -E 'passed|failed' | head -1; done
" 2>&1 | grep -v 'AdaptiveCpp Warning'

log "matrix: rocm-gpu omp-gpu omp-cpu"
bash "$HERE/scripts/matrix-ws-amd.sh" "$DATE" rocm-gpu omp-gpu omp-cpu

log "perf cross-check (CPU idle)"
rm -rf "$HERE/results/ws-amd/perf-$DATE"
OUTDIR="$HERE/results/ws-amd/perf-$DATE" bash "$HERE/scripts/perf-crosscheck.sh" mnist-512-256 256 || echo "perf failed"

log "analyze"
distrobox enter fnn-rocm -- bash -lc "export PATH=$P/venv/bin:\$PATH; bash $HERE/scripts/analyze.sh" 2>&1 | grep -vE 'Warning' | tail -8
log "pass3 ws-amd GPU half done"
