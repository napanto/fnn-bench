#!/usr/bin/env bash
# Second local pass on ws-amd (run after the first matrix-ws-amd.sh instance exits):
#  1. rebuild the venv wheels at the current library revisions (tiled kernels with the
#     transposed A tile), 2. GPU parity suites with blas=tiled, 3. supersede the rows
#     measured with the older harness, 4. matrix-ws-amd.sh again (done rows are skipped,
#     E5/E7/tiled peaks are new), 5. the perf cross-check on the idle CPU.
set -uo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
AC=$(cd "$HERE/.." && pwd)
DATE=${DATE:-2026-09-03}
P=$HOME/.local/opt/fnn-rocm
log() { echo "==== [$(date +%T)] $*"; }

log "rebuild venv wheels (syclnn acpp, cudann hip, ompnn amdclang++/gcc-14 amdgcn)"
distrobox enter fnn-rocm -- bash -lc "
    P=$P; set -o pipefail
    export PATH=\$P/acpp/bin:\$P/venv/bin:\$PATH CMAKE_BUILD_PARALLEL_LEVEL=8
    cd $AC/syclnn && CXX=acpp CC=clang-18 SYCLNN_SYCL_IMPL=adaptivecpp SYCLNN_ONEMATH_ROOT=\$P/onemath CMAKE_PREFIX_PATH=\$P/acpp:\$P/onemath pip install -q --no-deps --force-reinstall --config-settings=build-dir=/tmp/syclnn-build-acpp . && python -c 'import syclnn; print(\"syclnn\", syclnn.build_info()[\"git_sha\"])'
    cd $AC/cudann && CXX=hipcc CC=hipcc CUDANN_HIP=ON CUDANN_HIP_ARCHS=gfx1100 pip install -q --no-deps --force-reinstall --config-settings=build-dir=/tmp/cudann-build-hip . && python -c 'import cudann; print(\"cudann\", cudann.build_info()[\"git_sha\"])'
    cd $AC/ompnn
    for cfg in 'amdclang amdclang++ venv-amdclang' 'gcc14amd g++-14 venv-gcc14amd'; do set -- \$cfg
        PATH=\$P/\$3/bin:\$PATH CXX=\$2 OMPNN_TARGET=amd OMPNN_OFFLOAD_ARCH=gfx1100 OMPNN_BLAS=openblas OMPNN_BLAS_ROOT=\$P/openblas-openmp pip install -q --no-deps --force-reinstall --config-settings=build-dir=/tmp/ompnn-build-\$1-venv . && \$P/\$3/bin/python -c 'import ompnn; print(\"ompnn\", ompnn.build_info()[\"git_sha\"], ompnn.build_info()[\"compiler\"][:30])'
    done
"

log "GPU parity suites, blas=tiled (results also in results/ws-amd/$DATE/tiled-parity.txt)"
distrobox enter fnn-rocm -- bash -lc "
    P=$P; out=$HERE/results/ws-amd/$DATE/tiled-parity.txt; : > \$out
    export PATH=\$P/venv/bin:\$PATH OMP_NUM_THREADS=8
    cd $AC/syclnn; for a in 'gpu --blas tiled --dtype double' 'gpu --blas tiled --dtype float' 'cpu --blas tiled --dtype float'; do printf 'syclnn acpp %-34s ' \"\$a\" | tee -a \$out; python -m pytest -q --device \$a -p no:cacheprovider 2>&1 | grep -E 'passed|failed|rror' | head -1 | tee -a \$out; done
    cd $AC/cudann; for a in '--option blas=tiled --dtype double' '--option blas=tiled --dtype float' '--option blas=tiled --option queue=graph'; do printf 'cudann hip gpu %-40s ' \"\$a\" | tee -a \$out; python -m pytest -q --device gpu -p no:cacheprovider \$a 2>&1 | grep -E 'passed|failed|rror' | head -1 | tee -a \$out; done
    cd $AC/ompnn; for v in venv-amdclang venv-gcc14amd; do for a in '--option blas=tiled --dtype double' '--option blas=tiled --dtype float'; do printf 'ompnn %-14s gpu %-34s ' \$v \"\$a\" | tee -a \$out; \$P/\$v/bin/python -m pytest -q --device gpu -p no:cacheprovider \$a 2>&1 | grep -E 'passed|failed|rror' | head -1 | tee -a \$out; done; done
" 2>&1 | grep -v 'AdaptiveCpp Warning'

log "supersede rows measured with the older harness"
python3 "$HERE/scripts/supersede-stale.py" "$HERE/results/ws-amd/$DATE"

log "second matrix pass"
bash "$HERE/scripts/matrix-ws-amd.sh" "$DATE" rocm-gpu omp-gpu sycl-cpu sycl-generic omp-cpu

log "perf cross-check (CPU idle)"
bash "$HERE/scripts/perf-crosscheck.sh" mnist-512-256 256 || true
log "pass2 done"
