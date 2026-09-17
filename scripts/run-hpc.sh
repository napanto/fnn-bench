#!/usr/bin/env bash
# hpc-a30 driver: the department SLURM cluster (front-end + node09: 2x AMD EPYC 7301, NVIDIA A30),
# run from a workstation with ssh access to the front-end (HPC_HOST). The cluster side is
# scripts/hpc-blocks.sh (sandbox images + unprivileged apptainer, docs/toolchains.md).
#
#   run-hpc.sh sync            # rsync the four repos + the reference cache to the cluster
#   run-hpc.sh build           # wheels on the front-end (compile only) + the nvc++ wheels on node09
#   run-hpc.sh tests           # parity suites: 5 chained 30-minute jobs -> results/hpc-a30/<date>/parity/
#   run-hpc.sh matrix [blocks] # the experiment matrix: N chained 30-minute jobs per block (sweeps are
#                              # resumable: each job continues where the previous one was cut off),
#                              # then the block's extras (peaks, nsys traces) in one more job
#   run-hpc.sh status          # pending rows per block (dry run, on the front-end) + the queue
#   run-hpc.sh fetch           # copy results/hpc-a30/ back
#   blocks: sycl-gpu cuda omp-gpu acpp-gpu sycl-cpu sycl-generic omp-cpu acpp-cpu
set -uo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
AC=$(cd "$HERE/.." && pwd)
HOST=${HPC_HOST:-hpc-a30}
REMOTE=${HPC_DIR:-fnn}                  # relative to the cluster home
DATE=${DATE:-$(date +%F)}
BLK="$REMOTE/fnn-bench/scripts/hpc-blocks.sh"
GPUQ="-p gpu-excl -w node09"
# 30-minute jobs per block: generous upper bounds from the workstation spans; unused jobs exit in a
# minute (every row already measured)
declare -A NJOBS=([sycl-gpu]=10 [cuda]=6 [omp-gpu]=8 [acpp-gpu]=4 [sycl-cpu]=8 [sycl-generic]=6 [omp-cpu]=5 [acpp-cpu]=2)
ALL="sycl-gpu cuda omp-gpu acpp-gpu sycl-cpu sycl-generic omp-cpu acpp-cpu"

sync() {
    ssh "$HOST" "mkdir -p $REMOTE logs/slurm"
    for r in syclnn cudann ompnn fnn-bench; do
        rsync -a --delete --exclude build --exclude .venv --exclude .wheels --exclude pylib --exclude __pycache__ --exclude results \
              --exclude .cache --exclude analysis --exclude .rocprofv3 "$AC/$r/" "$HOST:$REMOTE/$r/"
    done
    rsync -a "$HERE/.cache/ref/" "$HOST:$REMOTE/fnn-bench/.cache/ref/"
}
build() {
    ssh "$HOST" "DATE=$DATE bash $BLK build-wheels"
    ssh "$HOST" "sbatch --wait -p gpu-shared -w node09 -t 00:10:00 -c 16 -J hpc-build-nvcpp -o logs/slurm/%x-%j.out --wrap 'DATE=$DATE bash $BLK build-nvcpp'; cat logs/slurm/hpc-build-nvcpp-*.out | tail -6; ls $REMOTE/fnn-bench/.wheels"
}
submit() { # submit <name> <cmd> [dependency-job-id] -> job id
    local dep=""; [ -n "${3:-}" ] && dep="-d afterany:$3"
    ssh "$HOST" "sbatch --parsable $GPUQ -t 00:30:00 -J $1 -o logs/slurm/%x-%j.out $dep --wrap 'DATE=$DATE bash $BLK $2'"
}
tests() {
    local prev=""
    for p in sycl cuda omp-gpu acpp cpu; do prev=$(submit hpc-parity-$p parity-$p "$prev"); echo "parity-$p -> job $prev"; done
}
matrix() {
    for b in ${*:-$ALL}; do
        local prev="" n=${HPC_NJOBS:-${NJOBS[$b]:-4}}   # HPC_NJOBS overrides the table (re-runs of a block)
        for i in $(seq 1 $n); do prev=$(submit hpc-$b $b "$prev"); done
        local ex; ex=$(submit hpc-$b-extras $b-extras "$prev")
        echo "$b: $n chained jobs (last $prev), extras job $ex"
    done
}
extras() { for b in ${*:-$ALL}; do echo "$b-extras -> job $(submit hpc-$b-extras $b-extras)"; done; }
status() {
    ssh "$HOST" "DATE=$DATE bash $BLK status" 2>&1 | grep -vE 'AdaptiveCpp Warning'
    ssh "$HOST" "squeue -u \$USER -o '%.8i %.10P %.22j %.8T %.10M %.6D %R' | head -40"
}
fetch() { mkdir -p "$HERE/results/hpc-a30"; rsync -a "$HOST:$REMOTE/fnn-bench/results/hpc-a30/" "$HERE/results/hpc-a30/"; }

case ${1:-} in
sync) sync ;; build) build ;; tests) tests ;; matrix) shift; matrix "$@" ;; extras) shift; extras "$@" ;;
status) status ;; fetch) fetch ;;
all) sync; build; tests; matrix ;;
*) echo "usage: $0 sync|build|tests|matrix [blocks]|extras [blocks]|status|fetch|all" >&2; exit 1 ;;
esac
