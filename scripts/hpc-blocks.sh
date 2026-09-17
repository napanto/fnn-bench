#!/usr/bin/env bash
# hpc-a30 (department SLURM cluster: front-end + node09 = 2x AMD EPYC 7301 + NVIDIA A30), cluster side.
# Every step runs inside the fnn-* sandbox directories (docs/toolchains.md) with an unprivileged
# apptainer: the node has no container runtime, the front-end has singularity to convert the
# registry images into sandboxes. Builds run on the front-end, the sweeps inside SLURM jobs
# (scripts/run-hpc.sh submits them: 30-minute jobs chained on the exclusive GPU partition,
# `fnnbench sweep` skips the rows a previous job already measured).
#
#   scripts/hpc-blocks.sh build-wheels | build-nvcpp | parity-<sycl|cuda|omp-gpu|acpp|cpu> |
#                         <block> | <block>-extras | nsys | status
#   blocks: sycl-gpu cuda omp-gpu acpp-gpu sycl-cpu sycl-generic omp-cpu acpp-cpu
#
# Environment: HPC_DIR (repos, default ~/fnn), HPC_SANDBOXES (default ~/sandboxes), APPTAINER, DATE.
set -uo pipefail
R=${HPC_DIR:-$HOME/fnn}; B=$R/fnn-bench; SB=${HPC_SANDBOXES:-$HOME/sandboxes}
APPT=${APPTAINER:-$HOME/.local/apptainer-1.5.3/usr/bin/apptainer}
DATE=${DATE:-$(date +%F)}; RES=results/hpc-a30/$DATE
W=$B/.wheels; PL=$B/pylib
NVHPC=/opt/nvidia/hpc_sdk/Linux_x86_64/26.5
# the nvc++ GPU wheel links the SDK's cuBLAS/cudart 13 (CUDA::cublas from CUDAToolkit_ROOT=$NVHPC/cuda/13.2)
# by soname without an rpath: they must be on the library path at run time (the images ship CUDA 12.9)
NVLIBS="$NVHPC/math_libs/13.2/lib64:$NVHPC/cuda/13.2/lib64:$NVHPC/compilers/lib"
FB="python -m fnnbench.cli"
# 32 physical cores (2 x 16, SMT off in the runtimes): every CPU runtime on the cores, like the
# workstations (docs/methodology.md, "CPU rows: thread counts, affinity and the OpenBLAS variant")
CORES=32
log() { printf '\n==== [%s] %s\n' "$(date +%T)" "$*"; }
binds="-B /tmp"; [ -d /scratch ] && binds="$binds,/scratch"; [ -d /opt/nvidia ] && binds="$binds,/opt/nvidia"
nv=""; command -v nvidia-smi >/dev/null 2>&1 && nv="--nv"
# run <image> <script>: bash inside the sandbox, in fnn-bench, with the reference cache and pylib
run() { local img=$1; shift; $APPT exec $nv $binds "$SB/$img" bash -c "export FNN_REF_CACHE=$B/.cache/ref FNN_DATA_DIR=$HOME/.cache/fnn-testkit PYTHONPATH=$PL:\${PYTHONPATH:-}; cd $B; $*"; }
wheel() { echo "export PYTHONPATH=$PL:$W/hpc-$1"; }
pip_target() { # pip_target <wheel-dir> <src-dir> <build-name>
    echo "rm -rf $W/hpc-$1 && pip install -q --no-deps --target $W/hpc-$1 --config-settings=build-dir=/tmp/hpc-b-$3 $2 2>&1 | grep -E 'error|Error' ; ls $W/hpc-$1 >/dev/null 2>&1 && echo 'BUILD OK $1' || echo 'BUILD FAILED $1'"
}
DRY=${DRY:-}   # DRY=--dry-run: only count pending rows (status)
sw() { local plan=$1 out=$2; shift 2; $FB sweep --plan plans/$plan.json --out $RES/$out $DRY "$@"; }
export -f sw 2>/dev/null || true

case ${1:-} in
build-wheels)
    log "wheels (front-end, compile only): syclnn DPC++ sm_80, generic, AdaptiveCpp; cudann sm_80; ompnn clang-18/gcc-14 nvptx + host compilers"
    run fnn-sycl "export CC=clang CXX=clang++ SYCLNN_TARGETS='spir64;nvidia_gpu_sm_80' SYCLNN_ONEMATH_ROOT=/opt/onemath CMAKE_BUILD_PARALLEL_LEVEL=16; $(pip_target sycl $R/syclnn sycl)"
    run fnn-sycl-generic "export CC=clang CXX=clang++ SYCLNN_TARGETS='spir64' SYCLNN_ONEMATH_ROOT=/opt/onemath-generic CMAKE_BUILD_PARALLEL_LEVEL=16; $(pip_target sycl-generic $R/syclnn sycl-generic)"
    run fnn-acpp-cuda "export CXX=/opt/acpp/bin/acpp CC=clang-18 SYCLNN_SYCL_IMPL=adaptivecpp SYCLNN_ONEMATH_ROOT=/opt/onemath-acpp CMAKE_PREFIX_PATH=/opt/acpp:/opt/onemath-acpp SYCLNN_CT_BACKENDS=netlib ACPP_TARGETS=generic CMAKE_BUILD_PARALLEL_LEVEL=16; $(pip_target sycl-acpp $R/syclnn sycl-acpp)"
    run fnn-cuda "export CUDANN_CUDA_ARCHS=80 CUDAHOSTCXX=g++-13 CMAKE_BUILD_PARALLEL_LEVEL=16; $(pip_target cuda $R/cudann cuda)"
    for cfg in "clang18nv clang++-18 nvidia sm_80 openblas /opt/openblas-openmp" "gcc14nv g++-14 nvidia sm_80 openblas /opt/openblas-openmp" \
               "clang22 clang++-22 cpu - openblas /opt/openblas-openmp" "gcc14 g++-14 cpu - openblas /opt/openblas-openmp" \
               "clang18 clang++-18 cpu - openblas /opt/openblas-openmp" "gcc14mkl g++-14 cpu - mkl /opt/venv"; do
        set -- $cfg
        run fnn-cuda "export CXX=$2 OMPNN_TARGET=$3 OMPNN_OFFLOAD_ARCH=$([ $4 = - ] && echo || echo $4) OMPNN_BLAS=$5 OMPNN_BLAS_ROOT=$6 CMAKE_BUILD_PARALLEL_LEVEL=16; $(pip_target omp-$1 $R/ompnn omp-$1)"
    done
    run fnn-cuda "rm -rf $PL && pip install -q --no-deps --target $PL $B/testkit $B/bench && PYTHONPATH=$PL python -c 'import fnn_testkit, fnnbench; print(\"pylib ok\")'"
    run fnn-cuda "PYTHONPATH=$PL python -c 'from fnn_testkit import datasets; datasets.mnist(\"train\"); datasets.mnist(\"test\"); print(\"mnist cached\")'"
    ls $W ;;
build-nvcpp)
    # nvc++ (HPC SDK on the node) inside the fnn-cuda sandbox: same OpenBLAS/Python as the other
    # OpenMP variants; the SDK needs a localrc for the container's gcc (makelocalrc)
    log "ompnn with nvc++ 26.5 (needs node09: /opt/nvidia): target nvidia cc80 and host -mp=multicore"
    run fnn-cuda "export PATH=$NVHPC/compilers/bin:\$PATH; mkdir -p $R/nvlocalrc; [ -f $R/nvlocalrc/localrc ] || makelocalrc -x -d $R/nvlocalrc -gcc /usr/bin/gcc-13 -gpp /usr/bin/g++-13 -g77 /usr/bin/gfortran-13 $NVHPC/compilers/bin >/dev/null 2>&1; export NVLOCALRC=$R/nvlocalrc/localrc CC=nvc CXX=nvc++ OMPNN_BLAS=openblas OMPNN_BLAS_ROOT=/opt/openblas-openmp CUDAToolkit_ROOT=$NVHPC/cuda/13.2 CMAKE_BUILD_PARALLEL_LEVEL=16
        export OMPNN_TARGET=nvidia OMPNN_OFFLOAD_ARCH=sm_80; $(pip_target omp-nvcpp $R/ompnn omp-nvcpp)
        export OMPNN_TARGET=cpu OMPNN_OFFLOAD_ARCH=; $(pip_target omp-nvcppmc $R/ompnn omp-nvcppmc)" ;;
parity-sycl)
    mkdir -p $B/$RES/parity; P=$B/$RES/parity/sycl-dpcpp.txt; : > $P
    run fnn-sycl "$(wheel sycl); export DPCPP_CPU_NUM_CUS=$CORES OMP_NUM_THREADS=$CORES OMP_PROC_BIND=close OMP_PLACES=cores; cd $R/syclnn; python -c 'import syclnn; print(syclnn.build_info()); print(syclnn.devices())' | tee -a $P
        for o in '--dtype double' '--dtype float' '--option memory=shared' '--option queue=in_order' '--option fine_deps=False' '--blas tiled --dtype double' '--blas tiled --dtype float'; do printf 'syclnn DPC++ cuda:gpu %-28s ' \"\$o\" | tee -a $P; pytest -q --device gpu -p no:cacheprovider \$o 2>&1 | grep -E 'passed|failed|error' | tail -1 | tee -a $P; done
        printf 'syclnn DPC++ cuda:gpu %-28s ' '--run-slow -k mnist' | tee -a $P; pytest -q --device gpu --run-slow -k mnist -p no:cacheprovider 2>&1 | grep -E 'passed|failed|error' | tail -1 | tee -a $P
        for o in '--dtype double' '--dtype float' '--blas netlib --dtype float' '--blas tiled --dtype float'; do printf 'syclnn DPC++ opencl:cpu %-28s ' \"\$o\" | tee -a $P; pytest -q --device cpu -p no:cacheprovider \$o 2>&1 | grep -E 'passed|failed|error' | tail -1 | tee -a $P; done"
    run fnn-sycl-generic "$(wheel sycl-generic); export DPCPP_CPU_NUM_CUS=$CORES OMP_NUM_THREADS=$CORES; cd $R/syclnn; for o in '--blas generic --dtype double' '--blas generic --dtype float'; do printf 'syclnn DPC++ opencl:cpu %-28s ' \"\$o\" | tee -a $P; pytest -q --device cpu -p no:cacheprovider \$o 2>&1 | grep -E 'passed|failed|error' | tail -1 | tee -a $P; done" ;;
parity-cuda)
    mkdir -p $B/$RES/parity; P=$B/$RES/parity/cudann.txt; : > $P
    run fnn-cuda "$(wheel cuda); cd $R/cudann; python -c 'import cudann; print(cudann.build_info()); print(cudann.devices())' | tee -a $P
        for o in '--dtype double' '--dtype float' '--option memory=shared' '--option memory=host' '--option queue=in_order' '--option queue=graph' '--option streams=8' '--option queue=graph --option streams=4' '--option blas=tiled --dtype double' '--option blas=tiled --dtype float' '--option blas=tiled --option queue=graph'; do printf 'cudann %-40s ' \"\$o\" | tee -a $P; pytest -q --device gpu -p no:cacheprovider \$o 2>&1 | grep -E 'passed|failed|error' | tail -1 | tee -a $P; done
        printf 'cudann %-40s ' '--run-slow -k mnist' | tee -a $P; pytest -q --device gpu --run-slow -k mnist -p no:cacheprovider 2>&1 | grep -E 'passed|failed|error' | tail -1 | tee -a $P" ;;
parity-omp-gpu)
    mkdir -p $B/$RES/parity; P=$B/$RES/parity/ompnn-gpu.txt; : > $P
    for w in clang18nv gcc14nv nvcpp; do
        nvl=""; [ $w = nvcpp ] && nvl="export LD_LIBRARY_PATH=$NVLIBS:\${LD_LIBRARY_PATH:-};"
        run fnn-cuda "$(wheel omp-$w); export PATH=$NVHPC/compilers/bin:\$PATH; $nvl cd $R/ompnn; python -c 'import ompnn; print(ompnn.build_info())' | tee -a $P
            for o in '--dtype double' '--dtype float' '--option blas=tiled --dtype double' '--option blas=tiled --dtype float' '--option blas=omp --dtype float'; do printf 'ompnn %-10s gpu %-34s ' $w \"\$o\" | tee -a $P; pytest -q --device gpu -p no:cacheprovider \$o 2>&1 | grep -E 'passed|failed|error' | tail -1 | tee -a $P; done"
    done ;;
parity-acpp)
    mkdir -p $B/$RES/parity; P=$B/$RES/parity/sycl-acpp.txt; : > $P
    run fnn-acpp-cuda "$(wheel sycl-acpp); export OMP_NUM_THREADS=$CORES; cd $R/syclnn; python -c 'import syclnn; print(syclnn.build_info()); print(syclnn.devices())' | tee -a $P
        for o in '--dtype double' '--dtype float' '--option queue=in_order' '--option streams=1' '--option blas_queue=dedicated' '--option memory=shared' '--option memory=host' '--option blas=tiled --dtype float'; do printf 'syclnn AdaptiveCpp cuda:gpu %-34s ' \"\$o\" | tee -a $P; pytest -q --device gpu -p no:cacheprovider \$o 2>&1 | grep -E 'passed|failed|error' | tail -1 | tee -a $P; done
        printf 'syclnn AdaptiveCpp cuda:gpu %-34s ' '--run-slow -k mnist' | tee -a $P; pytest -q --device gpu --run-slow -k mnist -p no:cacheprovider 2>&1 | grep -E 'passed|failed|error' | tail -1 | tee -a $P
        for o in '--dtype double' '--dtype float' '--blas tiled --dtype float'; do printf 'syclnn AdaptiveCpp omp:cpu %-34s ' \"\$o\" | tee -a $P; pytest -q --device cpu -p no:cacheprovider \$o 2>&1 | grep -E 'passed|failed|error' | tail -1 | tee -a $P; done" 2>&1 | grep -v 'AdaptiveCpp Warning' ;;
parity-cpu)
    mkdir -p $B/$RES/parity; P=$B/$RES/parity/ompnn-cpu.txt; : > $P
    for w in gcc14 clang22 clang18 gcc14mkl nvcppmc; do
        case $w in clang*) bind="OMP_PROC_BIND=false" ;; *) bind="OMP_PROC_BIND=close OMP_PLACES=cores" ;; esac
        run fnn-cuda "$(wheel omp-$w); export PATH=$NVHPC/compilers/bin:\$PATH OMP_NUM_THREADS=$CORES $bind MKL_THREADING_LAYER=GNU; cd $R/ompnn; python -c 'import ompnn; print(ompnn.build_info())' | tee -a $P
            for o in '--dtype double' '--dtype float' '--option blas=tiled --dtype float'; do printf 'ompnn %-10s cpu %-30s ' $w \"\$o\" | tee -a $P; pytest -q --device cpu -p no:cacheprovider \$o 2>&1 | grep -E 'passed|failed|error' | tail -1 | tee -a $P; done"
    done ;;
# ---------------------------------------------------------------- GPU blocks (gpu-excl, node09)
sycl-gpu)
    run fnn-sycl "$(wheel sycl); $(declare -f sw); FB='$FB'; RES=$RES; DRY='$DRY'
        sw e2_sycl_cpu_gpu sycl-gpu --select device=gpu
        sw e6_sycl_ablations sycl-gpu-ablations --select device=gpu
        sw w4_sweep_gpu gpu-w4-sycl
        sw e3_cuda_vs_sycl_gpu gpu-cuda-vs-sycl --select backend=syclnn
        sw e7_tiled_gemm e7-tiled-gpu --select device=gpu --select backend=syclnn
        sw e5_inference e5-infer-gpu --select device=gpu --select backend=syclnn
        sw e5_breakdown e5-breakdown-gpu --select device=gpu --select backend=syclnn" ;;
sycl-gpu-extras)
    run fnn-sycl "$(wheel sycl); mkdir -p $RES
        for dt in float double; do $FB peak --backend syclnn --device gpu --dtype \$dt --size 8192 --out $RES/peaks; done
        $FB peak --backend syclnn --device gpu --dtype float --size 8192 --option blas=tiled --out $RES/peaks-tiled" ;;
cuda)
    run fnn-cuda "$(wheel cuda); $(declare -f sw); FB='$FB'; RES=$RES; DRY='$DRY'
        sw e3_cuda_vs_sycl_gpu gpu-cuda-vs-sycl --select backend=cudann
        sw w4_sweep_gpu gpu-w4-cuda --backend cudann
        sw e7_tiled_gemm e7-tiled-gpu --select device=gpu --select backend=cudann
        sw e5_inference e5-infer-gpu --select device=gpu --select backend=cudann
        sw e5_breakdown e5-breakdown-gpu --select device=gpu --select backend=cudann" ;;
cuda-extras)
    run fnn-cuda "$(wheel cuda); mkdir -p $RES
        for dt in float double; do $FB peak --backend cudann --device gpu --dtype \$dt --size 8192 --out $RES/peaks; done
        $FB peak --backend cudann --device gpu --dtype float --size 8192 --option blas=tiled --out $RES/peaks-tiled
        for n in 1024 2048 4096; do $FB peak --backend cudann --device gpu --dtype float --size \$n --out $RES/peaks-sizes; $FB peak --backend syclnn --device gpu --dtype float --size \$n --out $RES/peaks-sizes 2>/dev/null; done" ;;

omp-gpu)
    # the gcc-14 nvptx tiled GEMM (E7) runs last: ~100 s per epoch on the A30 and the kernel aborts
    # the process on some rows (documented in toolchains.md), so it must not stand between the other
    # wheels' sweeps and the 30-minute limit
    for w in clang18nv gcc14nv nvcpp; do
        nvl=""; [ $w = nvcpp ] && nvl="export LD_LIBRARY_PATH=$NVLIBS:\${LD_LIBRARY_PATH:-};"
        e7="sw e7_tiled_gemm e7-tiled-omp-gpu-$w --select device=gpu --select backend=ompnn"; [ $w = gcc14nv ] && e7=""
        run fnn-cuda "$(wheel omp-$w); export PATH=$NVHPC/compilers/bin:\$PATH; $nvl $(declare -f sw); FB='$FB'; RES=$RES; DRY='$DRY'
            sw e4_omp_gpu omp-gpu-$w
            sw e5_inference e5-infer-omp-gpu-$w --select device=gpu --select backend=ompnn
            sw e5_breakdown e5-breakdown-omp-gpu-$w --select device=gpu --select backend=ompnn
            $e7"
    done
    run fnn-cuda "$(wheel omp-gcc14nv); $(declare -f sw); FB='$FB'; RES=$RES; DRY='$DRY'
        sw e7_tiled_gemm e7-tiled-omp-gpu-gcc14nv --select device=gpu --select backend=ompnn --select workload=monk,cup,mnist-512-256,sweep-w256-d4-b256,sweep-w1024-d4-b256" ;;
omp-gpu-extras)
    for w in clang18nv gcc14nv nvcpp; do
        nvl=""; [ $w = nvcpp ] && nvl="export LD_LIBRARY_PATH=$NVLIBS:\${LD_LIBRARY_PATH:-};"
        tiled="$FB peak --backend ompnn --device gpu --dtype float --size 8192 --option blas=tiled --out $RES/peaks-tiled-omp-$w"; [ $w = gcc14nv ] && tiled=""
        run fnn-cuda "$(wheel omp-$w); export PATH=$NVHPC/compilers/bin:\$PATH; $nvl mkdir -p $RES
            $FB peak --backend ompnn --device gpu --dtype float --size 8192 --out $RES/peaks-omp-$w
            $tiled"
    done ;;
acpp-gpu)
    run fnn-acpp-cuda "$(wheel sycl-acpp); export OMP_NUM_THREADS=$CORES; $(declare -f sw); FB='$FB'; RES=$RES; DRY='$DRY'
        sw e2_sycl_cpu_gpu sycl-gpu-acpp --select device=gpu
        sw e3_cuda_vs_sycl_gpu sycl-gpu-acpp --select backend=syclnn
        sw e7_tiled_gemm sycl-gpu-acpp --select device=gpu --select backend=syclnn
        sw e5_inference sycl-gpu-acpp --select device=gpu --select backend=syclnn
        sw e5_breakdown sycl-gpu-acpp --select device=gpu --select backend=syclnn
        sw e6_sycl_ablations sycl-gpu-acpp --select device=gpu" 2>&1 | grep -v 'AdaptiveCpp Warning' ;;
acpp-gpu-extras)
    run fnn-acpp-cuda "$(wheel sycl-acpp); mkdir -p $RES; for dt in float double; do $FB peak --backend syclnn --device gpu --dtype \$dt --size 8192 --out $RES/peaks-acpp; done" 2>&1 | grep -v 'AdaptiveCpp Warning' ;;
# ---------------------------------------------------------------- CPU blocks (the node's EPYC, gpu-excl for exclusivity)
sycl-cpu)
    # the mnist batch-1 double inference row of the DPC++ CPU device exceeds the 30-minute job limit on
    # this node (2048 one-sample launch chains: the float row takes 71 s per pass, double several times
    # that with the check and the repeats), so the batch-1 mnist rows run in float only (documented)
    run fnn-sycl "$(wheel sycl); export OMP_PROC_BIND=close OMP_PLACES=cores DPCPP_CPU_NUM_CUS=$CORES OMP_NUM_THREADS=$CORES; $(declare -f sw); FB='$FB'; RES=$RES; DRY='$DRY'
        sw e1_cpu_blas sycl-cpu
        sw e1_cpu_threads sycl-cpu-threads
        sw e2_sycl_cpu_gpu sycl-cpu --select device=cpu --select dtype=float
        sw e2_sycl_cpu_gpu sycl-cpu --select device=cpu --select dtype=double --select workload=monk,cup,mnist-512-256
        sw e2_sycl_cpu_gpu sycl-cpu --select device=cpu --select dtype=double --select workload=mnist --select batch=4096
        sw e6_sycl_ablations sycl-cpu-ablations --select device=cpu
        sw w4_sweep_cpu sycl-cpu-w4
        sw e7_tiled_gemm e7-tiled-cpu-dpcpp --select device=cpu --select backend=syclnn
        sw e5_inference e5-infer-cpu-dpcpp --select device=cpu --select backend=syclnn --select dtype=float
        sw e5_inference e5-infer-cpu-dpcpp --select device=cpu --select backend=syclnn --select dtype=double --select workload=monk,cup,mnist-512-256
        sw e5_inference e5-infer-cpu-dpcpp --select device=cpu --select backend=syclnn --select dtype=double --select workload=mnist --select batch=4096
        sw e5_breakdown e5-breakdown-cpu-dpcpp --select device=cpu --select backend=syclnn --select dtype=float
        sw e5_breakdown e5-breakdown-cpu-dpcpp --select device=cpu --select backend=syclnn --select dtype=double --select workload=monk,cup,mnist-512-256
        sw e5_breakdown e5-breakdown-cpu-dpcpp --select device=cpu --select backend=syclnn --select dtype=double --select workload=mnist --select batch=4096" ;;
sycl-cpu-extras)
    run fnn-sycl "$(wheel sycl); export OMP_PROC_BIND=close OMP_PLACES=cores DPCPP_CPU_NUM_CUS=$CORES OMP_NUM_THREADS=$CORES; mkdir -p $RES
        $FB peak --backend syclnn --device cpu --dtype float --size 4096 --option blas=mklcpu --out $RES/peaks
        $FB peak --backend syclnn --device cpu --dtype double --size 4096 --option blas=mklcpu --out $RES/peaks
        $FB peak --backend syclnn --device cpu --dtype float --size 4096 --option blas=netlib --out $RES/peaks" ;;
sycl-generic)
    run fnn-sycl-generic "$(wheel sycl-generic); export OMP_PROC_BIND=close OMP_PLACES=cores DPCPP_CPU_NUM_CUS=$CORES OMP_NUM_THREADS=$CORES; $(declare -f sw); FB='$FB'; RES=$RES; DRY='$DRY'
        sw e1_cpu_generic sycl-cpu-generic" ;;
sycl-generic-extras)
    run fnn-sycl-generic "$(wheel sycl-generic); export OMP_PROC_BIND=close OMP_PLACES=cores DPCPP_CPU_NUM_CUS=$CORES OMP_NUM_THREADS=$CORES; mkdir -p $RES
        $FB peak --backend syclnn --device cpu --dtype float --size 4096 --option blas=generic --out $RES/peaks" ;;
omp-cpu)
    for w in gcc14 clang22 clang18 gcc14mkl nvcppmc; do
        case $w in clang*) bind="OMP_PROC_BIND=false" ;; *) bind="OMP_PROC_BIND=close OMP_PLACES=cores" ;; esac
        run fnn-cuda "$(wheel omp-$w); export PATH=$NVHPC/compilers/bin:\$PATH OMP_NUM_THREADS=$CORES $bind MKL_THREADING_LAYER=GNU; $(declare -f sw); FB='$FB'; RES=$RES; DRY='$DRY'
            sw e4_omp_cpu omp-cpu-$w
            if [ $w = gcc14 ]; then
                sw e7_tiled_gemm e7-tiled-omp-cpu-gcc14 --select device=cpu --select backend=ompnn
                sw e5_inference e5-infer-omp-cpu-gcc14 --select device=cpu --select backend=ompnn
                sw e5_breakdown e5-breakdown-omp-cpu-gcc14 --select device=cpu --select backend=ompnn
            fi"
    done ;;
omp-cpu-extras)
    for w in gcc14 clang22 clang18 gcc14mkl nvcppmc; do
        case $w in clang*) bind="OMP_PROC_BIND=false" ;; *) bind="OMP_PROC_BIND=close OMP_PLACES=cores" ;; esac
        run fnn-cuda "$(wheel omp-$w); export PATH=$NVHPC/compilers/bin:\$PATH OMP_NUM_THREADS=$CORES $bind MKL_THREADING_LAYER=GNU; mkdir -p $RES; $FB peak --backend ompnn --device cpu --dtype float --size 4096 --out $RES/peaks-omp-$w"
    done ;;
acpp-cpu)
    run fnn-acpp-cuda "$(wheel sycl-acpp); export OMP_NUM_THREADS=$CORES OMP_PROC_BIND=close OMP_PLACES=cores; $(declare -f sw); FB='$FB'; RES=$RES; DRY='$DRY'
        sw e2_sycl_cpu_gpu sycl-cpu-acpp --select device=cpu --select dtype=float
        sw e5_breakdown e5-breakdown-cpu-acpp --select device=cpu --select backend=syclnn --select dtype=float
        sw e7_tiled_gemm e7-tiled-cpu-acpp --select device=cpu --select backend=syclnn" 2>&1 | grep -v 'AdaptiveCpp Warning' ;;
acpp-cpu-extras)
    run fnn-acpp-cuda "$(wheel sycl-acpp); export OMP_NUM_THREADS=$CORES OMP_PROC_BIND=close OMP_PLACES=cores; mkdir -p $RES; $FB peak --backend syclnn --device cpu --dtype float --size 4096 --out $RES/peaks-acpp" 2>&1 | grep -v 'AdaptiveCpp Warning' ;;
nsys)
    # the traces for the profiler cross-validation (docs/methodology.md): the images' nsys is only the
    # CUDA 12.9 stub ("hasn't been installed with CUDA Toolkit 12.9"), so the node's own Nsight Systems
    # (under /opt/nvidia, bind-mounted) is used inside the sandboxes
    NSYS=""; for c in /opt/nvidia/nsight-systems/2026.1.3/bin/nsys /opt/nvidia/nsight-systems/2026.1.3/target-linux-x64/nsys /opt/nvidia/nsight-systems/2023.3.3/bin/nsys; do [ -x $c ] && { NSYS=$c; break; }; done
    [ -n "$NSYS" ] || { echo "no Nsight Systems under /opt/nvidia on $(hostname)"; exit 1; }; echo "nsys: $NSYS ($($NSYS --version 2>/dev/null | head -1))"
    trace() { # trace <image> <wheel> <backend> <tag> [env...]
        local img=$1 w=$2 be=$3 tag=$4; shift 4
        run $img "$(wheel $w); export PATH=$NVHPC/compilers/bin:\$PATH $*; mkdir -p $RES
            $NSYS profile -o $RES/nsys-$tag --force-overwrite true $FB run --backend $be --device gpu --workload mnist-512-256 --batch 256 --dtype float --epochs 1 --repeat 1 --warmup 1 --option profile=True --samples 8192 2>&1 | tail -3
            for rep in cuda_gpu_kern_sum cuda_api_sum cuda_gpu_mem_time_sum; do $NSYS stats --report \$rep --format csv -o $RES/nsys-$tag-\$rep $RES/nsys-$tag.nsys-rep >/dev/null 2>&1 || echo \"stats \$rep failed\"; done; ls -la $RES | grep nsys-$tag | cut -c30-"
    }
    trace fnn-sycl sycl syclnn syclnn-mnist
    trace fnn-cuda cuda cudann cudann-mnist
    trace fnn-cuda omp-clang18nv ompnn ompnn-clang18nv-mnist
    trace fnn-cuda omp-nvcpp ompnn ompnn-nvcpp-mnist LD_LIBRARY_PATH=$NVLIBS:\${LD_LIBRARY_PATH:-} ;;
status)
    for b in sycl-gpu cuda omp-gpu acpp-gpu sycl-cpu sycl-generic omp-cpu acpp-cpu; do
        echo "--- $b"; DRY=--dry-run "$0" $b 2>&1 | grep -E 'dry run:|Error|error' | sed 's/^/    /'
    done ;;
*) echo "usage: $0 build-wheels|build-nvcpp|parity-{sycl,cuda,omp-gpu,acpp,cpu}|<block>|<block>-extras|status" >&2; exit 2 ;;
esac
