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
# Extras: nsys-epoch | nsys-gap | wallprof | overlap-probe | ncu-check | ncu-tiled <cuda|sycl|omp> | stat-confirm |
#         sanitizer-check | sanitizer <tool> | stream. Each has its own result directory.
#   blocks: sycl-gpu cuda omp-gpu acpp-gpu sycl-cpu sycl-generic omp-cpu acpp-cpu
#   hardware-thread points of the scaling series: sycl-cpu-threads-smt omp-cpu-threads-smt
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
# workstations (docs/methodology.md, "CPU thread policy")
CORES=32
log() { printf '\n==== [%s] %s\n' "$(date +%T)" "$*"; }
binds="-B /tmp"; [ -d /scratch ] && binds="$binds,/scratch"; [ -d /opt/nvidia ] && binds="$binds,/opt/nvidia"
nv=""; command -v nvidia-smi >/dev/null 2>&1 && nv="--nv"
# run <image> <script>: bash inside the sandbox, in fnn-bench, with the reference cache and pylib
run() { local img=$1; shift; $NUMA $APPT exec $nv $binds "$SB/$img" bash -c "export FNN_REF_CACHE=$B/.cache/ref FNN_DATA_DIR=$HOME/.cache/fnn-testkit PYTHONPATH=$PL:\${PYTHONPATH:-}; cd $B; $*"; }
wheel() { echo "export PYTHONPATH=$PL:$W/hpc-$1"; }
pip_target() { # pip_target <wheel-dir> <src-dir> <build-name>
    local link=""
    # Apptainer puts /lib/x86_64-linux-gnu on LD_LIBRARY_PATH. DT_RUNPATH loses
    # to its alternatives link (pthread); DT_RPATH preserves the linked variant.
    # nvc++ multicore keeps pthread OpenBLAS, whose own
    # OpenMP runtime would otherwise coexist with OpenBLAS's libgomp.
    [[ $1 = omp-* && $1 != omp-nvcpp* && $1 != omp-gcc14mkl && $1 != omp-clang18wait ]] && link="--config-settings=cmake.define.CMAKE_MODULE_LINKER_FLAGS=-Wl,--disable-new-dtags"
    echo "rm -rf $W/hpc-$1 && pip install -q --no-deps --target $W/hpc-$1 --config-settings=build-dir=/tmp/hpc-b-$3 $link $2 && echo 'BUILD OK $1' || { echo 'BUILD FAILED $1'; exit 1; }"
}
DRY=${DRY:-}   # DRY=--dry-run: only count pending rows (status)
sw() { local plan=$1 out=$2; shift 2; $FB sweep --plan plans/$plan.json --out $RES/$out $DRY "$@"; }
export -f sw 2>/dev/null || true

extra_dir() { RES=results/hpc-a30/$1-$DATE; mkdir -p "$B/$RES"; }
find_ncu() { find /opt/nvidia -type f -name ncu -executable -path '*x64*' 2>/dev/null | sort -V | tail -1; }
extra_policy() {
    local policy=$B/results/hpc-a30/NUMA-POLICY
    [ -s "$policy" ] || policy=$B/$RES/NUMA-POLICY
    [ -s "$policy" ] || { echo 'waiting for NUMA-POLICY'; return 1; }
    NUMA=$(python3 "$B/scripts/gpu-numa-prefix.py" "$policy") || return 1
    [ "$policy" = "$B/$RES/NUMA-POLICY" ] || cp "$policy" "$B/$RES/NUMA-POLICY"
}
kernel_roofline() {
    local nc=results/hpc-a30/ncu-tiled-$DATE stream=results/hpc-a30/stream-$DATE
    local figure=results/analysis/figures/hpc-a30-kernel-$DATE
    [ -s "$B/$nc/kernels.json" ] && [ -s "$B/$stream/summary.json" ] || return 0
    # Runs after measurement, in the exclusive allocation. Plot dependencies are
    # isolated from pylib and the benchmark wheels; the last producer refreshes it.
    if run fnn-cuda "export PYTHONPATH=$B/results/analysis/engineer-check-env-$DATE:$PL; python scripts/kernel-roofline.py --ncu $nc --stream $stream --out $figure" > "$B/$nc/roofline.log" 2>&1; then
        for dir in "$nc" "$stream"; do
            printf '\nPer-kernel roofline: [PNG](../../analysis/figures/hpc-a30-kernel-%s/kernel-roofline.png), separate from application W4.\n' "$DATE" >> "$B/$dir/README.md"
        done
    else
        printf '\nPer-kernel figure not generated; see ncu-tiled-%s/roofline.log. Counter/STREAM tables remain available.\n' "$DATE" >> "$B/$stream/README.md"
    fi
}

# Memory policy (docs/methodology.md, "Memory placement"): the CPU blocks run with interleaved memory;
# the GPU blocks keep the default policy, since binding the process to the A30's node changed no GPU
# row beyond the noise (results/hpc-a30/numa-pilot-2026-11-03, results/hpc-a30/NUMA-POLICY)
NUMA=""
case ${1:-} in sycl-cpu*|sycl-generic*|omp-cpu*|acpp-cpu*) NUMA="numactl --interleave=all" ;; esac
# ompnn links the OpenMP-threaded OpenBLAS through its RUNPATH, but the sandbox's LD_LIBRARY_PATH
# (/lib/x86_64-linux-gnu) is searched first and the alternatives symlink there is the pthread build:
# put the OpenMP variant first, as it resolves on the workstations (the 2026-09-17 rows loaded pthread).
# nvc++ multicore keeps the build the image resolves: the OpenMP variant would bring libgomp next to
# nvc++'s own OpenMP runtime.
ompblas() { case $1 in gcc14mkl|nvcppmc) ;; *) echo "export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/openblas-openmp:\${LD_LIBRARY_PATH:-};" ;; esac; }

case ${1:-} in
nsys-epoch)
    extra_dir nsys-epoch; extra_policy || exit 1
    NSYS=$(find /opt/nvidia -type f -name nsys -executable -path '*nsight-systems*x64*' 2>/dev/null | sort -V | tail -1)
    [ -n "$NSYS" ] || { echo 'Nsight Systems unavailable'; exit 1; }
    "$NSYS" --version > "$B/$RES/versions.txt" 2>&1
    printf 'NUMA prefix: %s\n' "${NUMA:-none (adopted default)}" > "$B/$RES/commands.txt"
    run fnn-cuda "g++-13 -O2 -shared -fPIC -I/usr/local/cuda-12.9/include scripts/nvtx-range.cpp -ldl -o $RES/nvtx-range.so" || exit 1
    for cfg in 'fnn-cuda cuda cudann' 'fnn-sycl sycl syclnn' 'fnn-cuda omp-clang18nv ompnn'; do
        set -- $cfg; img=$1; w=$2; be=$3
        for profiling in off on; do
            opt=""; [ "$profiling" = on ] && opt=--profile
            tag=$be-$profiling
            cmd="$NSYS profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --capture-range=cudaProfilerApi --capture-range-end=stop --force-overwrite true -o $RES/$tag python scripts/nsys_epoch.py --backend $be --nvtx-lib $B/$RES/nvtx-range.so --out $RES/$tag.json $opt"
            printf '%s\n' "$cmd" >> "$B/$RES/commands.txt"
            run "$img" "$(wheel "$w"); $cmd" > "$B/$RES/$tag.log" 2>&1 || continue
            run "$img" "$NSYS stats --report cuda_gpu_trace --format csv -o $RES/$tag-gpu $RES/$tag.nsys-rep
                $NSYS stats --report cuda_api_sum --format csv -o $RES/$tag-api $RES/$tag.nsys-rep
                $NSYS export --type sqlite --force-overwrite true -o $RES/$tag.sqlite $RES/$tag.nsys-rep" >> "$B/$RES/$tag.log" 2>&1
        done
    done
    run fnn-cuda "python scripts/nsys-epoch-table.py $RES" ;;
nsys-gap)  # one delimited capture per execution path, to attribute the GPU idle time of an epoch
    NS=${NS:-60000}; WARM=${WARM:-1}; extra_dir nsys-gap-n$NS-w$WARM; extra_policy || exit 1
    NSYS=$(find /opt/nvidia -type f -name nsys -executable -path '*nsight-systems*x64*' 2>/dev/null | sort -V | tail -1)
    [ -n "$NSYS" ] || { echo 'Nsight Systems unavailable'; exit 1; }
    "$NSYS" --version > "$B/$RES/versions.txt" 2>&1
    { echo "NUMA prefix: ${NUMA:-none (adopted default)}"; echo "perf_event_paranoid $(cat /proc/sys/kernel/perf_event_paranoid)"; echo "rtprio limit $(ulimit -r)"
      sort /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor | uniq -c; numactl --hardware | head -3; nvidia-smi -q -d CLOCK,PERFORMANCE | sed -n 1,40p; } > "$B/$RES/node-state.txt" 2>&1
    run fnn-cuda "g++-13 -O2 -shared -fPIC -I/usr/local/cuda-12.9/include scripts/nvtx-range.cpp -ldl -o $RES/nvtx-range.so" || exit 1
    # image wheel backend tag trace extra-options library-path
    while read -r img w be tag trace opt nvl; do
        pre=""; [ "$nvl" = nv ] && pre="export LD_LIBRARY_PATH=$NVLIBS:\${LD_LIBRARY_PATH:-};"
        o=""; [ "$opt" = - ] || o="--option $opt"
        cmd="$NSYS profile --trace=$trace --sample=none --cpuctxsw=none --capture-range=cudaProfilerApi --capture-range-end=stop --force-overwrite true -o $RES/$tag python scripts/nsys_epoch.py --backend $be $o --samples $NS --warm-epochs $WARM --nvtx-lib $B/$RES/nvtx-range.so --out $RES/$tag.json"
        printf '%s\n' "$cmd" >> "$B/$RES/commands.txt"
        run "$img" "$pre $(wheel "$w"); $cmd" > "$B/$RES/$tag.log" 2>&1 || continue
        run "$img" "$NSYS stats --report cuda_gpu_trace --format csv -o $RES/$tag-gpu $RES/$tag.nsys-rep
            $NSYS stats --report cuda_api_sum --format csv -o $RES/$tag-api $RES/$tag.nsys-rep
            $NSYS export --type sqlite --force-overwrite true -o $RES/$tag.sqlite $RES/$tag.nsys-rep" >> "$B/$RES/$tag.log" 2>&1
    done <<'CAPTURES'
fnn-cuda cuda cudann cudann-default cuda,nvtx - -
fnn-cuda cuda cudann cudann-inorder cuda,nvtx queue=in_order -
fnn-sycl sycl syclnn syclnn-default cuda,nvtx - -
fnn-sycl sycl syclnn syclnn-osrt cuda,nvtx,osrt - -
fnn-cuda omp-clang18nv ompnn ompnn-clang18 cuda,nvtx - -
fnn-cuda omp-nvcpp ompnn ompnn-nvcpp cuda,nvtx - nv
fnn-cuda omp-gcc14nv ompnn ompnn-gcc14 cuda,nvtx - -
CAPTURES
    ;;
launch-probe-build)  # front-end, no GPU: CUDA, DPC++, clang and gcc probes; nvc++ needs the node (launch-probe)
    extra_dir launch-probe; P=$B/scripts/launch-probe; O=$B/$RES
    run fnn-cuda "cd $P && nvcc -O3 -std=c++17 -arch=sm_80 -ccbin g++-13 probe_cuda.cu -o $O/probe-cuda" && echo 'BUILD OK cuda'
    run fnn-sycl "cd $P && clang++ -O3 -std=c++17 -fsycl -fsycl-targets=nvidia_gpu_sm_80 probe_sycl.cpp -o $O/probe-dpcpp" && echo 'BUILD OK dpcpp'
    run fnn-cuda "cd $P && clang++-18 -O3 -std=c++17 -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda --offload-arch=sm_80 -DCOMPILER='\"clang18\"' probe_omp.cpp -o $O/probe-clang18" && echo 'BUILD OK clang18'
    run fnn-cuda "cd $P && g++-14 -O3 -std=c++17 -fopenmp -foffload=nvptx-none -foffload-options=nvptx-none=-misa=sm_80 -fcf-protection=none -fno-stack-protector -DCOMPILER='\"gcc14\"' probe_omp.cpp -o $O/probe-gcc14" && echo 'BUILD OK gcc14'
    run fnn-cuda "cd $P && nvcc -O3 -std=c++17 -arch=sm_80 -ccbin g++-13 probe2_cuda.cu -lcublas -o $O/probe2-cuda" && echo 'BUILD OK cuda2'
    run fnn-sycl "cd $P && clang++ -O3 -std=c++17 -fsycl -fsycl-targets=nvidia_gpu_sm_80 -I/opt/onemath/include probe2_sycl.cpp -L/opt/onemath/lib -lonemath -Wl,-rpath,/opt/onemath/lib -o $O/probe2-dpcpp" && echo 'BUILD OK dpcpp2' ;;
launch-probe)  # one exclusive job: three alternating process rounds of every probe
    extra_dir launch-probe; O=$B/$RES; P=$B/scripts/launch-probe
    [ -x "$O/probe-nvcpp" ] || run fnn-cuda "export PATH=$NVHPC/compilers/bin:\$PATH NVLOCALRC=$R/nvlocalrc/localrc; cd $P && nvc++ -O3 -std=c++17 -mp=gpu -gpu=cc80 -DCOMPILER='\"nvcpp\"' probe_omp.cpp -o $O/probe-nvcpp" >> "$O/build-nvcpp.log" 2>&1
    nvidia-smi --query-gpu=clocks.gr,clocks.mem,pstate --format=csv > "$O/gpu-clocks.txt" 2>&1
    nvl="export LD_LIBRARY_PATH=$NVHPC/compilers/lib:\${LD_LIBRARY_PATH:-};"
    llvm="export LD_LIBRARY_PATH=/usr/lib/llvm-18/lib:\${LD_LIBRARY_PATH:-};"
    for round in 1 2 3; do
        list='cuda dpcpp clang18 gcc14 nvcpp cuda2 dpcpp2'; [ $((round % 2)) = 0 ] && list='dpcpp2 cuda2 nvcpp gcc14 clang18 dpcpp cuda'
        for w in $list; do
            [ -s "$O/$w-r$round.csv" ] && continue
            bin=probe-$w; case $w in cuda2) bin=probe2-cuda ;; dpcpp2) bin=probe2-dpcpp ;; esac
            img=fnn-cuda; case $w in dpcpp*) img=fnn-sycl ;; esac
            pre=""; [ "$w" = nvcpp ] && pre=$nvl; [ "$w" = clang18 ] && pre=$llvm
            [ -x "$O/$bin" ] || continue
            timeout -s KILL 150 $APPT exec $nv $binds "$SB/$img" bash -c "$pre OMP_TARGET_OFFLOAD=MANDATORY PROBE_R=9 $O/$bin" > "$O/$w-r$round.csv" 2> "$O/$w-r$round.err"
        done
    done
    echo 'launch probe done' ;;
gpu-placement)  # launch-bound rows by NUMA node of the submitting thread; the A30 sits on node 4
    extra_dir gpu-placement; O=$B/$RES
    numactl --hardware > "$O/numa.txt" 2>&1
    nvidia-smi --query-gpu=clocks.gr,clocks.mem,pstate,pcie.link.gen.current,pcie.link.width.current --format=csv > "$O/gpu.txt" 2>&1
    deadline=$(( $(date +%s) + ${MAX_S:-1320} ))
    one() { # one <placement> <round> <image> <wheel> <backend> <tag> <workload> <samples>
        local place=$1 k=$2 img=$3 w=$4 be=$5 tag=$6 wl=$7 ns=$8
        local out=$RES/$place/$tag-$wl/round-$k
        mkdir -p "$B/$out"
        ls "$B/$out"/*.jsonl >/dev/null 2>&1 && return 0
        [ "$(date +%s)" -ge "$deadline" ] && return 0
        local pre=""; [ "$tag" = nvcpp ] && pre="export LD_LIBRARY_PATH=$NVLIBS:\${LD_LIBRARY_PATH:-};"
        case $place in unbound*) NUMA="" ;; core*) NUMA="numactl --physcpubind=${place#core} --localalloc" ;; *) NUMA="numactl --cpunodebind=${place#node} --localalloc" ;; esac
        local proto="--batch 40 --epochs 5 --repeat 5"; [ "$wl" = mnist-512-256 ] && proto="--batch 256 --epochs 3 --repeat 3"
        run "$img" "$pre $(wheel "$w"); python -m fnnbench.cli run --backend $be --device gpu --dtype float --workload $wl $proto --warmup 1 --seed 1 --samples $ns --out $out & p=\$!; while kill -0 \$p 2>/dev/null; do ps -o psr= -p \$p; sleep 0.25; done > $out/psr.txt; wait \$p" > "$B/$out/run.log" 2>&1
    }
    for k in ${ROUNDS:-1 2 3}; do
        places=${PLACES:-unbound node4 node0 node1 node2 node3 node5 node6 node7}
        [ $((k % 2)) = 0 ] && places=$(echo "$places" | tr ' ' '\n' | tac | tr '\n' ' ')
        for place in $places; do
            for cfg in 'fnn-cuda cuda cudann cuda' 'fnn-sycl sycl syclnn dpcpp' 'fnn-cuda omp-clang18nv ompnn clang18' 'fnn-cuda omp-nvcpp ompnn nvcpp'; do
                set -- $cfg
                one "$place" "$k" "$1" "$2" "$3" "$4" cup 1000
                one "$place" "$k" "$1" "$2" "$3" "$4" monk 124
                [ -n "${WITH_MNIST:-}" ] && one "$place" "$k" "$1" "$2" "$3" "$4" mnist-512-256 60000
            done
        done
    done
    NUMA=""; echo "placement rows: $(find "$O" -name '*.jsonl' | wc -l)" ;;
clock-state)  # does the A30's clock depend on what ran just before? Long runs with a clock log, then short rows after idle and after load
    extra_dir clock-state; O=$B/$RES
    nvidia-smi --query-gpu=timestamp,clocks.gr,clocks.mem,utilization.gpu,power.draw,temperature.gpu,pstate --format=csv,noheader -lms 100 > "$O/clocks.csv" &
    logger=$!
    nvl="export LD_LIBRARY_PATH=$NVLIBS:\${LD_LIBRARY_PATH:-};"
    sleep 20
    run fnn-cuda "$(wheel cuda); python scripts/clock_ramp.py --backend cudann --out $RES/ramp-cudann.json" > "$O/ramp-cudann.log" 2>&1
    sleep 30
    run fnn-sycl "$(wheel sycl); python scripts/clock_ramp.py --backend syclnn --out $RES/ramp-syclnn.json" > "$O/ramp-syclnn.log" 2>&1
    sleep 30
    run fnn-cuda "$nvl $(wheel omp-nvcpp); python scripts/clock_ramp.py --backend ompnn --out $RES/ramp-nvcpp.json" > "$O/ramp-nvcpp.log" 2>&1
    rows() { # rows <state> <round>
        local state=$1 round=$2 img w be tag out
        while read -r img w be tag; do
            out=$RES/$state/$tag/round-$round; mkdir -p "$B/$out"
            run "$img" "$(wheel "$w"); date +%s.%N > $out/start.txt; python -m fnnbench.cli run --backend $be --device gpu --dtype float --workload monk --batch 40 --epochs 5 --repeat 5 --warmup 1 --seed 1 --samples 124 --out $out" > "$B/$out/run.log" 2>&1 < /dev/null
        done <<'ROWS'
fnn-cuda omp-clang18nv ompnn clang18
fnn-cuda cuda cudann cuda
fnn-sycl sycl syclnn dpcpp
ROWS
    }
    for round in 1 2 3 4; do
        sleep 40; rows after-idle "$round"
        run fnn-cuda "$(wheel cuda); python scripts/clock_ramp.py --backend cudann --seconds 6 --out $RES/load-$round.json" >> "$O/load.log" 2>&1
        rows after-load "$round"
    done
    kill $logger 2>/dev/null; echo 'clock state done' ;;
build-aocl)  # front-end: AMD's BLIS (AOCL 5.3) as a CBLAS for the OpenMP host path
    AO=$R/aocl; S=$R/aocl-src/blis
    run fnn-cuda "cd $S && CC=gcc-13 ./configure --enable-cblas --enable-threading=openmp --prefix=$AO amdzen > $R/aocl-src/configure.log 2>&1 && make -j8 > $R/aocl-src/make.log 2>&1 && make install >> $R/aocl-src/make.log 2>&1" || { echo 'BUILD FAILED blis'; exit 1; }
    mkdir -p "$R/aocl-cblas/lib" "$R/aocl-cblas/include"
    ln -sf "$AO/lib/libblis-mt.so" "$R/aocl-cblas/lib/libcblas.so"; ln -sf "$AO/lib/libblis-mt.so" "$R/aocl-cblas/lib/libblas.so"; ln -sf "$AO/lib/libblis-mt.so" "$R/aocl-cblas/lib/libopenblas.so"
    cp "$AO/include/blis/cblas.h" "$R/aocl-cblas/include/" 2>/dev/null || cp "$AO"/include/*/cblas.h "$R/aocl-cblas/include/"
    run fnn-cuda "export CXX=g++-14 OMPNN_TARGET=cpu OMPNN_OFFLOAD_ARCH= OMPNN_BLAS=openblas OMPNN_BLAS_ROOT=$R/aocl-cblas CMAKE_BUILD_PARALLEL_LEVEL=8; $(pip_target omp-gcc14aocl $R/ompnn omp-gcc14aocl)" ;;
cpu-blas-amd)  # gcc OpenMP host path on three CBLAS libraries, interleaved pages, three fresh processes each
    extra_dir cpu-blas-amd; O=$B/$RES
    lscpu | grep -E 'Model name|Flags' | cut -c1-400 > "$O/cpu.txt"
    run_row() { # run_row <wheel> <round> <workload> <batch> <samples> <extra env>
        local w=$1 k=$2 wl=$3 bs=$4 ns=$5 out=$RES/$1/$3-b$4/round-$2; shift 5
        mkdir -p "$B/$out"; ls "$B/$out"/*.jsonl >/dev/null 2>&1 && return 0
        NUMA="numactl --interleave=all"
        run fnn-cuda "$(wheel omp-$w); $(ompblas $w) export OMP_NUM_THREADS=$CORES OMP_PROC_BIND=close OMP_PLACES=cores MKL_THREADING_LAYER=GNU $*; python -m fnnbench.cli run --backend ompnn --device cpu --dtype float --workload $wl --batch $bs --epochs 3 --repeat 3 --warmup 1 --seed 1 --samples $ns --tag cpu-blas-amd --out $out" > "$B/$out/run.log" 2>&1 < /dev/null
        NUMA=""
    }
    for k in 1 2 3; do
        for w in gcc14 gcc14mkl gcc14aocl; do
            e=""; [ "$w" = gcc14aocl ] && e="LD_LIBRARY_PATH=$R/aocl/lib:\${LD_LIBRARY_PATH:-} BLIS_ARCH_TYPE=zen"
            [ "$w" = gcc14mkl ] && [ "$k" = 1 ] && e="MKL_VERBOSE=1"
            run_row "$w" "$k" mnist-512-256 256 60000 $e
            run_row "$w" "$k" cup 40 1000 $e
        done
    done
    echo 'cpu blas done' ;;
launch-probe-more)  # second pass of the probes: real matrix sizes, and the clauses of ompnn's kernels
    extra_dir launch-probe; O=$B/$RES; P=$B/scripts/launch-probe; mkdir -p "$O/more"
    run fnn-cuda "cd $P && clang++-18 -O3 -std=c++17 -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda --offload-arch=sm_80 -DCOMPILER='\"clang18\"' probe_omp.cpp -o $O/probe-clang18" > "$O/build2.log" 2>&1
    run fnn-cuda "cd $P && g++-14 -O3 -std=c++17 -fopenmp -foffload=nvptx-none -foffload-options=nvptx-none=-misa=sm_80 -fcf-protection=none -fno-stack-protector -DCOMPILER='\"gcc14\"' probe_omp.cpp -o $O/probe-gcc14" >> "$O/build2.log" 2>&1
    run fnn-cuda "export PATH=$NVHPC/compilers/bin:\$PATH NVLOCALRC=$R/nvlocalrc/localrc; cd $P && nvc++ -O3 -std=c++17 -mp=gpu -gpu=cc80 -DCOMPILER='\"nvcpp\"' probe_omp.cpp -o $O/probe-nvcpp" >> "$O/build2.log" 2>&1
    nvl="export LD_LIBRARY_PATH=$NVHPC/compilers/lib:\${LD_LIBRARY_PATH:-};"; llvm="export LD_LIBRARY_PATH=/usr/lib/llvm-18/lib:\${LD_LIBRARY_PATH:-};"
    for round in 1 2 3; do
        for M in 64 256; do
            timeout -s KILL 100 $APPT exec $nv $binds "$SB/fnn-cuda" bash -c "PROBE_M=$M PROBE_N=1000 PROBE_R=9 $O/probe2-cuda" > "$O/more/cuda2-m$M-r$round.csv" 2> /dev/null
            timeout -s KILL 100 $APPT exec $nv $binds "$SB/fnn-sycl" bash -c "PROBE_M=$M PROBE_N=1000 PROBE_R=9 $O/probe2-dpcpp" > "$O/more/dpcpp2-m$M-r$round.csv" 2> /dev/null
        done
        for w in clang18 gcc14 nvcpp; do
            pre=""; [ "$w" = nvcpp ] && pre=$nvl; [ "$w" = clang18 ] && pre=$llvm
            n=500; [ "$w" = gcc14 ] && n=100
            timeout -s KILL 100 $APPT exec $nv $binds "$SB/fnn-cuda" bash -c "$pre OMP_TARGET_OFFLOAD=MANDATORY PROBE_N=$n PROBE_R=5 $O/probe-$w" > "$O/more/$w-r$round.csv" 2> "$O/more/$w-r$round.err"
        done
    done
    echo 'more probes done' ;;
launch-probe3-build)  # front-end
    extra_dir launch-probe; P=$B/scripts/launch-probe; O=$B/$RES
    run fnn-cuda "cd $P && nvcc -O3 -std=c++17 -arch=sm_80 -ccbin g++-13 probe3_cuda.cu -o $O/probe3-cuda" && echo 'BUILD OK cuda3'
    run fnn-sycl "cd $P && clang++ -O3 -std=c++17 -fsycl -fsycl-targets=nvidia_gpu_sm_80 probe3_sycl.cpp -o $O/probe3-dpcpp" && echo 'BUILD OK dpcpp3' ;;
launch-probe3)  # one short job: launch cost with 12 arguments; gcc target regions against teams and threads
    extra_dir launch-probe; O=$B/$RES; mkdir -p "$O/more3"
    for round in 1 2 3; do
        timeout -s KILL 60 $APPT exec $nv $binds "$SB/fnn-cuda" bash -c "PROBE_R=9 $O/probe3-cuda" > "$O/more3/cuda3-r$round.csv" 2>/dev/null
        timeout -s KILL 60 $APPT exec $nv $binds "$SB/fnn-sycl" bash -c "PROBE_R=9 $O/probe3-dpcpp" > "$O/more3/dpcpp3-r$round.csv" 2>/dev/null
    done
    for teams in 1 8 56; do
        for threads in 1 4 12; do
            OMP=$(cat <<ENV
OMP_TARGET_OFFLOAD=MANDATORY OMP_NUM_TEAMS=$teams OMP_TEAMS_THREAD_LIMIT=$threads OMP_NUM_THREADS=$threads PROBE_N=100 PROBE_R=4
ENV
)
            timeout -s KILL 30 $APPT exec $nv $binds "$SB/fnn-cuda" bash -c "$OMP $O/probe-gcc14" > "$O/more3/gcc-teams$teams-threads$threads.txt" 2>&1
        done
    done
    echo 'probe3 done' ;;
launch-probe4-build)  # front-end
    extra_dir launch-probe; P=$B/scripts/launch-probe; O=$B/$RES
    run fnn-sycl "cd $P && clang++ -O3 -std=c++17 -fsycl -fsycl-targets=nvidia_gpu_sm_80 -I/usr/local/cuda-12.9/include -I/usr/local/cuda/include probe4_sycl.cpp -L/usr/local/cuda-12.9/lib64 -L/usr/local/cuda/lib64 -lcublas -lcuda -o $O/probe4-dpcpp" && echo 'BUILD OK dpcpp4' ;;
launch-probe4)
    extra_dir launch-probe; O=$B/$RES; mkdir -p "$O/more4"
    for round in 1 2 3; do
        timeout -s KILL 60 $APPT exec $nv $binds "$SB/fnn-sycl" bash -c "PROBE_R=9 $O/probe4-dpcpp" > "$O/more4/dpcpp4-r$round.csv" 2> "$O/more4/dpcpp4-r$round.err"
    done
    echo 'probe4 done' ;;
clang-spin)  # is clang's slow mode libomptarget's blocking wait? One short job, fresh processes, arms alternated
    extra_dir clang-spin; O=$B/$RES
    ls -d /usr/local/cuda*/lib64 > "$O/env.txt" 2>&1
    one() { # one <arm> <round> <workload> <samples> <batch> <seconds> <place>
        local arm=$1 k=$2 wl=$3 ns=$4 b=$5 secs=$6 place=$7 x=""
        [ "$arm" = spin ] && x=--spin
        NUMA=""; [ "$place" = node4 ] && NUMA="numactl --cpunodebind=4 --localalloc"
        run fnn-cuda "export LD_LIBRARY_PATH=/usr/local/cuda-12.9/lib64:\${LD_LIBRARY_PATH:-}; $(wheel omp-clang18nv); python scripts/clock_ramp.py --backend ompnn --workload $wl --samples $ns --batch $b --seconds $secs $x --out $RES/$arm-$wl-$place-r$k.json" > "$O/$arm-$wl-$place-r$k.log" 2>&1 < /dev/null
    }
    for k in 1 2 3 4; do
        arms='default spin'; [ $((k % 2)) = 0 ] && arms='spin default'
        for arm in $arms; do
            for place in unbound node4; do
                one "$arm" "$k" monk 124 40 4 "$place"
                one "$arm" "$k" cup 1000 40 4 "$place"
            done
            one "$arm" "$k" mnist-512-256 60000 256 8 unbound
        done
    done
    NUMA=""; echo 'clang spin done' ;;
build-ompnn-wait)  # front-end: ompnn 0.1.2 with clang-18 for NVIDIA, in its own wheel directory
    run fnn-cuda "export CXX=clang++-18 OMPNN_TARGET=nvidia OMPNN_OFFLOAD_ARCH=sm_80 OMPNN_BLAS=openblas OMPNN_BLAS_ROOT=/opt/openblas-openmp CMAKE_BUILD_PARALLEL_LEVEL=8; $(pip_target omp-clang18wait $R/ompnn omp-clang18wait)" ;;
wallprof)  # wall-clock stacks of the submitting thread: where an epoch goes on the machine that forbids perf events
    extra_dir wallprof; extra_policy || exit 1
    run fnn-cuda "gcc -O2 -shared -fPIC scripts/wallprof/wallprof.c -o $RES/wallprof.so" || exit 1
    one() { # one <image> <wheel> <backend> <tag> <epochs> <round> <library-path:nv|-> [option...]
        local img=$1 w=$2 be=$3 tag=$4 ep=$5 k=$6 nvl=$7; shift 7
        local out=$RES/$tag-r$k opts="" pre=""
        [ -s "$B/$out.stacks.json" ] && return 0
        [ "$nvl" = nv ] && pre="export LD_LIBRARY_PATH=$NVLIBS:\${LD_LIBRARY_PATH:-};"
        for o in "$@"; do opts="$opts --option $o"; done
        run "$img" "$pre $(wheel "$w"); python scripts/wallprof/wallprof_run.py --backend $be --epochs $ep --lib $RES/wallprof.so --out $out.dump $opts && python scripts/wallprof/wallprof_report.py $out.dump"
    }
    for k in 1 2 3; do
        one fnn-cuda cuda cudann cuda-inorder 200 "$k" - queue=in_order
        one fnn-sycl sycl syclnn dpcpp 100 "$k" -
        one fnn-cuda omp-nvcpp ompnn nvcpp 70 "$k" nv
        one fnn-cuda omp-clang18wait ompnn clang-spin 70 "$k" - wait=spin
    done
    ;;
overlap-probe)  # one short job: do kernels of different streams overlap untraced; the gcc epoch against thread_limit
    extra_dir overlap-probe; O=$B/$RES; P=$B/scripts/launch-probe; extra_policy || exit 1
    nvidia-smi --query-gpu=clocks.gr,clocks.mem,pstate --format=csv > "$O/gpu-clocks.txt" 2>&1
    run fnn-cuda "cd $P && nvcc -O3 -std=c++17 -arch=sm_80 -ccbin g++-13 probe5_cuda.cu -lcublas -o $O/probe5-cuda" || exit 1
    for round in 1 2 3; do
        run fnn-cuda "PROBE_R=9 $O/probe5-cuda" > "$O/probe5-r$round.csv" 2> "$O/probe5-r$round.err"
    done
    for round in 1 2 3; do
        for wg in 256 64 8; do
            out=$RES/gcc-tl$wg/round-$round; mkdir -p "$B/$out"
            run fnn-cuda "$(wheel omp-gcc14nv); python -m fnnbench.cli run --backend ompnn --device gpu --dtype float --workload mnist-512-256 --batch 256 --epochs 3 --repeat 3 --warmup 1 --seed 1 --out $out --option workgroup_size=$wg" >> "$O/gcc.log" 2>&1
        done
    done
    run fnn-cuda "gcc -O2 -shared -fPIC scripts/wallprof/wallprof.c -o $RES/wallprof.so" || exit 1
    perturb() { # perturb <image> <wheel> <backend> <tag> <epochs> <period-us> <round> [option]
        local img=$1 w=$2 be=$3 tag=$4 ep=$5 p=$6 k=$7 opt=""; [ -n "${8:-}" ] && opt="--option $8"
        run "$img" "$(wheel "$w"); python scripts/wallprof/wallprof_run.py --backend $be --epochs $ep --delay-ms 300 --period-us $p --max-samples 400000 --discard --lib $RES/wallprof.so --out $RES/perturb/$tag-p$p-r$k.dump $opt" >> "$O/perturb.log" 2>&1
    }
    mkdir -p "$O/perturb"
    for k in 1 2; do
        for p in 2000 400 200 100; do
            perturb fnn-cuda cuda cudann cuda-inorder 100 "$p" "$k" queue=in_order
            perturb fnn-sycl sycl syclnn dpcpp 50 "$p" "$k"
        done
    done
    echo 'overlap probe done' ;;
wallprof-sweep)  # the same stacks as parameters vary: batch, CUP, width; and the stream probe with its arms reversed
    extra_dir wallprof-sweep; O=$B/$RES; P=$B/scripts/launch-probe; extra_policy || exit 1
    run fnn-cuda "gcc -O2 -shared -fPIC scripts/wallprof/wallprof.c -o $RES/wallprof.so && cd $P && nvcc -O3 -std=c++17 -arch=sm_80 -ccbin g++-13 probe5_cuda.cu -lcublas -o $O/probe5-cuda" || exit 1
    nvidia-smi --query-gpu=timestamp,clocks.gr,clocks.mem --format=csv -l 1 > "$O/probe5-clocks.csv" 2>&1 &
    smi=$!
    for round in 1 2; do
        run fnn-cuda "PROBE_R=9 PROBE_REVERSE=1 $O/probe5-cuda" > "$O/probe5-rev-r$round.csv" 2> "$O/probe5-rev-r$round.err"
    done
    kill $smi 2>/dev/null
    one() { # one <image> <wheel> <backend> <tag> <workload> <samples> <batch> <warm> <epochs> <delay-ms> <round> [option]
        local img=$1 w=$2 be=$3 tag=$4 wl=$5 ns=$6 bs=$7 warm=$8 ep=$9 dl=${10} k=${11} opt=""
        [ -n "${12:-}" ] && opt="--option ${12}"
        local out=$RES/$tag-r$k
        [ -s "$B/$out.stacks.json" ] && return 0
        run "$img" "$(wheel "$w"); python scripts/wallprof/wallprof_run.py --backend $be --workload $wl --samples $ns --batch $bs --warm $warm --epochs $ep --delay-ms $dl --lib $RES/wallprof.so --out $out.dump $opt && python scripts/wallprof/wallprof_report.py $out.dump"
    }
    for k in 1 2; do
        #          tag      workload            samples batch warm CUDA DPC++ delay
        while read -r tag wl ns bs warm ec ed dl; do
            one fnn-cuda cuda cudann "cuda-$tag" "$wl" "$ns" "$bs" "$warm" "$ec" "$dl" "$k" queue=in_order
            one fnn-sycl sycl syclnn "dpcpp-$tag" "$wl" "$ns" "$bs" "$warm" "$ed" "$dl" "$k"
        done <<'CONDITIONS'
mnist-b64 mnist-512-256 60000 64 6 50 20 1000
mnist-b1024 mnist-512-256 60000 1024 40 350 150 1000
cup cup 1000 40 200 1200 500 1000
w256 sweep-w256-d4-b256 65536 256 12 100 40 1000
w1024 sweep-w1024-d4-b256 65536 256 6 25 25 1000
w4096 sweep-w4096-d4-b256 65536 256 2 3 3 2000
CONDITIONS
    done
    echo 'wallprof sweep done' ;;
build-sycl-discard)  # front-end: a scratch syclnn whose in-order queue discards events when SYCLNN_DISCARD is set
    run fnn-sycl "export CC=clang CXX=clang++ SYCLNN_TARGETS='spir64;nvidia_gpu_sm_80' SYCLNN_ONEMATH_ROOT=/opt/onemath CMAKE_BUILD_PARALLEL_LEVEL=16; $(pip_target sycl-discard $R/syclnn-discard sycl-discard)" ;;
discard-probe)  # one short job: what the profile suggests, with and without events, alternated
    extra_dir discard-probe; O=$B/$RES; extra_policy || exit 1
    run fnn-cuda "gcc -O2 -shared -fPIC scripts/wallprof/wallprof.c -o $RES/wallprof.so" || exit 1
    one() { # one <tag> <round> <prefix> [option...]
        local tag=$1 k=$2 pre=$3 opts=""; shift 3
        for o in "$@"; do opts="$opts --option $o"; done
        [ -s "$O/$tag-r$k.stacks.json" ] && return 0
        run fnn-sycl "$(wheel sycl-discard); $pre python scripts/wallprof/wallprof_run.py --backend syclnn --epochs 100 --lib $RES/wallprof.so --out $RES/$tag-r$k.dump $opts && python scripts/wallprof/wallprof_report.py $RES/$tag-r$k.dump"
    }
    T="loss_reduction=false bias_gemv=false workgroup_size=256"
    for k in 1 2 3; do
        one events "$k" ""
        one discard "$k" "SYCLNN_DISCARD=1"
        one tuned "$k" "" $T
        one tuned-discard "$k" "SYCLNN_DISCARD=1" $T
    done
    run fnn-sycl "$(wheel sycl-discard); SYCLNN_DISCARD=1 python -m fnnbench.cli run --backend syclnn --device gpu --dtype float --workload mnist-512-256 --batch 256 --epochs 3 --repeat 3 --warmup 1 --seed 1 --out $RES/check" > "$O/check.log" 2>&1
    echo 'discard probe done' ;;
tuned-rows)  # harness rows of what the profile suggests, numerically checked, fresh processes, alternated
    extra_dir tuned; O=$B/$RES; extra_policy || exit 1
    T="--option loss_reduction=False --option bias_gemv=False"
    one() { # one <arm> <workload-tag> <round> <image> <wheel> <backend> <prefix> <run arguments>
        local arm=$1 wl=$2 k=$3 img=$4 w=$5 be=$6 pre=$7; shift 7
        local out=$RES/$arm/$wl/round-$k; mkdir -p "$B/$out"
        ls "$B/$out"/*.jsonl >/dev/null 2>&1 && return 0
        run "$img" "$(wheel "$w"); $pre python -m fnnbench.cli run --backend $be --device gpu --dtype float --seed 1 --warmup 1 $* --out $out" > "$B/$out/run.log" 2>&1 < /dev/null
    }
    M="--workload mnist-512-256 --batch 256 --epochs 3 --repeat 3 --samples 60000"
    C="--workload cup --batch 40 --epochs 5 --repeat 5 --samples 1000"
    for k in 1 2 3; do
        one cuda-default mnist "$k" fnn-cuda cuda cudann "" $M --option queue=in_order
        one dpcpp-default mnist "$k" fnn-sycl sycl syclnn "" $M
        one cuda-tuned mnist "$k" fnn-cuda cuda cudann "" $M --option queue=in_order $T
        one dpcpp-tuned mnist "$k" fnn-sycl sycl syclnn "" $M $T --option workgroup_size=256
        one dpcpp-tuned-discard mnist "$k" fnn-sycl sycl-discard syclnn "SYCLNN_DISCARD=1" $M $T --option workgroup_size=256
        [ "$k" = 3 ] && continue
        one cuda-default cup "$k" fnn-cuda cuda cudann "" $C --option queue=in_order
        one dpcpp-default cup "$k" fnn-sycl sycl syclnn "" $C
        one cuda-tuned cup "$k" fnn-cuda cuda cudann "" $C --option queue=in_order $T
        one dpcpp-tuned cup "$k" fnn-sycl sycl syclnn "" $C $T --option workgroup_size=256
    done
    echo 'tuned rows done' ;;
clang-wait)  # steady-state rows of ompnn 0.1.2 with clang: Options::wait auto against spin, fresh processes, alternated
    extra_dir clang-wait; O=$B/$RES
    one() { # one <wait> <round> <workload> <batch> <epochs> <samples>
        local out=$RES/$1/$3-b$4/round-$2; mkdir -p "$B/$out"
        ls "$B/$out"/*.jsonl >/dev/null 2>&1 && return 0
        run fnn-cuda "$(wheel omp-clang18wait); python -m fnnbench.cli run --backend ompnn --device gpu --dtype float --workload $3 --batch $4 --epochs $5 --repeat $5 --warmup 1 --seed 1 --samples $6 --option wait=$1 --out $out" > "$B/$out/run.log" 2>&1 < /dev/null
    }
    for k in 1 2 3; do
        arms='auto spin'; [ $((k % 2)) = 0 ] && arms='spin auto'
        for w in $arms; do
            one "$w" "$k" monk 40 5 124
            one "$w" "$k" cup 40 5 1000
            one "$w" "$k" mnist-512-256 256 3 60000
            one "$w" "$k" mnist-512-256 1024 3 60000
            one "$w" "$k" mnist-512-256 64 3 60000
        done
    done
    echo 'clang wait done' ;;
stat-confirm)
    extra_dir stat-confirm; extra_policy || exit 1
    rounds=${STAT_ROUNDS:-3}   # odd rounds: CUDA first; even rounds: DPC++ first, CUP options reversed
    { printf 'SLURM job %s on %s, partition %s\n' "${SLURM_JOB_ID:-none}" "$(hostname)" "${SLURM_JOB_PARTITION:-none}"
      printf 'NUMA prefix: %s\nrounds: %s\n' "${NUMA:-none (adopted default)}" "$rounds"; } > "$B/$RES/commands.txt"
    { nvidia-smi --query-gpu=name,driver_version,clocks.sm,clocks.mem,utilization.gpu,memory.used --format=csv
      nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv; } > "$B/$RES/gpu-start.txt" 2>&1
    run fnn-cuda "python --version; python -m pip show fnnbench fnn-testkit numpy" > "$B/$RES/versions.txt" 2>&1
    for round in $(seq 1 "$rounds"); do
        order='cudann syclnn'; cup='default in-order graph'
        [ $((round % 2)) = 0 ] && { order='syclnn cudann'; cup='graph in-order default'; }
        for batch in 256 1024; do for be in $order; do
            img=fnn-cuda; w=cuda; [ "$be" = syclnn ] && { img=fnn-sycl; w=sycl; }
            run "$img" "$(wheel "$w"); python scripts/stat-confirm.py --run $be-mnist-b$batch --round $round --out $RES" || true
            # the matched pair of the headline: CUDA with one in-order stream, next to its default
            [ "$be" = cudann ] && [ "$batch" = 256 ] && { run fnn-cuda "$(wheel cuda); python scripts/stat-confirm.py --run cudann-mnist-b256-in-order --round $round --out $RES" || true; }
        done; done
        for opt in $cup; do
            run fnn-cuda "$(wheel cuda); python scripts/stat-confirm.py --run cudann-cup-$opt --round $round --out $RES" || true
        done
    done
    run fnn-cuda "python scripts/stat-confirm.py --out $RES" ;;
sanitizer|sanitizer-check)  # sanitizer <tool>, one exclusive job each; check is a short shared job
    extra_dir sanitizer
    tool=${2:-memcheck}
    smoke=""; [ "$1" = sanitizer-check ] && smoke=--smoke
    san=/usr/local/cuda-12.9/compute-sanitizer/compute-sanitizer
    [ ! -s "$B/$RES/sanitizer-path.txt" ] || san=$(cat "$B/$RES/sanitizer-path.txt")
    run fnn-cuda "$(wheel cuda); export PYTHONDONTWRITEBYTECODE=1; test -x '$san' || { echo 'CUDA 12.9 compute-sanitizer unavailable; use scripts/fetch-sanitizer.py in this sandbox'; exit 1; }; python scripts/sanitizer-cases.py --tool $tool --sanitizer '$san' --out $RES $smoke" ;;
stream)
    extra_dir stream; extra_policy || exit 1
    printf 'NUMA prefix: %s\nnvcc -O3 -std=c++17 -arch=sm_80 -ccbin g++-13 scripts/stream-probe.cu -o %s/stream-probe\n%s/stream-probe > %s/stream.csv\n' "${NUMA:-none (adopted default)}" "$RES" "$RES" "$RES" > "$B/$RES/commands.txt"
    run fnn-cuda "nvcc --version; python --version; nvidia-smi --query-gpu=name,driver_version,clocks.sm,clocks.mem --format=csv" > "$B/$RES/versions.txt" 2>&1
    run fnn-cuda "nvcc -O3 -std=c++17 -arch=sm_80 -ccbin g++-13 scripts/stream-probe.cu -o $RES/stream-probe && $RES/stream-probe > $RES/stream.csv && python scripts/stream-table.py $RES" || exit 1
    kernel_roofline ;;
ncu-tiled)  # ncu-tiled <cuda|sycl|omp>; one <=30-minute exclusive job per wheel
    extra_dir ncu-tiled
    grep -qx allowed "$B/$RES/access-status.txt" || { echo 'run ncu-check successfully first'; exit 1; }
    NCU=$(cat "$B/$RES/ncu-path.txt")
    case ${2:-cuda} in
        cuda) img=fnn-cuda; w=cuda; be=cudann; label=cudann-nvcc; kinds='tiled auto' ;;
        sycl) img=fnn-sycl; w=sycl; be=syclnn; label=syclnn-dpcpp; kinds=tiled ;;
        omp) img=fnn-cuda; w=omp-clang18nv; be=ompnn; label=ompnn-clang18nv; kinds=tiled ;;
        *) echo 'expected cuda, sycl or omp'; exit 2 ;;
    esac
    run fnn-cuda "find /usr/local /opt/nvidia -name compute-sanitizer -o -name 'libnvToolsExt.so*' 2>/dev/null > $RES/node-tools.txt"
    for kind in $kinds; do
        run "$img" "$(wheel "$w"); python scripts/ncu-experiments.py --ncu $NCU --backend $be --label $label --blas $kind --out $RES" || exit 1
    done
    run fnn-cuda "python scripts/ncu-table.py $RES" || exit 1
    kernel_roofline ;;
ncu-check)
    extra_dir ncu-tiled
    NCU=$(find_ncu)
    {
        printf '# Nsight Compute availability and counter access\n\n'
        printf 'Command: `DATE=%s bash scripts/hpc-blocks.sh ncu-check` (gpu-shared, <=5 minutes).\n\n' "$DATE"
        printf '| check | result |\n|---|---|\n'
        printf '| node | %s |\n| ncu | %s |\n' "$(hostname)" "${NCU:-not found}"
    } > "$B/$RES/README.md"
    [ -n "$NCU" ] || { echo 'unavailable: ncu not found under /opt/nvidia' > "$B/$RES/access-status.txt"; exit 1; }
    printf '%s\n' "$NCU" > "$B/$RES/ncu-path.txt"
    "$NCU" --version > "$B/$RES/ncu-version.txt" 2>&1
    run fnn-cuda "$(wheel cuda); timeout 180s $NCU --set basic --launch-count 1 --profile-from-start off --target-processes all --force-overwrite -o $RES/access-check python scripts/nsys_epoch.py --backend cudann --blas tiled --samples 256 --epochs 1 --out $RES/access-driver.json" > "$B/$RES/access-check.log" 2>&1
    rc=$?
    if grep -q ERR_NVGPUCTRPERM "$B/$RES/access-check.log"; then
        echo 'denied: ERR_NVGPUCTRPERM; no privilege escalation or repeated counter attempts' > "$B/$RES/access-status.txt"
    elif [ "$rc" = 0 ] && [ -s "$B/$RES/access-check.ncu-rep" ]; then
        echo 'allowed' > "$B/$RES/access-status.txt"
    else
        echo "failed: exit $rc; inspect access-check.log" > "$B/$RES/access-status.txt"
    fi
    printf '| counter probe | %s |\n| tool version | %s |\n' "$(cat "$B/$RES/access-status.txt")" "$(tr '\n' ' ' < "$B/$RES/ncu-version.txt")" >> "$B/$RES/README.md"
    exit "$rc" ;;
build-omp-cpu)
    # Rebuild only: no pylib or GPU wheel is removed by this case.
    for cfg in "gcc14 g++-14" "clang18 clang++-18" "clang22 clang++-22"; do
        set -- $cfg
        run fnn-cuda "export CXX=$2 OMPNN_TARGET=cpu OMPNN_OFFLOAD_ARCH= OMPNN_BLAS=openblas OMPNN_BLAS_ROOT=/opt/openblas-openmp CMAKE_BUILD_PARALLEL_LEVEL=16; $(pip_target omp-$1 $R/ompnn omp-$1)" || exit 1
    done ;;
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
        run fnn-cuda "$(wheel omp-$w); $(ompblas $w) export PATH=$NVHPC/compilers/bin:\$PATH OMP_NUM_THREADS=$CORES $bind MKL_THREADING_LAYER=GNU; $(declare -f sw); FB='$FB'; RES=$RES; DRY='$DRY'
            sw e4_omp_cpu omp-cpu-$w
            if [ $w = gcc14 ]; then
                sw e7_tiled_gemm e7-tiled-omp-cpu-gcc14 --select device=cpu --select backend=ompnn
                sw e5_inference e5-infer-omp-cpu-gcc14 --select device=cpu --select backend=ompnn
                sw e5_breakdown e5-breakdown-omp-cpu-gcc14 --select device=cpu --select backend=ompnn
            fi"
    done ;;
sycl-cpu-threads-smt)
    # hardware threads (64 workers on 32 cores): only the scaling series, same variables as sycl-cpu
    run fnn-sycl "$(wheel sycl); export OMP_PROC_BIND=close OMP_PLACES=cores DPCPP_CPU_NUM_CUS=$CORES OMP_NUM_THREADS=$CORES; $(declare -f sw); FB='$FB'; RES=$RES; DRY='$DRY'
        sw e1_cpu_threads_smt sycl-cpu-threads" ;;
omp-cpu-threads-smt)
    for w in gcc14 clang22 clang18 gcc14mkl nvcppmc; do
        case $w in clang*) bind="OMP_PROC_BIND=false" ;; *) bind="OMP_PROC_BIND=close OMP_PLACES=cores" ;; esac
        run fnn-cuda "$(wheel omp-$w); $(ompblas $w) export PATH=$NVHPC/compilers/bin:\$PATH OMP_NUM_THREADS=$CORES $bind MKL_THREADING_LAYER=GNU; $(declare -f sw); FB='$FB'; RES=$RES; DRY='$DRY'
            sw e4_omp_cpu_threads_smt omp-cpu-$w"
    done ;;
omp-cpu-extras)
    for w in gcc14 clang22 clang18 gcc14mkl nvcppmc; do
        case $w in clang*) bind="OMP_PROC_BIND=false" ;; *) bind="OMP_PROC_BIND=close OMP_PLACES=cores" ;; esac
        run fnn-cuda "$(wheel omp-$w); $(ompblas $w) export PATH=$NVHPC/compilers/bin:\$PATH OMP_NUM_THREADS=$CORES $bind MKL_THREADING_LAYER=GNU; mkdir -p $RES; $FB peak --backend ompnn --device cpu --dtype float --size 4096 --out $RES/peaks-omp-$w"
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
# ---------------------------------------------------------------- NUMA pilot (DATE=numa-pilot-<date>)
# Replays the rows in $RES/sel-*.jsonl (picked from the 2026-09-17 pass) under one memory policy, to
# size the effect of page placement on node09 (8 NUMA nodes of 4 cores; the A30 hangs off one of them).
numa-pilot-cpu)   # numa-pilot-cpu <first-touch|interleave> <round>
    pol=$2; k=$3; pre=""; [ "$pol" = interleave ] && pre="numactl --interleave=all"
    log "NUMA pilot, CPU rows, policy $pol, round $k"
    run fnn-sycl "$(wheel sycl); export OMP_PROC_BIND=close OMP_PLACES=cores DPCPP_CPU_NUM_CUS=$CORES OMP_NUM_THREADS=$CORES; $pre $FB replay $RES/sel-dpcpp-syclnn.jsonl --out $RES/$pol-r$k/dpcpp"
    run fnn-cuda "$(wheel omp-gcc14); export OMP_NUM_THREADS=$CORES OMP_PROC_BIND=close OMP_PLACES=cores MKL_THREADING_LAYER=GNU; $pre $FB replay $RES/sel-omp-gcc14-ompnn.jsonl --out $RES/$pol-r$k/omp-gcc14"
    run fnn-acpp-cuda "$(wheel sycl-acpp); export OMP_NUM_THREADS=$CORES OMP_PROC_BIND=close OMP_PLACES=cores; $pre $FB replay $RES/sel-acpp-cpu-syclnn.jsonl --out $RES/$pol-r$k/acpp-cpu" 2>&1 | grep -v 'AdaptiveCpp Warning' ;;
numa-pilot-gpu)   # default (unbound, as the pass) against the process bound to the A30's node, ABAB
    gnode=$(cat /sys/bus/pci/drivers/nvidia/0000:*/numa_node 2>/dev/null | head -1); log "NUMA pilot, GPU rows; the A30 is on node $gnode"
    for k in 1 2; do for pol in default gpu-node; do
        pre=""; [ $pol = gpu-node ] && pre="numactl --cpunodebind=$gnode --membind=$gnode"
        run fnn-cuda "$(wheel cuda); $pre $FB replay $RES/sel-gpu-cudann.jsonl --out $RES/$pol-r$k/gpu"
        run fnn-sycl "$(wheel sycl); $pre $FB replay $RES/sel-gpu-syclnn.jsonl --out $RES/$pol-r$k/gpu"
    done; done ;;
status)
    for b in sycl-gpu cuda omp-gpu acpp-gpu sycl-cpu sycl-generic omp-cpu acpp-cpu; do
        echo "--- $b"; DRY=--dry-run "$0" $b 2>&1 | grep -E 'dry run:|Error|error' | sed 's/^/    /'
    done ;;
*) echo "usage: $0 build-wheels|build-nvcpp|parity-{sycl,cuda,omp-gpu,acpp,cpu}|<block>|<block>-extras|status" >&2; exit 2 ;;
esac
