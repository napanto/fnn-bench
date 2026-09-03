#!/usr/bin/env bash
# Cross-validate the in-library profilers against rocprofv3 on the RX 7900 XTX.
#
# Each backend trains mnist-512-256 (8192 samples, batch 256, float) on the GPU
# twice in separate processes: A = 1 epoch, B = 2 epochs.  rocprofv3 sees every
# kernel of the process (including rocBLAS's one-time kernel loading in the
# first epoch), so the second epoch's device time is B - A per kernel family,
# and it is compared with the library's own profile of exactly that epoch
# (profile reset after the first epoch).  Output: results/ws-amd/rocprof/<backend>.txt
#
#   distrobox enter fnn-rocm -- bash fnn-bench/scripts/rocprof-crosscheck.sh
set -uo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
P=$HOME/.local/opt/fnn-rocm
OUT=$HERE/results/ws-amd/rocprof
TMP=/tmp/rocprof-xcheck
mkdir -p "$OUT"
rm -rf "$TMP"

run_backend() {
    local backend=$1 venv=$2
    export PATH=$P/$venv/bin:$PATH
    for epochs in 1 2; do
        rocprofv3 --kernel-trace --memory-copy-trace --stats --output-format csv -d "$TMP/$backend-$epochs" -o run -- python - "$backend" "$epochs" <<'EOF' 2>&1 | grep -E "^PROFILE" > "$TMP/$backend-$epochs.profile"
import sys, numpy as np
from fnn_testkit import datasets
from fnn_testkit.reference import mlp, Momentum
from fnn_testkit.backends import NetworkAdapter
backend, epochs = sys.argv[1], int(sys.argv[2])
X, Y = datasets.mnist("train", limit=8192)
spec = mlp([784, 512, 256, 10], "relu", "sigmoid", learning_rate=0.1, momentum=Momentum("classical", 0.9))
net = NetworkAdapter(backend, spec, dtype="float", device="gpu", seed=1, profile=True)
X = X.astype(np.float32); Y = Y.astype(np.float32)
net.train(X, Y, 256, 1)
net.reset_profile()
if epochs == 2:
    net.train(X, Y, 256, 1)
print("PROFILE", {k: v for k, v in net.profile.items() if k.endswith("_ns") or k == "launches"})
EOF
    done
    python - "$backend" "$TMP" <<'EOF' | tee "$OUT/$backend.txt"
import ast, csv, glob, sys, collections
backend, tmp = sys.argv[1], sys.argv[2]
def kernels(run):
    f = glob.glob(f"{tmp}/{backend}-{run}/**/*kernel_stats.csv", recursive=True)
    g = collections.Counter()
    for r in csv.DictReader(open(f[0])):
        name, ns = r["Name"], float(r["TotalDurationNs"])
        low = name.lower()
        if "gemv" in low:
            fam = "biasgrad(gemv)"
        elif "cijk" in low or "gemm" in low or "rocblas" in low or "tensile" in low:
            fam = "gemm"
        elif "activate" in low: fam = "act"
        elif "output_delta" in low or "loss" in low: fam = "loss"
        elif "hidden_delta" in low: fam = "delta"
        elif "bias_grad" in low: fam = "biasgrad"
        elif "update" in low: fam = "update"
        elif "fill" in low or "gather" in low or "copy" in low: fam = "other"
        else: fam = "other:" + name[:50]
        g[fam] += ns
    return g
def copies(run):
    f = glob.glob(f"{tmp}/{backend}-{run}/**/*memory_copy_stats.csv", recursive=True)
    g = collections.Counter()
    if f:
        for r in csv.DictReader(open(f[0])):
            g[r["Name"]] += float(r["TotalDurationNs"])
    return g
k1, k2 = kernels(1), kernels(2)
c1, c2 = copies(1), copies(2)
prof = ast.literal_eval(open(f"{tmp}/{backend}-2.profile").read().split("PROFILE", 1)[1].strip())
print(f"== {backend}: second epoch, rocprofv3 (B - A) vs in-library profile  [ms]")
print(f"{'family':28s} {'rocprof':>10s} {'profile':>10s} {'ratio':>7s}")
rows = {"gemm": prof["gemm_ns"], "act": prof["act_ns"], "delta": prof["delta_ns"], "loss": prof["loss_ns"],
        "update": prof["update_ns"], "biasgrad": prof["biasgrad_ns"], "other": prof["other_ns"]}
total_r = 0.0
for fam in sorted(set(list(k2) + list(rows))):
    r = (k2.get(fam, 0.0) - k1.get(fam, 0.0)) / 1e6
    base = fam.split("(")[0].split(":")[0]
    p = rows.get(base, 0.0) / 1e6 if fam == base or fam.startswith("biasgrad") else 0.0
    total_r += r
    ratio = f"{r / p:7.2f}" if p else "      -"
    print(f"{fam:28s} {r:10.3f} {p:10.3f} {ratio}")
copy_r = sum(c2.values()) - sum(c1.values())
print(f"{'memcpy (all)':28s} {copy_r / 1e6:10.3f} {(prof['h2d_ns'] + prof['d2h_ns']) / 1e6:10.3f}")
print(f"{'kernels total':28s} {total_r:10.3f} {sum(rows.values()) / 1e6:10.3f}   epoch wall {prof['wall_ns'] / 1e6:.1f} ms, launches {prof['launches']}")
EOF
}

run_backend cudann venv
run_backend syclnn venv
run_backend ompnn venv-amdclang
echo "written to $OUT"
