"""Capture everything needed to reproduce a run: host, CPU, GPU, driver, clocks,
governor, environment, container image, git SHAs, package versions."""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ENV_KEYS_PREFIXES = ("OMP_", "ONEAPI_", "SYCL_", "ACPP_", "HIP_", "CUDA_", "ROCR_", "HSA_", "MKL_", "OPENBLAS_",
                     "GOMP_", "KMP_", "LIBOMPTARGET", "FNN_", "NVIDIA_", "NVCOMPILER_", "DPCPP_", "LD_LIBRARY_PATH", "LD_PRELOAD", "HIPSYCL_",
)



def loaded_libs() -> list[str]:
    """BLAS / OpenMP / SYCL runtime libraries mapped in this process (provenance: e.g. which
    OpenBLAS variant the loader actually resolved)."""
    pats = ("openblas", "libblas", "mkl_rt", "mkl_sycl", "libgomp", "libomp", "libiomp", "libtbb", "libonemath", "libsycl", "libacpp", "hipsycl", "rocblas", "hipblas", "cublas", "libcudart", "libamdhip", "libOpenCL", "intelocl", "libur_")
    seen: list[str] = []
    try:
        for line in open("/proc/self/maps"):
            parts = line.split()
            if len(parts) < 6 or not parts[5].startswith("/"):
                continue
            path = parts[5]
            if any(p in path for p in pats) and path not in seen:
                seen.append(path)
    except OSError:
        pass
    return seen


def _run(cmd: list[str], timeout: float = 10) -> str | None:
    if shutil.which(cmd[0]) is None:
        return None
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout
    except Exception:
        return None


def _read(path: str) -> str | None:
    try:
        return Path(path).read_text().strip()
    except Exception:
        return None


def cpu_info() -> dict[str, Any]:
    info: dict[str, Any] = {"count": os.cpu_count()}
    txt = _read("/proc/cpuinfo") or ""
    m = re.search(r"model name\s*:\s*(.+)", txt)
    if m:
        info["model"] = m.group(1).strip()
    m = re.search(r"flags\s*:\s*(.+)", txt)
    if m:
        flags = set(m.group(1).split())
        info["isa"] = [f for f in ("avx", "avx2", "fma", "avx512f", "avx512_bf16", "amx_tile") if f in flags]
    lscpu = _run(["lscpu"])
    if lscpu:
        for key in ("Thread(s) per core", "Core(s) per socket", "Socket(s)", "NUMA node(s)", "L2 cache", "L3 cache"):
            m = re.search(re.escape(key) + r":\s*(.+)", lscpu)
            if m:
                info[key.lower().replace("(s)", "s").replace(" ", "_")] = m.group(1).strip()
    info["governor"] = _read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    info["max_khz"] = _read("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq")
    try:
        info["affinity"] = len(os.sched_getaffinity(0))  # of the calling (master) thread: pinned to one place under OMP_PROC_BIND
    except Exception:
        pass
    for f in ("/sys/fs/cgroup/cpuset.cpus.effective", "/sys/fs/cgroup/cpuset/cpuset.effective_cpus"):
        v = _read(f)
        if v:
            info["cpuset_effective"] = v  # what the process as a whole may use (the worker threads' places)
            break
    return info


def gpu_info() -> dict[str, Any]:
    out: dict[str, Any] = {}
    nv = _run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,clocks.max.sm,clocks.max.mem,clocks.sm,clocks.mem,power.limit,pstate,clocks_throttle_reasons.active",
               "--format=csv,noheader"])
    if nv:
        out["nvidia"] = [line.strip() for line in nv.strip().splitlines()]
    rocm = _run(["rocm-smi", "--showproductname", "--showclocks", "--json"], timeout=20)
    if rocm:
        try:
            out["rocm"] = json.loads(rocm)
        except Exception:
            out["rocm_raw"] = rocm[:2000]
    return out


def container_info() -> dict[str, Any]:
    info: dict[str, Any] = {}
    for key in ("FNN_IMAGE", "FNN_IMAGE_DIGEST", "container", "HOSTNAME"):
        if key in os.environ:
            info[key] = os.environ[key]
    env_file = _read("/run/.containerenv")
    if env_file is not None:
        info["containerenv"] = True
        for line in env_file.splitlines():
            if line.startswith(("image=", "imageid=", "name=")):
                k, _, v = line.partition("=")
                info[k] = v.strip('"')
    os_release = _read("/etc/os-release")
    if os_release:
        m = re.search(r'PRETTY_NAME="?([^"\n]+)', os_release)
        if m:
            info["os"] = m.group(1)
    return info


def git_sha(path: str | Path) -> dict[str, Any] | None:
    p = Path(path)
    if not (p / ".git").exists() and not (p / ".git").is_file():
        return None
    sha = _run(["git", "-C", str(p), "rev-parse", "HEAD"])
    dirty = _run(["git", "-C", str(p), "status", "--porcelain"])
    if sha is None:
        return None
    return {"sha": sha.strip(), "dirty": bool(dirty and dirty.strip())}


def environment() -> dict[str, str]:
    return {k: v for k, v in os.environ.items() if k.startswith(ENV_KEYS_PREFIXES)}


def packages() -> dict[str, str]:
    """Installed versions without importing the backends (importing a second
    SYCL/CUDA/OpenMP runtime into the benchmark process would perturb it)."""
    from importlib import metadata

    out = {}
    for name in ("numpy", "fnn-testkit", "fnnbench", "syclnn", "cudann", "ompnn"):
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    return out


def numa_info() -> dict[str, Any]:
    info: dict[str, Any] = {}
    status = _read("/proc/self/status") or ""
    for key in ("Cpus_allowed_list", "Mems_allowed_list"):
        m = re.search(key + r":\s*(\S+)", status)
        if m:
            info[key.lower()] = m.group(1)
    show = _run(["numactl", "--show"])
    if show:
        info["numactl_show"] = show.strip().splitlines()
    return info


def omp_max_threads() -> int | None:
    """What the OpenMP runtime of *this* process would use (ctypes, no import of a backend)."""
    import ctypes.util

    for lib in ("gomp", "omp", "iomp5"):
        path = ctypes.util.find_library(lib)
        if not path:
            continue
        try:
            handle = ctypes.CDLL(path)
            return int(handle.omp_get_max_threads())
        except Exception:
            continue
    v = os.environ.get("OMP_NUM_THREADS")
    return int(v.split(",")[0]) if v and v.split(",")[0].isdigit() else None


def _meminfo_total() -> int | None:
    m = re.search(r"MemTotal:\s*(\d+)", _read("/proc/meminfo") or "")
    return int(m.group(1)) if m else None


def collect(repos: dict[str, str] | None = None) -> dict[str, Any]:
    info = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "hostname": socket.gethostname(),
        "kernel": platform.release(),
        "python": platform.python_version(),
        "cpu": cpu_info(),
        "libs": loaded_libs(),
        "gpu": gpu_info(),
        "container": container_info(),
        "env": environment(),
        "numa": numa_info(),
        "omp_max_threads": omp_max_threads(),
        "meminfo_total_kb": _meminfo_total(),
        "packages": packages(),
        "git": {},
    }
    for name, path in (repos or {}).items():
        g = git_sha(path)
        if g:
            info["git"][name] = g
    return info
