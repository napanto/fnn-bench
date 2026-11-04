"""Recorded conditions shared by the workstation experiment reports."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import subprocess
import sys


def command(argv):
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=15)
        return {"command": argv, "exit": p.returncode, "stdout": p.stdout.strip(), "stderr": p.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"command": argv, "error": str(exc)}


def read(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def snapshot(out, machine, binding):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bench"))
    from fnnbench import sysinfo
    data = {"timestamp": datetime.now(timezone.utc).isoformat(), "machine": machine,
            "kernel": platform.release(), "binding": binding, "niceness": os.getpriority(os.PRIO_PROCESS, 0),
            "cpu_cap_khz": read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq"),
            "boost": read("/sys/devices/system/cpu/cpufreq/boost"),
            "numa_balancing": read("/proc/sys/kernel/numa_balancing"), "sysinfo": sysinfo.collect(),
            "numactl": command(["numactl", "--show"]),
            "background_note": "Other host services were running; their CPU and GPU use is not controlled."}
    if machine == "ws-nvidia":
        data["nvidia_parameters"] = read("/proc/driver/nvidia/params")
        data["nvidia_smi"] = command(["nvidia-smi"])
        data["compute_processes"] = command(["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv"])
        data["gpu_clocks"] = command(["nvidia-smi", "--query-gpu=name,driver_version,clocks.current.sm,clocks.current.memory,power.draw,temperature.gpu,utilization.gpu,utilization.memory", "--format=csv"])
        data["limits"] = "Pascal GPU; other GPU work remains possible. Bench mode only changes the explicitly authorised runtime settings. No comparisons with the historical pass assume identical conditions."
    else:
        data["gpu_clocks"] = {str(p): read(p) for pattern in ("card*/device/pp_dpm_sclk", "card*/device/pp_dpm_mclk", "card*/device/power_dpm_force_performance_level") for p in Path("/sys/class/drm").glob(pattern)}
        data["gpu_hwmon"] = {str(p): read(p) for pattern in ("card*/device/hwmon/hwmon*/freq*_input", "card*/device/hwmon/hwmon*/temp*_input", "card*/device/hwmon/hwmon*/power*_average") for p in Path("/sys/class/drm").glob(pattern)}
        data["limits"] = "Kernel 7.3.0 differs from the 7.2.3 pass; we observed 20–50% slower/noisy GPU timings. These experiments prioritise counters. Timings and bandwidth describe this run only, not a programming-model speedup against the pass. CPU cap/boost were left unchanged."
    out.mkdir(parents=True, exist_ok=True)
    (out / "conditions.json").write_text(json.dumps(data, indent=2)+"\n")


def footer(out):
    result = ["\n## Conditions and limits\n\n"]
    path = out / "conditions.json"
    if path.exists():
        d = json.loads(path.read_text())
        cap = d.get("cpu_cap_khz") or "not exposed by guest"
        result.append(f"{d['machine']}; kernel `{d['kernel']}`; CPU maximum {cap} kHz; boost={d.get('boost')}; "
                      f"binding `{d['binding']}`; process nice={d['niceness']}; NUMA balancing={d.get('numa_balancing')}. "
                      "GPU policy and clock snapshots are in conditions.json and any runtime monitors. "
                      "A profiling policy named in the binding was temporary and released after the run.\n\n")
        result.append(d['background_note']+" "+d['limits']+"\n")
        result += ["\nGPU clock snapshot at collection start (not a clock lock or a time average):\n\n```text\n",
                   json.dumps(d.get('gpu_clocks'), indent=2), "\n```\n"]
    else:
        result.append("Conditions snapshot not available; do not interpret missing conditions as controlled.\n")
    for name in ("commands.txt", "versions.txt", "bench-mode-status.txt", "background-host.txt"):
        path = out / name
        if path.exists():
            result += [f"\n{name}:\n\n```text\n", path.read_text(), "\n```\n"]
    return "".join(result)


def write_report(out, text):
    (out / "README.md").write_text(text + footer(out))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--machine", choices=("ws-amd", "ws-nvidia"), required=True)
    parser.add_argument("--binding", required=True)
    a = parser.parse_args()
    snapshot(a.out, a.machine, a.binding)
