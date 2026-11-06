"""Recorded conditions shared by the workstation experiment reports."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys


BACKGROUND_NOTE = ("other services were running on the host and in the VM, as in every "
                   "ws-nvidia pass; their CPU and GPU use is not controlled")
GPU_SHARING_NOTE = "The GPU was not dedicated to the benchmark."


def public_bench_status(text):
    """Keep measurement policy fields without host identifiers or service inventories."""
    lines = []
    for line in text.splitlines():
        stamp = re.match(r"^(?:\S+|the host) (\d{4}-\d{2}-\d{2}T\S+)", line)
        vm = re.match(r"^(?:\S+|the VM): (running \(booted\)\s+hugepage guard:.*)$", line)
        if stamp:
            lines.append("the host " + stamp[1])
        elif vm:
            lines.append("the VM: " + vm[1])
        elif re.match(r"^\s*(cpufreq:|policies:|scaling_cur_freq kHz|cpuidle:|"
                      r"idle states allowed by PM QoS:|kernel\.numa_balancing=|"
                      r"THP enabled=|node\d+ \d+M hugepages|ZFS ARC |zfs_arc_max param=)", line):
            lines.append(line.rstrip())
    return "\n".join(lines) + ("\n" if lines else "")


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
    os.environ["FNN_MACHINE_ID"] = machine
    data = {"timestamp": datetime.now(timezone.utc).isoformat(), "machine": machine,
            "kernel": platform.release(), "binding": binding, "niceness": os.getpriority(os.PRIO_PROCESS, 0),
            "cpu_cap_khz": read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq"),
            "boost": read("/sys/devices/system/cpu/cpufreq/boost"),
            "numa_balancing": read("/proc/sys/kernel/numa_balancing"), "sysinfo": sysinfo.collect(),
            "numactl": command(["numactl", "--show"]),
            "background_note": BACKGROUND_NOTE if machine == "ws-nvidia" else
                               "Other host services were running; their CPU and GPU use is not controlled."}
    if machine == "ws-nvidia":
        data["nvidia_parameters"] = read("/proc/driver/nvidia/params")
        data["gpu_sharing"] = GPU_SHARING_NOTE
        data["gpu_state"] = command(["nvidia-smi", "--query-gpu=name,driver_version,pstate,persistence_mode,memory.used,memory.total,power.limit", "--format=csv"])
        data["gpu_clocks"] = command(["nvidia-smi", "--query-gpu=name,driver_version,clocks.current.sm,clocks.current.memory,power.draw,temperature.gpu,utilization.gpu,utilization.memory", "--format=csv"])
        data["limits"] = "Pascal GPU; other GPU work remains possible. Bench mode only changes the listed runtime settings. No comparisons with the historical pass assume identical conditions."
    else:
        data["gpu_clocks"] = {str(p): read(p) for pattern in ("card*/device/pp_dpm_sclk", "card*/device/pp_dpm_mclk", "card*/device/power_dpm_force_performance_level") for p in Path("/sys/class/drm").glob(pattern)}
        data["gpu_hwmon"] = {str(p): read(p) for pattern in ("card*/device/hwmon/hwmon*/freq*_input", "card*/device/hwmon/hwmon*/temp*_input", "card*/device/hwmon/hwmon*/power*_average") for p in Path("/sys/class/drm").glob(pattern)}
        data["limits"] = "Same kernel, CPU cap and boost setting as the pass. These experiments prioritise counters. Timings and bandwidth describe this run only, not a programming-model speedup against the pass."
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
        result.append(d['background_note'].rstrip('.')+". "+d['limits']+"\n")
        result += ["\nGPU clock snapshot at collection start (not a clock lock or a time average):\n\n```text\n",
                   json.dumps(d.get('gpu_clocks'), indent=2), "\n```\n"]
        if d.get("gpu_sharing"):
            result.append("\n" + d["gpu_sharing"] + "\n")
    else:
        result.append("Conditions snapshot not available; do not interpret missing conditions as controlled.\n")
    for name in ("commands.txt", "versions.txt", "bench-mode-status.txt", "background-host.txt"):
        path = out / name
        if path.exists():
            content = path.read_text()
            if name == "bench-mode-status.txt":
                content = public_bench_status(content)
            elif name == "background-host.txt":
                content = BACKGROUND_NOTE + ".\n" + GPU_SHARING_NOTE + "\n"
            result += [f"\n{name}:\n\n```text\n", content, "\n```\n"]
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
