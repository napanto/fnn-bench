"""Background sampler of CPU temperature and clock during the timed region.

Temperatures from hwmon (`k10temp` on AMD: Tctl/Tdie/Tccd*, `coretemp` on Intel: the
package sensors), the mean core clock from cpufreq's `scaling_cur_freq` (falls back
to /proc/cpuinfo). Sampling every 200 ms costs nothing measurable; every source is
best-effort and yields `None` when absent (containers usually see the host's hwmon).

Why: the Threadripper 2950X of ws-amd reaches its throttle point (Tdie 68 C, Tctl
95 C) within 20 s of a 16-thread load and froze once under the generic SYCL
BLAS sweep (2026-09-05, 18:22); the third-pass CPU rows are re-measured at a fixed
2.8 GHz and this sampler records the clock and temperature every row ran at."""
from __future__ import annotations

import glob
import os
import threading
import time
from typing import Any

INTERVAL_S = 0.2


def _read(path: str) -> float | None:
    try:
        with open(path) as f:
            return float(f.read().strip())
    except (OSError, ValueError):
        return None


def _temp_sources() -> dict[str, str]:
    """label -> temp*_input path for the CPU hwmon chip(s)."""
    out: dict[str, str] = {}
    for hw in glob.glob("/sys/class/hwmon/hwmon*"):
        try:
            name = open(os.path.join(hw, "name")).read().strip()
        except OSError:
            continue
        if name not in ("k10temp", "coretemp", "zenpower"):
            continue
        for inp in sorted(glob.glob(os.path.join(hw, "temp*_input"))):
            lab = inp.replace("_input", "_label")
            try:
                label = open(lab).read().strip()
            except OSError:
                label = os.path.basename(inp).replace("_input", "")
            if name == "coretemp" and not label.startswith("Package"):
                continue  # one sensor per package is enough
            out.setdefault(label if label not in out else f"{label}#{len(out)}", inp)
    return out


def _clock_paths() -> list[str]:
    return sorted(glob.glob("/sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_cur_freq"))


def _mean_mhz(paths: list[str]) -> float | None:
    if paths:
        vals = [v for v in (_read(p) for p in paths) if v is not None]
        return (sum(vals) / len(vals) / 1000.0) if vals else None
    try:  # /proc/cpuinfo fallback
        tot, n = 0.0, 0
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("cpu MHz"):
                    tot += float(line.split(":")[1])
                    n += 1
        return tot / n if n else None
    except (OSError, ValueError):
        return None


class CpuSampler:
    """`with CpuSampler() as s: ...; s.summary()`"""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.samples: list[tuple[float, dict[str, float | None], float | None]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._temps: dict[str, str] = {}
        self._clocks: list[str] = []

    def _loop(self) -> None:
        while not self._stop.is_set():
            t = time.perf_counter()
            temps = {k: (v / 1000.0 if v is not None else None) for k, v in ((k, _read(p)) for k, p in self._temps.items())}
            self.samples.append((t, temps, _mean_mhz(self._clocks)))
            self._stop.wait(INTERVAL_S)

    def __enter__(self) -> "CpuSampler":
        if not self.enabled:
            return self
        self._temps = _temp_sources()
        self._clocks = _clock_paths()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def summary(self) -> dict[str, Any] | None:
        if not self.samples:
            return None
        out: dict[str, Any] = {"samples": len(self.samples), "span_s": self.samples[-1][0] - self.samples[0][0]}
        for label in self._temps:
            vals = [s[1].get(label) for s in self.samples if s[1].get(label) is not None]
            if vals:
                out[f"{label}_c_mean"] = sum(vals) / len(vals)
                out[f"{label}_c_max"] = max(vals)
        mhz = [s[2] for s in self.samples if s[2] is not None]
        if mhz:
            out["mhz_mean"] = sum(mhz) / len(mhz)
            out["mhz_min"] = min(mhz)
            out["mhz_max"] = max(mhz)
        return out
