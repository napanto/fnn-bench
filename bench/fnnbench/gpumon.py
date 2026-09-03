"""Background sampler of GPU utilisation, power and clock during the timed region.

AMD: sysfs (`gpu_busy_percent`, hwmon `power1_average` in uW, `freq1_input` in Hz)
of the discrete card (largest power cap). NVIDIA: `nvidia-smi --query-gpu ... -lms`.
Sampling every 200 ms costs nothing measurable; everything is best-effort and a
missing source yields `None` (the CPU package energy needs root for RAPL and is
not sampled)."""
from __future__ import annotations

import glob
import os
import subprocess
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


def _amd_card() -> dict[str, str] | None:
    best = None
    for dev in glob.glob("/sys/class/drm/card*/device"):
        if not os.path.exists(os.path.join(dev, "gpu_busy_percent")):
            continue
        hw = glob.glob(os.path.join(dev, "hwmon", "hwmon*"))
        cap = _read(os.path.join(hw[0], "power1_cap")) if hw else None
        cand = {"busy": os.path.join(dev, "gpu_busy_percent"),
                "power": os.path.join(hw[0], "power1_average") if hw else "",
                "clock": os.path.join(hw[0], "freq1_input") if hw else "",
                "cap_w": (cap or 0) / 1e6}
        if best is None or cand["cap_w"] > best["cap_w"]:
            best = cand
    return best


class GpuSampler:
    """`with GpuSampler(enabled) as s: ...; s.summary()`"""

    def __init__(self, enabled: bool = True, ordinal: int | None = None):
        self.enabled = enabled
        self.ordinal = ordinal
        self.samples: list[tuple[float, float | None, float | None, float | None]] = []
        self.source: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._proc: subprocess.Popen | None = None

    # ---- sources
    def _amd_loop(self, card: dict[str, str]) -> None:
        while not self._stop.is_set():
            t = time.perf_counter()
            busy = _read(card["busy"])
            p = _read(card["power"]) if card["power"] else None
            c = _read(card["clock"]) if card["clock"] else None
            self.samples.append((t, busy, p / 1e6 if p is not None else None, c / 1e6 if c is not None else None))
            self._stop.wait(INTERVAL_S)

    def _nvidia_loop(self) -> None:
        cmd = ["nvidia-smi", "--query-gpu=utilization.gpu,power.draw,clocks.sm", "--format=csv,noheader,nounits",
               "-lms", str(int(INTERVAL_S * 1000))]
        if self.ordinal is not None:
            cmd += ["-i", str(self.ordinal)]
        try:
            self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        except OSError:
            return
        assert self._proc.stdout is not None
        for line in self._proc.stdout:
            if self._stop.is_set():
                break
            parts = [x.strip() for x in line.split(",")]
            try:
                self.samples.append((time.perf_counter(), float(parts[0]), float(parts[1]), float(parts[2])))
            except (ValueError, IndexError):
                pass

    # ---- context manager
    def __enter__(self) -> "GpuSampler":
        if not self.enabled:
            return self
        card = _amd_card()
        if card is not None:
            self.source = "sysfs-amdgpu"
            self._thread = threading.Thread(target=self._amd_loop, args=(card,), daemon=True)
        elif any(os.access(os.path.join(p, "nvidia-smi"), os.X_OK) for p in os.environ.get("PATH", "").split(os.pathsep)):
            self.source = "nvidia-smi"
            self._thread = threading.Thread(target=self._nvidia_loop, daemon=True)
        else:
            return self
        self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._stop.set()
        if self._proc is not None:
            try:
                self._proc.terminate()
            except OSError:
                pass
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def summary(self) -> dict[str, Any] | None:
        if not self.samples:
            return None
        ts = [s[0] for s in self.samples]
        busy = [s[1] for s in self.samples if s[1] is not None]
        power = [s[2] for s in self.samples if s[2] is not None]
        clock = [s[3] for s in self.samples if s[3] is not None]
        span = ts[-1] - ts[0] if len(ts) > 1 else 0.0
        mean = lambda v: (sum(v) / len(v)) if v else None  # noqa: E731
        out = {"source": self.source, "samples": len(self.samples), "span_s": span,
               "util_pct_mean": mean(busy), "util_pct_max": max(busy) if busy else None,
               "power_w_mean": mean(power), "power_w_max": max(power) if power else None,
               "clock_mhz_mean": mean(clock)}
        # energy over the sampled span (trapezoid on the power samples)
        if len(power) > 1 and span > 0:
            pts = [(s[0], s[2]) for s in self.samples if s[2] is not None]
            e = 0.0
            for (t0, p0), (t1, p1) in zip(pts, pts[1:]):
                e += 0.5 * (p0 + p1) * (t1 - t0)
            out["energy_j"] = e
        return out
