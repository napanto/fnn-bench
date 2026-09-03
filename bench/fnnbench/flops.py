"""Analytic FLOP / byte model of one training step and of inference.

For a network with layer sizes n_0 … n_L and a batch of B samples
(one multiply-add = 2 FLOP):

* forward GEMMs            2·B·Σ_l n_l·n_{l+1}
* backward delta GEMMs     2·B·Σ_{l≥1} n_l·n_{l+1}   (no delta for the input layer)
* weight-gradient GEMMs    2·B·Σ_l n_l·n_{l+1}
* element-wise work        O(B·Σ_{l≥1} n_l)  (bias+activation, deltas, loss)
* update work              O(Σ_l n_l·n_{l+1} + n_{l+1})  per step, independent of B

`gemm_flops` counts only the GEMMs (what the BLAS library executes);
`total_flops` adds the element-wise and update work with a small per-element
constant, so that the "hand-written kernels" share can be reported as well.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class StepFlops:
    forward_gemm: float
    delta_gemm: float
    grad_gemm: float
    elementwise: float
    update: float

    @property
    def gemm(self) -> float:
        return self.forward_gemm + self.delta_gemm + self.grad_gemm

    @property
    def total(self) -> float:
        return self.gemm + self.elementwise + self.update


def weights(sizes: Sequence[int]) -> int:
    return sum(sizes[i] * sizes[i + 1] for i in range(len(sizes) - 1))


def params(sizes: Sequence[int]) -> int:
    return weights(sizes) + sum(sizes[1:])


def train_step(sizes: Sequence[int], batch: int, act_cost: int = 8, update_cost: int = 12) -> StepFlops:
    """FLOPs of one mini-batch step (forward + backward + update).

    act_cost: FLOPs charged per activation element (bias add, f, f', delta
    products, loss) — a coarse constant, transcendental functions cost more.
    update_cost: FLOPs charged per parameter per step (Adam-like worst case)."""
    w = weights(sizes)
    fwd = 2.0 * batch * w
    delta = 2.0 * batch * sum(sizes[i] * sizes[i + 1] for i in range(1, len(sizes) - 1))
    grad = 2.0 * batch * w
    elem = float(act_cost) * batch * sum(sizes[1:])
    upd = float(update_cost) * params(sizes)
    return StepFlops(fwd, delta, grad, elem, upd)


def train_epoch(sizes: Sequence[int], n_samples: int, batch: int) -> StepFlops:
    """Sum over the batches of one epoch (the last batch may be smaller)."""
    full, rem = divmod(n_samples, batch)
    a = train_step(sizes, batch)
    parts = [(full, a)]
    if rem:
        parts.append((1, train_step(sizes, rem)))
    return StepFlops(*(sum(k * getattr(s, f) for k, s in parts) for f in ("forward_gemm", "delta_gemm", "grad_gemm", "elementwise", "update")))


def inference(sizes: Sequence[int], n_samples: int, act_cost: int = 4) -> StepFlops:
    w = weights(sizes)
    return StepFlops(2.0 * n_samples * w, 0.0, 0.0, float(act_cost) * n_samples * sum(sizes[1:]), 0.0)


def gemm_bytes(m: int, n: int, k: int, itemsize: int) -> int:
    """Minimum traffic of one GEMM C[m×n] = A[m×k]·B[k×n] (read A, B; write C)."""
    return (m * k + k * n + m * n) * itemsize


def arithmetic_intensity(m: int, n: int, k: int, itemsize: int) -> float:
    return 2.0 * m * n * k / gemm_bytes(m, n, k, itemsize)


def step_intensity(sizes: Sequence[int], batch: int, itemsize: int) -> float:
    """Aggregate arithmetic intensity (FLOP/byte) of the GEMMs of one step, used
    to place a run on the roofline."""
    flops = 0.0
    byts = 0
    for i in range(len(sizes) - 1):
        n_in, n_out = sizes[i], sizes[i + 1]
        flops += 2.0 * n_out * batch * n_in  # forward
        byts += gemm_bytes(n_out, batch, n_in, itemsize)
        if i >= 1:  # delta
            flops += 2.0 * n_in * batch * n_out
            byts += gemm_bytes(n_in, batch, n_out, itemsize)
        flops += 2.0 * n_out * n_in * batch  # weight gradient
        byts += gemm_bytes(n_out, n_in, batch, itemsize)
    return flops / byts


def inference_intensity(sizes: Sequence[int], batch: int, itemsize: int) -> float:
    """Aggregate arithmetic intensity of the forward GEMMs only (inference rows)."""
    flops = 0.0
    byts = 0
    for i in range(len(sizes) - 1):
        n_in, n_out = sizes[i], sizes[i + 1]
        flops += 2.0 * n_out * batch * n_in
        byts += gemm_bytes(n_out, batch, n_in, itemsize)
    return flops / byts
