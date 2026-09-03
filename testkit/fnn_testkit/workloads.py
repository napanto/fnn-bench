"""Named workloads shared by the parity tests and the benchmark harness.

W1 ``monk``   17-4-1 tanh/sigmoid, batch 40, Adam, L2 — the ML project's real run.
W2 ``cup``    12-64-64-3 tanh/linear, batch 40 — ML-CUP-like regression (synthetic data).
W3 ``mnist``  784-1024-1024-10 (and 784-512-256-10) ReLU/sigmoid, SGD+momentum.
W4 ``sweep``  synthetic width/depth/batch grid.
W5 ``infer``  forward-only variants of W1–W3.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from . import datasets
from .reference import AdaptiveLR, Momentum, NetSpec, Regularization, StopCriteria, mlp


@dataclass(frozen=True)
class Workload:
    name: str
    spec: NetSpec
    dataset: str
    dataset_kw: dict = field(default_factory=dict)
    batch_size: int = 40
    epochs: int = 5
    description: str = ""

    def data(self) -> tuple[np.ndarray, np.ndarray]:
        return datasets.load(self.dataset, **self.dataset_kw)

    def with_layers(self, sizes, hidden="relu", output="sigmoid") -> "Workload":
        return replace(self, spec=mlp(sizes, hidden, output, learning_rate=self.spec.learning_rate,
                                      regularization=self.spec.regularization, adaptive=self.spec.adaptive,
                                      stop=self.spec.stop, momentum=self.spec.momentum))


def monk_workload() -> Workload:
    # ML project, MONK-1 best configuration: 17-4-1, tanh/sigmoid, batch 40, Adam lr 0.01, L2 1e-3
    spec = mlp(
        [17, 4, 1],
        "tanh",
        "sigmoid",
        learning_rate=0.01,
        regularization=Regularization("l2", 0.001, 0.001),
        adaptive=AdaptiveLR("adam", 1e-8, 0.9, 0.999),
        stop=StopCriteria("max_epochs"),
        momentum=Momentum("disabled"),
    )
    return Workload("monk", spec, "monk1", {"split": "train"}, batch_size=40, epochs=1000,
                    description="W1: MONK-1, ML-project configuration (launch-overhead regime)")


def cup_workload() -> Workload:
    spec = mlp(
        [12, 64, 64, 3],
        "tanh",
        "disabled",
        learning_rate=0.001,
        regularization=Regularization("l2", 0.0, 1e-4),
        adaptive=AdaptiveLR("adam"),
    )
    return Workload("cup", spec, "cup", {"n": 1000, "seed": 1}, batch_size=40, epochs=200,
                    description="W2: ML-CUP-like regression 12-64-64-3 (synthetic data)")


def mnist_workload(sizes=(784, 1024, 1024, 10), batch_size: int = 256) -> Workload:
    spec = mlp(
        list(sizes),
        "relu",
        "sigmoid",
        learning_rate=0.1,
        adaptive=AdaptiveLR("constant"),
        momentum=Momentum("classical", 0.9),
    )
    name = "mnist" if tuple(sizes) == (784, 1024, 1024, 10) else "mnist-" + "-".join(map(str, sizes[1:-1]))
    return Workload(name, spec, "mnist", {"split": "train"}, batch_size=batch_size, epochs=5,
                    description=f"W3: MNIST {'-'.join(map(str, sizes))} ReLU/sigmoid, SGD+momentum")


def sweep_workload(width: int, depth: int, batch_size: int, n: int = 65536, n_in: int = 256, n_out: int = 16) -> Workload:
    sizes = [n_in] + [width] * depth + [n_out]
    spec = mlp(sizes, "relu", "disabled", learning_rate=0.001, adaptive=AdaptiveLR("adam"))
    return Workload(f"sweep-w{width}-d{depth}-b{batch_size}", spec, "synthetic",
                    {"n": n, "n_in": n_in, "n_out": n_out, "seed": 7}, batch_size=batch_size, epochs=3,
                    description="W4: synthetic scaling sweep")


def registry() -> dict[str, Workload]:
    reg = {w.name: w for w in (monk_workload(), cup_workload(), mnist_workload(), mnist_workload((784, 512, 256, 10)))}
    for width in (256, 1024, 4096):
        for depth in (2, 4, 8):
            for batch in (32, 256, 4096):
                w = sweep_workload(width, depth, batch)
                reg[w.name] = w
    return reg


def get(name: str) -> Workload:
    return registry()[name]
