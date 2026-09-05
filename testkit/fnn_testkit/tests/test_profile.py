"""The per-phase profiler exposed by every compiled backend."""

import math

import pytest

from fnn_testkit import datasets
from fnn_testkit.reference import mlp

pytestmark = pytest.mark.compiled_only

PHASES = ("h2d", "d2h", "gemm", "act", "delta", "biasgrad", "update", "loss", "reg")
KEYS = tuple(f"{p}_ns" for p in PHASES) + ("wall_ns", "launches", "bytes_h2d", "bytes_d2h", "epochs", "batches")


@pytest.fixture
def graph_mode(backend_options):
    return str(backend_options.get("queue", "")).lower() == "graph"


def test_profile_shape_and_counts(make_net, graph_mode):
    X, T = datasets.synthetic(100, 16, 4, seed=1)
    spec = mlp([16, 32, 4], "relu", "sigmoid", learning_rate=0.01)
    net = make_net(spec, seed=1, options={"profile": True})
    net.train(X, T, 32, 3)  # 4 batches per epoch (32, 32, 32, 4)
    p = net.profile
    for k in KEYS:
        assert k in p, k
        assert p[k] >= 0, (k, p[k])
    assert p["epochs"] == 3
    assert p["batches"] == 3 * math.ceil(100 / 32)
    assert p["wall_ns"] > 0
    if graph_mode:
        # a replayed CUDA Graph is one launch; per-phase events are not recorded inside a capture
        assert p["other_ns"] > 0
    else:
        assert p["gemm_ns"] > 0 and p["update_ns"] > 0 and p["act_ns"] > 0
    assert p["launches"] > 0
    device_total = sum(p[f"{ph}_ns"] for ph in PHASES)
    assert device_total > 0
    net.reset_profile()
    p = net.profile
    assert all(p[k] == 0 for k in KEYS)


def test_profile_is_zero_when_disabled(make_net):
    """Device phases and launch counts need the event profiler; the per-epoch wall
    clock, the epoch/batch counters and the call wall time are always recorded."""
    X, T = datasets.synthetic(50, 8, 2, seed=2)
    spec = mlp([8, 8, 2], learning_rate=0.01)
    net = make_net(spec, seed=1, options={"profile": False})
    net.train(X, T, 25, 2)
    p = net.profile
    always = {"epochs", "batches", "wall_ns", "epoch_wall_ns"}
    assert all(p[k] == 0 for k in KEYS if k not in always)
    assert p["epochs"] == 2 and p["batches"] == 4 and p["wall_ns"] > 0
    assert len(p["epoch_wall_ns"]) == 2 and all(v > 0 for v in p["epoch_wall_ns"])


def test_profile_covers_predict(make_net):
    X, T = datasets.synthetic(64, 8, 2, seed=2)
    spec = mlp([8, 8, 2], learning_rate=0.01)
    net = make_net(spec, seed=1, options={"profile": True})
    net.predict(X)
    p = net.profile
    assert p["gemm_ns"] > 0 and p["act_ns"] > 0
    assert p["epochs"] == 0 and p["batches"] == 0
