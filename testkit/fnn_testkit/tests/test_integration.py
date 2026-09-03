"""End-to-end learning: the backend must actually learn, not only match numbers."""

import numpy as np
import pytest

from fnn_testkit import datasets
from fnn_testkit.reference import AdaptiveLR, Momentum, Regularization, mlp


def test_xor_converges(make_net):
    X, T = datasets.xor()
    spec = mlp([2, 4, 1], "tanh", "sigmoid", learning_rate=0.05, adaptive=AdaptiveLR("adam"))
    net = make_net(spec, seed=2024)
    losses = net.train(X, T, 4, 3000)
    assert losses[-1] < 1e-3, losses[-1]
    out = net.predict(X)
    assert np.array_equal((out > 0.5).astype(int), T.astype(int))


def test_monk1_accuracy(make_net):
    """17-8-1 tanh/sigmoid, batch 40, Adam lr 0.05: 100 % test accuracy for every
    seed tried with the oracle (the ML project's lr 0.01 + L2 plateaus at 82 %)."""
    Xtr, Ttr = datasets.monk(1, "train")
    Xte, Tte = datasets.monk(1, "test")
    spec = mlp([17, 8, 1], "tanh", "sigmoid", learning_rate=0.05, adaptive=AdaptiveLR("adam"))
    net = make_net(spec, seed=1)
    losses = net.train(Xtr, Ttr, 40, 500)
    assert np.all(np.isfinite(losses))
    pred = (net.predict(Xte) > 0.5).astype(int)
    acc = float((pred == Tte.astype(int)).mean())
    assert acc >= 0.97, acc


@pytest.mark.slow
def test_mnist_accuracy(make_net):
    Xtr, Ttr = datasets.mnist("train")
    Xte, Tte = datasets.mnist("test")
    spec = mlp([784, 128, 10], "relu", "sigmoid", learning_rate=0.1, momentum=Momentum("classical", 0.9))
    net = make_net(spec, seed=3)
    net.train(Xtr, Ttr, 64, 3)
    pred = net.predict(Xte, batch_size=1000).argmax(axis=1)
    acc = float((pred == Tte.argmax(axis=1)).mean())
    assert acc >= 0.95, acc
