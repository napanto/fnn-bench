"""Every optimiser × momentum × regularisation combination for three epochs on a
tiny set whose last batch is a remainder batch (N = 10, B = 4)."""

import itertools

import numpy as np
import pytest

from fnn_testkit.plugin import assert_close, random_params
from fnn_testkit.reference import AdaptiveLR, Momentum, Regularization, mlp

STRATEGIES = ["constant", "linear_decay", "adagrad", "rmsprop", "adam"]
MOMENTA = ["disabled", "classical"]
REGS = ["disabled", "l1", "l2", "elastic_net"]


@pytest.mark.parametrize("strategy,momentum,reg", list(itertools.product(STRATEGIES, MOMENTA, REGS)))
def test_three_epochs_match_reference(make_net, make_ref, tol, rng, strategy, momentum, reg):
    spec = mlp(
        [3, 5, 2],
        hidden="tanh",
        output="sigmoid",
        learning_rate=0.1,
        regularization=Regularization(reg, 0.01, 0.02),
        adaptive=AdaptiveLR(strategy, 1e-8, 0.9, 0.999, final_lr=0.01),
        momentum=Momentum(momentum, 0.9),
    )
    Ws, bs = random_params(spec, rng)
    X, T = rng.uniform(-1, 1, (10, 3)), rng.uniform(0, 1, (10, 2))
    net = make_net(spec, initial_weights=Ws, initial_biases=bs)
    ref = make_ref(spec, initial_weights=Ws, initial_biases=bs)
    losses = net.train(X, T, batch_size=4, max_epochs=3)
    expected = ref.train(X, T, batch_size=4, max_epochs=3)
    assert_close(losses, expected, tol, "loss")
    for l in range(2):
        assert_close(net.weights[l], ref.W[l], tol, f"W{l}")
        assert_close(net.biases[l], ref.b[l], tol, f"b{l}")


def test_adam_state_persists_across_train_calls(make_net, make_ref, tol, rng):
    """Moments survive a second train() call, the step counter restarts — as in
    network.hpp."""
    spec = mlp([3, 4, 1], learning_rate=0.05, adaptive=AdaptiveLR("adam"))
    Ws, bs = random_params(spec, rng)
    X, T = rng.uniform(-1, 1, (8, 3)), rng.uniform(0, 1, (8, 1))
    net = make_net(spec, initial_weights=Ws, initial_biases=bs)
    ref = make_ref(spec, initial_weights=Ws, initial_biases=bs)
    for _ in range(2):
        losses = net.train(X, T, 4, 2)
        expected = ref.train(X, T, 4, 2)
        assert_close(losses, expected, tol)
    assert_close(net.weights[1], ref.W[1], tol)
