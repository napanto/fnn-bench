"""Cross-implementation parity: identical initial weights, three epochs on
real-shaped workloads, per-epoch loss and final weights versus the oracle."""

import numpy as np
import pytest

from fnn_testkit import datasets
from fnn_testkit.plugin import assert_close, random_params
from fnn_testkit.reference import AdaptiveLR, Momentum, Regularization, mlp

CASES = {
    "monk1": dict(
        spec=mlp([17, 4, 1], "tanh", "sigmoid", learning_rate=0.01,
                 regularization=Regularization("l2", 0.001, 0.001), adaptive=AdaptiveLR("adam")),
        data=lambda: datasets.monk(1, "train"),
        batch=40,
        epochs=3,
    ),
    "xor": dict(
        spec=mlp([2, 4, 1], "tanh", "sigmoid", learning_rate=0.5, momentum=Momentum("classical", 0.9)),
        data=datasets.xor,
        batch=4,
        epochs=50,
    ),
    "cup-like": dict(
        spec=mlp([12, 16, 16, 3], "tanh", "disabled", learning_rate=0.001, adaptive=AdaptiveLR("adam"),
                 regularization=Regularization("l2", 0.0, 1e-4)),
        data=lambda: datasets.cup_like(200, seed=3),
        batch=40,
        epochs=3,
    ),
    "relu-rmsprop": dict(
        spec=mlp([8, 32, 32, 4], "relu", "sigmoid", learning_rate=0.001, adaptive=AdaptiveLR("rmsprop", beta1=0.9)),
        data=lambda: datasets.synthetic(130, 8, 4, seed=5),
        batch=32,
        epochs=3,
    ),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_parity_with_reference(make_net, make_ref, tol, rng, case):
    c = CASES[case]
    X, T = c["data"]()
    Ws, bs = random_params(c["spec"], rng)
    net = make_net(c["spec"], initial_weights=Ws, initial_biases=bs)
    ref = make_ref(c["spec"], initial_weights=Ws, initial_biases=bs)
    losses = net.train(X, T, c["batch"], c["epochs"])
    expected = ref.train(X, T, c["batch"], c["epochs"])
    assert_close(losses, expected, tol, f"{case}: loss")
    for l, (W, b) in enumerate(zip(ref.W, ref.b)):
        assert_close(net.weights[l], W, tol, f"{case}: W{l}")
        assert_close(net.biases[l], b, tol, f"{case}: b{l}")
    assert_close(net.predict(X[:16]), ref.predict(X[:16]), tol, f"{case}: predict")


def test_shuffle_permutation_matches_reference(make_net, make_ref, tol, rng):
    """With shuffle on, both sides draw the same std::mt19937 Fisher–Yates permutation."""
    spec = mlp([3, 5, 2], learning_rate=0.1)
    Ws, bs = random_params(spec, rng)
    X, T = datasets.synthetic(23, 3, 2, seed=9)
    net = make_net(spec, initial_weights=Ws, initial_biases=bs, options={"shuffle": True, "shuffle_seed": 11})
    ref = make_ref(spec, initial_weights=Ws, initial_biases=bs, shuffle=True, shuffle_seed=11)
    got = net.train(X, T, 5, 3)
    exp = ref.train(X, T, 5, 3)
    unshuffled = make_ref(spec, initial_weights=Ws, initial_biases=bs).train(X, T, 5, 3)
    assert not np.allclose(exp, unshuffled)  # the shuffle changes the trajectory ...
    assert_close(got, exp, tol)  # ... identically on both sides
