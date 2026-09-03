"""Forward pass / predict."""

import numpy as np
import pytest

from fnn_testkit.plugin import Tolerance, assert_close, random_params
from fnn_testkit.reference import ACTIVATIONS, Layer, NetSpec, ReferenceNetwork, activate, mlp


@pytest.mark.parametrize("kind", ACTIVATIONS)
def test_single_layer_forward(make_net, dtype, rng, kind):
    spec = NetSpec((Layer(5, "disabled"), Layer(3, kind)), learning_rate=0.1)
    Ws, bs = random_params(spec, rng)
    net = make_net(spec, initial_weights=Ws, initial_biases=bs)
    X = rng.uniform(-2, 2, size=(7, 5))
    expected = activate(kind, X @ Ws[0].T + bs[0])
    tol = Tolerance(1e-12, 1e-14) if dtype == "double" else Tolerance(1e-5, 1e-6)
    assert_close(net.predict(X), expected, tol)


def test_deep_forward_matches_reference(make_net, make_ref, tol, rng):
    spec = mlp([6, 9, 7, 4], hidden="tanh", output="sigmoid", learning_rate=0.1)
    Ws, bs = random_params(spec, rng)
    net = make_net(spec, initial_weights=Ws, initial_biases=bs)
    ref = make_ref(spec, initial_weights=Ws, initial_biases=bs)
    X = rng.uniform(-1, 1, size=(11, 6))
    assert_close(net.predict(X), ref.predict(X), tol)


def test_predict_batching_is_transparent(make_net, dtype, rng):
    spec = mlp([4, 8, 2], learning_rate=0.1)
    Ws, bs = random_params(spec, rng)
    net = make_net(spec, initial_weights=Ws, initial_biases=bs)
    X = rng.uniform(-1, 1, size=(10, 4))
    full = net.predict(X)
    # GEMM kernels pick different blockings for different N, so allow rounding noise
    tol = Tolerance(1e-13, 1e-14) if dtype == "double" else Tolerance(1e-6, 1e-7)
    for b in (1, 3, 4, 10, 64):
        assert_close(net.predict(X, batch_size=b), full, tol, f"batch {b}")


def test_predict_after_training_uses_updated_weights(make_net, make_ref, tol, rng):
    spec = mlp([3, 5, 2], learning_rate=0.2)
    Ws, bs = random_params(spec, rng)
    net = make_net(spec, initial_weights=Ws, initial_biases=bs)
    X, T = rng.uniform(-1, 1, (9, 3)), rng.uniform(-1, 1, (9, 2))
    net.train(X, T, 4, 2)
    # the weights read back must reproduce the backend's own predictions
    ref = ReferenceNetwork(spec, net.weights, net.biases)
    assert_close(net.predict(X), ref.predict(X), tol)


def test_predict_validation(make_net, rng):
    spec = mlp([3, 2], learning_rate=0.1)
    net = make_net(spec, seed=1)
    with pytest.raises((ValueError, RuntimeError)):
        net.predict(np.zeros((4, 5)))
    assert net.predict(np.zeros((0, 3))).shape == (0, 2)
