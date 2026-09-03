"""Activation functions and their derivatives inside the backend's kernels."""

import numpy as np
import pytest

from fnn_testkit.plugin import Tolerance, assert_close
from fnn_testkit.reference import ACTIVATIONS, AdaptiveLR, Layer, NetSpec, activate, derivative


def _grid():
    z = np.linspace(-4.0, 4.0, 33)
    return z[np.abs(z) > 1e-6]


@pytest.mark.parametrize("kind", ACTIVATIONS)
def test_activation_values(make_net, dtype, kind):
    z = _grid()
    n = z.size
    # identity weights, zero bias: out = f(x)
    spec = NetSpec((Layer(n, "disabled"), Layer(n, kind)), learning_rate=0.1)
    net = make_net(spec, initial_weights=[np.eye(n)], initial_biases=[np.zeros(n)])
    X = np.tile(z, (3, 1))  # three identical samples
    out = net.predict(X)
    tol = Tolerance(1e-12, 1e-14) if dtype == "double" else Tolerance(1e-5, 1e-6)
    assert_close(out, np.tile(activate(kind, z), (3, 1)), tol, f"f({kind})")


@pytest.mark.parametrize("kind", ACTIVATIONS)
def test_activation_derivatives(make_net, dtype, kind):
    """One SGD step with unit one-hot inputs isolates f'(z): with W = [z_0 .. z_{B-1}],
    x_n = e_n, t_n = f(z_n) + 1, lr = 1 the weight update is f'(z_n)/B."""
    z = _grid()
    B = z.size
    spec = NetSpec((Layer(B, "disabled"), Layer(1, kind)), learning_rate=1.0, adaptive=AdaptiveLR("constant"))
    W0 = z.reshape(1, B)
    net = make_net(spec, initial_weights=[W0], initial_biases=[np.zeros(1)])
    X = np.eye(B)
    T = (activate(kind, z) + 1.0).reshape(B, 1)
    net.train(X, T, batch_size=B, max_epochs=1)
    W1 = net.weights[0]
    got = (W1 - W0)[0] * B
    tol = Tolerance(1e-10, 1e-12) if dtype == "double" else Tolerance(2e-4, 2e-5)
    assert_close(got, derivative(kind, z), tol, f"f'({kind})")
