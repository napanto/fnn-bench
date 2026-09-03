"""Back-propagation gradients: extracted from one SGD step (lr = 1, no
regularisation, a single batch) and compared with (a) the reference's analytic
gradient and (b) central finite differences of the backend's *own* loss."""

import numpy as np
import pytest

from fnn_testkit.plugin import Tolerance, assert_close, random_params
from fnn_testkit.reference import ACTIVATIONS, AdaptiveLR, mlp


def _extract_gradients(net, Ws, bs, X, T):
    net.set_weights_biases(Ws, bs)
    net.train(X, T, batch_size=X.shape[0], max_epochs=1)
    gW = [W1 - W0 for W1, W0 in zip(net.weights, Ws)]
    gb = [b1 - b0 for b1, b0 in zip(net.biases, bs)]
    return gW, gb


@pytest.mark.parametrize("kind", ACTIVATIONS)
@pytest.mark.parametrize("depth", [1, 2, 3])
def test_gradients_match_reference(make_net, make_ref, tol, rng, kind, depth):
    sizes = [5] + [4] * (depth - 1) + [3]
    spec = mlp(sizes, hidden=kind, output=kind, learning_rate=1.0, adaptive=AdaptiveLR("constant"))
    Ws, bs = random_params(spec, rng)
    X, T = rng.uniform(-1, 1, (6, 5)), rng.uniform(-1, 1, (6, 3))
    net = make_net(spec, initial_weights=Ws, initial_biases=bs)
    ref = make_ref(spec, initial_weights=Ws, initial_biases=bs)
    gW_ref, gb_ref, _ = ref.gradients(X, T)
    gW, gb = _extract_gradients(net, Ws, bs, X, T)
    t = Tolerance(tol.rtol * 10, tol.atol * 10)  # one subtraction of O(1) numbers
    for l in range(depth):
        assert_close(gW[l], gW_ref[l], t, f"dW{l}")
        assert_close(gb[l], gb_ref[l], t, f"db{l}")


@pytest.mark.parametrize("kind", ["sigmoid", "tanh", "elu"])
@pytest.mark.parametrize("depth", [1, 3])
def test_gradients_match_finite_differences(make_net, dtype, rng, kind, depth):
    if dtype != "double":
        pytest.skip("finite differences need double precision")
    sizes = [4] + [3] * (depth - 1) + [2]
    spec = mlp(sizes, hidden=kind, output=kind, learning_rate=1.0, adaptive=AdaptiveLR("constant"))
    Ws, bs = random_params(spec, rng)
    X, T = rng.uniform(-1, 1, (5, 4)), rng.uniform(-1, 1, (5, 2))
    net = make_net(spec, initial_weights=Ws, initial_biases=bs)
    gW, gb = _extract_gradients(net, Ws, bs, X, T)
    B = X.shape[0]

    def loss(ws, bs_):
        net.set_weights_biases(ws, bs_)
        out = net.predict(X)
        return 0.5 * np.sum((T - out) ** 2) / B

    h = 1e-6
    for l in range(depth):
        for idx in [(0, 0), (sizes[l + 1] - 1, sizes[l] - 1)]:
            Wp = [W.copy() for W in Ws]
            Wm = [W.copy() for W in Ws]
            Wp[l][idx] += h
            Wm[l][idx] -= h
            fd = -(loss(Wp, bs) - loss(Wm, bs)) / (2 * h)
            assert abs(fd - gW[l][idx]) < 1e-6, (l, idx, fd, gW[l][idx])
        bp = [b.copy() for b in bs]
        bm = [b.copy() for b in bs]
        bp[l][0] += h
        bm[l][0] -= h
        fd = -(loss(Ws, bp) - loss(Ws, bm)) / (2 * h)
        assert abs(fd - gb[l][0]) < 1e-6, (l, fd, gb[l][0])
