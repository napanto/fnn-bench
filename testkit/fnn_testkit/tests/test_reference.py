"""Self-checks of the NumPy oracle: these do not depend on any backend."""

import numpy as np
import pytest

from fnn_testkit.plugin import Tolerance, assert_close, random_params
from fnn_testkit.reference import (
    ACTIVATIONS,
    MT19937,
    AdaptiveLR,
    Layer,
    Momentum,
    NetSpec,
    ReferenceNetwork,
    Regularization,
    activate,
    derivative,
    mlp,
    shuffle_permutation,
)

pytestmark = pytest.mark.reference_only


def test_mt19937_matches_std():
    # C++ standard: the 10000th invocation of a default-seeded std::mt19937 is 4123659995.
    rng = MT19937(5489)
    first = rng()
    assert first == 3499211612
    for _ in range(9998):
        rng()
    assert rng() == 4123659995


def test_shuffle_permutation_is_a_permutation():
    perm = shuffle_permutation(100, MT19937(1))
    assert sorted(perm.tolist()) == list(range(100))
    assert not np.array_equal(perm, np.arange(100))


@pytest.mark.parametrize("kind", ACTIVATIONS)
def test_derivative_matches_finite_difference(kind):
    z = np.linspace(-4.0, 4.0, 81)
    z = z[np.abs(z) > 1e-6]  # avoid the kink of relu-like functions
    h = 1e-6
    fd = (activate(kind, z + h) - activate(kind, z - h)) / (2 * h)
    assert_close(derivative(kind, z), fd, Tolerance(1e-6, 1e-8), f"f'({kind})")


@pytest.mark.parametrize("kind", ACTIVATIONS)
@pytest.mark.parametrize("depth", [1, 2, 3])
def test_backprop_gradient_matches_finite_difference(kind, depth):
    rng = np.random.default_rng(0)
    sizes = [5] + [4] * (depth - 1) + [3]
    spec = mlp(sizes, hidden=kind, output=kind, learning_rate=0.1)
    Ws, bs = random_params(spec, rng)
    net = ReferenceNetwork(spec, Ws, bs)
    X = rng.uniform(-1, 1, size=(6, 5))
    T = rng.uniform(-1, 1, size=(6, 3))
    gW, gb, _ = net.gradients(X, T)
    B = X.shape[0]

    def loss_fn():
        out = net.predict(X)
        return 0.5 * np.sum((T - out) ** 2) / B

    h = 1e-6
    for l in range(depth):
        for idx in [(0, 0), (sizes[l + 1] - 1, sizes[l] - 1), (sizes[l + 1] // 2, sizes[l] // 2)]:
            w0 = net.W[l][idx]
            net.W[l][idx] = w0 + h
            lp = loss_fn()
            net.W[l][idx] = w0 - h
            lm = loss_fn()
            net.W[l][idx] = w0
            # gW is the *ascent* direction of (t-o), i.e. -dE/dW
            assert abs(-(lp - lm) / (2 * h) - gW[l][idx]) < 1e-6, (l, idx)
        j = sizes[l + 1] - 1
        b0 = net.b[l][j]
        net.b[l][j] = b0 + h
        lp = loss_fn()
        net.b[l][j] = b0 - h
        lm = loss_fn()
        net.b[l][j] = b0
        assert abs(-(lp - lm) / (2 * h) - gb[l][j]) < 1e-6


def _one_step(spec, Ws, bs, X, T, **kw):
    net = ReferenceNetwork(spec, Ws, bs, **kw)
    gW, gb, _ = net.gradients(X, T)
    net.train(X, T, batch_size=X.shape[0], max_epochs=1)
    return net, gW, gb


def test_sgd_update_closed_form():
    rng = np.random.default_rng(1)
    spec = mlp([3, 4, 2], learning_rate=0.3, regularization=Regularization("elastic_net", 0.01, 0.02))
    Ws, bs = random_params(spec, rng)
    X, T = rng.uniform(-1, 1, (5, 3)), rng.uniform(-1, 1, (5, 2))
    net, gW, gb = _one_step(spec, Ws, bs, X, T)
    for l in range(2):
        expected = Ws[l] + 0.3 * (gW[l] - (0.01 * np.sign(Ws[l]) + 0.02 * Ws[l]))
        assert_close(net.W[l], expected, Tolerance(1e-12, 1e-14), f"W{l}")
        assert_close(net.b[l], bs[l] + 0.3 * gb[l], Tolerance(1e-12, 1e-14), f"b{l}")


def test_momentum_update_closed_form():
    rng = np.random.default_rng(2)
    spec = mlp([3, 2], learning_rate=0.1, momentum=Momentum("classical", 0.9))
    Ws, bs = random_params(spec, rng)
    X, T = rng.uniform(-1, 1, (4, 3)), rng.uniform(-1, 1, (4, 2))
    net = ReferenceNetwork(spec, Ws, bs)
    g1, gb1, _ = net.gradients(X, T)
    net.train(X, T, 4, 1)
    v = 0.1 * g1[0]
    W1 = Ws[0] + v
    assert_close(net.W[0], W1, Tolerance(1e-12, 1e-14))
    g2, _, _ = net.gradients(X, T)
    net.train(X, T, 4, 1)
    v = 0.9 * v + 0.1 * g2[0]
    assert_close(net.W[0], W1 + v, Tolerance(1e-12, 1e-14))


def test_linear_decay_schedule():
    rng = np.random.default_rng(3)
    spec = mlp([2, 1], learning_rate=1.0, adaptive=AdaptiveLR("linear_decay", final_lr=0.0))
    Ws, bs = random_params(spec, rng)
    X, T = rng.uniform(-1, 1, (3, 2)), rng.uniform(-1, 1, (3, 1))
    net = ReferenceNetwork(spec, Ws, bs)
    # epochs 0,1,2 of max_epochs=3 -> gamma 0, 0.5, 1 -> lr 1.0, 0.5, 0.0
    assert net._effective_lr(0, 3) == 1.0 and net._effective_lr(1, 3) == 0.5 and net._effective_lr(2, 3) == 0.0
    assert net._effective_lr(0, 1) == 1.0  # max_epochs == 1 -> gamma 0
    net.train(X, T, 3, 3)
    # third epoch has lr 0 -> weights unchanged during it
    hist = net.weights_history
    assert len(hist) == 2  # initial + final (record_history off)


def test_adaptive_updates_closed_form():
    rng = np.random.default_rng(4)
    X, T = rng.uniform(-1, 1, (4, 3)), rng.uniform(-1, 1, (4, 2))
    for strategy in ("adagrad", "rmsprop", "adam"):
        spec = mlp([3, 2], learning_rate=0.05, adaptive=AdaptiveLR(strategy, 1e-8, 0.9, 0.999),
                   regularization=Regularization("l2", 0.0, 0.1))
        Ws, bs = random_params(spec, rng)
        net = ReferenceNetwork(spec, Ws, bs)
        gW, gb, _ = net.gradients(X, T)
        net.train(X, T, 4, 1)
        G = -(gW[0] - 0.1 * Ws[0])  # descent direction, regularised
        if strategy == "adagrad":
            step = (0.05 / (np.sqrt(G * G) + 1e-8)) * (-G)
        elif strategy == "rmsprop":
            v = 0.1 * G * G
            step = (0.05 / (np.sqrt(v) + 1e-8)) * (-G)
        else:
            m = 0.1 * G
            v = 0.001 * G * G
            m_hat = m / (1 - 0.9)
            v_hat = v / (1 - 0.999)
            step = -(0.05 * m_hat / (np.sqrt(v_hat) + 1e-8))
        assert_close(net.W[0], Ws[0] + step, Tolerance(1e-10, 1e-12), strategy)


def test_epoch_loss_definition():
    rng = np.random.default_rng(5)
    spec = mlp([3, 4, 2], learning_rate=0.1, regularization=Regularization("elastic_net", 0.01, 0.02))
    Ws, bs = random_params(spec, rng)
    X, T = rng.uniform(-1, 1, (7, 3)), rng.uniform(-1, 1, (7, 2))
    net = ReferenceNetwork(spec, Ws, bs)
    data_before = net.data_loss(X, T)  # weights before the (single) update of the epoch
    losses = net.train(X, T, batch_size=7, max_epochs=1)
    penalty_after = sum(0.01 * np.abs(W).sum() + 0.5 * 0.02 * (W * W).sum() for W in net.W)
    assert losses.shape == (1,)
    assert abs(losses[0] - (data_before + penalty_after)) < 1e-12


def test_remainder_batch_and_history():
    rng = np.random.default_rng(6)
    spec = mlp([2, 3, 1], learning_rate=0.1)
    Ws, bs = random_params(spec, rng)
    X, T = rng.uniform(-1, 1, (7, 2)), rng.uniform(-1, 1, (7, 1))
    net = ReferenceNetwork(spec, Ws, bs, record_history=True)
    losses = net.train(X, T, batch_size=3, max_epochs=4)  # batches of 3,3,1
    assert losses.shape == (4,)
    hw, hb = net.weights_biases()
    assert len(hw) == 5 and len(hb) == 5
    assert_close(hw[0][0], Ws[0], Tolerance(0, 0))
    assert_close(hw[-1][0], net.W[0], Tolerance(0, 0))


def test_stop_criteria():
    rng = np.random.default_rng(7)
    spec = mlp([2, 3, 1], learning_rate=0.5)
    Ws, bs = random_params(spec, rng)
    X, T = rng.uniform(-1, 1, (8, 2)), rng.uniform(-1, 1, (8, 1))
    ref = ReferenceNetwork(spec, Ws, bs).train(X, T, 8, 20)
    k = 5
    spec_e = NetSpec(spec.layers, 0.5, stop=__import__("fnn_testkit.reference", fromlist=["StopCriteria"]).StopCriteria("min_error", float(ref[k]) + 1e-15))
    got = ReferenceNetwork(spec_e, Ws, bs).train(X, T, 8, 20)
    assert got.shape == (k + 1,)
    diffs = np.abs(np.diff(ref))
    spec_c = NetSpec(spec.layers, 0.5, stop=__import__("fnn_testkit.reference", fromlist=["StopCriteria"]).StopCriteria("min_error_change", float(diffs[k - 1]) + 1e-15))
    got = ReferenceNetwork(spec_c, Ws, bs).train(X, T, 8, 20)
    first = int(np.argmax(diffs < diffs[k - 1] + 1e-15))
    assert got.shape == (first + 2,)


def test_validation_errors():
    spec = mlp([2, 1], learning_rate=0.1)
    net = ReferenceNetwork(spec, seed=0)
    X, T = np.zeros((4, 2)), np.zeros((4, 1))
    with pytest.raises(ValueError):
        net.train(X, T, 2, 0)
    with pytest.raises(ValueError):
        net.train(X, T, 0, 1)
    with pytest.raises(ValueError):
        net.train(np.zeros((4, 3)), T, 2, 1)
    with pytest.raises(ValueError):
        net.train(X, np.zeros((4, 2)), 2, 1)
    with pytest.raises(ValueError):
        net.train(np.zeros((0, 2)), np.zeros((0, 1)), 2, 1)
    with pytest.raises(ValueError):
        net.predict(np.zeros((3, 5)))
    with pytest.raises(ValueError):
        ReferenceNetwork(spec, [np.zeros((1, 3))], [np.zeros(1)])
    with pytest.raises(ValueError):
        NetSpec((Layer(3),), 0.1)
    with pytest.raises(ValueError):
        ReferenceNetwork(mlp([2, 0, 1], learning_rate=0.1), seed=0)
    assert net.predict(np.zeros((0, 2))).shape == (0, 1)
    # batch_size > N is clamped
    Ws, bs = random_params(spec, np.random.default_rng(1))
    a = ReferenceNetwork(spec, Ws, bs).train(X + 0.5, T + 0.5, 4, 2)
    b = ReferenceNetwork(spec, Ws, bs).train(X + 0.5, T + 0.5, 100, 2)
    assert np.array_equal(a, b)
