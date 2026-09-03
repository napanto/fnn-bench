"""Training-loop semantics: batching, stop criteria, history, validation,
determinism, precision."""

import numpy as np
import pytest

from fnn_testkit import backends as be
from fnn_testkit.plugin import Tolerance, assert_close, random_params
from fnn_testkit.reference import AdaptiveLR, Layer, NetSpec, StopCriteria, mlp


def _data(rng, n=7, n_in=2, n_out=1):
    return rng.uniform(-1, 1, (n, n_in)), rng.uniform(0, 1, (n, n_out))


def test_remainder_batch(make_net, make_ref, tol, rng):
    spec = mlp([2, 3, 1], learning_rate=0.2)
    Ws, bs = random_params(spec, rng)
    X, T = _data(rng, 7)
    got = make_net(spec, initial_weights=Ws, initial_biases=bs).train(X, T, 3, 3)  # 3 + 3 + 1
    exp = make_ref(spec, initial_weights=Ws, initial_biases=bs).train(X, T, 3, 3)
    assert_close(got, exp, tol)


def test_batch_size_larger_than_dataset_is_clamped(make_net, dtype, rng):
    spec = mlp([2, 3, 1], learning_rate=0.2)
    Ws, bs = random_params(spec, rng)
    X, T = _data(rng, 5)
    a = make_net(spec, initial_weights=Ws, initial_biases=bs).train(X, T, 5, 2)
    b = make_net(spec, initial_weights=Ws, initial_biases=bs).train(X, T, 1000, 2)
    # multithreaded BLAS is not bitwise reproducible run to run
    assert_close(a, b, Tolerance(1e-12, 1e-14) if dtype == "double" else Tolerance(1e-6, 1e-7))


def test_invalid_arguments_raise(make_net, rng):
    spec = mlp([2, 3, 1], learning_rate=0.2)
    net = make_net(spec, seed=3)
    X, T = _data(rng, 4)
    err = (ValueError, RuntimeError)
    with pytest.raises(err):
        net.train(X, T, 2, 0)
    with pytest.raises(err):
        net.train(X, T, 0, 1)
    with pytest.raises(err):
        net.train(np.zeros((4, 3)), T, 2, 1)
    with pytest.raises(err):
        net.train(X, np.zeros((4, 2)), 2, 1)
    with pytest.raises(err):
        net.train(np.zeros((0, 2)), np.zeros((0, 1)), 2, 1)
    with pytest.raises(err):
        make_net(spec, initial_weights=[np.zeros((3, 2)), np.zeros((1, 4))], initial_biases=[np.zeros(3), np.zeros(1)])
    with pytest.raises(err):
        make_net(NetSpec((Layer(2), Layer(0), Layer(1)), 0.1), seed=1)
    with pytest.raises(err):
        net.set_weights_biases([np.zeros((3, 2))], [np.zeros(3)])


def test_stop_criteria_epoch_counts(make_net, make_ref, tol, rng):
    spec = mlp([2, 3, 1], learning_rate=0.5)
    Ws, bs = random_params(spec, rng)
    X, T = _data(rng, 8)
    ref = make_ref(spec, initial_weights=Ws, initial_biases=bs).train(X, T, 8, 20)
    k = 5
    # thresholds half-way between consecutive reference values, so that float
    # rounding of the backend's loss cannot move the stopping epoch
    assert ref[k] < ref[k - 1], "loss must decrease for this test to be meaningful"
    thr = 0.5 * (ref[k] + ref[k - 1])
    first = int(np.argmax(ref < thr))
    # MinError: stop as soon as loss < threshold (checked after every epoch)
    s = NetSpec(spec.layers, 0.5, stop=StopCriteria("min_error", float(thr)))
    got = make_net(s, initial_weights=Ws, initial_biases=bs).train(X, T, 8, 20)
    assert got.shape == (first + 1,)
    assert_close(got, ref[: first + 1], tol)
    # MinErrorChange: |loss[e-1] - loss[e]| < threshold, e > 0
    diffs = np.abs(np.diff(ref))
    assert diffs[k] < diffs[k - 1]
    thr_c = 0.5 * (diffs[k] + diffs[k - 1])
    first_c = int(np.argmax(diffs < thr_c))
    s = NetSpec(spec.layers, 0.5, stop=StopCriteria("min_error_change", float(thr_c)))
    got = make_net(s, initial_weights=Ws, initial_biases=bs).train(X, T, 8, 20)
    assert got.shape == (first_c + 2,)
    # MaxEpochs is always a hard cap
    s = NetSpec(spec.layers, 0.5, stop=StopCriteria("min_error", 0.0))
    assert make_net(s, initial_weights=Ws, initial_biases=bs).train(X, T, 8, 4).shape == (4,)


def test_history_semantics(make_net, rng):
    spec = mlp([2, 3, 1], learning_rate=0.2)
    Ws, bs = random_params(spec, rng)
    X, T = _data(rng, 6)
    # default: history = [initial, final]
    net = make_net(spec, initial_weights=Ws, initial_biases=bs)
    cast = lambda a: np.asarray(a, dtype=net.np_dtype)  # noqa: E731
    hw, hb = net.history()
    assert len(hw) == 1 and len(hb) == 1
    assert_close(hw[0][0], cast(Ws[0]), Tolerance(0, 0))
    net.train(X, T, 3, 4)
    hw, hb = net.history()
    assert len(hw) == 2
    assert_close(hw[0][1], cast(Ws[1]), Tolerance(0, 0))
    assert_close(hw[-1][0], net.weights[0], Tolerance(0, 0))
    assert_close(hb[-1][1], net.biases[1], Tolerance(0, 0))
    # record_history: one snapshot per epoch
    net = make_net(spec, initial_weights=Ws, initial_biases=bs, options={"record_history": True})
    net.train(X, T, 3, 4)
    hw, hb = net.history()
    assert len(hw) == 5 and len(hb) == 5
    assert_close(hw[-1][0], net.weights[0], Tolerance(0, 0))
    # ... and the intermediate snapshots are the states a shorter run ends in
    short = make_net(spec, initial_weights=Ws, initial_biases=bs)
    short.train(X, T, 3, 2)
    assert_close(hw[2][0], short.weights[0], Tolerance(0, 0))


def test_set_weights_biases(make_net, make_ref, tol, rng):
    spec = mlp([2, 3, 1], learning_rate=0.2)
    Ws, bs = random_params(spec, rng)
    W2, b2 = random_params(spec, rng, scale=0.3)
    X, T = _data(rng, 6)
    net = make_net(spec, initial_weights=Ws, initial_biases=bs)
    net.train(X, T, 3, 2)
    net.set_weights_biases(W2, b2)
    cast = lambda a: np.asarray(a, dtype=net.np_dtype)  # noqa: E731
    for l in range(2):
        assert_close(net.weights[l], cast(W2[l]), Tolerance(0, 0))
        assert_close(net.biases[l], cast(b2[l]), Tolerance(0, 0))
    hw, _ = net.history()
    assert len(hw) == 1
    assert_close(hw[0][0], cast(W2[0]), Tolerance(0, 0))
    # plain SGD has no optimiser state: continuing equals a fresh network
    got = net.train(X, T, 3, 2)
    exp = make_ref(spec, initial_weights=W2, initial_biases=b2).train(X, T, 3, 2)
    assert_close(got, exp, tol)


@pytest.mark.compiled_only
def test_seeded_initialisation_is_deterministic(make_net):
    spec = mlp([5, 7, 3], learning_rate=0.1)
    a = make_net(spec, seed=42)
    b = make_net(spec, seed=42)
    c = make_net(spec, seed=43)
    for l in range(2):
        assert np.array_equal(a.weights[l], b.weights[l])
        assert np.array_equal(a.biases[l], b.biases[l])
        assert np.abs(a.weights[l]).max() <= 0.1 and np.abs(a.biases[l]).max() <= 0.1
    assert not np.array_equal(a.weights[0], c.weights[0])
    assert a.weights[0].std() > 0.01


def test_float_and_double_agree(backend, device, backend_options, rng):
    spec = mlp([4, 6, 2], hidden="tanh", output="sigmoid", learning_rate=0.1, adaptive=AdaptiveLR("adam"))
    Ws, bs = random_params(spec, rng)
    X, T = _data(rng, 12, 4, 2)
    nets = {d: be.NetworkAdapter(backend, spec, dtype=d, device=device, initial_weights=Ws, initial_biases=bs, **backend_options) for d in ("double", "float")}
    losses = {d: n.train(X, T, 5, 3) for d, n in nets.items()}
    assert_close(losses["float"], losses["double"], Tolerance(1e-3, 1e-4), "loss")
    assert_close(nets["float"].weights[0], nets["double"].weights[0], Tolerance(1e-3, 1e-4), "W0")


def test_loss_is_finite_and_decreasing_on_easy_problem(make_net, rng):
    spec = mlp([2, 4, 1], hidden="tanh", output="disabled", learning_rate=0.05)
    X = rng.uniform(-1, 1, (32, 2))
    T = (X[:, :1] * 0.5 + X[:, 1:] * 0.25)
    net = make_net(spec, seed=7)
    losses = net.train(X, T, 8, 30)
    assert np.all(np.isfinite(losses))
    assert losses[-1] < losses[0]
