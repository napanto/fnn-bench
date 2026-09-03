"""Regularisation: penalty term in the epoch loss and its gradient."""

import numpy as np
import pytest

from fnn_testkit.plugin import Tolerance, assert_close, random_params
from fnn_testkit.reference import ReferenceNetwork, Regularization, mlp


@pytest.mark.parametrize("reg", ["l1", "l2", "elastic_net"])
def test_penalty_in_epoch_loss(make_net, tol, rng, reg):
    spec = mlp([3, 4, 2], learning_rate=0.1, regularization=Regularization(reg, 0.05, 0.1))
    Ws, bs = random_params(spec, rng)
    X, T = rng.uniform(-1, 1, (6, 3)), rng.uniform(0, 1, (6, 2))
    net = make_net(spec, initial_weights=Ws, initial_biases=bs)
    data_before = ReferenceNetwork(spec, Ws, bs).data_loss(X, T)
    losses = net.train(X, T, batch_size=6, max_epochs=1)
    W_after = net.weights
    penalty = 0.0
    for W in W_after:
        if reg in ("l1", "elastic_net"):
            penalty += 0.05 * np.abs(W).sum()
        if reg in ("l2", "elastic_net"):
            penalty += 0.5 * 0.1 * (W * W).sum()
    assert_close(losses, [data_before + penalty], tol, "epoch loss")


def test_unused_lambda_is_ignored(make_net, rng):
    """Regularization(L2, lambda1=x) must not depend on x (and vice versa), as in
    the original library where the ML project passes both lambdas with type L2."""
    X, T = rng.uniform(-1, 1, (6, 3)), rng.uniform(0, 1, (6, 2))
    base = mlp([3, 4, 2], learning_rate=0.1)
    Ws, bs = random_params(base, rng)
    for typ, other in (("l2", "lambda1"), ("l1", "lambda2")):
        kw1 = {"lambda1": 0.01, "lambda2": 0.01}
        kw2 = dict(kw1)
        kw2[other] = 0.5
        s1 = mlp([3, 4, 2], learning_rate=0.1, regularization=Regularization(typ, **kw1))
        s2 = mlp([3, 4, 2], learning_rate=0.1, regularization=Regularization(typ, **kw2))
        l1 = make_net(s1, initial_weights=Ws, initial_biases=bs).train(X, T, 3, 2)
        l2 = make_net(s2, initial_weights=Ws, initial_biases=bs).train(X, T, 3, 2)
        assert_close(l1, l2, Tolerance(0, 0), typ)


def test_biases_are_not_regularised(make_net, make_ref, tol, rng):
    spec = mlp([2, 3, 1], learning_rate=0.3, regularization=Regularization("l1", 0.2, 0.0))
    Ws, bs = random_params(spec, rng)
    X, T = rng.uniform(-1, 1, (4, 2)), rng.uniform(0, 1, (4, 1))
    net = make_net(spec, initial_weights=Ws, initial_biases=bs)
    ref = make_ref(spec, initial_weights=Ws, initial_biases=bs)
    gW, gb, _ = ref.gradients(X, T)
    net.train(X, T, 4, 1)
    # bias update is lr * gb (no L1 term), weight update includes -lr * lambda1 * sgn(W)
    assert_close(net.biases[0], bs[0] + 0.3 * gb[0], tol, "b0")
    assert_close(net.weights[0], Ws[0] + 0.3 * (gW[0] - 0.2 * np.sign(Ws[0])), tol, "W0")
