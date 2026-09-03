"""NumPy float64 reference implementation of the feed-forward network trained by
``syclnn`` / ``cudann`` / ``ompnn``.

This module is the executable specification of the algorithm: every backend is
required to reproduce it (up to floating-point summation order) and the parity
tests in :mod:`fnn_testkit.tests` check exactly that.  The maths mirrors
``network.hpp`` line by line, including its sign conventions and its quirks:

* column-major weights ``W_l`` of shape ``(n_{l+1}, n_l)`` connecting layer ``l``
  to ``l+1`` and biases ``b_l`` of shape ``(n_{l+1},)``;
* forward: ``net = W @ x + b`` (the stored net input *includes* the bias),
  ``out = f(net)``;
* output delta ``δ_L = (t − o) ∘ f'(net_L)``, data loss ``0.5·Σ(t − o)²``;
* hidden delta ``δ_l = (W_lᵀ δ_{l+1}) ∘ f'(net_l)``;
* gradients averaged over the *current* batch: ``∂E/∂W = δ_{l+1} a_lᵀ / B`` and
  ``∂E/∂b = Σ_batch δ_{l+1} / B``;
* update ``W ← W + step`` where the ascent direction is ``g = ∂E/∂W − r(W)`` with
  ``r`` the regularisation gradient (``λ1·sgn(W)``, ``λ2·W`` or both); biases are
  never regularised;
* optimisers: constant / linear-decay learning rate with optional classical
  momentum; AdaGrad, RMSProp and Adam accumulate the *descent* direction
  ``G = −g`` (that is what the original code does) and ignore momentum;
* the epoch loss is ``data_loss / N + Σ_l (λ1·Σ|W_l| + ½·λ2·Σ W_l²)`` with the
  penalty evaluated on the weights *after* the epoch's updates;
* stop criteria are evaluated after each epoch; ``max_epochs`` is always a hard
  cap; history snapshots are taken at construction, after ``set_weights_biases``
  and after every epoch (when ``record_history`` is on) or at the end of training.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

# --------------------------------------------------------------------------- enums

ACTIVATIONS = ("disabled", "sigmoid", "tanh", "relu", "leaky_relu", "elu")
REGULARIZATIONS = ("disabled", "l1", "l2", "elastic_net")
STOP_CRITERIA = ("max_epochs", "min_error", "min_error_change")
MOMENTA = ("disabled", "classical")
STRATEGIES = ("constant", "linear_decay", "adagrad", "rmsprop", "adam")

LEAKY_RELU_ALPHA = 0.01
ELU_ALPHA = 1.0


def activate(kind: str, z: np.ndarray) -> np.ndarray:
    """f(z) for every activation kind, vectorised."""
    z = np.asarray(z)
    if kind == "disabled":
        return z.copy()
    if kind == "sigmoid":
        return 1.0 / (1.0 + np.exp(-z))
    if kind == "tanh":
        return np.tanh(z)
    if kind == "relu":
        return np.maximum(z, 0.0)
    if kind == "leaky_relu":
        return np.where(z > 0.0, z, LEAKY_RELU_ALPHA * z)
    if kind == "elu":
        return np.where(z > 0.0, z, ELU_ALPHA * (np.exp(np.minimum(z, 0.0)) - 1.0))
    raise ValueError(f"unknown activation {kind!r}")


def derivative(kind: str, z: np.ndarray) -> np.ndarray:
    """f'(z) evaluated from the net input (bias included), as ``network.hpp`` does."""
    z = np.asarray(z)
    if kind == "disabled":
        return np.ones_like(z)
    if kind == "sigmoid":
        s = 1.0 / (1.0 + np.exp(-z))
        return s * (1.0 - s)
    if kind == "tanh":
        t = np.tanh(z)
        return 1.0 - t * t
    if kind == "relu":
        return np.where(z > 0.0, 1.0, 0.0)
    if kind == "leaky_relu":
        return np.where(z > 0.0, 1.0, LEAKY_RELU_ALPHA)
    if kind == "elu":
        return np.where(z > 0.0, 1.0, ELU_ALPHA * np.exp(np.minimum(z, 0.0)))
    raise ValueError(f"unknown activation {kind!r}")


# --------------------------------------------------------------------------- config


@dataclass(frozen=True)
class Layer:
    neurons: int
    activation: str = "disabled"

    def __post_init__(self) -> None:
        if self.activation not in ACTIVATIONS:
            raise ValueError(f"unknown activation {self.activation!r}")


@dataclass(frozen=True)
class Regularization:
    type: str = "disabled"
    lambda1: float = 0.0
    lambda2: float = 0.0

    @property
    def uses_l1(self) -> bool:
        return self.type in ("l1", "elastic_net")

    @property
    def uses_l2(self) -> bool:
        return self.type in ("l2", "elastic_net")


@dataclass(frozen=True)
class StopCriteria:
    type: str = "max_epochs"
    threshold: float = 1e-4


@dataclass(frozen=True)
class Momentum:
    type: str = "disabled"
    rate: float = 0.0


@dataclass(frozen=True)
class AdaptiveLR:
    strategy: str = "constant"
    epsilon: float = 1e-8
    beta1: float = 0.9
    beta2: float = 0.999
    final_lr: float = 1e-4


@dataclass(frozen=True)
class NetSpec:
    """Everything the constructor of ``Network<T>`` takes, backend-agnostic."""

    layers: tuple[Layer, ...]
    learning_rate: float
    regularization: Regularization = Regularization()
    adaptive: AdaptiveLR = AdaptiveLR()
    stop: StopCriteria = StopCriteria()
    momentum: Momentum = Momentum()

    def __post_init__(self) -> None:
        object.__setattr__(self, "layers", tuple(self.layers))
        if len(self.layers) < 2:
            raise ValueError("a network needs at least an input and an output layer")
        for name, value, allowed in (
            ("regularization.type", self.regularization.type, REGULARIZATIONS),
            ("adaptive.strategy", self.adaptive.strategy, STRATEGIES),
            ("stop.type", self.stop.type, STOP_CRITERIA),
            ("momentum.type", self.momentum.type, MOMENTA),
        ):
            if value not in allowed:
                raise ValueError(f"{name}={value!r} not in {allowed}")

    @property
    def sizes(self) -> tuple[int, ...]:
        return tuple(l.neurons for l in self.layers)

    @property
    def n_in(self) -> int:
        return self.layers[0].neurons

    @property
    def n_out(self) -> int:
        return self.layers[-1].neurons

    @property
    def n_params(self) -> int:
        s = self.sizes
        return sum(s[i] * s[i + 1] + s[i + 1] for i in range(len(s) - 1))


def mlp(sizes: Sequence[int], hidden: str = "tanh", output: str = "sigmoid", **kw) -> NetSpec:
    """Shorthand: ``mlp([17, 4, 1], learning_rate=0.1)``."""
    layers = [Layer(sizes[0], "disabled")]
    layers += [Layer(n, hidden) for n in sizes[1:-1]]
    layers.append(Layer(sizes[-1], output))
    return NetSpec(tuple(layers), **kw)


# --------------------------------------------------------------------------- MT19937
# std::mt19937 clone, so the optional per-epoch shuffle of the C++ libraries can be
# reproduced bit-exactly (Fisher–Yates with j = i + mt() % (n - i)).


class MT19937:
    """Bit-exact ``std::mt19937`` (32-bit outputs, ``init_genrand`` seeding)."""

    N, M = 624, 397

    def __init__(self, seed: int = 5489) -> None:
        self.mt = np.zeros(self.N, dtype=np.uint64)
        self.mt[0] = np.uint64(seed & 0xFFFFFFFF)
        for i in range(1, self.N):
            prev = int(self.mt[i - 1])
            self.mt[i] = np.uint64((1812433253 * (prev ^ (prev >> 30)) + i) & 0xFFFFFFFF)
        self.index = self.N

    def _twist(self) -> None:
        mt = self.mt.astype(np.uint64)
        upper, lower = np.uint64(0x80000000), np.uint64(0x7FFFFFFF)
        for i in range(self.N):
            y = (mt[i] & upper) | (mt[(i + 1) % self.N] & lower)
            v = mt[(i + self.M) % self.N] ^ (y >> np.uint64(1))
            if int(y) & 1:
                v ^= np.uint64(0x9908B0DF)
            mt[i] = v
        self.mt = mt
        self.index = 0

    def __call__(self) -> int:
        if self.index >= self.N:
            self._twist()
        y = int(self.mt[self.index])
        self.index += 1
        y ^= y >> 11
        y ^= (y << 7) & 0x9D2C5680
        y ^= (y << 15) & 0xEFC60000
        y ^= y >> 18
        return y & 0xFFFFFFFF


def shuffle_permutation(n: int, rng: MT19937) -> np.ndarray:
    """The Fisher–Yates permutation the C++ libraries draw for one epoch."""
    perm = np.arange(n)
    for i in range(n - 1):
        j = i + rng() % (n - i)
        perm[i], perm[j] = perm[j], perm[i]
    return perm


# --------------------------------------------------------------------------- network


@dataclass
class TrainingOptions:
    record_history: bool = False
    shuffle: bool = False
    shuffle_seed: int = 1


class ReferenceNetwork:
    """The oracle. Float64 by default; ``dtype=np.float32`` mimics a float backend's
    storage precision (but not its summation order)."""

    def __init__(
        self,
        spec: NetSpec,
        initial_weights: Sequence[np.ndarray] | None = None,
        initial_biases: Sequence[np.ndarray] | None = None,
        seed: int | None = None,
        dtype=np.float64,
        record_history: bool = False,
        shuffle: bool = False,
        shuffle_seed: int = 1,
    ) -> None:
        self.spec = spec
        self.dtype = np.dtype(dtype)
        self.record_history = record_history
        self.shuffle = shuffle
        self.shuffle_seed = shuffle_seed
        sizes = spec.sizes
        if any(n == 0 for n in sizes):
            raise ValueError("layer neuron count must be greater than 0")
        self.W: list[np.ndarray] = []
        self.b: list[np.ndarray] = []
        if initial_weights is None or initial_biases is None:
            rng = np.random.default_rng(seed)
            for n_in, n_out in zip(sizes[:-1], sizes[1:]):
                self.W.append(rng.uniform(-0.1, 0.1, size=(n_out, n_in)).astype(self.dtype))
                self.b.append(rng.uniform(-0.1, 0.1, size=(n_out,)).astype(self.dtype))
        else:
            if len(initial_weights) != len(sizes) - 1 or len(initial_biases) != len(sizes) - 1:
                raise ValueError("initial weights/biases have the wrong number of layers")
            for l, (n_in, n_out) in enumerate(zip(sizes[:-1], sizes[1:])):
                W = np.asarray(initial_weights[l], dtype=self.dtype)
                b = np.asarray(initial_biases[l], dtype=self.dtype)
                if W.shape != (n_out, n_in):
                    raise ValueError(f"initial weights of layer {l} have shape {W.shape}, expected {(n_out, n_in)}")
                if b.shape != (n_out,):
                    raise ValueError(f"initial biases of layer {l} have shape {b.shape}, expected {(n_out,)}")
                self.W.append(W.copy())
                self.b.append(b.copy())
        self._reset_optimizer_state()
        self.weights_history: list[list[np.ndarray]] = []
        self.biases_history: list[list[np.ndarray]] = []
        self.loss_history: list[float] = []
        self._snapshot()

    # ------------------------------------------------------------------ state
    def _reset_optimizer_state(self) -> None:
        self.mW = [np.zeros_like(W) for W in self.W]
        self.mb = [np.zeros_like(b) for b in self.b]
        self.vW = [np.zeros_like(W) for W in self.W]
        self.vb = [np.zeros_like(b) for b in self.b]
        self.adam_step = 0

    def _snapshot(self) -> None:
        self.weights_history.append([W.copy() for W in self.W])
        self.biases_history.append([b.copy() for b in self.b])

    @property
    def weights(self) -> list[np.ndarray]:
        return [W.copy() for W in self.W]

    @property
    def biases(self) -> list[np.ndarray]:
        return [b.copy() for b in self.b]

    def weights_biases(self) -> tuple[list[list[np.ndarray]], list[list[np.ndarray]]]:
        return (
            [[W.copy() for W in ws] for ws in self.weights_history],
            [[b.copy() for b in bs] for bs in self.biases_history],
        )

    def set_weights_biases(self, weights: Sequence[np.ndarray], biases: Sequence[np.ndarray]) -> None:
        if len(weights) != len(self.W) or len(biases) != len(self.b):
            raise ValueError("layer count mismatch")
        for l, (W, b) in enumerate(zip(weights, biases)):
            W = np.asarray(W, dtype=self.dtype)
            b = np.asarray(b, dtype=self.dtype)
            if W.shape != self.W[l].shape or b.shape != self.b[l].shape:
                raise ValueError(f"shape mismatch for layer {l}")
            self.W[l] = W.copy()
            self.b[l] = b.copy()
        # network.hpp resets the history but keeps the optimiser state.
        self.weights_history.clear()
        self.biases_history.clear()
        self._snapshot()

    # ------------------------------------------------------------------ forward
    def forward(self, X: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Return ``(nets, outs)``: ``outs[0]`` is the input batch (column-major, one
        sample per column), ``nets[l]``/``outs[l+1]`` belong to layer ``l+1``."""
        A = np.asarray(X, dtype=self.dtype).T  # (n_in, B)
        outs = [A]
        nets = []
        for l, (W, b) in enumerate(zip(self.W, self.b)):
            net = (W @ outs[-1] + b[:, None]).astype(self.dtype, copy=False)
            out = activate(self.spec.layers[l + 1].activation, net).astype(self.dtype, copy=False)
            nets.append(net)
            outs.append(out)
        return nets, outs

    def predict(self, X: np.ndarray, batch_size: int | None = None) -> np.ndarray:
        X = np.asarray(X, dtype=self.dtype)
        if X.ndim != 2 or X.shape[1] != self.spec.n_in:
            raise ValueError(f"predict expects (N, {self.spec.n_in}) inputs, got {X.shape}")
        if X.shape[0] == 0:
            return np.zeros((0, self.spec.n_out), dtype=self.dtype)
        if not batch_size:
            batch_size = X.shape[0]
        chunks = [self.forward(X[s : s + batch_size])[1][-1].T for s in range(0, X.shape[0], batch_size)]
        return np.concatenate(chunks, axis=0)

    # ------------------------------------------------------------------ backward
    def _regularization_gradient(self, W: np.ndarray) -> np.ndarray:
        reg = self.spec.regularization
        r = np.zeros_like(W)
        if reg.uses_l1:
            r = r + reg.lambda1 * np.sign(W)
        if reg.uses_l2:
            r = r + reg.lambda2 * W
        return r

    def _step(self, param, grad_data, reg_grad, m, v, lr_eff):
        """One optimiser step; returns the update to add to ``param`` and mutates the
        moment buffers in place.  ``grad_data - reg_grad`` is the ascent direction."""
        cfg = self.spec.adaptive
        g = grad_data - reg_grad
        strategy = cfg.strategy
        if strategy in ("constant", "linear_decay"):
            if self.spec.momentum.type == "classical":
                m[...] = self.spec.momentum.rate * m + lr_eff * g
                return m.copy()
            return lr_eff * g
        G = -g  # the "positive total gradient" of the original code
        if strategy == "adagrad":
            v[...] = v + G * G
            return (self.spec.learning_rate / (np.sqrt(v) + cfg.epsilon)) * g
        if strategy == "rmsprop":
            v[...] = cfg.beta1 * v + (1.0 - cfg.beta1) * G * G
            return (self.spec.learning_rate / (np.sqrt(v) + cfg.epsilon)) * g
        if strategy == "adam":
            t = self.adam_step
            m[...] = cfg.beta1 * m + (1.0 - cfg.beta1) * G
            v[...] = cfg.beta2 * v + (1.0 - cfg.beta2) * G * G
            m_hat = m / (1.0 - cfg.beta1**t)
            v_hat = v / (1.0 - cfg.beta2**t)
            return -(self.spec.learning_rate * m_hat / (np.sqrt(v_hat) + cfg.epsilon))
        raise ValueError(strategy)

    def _effective_lr(self, epoch: int, max_epochs: int) -> float:
        cfg = self.spec.adaptive
        if cfg.strategy == "linear_decay":
            gamma = epoch / (max_epochs - 1) if max_epochs > 1 else 0.0
            return self.spec.learning_rate * (1.0 - gamma) + gamma * cfg.final_lr
        return self.spec.learning_rate

    def train_batch(self, Xb: np.ndarray, Tb: np.ndarray, epoch: int, max_epochs: int) -> float:
        """One mini-batch of back-propagation. Returns the batch's data loss
        ``0.5·Σ(t−o)²`` (not yet divided by N)."""
        B = Xb.shape[0]
        L = len(self.W)
        nets, outs = self.forward(Xb)
        T = np.asarray(Tb, dtype=self.dtype).T
        err = T - outs[-1]
        loss = 0.5 * float(np.sum(err.astype(np.float64) ** 2))
        deltas = [None] * L
        deltas[L - 1] = err * derivative(self.spec.layers[L].activation, nets[L - 1])
        for l in range(L - 2, -1, -1):
            deltas[l] = (self.W[l + 1].T @ deltas[l + 1]) * derivative(self.spec.layers[l + 1].activation, nets[l])
        gW = [(deltas[l] @ outs[l].T) / B for l in range(L)]
        gb = [deltas[l].sum(axis=1) / B for l in range(L)]
        if self.spec.adaptive.strategy == "adam":
            self.adam_step += 1
        lr_eff = self._effective_lr(epoch, max_epochs)
        for l in range(L):
            dW = self._step(self.W[l], gW[l], self._regularization_gradient(self.W[l]), self.mW[l], self.vW[l], lr_eff)
            db = self._step(self.b[l], gb[l], np.zeros_like(gb[l]), self.mb[l], self.vb[l], lr_eff)
            self.W[l] = (self.W[l] + dW).astype(self.dtype, copy=False)
            self.b[l] = (self.b[l] + db).astype(self.dtype, copy=False)
        return loss

    def penalty(self) -> float:
        reg = self.spec.regularization
        p = 0.0
        for W in self.W:
            W64 = W.astype(np.float64)
            if reg.uses_l1:
                p += reg.lambda1 * float(np.sum(np.abs(W64)))
            if reg.uses_l2:
                p += 0.5 * reg.lambda2 * float(np.sum(W64 * W64))
        return p

    def train(self, X: np.ndarray, Y: np.ndarray, batch_size: int, max_epochs: int) -> np.ndarray:
        X = np.asarray(X, dtype=self.dtype)
        Y = np.asarray(Y, dtype=self.dtype)
        if X.ndim != 2 or X.shape[1] != self.spec.n_in:
            raise ValueError(f"train expects (N, {self.spec.n_in}) inputs, got {X.shape}")
        if Y.ndim != 2 or Y.shape != (X.shape[0], self.spec.n_out):
            raise ValueError(f"train expects (N, {self.spec.n_out}) targets, got {Y.shape}")
        N = X.shape[0]
        if N == 0:
            raise ValueError("number of samples cannot be zero")
        if batch_size <= 0:
            raise ValueError("batch size must be greater than 0")
        if max_epochs <= 0:
            raise ValueError("max_epochs must be greater than 0")
        batch_size = min(batch_size, N)
        self.adam_step = 0
        self.loss_history = []
        rng = MT19937(self.shuffle_seed) if self.shuffle else None
        for epoch in range(max_epochs):
            if rng is not None:
                perm = shuffle_permutation(N, rng)
                Xe, Ye = X[perm], Y[perm]
            else:
                Xe, Ye = X, Y
            data_loss = 0.0
            for start in range(0, N, batch_size):
                data_loss += self.train_batch(Xe[start : start + batch_size], Ye[start : start + batch_size], epoch, max_epochs)
            total = data_loss / N + self.penalty()
            self.loss_history.append(total)
            if self.record_history:
                self._snapshot()
            stop = self.spec.stop
            if stop.type == "min_error" and total < stop.threshold:
                break
            if stop.type == "min_error_change" and epoch > 0 and abs(self.loss_history[-2] - total) < stop.threshold:
                break
        if not self.record_history:
            self._snapshot()
        return np.asarray(self.loss_history, dtype=np.float64)

    # ------------------------------------------------------------------ helpers
    def gradients(self, Xb: np.ndarray, Tb: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray], float]:
        """Data-loss gradients ``(gW, gb, loss)`` of one batch *without* updating —
        used by the finite-difference tests."""
        B = Xb.shape[0]
        L = len(self.W)
        nets, outs = self.forward(Xb)
        T = np.asarray(Tb, dtype=self.dtype).T
        err = T - outs[-1]
        loss = 0.5 * float(np.sum(err.astype(np.float64) ** 2))
        deltas = [None] * L
        deltas[L - 1] = err * derivative(self.spec.layers[L].activation, nets[L - 1])
        for l in range(L - 2, -1, -1):
            deltas[l] = (self.W[l + 1].T @ deltas[l + 1]) * derivative(self.spec.layers[l + 1].activation, nets[l])
        gW = [(deltas[l] @ outs[l].T) / B for l in range(L)]
        gb = [deltas[l].sum(axis=1) / B for l in range(L)]
        return gW, gb, loss

    def data_loss(self, X: np.ndarray, Y: np.ndarray) -> float:
        """``0.5·Σ(t−o)² / N`` for the whole set with the current weights."""
        out = self.predict(X)
        err = np.asarray(Y, dtype=np.float64) - out.astype(np.float64)
        return 0.5 * float(np.sum(err * err)) / X.shape[0]


def flatten_weights(W: np.ndarray) -> np.ndarray:
    """Column-major flattening used by the C++ libraries (``flat[i + rows*j] = W[i, j]``)."""
    return np.asarray(W).flatten(order="F")


def unflatten_weights(flat: Iterable[float], n_out: int, n_in: int, dtype=np.float64) -> np.ndarray:
    return np.asarray(list(flat) if not isinstance(flat, np.ndarray) else flat, dtype=dtype).reshape((n_out, n_in), order="F")
