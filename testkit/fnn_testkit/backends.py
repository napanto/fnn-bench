"""Uniform adapter over the NumPy reference and the three compiled libraries.

The compiled libraries (``syclnn``, ``cudann``, ``ompnn``) expose the *same*
pybind11 API, differing only in the module name; this module hides the last
differences (flattened column-major weights, ``n_samples`` arguments, enum
spellings) behind :class:`NetworkAdapter` so that tests and benchmarks are
written once.

Contract expected from a compiled backend module ``mod`` (version 0.2 API):

* ``mod.__version__``; ``mod.devices() -> list[dict]`` with at least
  ``index``, ``name``, ``type`` (``"cpu"``/``"gpu"``/``"accelerator"``) and
  ``backend``; ``mod.build_info() -> dict``.
* ``mod.Options()`` with attributes ``device``, ``blas``, ``profile``,
  ``record_history``, ``shuffle``, ``shuffle_seed`` plus the backend's ablation
  switches (unknown keys raise unless ``strict=False``).
* ``mod.Network_double`` / ``mod.Network_float`` constructed with keyword
  arguments ``layers, learning_rate, regularization, backpropagation,
  adaptive_learning_rate, stop_criteria, momentum`` and either ``seed`` or
  ``initial_weights``/``initial_biases`` (flattened column-major), plus
  ``options``; methods ``train(x, y, n_samples, batch_size, max_epochs)``,
  ``predict(x, n_samples, batch_size=0)``, properties ``weights``, ``biases``,
  ``weights_biases``, ``profile``, ``device_name``; ``set_weights_biases(w, b)``
  and ``reset_profile()``.
"""

from __future__ import annotations

import importlib
from typing import Any, Sequence

import numpy as np

from .reference import NetSpec, ReferenceNetwork, flatten_weights, unflatten_weights

BACKENDS = ("numpy", "syclnn", "cudann", "ompnn")
DTYPES = {"double": np.float64, "float": np.float32}

_ENUM = {
    "activation": {
        "disabled": "Disabled",
        "sigmoid": "Sigmoid",
        "tanh": "Tanh",
        "relu": "ReLU",
        "leaky_relu": "LeakyReLU",
        "elu": "ELU",
    },
    "regularization": {"disabled": "Disabled", "l1": "L1", "l2": "L2", "elastic_net": "ElasticNet"},
    "stop": {"max_epochs": "MaxEpochs", "min_error": "MinError", "min_error_change": "MinErrorChange"},
    "momentum": {"disabled": "Disabled", "classical": "Classical"},
    "strategy": {
        "constant": "Constant",
        "linear_decay": "LinearDecay",
        "adagrad": "AdaGrad",
        "rmsprop": "RMSProp",
        "adam": "Adam",
    },
}


def load_backend(name: str):
    """Import the backend module (``None`` for numpy). Raises ImportError if absent."""
    if name == "numpy":
        return None
    if name not in BACKENDS:
        raise ValueError(f"unknown backend {name!r}; choose from {BACKENDS}")
    return importlib.import_module(name)


def backend_available(name: str) -> bool:
    try:
        load_backend(name)
        return True
    except ImportError:
        return False


def list_devices(name: str) -> list[dict[str, Any]]:
    mod = load_backend(name)
    if mod is None:
        return [{"index": 0, "name": "numpy (host)", "type": "cpu", "backend": "numpy"}]
    return list(mod.devices())


def build_info(name: str) -> dict[str, Any]:
    mod = load_backend(name)
    if mod is None:
        return {"backend": "numpy", "version": np.__version__}
    info = dict(mod.build_info())
    info.setdefault("version", getattr(mod, "__version__", "?"))
    return info


def resolve_device(name: str, device: str | None) -> str:
    """Map the CLI's ``--device`` (``cpu``, ``gpu``, ``auto``, ``cuda:0`` ...) to a
    string the backend understands; ``auto`` = the backend's default device."""
    if device in (None, "", "auto", "default"):
        return "default"
    return device


def make_options(mod, device: str | None, strict: bool = True, **options):
    opts = mod.Options()
    opts.device = resolve_device(mod.__name__, device)
    for key, value in options.items():
        if not hasattr(opts, key):
            if strict:
                raise AttributeError(f"{mod.__name__}.Options has no option {key!r}")
            continue
        setattr(opts, key, value)
    return opts


class NetworkAdapter:
    """Backend-agnostic view of a network; arrays in and out are NumPy ``(N, n)``
    row-major, weights are ``(n_out, n_in)`` matrices."""

    def __init__(
        self,
        backend: str,
        spec: NetSpec,
        dtype: str = "double",
        device: str | None = None,
        initial_weights: Sequence[np.ndarray] | None = None,
        initial_biases: Sequence[np.ndarray] | None = None,
        seed: int | None = None,
        strict_options: bool = True,
        **options,
    ) -> None:
        if dtype not in DTYPES:
            raise ValueError("dtype must be 'double' or 'float'")
        self.backend = backend
        self.spec = spec
        self.dtype = dtype
        self.np_dtype = DTYPES[dtype]
        self.device = device
        self.options = dict(options)
        self.mod = load_backend(backend)
        sizes = spec.sizes
        if self.mod is None:
            ref_opts = {k: v for k, v in options.items() if k in ("record_history", "shuffle", "shuffle_seed")}
            self.net = ReferenceNetwork(
                spec,
                initial_weights=initial_weights,
                initial_biases=initial_biases,
                seed=seed,
                dtype=self.np_dtype,
                **ref_opts,
            )
            self.device_name = "numpy (host)"
            return
        m = self.mod
        sfx = dtype
        A = m.ActivationType
        layers = [m.LayerDescription(l.neurons, getattr(A, _ENUM["activation"][l.activation])) for l in spec.layers]
        reg_t = getattr(m, f"RegularizationType_{sfx}")
        reg = getattr(m, f"Regularization_{sfx}")(
            getattr(reg_t, _ENUM["regularization"][spec.regularization.type]),
            spec.regularization.lambda1,
            spec.regularization.lambda2,
        )
        stop_t = getattr(m, f"StopCriteriaType_{sfx}")
        stop = getattr(m, f"StopCriteria_{sfx}")(getattr(stop_t, _ENUM["stop"][spec.stop.type]), spec.stop.threshold)
        mom_t = getattr(m, f"MomentumType_{sfx}")
        mom = getattr(m, f"MomentumConfig_{sfx}")(getattr(mom_t, _ENUM["momentum"][spec.momentum.type]), spec.momentum.rate)
        alr_t = getattr(m, f"AdaptiveLearningStrategy_{sfx}")
        alr = getattr(m, f"AdaptiveLearningRate_{sfx}")(
            getattr(alr_t, _ENUM["strategy"][spec.adaptive.strategy]),
            spec.adaptive.epsilon,
            spec.adaptive.beta1,
            spec.adaptive.beta2,
            spec.adaptive.final_lr,
        )
        opts = make_options(m, device, strict=strict_options, **options)
        kwargs = dict(
            layers=layers,
            learning_rate=spec.learning_rate,
            regularization=reg,
            backpropagation=m.BackPropagation.Standard,
            adaptive_learning_rate=alr,
            stop_criteria=stop,
            momentum=mom,
            options=opts,
        )
        Net = getattr(m, f"Network_{sfx}")
        if initial_weights is not None and initial_biases is not None:
            ws = [flatten_weights(np.asarray(W, dtype=self.np_dtype)) for W in initial_weights]
            bs = [np.ascontiguousarray(np.asarray(b, dtype=self.np_dtype)) for b in initial_biases]
            self.net = Net(initial_weights=ws, initial_biases=bs, **kwargs)
        elif seed is not None:
            self.net = Net(seed=int(seed), **kwargs)
        else:
            self.net = Net(**kwargs)
        self.device_name = self.net.device_name

    # ------------------------------------------------------------------ API
    @property
    def is_reference(self) -> bool:
        return self.mod is None

    def train(self, X: np.ndarray, Y: np.ndarray, batch_size: int, max_epochs: int) -> np.ndarray:
        X = np.ascontiguousarray(X, dtype=self.np_dtype)
        Y = np.ascontiguousarray(Y, dtype=self.np_dtype)
        if self.mod is None:
            return self.net.train(X, Y, batch_size, max_epochs)
        if X.ndim != 2 or Y.ndim != 2 or X.shape[0] != Y.shape[0]:
            raise ValueError("train expects X (N, n_in) and Y (N, n_out)")
        losses = self.net.train(X.reshape(-1), Y.reshape(-1), int(X.shape[0]), int(batch_size), int(max_epochs))
        return np.asarray(losses, dtype=np.float64)

    def predict(self, X: np.ndarray, batch_size: int | None = None) -> np.ndarray:
        X = np.ascontiguousarray(X, dtype=self.np_dtype)
        if self.mod is None:
            return self.net.predict(X, batch_size)
        if X.ndim != 2:
            raise ValueError("predict expects X (N, n_in)")
        out = self.net.predict(X.reshape(-1), int(X.shape[0]), int(batch_size or 0))
        return np.asarray(out, dtype=self.np_dtype).reshape(X.shape[0], self.spec.n_out)

    def _unflatten(self, flat_ws, flat_bs):
        sizes = self.spec.sizes
        Ws = [unflatten_weights(np.asarray(w), sizes[l + 1], sizes[l], self.np_dtype) for l, w in enumerate(flat_ws)]
        bs = [np.asarray(b, dtype=self.np_dtype) for b in flat_bs]
        return Ws, bs

    @property
    def weights(self) -> list[np.ndarray]:
        if self.mod is None:
            return self.net.weights
        return self._unflatten(self.net.weights, self.net.biases)[0]

    @property
    def biases(self) -> list[np.ndarray]:
        if self.mod is None:
            return self.net.biases
        return self._unflatten(self.net.weights, self.net.biases)[1]

    def history(self) -> tuple[list[list[np.ndarray]], list[list[np.ndarray]]]:
        if self.mod is None:
            return self.net.weights_biases()
        hw, hb = self.net.weights_biases
        out_w, out_b = [], []
        for ws, bs in zip(hw, hb):
            Ws, Bs = self._unflatten(ws, bs)
            out_w.append(Ws)
            out_b.append(Bs)
        return out_w, out_b

    def set_weights_biases(self, weights: Sequence[np.ndarray], biases: Sequence[np.ndarray]) -> None:
        if self.mod is None:
            self.net.set_weights_biases(weights, biases)
            return
        ws = [flatten_weights(np.asarray(W, dtype=self.np_dtype)) for W in weights]
        bs = [np.ascontiguousarray(np.asarray(b, dtype=self.np_dtype)) for b in biases]
        self.net.set_weights_biases(ws, bs)

    @property
    def profile(self) -> dict[str, Any] | None:
        if self.mod is None:
            return None
        return dict(self.net.profile)

    def reset_profile(self) -> None:
        if self.mod is not None:
            self.net.reset_profile()


def make_network(backend: str, spec: NetSpec, **kw) -> NetworkAdapter:
    return NetworkAdapter(backend, spec, **kw)
