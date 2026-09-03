"""Datasets used by the parity/integration tests and by the benchmark workloads.

All loaders return ``(X, Y)`` as float64 C-contiguous arrays of shape ``(N, n_in)``
and ``(N, n_out)``; backends convert to their own precision.  Downloads are cached
under ``$FNN_DATA_DIR`` (default ``~/.cache/fnn-testkit``).
"""

from __future__ import annotations

import gzip
import os
import struct
import urllib.request
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_MONK_DIR = _HERE / "data" / "monk"
_MONK_CARDINALITIES = (3, 3, 2, 3, 4, 2)  # a1..a6 -> 17 one-hot inputs


def cache_dir() -> Path:
    d = Path(os.environ.get("FNN_DATA_DIR", Path.home() / ".cache" / "fnn-testkit"))
    d.mkdir(parents=True, exist_ok=True)
    return d


# --------------------------------------------------------------------------- toy


def xor() -> tuple[np.ndarray, np.ndarray]:
    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    Y = np.array([[0.0], [1.0], [1.0], [0.0]])
    return X, Y


# --------------------------------------------------------------------------- MONK


def monk(problem: int = 1, split: str = "train") -> tuple[np.ndarray, np.ndarray]:
    """MONK's problems (UCI) with the usual one-hot encoding: 17 binary inputs, one
    binary target.  The files are vendored (public domain, ~30 kB)."""
    if problem not in (1, 2, 3) or split not in ("train", "test"):
        raise ValueError("monk(problem in {1,2,3}, split in {'train','test'})")
    path = _MONK_DIR / f"monks-{problem}.{split}"
    rows = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 7:
            continue
        rows.append([int(p) for p in parts[:7]])
    raw = np.asarray(rows, dtype=np.int64)  # class a1 a2 a3 a4 a5 a6
    X = np.zeros((raw.shape[0], sum(_MONK_CARDINALITIES)), dtype=np.float64)
    col = 0
    for j, card in enumerate(_MONK_CARDINALITIES):
        X[np.arange(raw.shape[0]), col + raw[:, 1 + j] - 1] = 1.0
        col += card
    Y = raw[:, :1].astype(np.float64)
    return np.ascontiguousarray(X), np.ascontiguousarray(Y)


# --------------------------------------------------------------------------- MNIST

_MNIST_URLS = (
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
)
_MNIST_FILES = {
    "train": ("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz"),
    "test": ("t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"),
}


def _download(name: str, dest: Path) -> None:
    last = None
    for base in _MNIST_URLS:
        try:
            urllib.request.urlretrieve(base + name, dest)
            return
        except Exception as exc:  # pragma: no cover - network
            last = exc
    raise RuntimeError(f"could not download {name}: {last}")


def _read_idx(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        ndim = magic & 0xFF
        shape = [n] + [struct.unpack(">I", f.read(4))[0] for _ in range(ndim - 1)]
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)


def mnist(split: str = "train", limit: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """MNIST as ``X in [0,1]`` of shape ``(N, 784)`` and one-hot ``Y`` of shape ``(N, 10)``.
    Downloaded once (11 MB) and cached as ``.npz``."""
    if split not in _MNIST_FILES:
        raise ValueError("split must be 'train' or 'test'")
    npz = cache_dir() / f"mnist-{split}.npz"
    if not npz.exists():
        imgs, labels = _MNIST_FILES[split]
        for name in (imgs, labels):
            p = cache_dir() / name
            if not p.exists():
                _download(name, p)
        X = _read_idx(cache_dir() / imgs).reshape(-1, 784)
        y = _read_idx(cache_dir() / labels)
        np.savez_compressed(npz, X=X, y=y)
    data = np.load(npz)
    X, y = data["X"], data["y"]
    if limit is not None:
        X, y = X[:limit], y[:limit]
    Xf = X.astype(np.float64) / 255.0
    Y = np.zeros((X.shape[0], 10), dtype=np.float64)
    Y[np.arange(X.shape[0]), y] = 1.0
    return np.ascontiguousarray(Xf), Y


# --------------------------------------------------------------------------- synthetic


def synthetic(n: int, n_in: int, n_out: int, seed: int = 0, hidden: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic regression data from a random tanh teacher network with a
    little Gaussian noise; targets are scaled to roughly ``[-1, 1]``."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, size=(n, n_in))
    A = rng.normal(0.0, 1.0 / np.sqrt(n_in), size=(hidden, n_in))
    c = rng.normal(0.0, 0.1, size=(hidden,))
    Bm = rng.normal(0.0, 1.0 / np.sqrt(hidden), size=(n_out, hidden))
    H = np.tanh(X @ A.T + c)
    Y = np.tanh(H @ Bm.T) + rng.normal(0.0, 0.01, size=(n, n_out))
    return np.ascontiguousarray(X), np.ascontiguousarray(Y)


def cup_like(n: int = 1000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Same *shape* as ML-CUP24 (12 inputs, 3 regression targets); the real CUP
    data is course-private and is not used."""
    return synthetic(n, 12, 3, seed=seed)


def load(name: str, **kw) -> tuple[np.ndarray, np.ndarray]:
    """Dispatch by name: xor, monk1/monk2/monk3 (+ ``split=``), mnist (+ ``split``,
    ``limit``), synthetic (+ ``n``, ``n_in``, ``n_out``, ``seed``), cup."""
    if name == "xor":
        return xor()
    if name.startswith("monk"):
        return monk(int(name[4:] or 1), **kw)
    if name == "mnist":
        return mnist(**kw)
    if name == "synthetic":
        return synthetic(**kw)
    if name == "cup":
        return cup_like(**kw)
    raise ValueError(f"unknown dataset {name!r}")
