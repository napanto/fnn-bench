"""pytest plugin: ``--backend``, ``--device``, ``--dtype``, ``--blas``, ``--option``
command-line options and the fixtures the shared test-suite is written against.

Run the suite against any backend with, e.g.::

    pytest --pyargs fnn_testkit --backend syclnn --device cpu --dtype double
    pytest --pyargs fnn_testkit --backend cudann --device cuda:0 --dtype float --option memory=managed
    pytest --pyargs fnn_testkit                       # NumPy reference checks itself
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

import numpy as np
import pytest

from . import backends as be
from .reference import NetSpec, ReferenceNetwork


def pytest_addoption(parser: pytest.Parser) -> None:
    g = parser.getgroup("fnn-testkit")
    g.addoption("--backend", default="numpy", help="numpy | syclnn | cudann | ompnn")
    g.addoption("--device", default="auto", help="device string understood by the backend (cpu, gpu, cuda:0, ...)")
    g.addoption("--dtype", default="both", help="double | float | both")
    g.addoption("--blas", default=None, help="BLAS backend option forwarded as options.blas")
    g.addoption(
        "--option",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="extra backend option (repeatable); values are parsed as Python literals",
    )
    g.addoption("--run-slow", action="store_true", default=False, help="run tests marked slow (MNIST)")
    g.addoption("--lenient", action="store_true", default=False, help="loosen parity tolerances 10x")


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "slow: long-running integration test (needs --run-slow)")
    config.addinivalue_line("markers", "reference_only: test that only makes sense for the NumPy reference")
    config.addinivalue_line("markers", "compiled_only: test that only makes sense for a compiled backend")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    backend = config.getoption("--backend")
    run_slow = config.getoption("--run-slow")
    for item in items:
        if "slow" in item.keywords and not run_slow:
            item.add_marker(pytest.mark.skip(reason="needs --run-slow"))
        if "reference_only" in item.keywords and backend != "numpy":
            item.add_marker(pytest.mark.skip(reason="reference-only test"))
        if "compiled_only" in item.keywords and backend == "numpy":
            item.add_marker(pytest.mark.skip(reason="needs a compiled backend"))


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "dtype" in metafunc.fixturenames:
        want = metafunc.config.getoption("--dtype")
        dtypes = ["double", "float"] if want == "both" else [want]
        metafunc.parametrize("dtype", dtypes, scope="session")


@dataclass(frozen=True)
class Tolerance:
    rtol: float
    atol: float

    def scaled(self, k: float) -> "Tolerance":
        return Tolerance(self.rtol * k, self.atol * k)


def _parse_options(raw: list[str]) -> dict:
    out = {}
    for item in raw:
        if "=" not in item:
            raise pytest.UsageError(f"--option expects KEY=VALUE, got {item!r}")
        k, v = item.split("=", 1)
        try:
            out[k] = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            out[k] = v
    return out


@pytest.fixture(scope="session")
def backend(request: pytest.FixtureRequest) -> str:
    name = request.config.getoption("--backend")
    if name != "numpy":
        try:
            be.load_backend(name)
        except ImportError as exc:
            pytest.exit(f"backend {name!r} cannot be imported: {exc}", returncode=3)
    return name


@pytest.fixture(scope="session")
def device(request: pytest.FixtureRequest, backend: str) -> str | None:
    dev = request.config.getoption("--device")
    return None if dev == "auto" else dev


@pytest.fixture(scope="session")
def backend_options(request: pytest.FixtureRequest) -> dict:
    opts = _parse_options(request.config.getoption("--option"))
    blas = request.config.getoption("--blas")
    if blas:
        opts["blas"] = blas
    return opts


@pytest.fixture
def tol(request: pytest.FixtureRequest, dtype: str) -> Tolerance:
    """Parity tolerance versus the float64 reference: different BLAS summation
    orders make bitwise equality impossible; these bounds are justified in the docs."""
    t = Tolerance(1e-9, 1e-11) if dtype == "double" else Tolerance(2e-4, 2e-5)
    if request.config.getoption("--lenient"):
        t = t.scaled(10.0)
    return t


@pytest.fixture
def make_net(backend: str, device: str | None, backend_options: dict, dtype: str):
    """Factory: ``make_net(spec, initial_weights=..., initial_biases=..., **opts)``."""

    def _make(spec: NetSpec, **kw) -> be.NetworkAdapter:
        opts = dict(backend_options)
        opts.update(kw.pop("options", {}))
        return be.NetworkAdapter(backend, spec, dtype=dtype, device=device, **opts, **kw)

    return _make


@pytest.fixture
def make_ref():
    """Factory for the float64 oracle with the same signature as ``make_net``."""

    def _make(spec: NetSpec, **kw) -> ReferenceNetwork:
        kw.pop("options", None)
        kw = {k: v for k, v in kw.items() if k in ("initial_weights", "initial_biases", "seed", "record_history", "shuffle", "shuffle_seed")}
        return ReferenceNetwork(spec, dtype=np.float64, **kw)

    return _make


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(12345)


def random_params(spec: NetSpec, rng: np.random.Generator, scale: float = 0.5):
    """Random initial weights/biases (as ``(n_out, n_in)`` / ``(n_out,)`` arrays)."""
    sizes = spec.sizes
    Ws = [rng.uniform(-scale, scale, size=(sizes[l + 1], sizes[l])) for l in range(len(sizes) - 1)]
    bs = [rng.uniform(-scale, scale, size=(sizes[l + 1],)) for l in range(len(sizes) - 1)]
    return Ws, bs


@pytest.fixture
def params():
    return random_params


def assert_close(actual, expected, tol: Tolerance, what: str = "") -> None:
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    assert actual.shape == expected.shape, f"{what}: shape {actual.shape} != {expected.shape}"
    if not np.allclose(actual, expected, rtol=tol.rtol, atol=tol.atol):
        diff = np.abs(actual - expected)
        rel = diff / np.maximum(np.abs(expected), 1e-300)
        i = np.unravel_index(int(np.argmax(diff)), diff.shape)
        raise AssertionError(
            f"{what}: max abs diff {diff.max():.3e} (rel {rel.max():.3e}) at {i}: "
            f"got {actual[i]!r}, expected {expected[i]!r}; tol rtol={tol.rtol} atol={tol.atol}"
        )
