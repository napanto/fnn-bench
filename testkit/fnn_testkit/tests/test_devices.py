"""Device enumeration / selection and build metadata of the compiled backends."""

import pytest

from fnn_testkit import backends as be
from fnn_testkit.reference import mlp

pytestmark = pytest.mark.compiled_only


def test_devices_and_build_info(backend):
    devs = be.list_devices(backend)
    assert devs, "no devices"
    for d in devs:
        for k in ("index", "name", "type", "backend"):
            assert k in d, k
        assert d["type"] in ("cpu", "gpu", "accelerator", "host")
    info = be.build_info(backend)
    for k in ("version", "compiler"):
        assert k in info, k
    mod = be.load_backend(backend)
    assert isinstance(mod.__version__, str) and mod.__version__


def test_device_selection(backend, device, backend_options):
    spec = mlp([2, 2], learning_rate=0.1)
    devs = be.list_devices(backend)
    names = {d["name"] for d in devs}
    net = be.NetworkAdapter(backend, spec, device=device, seed=1, **backend_options)
    assert net.device_name in names, (net.device_name, names)
    # explicit selection by index must give exactly that device (a device-specific
    # BLAS choice such as --blas rocblas cannot apply to every device: drop it)
    generic = {k: v for k, v in backend_options.items() if k != "blas"}
    for d in devs:
        by_index = be.NetworkAdapter(backend, spec, device=f"index:{d['index']}", seed=1, **generic)
        assert by_index.device_name == d["name"]
    with pytest.raises((ValueError, RuntimeError)):
        be.NetworkAdapter(backend, spec, device="no-such-device:99", seed=1, **generic)


def test_unknown_option_is_rejected(backend, device):
    spec = mlp([2, 2], learning_rate=0.1)
    with pytest.raises(AttributeError):
        be.NetworkAdapter(backend, spec, device=device, seed=1, no_such_option=True)
