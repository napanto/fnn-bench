"""fnn-testkit: the NumPy reference implementation of the syclnn/cudann/ompnn
feed-forward network and the parity test-suite shared by the three libraries."""

from .backends import BACKENDS, NetworkAdapter, backend_available, build_info, list_devices, load_backend, make_network
from .reference import (
    ACTIVATIONS,
    AdaptiveLR,
    Layer,
    Momentum,
    NetSpec,
    ReferenceNetwork,
    Regularization,
    StopCriteria,
    activate,
    derivative,
    flatten_weights,
    mlp,
    unflatten_weights,
)

__version__ = "1.0.0"

__all__ = [
    "ACTIVATIONS",
    "AdaptiveLR",
    "BACKENDS",
    "Layer",
    "Momentum",
    "NetSpec",
    "NetworkAdapter",
    "ReferenceNetwork",
    "Regularization",
    "StopCriteria",
    "activate",
    "backend_available",
    "build_info",
    "derivative",
    "flatten_weights",
    "list_devices",
    "load_backend",
    "make_network",
    "mlp",
    "unflatten_weights",
]
