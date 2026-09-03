# fnn-testkit

NumPy float64 reference implementation of the feed-forward network trained by
[syclnn](https://github.com/napanto/syclnn), [cudann](https://github.com/napanto/cudann)
and [ompnn](https://github.com/napanto/ompnn), plus the pytest suite that every
backend has to pass (the *parity oracle*).

```sh
pip install "fnn-testkit @ git+https://github.com/napanto/fnn-bench#subdirectory=testkit"

pytest --pyargs fnn_testkit                                   # the oracle checks itself
pytest --pyargs fnn_testkit --backend syclnn --device cpu     # a compiled backend, both dtypes
pytest --pyargs fnn_testkit --backend cudann --device cuda:0 --dtype float --option memory=managed
pytest --pyargs fnn_testkit --backend ompnn --run-slow        # + MNIST
```

Options: `--backend numpy|syclnn|cudann|ompnn`, `--device <string>` (backend
specific: `cpu`, `gpu`, `cuda:0`, `index:2`, ...), `--dtype double|float|both`,
`--blas <name>` (forwarded as `options.blas`), `--option KEY=VALUE` (repeatable,
any `Options` attribute, e.g. an ablation switch), `--run-slow`, `--lenient`.

## What is tested

| file | contents |
|---|---|
| `test_reference.py` | the oracle against closed forms and finite differences; `std::mt19937` clone |
| `test_activations.py` | f and f' of every activation inside the backend kernels |
| `test_forward.py` | single-layer/deep forward, transparent `predict` batching, read-back of weights |
| `test_gradients.py` | back-prop gradients vs the oracle and vs finite differences of the backend's own loss |
| `test_optimizers.py` | 5 strategies × 2 momenta × 4 regularisations for 3 epochs with a remainder batch; Adam state across `train()` calls |
| `test_regularization.py` | penalty in the epoch loss, unused λ ignored, biases unregularised |
| `test_training.py` | remainder batch, `batch_size > N`, error handling, stop criteria epoch counts, history semantics, `set_weights_biases`, seeded determinism, float vs double |
| `test_parity.py` | MONK-1 / XOR / CUP-like / ReLU-RMSProp from identical initial weights; shuffle permutation |
| `test_integration.py` | XOR converges, MONK-1 ≥ 97 % test accuracy, MNIST ≥ 95 % (slow) |
| `test_profile.py` | phase profiler keys, counts, reset, disabled |
| `test_devices.py` | device list, selection by index, build info, option validation |

## Tolerances

The oracle accumulates in float64 and every BLAS library sums in its own order,
so bit-exact equality is impossible.  Parity uses `rtol 1e-9 / atol 1e-11` for
double and `rtol 2e-4 / atol 2e-5` for float; the gradient tests allow 10×
because they subtract two O(1) weights.  `--lenient` scales them by 10.

## Semantics fixed by the oracle (version 0.2 of the libraries)

* `max_epochs == 0`, `batch_size == 0`, empty datasets and shape mismatches raise;
  `batch_size > N` is clamped to `N`.
* History: snapshot at construction and after `set_weights_biases`; one per epoch
  when `record_history` is on, otherwise only the final state is appended.
* The regularisation penalty is evaluated on the weights *after* the epoch.
* Adam's step counter restarts at every `train()`, its moments persist.
* Shuffling (off by default) uses `std::mt19937(seed)` and Fisher–Yates with
  `j = i + mt() % (n - i)`.
