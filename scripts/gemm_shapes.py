"""GEMM submission order in network.hpp: forward, hidden delta, gradients."""
MNIST = [('NN', 512, 256, 784), ('NN', 256, 256, 512), ('NN', 10, 256, 256),
         ('TN', 256, 256, 10), ('TN', 512, 256, 256),
         ('NT', 512, 784, 256), ('NT', 256, 512, 256), ('NT', 10, 256, 256)]
LARGE = [('NN', 4096, 4096, 256), ('NN', 4096, 4096, 4096), ('NN', 16, 4096, 4096),
         ('TN', 4096, 4096, 16), ('TN', 4096, 4096, 4096),
         ('NT', 4096, 256, 4096), ('NT', 4096, 4096, 4096), ('NT', 16, 4096, 4096)]
TARGETS = {('NN', 512, 256, 784), ('TN', 512, 256, 256), ('NT', 512, 784, 256), ('NN', 4096, 4096, 4096)}


def is_gemm(name):
    lower = name.lower()
    return any(k in lower for k in ('gemm', 'cijk')) and not any(k in lower for k in ('reduce', 'epilogue'))


def shapes(size):
    return MNIST if size == 'mnist' else LARGE


def driver_args(backend, blas, size, out, runtime):
    args = ['python', 'scripts/nsys_epoch.py', '--runtime', runtime, '--backend', backend,
            '--blas', blas, '--epochs', '1', '--out', str(out)]
    if size == 'mnist':
        return args + ['--samples', '256', '--batch', '256']
    return args + ['--workload', 'sweep-w4096-d2-b4096', '--samples', '4096', '--batch', '4096']
