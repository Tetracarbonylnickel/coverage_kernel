"""coverage_kernel: coverage-dependent branching-probability models for DSMC gas-surface boundary conditions.

    from coverage_kernel import load_model
    model = load_model("branching_nn_ensemble_weighted.h5")
    p = model.probs(coverage, velocity, angle)

See README.md for the models, the coverage physics and the Fortran port.
"""

__version__ = "0.2.0"


def load_model(path):
    """Load a grid or NN-ensemble model file (see export_hdf5.load_model)."""
    from .export_hdf5 import load_model as _load
    return _load(path)
