"""
generate_reference_data.py
===========================
Builds small synthetic pooled and grid coverage models (same shape/physics
conventions as the real project: 3 outcome classes indexed
0=adsorption/1=O2 desorption/2=AO desorption -- the exact class *meaning*
doesn't matter here, only that the Fortran port reproduces the same numbers
the Python reference implementation produces), exports them to HDF5, and
dumps a battery of query points with their expected outputs to plain text
files that test_coverage_kernel.f90 reads and checks against.

This is a test-only tool, not part of the coverage_kernel library -- fixed
dimensions here (3 classes, specific grid values) are fine, unlike in the
library itself which must stay fully general.
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "python"))

from coverage_kernel import pooled_classifier as pc
from coverage_kernel import grid_classifier as gc
from coverage_kernel import coverage_model as cm
from coverage_kernel import export_hdf5 as eh
from coverage_kernel.features import physics_features

DATA_DIR = HERE / "data"
DATA_DIR.mkdir(exist_ok=True)

N_SITES = 16
PHYSICS = cm.CoveragePhysics(n_sites=N_SITES, zeroed_zero_index=1, zeroed_full_index=0, absorbing_full_index=2)
COVERAGE_DELTA = [1, -1, -1]

VELOCITIES = [2000.0, 5000.0, 9000.0]
ANGLES = [0.0, 40.0, 80.0]

QUERY_COVERAGES = [0.0, 1.0, 3.0, 6.0, 8.0, 12.0, 16.0, 20.0]
QUERY_VELOCITIES = [1000.0, 2000.0, 4000.0, 5000.0, 7000.0, 9000.0, 10000.0]
QUERY_ANGLES = [0.0, 20.0, 40.0, 60.0, 80.0, 89.0]


def synth_labels(rng, coverage, n_sites):
    logits = np.column_stack([
        0.30 * (n_sites - coverage),
        0.15 * coverage,
        0.05 * coverage + 1.0,
    ])
    p = np.exp(logits)
    p /= p.sum(axis=1, keepdims=True)
    return np.array([rng.choice(3, p=pi) for pi in p])


def build_pooled_model():
    rng = np.random.default_rng(1234)
    n = 6000
    coverage = rng.uniform(0, N_SITES, n)
    velocity = rng.choice(VELOCITIES, n)
    angle = rng.choice(ANGLES, n)
    labels = synth_labels(rng, coverage, N_SITES)

    x_raw = np.stack([physics_features(c, v, a) for c, v, a in zip(coverage, velocity, angle)], axis=0)
    terms = [pc.SplineTerm(0, n_knots=5, degree=2), pc.SplineTerm(1, n_knots=4, degree=2), pc.LinearTerm(2)]

    model = cm.fit_coverage_pooled_model(
        x_raw, labels, terms, physics_features,
        n_classes=3, physics=PHYSICS, coverage_delta=COVERAGE_DELTA,
        ceiling_axes_values=[VELOCITIES, ANGLES], mc_steps=4000, mc_tail=1000, mc_seed=7,
    )
    return model


def build_grid_model():
    rng = np.random.default_rng(4321)
    n = 6000
    velocity = rng.choice(VELOCITIES, n)
    angle = rng.choice(ANGLES, n)
    coverage = rng.uniform(0, N_SITES, n)
    labels = synth_labels(rng, coverage, N_SITES)

    axes = [gc.AxisSpec(np.array(VELOCITIES), edge_policy="linear_extrapolate"),
            gc.AxisSpec(np.array(ANGLES), edge_policy="clip")]
    event_axis_values = np.column_stack([velocity, angle])
    terms = [pc.SplineTerm(0, n_knots=5, degree=2)]

    model = cm.fit_coverage_grid_model(
        axes, event_axis_values, coverage, labels, terms,
        n_classes=3, physics=PHYSICS, coverage_delta=COVERAGE_DELTA,
        mc_steps=4000, mc_tail=1000, mc_seed=7, method="pchip",
    )
    return model


def dump_pooled_reference(model):
    eh.export_pooled_model(str(DATA_DIR / "pooled_test.h5"), model, coverage_delta=COVERAGE_DELTA,
                            model_area_nm2=3.5)
    lines = []
    for c in QUERY_COVERAGES:
        for v in QUERY_VELOCITIES:
            for a in QUERY_ANGLES:
                p = model.probs(c, v, a)
                lines.append(f"{c:.6f} {v:.6f} {a:.6f} " + " ".join(f"{pi:.10e}" for pi in p))
    (DATA_DIR / "reference_pooled.txt").write_text("\n".join(lines) + "\n")


def dump_grid_reference(model):
    eh.export_grid_model(str(DATA_DIR / "grid_test.h5"), model, coverage_delta=COVERAGE_DELTA)
    lines = []
    for c in QUERY_COVERAGES:
        for v in QUERY_VELOCITIES:
            for a in QUERY_ANGLES:
                p = model.eval(np.array([c]), [v, a])
                lines.append(f"{c:.6f} {v:.6f} {a:.6f} " + " ".join(f"{pi:.10e}" for pi in p))
    (DATA_DIR / "reference_grid.txt").write_text("\n".join(lines) + "\n")


def dump_sampling_reference(pooled_model):
    """A handful of (coverage, velocity, angle, xi) -> (outcome, delta_coverage)
    cases exercising SampleCoverageOutcome's inverse-CDF, including exact
    cumsum-boundary values where NumPy's searchsorted(side='right') behavior
    matters (see MOD_CoverageKernel's SampleOutcome docstring)."""
    cases = [(6.0, 5000.0, 40.0, 0.0), (6.0, 5000.0, 40.0, 0.999999),
             (0.0, 2000.0, 0.0, 0.5), (16.0, 9000.0, 80.0, 0.5)]
    lines = []
    for c, v, a, xi in cases:
        p = pooled_model.probs(c, v, a)
        idx, _ = cm.sample_outcome(p, xi)
        delta = COVERAGE_DELTA[idx]
        lines.append(f"{c:.6f} {v:.6f} {a:.6f} {xi:.6f} {idx} {delta:.6f}")
    (DATA_DIR / "reference_sampling.txt").write_text("\n".join(lines) + "\n")


def dump_density_reference():
    # model_area must match the pooled model's exported model_area_nm2 (see
    # dump_pooled_reference) since the Fortran test calls
    # MapAreaDensityToCoverage on the *loaded* pooled model, which reads its
    # own ModelAreaNm2 rather than taking it as a query argument.
    model_area = 3.5
    cases = [(2.0, 10.0, 0.0), (2.0, 10.0, 0.99), (1.25, 5.0, 0.5)]
    lines = []
    for area_density, element_area, xi in cases:
        local_density = area_density / element_area
        local_coverage = local_density * model_area
        n = cm.stochastic_round(local_coverage, xi)
        lines.append(f"{area_density:.6f} {element_area:.6f} {xi:.6f} {n}")
    (DATA_DIR / "reference_density.txt").write_text("\n".join(lines) + "\n")


REAL_MODELS_DIR = ROOT / "examples" / "python" / "models"
REAL_QUERY_COVERAGES = [0.0, 1.0, 5.0, 10.0, 20.0, 40.0, 48.0, 55.0]
REAL_QUERY_VELOCITIES = [1000.0, 2000.0, 5000.0, 6527.8, 9319.4, 10000.0, 11000.0]
REAL_QUERY_ANGLES = [0.0, 20.0, 40.0, 60.0, 80.0, 89.0]


def dump_real_pooled_reference():
    """Cross-validate against the actual production model fitted in
    train_models.ipynb on the real ~400k-event MLIP dataset (8 velocities x
    5 angles, n_sites=48) -- a stronger check than the small synthetic
    model above, since it exercises the real grid shape, real ceiling grid,
    and real fitted coefficients the Fortran port is actually meant to serve."""
    path = REAL_MODELS_DIR / "pooled_model.h5"
    if not path.exists():
        print(f"skipping real pooled model reference: {path} not found")
        return
    model, _, _ = eh.load_pooled_model(str(path), physics_features)
    eh.export_pooled_model(str(DATA_DIR / "pooled_real.h5"), model, coverage_delta=[1, 0, -1])
    lines = []
    for c in REAL_QUERY_COVERAGES:
        for v in REAL_QUERY_VELOCITIES:
            for a in REAL_QUERY_ANGLES:
                p = model.probs(c, v, a)
                lines.append(f"{c:.6f} {v:.6f} {a:.6f} " + " ".join(f"{pi:.10e}" for pi in p))
    (DATA_DIR / "reference_pooled_real.txt").write_text("\n".join(lines) + "\n")


def dump_real_grid_reference():
    path = REAL_MODELS_DIR / "grid_model.h5"
    if not path.exists():
        print(f"skipping real grid model reference: {path} not found")
        return
    model, _, _ = eh.load_grid_model(str(path))
    eh.export_grid_model(str(DATA_DIR / "grid_real.h5"), model, coverage_delta=[1, 0, -1])
    lines = []
    for c in REAL_QUERY_COVERAGES:
        for v in REAL_QUERY_VELOCITIES:
            for a in REAL_QUERY_ANGLES:
                p = model.eval(np.array([c]), [v, a])
                lines.append(f"{c:.6f} {v:.6f} {a:.6f} " + " ".join(f"{pi:.10e}" for pi in p))
    (DATA_DIR / "reference_grid_real.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    pooled_model = build_pooled_model()
    grid_model = build_grid_model()
    dump_pooled_reference(pooled_model)
    dump_grid_reference(grid_model)
    dump_sampling_reference(pooled_model)
    dump_density_reference()
    dump_real_pooled_reference()
    dump_real_grid_reference()
    print("Reference data written to", DATA_DIR)
