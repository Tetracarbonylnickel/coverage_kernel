"""
Integration check against the real, already-fitted models shipped in the
old coverage_experiment/coverage_model package. Not required for
coverage_kernel to be usable on its own (see the "runs anywhere on synthetic
data" convention the rest of this test suite follows) -- this is an extra
regression check confirming the rewrite reproduces the original's validated
numerics, skipped automatically if that external package/data isn't present
(e.g. a fresh checkout of coverage_kernel alone, without the sibling repo).
"""

import sys
from pathlib import Path

import numpy as np
import pytest

_OLD_PKG = Path("/users/fs3/segreto/miklas_imps/coverage_experiment/coverage_model")

pytestmark = pytest.mark.skipif(
    not (_OLD_PKG / "examples/models/piecewise_global/model.joblib").exists(),
    reason="old coverage_model package/shipped models not available in this checkout",
)


@pytest.fixture(scope="module")
def old_pkg_on_path():
    sys.path.insert(0, str(_OLD_PKG))
    yield
    sys.path.remove(str(_OLD_PKG))


def _build_piecewise_grid_classifier(state):
    from coverage_kernel import pooled_classifier as pc
    from coverage_kernel import grid_classifier as gc
    from coverage_kernel import coverage_model as cm

    velocities, angles = state["velocities"], state["angles"]
    n_sites = state["n_sites"]
    physics = cm.CoveragePhysics(n_sites=n_sites, zeroed_zero_index=2, zeroed_full_index=0, absorbing_full_index=2)

    cell_classifiers = {}
    for vi, v in enumerate(velocities):
        for ai, a in enumerate(angles):
            pipe = state["cell_pipelines"][(v, a)]
            spline, scaler, logit = pipe.named_steps["spline"], pipe.named_steps["scale"], pipe.named_steps["logit"]
            raw_clf = pc.PooledClassifier(
                [pc.SplineTerm(0, n_knots=5, degree=2)], [spline.bsplines_[0].t],
                scaler.mean_, scaler.scale_, logit.coef_, logit.intercept_, logit.classes_, n_classes=3)
            cell_classifiers[(vi, ai)] = cm.CoverageCell(
                raw_clf, physics, state["cell_coverage_max"][(v, a)], state["cell_ceiling"][(v, a)])

    axes = [gc.AxisSpec(np.array(velocities), edge_policy="linear_extrapolate"),
            gc.AxisSpec(np.array(angles), edge_policy="clip")]
    return gc.GridClassifier(axes, cell_classifiers, n_classes=3, method="pchip"), velocities


def test_piecewise_global_matches_shipped_model_in_range(old_pkg_on_path):
    import joblib
    from coverage_model.fitting import load_coverage_model

    path = _OLD_PKG / "examples/models/piecewise_global/model.joblib"
    orig_model = load_coverage_model(str(path))
    state = joblib.load(str(path))["state"]
    mine, velocities = _build_piecewise_grid_classifier(state)

    rng = np.random.default_rng(9)
    max_err = 0.0
    for _ in range(500):
        coverage = rng.choice([0.0, rng.uniform(0, 50)])
        velocity = rng.uniform(velocities[0], velocities[-1])  # stay in-range: see design-deviation note below
        angle = rng.uniform(-20, 100)
        expected = orig_model.probs(coverage, angle, velocity)
        got = mine.eval(np.array([coverage]), [velocity, angle])
        max_err = max(max_err, np.max(np.abs(got - expected)))

    # In-range queries (both axes) should match to floating-point precision.
    # Out-of-range-velocity queries deliberately differ from the original by
    # design (decoupled per-axis interpolation method, agreed as the
    # intentional generalization -- see coverage_kernel design notes) and
    # are excluded here on purpose, not because they're expected to match.
    assert max_err < 1e-6


def test_global_matches_shipped_model(old_pkg_on_path):
    import joblib
    from coverage_model.fitting import load_coverage_model
    from coverage_kernel import pooled_classifier as pc
    from coverage_kernel import coverage_model as cm

    path = _OLD_PKG / "examples/models/global/model.joblib"
    orig_model = load_coverage_model(str(path))
    state = joblib.load(str(path))["state"]
    pipe = state["pipeline"]
    ct, scaler, logit = pipe.named_steps["spline"], pipe.named_steps["scale"], pipe.named_steps["logit"]
    cov_knots = ct.transformers_[0][1].bsplines_[0].t
    en_knots = ct.transformers_[1][1].bsplines_[0].t

    raw_clf = pc.PooledClassifier(
        [pc.SplineTerm(0, n_knots=5, degree=2), pc.SplineTerm(1, n_knots=5, degree=2), pc.LinearTerm(2)],
        [cov_knots, en_knots, None], scaler.mean_, scaler.scale_, logit.coef_, logit.intercept_,
        logit.classes_, n_classes=3)

    def feature_fn(coverage, velocity, angle):
        ar = np.radians(angle)
        v_normal, v_parallel = velocity * np.cos(ar), velocity * np.sin(ar)
        return np.array([coverage, v_normal ** 2, v_parallel])

    physics = cm.CoveragePhysics(n_sites=state["n_sites"], zeroed_zero_index=2, zeroed_full_index=0,
                                  absorbing_full_index=2)
    ceiling_velocities, ceiling_angles = state["ceiling_velocities"], state["ceiling_angles"]
    ceiling_grid = np.array([[state["cell_ceiling"][(v, a)] for a in ceiling_angles] for v in ceiling_velocities])

    mine = cm.PooledCoverageModel(raw_clf, feature_fn, physics, state["coverage_max"],
                                  ceiling_axes=[np.array(ceiling_velocities), np.array(ceiling_angles)],
                                  ceiling_grid=ceiling_grid)

    rng = np.random.default_rng(10)
    max_err = 0.0
    for _ in range(500):
        coverage = rng.choice([0.0, rng.uniform(0, 50)])
        velocity = rng.uniform(1000, 12000)
        angle = rng.uniform(-10, 90)
        expected = orig_model.probs(coverage, angle, velocity)
        got = mine.probs(coverage, velocity, angle)
        max_err = max(max_err, np.max(np.abs(got - expected)))
    assert max_err < 1e-6
