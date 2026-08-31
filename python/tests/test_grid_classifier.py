import numpy as np
import pytest
from scipy.interpolate import RegularGridInterpolator

from coverage_kernel import pooled_classifier as pc
from coverage_kernel import grid_classifier as gc


@pytest.fixture
def rng():
    return np.random.default_rng(6)


def _make_synthetic_grid(rng, n_v=6, n_a=5, n_classes=3):
    """Build a small synthetic grid of independently-fitted PooledClassifiers,
    one per cell, mirroring what fit_grid_classifier would produce."""
    velocities = np.linspace(2000, 9000, n_v)
    angles = np.linspace(0, 80, n_a)
    cell_classifiers = {}
    for vi in range(n_v):
        for ai in range(n_a):
            N = 200
            coverage = rng.uniform(0, 16, N)
            logits = np.column_stack([0.1 * coverage + vi, -0.05 * coverage + ai, 0.02 * coverage])
            p = np.exp(logits)
            p /= p.sum(axis=1, keepdims=True)
            y = np.array([rng.choice(n_classes, p=pp) for pp in p])
            clf = pc.fit_pooled_classifier(coverage.reshape(-1, 1), y,
                                            [pc.SplineTerm(0, n_knots=4, degree=2)], n_classes=n_classes)
            cell_classifiers[(vi, ai)] = clf
    return velocities, angles, cell_classifiers


def test_tensor_product_pchip_matches_scipy(rng):
    velocities, angles, cell_classifiers = _make_synthetic_grid(rng)
    axes = [gc.AxisSpec(velocities, edge_policy="clip"), gc.AxisSpec(angles, edge_policy="clip")]
    model = gc.GridClassifier(axes, cell_classifiers, n_classes=3, method="pchip")

    n_v, n_a = len(velocities), len(angles)
    max_err = 0.0
    for _ in range(200):
        coverage = rng.uniform(0.5, 15.5)
        velocity = rng.uniform(velocities[0], velocities[-1])
        angle = rng.uniform(angles[0], angles[-1])

        grid = np.array([[cell_classifiers[(vi, ai)].eval(np.array([coverage]))
                           for ai in range(n_a)] for vi in range(n_v)])
        ref = RegularGridInterpolator((velocities, angles), grid, method="pchip",
                                       bounds_error=False, fill_value=None)
        expected = ref([[velocity, angle]])[0]
        expected = np.clip(expected, 0, None)
        expected = expected / expected.sum() if expected.sum() > 0 else expected

        got = model.eval(np.array([coverage]), [velocity, angle])
        max_err = max(max_err, np.max(np.abs(got - expected)))
    assert max_err < 1e-9


def test_linear_method_matches_scipy(rng):
    velocities, angles, cell_classifiers = _make_synthetic_grid(rng)
    axes = [gc.AxisSpec(velocities, edge_policy="clip"), gc.AxisSpec(angles, edge_policy="clip")]
    model = gc.GridClassifier(axes, cell_classifiers, n_classes=3, method="linear")

    n_v, n_a = len(velocities), len(angles)
    coverage, velocity, angle = 5.0, 4000.0, 33.0
    grid = np.array([[cell_classifiers[(vi, ai)].eval(np.array([coverage]))
                       for ai in range(n_a)] for vi in range(n_v)])
    ref = RegularGridInterpolator((velocities, angles), grid, method="linear",
                                   bounds_error=False, fill_value=None)
    expected = ref([[velocity, angle]])[0]
    got = model.eval(np.array([coverage]), [velocity, angle])
    assert np.max(np.abs(got - expected)) < 1e-10


def test_clip_edge_policy_clamps(rng):
    velocities, angles, cell_classifiers = _make_synthetic_grid(rng)
    axes = [gc.AxisSpec(velocities, edge_policy="clip"), gc.AxisSpec(angles, edge_policy="clip")]
    model = gc.GridClassifier(axes, cell_classifiers, n_classes=3, method="pchip")

    coverage = 5.0
    at_edge = model.eval(np.array([coverage]), [velocities[-1], angles[-1]])
    beyond = model.eval(np.array([coverage]), [velocities[-1] + 5000, angles[-1] + 30])
    assert np.max(np.abs(at_edge - beyond)) < 1e-9


def test_linear_extrapolate_edge_policy_continues_slope(rng):
    velocities, angles, cell_classifiers = _make_synthetic_grid(rng)
    axes = [gc.AxisSpec(velocities, edge_policy="linear_extrapolate"), gc.AxisSpec(angles, edge_policy="clip")]
    model = gc.GridClassifier(axes, cell_classifiers, n_classes=3, method="pchip")

    coverage = 5.0
    p_last = model.eval(np.array([coverage]), [velocities[-1], 40.0])
    p_beyond = model.eval(np.array([coverage]), [velocities[-1] + 500, 40.0])
    # should differ (extrapolating), not clamp to the same value as "clip" would
    assert np.max(np.abs(p_last - p_beyond)) > 1e-6


def test_fit_grid_classifier_end_to_end(rng):
    velocities = [2000.0, 5000.0, 9000.0]
    angles = [0.0, 40.0, 80.0]
    N = 2000
    event_axis_values = np.column_stack([rng.choice(velocities, N), rng.choice(angles, N)])
    coverage = rng.uniform(0, 16, N)
    logits = np.column_stack([0.05 * (16 - coverage), 0.02 * coverage, 0.1 * coverage])
    p = np.exp(logits)
    p /= p.sum(axis=1, keepdims=True)
    labels = np.array([rng.choice(3, p=pi) for pi in p])

    model = gc.fit_grid_classifier([velocities, angles], event_axis_values, coverage, labels,
                                    [pc.SplineTerm(0, n_knots=4, degree=2)], n_classes=3)
    probs = model.eval(np.array([5.0]), [4000.0, 30.0])
    assert probs.shape == (3,)
    assert np.all(probs >= 0)
    assert abs(probs.sum() - 1.0) < 1e-9

    # events snapped to a noisy-but-close axis value should land in the same cell
    noisy_axis_values = event_axis_values + np.array([[0.5, 1e-6]])
    idx_clean = gc._nearest_indices([np.array(velocities), np.array(angles)], event_axis_values)
    idx_noisy = gc._nearest_indices([np.array(velocities), np.array(angles)], noisy_axis_values)
    assert np.array_equal(idx_clean, idx_noisy)
