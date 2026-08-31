import numpy as np
import pytest

from coverage_kernel import pooled_classifier as pc
from coverage_kernel import grid_classifier as gc
from coverage_kernel import coverage_model as cm
from coverage_kernel import export_hdf5 as eh


@pytest.fixture
def physics():
    return cm.CoveragePhysics(n_sites=48, zeroed_zero_index=2, zeroed_full_index=0, absorbing_full_index=2)


def _feature_fn(coverage, velocity, angle):
    ar = np.radians(angle)
    return np.array([coverage, (velocity * np.cos(ar)) ** 2, velocity * np.sin(ar)])


def test_pooled_round_trip(tmp_path, physics):
    rng = np.random.default_rng(20)
    velocities = [2000.0, 5000.0, 9000.0]
    angles = [0.0, 40.0, 80.0]
    N = 1500
    coverage = rng.uniform(0, 16, N)
    logits = np.column_stack([0.05 * (16 - coverage), 0.02 * coverage, 0.1 * coverage])
    p = np.exp(logits); p /= p.sum(axis=1, keepdims=True)
    labels = np.array([rng.choice(3, p=pi) for pi in p])
    velocity_events = rng.choice(velocities, N).astype(float)
    angle_events = rng.choice(angles, N).astype(float)
    X_raw = np.stack([_feature_fn(c, v, a) for c, v, a in zip(coverage, velocity_events, angle_events)], axis=0)

    model = cm.fit_coverage_pooled_model(
        X_raw, labels,
        [pc.SplineTerm(0, n_knots=4, degree=2), pc.SplineTerm(1, n_knots=4, degree=2), pc.LinearTerm(2)],
        _feature_fn, n_classes=3, physics=physics, coverage_delta=[1, 0, -1],
        ceiling_axes_values=[velocities, angles], mc_steps=1000, mc_tail=300)

    path = tmp_path / "pooled.h5"
    eh.export_pooled_model(str(path), model, coverage_delta=[1, 0, -1], model_area_nm2=3.179)
    loaded, coverage_delta, model_area_nm2 = eh.load_pooled_model(str(path), _feature_fn)

    assert model_area_nm2 == pytest.approx(3.179)
    assert list(coverage_delta) == [1, 0, -1]

    for _ in range(100):
        c = rng.choice([0.0, rng.uniform(0, 40)])
        v = rng.uniform(1000, 11000)
        a = rng.uniform(-10, 90)
        assert np.allclose(model.probs(c, v, a), loaded.probs(c, v, a), atol=1e-12)


def test_grid_round_trip(physics, tmp_path):
    rng = np.random.default_rng(21)
    velocities = [2000.0, 5000.0, 9000.0]
    angles = [0.0, 40.0, 80.0]
    N = 1500
    event_axis_values = np.column_stack([rng.choice(velocities, N), rng.choice(angles, N)])
    coverage = rng.uniform(0, 16, N)
    logits = np.column_stack([0.05 * (16 - coverage), 0.02 * coverage, 0.1 * coverage])
    p = np.exp(logits); p /= p.sum(axis=1, keepdims=True)
    labels = np.array([rng.choice(3, p=pi) for pi in p])

    axes = [gc.AxisSpec(np.array(velocities), edge_policy="linear_extrapolate"),
            gc.AxisSpec(np.array(angles), edge_policy="clip")]
    model = cm.fit_coverage_grid_model(
        axes, event_axis_values, coverage, labels, [pc.SplineTerm(0, n_knots=4, degree=2)],
        n_classes=3, physics=physics, coverage_delta=[1, 0, -1], mc_steps=1000, mc_tail=300)

    path = tmp_path / "grid.h5"
    eh.export_grid_model(str(path), model, coverage_delta=[1, 0, -1])
    loaded, coverage_delta, model_area_nm2 = eh.load_grid_model(str(path))

    assert model_area_nm2 is None
    assert list(coverage_delta) == [1, 0, -1]

    for _ in range(100):
        c = rng.choice([0.0, rng.uniform(0, 40)])
        v = rng.uniform(500, 11000)
        a = rng.uniform(-10, 90)
        assert np.allclose(model.eval(np.array([c]), [v, a]), loaded.eval(np.array([c]), [v, a]), atol=1e-12)


def test_round_trip_preserves_missing_class_masking(physics):
    """The tricky case the schema is specifically designed for: a cell/fit
    where a class is entirely absent from training must round-trip as
    "absent from the softmax" (present_classes/n_present_classes), not as
    a dense zero-coefficient row (which would leak non-zero probability)."""
    rng = np.random.default_rng(22)
    N = 300
    coverage = rng.uniform(0, 16, N)
    y = np.where(coverage < 8, 0, 2)   # class 1 never observed
    clf = pc.fit_pooled_classifier(coverage.reshape(-1, 1), y, [pc.SplineTerm(0, n_knots=4, degree=2)],
                                    n_classes=3, C=1.0)
    assert len(clf.present_classes) == 2   # sanity: this really is the missing-class case

    from coverage_kernel.export_hdf5 import _padded_present, _classifier_from_padded
    n_present, present_padded, coef_padded, intercept_padded = _padded_present(clf, n_classes=3)
    assert n_present == 2
    assert set(present_padded[:n_present]) == {0, 2}
    assert present_padded[n_present:].tolist() == [-1] * (3 - n_present)

    reconstructed = _classifier_from_padded(clf.terms, clf.knot_vectors, clf.scaler_mean, clf.scaler_scale,
                                             n_present, present_padded, coef_padded, intercept_padded, 3)
    test_x = np.array([[1.0], [7.0], [8.0], [15.0]])
    probs = reconstructed.eval(test_x)
    assert np.all(probs[:, 1] == 0.0)   # class 1 must still be exactly absent, not just near-zero
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert np.allclose(probs, clf.eval(test_x))
