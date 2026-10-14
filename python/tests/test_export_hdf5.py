import h5py
import numpy as np
import pytest

from coverage_kernel import spline_classifier as sc
from coverage_kernel import grid_classifier as gc
from coverage_kernel import coverage_model as cm
from coverage_kernel import export_hdf5 as eh
from coverage_kernel import nn
from coverage_kernel import load_model


@pytest.fixture
def physics():
    return cm.CoveragePhysics(n_sites=48, zeroed_zero_index=2, zeroed_full_index=0, absorbing_full_index=2)


def _toy_nn_model(physics, rng, n_members=2, hidden=(5, 4), with_ceiling=True):
    members = []
    for _ in range(n_members):
        sizes = [3, *hidden, 3]
        layers = [(rng.normal(size=(i, o)), rng.normal(size=o), "silu" if k < len(sizes) - 2 else "linear")
                  for k, (i, o) in enumerate(zip(sizes[:-1], sizes[1:]))]
        members.append(nn.MLP(layers, [7000.0 + rng.normal(), 40.0 + rng.normal()], [1400.0, 28.0]))
    ens = nn.NNEnsemble(members, physics.n_sites, offset_class=2, zeroed_zero_index=physics.zeroed_zero_index)
    axes = [[5000.0, 7000.0, 9000.0], [0.0, 40.0, 80.0]]
    grid = rng.uniform(8, 22, (3, 3)) if with_ceiling else None
    return nn.NNEnsembleCoverageModel(ens, physics, 16.0, axes if with_ceiling else None, grid,
                                      coverage_delta=[1, 0, -1], model_area_nm2=3.179,
                                      metadata={"description": "toy", "n_members": n_members})


def _toy_grid_model(physics, seed):
    rng = np.random.default_rng(seed)
    velocities, angles = [2000.0, 5000.0, 9000.0], [0.0, 40.0, 80.0]
    N = 1500
    event_axis_values = np.column_stack([rng.choice(velocities, N), rng.choice(angles, N)])
    coverage = rng.uniform(0, 16, N)
    logits = np.column_stack([0.05 * (16 - coverage), 0.02 * coverage, 0.1 * coverage])
    p = np.exp(logits); p /= p.sum(axis=1, keepdims=True)
    labels = np.array([rng.choice(3, p=pi) for pi in p])
    axes = [gc.AxisSpec(np.array(velocities), edge_policy="linear_extrapolate"),
            gc.AxisSpec(np.array(angles), edge_policy="clip")]
    return cm.fit_coverage_grid_model(axes, event_axis_values, coverage, labels, [sc.SplineTerm(0, n_knots=4, degree=2)],
                                      n_classes=3, physics=physics, coverage_delta=[1, 0, -1],
                                      mc_steps=1000, mc_tail=300, model_area_nm2=3.179)


def test_grid_round_trip(physics, tmp_path):
    rng = np.random.default_rng(21)
    model = _toy_grid_model(physics, 21)
    path = tmp_path / "grid.h5"
    eh.export_grid_model(str(path), model)
    loaded = eh.load_grid_model(str(path))
    assert isinstance(load_model(str(path)), cm.GridCoverageModel)
    assert loaded.model_area_nm2 == pytest.approx(3.179)
    assert list(loaded.coverage_delta) == [1, 0, -1]
    assert loaded.physics == physics
    for _ in range(100):
        c = rng.choice([0.0, rng.uniform(0, 40)])
        v, a = rng.uniform(500, 11000), rng.uniform(-10, 90)
        assert np.allclose(model.probs(c, v, a), loaded.probs(c, v, a), atol=1e-12)


def test_nn_round_trip(physics, tmp_path):
    rng = np.random.default_rng(5)
    model = _toy_nn_model(physics, rng)
    path = tmp_path / "nn.h5"
    eh.export_nn_model(str(path), model)
    loaded = load_model(str(path))
    assert isinstance(loaded, nn.NNEnsembleCoverageModel)
    assert loaded.physics == physics and loaded.coverage_max == 16.0
    assert loaded.model_area_nm2 == pytest.approx(3.179)
    assert np.array_equal(loaded.ceiling_grid, model.ceiling_grid)
    assert loaded.metadata["description"] == "toy"
    for _ in range(100):
        c = rng.choice([0.0, rng.uniform(0, 50)])
        v, a = rng.uniform(3000, 12000), rng.uniform(0, 90)
        assert np.allclose(model.probs(c, v, a), loaded.probs(c, v, a), atol=1e-14)


def test_nn_round_trip_without_ceiling(physics, tmp_path):
    model = _toy_nn_model(physics, np.random.default_rng(6), n_members=1, with_ceiling=False)
    path = tmp_path / "nn1.h5"
    eh.export_nn_model(str(path), model)
    loaded = eh.load_nn_model(str(path))
    assert loaded.ceiling_grid is None and loaded.members == []
    assert np.allclose(loaded.probs(30.0, 7000.0, 30.0), model.probs(30.0, 7000.0, 30.0), atol=1e-14)


def test_readers_check_version_and_scheme(physics, tmp_path):
    path = tmp_path / "nn.h5"
    eh.export_nn_model(str(path), _toy_nn_model(physics, np.random.default_rng(7)))
    with pytest.raises(ValueError):
        eh.load_grid_model(str(path))
    with h5py.File(path, "a") as f:
        f.attrs["format_version"] = np.array("9.9", dtype="S32")
    with pytest.raises(ValueError):
        eh.load_nn_model(str(path))


def test_missing_class_survives_round_trip(physics, tmp_path):
    """A cell where a class never occurred must come back as 'absent from the softmax'
    (present_classes / n_present_classes), not as a zero coefficient row."""
    rng = np.random.default_rng(22)
    coverage = rng.uniform(0, 16, 300)
    y = np.where(coverage < 8, 0, 2)   # class 1 never observed
    clf = sc.fit_spline_classifier(coverage.reshape(-1, 1), y, [sc.SplineTerm(0, n_knots=4, degree=2)], n_classes=3)
    assert len(clf.present_classes) == 2
    cell = cm.CoverageCell(clf, physics, coverage_max=16.0, ceiling=None)
    grid = gc.GridClassifier([gc.AxisSpec(np.array([5000.0])), gc.AxisSpec(np.array([40.0]))], {(0, 0): cell}, 3)
    model = cm.GridCoverageModel(grid, physics, coverage_delta=[1, 0, -1])
    path = tmp_path / "missing.h5"
    eh.export_grid_model(str(path), model)
    loaded = eh.load_grid_model(str(path))
    for c in (1.0, 7.0, 8.0, 15.0):
        p = loaded.probs(c, 5000.0, 40.0)
        assert p[1] == 0.0
        assert np.allclose(p, model.probs(c, 5000.0, 40.0), atol=1e-14)
