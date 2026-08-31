import numpy as np
import pytest

from coverage_kernel import coverage_model as cm
from coverage_kernel import pooled_classifier as pc


@pytest.fixture
def physics():
    return cm.CoveragePhysics(n_sites=48, zeroed_zero_index=2, zeroed_full_index=0, absorbing_full_index=2)


def test_enforce_zero_at_zero_feature():
    probs = np.array([0.3, 0.3, 0.4])
    gated = cm.enforce_zero_at_zero_feature(probs, 0, zeroed_index=2)
    assert gated[2] == 0.0
    assert abs(gated.sum() - 1.0) < 1e-12
    # untouched away from feature==0
    ungated = cm.enforce_zero_at_zero_feature(probs, 5, zeroed_index=2)
    assert np.array_equal(ungated, probs)


def test_enforce_zero_at_full_feature():
    probs = np.array([0.2, 0.3, 0.5])
    gated = cm.enforce_zero_at_full_feature(probs, 48, 48, zeroed_index=0, absorbing_index=2)
    assert gated[0] == 0.0
    assert gated[1] == 0.3          # untouched
    assert abs(gated[2] - 0.7) < 1e-12   # absorbed all of class 0's mass
    assert abs(gated.sum() - 1.0) < 1e-12


def test_blend_stays_in_simplex(physics):
    def raw(c):
        return np.array([max(0.6 - 0.02 * c, 0.05), 0.1, 0.3 + 0.02 * c])

    for c in np.linspace(0, 60, 50):
        p = cm.blend_beyond_sampled_range(raw, c, trusted_max=16, feature_cap=physics.n_sites,
                                           zeroed_index=physics.zeroed_full_index,
                                           absorbing_index=physics.absorbing_full_index)
        assert np.all(p >= -1e-12)
        assert abs(p.sum() - 1.0) < 1e-9


def test_blend_reaches_zero_ads_at_cap(physics):
    def raw(c):
        return np.array([0.5, 0.1, 0.4])   # deliberately doesn't decay on its own

    p_cap = cm.blend_beyond_sampled_range(raw, physics.n_sites, trusted_max=16, feature_cap=physics.n_sites,
                                           zeroed_index=0, absorbing_index=2)
    assert p_cap[0] == 0.0
    assert abs(p_cap[1] - 0.1) < 1e-12   # untouched class


def test_blend_respects_custom_ceiling_below_trusted_max(physics):
    def raw(c):
        return np.array([0.5, 0.1, 0.4])

    # ceiling (10) below trusted_max (16): blend_lo=min(16,10)=10, blend_hi=max(16,10)=16
    p_below = cm.blend_beyond_sampled_range(raw, 5.0, trusted_max=16, feature_cap=physics.n_sites,
                                             zeroed_index=0, absorbing_index=2, ceiling=10.0)
    assert np.array_equal(p_below, raw(5.0))   # below blend_lo=10: untouched trusted raw value

    p_at_blend_lo = cm.blend_beyond_sampled_range(raw, 10.0, trusted_max=16, feature_cap=physics.n_sites,
                                                   zeroed_index=0, absorbing_index=2, ceiling=10.0)
    assert np.array_equal(p_at_blend_lo, raw(10.0))   # exactly at blend_lo: still the raw (untouched) value

    p_at_blend_hi = cm.blend_beyond_sampled_range(raw, 16.0, trusted_max=16, feature_cap=physics.n_sites,
                                                   zeroed_index=0, absorbing_index=2, ceiling=10.0)
    assert p_at_blend_hi[0] == 0.0   # at/past blend_hi=16: fully blended to the zeroed endpoint
    assert abs(p_at_blend_hi[2] - 0.9) < 1e-12   # absorbed class 0's 0.5 mass on top of its own 0.4


def test_sample_outcome_distribution():
    rng = np.random.default_rng(0)
    probs = np.array([0.2, 0.3, 0.5])
    counts = np.zeros(3)
    n = 20000
    for _ in range(n):
        idx, _ = cm.sample_outcome(probs, rng.random())
        counts[idx] += 1
    freq = counts / n
    assert np.max(np.abs(freq - probs)) < 0.02


def test_sample_outcome_boundary_cases():
    probs = np.array([0.2, 0.3, 0.5])
    assert cm.sample_outcome(probs, 0.0)[0] == 0
    assert cm.sample_outcome(probs, 0.1999999)[0] == 0
    assert cm.sample_outcome(probs, 0.2000001)[0] == 1
    assert cm.sample_outcome(probs, 0.999999)[0] == 2


def test_stochastic_round_matches_expectation():
    rng = np.random.default_rng(1)
    x = 6.68
    draws = np.array([cm.stochastic_round(x, rng.random()) for _ in range(50000)])
    assert abs(draws.mean() - x) < 0.02
    assert set(np.unique(draws)) <= {6, 7}


def test_map_area_density_to_coverage():
    # local_coverage = 2.1*3.0 = 6.3 -> floor=6, frac=0.3.
    # stochastic_round rounds up iff xi_round < frac, so xi_round=0.99 (>= frac) forces the floor,
    # and xi_round=0.0 (< frac, since frac>0) forces the ceiling.
    got_floor = cm.map_area_density_to_coverage(area_density=2.1, element_area=1.0, model_area=3.0, xi_round=0.99)
    assert got_floor == 6
    got_ceil = cm.map_area_density_to_coverage(area_density=2.1, element_area=1.0, model_area=3.0, xi_round=0.0)
    assert got_ceil == 7


def test_run_coverage_walk_bounded(physics):
    def probs_fn(c):
        return cm.blend_beyond_sampled_range(
            lambda cc: np.array([max(0.6 - 0.03 * cc, 0.0), 0.1, min(0.3 + 0.03 * cc, 0.9)]),
            c, trusted_max=16, feature_cap=physics.n_sites,
            zeroed_index=physics.zeroed_full_index, absorbing_index=physics.absorbing_full_index)

    trace = cm.run_coverage_walk(probs_fn, coverage_delta=[1, 0, -1], n_steps=5000, seed=0)
    assert trace.min() >= 0
    assert trace.max() <= physics.n_sites + 1e-6


def test_compute_ceiling_converges_and_is_bounded(physics):
    def raw(c):
        p = np.array([max(0.6 - 0.03 * c, 0.05), 0.1, 0.3 + 0.03 * c])
        return p / p.sum()

    ceiling = cm.compute_ceiling(raw, physics, coverage_max=16, coverage_delta=[1, 0, -1],
                                  n_steps=4000, n_tail=1000, seed=0)
    assert 0 < ceiling <= physics.n_sites


def test_fit_coverage_grid_model_end_to_end(physics):
    rng = np.random.default_rng(7)
    velocities = [2000.0, 5000.0, 9000.0]
    angles = [0.0, 40.0, 80.0]
    N = 1500
    event_axis_values = np.column_stack([rng.choice(velocities, N), rng.choice(angles, N)])
    coverage = rng.uniform(0, 16, N)
    logits = np.column_stack([0.05 * (16 - coverage), 0.02 * coverage, 0.1 * coverage])
    p = np.exp(logits)
    p /= p.sum(axis=1, keepdims=True)
    labels = np.array([rng.choice(3, p=pi) for pi in p])

    model = cm.fit_coverage_grid_model(
        [velocities, angles], event_axis_values, coverage, labels,
        [pc.SplineTerm(0, n_knots=4, degree=2)],
        n_classes=3, physics=physics, coverage_delta=[1, 0, -1],
        mc_steps=1500, mc_tail=400)

    probs = model.eval(np.array([5.0]), [4000.0, 30.0])
    assert abs(probs.sum() - 1.0) < 1e-9
    assert np.all(probs >= 0)
    # coverage=0 gate: zeroed_zero_index (2) must be exactly 0
    probs0 = model.eval(np.array([0.0]), [4000.0, 30.0])
    assert probs0[2] == 0.0


def test_fit_coverage_pooled_model_end_to_end(physics):
    rng = np.random.default_rng(8)
    velocities = [2000.0, 5000.0, 9000.0]
    angles = [0.0, 40.0, 80.0]
    N = 1500
    coverage = rng.uniform(0, 16, N)
    logits = np.column_stack([0.05 * (16 - coverage), 0.02 * coverage, 0.1 * coverage])
    p = np.exp(logits)
    p /= p.sum(axis=1, keepdims=True)
    labels = np.array([rng.choice(3, p=pi) for pi in p])

    def feature_fn(c, velocity, angle):
        ar = np.radians(angle)
        return np.array([c, (velocity * np.cos(ar)) ** 2, velocity * np.sin(ar)])

    velocity_events = rng.choice(velocities, N).astype(float)
    angle_events = rng.choice(angles, N).astype(float)
    X_raw = np.stack([feature_fn(c, v, a) for c, v, a in zip(coverage, velocity_events, angle_events)], axis=0)

    model = cm.fit_coverage_pooled_model(
        X_raw, labels,
        [pc.SplineTerm(0, n_knots=4, degree=2), pc.SplineTerm(1, n_knots=4, degree=2), pc.LinearTerm(2)],
        feature_fn, n_classes=3, physics=physics, coverage_delta=[1, 0, -1],
        ceiling_axes_values=[velocities, angles], mc_steps=1500, mc_tail=400)

    probs = model.probs(5.0, 4000.0, 30.0)
    assert abs(probs.sum() - 1.0) < 1e-9
    assert np.all(probs >= 0)
    probs0 = model.probs(0.0, 4000.0, 30.0)
    assert probs0[2] == 0.0
