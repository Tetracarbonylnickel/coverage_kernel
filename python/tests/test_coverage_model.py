import numpy as np
import pytest

from coverage_kernel import coverage_model as cm
from coverage_kernel import spline_classifier as sc


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


def test_blend_with_ceiling_below_trusted_max(physics):
    """The fitted model is used up to trusted_max even if the ceiling is lower;
    above trusted_max the no-adsorption endpoint applies at once."""
    def raw(c):
        return np.array([0.5 - 0.01 * c, 0.1, 0.4 + 0.01 * c])

    kw = dict(trusted_max=16, feature_cap=physics.n_sites, zeroed_index=0, absorbing_index=2, ceiling=10.0)
    assert cm.blend_limits(16, physics.n_sites, 10.0) == (16, 16)
    for c in (5.0, 10.0, 12.0, 16.0):
        assert np.array_equal(cm.blend_beyond_sampled_range(raw, c, **kw), raw(c))
    p = cm.blend_beyond_sampled_range(raw, 16.5, **kw)
    assert p[0] == 0.0 and abs(p[2] - (raw(16.0)[0] + raw(16.0)[2])) < 1e-12   # endpoint built at 16


def test_blend_with_ceiling_above_trusted_max(physics):
    def raw(c):
        return np.array([0.5, 0.1, 0.4])

    kw = dict(trusted_max=16, feature_cap=physics.n_sites, zeroed_index=0, absorbing_index=2, ceiling=20.0)
    assert cm.blend_limits(16, physics.n_sites, 20.0) == (16, 20.0)
    assert np.array_equal(cm.blend_beyond_sampled_range(raw, 16.0, **kw), raw(16.0))
    assert np.allclose(cm.blend_beyond_sampled_range(raw, 18.0, **kw), [0.25, 0.1, 0.65])   # half way
    assert np.allclose(cm.blend_beyond_sampled_range(raw, 25.0, **kw), [0.0, 0.1, 0.9])


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


def test_map_adsorbate_count_to_coverage():
    # 4.2 atoms on 2.0 nm^2 -> 2.1 per nm^2 -> 6.3 on a 3.0 nm^2 patch: floor 6, fraction 0.3.
    # stochastic_round rounds up iff xi_round < fraction.
    assert cm.map_adsorbate_count_to_coverage(4.2, element_area=2.0, model_area=3.0, xi_round=0.99) == 6
    assert cm.map_adsorbate_count_to_coverage(4.2, element_area=2.0, model_area=3.0, xi_round=0.0) == 7


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
        [sc.SplineTerm(0, n_knots=4, degree=2)],
        n_classes=3, physics=physics, coverage_delta=[1, 0, -1],
        mc_steps=1500, mc_tail=400)

    probs = model.probs(5.0, 4000.0, 30.0)
    assert abs(probs.sum() - 1.0) < 1e-9
    assert np.all(probs >= 0)
    assert np.allclose(model.probs_batch([5.0], 4000.0, 30.0)[0], probs)
    assert np.allclose(model.probs_fn(4000.0, 30.0)(5.0), probs)
    # coverage=0 gate: zeroed_zero_index (2) must be exactly 0
    assert model.probs(0.0, 4000.0, 30.0)[2] == 0.0
    assert model.members == []


def test_stationary_distribution_matches_long_walk():
    def probs_fn(c):
        p = np.array([max(0.7 - 0.05 * c, 0.0), 0.1, min(0.2 + 0.05 * c, 0.9)])
        return p / p.sum()

    P = np.array([probs_fn(c) for c in range(49)])
    pi = cm.stationary_distribution(P, [1, 0, -1])
    exact_mean = pi @ np.arange(49)
    trace = cm.run_coverage_walk(probs_fn, [1, 0, -1], n_steps=200000, seed=3)
    assert abs(trace[1000:].mean() - exact_mean) < 0.05
    assert abs(pi.sum() - 1.0) < 1e-12


def test_stationary_distribution_rejects_other_coverage_changes():
    with pytest.raises(ValueError):
        cm.stationary_distribution(np.full((5, 3), 1 / 3), [1, 0, -2])


def test_net_flux_crossing_analytic():
    # p_ads - p_O2 = 0.6 - 0.05 c  ->  crossing at c = 12
    def fn(c):
        return np.array([0.7 - 0.025 * c, 0.2, 0.1 + 0.025 * c])

    assert abs(cm.net_flux_crossing(fn, [1, 0, -1]) - 12.0) < 1e-9
    assert np.isnan(cm.net_flux_crossing(lambda c: np.array([0.8, 0.1, 0.1]), [1, 0, -1]))


def test_ceiling_model_without_ceiling_blends_to_n_sites(physics):
    def raw_batch(c, v, a):
        c = np.asarray(c, float)
        return np.stack([np.full(c.shape, 0.5), np.full(c.shape, 0.2), np.full(c.shape, 0.3)], 1)

    m = cm.CeilingCoverageModel(raw_batch, physics, coverage_max=16, ceiling_axes=None, ceiling_grid=None)
    assert m.ceiling(7000.0, 40.0) is None
    assert m.probs(0.0, 7000.0, 40.0)[2] == 0.0                       # zero-at-zero gate
    assert np.allclose(m.probs(10.0, 7000.0, 40.0), [0.5, 0.2, 0.3])   # inside the sampled range
    assert m.probs(48.0, 7000.0, 40.0)[0] == 0.0                      # no adsorption at full coverage
