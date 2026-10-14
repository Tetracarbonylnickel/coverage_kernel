"""
coverage_model.py
=================
Domain layer: the coverage-specific physics on top of the raw models
(the spline classifiers of the grid scheme in grid_classifier.py, the neural
networks in nn.py). This module is where "coverage", "adsorption" and
"n_sites" first appear; the class indices are never hardcoded.

Physics, for any number of outcome classes:

1. Zero-at-zero-coverage gate: one outcome (recombinative O2 desorption) is
   impossible on an empty surface -- its probability is set to zero at
   coverage 0 and the rest renormalized.
2. Blend beyond the sampled coverage range: one outcome (adsorption) is
   impossible once every site is occupied. Instead of trusting a raw model
   far beyond the coverages it was trained on, the raw probabilities are
   used up to blend_lo and then blended linearly toward the endpoint in which
   the adsorption probability is handed to the absorbing class, reached at
   blend_hi (see blend_limits). Blending two valid probability vectors cannot
   leave the simplex.
3. The far end of the blend ("ceiling") is a self-consistent Monte Carlo
   fixed point per sampled condition: run the blended model's own coverage
   walk and take mean + k * standard deviation of its tail (compute_ceiling).

Model wrappers (all expose probs(coverage, velocity, angle),
probs_batch(coverages, velocity, angle) and probs_fn(velocity, angle)):
  - GridCoverageModel: grid of per-condition spline classifiers, each wrapped
    in a CoverageCell (blend per cell first, then interpolate across cells);
  - CeilingCoverageModel: any raw model given as a vectorized callable
    raw_batch(coverages, velocity, angle), with a ceiling grid over
    (velocity, angle) interpolated multilinearly; nn.NNEnsembleCoverageModel
    builds on it.

Analysis helpers: run_coverage_walk (the per-impact loop a DSMC boundary
condition runs), stationary_distribution (exact long-run coverage
distribution of that walk) and net_flux_crossing (coverage where the
expected coverage change per impact vanishes).
"""

from dataclasses import dataclass

import numpy as np
from joblib import Parallel, delayed

from . import interp1d
from .grid_classifier import fit_grid_classifier, GridClassifier


# --------------------------------------------------------------------------- #
# Gates and blend                                                             #
# --------------------------------------------------------------------------- #

def enforce_zero_at_zero_feature(probs, feature_value, zeroed_index):
    """Hard-zero probs[zeroed_index] when feature_value == 0, renormalizing the rest."""
    probs = np.asarray(probs, dtype=float)
    if feature_value == 0 and probs[zeroed_index] != 0:
        probs = probs.copy()
        probs[zeroed_index] = 0.0
        total = probs.sum()
        if total > 0:
            probs = probs / total
    return probs


def enforce_zero_at_full_feature(probs, feature_value, feature_cap, zeroed_index, absorbing_index):
    """Hard-zero probs[zeroed_index] once feature_value >= feature_cap, handing all
    of its mass to probs[absorbing_index] (every other class untouched)."""
    probs = np.asarray(probs, dtype=float)
    if feature_value >= feature_cap and probs[zeroed_index] != 0:
        probs = probs.copy()
        probs[absorbing_index] += probs[zeroed_index]
        probs[zeroed_index] = 0.0
    return probs


def blend_limits(trusted_max, feature_cap, ceiling=None):
    """Coverage range (blend_lo, blend_hi) of the blend toward the no-adsorption endpoint.

    trusted_max : highest sampled coverage of the raw model. The raw model is always
        used up to here, also where the ceiling is lower: the data reach this far, and
        a surface element can sit at such a coverage because of other incidence conditions.
    ceiling : self-consistent ceiling of the condition, or None (blend to feature_cap).
        At or below trusted_max, the endpoint applies immediately above trusted_max.

    (Until 2026-10-07 the blend started at min(trusted_max, ceiling), which replaced the
    fitted model between a low ceiling and trusted_max.)
    """
    c = feature_cap if ceiling is None else ceiling
    return trusted_max, max(trusted_max, c)


def blend_beyond_sampled_range(raw_probs_fn, feature_value, trusted_max, feature_cap,
                               zeroed_index, absorbing_index, ceiling=None):
    """raw_probs_fn as-is up to blend_lo; beyond it a linear blend from
    raw_probs_fn(blend_lo) toward the endpoint built by enforce_zero_at_full_feature,
    reached at blend_hi (see blend_limits)."""
    blend_lo, blend_hi = blend_limits(trusted_max, feature_cap, ceiling)

    if feature_value <= blend_lo:
        return raw_probs_fn(feature_value)

    p_boundary = raw_probs_fn(blend_lo)
    p_end = enforce_zero_at_full_feature(p_boundary, feature_cap, feature_cap, zeroed_index, absorbing_index)

    if feature_value >= blend_hi:
        return p_end

    t = (feature_value - blend_lo) / (blend_hi - blend_lo)
    return (1 - t) * p_boundary + t * p_end


@dataclass
class CoveragePhysics:
    """Constants of the gate and blend policy, properties of the dataset and of
    the outcome labelling -- always supplied explicitly."""
    n_sites: int
    zeroed_zero_index: int      # class impossible at coverage 0 (recombinative desorption)
    zeroed_full_index: int      # class impossible at coverage >= n_sites (adsorption)
    absorbing_full_index: int   # class that takes over the zeroed_full_index mass


# --------------------------------------------------------------------------- #
# Grid scheme                                                                 #
# --------------------------------------------------------------------------- #

class CoverageCell:
    """One raw classifier (one grid cell's SplineClassifier, or anything with
    .eval(feature) -> probs) with the gate and blend of its own condition.
    Exposes .eval(coverage) -> probs, so a GridClassifier can hold it as a cell."""

    def __init__(self, raw_classifier, physics: CoveragePhysics, coverage_max, ceiling=None):
        self.raw_classifier = raw_classifier
        self.physics = physics
        self.coverage_max = coverage_max
        self.ceiling = ceiling

    def eval(self, coverage):
        coverage = float(np.asarray(coverage).reshape(-1)[0]) if np.ndim(coverage) else float(coverage)

        def raw(c):
            p = self.raw_classifier.eval(np.array([c]))
            return enforce_zero_at_zero_feature(p, c, self.physics.zeroed_zero_index)

        return blend_beyond_sampled_range(
            raw, coverage, self.coverage_max, self.physics.n_sites,
            self.physics.zeroed_full_index, self.physics.absorbing_full_index,
            ceiling=self.ceiling)


class GridCoverageModel:
    """Grid scheme: a GridClassifier whose cells are CoverageCells (blend per
    cell, then interpolation across cells -- every cell's own dynamics set its
    own blend). `grid` is the underlying GridClassifier."""

    def __init__(self, grid: GridClassifier, physics: CoveragePhysics, coverage_delta=None, model_area_nm2=None,
                 name="grid"):
        self.grid = grid
        self.physics = physics
        self.coverage_delta = None if coverage_delta is None else np.asarray(coverage_delta, dtype=float)
        self.model_area_nm2 = model_area_nm2
        self.name = name

    @property
    def n_classes(self):
        return self.grid.n_classes

    @property
    def members(self):
        """No ensemble members (same interface as nn.NNEnsembleCoverageModel)."""
        return []

    def probs(self, coverage, *conditioning):
        return np.ravel(self.grid.eval(np.array([float(coverage)]), list(conditioning)))

    def probs_fn(self, *conditioning):
        grid, cond = self.grid, list(conditioning)
        return lambda c: np.ravel(grid.eval(np.array([float(c)]), cond))

    def probs_batch(self, coverages, *conditioning):
        return np.array([self.probs(c, *conditioning) for c in np.asarray(coverages, dtype=float)])


# --------------------------------------------------------------------------- #
# Any raw model + ceiling grid                                                #
# --------------------------------------------------------------------------- #

class CoverageTable:
    """Raw probabilities of one condition tabulated on a fine coverage grid
    (0 to n_sites); exact at the grid points, linear in between. The walk only
    visits integer coverages plus one blend boundary, so a step of 0.01 adsorbed
    atoms is far below anything that matters. Used to make Monte Carlo walks of
    expensive raw models (networks) fast."""

    def __init__(self, raw_batch, velocity, angle, n_sites, step=0.01):
        self.grid = np.arange(0.0, n_sites + step / 2, step)
        self.values = np.asarray(raw_batch(self.grid, velocity, angle), dtype=float)

    def __call__(self, c):
        c = min(max(float(c), self.grid[0]), self.grid[-1])
        return np.array([np.interp(c, self.grid, self.values[:, k]) for k in range(self.values.shape[1])])


def _gated(raw_fn, physics):
    def raw(c):
        return enforce_zero_at_zero_feature(raw_fn(c), c, physics.zeroed_zero_index)
    return raw


def _ceiling_for_table(table, physics, coverage_max, coverage_delta, k, n_steps, n_tail, seed):
    return compute_ceiling(_gated(table, physics), physics, coverage_max, coverage_delta,
                           k=k, n_steps=n_steps, n_tail=n_tail, seed=seed)


class CeilingCoverageModel:
    """Any raw model with the gate, the blend and a self-consistent ceiling grid.

    raw_batch(coverages, velocity, angle) -> (n, n_classes) raw probabilities.
    ceiling_axes = [velocities, angles], ceiling_grid (n_v, n_a); the ceiling of a
    query is interpolated multilinearly (clipped to the axes). ceiling_grid None:
    blend all the way to n_sites."""

    def __init__(self, raw_batch, physics: CoveragePhysics, coverage_max, ceiling_axes, ceiling_grid,
                 name="model", coverage_delta=None, model_area_nm2=None):
        self.raw_batch = raw_batch
        self.physics = physics
        self.coverage_max = float(coverage_max)
        self.ceiling_axes = None if ceiling_grid is None else [np.asarray(a, dtype=float) for a in ceiling_axes]
        self.ceiling_grid = None if ceiling_grid is None else np.asarray(ceiling_grid, dtype=float)
        self.name = name
        self.coverage_delta = None if coverage_delta is None else np.asarray(coverage_delta, dtype=float)
        self.model_area_nm2 = model_area_nm2

    @classmethod
    def fit(cls, raw_batch, physics, coverage_max, coverage_delta, velocities, angles, *,
            k=3.0, n_steps=20000, n_tail=8000, seed=0, n_jobs=-1, name="model"):
        """Self-consistent ceiling at every (velocity, angle) of the given axes."""
        points = [(v, a) for v in velocities for a in angles]
        tables = [CoverageTable(raw_batch, v, a, physics.n_sites) for v, a in points]
        ceilings = Parallel(n_jobs=n_jobs)(
            delayed(_ceiling_for_table)(t, physics, coverage_max, coverage_delta, k, n_steps, n_tail, seed)
            for t in tables)
        grid = np.array(ceilings).reshape(len(velocities), len(angles))
        return cls(raw_batch, physics, coverage_max, [velocities, angles], grid, name=name,
                   coverage_delta=coverage_delta)

    def ceiling(self, velocity, angle):
        if self.ceiling_grid is None:
            return None
        return float(interp1d.multilinear_clipped(self.ceiling_axes, self.ceiling_grid, [velocity, angle]))

    def probs_fn(self, velocity, angle, table=True):
        """Scalar probs(coverage) for one condition, as a coverage walk or a DSMC
        boundary condition calls it. table=True reads the raw model from a
        CoverageTable (fast, linear between 0.01-spaced coverages); table=False
        calls the raw model for every query (exact)."""
        raw_fn = CoverageTable(self.raw_batch, velocity, angle, self.physics.n_sites) if table else \
            (lambda c: np.asarray(self.raw_batch(np.array([c]), velocity, angle))[0])
        raw = _gated(raw_fn, self.physics)
        ceiling = self.ceiling(velocity, angle)
        # bind plain values only, so the returned function holds no reference to the
        # raw model (picklable for parallel walks when table=True)
        coverage_max, n_sites = self.coverage_max, self.physics.n_sites
        zeroed, absorbing = self.physics.zeroed_full_index, self.physics.absorbing_full_index

        def probs(c):
            return blend_beyond_sampled_range(raw, float(c), coverage_max, n_sites, zeroed, absorbing,
                                              ceiling=ceiling)
        return probs

    def probs(self, coverage, velocity, angle):
        return self.probs_fn(velocity, angle, table=False)(coverage)

    def probs_batch(self, coverages, velocity, angle):
        fn = self.probs_fn(velocity, angle)
        return np.array([fn(c) for c in np.asarray(coverages, dtype=float)])


# --------------------------------------------------------------------------- #
# Sampling and coverage mapping                                               #
# --------------------------------------------------------------------------- #

def sample_outcome(probs, xi):
    """Inverse-CDF categorical sample: smallest index i with xi < cumsum(probs)[i].
    xi in [0, 1) is supplied by the caller (the host code owns the random numbers).
    Returns (outcome_index, probs)."""
    probs = np.asarray(probs, dtype=float)
    cum = np.cumsum(probs)
    idx = int(np.searchsorted(cum, xi, side="right"))
    idx = min(idx, len(probs) - 1)
    return idx, probs


def stochastic_round(x, xi_round):
    """floor(x) + 1 with probability frac(x), using the caller's xi_round in [0, 1).
    Keeps every query at an integer coverage the model was trained on."""
    floor = int(np.floor(x))
    return floor + (1 if xi_round < (x - floor) else 0)


def map_adsorbate_count_to_coverage(n_adsorbed, element_area, model_area, xi_round):
    """Number of adsorbed atoms n_adsorbed on a surface element of area element_area
    -> integer coverage on the model's training patch of area model_area (same
    unit), by stochastic rounding of the expected count n_adsorbed / element_area * model_area."""
    return stochastic_round(n_adsorbed / element_area * model_area, xi_round)


# --------------------------------------------------------------------------- #
# Coverage walk, ceiling, equilibrium                                          #
# --------------------------------------------------------------------------- #

def run_coverage_walk(probs_fn, coverage_delta, n_steps, seed, start_coverage=0.0):
    """Coverage random walk: at each step sample an outcome from probs_fn(coverage)
    by inverse CDF and move the coverage by coverage_delta[outcome] (floored at 0).
    This is the per-impact loop of a coverage-tracking DSMC boundary condition.
    Returns the trace (length n_steps + 1)."""
    rng = np.random.default_rng(seed)
    coverage_delta = np.asarray(coverage_delta, dtype=float)
    coverage = float(start_coverage)
    trace = [coverage]
    for _ in range(n_steps):
        idx, _ = sample_outcome(probs_fn(coverage), rng.random())
        coverage = max(0.0, coverage + coverage_delta[idx])
        trace.append(coverage)
    return np.array(trace)


def compute_ceiling(raw_probs_fn, physics: CoveragePhysics, coverage_max, coverage_delta, *,
                    k=3.0, n_steps=20000, n_tail=8000, seed=0, max_iter=10, tol=0.05):
    """Self-consistent blend ceiling of one condition: run the blended model's
    coverage walk, take mean + k * std of the last n_tail steps (capped at
    n_sites), feed it back as the next ceiling, until it changes by less than tol.
    Starts at n_sites, the one ceiling that needs no prior estimate."""
    ceil = physics.n_sites
    for _ in range(max_iter):
        def wrapped(c, ceil=ceil):
            return blend_beyond_sampled_range(
                raw_probs_fn, c, coverage_max, physics.n_sites,
                physics.zeroed_full_index, physics.absorbing_full_index, ceiling=ceil)

        trace = run_coverage_walk(wrapped, coverage_delta, n_steps, seed)
        tail = trace[-n_tail:]
        new_ceil = min(physics.n_sites, tail.mean() + k * tail.std())
        if abs(new_ceil - ceil) < tol:
            return new_ceil
        ceil = new_ceil
    return ceil


def _up_down(coverage_delta):
    delta = np.asarray(coverage_delta, dtype=float)
    if not set(delta.tolist()) <= {-1.0, 0.0, 1.0} or (delta == 1).sum() != 1 or (delta == -1).sum() != 1:
        raise ValueError("needs exactly one class with coverage change +1 and one with -1, the rest 0")
    return int(np.flatnonzero(delta == 1)[0]), int(np.flatnonzero(delta == -1)[0])


def stationary_distribution(P, coverage_delta=(1, 0, -1)):
    """Exact stationary distribution of the coverage walk on the integer coverages
    0..len(P)-1, where P[c] are the outcome probabilities at coverage c. The walk is
    a birth-death chain (up with the +1 class, down with the -1 class), so
    pi[c+1] / pi[c] = P[c, up] / P[c+1, down]. Mean coverage = pi @ arange(len(P))."""
    up_i, down_i = _up_down(coverage_delta)
    P = np.asarray(P, dtype=float)
    logpi = np.zeros(len(P))
    for c in range(len(P) - 1):
        up, down = P[c, up_i], P[c + 1, down_i]
        if up <= 0 or down <= 0:
            logpi[c + 1:] = -np.inf
            break
        logpi[c + 1] = logpi[c] + np.log(up) - np.log(down)
    pi = np.exp(logpi - logpi.max())
    return pi / pi.sum()


def net_flux_crossing(probs_fn, coverage_delta=(1, 0, -1), c_max=24.0, n_points=2401):
    """First coverage at which the expected coverage change per impact,
    sum_k coverage_delta[k] p_k (adsorption minus O2 desorption here), changes sign,
    searched on linspace(0, c_max, n_points) and interpolated linearly. NaN if none."""
    up_i, down_i = _up_down(coverage_delta)
    g = np.linspace(0, c_max, n_points)
    p = np.array([probs_fn(c) for c in g])
    flux = p[:, up_i] - p[:, down_i]
    sc = np.where(np.sign(flux[:-1]) != np.sign(flux[1:]))[0]
    if len(sc) == 0:
        return np.nan
    i = sc[0]
    return g[i] + (g[i + 1] - g[i]) * flux[i] / (flux[i] - flux[i + 1])


# --------------------------------------------------------------------------- #
# Fit entry point of the grid scheme                                          #
# --------------------------------------------------------------------------- #

def fit_coverage_grid_model(axes, event_axis_values, event_features, event_labels, terms, *,
                            n_classes, physics: CoveragePhysics, coverage_delta,
                            C=1.0, n_jobs=1, method="pchip",
                            ceiling_k=3.0, mc_steps=20000, mc_tail=8000, mc_seed=0, model_area_nm2=None):
    """Fit the grid scheme end-to-end: one independent SplineClassifier per grid
    cell (fit_grid_classifier), each wrapped in a CoverageCell with its own
    coverage_max (highest coverage of the cell, column 0 of event_features) and
    its own self-consistent ceiling. Returns a GridCoverageModel."""
    raw_grid = fit_grid_classifier(axes, event_axis_values, event_features, event_labels, terms,
                                   n_classes=n_classes, C=C, n_jobs=n_jobs, method=method)

    def _wrap(idx_clf):
        idx, clf = idx_clf
        coverage_max = float(clf.raw_feature_ranges[0, 1])
        ceiling = compute_ceiling(clf.eval, physics, coverage_max, coverage_delta,
                                  k=ceiling_k, n_steps=mc_steps, n_tail=mc_tail, seed=mc_seed)
        return idx, CoverageCell(clf, physics, coverage_max, ceiling)

    results = Parallel(n_jobs=n_jobs)(delayed(_wrap)(item) for item in raw_grid.cell_classifiers.items())
    grid = GridClassifier(raw_grid.axes, dict(results), n_classes, method=method)
    return GridCoverageModel(grid, physics, coverage_delta=coverage_delta, model_area_nm2=model_area_nm2)
