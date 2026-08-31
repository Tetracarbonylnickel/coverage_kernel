"""
coverage_model.py
==================
Domain layer: the coverage-specific physics on top of the generic
classifiers in pooled_classifier.py / grid_classifier.py. Nothing in
grid_classifier.py or pooled_classifier.py knows any of this -- this module
is where "coverage," "adsorption," and "n_sites" first appear.

Three pieces of physics, all generalized to an arbitrary number of outcome
classes and an arbitrary class-index assignment (not hardcoded to this
project's 3-outcome ads/O_des/O2_des set):

1. Zero-at-zero-coverage gate: some outcome (e.g. recombinative desorption)
   is physically impossible with nothing on the surface -- hard-zero that
   class's probability at coverage==0 and renormalize the rest.
2. Zero-at-full-coverage gate + blend: some outcome (e.g. adsorption) is
   physically impossible once every site is occupied. Rather than trusting
   a fitted classifier's raw output all the way out to that boundary (it
   was never trained anywhere near there), the raw fit is trusted up to a
   per-condition boundary and then linearly blended toward the known,
   physically-certain endpoint. Blending two valid probability vectors
   can't leave the simplex, unlike extrapolating a fitted coefficient.
3. The blend's far boundary ("ceiling") is itself a self-consistent
   Monte-Carlo fixed point: run the (blended) model's own dynamics forward
   as a random walk and see where it settles.

A CoverageCell wraps a raw classifier (a PooledClassifier, or one grid cell
of a GridClassifier) with this physics, exposing the same `.eval(feature)`
interface a bare classifier does -- so it can be substituted transparently
wherever a classifier is expected, including as a GridClassifier cell (see
grid_classifier.py's duck-typed cell_classifiers dict). This matters
because the *order of operations* is: blend first (per condition), *then*
interpolate across the grid -- not the other way around -- matching how
each grid cell's own dynamics differ enough that a shared blend boundary
across the whole grid would be wrong.
"""

from dataclasses import dataclass

import numpy as np
from joblib import Parallel, delayed

from . import interp1d
from .pooled_classifier import fit_pooled_classifier
from .grid_classifier import fit_grid_classifier, GridClassifier


def enforce_zero_at_zero_feature(probs, feature_value, zeroed_index):
    """Hard-zero probs[zeroed_index] when feature_value == 0, renormalizing
    the rest. Generalizes _enforce_no_o2_at_zero_coverage.
    """
    probs = np.asarray(probs, dtype=float)
    if feature_value == 0 and probs[zeroed_index] != 0:
        probs = probs.copy()
        probs[zeroed_index] = 0.0
        total = probs.sum()
        if total > 0:
            probs = probs / total
    return probs


def enforce_zero_at_full_feature(probs, feature_value, feature_cap, zeroed_index, absorbing_index):
    """Hard-zero probs[zeroed_index] once feature_value >= feature_cap,
    handing all of its mass to probs[absorbing_index] (every other class
    untouched). Generalizes _enforce_no_ads_at_full_coverage.
    """
    probs = np.asarray(probs, dtype=float)
    if feature_value >= feature_cap and probs[zeroed_index] != 0:
        probs = probs.copy()
        probs[absorbing_index] += probs[zeroed_index]
        probs[zeroed_index] = 0.0
    return probs


def blend_beyond_sampled_range(raw_probs_fn, feature_value, trusted_max, feature_cap,
                                zeroed_index, absorbing_index, ceiling=None):
    """Trust raw_probs_fn as-is up to blend_lo = min(trusted_max, ceiling),
    then linearly blend toward the fully-saturated endpoint (built by
    enforce_zero_at_full_feature at blend_lo) out to
    blend_hi = max(trusted_max, ceiling). ceiling=None reproduces blending
    all the way to feature_cap (the "everyday" case where the condition's
    own dynamics settle beyond the dataset-wide trusted range).
    Generalizes _extrapolate_beyond_sampled_coverage.
    """
    c = feature_cap if ceiling is None else ceiling
    blend_lo = min(trusted_max, c)
    blend_hi = max(trusted_max, c)

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
    """The per-model-instance constants the blend/gate policy needs.
    n_sites, zeroed_zero_index, zeroed_full_index, absorbing_full_index are
    properties of the dataset/supercell and outcome-labeling convention --
    never hardcoded, always supplied explicitly.
    """
    n_sites: int
    zeroed_zero_index: int      # class impossible at feature==0 (e.g. recombinative desorption)
    zeroed_full_index: int      # class impossible at feature>=n_sites (e.g. adsorption)
    absorbing_full_index: int   # class that absorbs the zeroed_full_index class's freed mass


class CoverageCell:
    """Wraps one raw classifier (a PooledClassifier, or anything with a
    matching .eval(feature)->probs) with the coverage blend/gate physics
    for one specific condition (e.g. one grid cell, or the single pooled
    model evaluated at one query condition). Exposes .eval(coverage)->probs,
    so it can substitute for a bare classifier wherever one is expected.
    """

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


class PooledCoverageModel:
    """Domain wrapper for the pooled/global scheme: a single PooledClassifier
    evaluated at engineered features built from (coverage, *conditioning) via
    `feature_fn` (e.g. this project's normal-energy/parallel-velocity
    decomposition). Since there's only one shared classifier rather than
    one per grid cell, the blend's per-condition ceiling is generalized to
    any conditioning value by interpolating a small precomputed ceiling
    grid (interp1d.multilinear_clipped) instead of a per-cell lookup.
    """

    def __init__(self, pooled_classifier, feature_fn, physics: CoveragePhysics, coverage_max,
                 ceiling_axes=None, ceiling_grid=None):
        self.pooled_classifier = pooled_classifier
        self.feature_fn = feature_fn
        self.physics = physics
        self.coverage_max = coverage_max
        self.ceiling_axes = ceiling_axes
        self.ceiling_grid = ceiling_grid

    def probs(self, coverage, *conditioning):
        def raw(c):
            x = self.feature_fn(c, *conditioning)
            p = self.pooled_classifier.eval(x)
            return enforce_zero_at_zero_feature(p, c, self.physics.zeroed_zero_index)

        ceiling = None
        if self.ceiling_grid is not None:
            ceiling = interp1d.multilinear_clipped(self.ceiling_axes, self.ceiling_grid, list(conditioning))

        return blend_beyond_sampled_range(
            raw, coverage, self.coverage_max, self.physics.n_sites,
            self.physics.zeroed_full_index, self.physics.absorbing_full_index,
            ceiling=ceiling)


def sample_outcome(probs, xi):
    """Inverse-CDF categorical sample: smallest index i with xi <= cumsum(probs)[i].
    xi in [0, 1) supplied by the caller (e.g. PICLas's own RNG stream) --
    this module never owns randomness. Returns (outcome_index, probs).
    """
    probs = np.asarray(probs, dtype=float)
    cum = np.cumsum(probs)
    idx = int(np.searchsorted(cum, xi, side="right"))
    idx = min(idx, len(probs) - 1)
    return idx, probs


def stochastic_round(x, xi_round):
    """floor(x) + 1 with probability frac(x), using caller-supplied xi_round
    in [0,1) -- generalizes DensityCoverageModel._stochastic_round. Keeps
    every query at an integer coverage the classifier was actually trained
    on, rather than silently extrapolating to a fractional value.
    """
    floor = int(np.floor(x))
    return floor + (1 if xi_round < (x - floor) else 0)


def map_area_density_to_coverage(area_density, element_area, model_area, xi_round):
    """Map an areal adsorbate density on a surface element of area
    `element_area` to an integer coverage on the model's own training-patch
    area `model_area`, via stochastic rounding of the expected count.
    Generalizes DensityCoverageModel.probs's local_density/local_coverage
    computation.
    """
    local_density = area_density / element_area
    local_coverage = local_density * model_area
    return stochastic_round(local_coverage, xi_round)


# --------------------------------------------------------------------------- #
# Self-consistent ceiling search (fit-time only)                              #
# --------------------------------------------------------------------------- #

def run_coverage_walk(probs_fn, coverage_delta, n_steps, seed, start_coverage=0.0):
    """Generic coverage random walk: at each step, sample an outcome from
    probs_fn(coverage) via inverse-CDF, move coverage by coverage_delta[outcome]
    (clipped at 0), repeat. coverage_delta: (n_classes,) array/list of the net
    coverage change each outcome causes (e.g. +1/0/-1 for ads/O_des/O2_des).
    Returns the full trace (length n_steps+1). Domain-specific only in that
    it always clips coverage at 0 (a physical floor: coverage can't go
    negative) -- otherwise this is what a DSMC/PICLas boundary condition
    loop does per particle, just self-contained here for calibration.
    """
    rng = np.random.default_rng(seed)
    coverage_delta = np.asarray(coverage_delta, dtype=float)
    coverage = float(start_coverage)
    trace = [coverage]
    for _ in range(n_steps):
        idx, _ = sample_outcome(probs_fn(coverage), rng.random())
        coverage = max(coverage + coverage_delta[idx], 0.0)
        trace.append(coverage)
    return np.array(trace)


def compute_ceiling(raw_probs_fn, physics: CoveragePhysics, coverage_max, coverage_delta, *,
                     k=3.5, n_steps=20000, n_tail=8000, seed=0, max_iter=10, tol=0.05):
    """Self-consistent per-condition blend ceiling: iteratively run
    raw_probs_fn's own (blended) dynamics forward as a random walk and take
    mean + k*std of the tail, capped at n_sites, feeding the result back in
    as the next iteration's blend ceiling until it stabilizes. Generalizes
    _compute_mc_ceiling.

    Seeded at n_sites (the one ceiling value that needs no prior estimate,
    since p=0 there is a hard physical fact). Iterating through the *full*
    blend (not a cruder wrapper) at every step matters: a condition whose
    raw fit still favors the zeroed_full_index class at its own boundary
    would otherwise just ride the n_sites wall with nothing pushing back,
    giving a degenerate "ceiling" of exactly n_sites rather than a real
    self-consistent equilibrium.
    """
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


# --------------------------------------------------------------------------- #
# Top-level fit entry points                                                  #
# --------------------------------------------------------------------------- #

def fit_coverage_pooled_model(X_raw, y, terms, feature_fn, *, n_classes, physics: CoveragePhysics,
                               coverage_delta, C=1.0, coverage_column=0,
                               ceiling_axes_values=None, ceiling_k=3.5, mc_steps=20000,
                               mc_tail=8000, mc_seed=0, ceiling_n_jobs=1):
    """Fit the pooled scheme end-to-end: a single PooledClassifier over all
    events, wrapped with the coverage blend/gate physics.

    X_raw, y, terms, C : passed straight to fit_pooled_classifier.
    feature_fn(coverage, *conditioning) -> raw feature vector : must match
        the same feature engineering `terms` expects (e.g. this project's
        normal-energy/parallel-velocity decomposition) -- supplied by the
        caller rather than hardcoded, so a different physical system's
        feature choice doesn't require touching this function.
    coverage_column : which column of X_raw is coverage (default 0, this
        project's convention throughout, but not assumed elsewhere).
    ceiling_axes_values : optional list of per-conditioning-dimension grid
        values (e.g. [velocities, angles]) at which to compute a self-
        consistent ceiling via compute_ceiling, generalized across the rest
        of the (conditioning) space by interpolation at query time (see
        PooledCoverageModel). None skips ceiling computation entirely
        (blend goes all the way to n_sites for every condition -- the
        `blend_beyond_sampled_range` default when ceiling=None).
    """
    clf = fit_pooled_classifier(X_raw, y, terms, n_classes=n_classes, C=C)
    coverage_max = float(X_raw[:, coverage_column].max())

    ceiling_grid = None
    if ceiling_axes_values is not None:
        def raw(c, conditioning):
            x = feature_fn(c, *conditioning)
            p = clf.eval(x)
            return enforce_zero_at_zero_feature(p, c, physics.zeroed_zero_index)

        mesh = np.meshgrid(*ceiling_axes_values, indexing="ij")
        points = list(zip(*(m.ravel() for m in mesh)))

        def _one(point):
            return compute_ceiling(lambda c: raw(c, point), physics, coverage_max, coverage_delta,
                                    k=ceiling_k, n_steps=mc_steps, n_tail=mc_tail, seed=mc_seed)

        results = Parallel(n_jobs=ceiling_n_jobs)(delayed(_one)(p) for p in points)
        ceiling_grid = np.array(results).reshape(mesh[0].shape)

    return PooledCoverageModel(clf, feature_fn, physics, coverage_max,
                                ceiling_axes=ceiling_axes_values, ceiling_grid=ceiling_grid)


def fit_coverage_grid_model(axes, event_axis_values, event_features, event_labels, terms, *,
                             n_classes, physics: CoveragePhysics, coverage_delta,
                             C=1.0, n_jobs=1, method="pchip",
                             ceiling_k=3.5, mc_steps=20000, mc_tail=8000, mc_seed=0):
    """Fit the grid scheme end-to-end: one independent PooledClassifier per
    grid cell (via fit_grid_classifier), each wrapped in a CoverageCell
    (its own coverage_max -- read off PooledClassifier.raw_feature_ranges,
    column 0 by convention -- and its own self-consistent MC ceiling).

    Blending happens per cell, *before* cross-cell interpolation (see
    CoverageCell/GridClassifier docstrings for why this order matters) --
    achieved simply by substituting CoverageCell-wrapped classifiers into a
    GridClassifier in place of the bare PooledClassifiers fit_grid_classifier
    itself returns, relying on GridClassifier's duck-typed cell_classifiers.
    """
    raw_grid = fit_grid_classifier(axes, event_axis_values, event_features, event_labels, terms,
                                    n_classes=n_classes, C=C, n_jobs=n_jobs, method=method)

    def _wrap(idx_clf):
        idx, clf = idx_clf
        coverage_max = float(clf.raw_feature_ranges[0, 1])
        ceiling = compute_ceiling(clf.eval, physics, coverage_max, coverage_delta,
                                   k=ceiling_k, n_steps=mc_steps, n_tail=mc_tail, seed=mc_seed)
        return idx, CoverageCell(clf, physics, coverage_max, ceiling)

    results = Parallel(n_jobs=n_jobs)(delayed(_wrap)(item) for item in raw_grid.cell_classifiers.items())
    wrapped_cells = dict(results)

    return GridClassifier(raw_grid.axes, wrapped_cells, n_classes, method=method)
