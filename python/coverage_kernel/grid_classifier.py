"""
grid_classifier.py
===================
Generic building block: an N-dimensional grid of independent classifiers
(one per grid cell, each a pooled_classifier.PooledClassifier) with local,
lazy interpolation across the grid at query time.

This has no idea what the grid axes or the per-cell feature mean -- it
doesn't know about velocity, angle, or coverage.

The key design point is that:
evaluating a query only ever touches the grid cells inside the *local*
interpolation stencil around the query point -- never the whole grid.
Both supported interpolation methods build that stencil from only the
immediate neighbourhood of each axis:
  - "linear": 2 points per axis (the bracketing pair).
  - "pchip": up to 4 points per axis (see interp1d.py) -- shape-preserving,
    matches the current validated model's behavior (chosen there after
    real overshoot testing against a plain cubic spline), at higher
    implementation complexity than linear.
Tensor-product interpolation across n_axes dimensions is done by recursively
collapsing one axis at a time: fix a window index on axis 0, recurse to
resolve axes 1..n-1 (down to actually evaluating the corresponding cells'
classifiers), interpolate the results along axis 0. Cost is the product of
the per-axis window sizes (<=4 per axis for pchip), never the full grid.

Each axis has its own *edge policy* for queries outside its sampled range:
  - "clip": clamp the query into range before interpolating (always inside
    the simplex/stencil, never extrapolates).
  - "linear_extrapolate": if outside range, switch that axis (only) to a
    2-point linear extrapolation off the boundary segment's slope,
    regardless of what `method` is configured for in-range queries. This
    mirrors a deliberate real finding in the model this generalizes:
    letting "pchip" extrapolate on its own continues the boundary
    *cubic* segment, which can reverse sign/diverge, whereas continuing
    the boundary *slope* (linear) cannot.
"""

from dataclasses import dataclass

import numpy as np
from joblib import Parallel, delayed

from . import interp1d
from .pooled_classifier import fit_pooled_classifier


def _nearest_indices(axes, event_axis_values):
    """Snap each event's (possibly noisy) per-axis coordinate to its
    nearest grid value, giving each event a definite grid-cell index tuple.
    No tolerance parameter is needed (and none is dataset-specific): "nearest"
    is well-defined for any axis as long as events are closer to their
    intended grid point than to any neighbouring one, which holds for any
    reasonably-sampled grid.
    """
    event_axis_values = np.asarray(event_axis_values, dtype=float)
    N, n_axes = event_axis_values.shape
    idx = np.empty((N, n_axes), dtype=int)
    for d in range(n_axes):
        grid = np.asarray(axes[d], dtype=float)
        diffs = np.abs(grid[None, :] - event_axis_values[:, d, None])
        idx[:, d] = diffs.argmin(axis=1)
    return idx


@dataclass(frozen=True)
class AxisSpec:
    """One grid axis: its sampled values (sorted) and its out-of-range
    query policy."""
    values: np.ndarray
    edge_policy: str = "clip"   # "clip" or "linear_extrapolate"

    def __post_init__(self):
        if self.edge_policy not in ("clip", "linear_extrapolate"):
            raise ValueError(f"unknown edge_policy {self.edge_policy!r}")


class GridClassifier:
    """A fitted grid of independent classifiers. Construct via
    fit_grid_classifier; evaluate via .eval(feature, axis_query).
    """

    def __init__(self, axes, cell_classifiers, n_classes, method="pchip"):
        self.axes = axes                        # list[AxisSpec]
        self.cell_classifiers = cell_classifiers  # {index_tuple: PooledClassifier}
        self.n_classes = n_classes
        self.method = method

    def eval(self, feature, axis_query):
        """feature: the raw per-cell feature vector/scalar (e.g. coverage).
        axis_query: sequence of len(self.axes) query coordinates.
        Returns (n_classes,) probabilities, clipped to >=0 and renormalized
        as a safety net (pchip has no structural simplex guarantee the way
        a convex-combination method like "linear" does, though it was
        chosen for empirically negligible overshoot).
        """
        n_axes = len(self.axes)

        def resolve_axis(d, q):
            axis = self.axes[d]
            grid = axis.values
            lo, hi = grid[0], grid[-1]
            if axis.edge_policy == "clip":
                qv = min(max(q, lo), hi)
                eff_method = self.method
            else:  # linear_extrapolate
                qv = q
                eff_method = self.method if lo <= q <= hi else "linear"
            if len(grid) == 1:
                return [0], grid, qv, eff_method
            i, w_lo, w_hi = interp1d.local_window(grid, qv, eff_method)
            return list(range(w_lo, w_hi + 1)), grid, qv, eff_method

        def eval_recursive(d, fixed_indices):
            if d == n_axes:
                return self.cell_classifiers[tuple(fixed_indices)].eval(feature)

            window, grid, qv, eff_method = resolve_axis(d, axis_query[d])
            if len(window) == 1:
                return eval_recursive(d + 1, fixed_indices + [window[0]])

            values = np.array([eval_recursive(d + 1, fixed_indices + [idx]) for idx in window])
            x_window = grid[window]

            if eff_method == "linear":
                return interp1d.linear_eval_window(x_window, values, qv)
            else:  # pchip
                is_left_edge = window[0] == 0
                is_right_edge = window[-1] == len(grid) - 1
                return interp1d.pchip_eval_window(x_window, values, qv,
                                                   is_left_edge=is_left_edge,
                                                   is_right_edge=is_right_edge)

        raw = eval_recursive(0, [])
        raw = np.clip(raw, 0, None)
        total = raw.sum()
        return raw / total if total > 0 else raw


def fit_grid_classifier(axes, event_axis_values, event_features, event_labels, terms, *,
                         n_classes, C=1.0, n_jobs=1, method="pchip"):
    """Fit one independent PooledClassifier per grid cell.

    Parameters
    ----------
    axes : list[AxisSpec] (or plain arrays, defaulting to edge_policy="clip") --
        the sampled grid values along each dimension, e.g. [velocities, angles].
    event_axis_values : (N, n_axes) each event's own (possibly noisy) coordinate
        along every axis -- snapped to the nearest grid point per axis (see
        _nearest_indices) to decide which cell it belongs to.
    event_features : (N,) or (N, d) the raw feature(s) each cell's own
        classifier is fit on (e.g. coverage alone, for the scheme this
        generalizes -- but any number of per-cell features is supported).
    event_labels : (N,) integer class labels in [0, n_classes).
    terms : list of SplineTerm/LinearTerm applied to event_features' columns,
        shared by every cell (each cell fits its own coefficients for
        these terms independently -- see pooled_classifier.py).
    n_classes, C : see fit_pooled_classifier.
    n_jobs : parallel workers across cells (joblib.Parallel) -- each cell's
        fit is completely independent of every other, so this parallelizes
        trivially, unlike the pooled scheme.
    method : default cross-cell interpolation method for the returned
        GridClassifier ("pchip" or "linear") -- can be overridden per-call
        via a different GridClassifier.method if needed.
    """
    axes = [a if isinstance(a, AxisSpec) else AxisSpec(np.asarray(a, dtype=float)) for a in axes]
    event_features = np.atleast_2d(np.asarray(event_features, dtype=float).T).T if np.ndim(event_features) == 1 \
        else np.asarray(event_features, dtype=float)
    event_labels = np.asarray(event_labels, dtype=int)

    cell_index = _nearest_indices([a.values for a in axes], event_axis_values)

    groups = {}
    for row, idx in enumerate(map(tuple, cell_index)):
        groups.setdefault(idx, []).append(row)

    def _fit_one(idx, rows):
        rows = np.asarray(rows)
        clf = fit_pooled_classifier(event_features[rows], event_labels[rows], terms,
                                     n_classes=n_classes, C=C)
        return idx, clf

    results = Parallel(n_jobs=n_jobs)(delayed(_fit_one)(idx, rows) for idx, rows in groups.items())
    cell_classifiers = dict(results)

    return GridClassifier(axes, cell_classifiers, n_classes, method=method)
