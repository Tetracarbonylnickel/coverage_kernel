"""
interp1d.py
===========
Generic 1-D interpolation building blocks used by grid_classifier.py to
interpolate across a sampled grid axis (e.g. velocity or angle). Nothing in
this module knows about coverage, outcomes, or any other domain concept --
it operates on plain (x, y) arrays.

Two methods are supported, both evaluated from a *local window* of grid
points rather than the whole axis:

  "linear" -- ordinary linear interpolation between the two bracketing
      points. Needs only those 2 points. Always a convex combination of its
      neighbours (can't overshoot), but not smooth (kinked at every node).

  "pchip" -- shape-preserving monotone cubic Hermite interpolation
      (Fritsch-Carlson method, matching scipy.interpolate.PchipInterpolator
      exactly). Needs up to 4 points around the query (2 each side; fewer
      at the ends of the axis, using scipy's one-sided edge derivative
      formula there). Smooth and, empirically, far less prone to
      overshooting past its local bracketing points than a plain cubic
      spline (see the docs this project inherited that empirical finding
      from) -- but costs more to evaluate than "linear".

The point of doing this from scratch instead of calling
scipy.interpolate.RegularGridInterpolator is that PCHIP's derivative at any
node depends only on its immediate neighbours -- never the whole array --
so evaluating at one query point never requires visiting every grid node,
only the local window around it. That's what makes lazy/local grid-cell
evaluation possible in grid_classifier.py: only the cells inside the
window this module reports are ever evaluated for a given query.
"""

import numpy as np


def bracket_index(grid, x):
    """Index i such that grid[i] <= x < grid[i+1] (bracket for interpolation),
    clamped to [0, len(grid)-2] so it's always a valid interval start --
    i.e. x outside [grid[0], grid[-1]] returns the nearest boundary
    interval, the caller is responsible for deciding whether that counts
    as "in range" or needs its own extrapolation policy.
    """
    n = len(grid)
    i = int(np.searchsorted(grid, x, side="right") - 1)
    return max(0, min(i, n - 2))


def local_window(grid, x, method):
    """Return the (lo, hi) inclusive index range of grid points needed to
    evaluate `method` at query `x`, and the bracket index i (grid[i] <=
    x <= grid[i+1] once clamped).

    "linear" needs only {i, i+1}. "pchip" needs up to {i-1, i, i+1, i+2},
    clamped to the array bounds (fewer points are needed near either edge
    of the axis -- see pchip_eval_window's edge-derivative handling).
    """
    n = len(grid)
    i = bracket_index(grid, x)
    if method == "linear":
        return i, i, i + 1
    elif method == "pchip":
        lo = max(i - 1, 0)
        hi = min(i + 2, n - 1)
        return i, lo, hi
    raise ValueError(f"unknown interpolation method {method!r}")


def multilinear_clipped(axes, values, query):
    """N-dimensional multilinear interpolation of a small precomputed array
    `values` (shape matching the length of each axis in `axes`), clipping
    the query into range on every axis first (never extrapolates). Used for
    interpolating small scalar/vector grids -- e.g. a per-condition "safety
    ceiling" value -- where the *value* being interpolated is smooth/
    monotone-safe enough that plain multilinear is preferable to pchip
    (no shape-preservation concern), and the grid is cheap enough that
    there's no lazy-evaluation benefit to worry about, unlike
    grid_classifier.py's per-cell classifiers.
    """
    n_axes = len(axes)

    def recurse(d, fixed_indices):
        if d == n_axes:
            return values[tuple(fixed_indices)]
        grid = axes[d]
        qv = min(max(query[d], grid[0]), grid[-1])
        if len(grid) == 1:
            return recurse(d + 1, fixed_indices + [0])
        _, lo, hi = local_window(grid, qv, "linear")
        v_lo = recurse(d + 1, fixed_indices + [lo])
        v_hi = recurse(d + 1, fixed_indices + [hi])
        t = (qv - grid[lo]) / (grid[hi] - grid[lo])
        return v_lo + t * (v_hi - v_lo)

    return recurse(0, [])


def linear_eval_window(x_window, y_window, x_query):
    """x_window, y_window: the 2 points {i, i+1} from local_window.
    y_window may be a (2,) array (scalar y) or (2, K) (vector y, e.g. a
    probability vector per node) -- broadcasts either way.
    """
    x0, x1 = x_window
    y0, y1 = y_window[0], y_window[1]
    if x1 == x0:
        return y0
    t = (x_query - x0) / (x1 - x0)
    return y0 + t * (y1 - y0)


def _edge_derivative(h0, h1, m0, m1):
    """One-sided 3-point derivative estimate at an axis endpoint
    (scipy.interpolate._cubic._edge_case). h0/m0 are the step/slope of the
    interval touching the endpoint; h1/m1 are the next interval inward.

    m0/m1 (and thus the result) may be scalars or arrays (elementwise, e.g.
    one slope per outcome in a vector-valued grid_classifier probability).
    """
    m0 = np.asarray(m0, dtype=float)
    m1 = np.asarray(m1, dtype=float)
    d = ((2 * h0 + h1) * m0 - h0 * m1) / (h0 + h1)
    d = np.where(np.sign(d) != np.sign(m0), 0.0, d)
    override = (np.sign(m0) != np.sign(m1)) & (np.abs(d) > 3.0 * np.abs(m0))
    d = np.where(override, 3.0 * m0, d)
    return d


def _interior_derivative(h_prev, h_next, m_prev, m_next):
    """Fritsch-Carlson weighted-harmonic-mean derivative estimate at an
    interior node, given the two secant slopes either side of it
    (scipy.interpolate._cubic._find_derivatives' interior-point branch).
    m_prev/m_next may be arrays (per-outcome), evaluated elementwise.
    """
    m_prev = np.asarray(m_prev, dtype=float)
    m_next = np.asarray(m_next, dtype=float)
    condition = (np.sign(m_prev) != np.sign(m_next)) | (m_prev == 0) | (m_next == 0)
    w1 = 2 * h_next + h_prev
    w2 = h_next + 2 * h_prev
    with np.errstate(divide="ignore", invalid="ignore"):
        whmean = (w1 / np.where(m_prev == 0, 1.0, m_prev) +
                  w2 / np.where(m_next == 0, 1.0, m_next)) / (w1 + w2)
        d = np.where(condition, 0.0, 1.0 / whmean)
    return d


def pchip_eval_window(x_window, y_window, x_query, *, is_left_edge, is_right_edge):
    """Evaluate PCHIP at x_query using only a local window of grid points.

    x_window, y_window: arrays for indices {lo..hi} from local_window("pchip", ...),
    i.e. length 3 or 4 (fewer only if the axis itself has < 4 points total).
    y_window may be (n,) or (n, K).

    is_left_edge / is_right_edge: whether x_window[0] / x_window[-1] is the
    *global* first/last point of the full axis (not just the local window) --
    this determines whether the derivative at that end of the window uses
    scipy's one-sided edge formula or the interior weighted-harmonic-mean
    formula, exactly mirroring what evaluating the full axis with
    scipy.interpolate.PchipInterpolator would do. Getting this flag wrong
    at a window that happens to sit at the true axis boundary would silently
    reproduce the wrong (interior) formula there.
    """
    x_window = np.asarray(x_window, dtype=float)
    y_window = np.asarray(y_window, dtype=float)
    n = len(x_window)

    h = x_window[1:] - x_window[:-1]                 # (n-1,)
    m = (y_window[1:] - y_window[:-1]) / h[:, None] if y_window.ndim == 2 \
        else (y_window[1:] - y_window[:-1]) / h       # (n-1,) or (n-1, K)

    if n == 2:
        # whole axis has only 2 points: pchip degenerates to the single secant slope
        d0 = d1 = m[0]
        i0, i1 = 0, 1
    else:
        # bracket is always the *middle* interval of the window as constructed
        # by local_window: window is {i-1,i,i+1,i+2} clamped, bracket is [i,i+1].
        # Find the bracket by locating x_query between two consecutive window points.
        i0 = int(np.searchsorted(x_window, x_query, side="right") - 1)
        i0 = max(0, min(i0, n - 2))
        i1 = i0 + 1

        # derivative at window-index i0
        if i0 == 0 and is_left_edge:
            d0 = _edge_derivative(h[0], h[1], m[0], m[1])
        elif i0 == n - 1 and is_right_edge:
            d0 = _edge_derivative(h[-1], h[-2], m[-1], m[-2])
        else:
            d0 = _interior_derivative(h[i0 - 1], h[i0], m[i0 - 1], m[i0])

        # derivative at window-index i1
        if i1 == 0 and is_left_edge:
            d1 = _edge_derivative(h[0], h[1], m[0], m[1])
        elif i1 == n - 1 and is_right_edge:
            d1 = _edge_derivative(h[-1], h[-2], m[-1], m[-2])
        else:
            d1 = _interior_derivative(h[i1 - 1], h[i1], m[i1 - 1], m[i1])

    x0, x1 = x_window[i0], x_window[i1]
    y0, y1 = y_window[i0], y_window[i1]
    hb = x1 - x0
    t = (x_query - x0) / hb

    h00 = 2 * t**3 - 3 * t**2 + 1
    h10 = t**3 - 2 * t**2 + t
    h01 = -2 * t**3 + 3 * t**2
    h11 = t**3 - t**2

    if y_window.ndim == 2:
        t = t  # noqa (already broadcasts fine for the scalar-t, vector-y case)
    return h00 * y0 + h10 * hb * d0 + h01 * y1 + h11 * hb * d1
