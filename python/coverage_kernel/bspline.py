"""
bspline.py
==========
Generic B-spline basis evaluation. Nothing here knows about coverage or any
other domain concept -- it's pure basis-function math, reused by
spline_classifier.py (the per-cell classifier of grid_classifier.py).

Matches scikit-learn's `SplineTransformer(knots="uniform", extrapolation=
"constant", include_bias=False)` exactly: same knot-vector convention, same
"drop the last raw basis function" rule (verified directly against
sklearn).

Implements the standard Cox-de Boor recursion (`BasisFuns`, The NURBS Book
A2.2) rather than going through scipy.interpolate.BSpline, so that
evaluating one point only costs O(degree**2), touching only the `degree+1`
basis functions that are actually nonzero there -- consistent with this
project's general approach of never doing more work than the query needs.
"""

import numpy as np


def uniform_knots(x_min, x_max, n_knots, degree):
    """Build the extended knot vector sklearn's SplineTransformer(knots=
    "uniform") would produce for `n_knots` interior knots evenly spaced over
    [x_min, x_max], extended by `degree` extra knots on each side at the
    same spacing (verified against the real shipped model: n_knots=5,
    degree=2, [x_min,x_max]=[0,16] -> interior {0,4,8,12,16}, extended to
    {-8,-4,0,4,8,12,16,20,24}).
    """
    interior = np.linspace(x_min, x_max, n_knots)
    spacing = interior[1] - interior[0]
    left = interior[0] - spacing * np.arange(degree, 0, -1)
    right = interior[-1] + spacing * np.arange(1, degree + 1)
    return np.concatenate([left, interior, right])


def n_basis_raw(knots, degree):
    """Number of raw B-spline basis functions for this knot vector (before
    dropping one for include_bias=False)."""
    return len(knots) - degree - 1


def nonzero_basis(knots, degree, x):
    """Evaluate the `degree+1` basis functions that are nonzero at `x`.

    Returns (span, values): values[j] is the value of raw basis function
    index (span - degree + j) for j in 0..degree, i.e. `values` are the
    only nonzero entries of the full n_basis_raw(knots, degree)-length
    basis vector at this x. Points outside [knots[degree],
    knots[-degree-1]] are clamped to the nearest boundary span, matching
    extrapolation="constant" (the basis freezes at its boundary value,
    since the boundary span's basis functions are used unchanged past the
    edge -- verified against sklearn's own extrapolation="constant" output).
    """
    n_total = n_basis_raw(knots, degree)
    lo_x, hi_x = knots[degree], knots[n_total]
    if x <= lo_x:
        span = degree
        x = lo_x   # freeze at the boundary value (extrapolation="constant")
    elif x >= hi_x:
        span = n_total - 1
        x = hi_x
    else:
        span = int(np.searchsorted(knots, x, side="right") - 1)
        span = max(degree, min(span, n_total - 1))

    N = np.zeros(degree + 1)
    left = np.zeros(degree + 1)
    right = np.zeros(degree + 1)
    N[0] = 1.0
    for j in range(1, degree + 1):
        left[j] = x - knots[span + 1 - j]
        right[j] = knots[span + j] - x
        saved = 0.0
        for r in range(j):
            temp = N[r] / (right[r + 1] + left[j - r])
            N[r] = saved + right[r + 1] * temp
            saved = left[j - r] * temp
        N[j] = saved

    return span, N


def basis_vector(knots, degree, x, n_features_out):
    """Full sparse-but-materialized basis vector of length n_features_out
    (== n_basis_raw(knots, degree) - 1, matching sklearn's include_bias=False
    convention of dropping the LAST raw basis function). This is the
    feature vector fed to a logistic regression's standardization/coef
    step, so it needs to be the same length/ordering sklearn produced at
    fit time.
    """
    span, values = nonzero_basis(knots, degree, x)
    out = np.zeros(n_features_out)
    for j in range(degree + 1):
        idx = span - degree + j
        if idx < n_features_out:  # drops contributions from the dropped last raw basis function
            out[idx] = values[j]
    return out
