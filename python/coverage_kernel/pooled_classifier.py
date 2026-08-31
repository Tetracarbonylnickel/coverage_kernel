"""
pooled_classifier.py
=====================
Generic building block: a K-class softmax (multinomial logistic) classifier
over a feature vector assembled from named "terms" applied to columns of a
raw input vector -- each term is either a B-spline basis expansion of one
column (lets the logit bend non-linearly in that column) or that column
passed through linearly.

This has no idea what the columns *mean* -- it doesn't know about
coverage, angle, velocity, or outcomes. It is used two ways elsewhere in
this project:
  - directly, as the "pooled" scheme (what the old codebase called
    "global": one fit shared across every sampled condition, spline bases
    on coverage/normal-energy, linear on parallel-velocity);
  - as the per-cell model inside grid_classifier.py.

Fitting uses sklearn's LogisticRegression. Evaluation is entirely from-scratch (bspline.py's
basis evaluator + a plain matmul + softmax), independent of sklearn, so the
exact same evaluation logic can be ported line-for-line to Fortran and
validated bit-for-bit against this module rather than against sklearn's
internals.

Handles classes missing from the training data explicitly: with n_classes
possible outcomes, a small/sparse fit (e.g. one grid cell of a fine grid)
may never observe every one of them. Missing classes are pinned to
probability exactly 0 in the output (not modeled by the softmax at all);
the classes that *were* observed still get a valid, properly normalized
softmax among themselves. This is standard sklearn behavior at the
coefficient level (`classes_` only lists observed labels) -- carried
through explicitly here so the rest of this project never has to special-
case it, and so a 2-observed-class fit doesn't hit sklearn's separate
binary-vs-multinomial code path (which returns a single coefficient row
using a sigmoid, not a softmax): this module always reconstructs a
uniform, always-softmax representation regardless of how many classes were
observed, by anchoring any single-row binary fit against an implicit
all-zero row for the class sklearn treated as reference.
"""

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression

from . import bspline


@dataclass(frozen=True)
class SplineTerm:
    """Expand column `column` of the raw feature vector into a B-spline
    basis with `n_knots` uniform interior knots and the given `degree`.
    """
    column: int
    n_knots: int = 5
    degree: int = 2


@dataclass(frozen=True)
class LinearTerm:
    """Pass column `column` of the raw feature vector through unchanged."""
    column: int


def _n_term_features(term):
    if isinstance(term, SplineTerm):
        return term.n_knots + term.degree - 1 - 1  # sklearn include_bias=False convention
    return 1


def _term_columns_raw(X_raw, terms, knot_vectors):
    """Build the expanded feature matrix (N, sum of each term's width) from
    raw columns. knot_vectors[i] is the fitted knot vector for terms[i] if
    it's a SplineTerm (ignored otherwise) -- passed in rather than
    recomputed, so fit and eval always use the exact same knots.
    """
    N = X_raw.shape[0]
    blocks = []
    for term, knots in zip(terms, knot_vectors):
        if isinstance(term, SplineTerm):
            n_out = _n_term_features(term)
            col = X_raw[:, term.column]
            block = np.empty((N, n_out))
            for i in range(N):
                block[i] = bspline.basis_vector(knots, term.degree, col[i], n_out)
            blocks.append(block)
        else:
            blocks.append(X_raw[:, [term.column]])
    return np.hstack(blocks)


class PooledClassifier:
    """A fitted pooled softmax-logistic classifier. Construct via
    fit_pooled_classifier; evaluate via .eval(x_raw).
    """

    def __init__(self, terms, knot_vectors, scaler_mean, scaler_scale,
                 coef, intercept, present_classes, n_classes, raw_feature_ranges=None):
        self.terms = terms
        self.knot_vectors = knot_vectors     # one per term (None for LinearTerm)
        self.scaler_mean = scaler_mean
        self.scaler_scale = scaler_scale
        self.coef = coef                     # (len(present_classes), n_features)
        self.intercept = intercept           # (len(present_classes),)
        self.present_classes = present_classes   # sorted array of observed class labels
        self.n_classes = n_classes           # total possible classes (K), incl. never-observed ones
        # (n_raw_columns, 2) observed (min, max) per raw column at fit time --
        # generic bookkeeping (this module has no idea which column, if any,
        # means "coverage"); the domain layer reads whichever column it cares
        # about to get that condition's own trusted feature range.
        self.raw_feature_ranges = raw_feature_ranges

    def _features(self, x_raw):
        """x_raw: (d,) single row or (N, d) batch. Returns (n_features,) or (N, n_features)."""
        x_raw = np.atleast_2d(x_raw)
        z = _term_columns_raw(x_raw, self.terms, self.knot_vectors)
        return (z - self.scaler_mean) / self.scaler_scale

    def eval(self, x_raw):
        """Evaluate P(class=k | x_raw) for k=0..n_classes-1. Returns (n_classes,)
        for a single row input, (N, n_classes) for a batch.
        """
        single = np.ndim(x_raw) == 1 or (np.ndim(x_raw) == 0)
        z = self._features(x_raw)                      # (N, n_features)
        logits = z @ self.coef.T + self.intercept       # (N, len(present_classes))
        logits -= logits.max(axis=1, keepdims=True)     # numerically-stable softmax
        exp = np.exp(logits)
        present_probs = exp / exp.sum(axis=1, keepdims=True)

        out = np.zeros((z.shape[0], self.n_classes))
        out[:, self.present_classes] = present_probs
        return out[0] if single else out


def fit_pooled_classifier(X_raw, y, terms, *, n_classes, C=1.0, max_iter=5000):
    """Fit a pooled softmax-logistic classifier.

    Parameters
    ----------
    X_raw : (N, d) raw feature matrix (before any term's spline/linear expansion)
    y : (N,) integer class labels in [0, n_classes)
    terms : list of SplineTerm/LinearTerm, each referencing a column of X_raw
    n_classes : total number of possible classes K (may exceed the number
        actually observed in y -- see PooledClassifier's docstring on how
        unobserved classes are handled)
    C : inverse L2 regularization strength (sklearn LogisticRegression convention)
    """
    X_raw = np.asarray(X_raw, dtype=float)
    y = np.asarray(y, dtype=int)
    raw_feature_ranges = np.column_stack([X_raw.min(axis=0), X_raw.max(axis=0)])

    knot_vectors = []
    for term in terms:
        if isinstance(term, SplineTerm):
            col = X_raw[:, term.column]
            knot_vectors.append(bspline.uniform_knots(col.min(), col.max(), term.n_knots, term.degree))
        else:
            knot_vectors.append(None)

    Z = _term_columns_raw(X_raw, terms, knot_vectors)
    mean = Z.mean(axis=0)
    scale = Z.std(axis=0)
    scale[scale == 0] = 1.0   # guard a constant feature (e.g. a degenerate spline column)
    Zs = (Z - mean) / scale

    present_classes = np.unique(y)
    if len(present_classes) == 1:
        # a single-class fit: every event had the same label. sklearn's
        # LogisticRegression outright refuses to fit with only one class
        # present, so this has to be special-cased *before* calling it, not
        # inferred from clf.classes_ afterwards. Represent as a softmax that
        # always returns that class with probability 1 (an all-zero
        # coef/intercept row achieves this without any special-cased
        # "constant" eval path -- keeps PooledClassifier.eval() uniform).
        coef = np.zeros((1, Zs.shape[1]))
        intercept = np.array([0.0])
    elif len(present_classes) == 2:
        clf = LogisticRegression(solver="lbfgs", C=C, max_iter=max_iter)
        clf.fit(Zs, y)
        # sklearn's binary special case: coef_/intercept_ have a single row,
        # implicitly "class present_classes[1] vs reference class
        # present_classes[0]" through a sigmoid. Reconstruct as an explicit
        # 2-row softmax anchored at an all-zero row for present_classes[0],
        # so PooledClassifier.eval() never needs a sigmoid special case.
        coef = np.vstack([np.zeros_like(clf.coef_[0]), clf.coef_[0]])
        intercept = np.array([0.0, clf.intercept_[0]])
    else:
        clf = LogisticRegression(solver="lbfgs", C=C, max_iter=max_iter)
        clf.fit(Zs, y)
        coef = clf.coef_
        intercept = clf.intercept_

    return PooledClassifier(terms, knot_vectors, mean, scale, coef, intercept,
                             present_classes, n_classes, raw_feature_ranges=raw_feature_ranges)
