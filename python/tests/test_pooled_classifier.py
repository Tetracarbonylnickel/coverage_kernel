import numpy as np
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

from coverage_kernel import pooled_classifier as pc


@pytest.fixture
def rng():
    return np.random.default_rng(3)


def test_three_class_matches_sklearn(rng):
    N = 3000
    coverage = rng.uniform(0, 16, N)
    e_normal = rng.uniform(1e6, 1e8, N)
    v_parallel = rng.uniform(0, 9000, N)
    X_raw = np.column_stack([coverage, e_normal, v_parallel])
    logits_true = np.column_stack([0.1 * coverage, -2e-8 * e_normal, 0.0002 * v_parallel])
    probs_true = np.exp(logits_true)
    probs_true /= probs_true.sum(axis=1, keepdims=True)
    y = np.array([rng.choice(3, p=p) for p in probs_true])

    terms = [pc.SplineTerm(0, n_knots=5, degree=2), pc.SplineTerm(1, n_knots=5, degree=2), pc.LinearTerm(2)]
    mine = pc.fit_pooled_classifier(X_raw, y, terms, n_classes=3, C=1.0)

    ct = ColumnTransformer([
        ("cov", SplineTransformer(n_knots=5, degree=2, knots="uniform",
                                   extrapolation="constant", include_bias=False), [0]),
        ("en", SplineTransformer(n_knots=5, degree=2, knots="uniform",
                                  extrapolation="constant", include_bias=False), [1]),
    ], remainder="passthrough")
    pipe = Pipeline([("spline", ct), ("scale", StandardScaler()),
                      ("logit", LogisticRegression(solver="lbfgs", C=1.0, max_iter=5000))])
    pipe.fit(X_raw, y)

    test_pts = np.column_stack([rng.uniform(0, 16, 30), rng.uniform(1e6, 1e8, 30), rng.uniform(0, 9000, 30)])
    assert np.max(np.abs(pipe.predict_proba(test_pts) - mine.eval(test_pts))) < 1e-8


def test_two_observed_classes(rng):
    N = 500
    coverage = rng.uniform(0, 16, N)
    X_raw = coverage.reshape(-1, 1)
    y = np.where(coverage < 8, 0, 2)   # class 1 never observed
    terms = [pc.SplineTerm(0, n_knots=5, degree=2)]
    mine = pc.fit_pooled_classifier(X_raw, y, terms, n_classes=3, C=1.0)
    test = np.array([[1.0], [7.0], [8.0], [9.0], [15.0]])
    probs = mine.eval(test)
    assert np.all(probs[:, 1] == 0.0)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_single_observed_class(rng):
    N = 200
    coverage = rng.uniform(0, 16, N)
    X_raw = coverage.reshape(-1, 1)
    y = np.zeros(N, dtype=int)
    terms = [pc.SplineTerm(0, n_knots=5, degree=2)]
    mine = pc.fit_pooled_classifier(X_raw, y, terms, n_classes=3, C=1.0)
    probs = mine.eval(np.array([[1.0], [7.0], [15.0]]))
    assert np.allclose(probs, np.array([1.0, 0.0, 0.0]))


def test_probabilities_valid(rng):
    N = 1000
    coverage = rng.uniform(0, 16, N)
    y = rng.integers(0, 3, N)
    terms = [pc.SplineTerm(0, n_knots=4, degree=2)]
    mine = pc.fit_pooled_classifier(coverage.reshape(-1, 1), y, terms, n_classes=3, C=1.0)
    probs = mine.eval(rng.uniform(-5, 25, (50, 1)))
    assert np.all(probs >= 0)
    assert np.allclose(probs.sum(axis=1), 1.0)
