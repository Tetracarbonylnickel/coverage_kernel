import numpy as np
import pytest
from sklearn.preprocessing import SplineTransformer

from coverage_kernel import bspline as bsp


@pytest.fixture
def rng():
    return np.random.default_rng(2)


def test_knots_and_basis_match_sklearn(rng):
    max_err = 0.0
    for _ in range(100):
        n_knots = rng.integers(3, 8)
        degree = rng.integers(1, 4)
        x_min, x_max = 0.0, rng.uniform(5, 30)

        st = SplineTransformer(n_knots=n_knots, degree=degree, knots="uniform",
                                extrapolation="constant", include_bias=False)
        st.fit(np.linspace(x_min, x_max, 100).reshape(-1, 1))
        n_out = st.n_features_out_

        my_knots = bsp.uniform_knots(x_min, x_max, n_knots, degree)
        assert np.max(np.abs(my_knots - st.bsplines_[0].t)) < 1e-9

        test_x = np.concatenate([
            rng.uniform(x_min, x_max, 15),
            np.linspace(x_min, x_max, n_knots),
            [x_min - 5, x_max + 5, x_min, x_max],
        ])
        sk_vals = st.transform(test_x.reshape(-1, 1))
        for i, x in enumerate(test_x):
            my_val = bsp.basis_vector(my_knots, degree, x, n_out)
            max_err = max(max_err, np.max(np.abs(my_val - sk_vals[i])))
    assert max_err < 1e-8


def test_extrapolation_is_constant():
    knots = bsp.uniform_knots(0, 16, 5, 2)
    n_out = bsp.n_basis_raw(knots, 2) - 1
    at_boundary = bsp.basis_vector(knots, 2, 16.0, n_out)
    past_boundary = bsp.basis_vector(knots, 2, 50.0, n_out)
    assert np.allclose(at_boundary, past_boundary)


def test_local_support():
    # degree=2: at most 3 of the K raw basis functions nonzero at any point
    knots = bsp.uniform_knots(0, 16, 5, 2)
    for x in np.linspace(-2, 18, 40):
        span, values = bsp.nonzero_basis(knots, 2, x)
        assert np.count_nonzero(values) <= 3
