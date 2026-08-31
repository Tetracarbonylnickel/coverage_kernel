import numpy as np
import pytest
from scipy.interpolate import PchipInterpolator

from coverage_kernel import interp1d as i1d


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def test_pchip_matches_scipy_scalar(rng):
    max_err = 0.0
    for _ in range(300):
        n = rng.integers(2, 12)
        x = np.unique(np.sort(rng.uniform(0, 20, n)))
        n = len(x)
        if n < 2:
            continue
        y = rng.normal(size=n)
        ref = PchipInterpolator(x, y, extrapolate=False)
        for _ in range(10):
            xq = rng.uniform(x[0], x[-1])
            expected = float(ref(xq))
            _, lo, hi = i1d.local_window(x, xq, "pchip")
            got = i1d.pchip_eval_window(x[lo:hi + 1], y[lo:hi + 1], xq,
                                         is_left_edge=(lo == 0), is_right_edge=(hi == n - 1))
            max_err = max(max_err, abs(got - expected))
    assert max_err < 1e-9


def test_pchip_matches_scipy_vector(rng):
    max_err = 0.0
    for _ in range(100):
        n = rng.integers(2, 10)
        x = np.unique(np.sort(rng.uniform(0, 20, n)))
        n = len(x)
        if n < 2:
            continue
        K = 3
        Y = rng.normal(size=(n, K))
        refs = [PchipInterpolator(x, Y[:, k], extrapolate=False) for k in range(K)]
        for _ in range(10):
            xq = rng.uniform(x[0], x[-1])
            expected = np.array([float(r(xq)) for r in refs])
            _, lo, hi = i1d.local_window(x, xq, "pchip")
            got = i1d.pchip_eval_window(x[lo:hi + 1], Y[lo:hi + 1], xq,
                                         is_left_edge=(lo == 0), is_right_edge=(hi == n - 1))
            max_err = max(max_err, np.max(np.abs(got - expected)))
    assert max_err < 1e-8


def test_pchip_exact_at_nodes(rng):
    n = 8
    x = np.unique(np.sort(rng.uniform(0, 20, n)))
    n = len(x)
    y = rng.normal(size=n)
    ref = PchipInterpolator(x, y, extrapolate=False)
    for k in range(n):
        xq = x[k]
        _, lo, hi = i1d.local_window(x, xq, "pchip")
        got = i1d.pchip_eval_window(x[lo:hi + 1], y[lo:hi + 1], xq,
                                     is_left_edge=(lo == 0), is_right_edge=(hi == n - 1))
        assert abs(got - float(ref(xq))) < 1e-9


def test_pchip_two_point_axis():
    x = np.array([0.0, 5.0])
    y = np.array([1.0, 3.0])
    ref = PchipInterpolator(x, y, extrapolate=False)
    for xq in [0.0, 1.3, 5.0]:
        _, lo, hi = i1d.local_window(x, xq, "pchip")
        got = i1d.pchip_eval_window(x[lo:hi + 1], y[lo:hi + 1], xq, is_left_edge=True, is_right_edge=True)
        assert abs(got - float(ref(xq))) < 1e-9


def test_linear_matches_numpy_interp(rng):
    x = np.unique(np.sort(rng.uniform(0, 20, 8)))
    y = rng.normal(size=len(x))
    for _ in range(20):
        xq = rng.uniform(x[0], x[-1])
        _, lo, hi = i1d.local_window(x, xq, "linear")
        got = i1d.linear_eval_window(x[lo:hi + 1], y[lo:hi + 1], xq)
        expected = np.interp(xq, x, y)
        assert abs(got - expected) < 1e-12


def test_linear_extrapolates_past_range():
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 1.0, 4.0])
    _, lo, hi = i1d.local_window(x, 3.0, "linear")
    got = i1d.linear_eval_window(x[lo:hi + 1], y[lo:hi + 1], 3.0)
    assert abs(got - 7.0) < 1e-12  # continues the boundary segment's slope (3) past x=2


def test_multilinear_clipped_matches_scipy():
    from scipy.interpolate import RegularGridInterpolator
    rng = np.random.default_rng(3)
    ax0 = np.array([0.0, 1.0, 3.0, 6.0])
    ax1 = np.array([-10.0, 0.0, 10.0])
    values = rng.normal(size=(4, 3))
    ref = RegularGridInterpolator((ax0, ax1), values, method="linear", bounds_error=False, fill_value=None)

    for _ in range(50):
        q0 = rng.uniform(-5, 10)
        q1 = rng.uniform(-20, 20)
        q0c = min(max(q0, ax0[0]), ax0[-1])
        q1c = min(max(q1, ax1[0]), ax1[-1])
        expected = ref([[q0c, q1c]])[0]
        got = i1d.multilinear_clipped([ax0, ax1], values, [q0, q1])
        assert abs(got - expected) < 1e-10
