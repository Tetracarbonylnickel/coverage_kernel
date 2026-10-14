import numpy as np
import pytest

from coverage_kernel import coverage_model as cm
from coverage_kernel import nn


@pytest.fixture
def physics():
    return cm.CoveragePhysics(n_sites=48, zeroed_zero_index=2, zeroed_full_index=0, absorbing_full_index=2)


def _member(rng, hidden=(6, 5), act="silu"):
    sizes = [3, *hidden, 3]
    layers = [(rng.normal(size=(i, o)), rng.normal(size=o), act if k < len(sizes) - 2 else "linear")
              for k, (i, o) in enumerate(zip(sizes[:-1], sizes[1:]))]
    return nn.MLP(layers, [7000.0, 40.0], [1400.0, 28.0])


def _by_hand(net, c, v, a, n_sites=48):
    """Explicit forward pass of one member for a single query."""
    x = np.array([(v - 7000.0) / 1400.0, (a - 40.0) / 28.0, c / n_sites])
    for w, b, act in net.layers:
        x = x @ w + b
        x = {"silu": lambda z: z / (1 + np.exp(-z)), "tanh": np.tanh, "linear": lambda z: z}[act](x)
    x[2] += np.log(max(c / n_sites, 1e-8))
    p = np.exp(x - x.max()); p /= p.sum()
    if c == 0:
        p[2] = 0; p /= p.sum()
    return p


@pytest.mark.parametrize("act", ["silu", "tanh"])
def test_member_matches_explicit_forward_pass(act):
    rng = np.random.default_rng(0)
    net = _member(rng, act=act)
    ens = nn.NNEnsemble([net], 48, offset_class=2)
    c = np.array([0.0, 0.5, 3.0, 16.0, 30.0])
    got = ens.member_probs_batch(0, c, 6500.0, 25.0)
    want = np.array([_by_hand(net, ci, 6500.0, 25.0) for ci in c])
    assert np.allclose(got, want, atol=1e-14)


def test_offset_and_gate():
    rng = np.random.default_rng(1)
    ens = nn.NNEnsemble([_member(rng)], 48, offset_class=2)
    p = ens.probs_batch(np.array([0.0, 1e-6, 1.0]), 7000.0, 40.0)
    assert p[0, 2] == 0.0                         # exact zero at coverage 0
    assert p[1, 2] < 1e-6 * 10                    # the log offset suppresses O2 near 0
    assert np.allclose(p.sum(1), 1.0)


def test_ensemble_is_mean_of_members():
    rng = np.random.default_rng(2)
    ens = nn.NNEnsemble([_member(rng) for _ in range(3)], 48, offset_class=2)
    c = np.linspace(0, 20, 11)
    mean = np.mean([ens.member_probs_batch(i, c, 8000.0, 60.0) for i in range(3)], axis=0)
    assert np.allclose(ens.probs_batch(c, 8000.0, 60.0), mean, atol=1e-15)


def test_unknown_activation_rejected():
    with pytest.raises(ValueError):
        nn.MLP([(np.eye(3), np.zeros(3), "swish")], [0, 0], [1, 1])


def test_coverage_model_members_share_ceiling(physics):
    rng = np.random.default_rng(3)
    ens = nn.NNEnsemble([_member(rng) for _ in range(2)], 48, offset_class=2)
    axes, grid = [[5000.0, 9000.0], [0.0, 80.0]], np.array([[20.0, 12.0], [19.0, 10.0]])
    model = nn.NNEnsembleCoverageModel(ens, physics, 16.0, axes, grid, coverage_delta=[1, 0, -1])
    members = model.members
    assert len(members) == 2
    for m in members:
        assert m.ceiling(7000.0, 40.0) == model.ceiling(7000.0, 40.0)
    # inside the sampled range the wrapped model is the raw ensemble
    assert np.allclose(model.probs(5.0, 7000.0, 40.0), ens.probs_batch([5.0], 7000.0, 40.0)[0], atol=1e-15)
    # far beyond the ceiling: no adsorption
    assert model.probs(40.0, 7000.0, 40.0)[0] == 0.0
    # table-based and exact evaluation agree at integer coverages
    fn = model.probs_fn(7000.0, 40.0)
    for c in range(0, 25):
        assert np.allclose(fn(c), model.probs(c, 7000.0, 40.0), atol=1e-12)


def test_fit_ceiling(physics):
    rng = np.random.default_rng(4)
    ens = nn.NNEnsemble([_member(rng)], 48, offset_class=2)
    model = nn.NNEnsembleCoverageModel.fit_ceiling(ens, physics, 16.0, [1, 0, -1], [5000.0, 9000.0], [0.0, 80.0],
                                                   n_steps=1500, n_tail=500, n_jobs=1)
    assert model.ceiling_grid.shape == (2, 2)
    assert np.all((model.ceiling_grid > 0) & (model.ceiling_grid <= 48))


def test_jax_training_matches_numpy_inference(physics):
    """Trained JAX network and its NumPy conversion give the same probabilities."""
    pytest.importorskip("jax")
    pytest.importorskip("flax")
    from coverage_kernel import nn_train
    import jax.numpy as jnp

    rng = np.random.default_rng(5)
    n = 4000
    v = rng.choice([5000.0, 7000.0, 9000.0], n); a = rng.choice([0.0, 40.0, 80.0], n)
    c = rng.integers(0, 17, n)
    p_ads = np.clip(0.8 - 0.04 * c, 0.05, 1); p_o2 = np.where(c == 0, 0, np.clip(0.03 * c, 0, 0.9))
    labels = np.array([rng.choice(3, p=[pa, 1 - pa - po, po] / np.sum([pa, 1 - pa - po, po]))
                       for pa, po in zip(p_ads, p_o2)])
    X, stats = nn_train.features(v, a, c, 48)
    model, curve = nn_train.train_network(X, jnp.asarray(labels, dtype=jnp.int32), X, jnp.asarray(labels, dtype=jnp.int32),
                                          seed=0, hidden=(8, 8), batch=512, epochs=3)
    assert curve.shape == (3, 3) and curve[-1, 1] < curve[0, 1] + 1e-3
    net = nn_train.to_numpy_mlp(model, stats)
    ens = nn.NNEnsemble([net], 48, offset_class=2)
    with __import__("jax").default_matmul_precision("highest"):
        p_jax = np.asarray(model(X[:200]), dtype=float)
    p_np = np.concatenate([ens.probs_batch([ci], vi, ai) for ci, vi, ai in zip(c[:200], v[:200], a[:200])])
    assert np.max(np.abs(p_jax - p_np)) < 1e-5     # float32 training network vs float64 inference


def test_coverage_weights():
    pytest.importorskip("jax")
    from coverage_kernel import nn_train
    cov = np.array([5] * 90 + [15] * 10)
    w = np.asarray(nn_train.coverage_weights(cov, gamma=0.5, cap=20.0))
    assert abs(w.mean() - 1.0) < 1e-6
    assert w[-1] / w[0] == pytest.approx(3.0, rel=1e-5)   # (0.1 / 0.9) ** -0.5
