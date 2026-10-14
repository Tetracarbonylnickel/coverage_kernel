"""
nn_train.py
===========
Training of the neural-network ensemble branching model (optional
dependencies: jax, flax, optax; install with `pip install -e .[train]`).

The network (ConstrainedSoftmaxMLP) maps (standardized velocity, standardized
angle, coverage / n_sites) to one logit per outcome class. The logit of the
O2 class is shifted by log(coverage fraction) during training as well, so
P(O2 | coverage 0) = 0 holds differentiably; at inference the class is also
set to exactly zero at coverage 0 (nn.py).

fit_coverage_nn_ensemble trains the ensemble used in the paper:
  - member i uses its own random 80/20 split (split seed i) and its own
    initialization and shuffling seed i,
  - fixed number of epochs, no early stopping, final weights kept,
  - Adam with a constant learning rate, mean cross-entropy loss,
  - optional coverage weighting: every training impact gets the weight
    share(c)^(-gamma), where share(c) is the fraction of training impacts at its
    coverage c, capped at `cap` times the weight of the most common coverage and
    normalized to mean 1. It raises the influence of the rare high coverages;
    the target p(outcome | inputs) is unchanged,
  - training and validation log-loss after every epoch (always unweighted),
  - the self-consistent coverage ceiling of the ensemble (coverage_model).
The trained networks are converted to the NumPy inference of nn.py, which is
what everything downstream (export, walks, Fortran) uses.

Matrix products run at "highest" precision, so GPUs that default to
TensorFloat-32 give the same training as CPUs (up to summation order).
"""

import time

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    import optax
    from flax import nnx
except ImportError as exc:  # pragma: no cover
    raise ImportError("coverage_kernel.nn_train needs jax, flax and optax: pip install -e .[train]") from exc

from . import nn
from .coverage_model import CoveragePhysics

ACTIVATIONS = {"silu": nnx.silu, "tanh": nnx.tanh, "relu": nnx.relu, "gelu": nnx.gelu}
STD_FLOOR = 1e-8   # added to the standard deviation of the inputs (as in all trained models)


# ── model ─────────────────────────────────────────────────────────────────────
class MLP(nnx.Module):
    """Feed-forward network: hidden layers with `act_func`, linear output layer."""

    def __init__(self, in_dim, hidden_list, out_dim, *, rngs, act_func=nnx.silu):
        self.act_func = act_func
        self.layers = []
        d = in_dim
        for h in hidden_list:
            self.layers.append(nnx.Linear(d, h, rngs=rngs))
            d = h
        self.layers.append(nnx.Linear(d, out_dim, rngs=rngs))

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = self.act_func(layer(x))
        return self.layers[-1](x)


class ConstrainedSoftmaxMLP(nnx.Module):
    """MLP -> logits -> log(coverage fraction) offset on `offset_class` -> softmax.

    Input x: (batch, 3) = (standardized velocity, standardized angle, coverage fraction),
    coverage fraction last and unscaled (the offset needs its true value)."""

    def __init__(self, in_dim=3, hidden_list=(64, 64, 64), act_func=nnx.silu, n_classes=3, offset_class=2, *, rngs):
        self.offset_class = offset_class
        self.n_classes = n_classes
        self.net = MLP(in_dim, hidden_list, n_classes, rngs=rngs, act_func=act_func)

    def logits(self, x):
        offset = jnp.log(jnp.maximum(x[:, -1], nn.OFFSET_FLOOR))
        return self.net(x).at[:, self.offset_class].add(offset)

    def __call__(self, x):
        probs = nnx.softmax(self.logits(x), axis=-1)
        zeroed = probs.at[:, self.offset_class].set(0.0)
        total = jnp.sum(zeroed, axis=-1, keepdims=True)
        return jnp.where((x[:, -1] == 0.0)[:, None], zeroed / jnp.where(total > 0, total, 1.0), probs)

    def loss(self, x, y):
        return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(self.logits(x), y))


@nnx.jit
def _step(model, optimizer, x, y):
    loss, grads = nnx.value_and_grad(lambda m: m.loss(x, y))(model)
    optimizer.update(grads)
    return loss


@nnx.jit
def _weighted_step(model, optimizer, x, y, w):
    def loss_fn(m):
        ce = optax.softmax_cross_entropy_with_integer_labels(m.logits(x), y)
        return jnp.sum(w * ce) / jnp.sum(w)
    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(grads)
    return loss


@nnx.jit
def _loss(model, x, y):
    return model.loss(x, y)


# ── data preparation ──────────────────────────────────────────────────────────
def features(velocity, angle, coverage, n_sites, stats=None):
    """Network input (n, 3) float32: standardized velocity, standardized angle, coverage / n_sites.
    stats = [(mean, std + STD_FLOOR)] for velocity and angle, computed from these rows if None."""
    cols = [np.asarray(velocity, dtype=float), np.asarray(angle, dtype=float)]
    if stats is None:
        stats = [(float(c.mean()), float(c.std() + STD_FLOOR)) for c in cols]
    scaled = [(c - m) / s for c, (m, s) in zip(cols, stats)]
    X = np.stack(scaled + [np.asarray(coverage, dtype=float) / n_sites], axis=1).astype(np.float32)
    return jnp.asarray(X), stats


def coverage_weights(coverage, gamma, cap=20.0):
    """Per-impact weight share(c)^(-gamma), capped at `cap` times the weight of the most
    common coverage, normalized to mean 1 (float32)."""
    coverage = np.asarray(coverage)
    values, counts = np.unique(coverage, return_counts=True)
    w_cov = (counts / counts.sum()) ** (-gamma)
    w_cov = np.minimum(w_cov / w_cov.min(), cap)
    w = w_cov[np.searchsorted(values, coverage)]
    return jnp.asarray((w / w.mean()).astype(np.float32))


# ── training ──────────────────────────────────────────────────────────────────
def train_network(X_tr, y_tr, X_va=None, y_va=None, *, seed, hidden=(64, 64, 64), activation="silu",
                  lr=3e-3, batch=8192, epochs=150, weights=None, n_classes=3, offset_class=2):
    """Train one ConstrainedSoftmaxMLP for a fixed number of epochs and keep the final weights.
    weights: per-impact training weights (None = unweighted). Returns (model, curve) with
    curve rows (epoch, training log-loss, validation log-loss), both unweighted."""
    with jax.default_matmul_precision("highest"):
        model = ConstrainedSoftmaxMLP(in_dim=X_tr.shape[1], hidden_list=tuple(hidden), act_func=ACTIVATIONS[activation],
                                      n_classes=n_classes, offset_class=offset_class, rngs=nnx.Rngs(seed))
        optimizer = nnx.Optimizer(model, optax.adam(lr), wrt=nnx.Param)
        n = X_tr.shape[0]
        steps = max(n // batch, 1)
        key = jax.random.key(seed)
        curve = []
        for epoch in range(1, epochs + 1):
            key, sub = jax.random.split(key)
            perm = jax.random.permutation(sub, n)[:steps * batch].reshape(steps, batch)
            model.train()
            for b in perm:
                if weights is None:
                    _step(model, optimizer, X_tr[b], y_tr[b])
                else:
                    _weighted_step(model, optimizer, X_tr[b], y_tr[b], weights[b])
            model.eval()
            va = float(_loss(model, X_va, y_va)) if X_va is not None else float("nan")
            curve.append((epoch, float(_loss(model, X_tr, y_tr)), va))
    return model, np.array(curve)


def to_numpy_mlp(model, stats, activation="silu"):
    """Convert a trained ConstrainedSoftmaxMLP into the NumPy inference network of nn.py."""
    layers = model.net.layers
    out = [(np.asarray(l.kernel.value, dtype=float), np.asarray(l.bias.value, dtype=float),
            activation if i < len(layers) - 1 else "linear") for i, l in enumerate(layers)]
    return nn.MLP(out, [s[0] for s in stats], [s[1] for s in stats])


def split_indices(n, member, val_fraction=0.2):
    """Random split of n impacts for one member (split seed = member index): (training, validation)."""
    perm = np.random.default_rng(member).permutation(n)
    n_val = int(val_fraction * n)
    return perm[n_val:], perm[:n_val]


def fit_coverage_nn_ensemble(velocity, angle, coverage, labels, *, physics: CoveragePhysics, coverage_delta,
                             n_members=8, val_fraction=0.2, gamma=0.0, cap=20.0, hidden=(64, 64, 64),
                             activation="silu", lr=3e-3, batch=8192, epochs=150, n_classes=3,
                             fit_ceiling=True, ceiling_velocities=None, ceiling_angles=None, ceiling_k=3.0,
                             mc_steps=20000, mc_tail=8000, mc_seed=0, model_area_nm2=None, name="nn_ensemble",
                             verbose=True):
    """Train the NN ensemble end-to-end (see module docstring).

    velocity, angle, coverage, labels : one entry per impact; labels in [0, n_classes).
    The O2 offset acts on physics.zeroed_zero_index. Ceiling conditions default to
    the distinct sampled velocities and angles.

    Returns (model, info): model is an nn.NNEnsembleCoverageModel; info holds the
    per-member learning curves, validation indices and timing.
    """
    velocity, angle = np.asarray(velocity, dtype=float), np.asarray(angle, dtype=float)
    coverage, labels = np.asarray(coverage), np.asarray(labels, dtype=np.int32)
    offset_class = physics.zeroed_zero_index
    members, curves, val_sets, rows = [], [], [], []
    for i in range(n_members):
        tr, va = split_indices(len(labels), i, val_fraction)
        X_tr, stats = features(velocity[tr], angle[tr], coverage[tr], physics.n_sites)
        X_va, _ = features(velocity[va], angle[va], coverage[va], physics.n_sites, stats)
        w = coverage_weights(coverage[tr], gamma, cap) if gamma > 0 else None
        t0 = time.time()
        model, curve = train_network(X_tr, jnp.asarray(labels[tr]), X_va, jnp.asarray(labels[va]), seed=i,
                                     hidden=hidden, activation=activation, lr=lr, batch=batch, epochs=epochs,
                                     weights=w, n_classes=n_classes, offset_class=offset_class)
        members.append(to_numpy_mlp(model, stats, activation))
        curves.append(curve)
        val_sets.append(va)
        rows.append({"member": i, "final_train_log_loss": curve[-1, 1], "final_validation_log_loss": curve[-1, 2],
                     "seconds": time.time() - t0})
        if verbose:
            print(f"member {i}: final training log-loss {curve[-1, 1]:.5f}, validation {curve[-1, 2]:.5f} "
                  f"({rows[-1]['seconds']:.0f} s)", flush=True)

    ensemble = nn.NNEnsemble(members, physics.n_sites, offset_class, physics.zeroed_zero_index)
    metadata = {"n_members": n_members, "hidden": list(hidden), "activation": activation, "learning_rate": lr,
                "batch_size": batch, "epochs": epochs, "coverage_weighting_gamma": gamma, "coverage_weighting_cap": cap,
                "validation_fraction": val_fraction, "split": "random, split seed = member index"}
    coverage_max = float(coverage.max())
    if fit_ceiling:
        vels = sorted(set(velocity.tolist())) if ceiling_velocities is None else list(ceiling_velocities)
        angs = sorted(set(angle.tolist())) if ceiling_angles is None else list(ceiling_angles)
        model = nn.NNEnsembleCoverageModel.fit_ceiling(ensemble, physics, coverage_max, coverage_delta, vels, angs,
                                                       k=ceiling_k, n_steps=mc_steps, n_tail=mc_tail, seed=mc_seed,
                                                       model_area_nm2=model_area_nm2, name=name, metadata=metadata)
    else:
        model = nn.NNEnsembleCoverageModel(ensemble, physics, coverage_max, None, None, coverage_delta=coverage_delta,
                                           model_area_nm2=model_area_nm2, name=name, metadata=metadata)
    return model, {"curves": curves, "validation_indices": val_sets, "members": rows}
