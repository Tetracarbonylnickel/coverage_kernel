"""
nn.py
=====
Neural-network ensemble branching model, evaluated in plain NumPy (float64).

Each member is a feed-forward network that maps
    (standardized velocity, standardized angle, coverage / n_sites)
to one logit per outcome class. Before the softmax, the logit of
`offset_class` (O2 desorption in this project) is shifted by
log(max(coverage / n_sites, OFFSET_FLOOR)), so that its probability vanishes
at zero coverage by construction; at exactly zero coverage the class is
also set to zero and the rest renormalized (zero-at-zero-coverage gate).
The ensemble probability is the mean of the member probabilities.

This is exactly the inference of the networks trained by nn_train.py
(ConstrainedSoftmaxMLP), so a model trained there and evaluated here gives
the same probabilities up to floating-point precision.

Beyond the highest sampled coverage the ensemble is wrapped in the same
coverage physics as the grid model (gates, linear blend toward the
no-adsorption endpoint, self-consistent ceiling), via
coverage_model.CeilingCoverageModel. NNEnsembleCoverageModel is that wrapper;
its `members` give the same wrapper around every single network (sharing
the ensemble's ceiling), for the member spread.

Training needs JAX (nn_train.py); this module only needs NumPy.
"""

import numpy as np

from .coverage_model import CeilingCoverageModel, CoveragePhysics

OFFSET_FLOOR = 1e-8   # log(coverage fraction) is floored here; the gate handles coverage == 0 exactly


def _silu(x):
    with np.errstate(over="ignore"):
        return x / (1.0 + np.exp(-x))


ACTIVATIONS = {
    "silu": _silu,
    "tanh": np.tanh,
    "relu": lambda x: np.maximum(x, 0.0),
    "linear": lambda x: x,
}


class MLP:
    """One feed-forward network.

    layers : list of (weights, biases, activation) with weights of shape
        (n_in, n_out), biases (n_out,), activation one of ACTIVATIONS.
    input_mean, input_std : standardization of the conditioning inputs
        (velocity, angle); the coverage fraction is appended unscaled.
    """

    def __init__(self, layers, input_mean, input_std):
        self.layers = [(np.asarray(w, dtype=float), np.asarray(b, dtype=float), str(act)) for w, b, act in layers]
        for _, _, act in self.layers:
            if act not in ACTIVATIONS:
                raise ValueError(f"unknown activation {act!r}")
        self.input_mean = np.asarray(input_mean, dtype=float)
        self.input_std = np.asarray(input_std, dtype=float)

    @property
    def n_out(self):
        return self.layers[-1][0].shape[1]

    def inputs(self, coverages, conditioning, n_sites):
        """(n, 3) network input for coverages (n,) at one condition (velocity, angle)."""
        c = np.asarray(coverages, dtype=float)
        cond = (np.asarray(conditioning, dtype=float) - self.input_mean) / self.input_std
        cols = [np.full(c.shape, x) for x in cond] + [c / n_sites]
        return np.stack(cols, axis=1)

    def logits(self, x):
        for w, b, act in self.layers:
            x = ACTIVATIONS[act](x @ w + b)
        return x


class NNEnsemble:
    """Raw ensemble (no blend, no ceiling): mean of the member probabilities.

    probs_batch(coverages, velocity, angle) -> (n, n_classes); member_probs_batch(i, ...)
    gives one member.
    """

    def __init__(self, members, n_sites, offset_class, zeroed_zero_index=None):
        self.members = list(members)
        self.n_sites = int(n_sites)
        self.offset_class = int(offset_class)
        self.zeroed_zero_index = self.offset_class if zeroed_zero_index is None else int(zeroed_zero_index)

    @property
    def n_classes(self):
        return self.members[0].n_out

    def member_probs_batch(self, i, coverages, velocity, angle):
        net = self.members[i]
        c = np.asarray(coverages, dtype=float)
        z = net.logits(net.inputs(c, (velocity, angle), self.n_sites))
        z[:, self.offset_class] += np.log(np.maximum(c / self.n_sites, OFFSET_FLOOR))
        p = np.exp(z - z.max(axis=1, keepdims=True))
        p /= p.sum(axis=1, keepdims=True)
        p[c == 0, self.zeroed_zero_index] = 0.0
        return p / p.sum(axis=1, keepdims=True)

    def probs_batch(self, coverages, velocity, angle):
        return np.mean([self.member_probs_batch(i, coverages, velocity, angle)
                        for i in range(len(self.members))], axis=0)

    def __call__(self, coverages, velocity, angle):
        return self.probs_batch(coverages, velocity, angle)


class _MemberRaw:
    """Picklable raw_batch callable for one member (used by NNEnsembleCoverageModel.members)."""

    def __init__(self, ensemble, i):
        self.ensemble, self.i = ensemble, i

    def __call__(self, coverages, velocity, angle):
        return self.ensemble.member_probs_batch(self.i, coverages, velocity, angle)


class NNEnsembleCoverageModel(CeilingCoverageModel):
    """The NN ensemble with the coverage physics (gate, blend, ceiling).

    Same interface as every coverage model of this package:
    probs(coverage, velocity, angle), probs_batch(coverages, velocity, angle),
    probs_fn(velocity, angle) -> scalar callable.
    """

    def __init__(self, ensemble: NNEnsemble, physics: CoveragePhysics, coverage_max, ceiling_axes, ceiling_grid,
                 coverage_delta=None, model_area_nm2=None, name="nn_ensemble", metadata=None):
        super().__init__(ensemble, physics, coverage_max, ceiling_axes, ceiling_grid, name=name,
                         coverage_delta=coverage_delta, model_area_nm2=model_area_nm2)
        self.ensemble = ensemble
        self.metadata = dict(metadata or {})

    @property
    def members(self):
        """Every single network with the same physics and the ensemble's ceiling."""
        if len(self.ensemble.members) == 1:
            return []
        return [CeilingCoverageModel(_MemberRaw(self.ensemble, i), self.physics, self.coverage_max,
                                     self.ceiling_axes, self.ceiling_grid, name=f"{self.name} member {i}",
                                     coverage_delta=self.coverage_delta, model_area_nm2=self.model_area_nm2)
                for i in range(len(self.ensemble.members))]

    @classmethod
    def fit_ceiling(cls, ensemble, physics, coverage_max, coverage_delta, velocities, angles, *,
                    k=3.0, n_steps=20000, n_tail=8000, seed=0, n_jobs=-1, model_area_nm2=None,
                    name="nn_ensemble", metadata=None):
        """Wrap a trained ensemble and compute its self-consistent ceiling at every
        sampled (velocity, angle) condition (see CeilingCoverageModel.fit)."""
        base = CeilingCoverageModel.fit(ensemble, physics, coverage_max, coverage_delta, velocities, angles,
                                        k=k, n_steps=n_steps, n_tail=n_tail, seed=seed, n_jobs=n_jobs, name=name)
        return cls(ensemble, physics, coverage_max, base.ceiling_axes, base.ceiling_grid,
                   coverage_delta=coverage_delta, model_area_nm2=model_area_nm2, name=name, metadata=metadata)
