"""
features.py
============
Physics feature engineering used by the pooled coverage model.

This has to live in a proper importable module, not be defined inline in a
notebook: PooledCoverageModel stores this function as an attribute so
probs() can call it later, and pickle/joblib only serializes a *reference*
to a function (its module + name), not its code -- a function defined in a
notebook's __main__ scope unpickles fine in the *same* kernel but fails in
a different process/notebook with "Can't get attribute ... on <module
'__main__'>", since there's no such name in that other process's __main__.
"""

import numpy as np


def physics_features(coverage, velocity, angle):
    """[coverage, normal energy E_n=(v cos alpha)², parallel velocity v sin alpha] --
    the physics-motivated feature decomposition used by the pooled scheme
    (see docs/global_linear_model.tex in the original coverage_model
    package for the physical motivation)."""
    angle_rad = np.radians(angle)
    v_normal = velocity * np.cos(angle_rad)
    v_parallel = velocity * np.sin(angle_rad)
    return np.array([coverage, v_normal**2, v_parallel])
