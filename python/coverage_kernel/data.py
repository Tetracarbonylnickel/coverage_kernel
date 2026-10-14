"""
data.py
=======
Reader for this project's MLIP impact data (optional dependencies: pandas,
znh5md). The generic core and the coverage physics never import this
module; they only see the arrays it produces.

Files:
  fragment_velocities_<velocity>.h5   one per incidence velocity, one HDF5 group
                                      per incidence angle with the datasets
                                      imp_idx, n_atoms, in_velo_mag, in_velo_z (and more)
  starting_frames.h5                  the starting structures (znh5md); the
                                      coverage of a structure is its atom count
                                      minus the substrate atoms

Impact imp_idx starts from structure imp_idx % n_frames.
"""

import h5py
import numpy as np


def load_starting_coverages(init_path, n_substrate_atoms):
    """Coverage (number of adsorbed atoms) of every starting structure, and the number of structures."""
    import znh5md

    frames = znh5md.IO(init_path)
    n_frames = len(frames)
    return np.array([len(frames[i].positions) - n_substrate_atoms for i in range(n_frames)]), n_frames


def load_trajectories(results_prefix, init_path, velocities, angles, *, n_substrate_atoms,
                      velocity_precision=0.01, angle_precision=0.01):
    """Read the impacts of every (velocity, angle) combination into one table.

    results_prefix : path prefix of the impact files; the file of velocity v is
        f"{results_prefix}{v}.h5", e.g. ".../labelled_impacts/fragment_velocities_".
    velocities, angles : nominal values naming the files and the groups in them.

    Every row is labelled with the mean real incidence speed and the mean real
    polar angle of its group (rounded to velocity_precision / angle_precision),
    so the labels are the true sampled conditions, not the rounded nominal ones.
    The polar angle is measured from the surface normal (0 = normal incidence),
    folded onto [0, 90] via |in_velo_z|.

    Returns a pandas.DataFrame with columns velocity, angle, coverage, n_atoms
    (one row per desorbing fragment; n_atoms = 0 is adsorption).
    """
    import pandas as pd

    coverage_by_frame, n_frames = load_starting_coverages(init_path, n_substrate_atoms)

    def _snap(x, precision):
        return round(round(x / precision) * precision, 10)

    parts = []
    for velo in velocities:
        with h5py.File(f"{results_prefix}{velo}.h5", "r") as f:
            for angle in angles:
                grp = f[str(angle)]
                idx = grp["imp_idx"][:]
                n = grp["n_atoms"][:]
                v_mag = grp["in_velo_mag"][:]
                v_z = grp["in_velo_z"][:]
                real_angle = np.degrees(np.arccos(np.clip(np.abs(v_z) / v_mag, -1, 1)))
                v_repr = _snap(float(v_mag.mean()), velocity_precision)
                a_repr = _snap(float(real_angle.mean()), angle_precision)
                parts.append(pd.DataFrame({
                    "velocity": np.full(len(n), v_repr, dtype=float),
                    "angle": np.full(len(n), a_repr, dtype=float),
                    "coverage": coverage_by_frame[idx % n_frames].astype(np.int64),
                    "n_atoms": n.astype(np.int64),
                }))
    return pd.concat(parts, ignore_index=True)
