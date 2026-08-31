"""
load_data.py
============
Reads the raw MLIP impact-trajectory HDF5 data (fragment_velocities_<v>.h5,
one per velocity, one HDF5 group per angle) plus the equilibrium starting-
frame coverages (starting_frames.h5) into a single tidy DataFrame with
columns: velocity, angle, coverage, n_atoms.

coverage_kernel's generic core and domain layer only ever see the tidy arrays this
produces (event_axis_values / event_features / event_labels).
"""

import h5py
import numpy as np
import pandas as pd
import znh5md


def load_starting_coverages(init_path, n_substrate_atoms):
    """O-atom coverage of each of the n_frames starting frames.

    n_substrate_atoms (required): atom count belonging to the substrate
    """
    frames = znh5md.IO(init_path)
    n_frames = len(frames)
    return np.array([len(frames[i].positions) - n_substrate_atoms for i in range(n_frames)]), n_frames


def load_trajectories(results_path, init_path, velocities, angles, *, n_substrate_atoms,
                       velocity_precision=0.01, angle_precision=0.01):
    """Read impact trajectories for every (velocity, angle) combination.

    Expects one file ``{results_path}/fragment_velocities_{velocity}.h5`` per
    velocity, with one group per angle containing ``imp_idx``, ``n_atoms``,
    ``in_velo_mag``, and ``in_velo_z`` datasets.

    Angle is the polar tilt from the surface normal (0 = incidence exactly
    along the normal, 90 = grazing/in-plane), folding every azimuthal
    direction and normal sign onto one [0,90] scale via abs(in_velo_z).

    Returns
    -------
    pandas.DataFrame with columns: velocity, angle, coverage, n_atoms.
    """
    coverage_by_frame, n_frames = load_starting_coverages(init_path, n_substrate_atoms)
    
    def _snap(x, precision):
        return round(round(x / precision) * precision, 10)

    records = []
    for velo in velocities:
        path = f"{results_path}{velo}.h5"
        with h5py.File(path, "r") as f:
            for angle in angles:
                grp = f[str(angle)]
                idx = grp["imp_idx"][:]
                n = grp["n_atoms"][:]
                v_mag = grp["in_velo_mag"][:]
                v_z = grp["in_velo_z"][:]
                real_angle = np.degrees(np.arccos(np.clip(np.abs(v_z) / v_mag, -1, 1)))
                v_mag_repr = _snap(float(v_mag.mean()), velocity_precision)
                angle_repr = _snap(float(real_angle.mean()), angle_precision)
                cov = coverage_by_frame[idx % n_frames]
                for i in range(len(n)):
                    records.append({
                        "velocity": v_mag_repr, "angle": angle_repr,
                        "coverage": int(cov[i]), "n_atoms": int(n[i]),
                    })
    return pd.DataFrame(records)
