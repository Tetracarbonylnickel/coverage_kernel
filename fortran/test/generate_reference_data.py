"""
generate_reference_data.py
==========================
Writes the HDF5 models and the expected outputs that test_coverage_kernel.f90
checks the Fortran port against:

  data/grid_test.h5, data/nn_test.h5    small synthetic models (n_sites 16,
                                        coverage range 0 to 10, so the blend region
                                        and both blend branches are exercised)
  data/grid_real.h5, data/nn_real.h5    the grid model and the coverage-weighted NN
                                        ensemble of the paper (copied unchanged; the
                                        grid file is format 1.0, so the reader's
                                        backward compatibility is checked too)
  data/reference_*.txt                  "coverage velocity angle p_1 .. p_K" per line,
                                        sampling and adsorbate-count cases

Class convention throughout (as in the project): 0 adsorption (+1),
1 O desorption (0), 2 O2 desorption (-1).

    python test/generate_reference_data.py      # from fortran/, with coverage_kernel installed
"""

import shutil
from pathlib import Path

import numpy as np

from coverage_kernel import coverage_model as cm
from coverage_kernel import export_hdf5 as eh
from coverage_kernel import grid_classifier as gc
from coverage_kernel import nn
from coverage_kernel import spline_classifier as sc

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR = HERE.parents[1] / "examples" / "data" / "models"
REAL_GRID = MODEL_DIR / "grid_model.h5"
REAL_NN = MODEL_DIR / "branching_nn_ensemble_weighted.h5"

N_SITES = 16
PHYSICS = cm.CoveragePhysics(n_sites=N_SITES, zeroed_zero_index=2, zeroed_full_index=0, absorbing_full_index=2)
COVERAGE_DELTA = [1, 0, -1]
VELOCITIES = [2000.0, 5000.0, 9000.0]
ANGLES = [0.0, 40.0, 80.0]

QUERY_COVERAGES = [0.0, 1.0, 3.0, 6.0, 8.0, 9.5, 10.0, 11.0, 12.0, 16.0, 20.0]
QUERY_VELOCITIES = [1000.0, 2000.0, 4000.0, 5000.0, 7000.0, 9000.0, 10000.0]
QUERY_ANGLES = [0.0, 20.0, 40.0, 60.0, 80.0, 89.0]
REAL_QUERY_COVERAGES = [0.0, 1.0, 5.0, 10.0, 14.0, 16.0, 17.0, 20.0, 30.0, 48.0, 55.0]
REAL_QUERY_VELOCITIES = [1000.0, 5000.0, 6527.8, 7923.9, 9319.4, 11000.0]
REAL_QUERY_ANGLES = [0.0, 20.0, 40.0, 60.0, 80.0, 89.0]


def synth_labels(rng, coverage):
    logits = np.column_stack([0.30 * (N_SITES - coverage), 0.15 * coverage + 1.0, 0.25 * coverage])
    p = np.exp(logits)
    p[coverage == 0, 2] = 0
    p /= p.sum(axis=1, keepdims=True)
    return np.array([rng.choice(3, p=pi) for pi in p])


def build_grid_model():
    rng = np.random.default_rng(4321)
    n = 6000
    velocity, angle = rng.choice(VELOCITIES, n), rng.choice(ANGLES, n)
    coverage = rng.integers(0, 11, n).astype(float)
    labels = synth_labels(rng, coverage)
    axes = [gc.AxisSpec(np.array(VELOCITIES), edge_policy="linear_extrapolate"),
            gc.AxisSpec(np.array(ANGLES), edge_policy="clip")]
    return cm.fit_coverage_grid_model(axes, np.column_stack([velocity, angle]), coverage, labels,
                                      [sc.SplineTerm(0, n_knots=5, degree=2)], n_classes=3, physics=PHYSICS,
                                      coverage_delta=COVERAGE_DELTA, mc_steps=4000, mc_tail=1000, mc_seed=7,
                                      method="pchip", model_area_nm2=3.5)


def build_nn_model():
    """Two random networks of different depth, standardization per member, ceiling grid
    with values on both sides of coverage_max (10)."""
    rng = np.random.default_rng(99)
    members = []
    for hidden, act in (((6, 5), "silu"), ((7,), "tanh")):
        sizes = [3, *hidden, 3]
        layers = [(rng.normal(scale=0.8, size=(i, o)), rng.normal(scale=0.5, size=o),
                   act if k < len(sizes) - 2 else "linear") for k, (i, o) in enumerate(zip(sizes[:-1], sizes[1:]))]
        members.append(nn.MLP(layers, [6000.0 + rng.normal(scale=50), 40.0 + rng.normal()],
                              [2000.0 + rng.normal(scale=10), 30.0 + rng.normal()]))
    ens = nn.NNEnsemble(members, N_SITES, offset_class=2, zeroed_zero_index=PHYSICS.zeroed_zero_index)
    ceiling = np.array([[14.0, 12.5, 9.0], [13.0, 11.0, 8.0], [15.0, 10.5, 7.5]])
    return nn.NNEnsembleCoverageModel(ens, PHYSICS, 10.0, [VELOCITIES, ANGLES], ceiling,
                                      coverage_delta=COVERAGE_DELTA, model_area_nm2=3.5, name="synthetic")


def dump_probs(model, name, coverages, velocities, angles):
    lines = [f"{c:.6f} {v:.6f} {a:.6f} " + " ".join(f"{p:.15e}" for p in model.probs(c, v, a))
             for c in coverages for v in velocities for a in angles]
    (DATA_DIR / f"reference_{name}.txt").write_text("\n".join(lines) + "\n")
    print(f"reference_{name}.txt: {len(lines)} cases")


def dump_sampling(model):
    """Inverse-CDF cases just below / above the cumulative boundaries (exact boundaries
    would depend on the last bit of the probabilities)."""
    lines = []
    for c, v, a in ((6.0, 5000.0, 40.0), (0.0, 2000.0, 0.0), (12.0, 9000.0, 80.0), (20.0, 7000.0, 20.0)):
        p = model.probs(c, v, a)
        cum = np.cumsum(p)
        for xi in (0.0, cum[0] - 1e-7, cum[0] + 1e-7, cum[1] - 1e-7, min(cum[1] + 1e-7, 0.9999999), 0.9999999):
            idx, _ = cm.sample_outcome(p, xi)
            lines.append(f"{c:.6f} {v:.6f} {a:.6f} {xi:.10f} {idx} {COVERAGE_DELTA[idx]:.6f}")
    (DATA_DIR / "reference_sampling.txt").write_text("\n".join(lines) + "\n")


def dump_count(model):
    lines = []
    for n_ads, area, xi in ((4.2, 2.0, 0.99), (4.2, 2.0, 0.0), (1.25, 5.0, 0.5), (10.0, 3.5, 0.3)):
        k = cm.map_adsorbate_count_to_coverage(n_ads, area, model.model_area_nm2, xi)
        lines.append(f"{n_ads:.6f} {area:.6f} {xi:.6f} {k}")
    (DATA_DIR / "reference_count.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    grid = build_grid_model()
    eh.export_grid_model(str(DATA_DIR / "grid_test.h5"), grid)
    dump_probs(eh.load_grid_model(str(DATA_DIR / "grid_test.h5")), "grid_test", QUERY_COVERAGES, QUERY_VELOCITIES,
               QUERY_ANGLES)

    nnm = build_nn_model()
    eh.export_nn_model(str(DATA_DIR / "nn_test.h5"), nnm)
    nn_loaded = eh.load_nn_model(str(DATA_DIR / "nn_test.h5"))
    dump_probs(nn_loaded, "nn_test", QUERY_COVERAGES, QUERY_VELOCITIES, QUERY_ANGLES)
    dump_sampling(nn_loaded)
    dump_count(nn_loaded)

    for src, dst, name in ((REAL_GRID, "grid_real.h5", "grid_real"), (REAL_NN, "nn_real.h5", "nn_real")):
        if not src.exists():
            print(f"skipping {name}: {src} not found")
            continue
        shutil.copyfile(src, DATA_DIR / dst)
        dump_probs(eh.load_model(str(DATA_DIR / dst)), name, REAL_QUERY_COVERAGES, REAL_QUERY_VELOCITIES,
                   REAL_QUERY_ANGLES)
    print("reference data written to", DATA_DIR)
