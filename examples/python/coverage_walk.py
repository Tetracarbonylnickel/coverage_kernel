"""
coverage_walk.py
================
Uses the paper's branching models the way a coverage-tracking DSMC boundary
condition does, through coverage_kernel only:

1. load the coverage-weighted NN ensemble (the working model) and the grid model;
2. print the outcome probabilities at a few conditions, with the spread of the
   eight networks;
3. sample the outcome of single impacts with caller-supplied random numbers;
4. sweep velocity and angle: zero crossing of the net O flux, exact long-run
   coverage average (stationary distribution of the walk) and the average of a
   Monte Carlo coverage walk;
5. write walk_sweep.csv. examples/fortran/monte_carlo_walk.f90 computes the same
   sweep through the Fortran API; `python coverage_walk.py --compare` checks the
   two (the exact averages must agree to rounding, the walks only statistically,
   since the random-number streams differ).

Outcomes: 0 adsorption (coverage +1), 1 O desorption (0), 2 O2 desorption (-1).
Coverage in O atoms on the 48-site training patch (3.179 nm^2).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from coverage_kernel import load_model
from coverage_kernel.coverage_model import (net_flux_crossing, run_coverage_walk, sample_outcome,
                                            stationary_distribution)

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE.parent / "data" / "models"   # examples/data/models
MODELS = {"nn_ensemble": MODEL_DIR / "branching_nn_ensemble_weighted.h5",
          "grid": MODEL_DIR / "grid_model.h5"}
VELOCITIES = [5000.0, 7923.9, 9319.0]
ANGLES = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]
N_STEPS, N_TAIL, SEED = 20000, 8000, 0


def sweep(models):
    rows, seed = [], SEED
    for name, model in models.items():
        delta = model.coverage_delta
        for v in VELOCITIES:
            for a in ANGLES:
                seed += 1   # own random stream per condition, so walk noise is not shared between rows
                # a walk that starts at 0 only visits whole coverages: tabulate them once per
                # condition (exact), as a DSMC code would for a fixed incidence condition
                P = np.array([model.probs(c, v, a) for c in range(model.physics.n_sites + 1)])
                pi = stationary_distribution(P, delta)
                trace = run_coverage_walk(lambda c: P[int(c)], delta, N_STEPS, seed)
                crossing = net_flux_crossing(model.probs_fn(v, a), delta, c_max=20.0, n_points=801)
                rows.append(dict(model=name, velocity=v, angle=a, crossing=crossing,
                                 stationary_average=pi @ np.arange(len(pi)), walk_average=trace[-N_TAIL:].mean()))
    return pd.DataFrame(rows)


def compare(csv_python, csv_fortran):
    py, fo = pd.read_csv(csv_python), pd.read_csv(csv_fortran)
    m = py.merge(fo, on=["model", "velocity", "angle"], suffixes=("_python", "_fortran"))
    d_exact = (m.stationary_average_python - m.stationary_average_fortran).abs().max()
    d_walk = (m.walk_average_fortran - m.stationary_average_python).abs()
    print(f"{len(m)} conditions; exact long-run average, Python vs Fortran: largest difference {d_exact:.2e}")
    print(f"Fortran walk average vs exact average: median |difference| {d_walk.median():.2f}, largest {d_walk.max():.2f} "
          f"O atoms (walk noise, {N_STEPS} impacts)")
    return d_exact


if __name__ == "__main__":
    if "--compare" in sys.argv:
        ok = compare(HERE / "walk_sweep.csv", HERE.parent / "fortran" / "walk_sweep_fortran.csv") < 1e-9
        sys.exit(0 if ok else 1)

    models = {name: load_model(path) for name, path in MODELS.items()}
    nnm = models["nn_ensemble"]

    print("P(adsorption, O desorption, O2 desorption), NN ensemble [spread of the 8 networks]:")
    members = nnm.members
    for c, v, a in ((0.0, 7923.9, 0.0), (6.0, 7923.9, 40.0), (12.0, 7923.9, 0.0), (6.0, 5000.0, 80.0)):
        p = nnm.probs(c, v, a)
        spread = np.std([m.probs(c, v, a) for m in members], axis=0)
        grid_p = models["grid"].probs(c, v, a)
        print(f"  coverage {c:4.1f}, {v:6.0f} m/s, {a:4.1f} deg: {np.round(p, 4)} [{np.round(spread, 4)}]"
              f"   grid model {np.round(grid_p, 4)}")

    rng = np.random.default_rng(1)          # the host code owns the random numbers
    coverage = 6.0
    for _ in range(5):
        p = nnm.probs(coverage, 7923.9, 40.0)
        k, _ = sample_outcome(p, rng.random())
        coverage = max(0.0, coverage + nnm.coverage_delta[k])
        print(f"  impact: outcome {['adsorption', 'O desorption', 'O2 desorption'][k]:14s} -> coverage {coverage:.0f}")

    df = sweep(models)
    df.to_csv(HERE / "walk_sweep.csv", index=False)
    pd.set_option("display.width", 200)
    print(df.round(2).to_string(index=False))
    print("saved", HERE / "walk_sweep.csv")
