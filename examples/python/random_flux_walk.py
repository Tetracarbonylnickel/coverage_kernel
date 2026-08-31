"""
random_flux_walk.py
====================
Coverage random walk under a *realistic* incidence flux rather than a fixed
(velocity, angle) condition: at every one of the walk's steps, a fresh
(angle, velocity) pair is redrawn from a physically motivated distribution
and fed to the fitted coverage_kernel models (pooled + grid), instead of
holding (velocity, angle) fixed as monte_carlo_walk.ipynb does.

Velocity ~ N(velo_mean, velo_sigma^2), unclipped: represents the spread of
impact speeds a real surface sees (spacecraft ram velocity plus the thermal
spread of the ambient AO), not a single fixed value.

Angle: sampled *isotropically over solid angle* on [0, angle_hi] -- i.e.
cos(angle) ~ Uniform(cos(angle_hi), 1). This is the same "solid-angle 
weight reproducing isotropic
incidence" convention already used in predicted_coverage.py (weight(alpha)
∝ cos(lo) - cos(hi)) and referenced in the main text's net-flux zero-
crossing analysis, applied here as a continuous sampling law instead of a
discrete per-angle weight.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

OUT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT_DIR.parent.parent / "python"))
sys.path.insert(0, str(OUT_DIR))

from coverage_kernel.coverage_model import sample_outcome  # noqa: E402
from coverage_kernel.export_hdf5 import load_pooled_model, load_grid_model  # noqa: E402
from coverage_kernel.features import physics_features  # noqa: E402
from load_data import load_starting_coverages  # noqa: E402

MODELS_DIR = OUT_DIR / "models"
FRAMES_PATH = str(OUT_DIR / "example_data" / "starting_frames.h5")
N_SUBSTRATE_ATOMS = 480

SCHEMES = ["pooled", "grid"]
COLORS = {"pooled": "#2a78d6", "grid": "#008300"}

IDX_ADS, IDX_O_DES, IDX_O2_DES = 0, 1, 2
COVERAGE_DELTA = [+1, 0, -1]

ANGLE_LO, ANGLE_HI = 0.0, 85.0   # real MD setup; 80 is the sampled grid's own max

VELO_MEAN = 8000.0
VELO_SIGMA = (12000.0 - 8000.0) / 3.0

N_STEPS, N_TAIL, SEED = 20_000, 8_000, 0


def sample_angle_isotropic(rng, angle_hi_deg):
    """Polar incidence angle sampled isotropically over solid angle on
    [0, angle_hi_deg]: cos(angle) ~ Uniform(cos(angle_hi), 1), i.e. density
    ∝ sin(angle) -- matches predicted_coverage.py's solid-angle weight
    w(alpha) ∝ cos(lo) - cos(hi), applied here as a continuous law rather
    than a discrete per-angle weight.
    """
    cos_hi = np.cos(np.radians(angle_hi_deg))
    cos_theta = rng.uniform(cos_hi, 1.0)
    return np.degrees(np.arccos(cos_theta))


def sample_condition(rng, velo_mean, velo_sigma):
    """One step's (angle, velocity): angle isotropic over solid angle (see
    sample_angle_isotropic), velocity ~ N(velo_mean, velo_sigma^2),
    deliberately unclipped (mean/sigma keep it effectively positive)."""
    angle = sample_angle_isotropic(rng, ANGLE_HI)
    velocity = rng.normal(velo_mean, velo_sigma)
    return angle, velocity


def run_random_flux_walk(probs_fn, n_steps, seed, velo_mean, velo_sigma, start_coverage=0.0):
    """Like coverage_kernel.coverage_model.run_coverage_walk, but
    (angle, velocity) is redrawn every step from sample_condition() instead
    of held fixed. probs_fn(coverage, angle, velocity) -> probs.
    """
    rng = np.random.default_rng(seed)
    coverage_delta = np.asarray(COVERAGE_DELTA, dtype=float)
    coverage = float(start_coverage)
    trace = np.empty(n_steps + 1)
    angles = np.empty(n_steps)
    velocities = np.empty(n_steps)
    trace[0] = coverage
    for t in range(n_steps):
        angle, velocity = sample_condition(rng, velo_mean, velo_sigma)
        probs = probs_fn(coverage, angle, velocity)
        idx, _ = sample_outcome(probs, rng.random())
        coverage = max(coverage + coverage_delta[idx], 0.0)
        trace[t + 1] = coverage
        angles[t] = angle
        velocities[t] = velocity
    return trace, angles, velocities


def make_diagnostic_figure(eq_mean, traces, velocities, angles):
    fig = plt.figure(figsize=(14, 8.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.3, 1], hspace=0.32, wspace=0.28)

    ax_trace = fig.add_subplot(gs[0, :])
    steps = np.arange(N_STEPS + 1)
    ax_trace.axvspan(N_STEPS - N_TAIL, N_STEPS, color="#e1e0d9", alpha=0.5, zorder=0,
                      label=f"tail window (last {N_TAIL:,})")
    for s in SCHEMES:
        ax_trace.plot(steps, traces[s], color=COLORS[s], linewidth=1.0, alpha=0.85, label=s)
    ax_trace.axhline(eq_mean, color="#d03b3b", linewidth=1.6, linestyle="--",
                      label=f"real MD equilibrium ({eq_mean:.2f})")
    ax_trace.set_xlabel("Step")
    ax_trace.set_ylabel("Coverage (O atoms)")
    ax_trace.set_title(
        f"Random-flux coverage trace -- angle ~ isotropic solid-angle(0-{ANGLE_HI:.0f} deg), "
        f"velocity ~ N({VELO_MEAN:.0f}, {VELO_SIGMA:.1f}²) m/s",
        fontsize=12, fontweight="bold")
    ax_trace.legend(fontsize=9, loc="upper left", ncol=3)
    ax_trace.grid(linestyle="--", alpha=0.3)

    ax_hist = fig.add_subplot(gs[1, 0])
    bins = np.linspace(0, max(traces[s][-N_TAIL:].max() for s in SCHEMES) + 1, 30)
    for s in SCHEMES:
        ax_hist.hist(traces[s][-N_TAIL:], bins=bins, color=COLORS[s], alpha=0.5,
                     density=True, label=s)
    ax_hist.axvline(eq_mean, color="#d03b3b", linewidth=1.6, linestyle="--")
    ax_hist.set_xlabel("Tail coverage (O atoms)")
    ax_hist.set_ylabel("Density")
    ax_hist.set_title("Tail coverage distribution", fontsize=11, fontweight="bold")
    ax_hist.legend(fontsize=8)
    ax_hist.grid(linestyle="--", alpha=0.3)

    ax_velo = fig.add_subplot(gs[1, 1])
    ax_velo.hist(velocities, bins=40, color="#6da7ec", edgecolor="white", linewidth=0.3)
    ax_velo.axvline(9319.4, color="#0b0b0b", linewidth=1.2, linestyle=":", label="sampled max (9319)")
    ax_velo.axvline(VELO_MEAN, color="#52514e", linewidth=1.0, linestyle="--", label=f"mean ({VELO_MEAN:.0f})")
    ax_velo.set_xlabel("Sampled velocity (m/s)")
    ax_velo.set_ylabel("Count")
    ax_velo.set_title("Drawn velocity distribution", fontsize=11, fontweight="bold")
    ax_velo.legend(fontsize=8)
    ax_velo.grid(linestyle="--", alpha=0.3)

    ax_angle = fig.add_subplot(gs[1, 2])
    ax_angle.hist(angles, bins=40, color="#eda100", edgecolor="white", linewidth=0.3)
    ax_angle.set_xlabel("Sampled angle (deg)")
    ax_angle.set_ylabel("Count")
    ax_angle.set_title("Drawn angle distribution (isotropic)", fontsize=11, fontweight="bold")
    ax_angle.grid(linestyle="--", alpha=0.3)

    fig.suptitle(
        "Random-flux Monte-Carlo walk (coverage_kernel pooled/grid) vs. real MD equilibrium",
        fontsize=13.5, fontweight="bold", y=1.01)

    pdf_path = OUT_DIR / "random_flux_walk.pdf"
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return pdf_path


def main():
    print("Computing real-MD equilibrium coverage reference...")
    coverage_by_frame, _ = load_starting_coverages(FRAMES_PATH, N_SUBSTRATE_ATOMS)
    eq_mean = coverage_by_frame.mean()
    print(f"  Real MD equilibrium coverage: {eq_mean:.2f} O atoms\n")

    print("Loading fitted models...")
    pooled_model, _, _ = load_pooled_model(str(MODELS_DIR / "pooled_model.h5")) # does not have to be called if default is used, physics_features)
    grid_model, _, _ = load_grid_model(str(MODELS_DIR / "grid_model.h5"))

    probs_fns = {
        "pooled": lambda c, a, v: pooled_model.probs(c, v, a),
        "grid": lambda c, a, v: grid_model.eval(np.array([c]), [v, a]),
    }

    print(f"Running {len(SCHEMES)} random-flux walks ({N_STEPS:,} steps each)...\n")
    traces, angles_by, velos_by = {}, {}, {}
    for s in SCHEMES:
        trace, angles, velocities = run_random_flux_walk(
            probs_fns[s], N_STEPS, SEED, VELO_MEAN, VELO_SIGMA)
        traces[s] = trace
        angles_by[s] = angles
        velos_by[s] = velocities

    print(f"=== Tail coverage summary (last {N_TAIL:,} of {N_STEPS:,} steps) ===")
    for s in SCHEMES:
        tail = traces[s][-N_TAIL:]
        print(f"  {s:8s} mean={tail.mean():6.2f}  std={tail.std():5.2f}"
              f"   diff from real MD ({eq_mean:.2f}): {tail.mean() - eq_mean:+.2f}")

    velocities = velos_by[SCHEMES[0]]
    angles = angles_by[SCHEMES[0]]
    frac_above_9319 = (velocities > 9319.4).mean()
    frac_angle_extrap = ((angles > 80) & (angles <= 85)).mean()
    print(f"\n  velocity samples > 9319 (sampled max, extrapolated): {frac_above_9319:.1%}")
    print(f"  angle samples in (80, 85] (extrapolated for pooled, clipped to 80 for grid): "
          f"{frac_angle_extrap:.1%}")

    pdf = make_diagnostic_figure(eq_mean, traces, velocities, angles)
    print(f"\nSaved {pdf}")
    print("DONE")


if __name__ == "__main__":
    main()
