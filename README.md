# coverage_kernel

A conditioned-classifier kernel for coverage-dependent surface reaction
probabilities, with a Fortran inference port meant to be embedded as a DSMC
gas-surface boundary condition (e.g. in PICLas).

Given a surface coverage and any number of conditioning variables (e.g. an
incident particle's angle and velocity), the model returns
`P(outcome | coverage, ...)` for a fixed set of outcome classes (e.g.
adsorption/desorption channels) — either as probabilities alone, or as a
sampled outcome plus the probability vector it was drawn from, given a
caller-supplied uniform random deviate.

Nothing about a specific dataset is hardcoded: grid size, number of outcome
classes, `n_sites`, coverage range, spline knot count/degree, and the
number of conditioning dimensions are all read from the fitted-model file.
A differently-shaped dataset (a different supercell, a different
conditioning sweep, an extra outcome channel) is served by the same code
after a re-fit/re-export — no code changes.

Fitting happens in Python; evaluating a fitted model at run time happens in
Fortran with no Python dependency (or in Python directly, if that's all you
need).

## Install

```
cd python
pip install -e .[test]
pytest
```

The Fortran side is optional and only needed if you're embedding inference
into a Fortran host code — see [Fortran side](#fortran-side) below.

## Quickstart (Python)

This fits a toy pooled model end-to-end on synthetic data, queries it, and
round-trips it through the HDF5 format used for persistence/Fortran
interop. No external dataset needed — copy-paste and run.

```python
import numpy as np
from coverage_kernel.pooled_classifier import SplineTerm, LinearTerm
from coverage_kernel.coverage_model import CoveragePhysics, fit_coverage_pooled_model
from coverage_kernel.export_hdf5 import export_pooled_model, load_pooled_model

# --- toy dataset: 3 outcome classes (0=stick, 1=channel A, 2=channel B),
# coverage in [0, 20], one extra conditioning dimension ("energy") ---
rng = np.random.default_rng(0)
n = 2000
coverage = rng.integers(0, 21, n).astype(float)
energy = rng.uniform(0, 10, n)
p_stick = np.clip(1 - coverage / 20 - 0.05 * energy, 0.02, 0.95)
p_a = (1 - p_stick) * 0.6
p_b = 1 - p_stick - p_a
labels = np.array([rng.choice(3, p=[ps, pa, pb])
                    for ps, pa, pb in zip(p_stick, p_a, p_b)])


def feature_fn(coverage, energy):
    """Raw feature vector fed to the classifier -- you supply this; the
    library never assumes what your conditioning variables mean."""
    return np.array([coverage, energy])


X_raw = np.stack([feature_fn(c, e) for c, e in zip(coverage, energy)])

physics = CoveragePhysics(n_sites=20, zeroed_zero_index=1,
                           zeroed_full_index=0, absorbing_full_index=1)

model = fit_coverage_pooled_model(
    X_raw, labels, terms=[SplineTerm(0, n_knots=5, degree=2), LinearTerm(1)],
    feature_fn=feature_fn, n_classes=3, physics=physics, coverage_delta=[+1, 0, -1],
)

probs = model.probs(10.0, 5.0)   # P(outcome | coverage=10, energy=5)
print(probs)

# persist to the same HDF5 format the Fortran port reads
export_pooled_model("model.h5", model, coverage_delta=[+1, 0, -1])
loaded, coverage_delta, _ = load_pooled_model("model.h5", feature_fn=feature_fn)
assert np.allclose(loaded.probs(10.0, 5.0), probs)
```

`SplineTerm`/`LinearTerm` say how each raw feature column enters the model
(B-spline expansion vs. linear); `CoveragePhysics` encodes the physical
gates (an outcome that's impossible at zero or full coverage) applied on
top of the raw classifier fit. See [Architecture](#architecture) below for
what each piece does.

For fitting/evaluating a real dataset instead of the toy one above, see
`examples/python/`.

## Architecture

Both the Python and Fortran sides are split into two layers:

- **Generic core** — a conditioned multinomial-classifier evaluator that
  knows nothing about coverage, adsorption, or any other physical meaning:
  B-spline/linear feature terms, a present-classes-masked softmax
  classifier, and local-window 1-D/N-D interpolation across a sampled grid.
  Parametrized entirely by shapes read from the fitted model (number of
  classes, number of terms, grid axis lengths, knot-vector lengths) — never
  a compiled-in constant.
- **Domain layer** — the coverage-specific physics on top: the
  zero-at-zero-coverage / zero-at-full-coverage gates, the linear blend
  toward the physically-certain full-coverage endpoint, the self-consistent
  Monte-Carlo ceiling search (fit-time only), inverse-CDF outcome sampling,
  and the areal-density → training-patch-coverage stochastic rounding. This
  layer is still reusable for a *different* coverage dataset, just not for
  an unrelated physical quantity.

Two fitting schemes share the generic core:

- **pooled** — one classifier shared across every sampled condition, with
  a caller-supplied `feature_fn` engineering the raw conditioning variables
  into regression columns (see the Quickstart above, or
  `python/coverage_kernel/features.py` for this project's own
  coverage/normal-energy/parallel-velocity choice).
- **grid** — one independent classifier per grid cell (e.g. a
  (velocity, angle) grid), interpolated across the grid at query time using
  only the local interpolation stencil around the query (≤4 grid cells per
  axis for PCHIP, 2 for linear) — never the whole grid. This is what makes
  inference on a fine grid cheap: evaluating one query touches only the
  handful of cells actually needed, not every fitted cell.

Both PCHIP (shape-preserving monotone cubic Hermite, matching
`scipy.interpolate.PchipInterpolator`) and plain linear interpolation are
supported as configurable per-grid-model methods, resolved independently
per axis based on whether that axis's query is in its sampled range.

## Directory layout

```
python/coverage_kernel/       Fitting + evaluation reference implementation (NumPy/scikit-learn)
python/tests/                 pytest suite
examples/python/               train_models.ipynb fits both schemes on a real trajectory
                              dataset and exports the fitted models straight to HDF5;
                              monte_carlo_walk.ipynb loads them and runs/validates a
                              coverage random walk
examples/fortran/               Worked integration example: loads the exported
                              models and computes the same coverage-vs-
                              (velocity,angle) heatmap as the Python notebook,
                              entirely through MOD_CoverageKernel's public API --
                              a template for what a DSMC boundary condition's
                              own per-particle loop looks like
docs/hdf5_schema.md            The self-describing HDF5 exchange format between
                              Python fitting and Fortran inference
fortran/src/                   The Fortran inference port (generic core + domain
                              layer + HDF5 loader), no Python dependency at run time
fortran/test/                   Cross-validation harness: exports a small synthetic
                              model (and, if present, the real fitted models in
                              examples/python/models/) to HDF5, and checks that
                              the Fortran port reproduces the Python reference
                              implementation's numbers
fortran/CMakeLists.txt         Builds libcoverage_kernel + the cross-validation test
```

## Fitting and exporting a real model

`examples/python/train_models.ipynb` fits both schemes on a real trajectory
dataset (`examples/python/example_data/`) and saves them straight to HDF5 in
`examples/python/models/`; `examples/python/monte_carlo_walk.ipynb` loads
those, runs a coverage random walk over a wider conditioning sweep than was
sampled, and validates extrapolation against held-out data.

To export a fitted model yourself:

```python
from coverage_kernel.export_hdf5 import export_pooled_model, export_grid_model
export_pooled_model("model.h5", pooled_model, coverage_delta=[+1, 0, -1])
export_grid_model("model.h5", grid_model, coverage_delta=[+1, 0, -1])
```

See `docs/hdf5_schema.md` for the full file format.

## Fortran side

### Building

Requires a Fortran 2008 compiler and an HDF5 build with Fortran bindings
(`h5fc`/`hdf5_fortran`). CMake's `find_package(HDF5 COMPONENTS Fortran)`
locates it (this is why `project()` also enables C — conda-forge's HDF5
CMake config pulls in `FindThreads`, which needs a C or CXX language
enabled even though the library itself is pure Fortran).

```
cd fortran
mkdir build && cd build
cmake ..
cmake --build .
```

This builds `libcoverage_kernel.a` and the `test_coverage_kernel`
cross-validation executable. To regenerate the test's reference data (a
small synthetic model, plus the real fitted models in
`examples/python/models/` if present) and re-run the check:

```
python test/generate_reference_data.py   # from fortran/, needs the python/ package importable
cd test && ../build/test_coverage_kernel
```

### Worked example: `examples/fortran/`

After running `examples/python/train_models.ipynb` (which writes
`examples/python/models/{pooled,grid}_model.h5` directly), `examples/fortran/`
builds and runs the same way:

```
cd examples/fortran
mkdir build && cd build
cmake .. && cmake --build .
cd ..
./build/monte_carlo_walk
```

`monte_carlo_walk.f90` loads both models and, using only
`MOD_CoverageKernel`'s public API plus Fortran's own `RANDOM_NUMBER`, runs
a 20000-step coverage random walk at every point of the same 9×11
(angle, velocity) sweep `examples/python/monte_carlo_walk.ipynb` uses,
writing `heatmap_pooled.csv`/`heatmap_grid.csv`. This is meant to read as a
template for a DSMC boundary condition's own per-particle loop: the host
code owns the RNG stream and the per-surface-element coverage state, and
calls `SampleCoverageOutcome` once per event.

### Public API (`MOD_CoverageKernel`)

```fortran
USE MOD_CoverageKernel

TYPE(t_CoverageModel) :: Model
CALL LoadCoverageModel("model.h5", Model)          ! IErr optional; see below

REAL(wp) :: Probs(Model%NClasses)
CALL EvalCoverageProbs(Model, Coverage, Angle, Velocity, Probs)

INTEGER  :: Outcome
REAL(wp) :: DeltaCoverage
CALL SampleCoverageOutcome(Model, Coverage, Angle, Velocity, Xi, Outcome, DeltaCoverage, Probs)
!   Xi: caller-supplied uniform [0,1) deviate -- this module never owns an
!   RNG, since the host DSMC code keeps its own RNG stream.
!   Outcome is 1-based; DeltaCoverage = Model%CoverageDelta(Outcome).

INTEGER :: LocalCoverage
LocalCoverage = MapAreaDensityToCoverage(Model, AreaDensity, ElementArea, Xi)
!   Maps a surface element's areal adsorbate density onto the model's own
!   (typically smaller) training-patch coverage via stochastic rounding.
```

`LoadCoverageModel`'s `IErr` is `OPTIONAL, INTENT(OUT)`: if supplied,
non-zero means the file couldn't be opened (the one failure mode a caller
should expect at run time, e.g. a bad path); if omitted, that failure
`ERROR STOP`s with a descriptive message instead. A file that opens but
doesn't match the schema (wrong version, corrupted) is always fatal
regardless of `IErr` — that indicates a build/deployment mistake, not a
recoverable runtime condition.

Indexing: HDF5/Python is 0-based throughout (outcome codes, gate indices);
everything in `t_CoverageModel` and every Fortran API argument is 1-based.
The 0→1-based conversion happens exactly once, inside
`MOD_CoverageKernel_IO` at load time.

### Module layout

```
MOD_CoverageKernel_Types       Shared derived types (t_CoverageModel, t_PooledModel,
                               t_GridModel, t_GridCell, t_PooledClassifier, t_Term, ...)
MOD_CoverageKernel_BSpline     GENERIC: Cox-de Boor B-spline basis evaluation
MOD_CoverageKernel_Interp1D    GENERIC: local-window PCHIP/linear + N-D multilinear
MOD_CoverageKernel_Classifier  GENERIC: feature assembly + present-classes-masked softmax
MOD_CoverageKernel_Grid        GENERIC: recursive local-stencil tensor-product grid eval
MOD_CoverageKernel_IO          Reads the HDF5 schema into t_CoverageModel
MOD_CoverageKernel             DOMAIN layer + public API (the only module a caller needs)
```

Each generic-core module is validated independently against its Python
counterpart's behavior (see `fortran/test/`); `MOD_CoverageKernel` owns
this project's own coverage-specific gate/blend physics and its (coverage,
normal energy, parallel velocity) feature engineering for the pooled
scheme — those are the one place this port isn't dataset-agnostic, by
design (see each module's header comment for the generic/domain split
rationale). A different physical system's pooled-scheme feature
engineering would need its own small Fortran routine mirroring whatever
`feature_fn` it was fit with in Python.

## Status

The Fortran port reproduces the Python reference implementation to ~1e-9
(double-precision, cross-language floating-point-order noise) across both
schemes, in-range and out-of-range/extrapolated queries, the coverage==0
gate, the deep-blend region beyond the sampled coverage range, outcome
sampling, and density mapping — checked against both a small synthetic
model and a real model fitted on a ~400k-event dataset.

Not yet done: a hyperparameter search over spline knot count/degree,
regularization `C`, and the MC ceiling search's parameters (`k`, step
count) — current defaults haven't been re-tuned.
