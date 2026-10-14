# coverage_kernel

Coverage-dependent branching probabilities for atomic oxygen hitting a surface, as a gas-surface boundary condition for direct simulation Monte Carlo (DSMC), for example in PICLas.

For an incoming atom with velocity v (m/s) and polar angle θ (degrees from the surface normal), hitting a surface with coverage c (adsorbed atoms on the training patch), a model returns the probability of each outcome:

| class | outcome | coverage change |
|---|---|---|
| 0 | adsorption | +1 |
| 1 | O desorption (reflection) | 0 |
| 2 | O₂ desorption (recombination with an adsorbed atom) | −1 |

Python fits, stores and evaluates the models. Fortran evaluates them at run time without any Python dependency.

## Models

The package holds two model types. Both carry the same coverage physics (next section) and store to the same kind of HDF5 file.

- **Neural-network ensemble (`nn_ensemble`, the working model).** Each of eight feed-forward networks (three hidden layers of 64 neurons, SiLU activation) maps the standardized velocity, the standardized angle and the coverage fraction c/48 to one logit per outcome. The O₂ logit is shifted by ln(c/48), so O₂ desorption vanishes on an empty surface by construction. The model probability is the mean over the networks, and their spread is an uncertainty estimate. The paper uses the coverage-weighted ensemble: during training, every impact is weighted by share(c)^(−0.5), where share(c) is the fraction of impacts at its coverage. This lifts the rare high coverages, where the equilibrium at small incidence angles is decided. The weighting changes which impacts dominate the fit, not the probability being fitted.
- **Grid model (`grid`, comparison model).** One spline-logistic classifier of coverage per sampled (velocity, angle) condition, interpolated between the conditions with PCHIP (a shape-preserving cubic interpolation). Outside the sampled range, velocity is extrapolated linearly and angle is clipped.

**Why the weighted ensemble is the working model.** It is smooth in velocity and angle, it shares information between neighbouring conditions, and it comes with an uncertainty. Without the weighting, a network places the equilibrium at 0° and 20° about one O atom too high. The grid model follows the data closely at every sampled condition, but it also follows their noise and gives no uncertainty. The comparison of all candidates is in `analysis/branching_models/comparison_plots/final_model_comparison/README.md`.

## Coverage physics

- **No O₂ at zero coverage.** The O₂ desorption probability is exactly 0 at c = 0, and the other two are renormalized.
- **Blend beyond the sampled coverage.** The training data reach coverage 16. Each fitted model is used as is up to that highest sampled coverage. Beyond it, the probabilities blend linearly toward the endpoint without adsorption, where the adsorption probability is handed to O₂ desorption. The endpoint is reached at the ceiling of the condition. If the ceiling lies at or below 16, the switch is immediate above 16.
- **Self-consistent ceiling.** The ceiling of a condition is the mean plus 3 standard deviations of the coverage in that condition's own coverage walk (20000 impacts, last 8000), iterated until it is stable. The networks have one ceiling per sampled (velocity, angle), interpolated linearly in between. The grid model has one per cell.

The blend matters only in two cases: a condition whose equilibrium lies near 16 (normal incidence), or a surface element whose coverage was raised by other incidence conditions. A coverage walk at one fixed condition rarely leaves the sampled range.

## Install

```
cd python
pip install -e ".[test]" --config-settings editable_mode=compat
pytest
```

Optional extras are `train` (JAX, Flax and Optax, for network training) and `data` (pandas and znh5md, for reading the impact data). The `compat` editable mode puts `python/` on the import path. Without it, a Python session started in the repository root resolves `coverage_kernel` to the outer folder of the same name, and imports fail.

## Quickstart (Python)

```python
from coverage_kernel import load_model
from coverage_kernel.coverage_model import sample_outcome, run_coverage_walk, stationary_distribution

model = load_model("examples/data/models/branching_nn_ensemble_weighted.h5")   # or examples/data/models/grid_model.h5
p = model.probs(6.0, 7923.9, 40.0)               # (coverage, velocity in m/s, angle in degrees) -> 3 probabilities
spread = [m.probs(6.0, 7923.9, 40.0) for m in model.members]   # the eight networks

k, _ = sample_outcome(p, xi)                     # xi in [0, 1) from the caller's random-number stream
coverage += model.coverage_delta[k]

trace = run_coverage_walk(model.probs_fn(7923.9, 40.0), model.coverage_delta, n_steps=20000, seed=0)
```

`examples/python/coverage_walk.py` runs this end to end. It loads both paper models, prints probabilities with the network spread, and samples single impacts. For a velocity and angle sweep it computes the zero crossing of the net O flux, the exact long-run coverage average and a Monte Carlo walk.

## Training

- **Networks:** `coverage_kernel.nn_train.fit_coverage_nn_ensemble(velocity, angle, coverage, labels, physics=..., coverage_delta=..., gamma=0.5)`. It trains the ensemble (split seed and training seed equal to the member index, 150 epochs, no early stopping), records learning curves and computes the ceilings. The result goes to `export_hdf5.export_nn_model`. The paper's ensembles come from `analysis/branching_models/train_ensemble_nn.py`.
- **Grid model:** `coverage_kernel.coverage_model.fit_coverage_grid_model(...)`, then `export_hdf5.export_grid_model`. The paper's grid model comes from `analysis/branching_models/grid_model/train_grid_model.py`.
- **Impact data:** `coverage_kernel.data.load_trajectories` reads the impact files in `examples/data/` (`labelled_impacts_with_coverage/fragment_velocities_<velocity>.h5` and `starting_frames.h5`).

The file format is in `docs/hdf5_schema.md`.

## Fortran

### Building and testing

You need a Fortran 2008 compiler and HDF5 with Fortran bindings. On this machine, the conda environment `coverage_kernel_fortran` has both.

```
cd fortran
cmake -S . -B build && cmake --build build
python test/generate_reference_data.py          # synthetic models + the paper's grid model and weighted ensemble
cd test && ../build/test_coverage_kernel         # Fortran vs Python, tolerance 1e-9
```

`examples/fortran/monte_carlo_walk.f90` runs the coverage walk of the Python example through the public API. It writes `walk_sweep_fortran.csv`, and `python examples/python/coverage_walk.py --compare` checks it against the Python results.

### Public API (`MOD_CoverageKernel`)

```fortran
USE MOD_CoverageKernel
TYPE(t_CoverageModel) :: Model
CALL LoadCoverageModel("branching_nn_ensemble_weighted.h5", Model)   ! optional IErr: 1 missing file, 2 not HDF5

REAL(wp) :: Probs(Model%NClasses)
CALL EvalCoverageProbs(Model, Coverage, Velocity, Angle, Probs)

INTEGER  :: Outcome          ! 1-based
REAL(wp) :: DeltaCoverage    ! Model%CoverageDelta(Outcome)
CALL SampleCoverageOutcome(Model, Coverage, Velocity, Angle, Xi, Outcome, DeltaCoverage, Probs)

INTEGER :: LocalCoverage     ! adsorbed atoms on an element -> coverage of the 3.179 nm^2 training patch
LocalCoverage = MapAdsorbateCountToCoverage(Model, AdsorbateCount, ElementArea_nm2, Xi)
```

- **Argument order.** Every call takes (Coverage, Velocity, Angle), as in Python.
- **Random numbers.** The host code owns the random-number stream and the coverage of every surface element. The model is read-only after loading.
- **HDF5 state.** Loading calls `h5open_f` but never `h5close_f`, so a host that uses HDF5 itself keeps a working library.
- **Error handling.** A file that opens but does not match the schema stops with a message naming the failing item.
- **No executable stack.** No internal procedure is passed as an argument, so the build needs no executable stack.
- **Cost.** One evaluation of the eight-network ensemble costs about 70000 multiply-adds, about 18 µs on one core of this machine. A grid-model evaluation takes about 4 µs. If that is too slow for a DSMC run, tabulate the probabilities once per surface element and incidence condition at the integer coverages 0 to 48. The walk only visits whole coverages, so the table is exact.

### Module layout

```
MOD_CoverageKernel_Types       derived types (t_CoverageModel, t_NNModel, t_MLP, t_GridModel, t_SplineClassifier, ...)
MOD_CoverageKernel_BSpline     B-spline basis (grid scheme)
MOD_CoverageKernel_Interp1D    local PCHIP/linear interpolation, multilinear ceiling interpolation
MOD_CoverageKernel_Classifier  spline-logistic classifier (grid cells)
MOD_CoverageKernel_MLP         dense feed-forward network
MOD_CoverageKernel_Grid        local-stencil interpolation across grid cells
MOD_CoverageKernel_IO          HDF5 reader
MOD_CoverageKernel             coverage physics + public API (the only module a caller uses)
```

## Directory layout

```
python/coverage_kernel/   nn.py (network inference), nn_train.py (training), coverage_model.py (coverage physics,
                          walk, stationary distribution, crossing), grid_classifier.py + spline_classifier.py +
                          bspline.py + interp1d.py (grid scheme), export_hdf5.py (files), data.py (impact data)
python/tests/             pytest suite
fortran/                  Fortran port, CMake build, Fortran-vs-Python test
examples/                 coverage_walk.py (Python) and monte_carlo_walk.f90 (Fortran)
examples/data/            models/ (the paper's weighted NN ensemble and grid model), the impact data
                          (labelled_impacts_with_coverage/) and the starting structures (starting_frames.h5)
docs/hdf5_schema.md       file format
```

## Validity and limitations

- **Velocity and angle range.** The models are trained on 5000 to 9319 m/s and 0 to 80°. Below 5000 m/s the equilibrium coverage lies near or beyond the sampled coverage range, and the outgoing velocities depend on coverage, so the models do not apply there. Beyond 80°, the grid model clips the angle, while the networks extrapolate smoothly.
- **Single-fragment approximation.** Impacts with several desorbing fragments count each fragment as one impact, and impacts with three or more atoms in one fragment are left out.
- **High coverage.** Coverages of 15 and 16 rest on 6 and 2 starting structures, so the models are least certain there.

## Status

- **Python vs Fortran.** The Fortran port reproduces the Python reference to about 1e-15 for the paper's weighted ensemble, the paper's grid model and synthetic models.
- **Conversion.** The trained networks were converted from their original files without retraining, bit-identically (`analysis/branching_models/convert_nn_models.py`).
- **History.** The former pooled spline scheme, its examples and tests are in `old/coverage_kernel_before_cleanup_2026-10-07/`.
