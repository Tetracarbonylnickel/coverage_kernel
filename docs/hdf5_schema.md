# coverage_kernel HDF5 schema (format 1.1)

One file holds one fitted model. Python writes it (`coverage_kernel.export_hdf5`) and the Python and Fortran readers load it (`coverage_kernel.load_model`, `MOD_CoverageKernel.LoadCoverageModel`).

Every array's shape comes from the fitted model. Nothing here is a fixed constant, and the Fortran reader allocates from what it reads.

## Root (all schemes)

```
/format_version              string attr "1.1" (readers also accept "1.0" grid files, same layout)
/scheme                      string attr: "grid" | "nn_ensemble"
/n_classes                   int attr
/n_sites                     int attr   -- number of adsorption sites of the training patch
/zeroed_zero_index           int attr   -- class with probability 0 at coverage 0 (0-based)
/zeroed_full_index           int attr   -- class with probability 0 at coverage n_sites (0-based)
/absorbing_full_index        int attr   -- class taking over that probability (0-based)
/coverage_delta              (n_classes,) float  -- coverage change of each outcome
/model_area_nm2              float attr (optional) -- area of the training patch, for
                                          map_adsorbate_count_to_coverage / MapAdsorbateCountToCoverage
```

## scheme == "nn_ensemble"

Mean of the member probabilities. Each member maps (standardized velocity, standardized angle, coverage / n_sites) to one logit per class. The logit of `offset_class` is shifted by log(max(coverage / n_sites, offset_floor)) before the softmax. At coverage 0, `zeroed_zero_index` is set to 0 and the rest renormalized.

```
/nn/n_members                    int attr
/nn/combination                  string attr "mean"
/nn/coverage_max                 float attr -- highest sampled coverage (the blend starts here)
/nn/offset_class                 int attr (0-based)
/nn/offset_floor                 float attr (1e-8)
/nn/has_ceiling                  int attr 0/1
/nn/inputs                       (3,) string: "velocity", "angle", "coverage_fraction" (informational)
/nn/training_*                   attrs, informational (training settings), ignored by the readers
/nn/member_<i>/n_layers          int attr
/nn/member_<i>/input_mean        (2,) float -- mean of velocity (m/s) and angle (degrees)
/nn/member_<i>/input_std         (2,) float -- their standard deviation
/nn/member_<i>/layer_<j>/weights (n_in, n_out) float -- output = activation(input @ weights + biases)
/nn/member_<i>/layer_<j>/biases  (n_out,) float
/nn/member_<i>/layer_<j>/activation   string attr: "silu" | "tanh" | "relu" | "linear"
/nn/ceiling_axes/<d>/values      (n_axis_d,) float -- d = 0 velocity, 1 angle
/nn/ceiling_grid                 (n_axis_0, n_axis_1) float -- interpolated multilinearly, clipped to the axes
```

## scheme == "grid"

One spline classifier of coverage per (velocity, angle) cell. The coverage blend is applied per cell, then the cells are interpolated.

```
/terms/n_terms                int attr
/terms/kind                   (n_terms,) string: "spline" | "linear"
/terms/column                 (n_terms,) int      -- raw feature column (0 = coverage)
/terms/degree                 (n_terms,) int      -- spline terms only, -1 otherwise
/terms/n_knots                (n_terms,) int      -- spline terms only (fit parameter, not the knot-vector length)

/grid/n_axes                  int attr
/grid/axes/<d>/values          (n_axis_d,) float
/grid/axes/<d>/edge_policy      string attr: "clip" | "linear_extrapolate"
/grid/interp_method              string attr: "pchip" | "linear"
/grid/cells/coverage_max           (n_axis_0, ..., n_axis_{d-1}) float
/grid/cells/ceiling                 (n_axis_0, ..., n_axis_{d-1}) float, NaN = no ceiling
/grid/cells/scaler_mean               (..., n_features) float
/grid/cells/scaler_scale                (..., n_features) float
/grid/cells/n_present_classes             (...) int
/grid/cells/present_classes                 (..., n_classes) int, padded with -1
/grid/cells/coef                              (..., n_classes, n_features) float, first n_present rows valid
/grid/cells/intercept                           (..., n_classes) float
/grid/cells/term_knots/<i>                        (..., knot_vector_len_i) float, one per spline term
```

## Notes

- **Missing classes.** In the grid scheme, `present_classes` with `n_present_classes` marks the classes a cell observed during fitting. A missing class is absent from the softmax and has probability exactly 0, which a zero coefficient row would not give.
- **Knot vectors.** Per-cell knot vectors are not ragged. `n_knots` and `degree` are shared, so every cell's knot vector has length `n_knots + 2 * degree`; only the values differ.
- **Strings.** All strings the readers use are fixed-width byte strings (`S32`). A fixed-width HDF5 string maps directly onto `CHARACTER(LEN=32)` in Fortran.
- **Flags.** Flags such as `has_ceiling` are plain integers 0/1, not booleans.
- **Array order.** Every multi-dimensional dataset is written in C order, last axis fastest (NumPy's default). The Fortran reader reads each one as a flat 1-D buffer and indexes it in C order, never into a native multi-dimensional Fortran array, which would transpose it. For layer weights, the flat (n_in, n_out) buffer reshaped to the Fortran shape (n_out, n_in) is the matrix W in output = W input + b.
- **Version 1.1.** Version 1.1 added the `nn_ensemble` scheme and removed the former `pooled` scheme.
