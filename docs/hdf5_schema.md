# coverage_kernel HDF5 export schema

Every array's *shape* comes from the fitted model (n_classes, grid shape,
n_features, etc.) -- nothing here is a fixed constant; Fortran allocates
based on what it reads.

```
/format_version              string attr, e.g. "1.0"
/scheme                      string attr: "pooled" | "grid"
/n_classes                   int
/n_sites                     int
/zeroed_zero_index           int
/zeroed_full_index           int
/absorbing_full_index        int
/coverage_delta              (n_classes,) float
/model_area_nm2              float            (optional -- only if density-scaling is used)

# Term spec (shared across the whole pooled classifier, or across every grid
# cell -- fit_grid_classifier fits the *same* term structure per cell, just
# different fitted numbers per cell):
/terms/n_terms                int
/terms/kind                   (n_terms,) string: "spline" | "linear"
/terms/column                 (n_terms,) int      -- which raw feature column
/terms/degree                 (n_terms,) int      -- spline terms only
/terms/n_knots                (n_terms,) int      -- spline terms only (fit param, not raw knot-vector length)

# scheme == "pooled":
/pooled/coverage_max
/pooled/scaler_mean                (n_features,)
/pooled/scaler_scale                (n_features,)
/pooled/n_present_classes            int
/pooled/present_classes                (n_classes,) int   -- first n_present_classes entries valid, rest -1
/pooled/coef                            (n_classes, n_features) float -- first n_present_classes rows valid
/pooled/intercept                        (n_classes,) float
/pooled/term_knots/<i>                    (knot_vector_len_i,) float, one per spline term
/pooled/has_ceiling                        bool attr
/pooled/ceiling_axes/<d>/values              (n_axis_d,) float, one per conditioning dim (e.g. velocity, angle)
/pooled/ceiling_grid                           (n_axis_0, n_axis_1, ...) float

# scheme == "grid":
/grid/n_axes                  int
/grid/axes/<d>/values          (n_axis_d,) float
/grid/axes/<d>/edge_policy      string attr: "clip" | "linear_extrapolate"
/grid/interp_method              string attr: "pchip" | "linear"
/grid/cells/coverage_max           (n_axis_0, ..., n_axis_{d-1}) float
/grid/cells/ceiling                 (n_axis_0, ..., n_axis_{d-1}) float
/grid/cells/scaler_mean               (n_axis_0, ..., n_axis_{d-1}, n_features) float
/grid/cells/scaler_scale                (n_axis_0, ..., n_axis_{d-1}, n_features) float
/grid/cells/n_present_classes             (n_axis_0, ..., n_axis_{d-1}) int
/grid/cells/present_classes                 (n_axis_0, ..., n_axis_{d-1}, n_classes) int, padded with -1
/grid/cells/coef                              (n_axis_0, ..., n_axis_{d-1}, n_classes, n_features) float
/grid/cells/intercept                           (n_axis_0, ..., n_axis_{d-1}, n_classes) float
/grid/cells/term_knots/<i>                        (n_axis_0, ..., n_axis_{d-1}, knot_vector_len_i) float
```

Notes:
- `present_classes` padding with `-1` (fixed-size `n_classes` slot, `n_present_classes`
  says how many are valid) avoids HDF5 ragged-array complexity while staying
  fully general -- the common case (`n_present_classes == n_classes`) just
  doesn't use the padding.
- Per-cell knot vectors are NOT ragged across cells: `n_knots`/`degree` are
  fixed by the shared term spec, so every cell's fitted knot vector for a
  given spline term has the same length (`n_knots + 2*degree`) -- only the
  *values* differ (each cell's own coverage range). Dense array, no padding
  needed there.
- All strings (`format_version`, `scheme`, `terms/kind`, `interp_method`,
  `edge_policy`) are **fixed-width** byte strings (`numpy.dtype("S32")`),
  not h5py's variable-length `string_dtype`. A Fortran reader can map a
  fixed-width HDF5 string directly onto `H5T_FORTRAN_S1`/`CHARACTER(LEN=32)`;
  variable-length strings need C-interop machinery Fortran has no plain,
  portable API for. All of this schema's strings are short fixed-vocabulary
  tags, so a generous fixed width costs nothing.
- `has_ceiling` is a plain int attr (0/1), not a Python bool -- h5py maps
  Python `bool` attrs onto an HDF5 enum-of-int8 type, which a Fortran reader
  would need special handling for; a plain integer is unambiguous on both
  sides.
- Every array with more than one leading grid dimension (every
  `/grid/cells/*` and `/pooled/ceiling_grid` dataset) is written in
  C-order (last axis fastest, NumPy's default) and must be read by a
  Fortran client as a flat buffer indexed with a matching C-order flat-index
  formula -- **not** into a native multi-dimensional Fortran array. Fortran
  arrays are column-major, so a direct `h5dread_f` into a
  `REAL(wp) :: Buf(n0,n1,...)`-shaped Fortran array would silently
  transpose the data relative to what indexing `Buf(i0+1,i1+1,...)` should
  mean. The reference Fortran implementation
  (`fortran/src/MOD_CoverageKernel_IO.f90`) reads every dataset as a flat
  1-D buffer and indexes it with the same C-order convention used
  internally by `MOD_CoverageKernel_Interp1D`'s `FlatIndexCOrder` and
  `MOD_CoverageKernel_Grid`'s cell flattening.
