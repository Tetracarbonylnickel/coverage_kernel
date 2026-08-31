"""
export_hdf5.py
===============
Writes a fitted pooled/grid coverage model to a single self-describing HDF5
file the Fortran inference module can load, and reads it back for
round-trip validation (also useful as a plain Python loader on its own).

Every array's shape comes from the fitted model (n_classes, grid shape,
number of features, etc.). See docs/hdf5_schema.md for the full layout.

One subtlety this format has to get right: a PooledClassifier only carries
coefficients for classes actually observed during its fit
(`present_classes`) -- a class missing from training isn't a zero
coefficient, it's *absent from the softmax entirely* and hard-set to
probability 0 afterward (see pooled_classifier.py). Padding coef/intercept
out to a dense (n_classes, n_features) array with zero rows for the
missing classes would silently change that behavior (a zero-coefficient
row still gets a non-zero softmax share), so `present_classes` and
`n_present_classes` are stored explicitly and Fortran must replicate the
same "only softmax over present_classes, then place into the full vector"
logic -- not just read the padded coefficients naively.

feature_fn (the pooled scheme's raw-feature engineering, e.g. this
project's coverage/E_n/v_parallel decomposition) is an arbitrary Python
callable and can't be serialized -- it's the caller's responsibility to
supply the same one when reloading a pooled model (see load_pooled_model,
which defaults to this project's own physics_features since that's the
only one this codebase actually fits models with; override it if loading
a model fit for a different feature engineering). The grid scheme has no
such function (its per-cell feature is just coverage directly), so this
only matters for the pooled scheme.
"""

import itertools

import h5py
import numpy as np

from .pooled_classifier import SplineTerm, LinearTerm, PooledClassifier
from .coverage_model import CoveragePhysics, CoverageCell, PooledCoverageModel
from .grid_classifier import AxisSpec, GridClassifier
from .features import physics_features

FORMAT_VERSION = "1.0"

# Fixed-width byte strings throughout (rather than h5py's variable-length
# string_dtype) -- a Fortran reader can map these directly onto an
# H5T_FORTRAN_S1 fixed-length string type; variable-length HDF5 strings need
# C-interop machinery Fortran doesn't have a plain, portable API for. Every
# string this schema stores (term kind, scheme, edge_policy, interp_method,
# format_version) is a short fixed vocabulary tag, so a generous fixed width
# costs nothing.
STR_DTYPE = "S32"


def _fixed_str_array(values):
    return np.array(values, dtype=STR_DTYPE)


def _decode_fixed_str(value):
    """h5py returns fixed-width HDF5 strings as numpy.bytes_ scalars (or
    plain bytes); normalize to a plain (unpadded) Python str either way."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


# --------------------------------------------------------------------------- #
# shared helpers                                                              #
# --------------------------------------------------------------------------- #

def _write_term_spec(group, terms):
    kind = _fixed_str_array(["spline" if isinstance(t, SplineTerm) else "linear" for t in terms])
    column = np.array([t.column for t in terms], dtype=np.int64)
    degree = np.array([t.degree if isinstance(t, SplineTerm) else -1 for t in terms], dtype=np.int64)
    n_knots = np.array([t.n_knots if isinstance(t, SplineTerm) else -1 for t in terms], dtype=np.int64)
    group.attrs["n_terms"] = len(terms)
    group.create_dataset("kind", data=kind)
    group.create_dataset("column", data=column)
    group.create_dataset("degree", data=degree)
    group.create_dataset("n_knots", data=n_knots)


def _read_term_spec(group):
    n_terms = int(group.attrs["n_terms"])
    kind = [k.decode() if isinstance(k, bytes) else k for k in group["kind"][:]]
    column = group["column"][:]
    degree = group["degree"][:]
    n_knots = group["n_knots"][:]
    terms = []
    for i in range(n_terms):
        if kind[i] == "spline":
            terms.append(SplineTerm(int(column[i]), n_knots=int(n_knots[i]), degree=int(degree[i])))
        else:
            terms.append(LinearTerm(int(column[i])))
    return terms


def _padded_present(clf: PooledClassifier, n_classes):
    """(n_present, present_padded[n_classes], coef_padded[n_classes,n_features],
    intercept_padded[n_classes]) -- see module docstring for why padding
    alone (without n_present/present_classes) would be wrong."""
    n_present = len(clf.present_classes)
    present_padded = np.full(n_classes, -1, dtype=np.int64)
    present_padded[:n_present] = clf.present_classes
    n_features = clf.coef.shape[1]
    coef_padded = np.zeros((n_classes, n_features))
    coef_padded[:n_present] = clf.coef
    intercept_padded = np.zeros(n_classes)
    intercept_padded[:n_present] = clf.intercept
    return n_present, present_padded, coef_padded, intercept_padded


def _classifier_from_padded(terms, knot_vectors, scaler_mean, scaler_scale,
                             n_present, present_padded, coef_padded, intercept_padded, n_classes):
    present_classes = present_padded[:n_present]
    coef = coef_padded[:n_present]
    intercept = intercept_padded[:n_present]
    return PooledClassifier(terms, knot_vectors, scaler_mean, scaler_scale,
                             coef, intercept, present_classes, n_classes)


def _write_physics_header(f, physics: CoveragePhysics, coverage_delta, model_area_nm2):
    """Writes everything except n_classes (a property of the classifier,
    not of CoveragePhysics) and scheme -- callers set those themselves."""
    f.attrs["format_version"] = _fixed_str_array(FORMAT_VERSION)
    f.attrs["n_sites"] = physics.n_sites
    f.attrs["zeroed_zero_index"] = physics.zeroed_zero_index
    f.attrs["zeroed_full_index"] = physics.zeroed_full_index
    f.attrs["absorbing_full_index"] = physics.absorbing_full_index
    f.create_dataset("coverage_delta", data=np.asarray(coverage_delta, dtype=np.float64))
    if model_area_nm2 is not None:
        f.attrs["model_area_nm2"] = float(model_area_nm2)


def _read_physics_header(f):
    physics = CoveragePhysics(
        n_sites=int(f.attrs["n_sites"]),
        zeroed_zero_index=int(f.attrs["zeroed_zero_index"]),
        zeroed_full_index=int(f.attrs["zeroed_full_index"]),
        absorbing_full_index=int(f.attrs["absorbing_full_index"]),
    )
    coverage_delta = f["coverage_delta"][:]
    model_area_nm2 = float(f.attrs["model_area_nm2"]) if "model_area_nm2" in f.attrs else None
    return physics, coverage_delta, model_area_nm2


# --------------------------------------------------------------------------- #
# pooled scheme                                                                #
# --------------------------------------------------------------------------- #

def export_pooled_model(path, model: PooledCoverageModel, coverage_delta, model_area_nm2=None):
    clf = model.pooled_classifier
    n_classes = clf.n_classes

    with h5py.File(path, "w") as f:
        f.attrs["scheme"] = _fixed_str_array("pooled")
        _write_physics_header(f, model.physics, coverage_delta, model_area_nm2)
        f.attrs["n_classes"] = n_classes

        _write_term_spec(f.create_group("terms"), clf.terms)

        g = f.create_group("pooled")
        g.attrs["coverage_max"] = float(model.coverage_max)
        g.create_dataset("scaler_mean", data=clf.scaler_mean)
        g.create_dataset("scaler_scale", data=clf.scaler_scale)

        n_present, present_padded, coef_padded, intercept_padded = _padded_present(clf, n_classes)
        g.attrs["n_present_classes"] = n_present
        g.create_dataset("present_classes", data=present_padded)
        g.create_dataset("coef", data=coef_padded)
        g.create_dataset("intercept", data=intercept_padded)

        tk = g.create_group("term_knots")
        for i, knots in enumerate(clf.knot_vectors):
            if knots is not None:
                tk.create_dataset(str(i), data=knots)

        has_ceiling = model.ceiling_grid is not None
        # plain int (0/1), not a Python bool -- h5py maps bool attrs onto an
        # HDF5 enum-of-int8 type, which a Fortran reader would need special
        # handling for; a plain integer is unambiguous on both sides.
        g.attrs["has_ceiling"] = int(has_ceiling)
        if has_ceiling:
            ca = g.create_group("ceiling_axes")
            for d, axis_vals in enumerate(model.ceiling_axes):
                ca.create_group(str(d)).create_dataset("values", data=np.asarray(axis_vals, dtype=np.float64))
            g.create_dataset("ceiling_grid", data=model.ceiling_grid)


def load_pooled_model(path, feature_fn=physics_features):
    """feature_fn: the same raw-feature-engineering callable the model was
    fit with (see module docstring -- not something HDF5 can store).
    Defaults to this project's own physics_features (coverage/E_n/v_parallel),
    the only one actually in use anywhere in this codebase -- pass a
    different one explicitly if loading a model fit for a different
    physical system's feature engineering."""
    with h5py.File(path, "r") as f:
        assert _decode_fixed_str(f.attrs["scheme"]) == "pooled"
        physics, coverage_delta, model_area_nm2 = _read_physics_header(f)
        n_classes = int(f.attrs["n_classes"])
        terms = _read_term_spec(f["terms"])

        g = f["pooled"]
        coverage_max = float(g.attrs["coverage_max"])
        scaler_mean = g["scaler_mean"][:]
        scaler_scale = g["scaler_scale"][:]
        n_present = int(g.attrs["n_present_classes"])
        present_padded = g["present_classes"][:]
        coef_padded = g["coef"][:]
        intercept_padded = g["intercept"][:]

        knot_vectors = []
        tk = g["term_knots"]
        for i, term in enumerate(terms):
            knot_vectors.append(tk[str(i)][:] if isinstance(term, SplineTerm) else None)

        clf = _classifier_from_padded(terms, knot_vectors, scaler_mean, scaler_scale,
                                       n_present, present_padded, coef_padded, intercept_padded, n_classes)

        ceiling_axes, ceiling_grid = None, None
        if bool(g.attrs["has_ceiling"]):
            ca = g["ceiling_axes"]
            ceiling_axes = [ca[str(d)]["values"][:] for d in range(len(ca))]
            ceiling_grid = g["ceiling_grid"][:]

    model = PooledCoverageModel(clf, feature_fn, physics, coverage_max,
                                 ceiling_axes=ceiling_axes, ceiling_grid=ceiling_grid)
    return model, coverage_delta, model_area_nm2


# --------------------------------------------------------------------------- #
# grid scheme                                                                  #
# --------------------------------------------------------------------------- #

def export_grid_model(path, model: GridClassifier, coverage_delta, model_area_nm2=None):
    sample_cell = next(iter(model.cell_classifiers.values()))
    physics = sample_cell.physics
    raw_clf0 = sample_cell.raw_classifier
    terms = raw_clf0.terms
    n_classes = model.n_classes
    n_features = raw_clf0.coef.shape[1]
    axes_shape = tuple(len(a.values) for a in model.axes)

    coverage_max_arr = np.zeros(axes_shape)
    ceiling_arr = np.full(axes_shape, np.nan)
    scaler_mean_arr = np.zeros(axes_shape + (n_features,))
    scaler_scale_arr = np.zeros(axes_shape + (n_features,))
    n_present_arr = np.zeros(axes_shape, dtype=np.int64)
    present_arr = np.full(axes_shape + (n_classes,), -1, dtype=np.int64)
    coef_arr = np.zeros(axes_shape + (n_classes, n_features))
    intercept_arr = np.zeros(axes_shape + (n_classes,))
    term_knot_arrs = {
        i: np.zeros(axes_shape + (len(knots),))
        for i, knots in enumerate(raw_clf0.knot_vectors) if knots is not None
    }

    for idx in itertools.product(*(range(n) for n in axes_shape)):
        cell = model.cell_classifiers[idx]
        raw_clf = cell.raw_classifier
        coverage_max_arr[idx] = cell.coverage_max
        if cell.ceiling is not None:
            ceiling_arr[idx] = cell.ceiling
        scaler_mean_arr[idx] = raw_clf.scaler_mean
        scaler_scale_arr[idx] = raw_clf.scaler_scale
        n_present, present_padded, coef_padded, intercept_padded = _padded_present(raw_clf, n_classes)
        n_present_arr[idx] = n_present
        present_arr[idx] = present_padded
        coef_arr[idx] = coef_padded
        intercept_arr[idx] = intercept_padded
        for i, knots in enumerate(raw_clf.knot_vectors):
            if knots is not None:
                term_knot_arrs[i][idx] = knots

    with h5py.File(path, "w") as f:
        f.attrs["scheme"] = _fixed_str_array("grid")
        _write_physics_header(f, physics, coverage_delta, model_area_nm2)
        f.attrs["n_classes"] = n_classes

        _write_term_spec(f.create_group("terms"), terms)

        g = f.create_group("grid")
        g.attrs["n_axes"] = len(model.axes)
        g.attrs["interp_method"] = _fixed_str_array(model.method)
        axg = g.create_group("axes")
        for d, axis in enumerate(model.axes):
            sub = axg.create_group(str(d))
            sub.create_dataset("values", data=np.asarray(axis.values, dtype=np.float64))
            sub.attrs["edge_policy"] = _fixed_str_array(axis.edge_policy)

        cg = g.create_group("cells")
        cg.create_dataset("coverage_max", data=coverage_max_arr)
        cg.create_dataset("ceiling", data=ceiling_arr)
        cg.create_dataset("scaler_mean", data=scaler_mean_arr)
        cg.create_dataset("scaler_scale", data=scaler_scale_arr)
        cg.create_dataset("n_present_classes", data=n_present_arr)
        cg.create_dataset("present_classes", data=present_arr)
        cg.create_dataset("coef", data=coef_arr)
        cg.create_dataset("intercept", data=intercept_arr)
        tk = cg.create_group("term_knots")
        for i, arr in term_knot_arrs.items():
            tk.create_dataset(str(i), data=arr)


def load_grid_model(path):
    with h5py.File(path, "r") as f:
        assert _decode_fixed_str(f.attrs["scheme"]) == "grid"
        physics, coverage_delta, model_area_nm2 = _read_physics_header(f)
        n_classes = int(f.attrs["n_classes"])
        terms = _read_term_spec(f["terms"])

        g = f["grid"]
        method = _decode_fixed_str(g.attrs["interp_method"])
        n_axes = int(g.attrs["n_axes"])
        axg = g["axes"]
        axes = []
        for d in range(n_axes):
            sub = axg[str(d)]
            axes.append(AxisSpec(sub["values"][:], edge_policy=_decode_fixed_str(sub.attrs["edge_policy"])))
        axes_shape = tuple(len(a.values) for a in axes)

        cg = g["cells"]
        coverage_max_arr = cg["coverage_max"][:]
        ceiling_arr = cg["ceiling"][:]
        scaler_mean_arr = cg["scaler_mean"][:]
        scaler_scale_arr = cg["scaler_scale"][:]
        n_present_arr = cg["n_present_classes"][:]
        present_arr = cg["present_classes"][:]
        coef_arr = cg["coef"][:]
        intercept_arr = cg["intercept"][:]
        tk = cg["term_knots"]
        term_knot_arrs = {i: tk[str(i)][:] for i, term in enumerate(terms) if isinstance(term, SplineTerm)}

        cell_classifiers = {}
        for idx in itertools.product(*(range(n) for n in axes_shape)):
            knot_vectors = [term_knot_arrs[i][idx] if i in term_knot_arrs else None for i in range(len(terms))]
            clf = _classifier_from_padded(
                terms, knot_vectors, scaler_mean_arr[idx], scaler_scale_arr[idx],
                int(n_present_arr[idx]), present_arr[idx], coef_arr[idx], intercept_arr[idx], n_classes)
            ceiling = None if np.isnan(ceiling_arr[idx]) else float(ceiling_arr[idx])
            cell_classifiers[idx] = CoverageCell(clf, physics, float(coverage_max_arr[idx]), ceiling)

    model = GridClassifier(axes, cell_classifiers, n_classes, method=method)
    return model, coverage_delta, model_area_nm2
