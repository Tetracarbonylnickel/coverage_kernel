"""
export_hdf5.py
==============
Writes a fitted coverage model (grid scheme or NN ensemble) to one
self-describing HDF5 file that the Fortran port reads, and reads it back.
See docs/hdf5_schema.md for the layout.

    load_model(path)                     -> GridCoverageModel or NNEnsembleCoverageModel
    export_grid_model(path, model, ...)  /  load_grid_model(path)
    export_nn_model(path, model, ...)    /  load_nn_model(path)

Every array's shape comes from the fitted model. The coverage physics
(n_sites, gate indices), coverage_delta and the optional model_area_nm2 are
stored in the root attributes and come back as attributes of the loaded model.

One subtlety of the grid scheme: a SplineClassifier only carries
coefficients for the classes observed during its fit (`present_classes`).
A missing class is absent from the softmax and set to probability 0
afterwards, which is not the same as a zero coefficient row (that would
still get a softmax share). So `present_classes` and `n_present_classes`
are stored explicitly, and the Fortran reader repeats the same logic.

All strings are fixed-width byte strings (S32), which Fortran maps onto
CHARACTER(LEN=32); flags are plain integers 0/1.
"""

import itertools

import h5py
import numpy as np

from .spline_classifier import SplineTerm, LinearTerm, SplineClassifier
from .coverage_model import CoveragePhysics, CoverageCell, GridCoverageModel
from .grid_classifier import AxisSpec, GridClassifier
from . import nn

FORMAT_VERSION = "1.1"               # 1.1: scheme "nn_ensemble" added, scheme "pooled" removed
READABLE_VERSIONS = ("1.0", "1.1")   # grid files written as 1.0 have the same layout
STR_DTYPE = "S32"


def _fixed_str_array(values):
    return np.array(values, dtype=STR_DTYPE)


def _decode_fixed_str(value):
    """h5py returns fixed-width strings as numpy.bytes_ (or bytes); return a plain str."""
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8")
    return str(value)


# --------------------------------------------------------------------------- #
# shared helpers                                                              #
# --------------------------------------------------------------------------- #

def _write_physics_header(f, scheme, physics: CoveragePhysics, n_classes, coverage_delta, model_area_nm2):
    if coverage_delta is None:
        raise ValueError("coverage_delta is required (one entry per outcome class)")
    f.attrs["format_version"] = _fixed_str_array(FORMAT_VERSION)
    f.attrs["scheme"] = _fixed_str_array(scheme)
    f.attrs["n_classes"] = int(n_classes)
    f.attrs["n_sites"] = int(physics.n_sites)
    f.attrs["zeroed_zero_index"] = int(physics.zeroed_zero_index)
    f.attrs["zeroed_full_index"] = int(physics.zeroed_full_index)
    f.attrs["absorbing_full_index"] = int(physics.absorbing_full_index)
    f.create_dataset("coverage_delta", data=np.asarray(coverage_delta, dtype=np.float64))
    if model_area_nm2 is not None:
        f.attrs["model_area_nm2"] = float(model_area_nm2)


def _read_physics_header(f, scheme):
    version = _decode_fixed_str(f.attrs["format_version"])
    if version not in READABLE_VERSIONS:
        raise ValueError(f"format_version {version!r} not supported (readable: {READABLE_VERSIONS})")
    found = _decode_fixed_str(f.attrs["scheme"])
    if found != scheme:
        raise ValueError(f"file holds scheme {found!r}, expected {scheme!r}")
    physics = CoveragePhysics(
        n_sites=int(f.attrs["n_sites"]),
        zeroed_zero_index=int(f.attrs["zeroed_zero_index"]),
        zeroed_full_index=int(f.attrs["zeroed_full_index"]),
        absorbing_full_index=int(f.attrs["absorbing_full_index"]),
    )
    coverage_delta = f["coverage_delta"][:]
    model_area_nm2 = float(f.attrs["model_area_nm2"]) if "model_area_nm2" in f.attrs else None
    return physics, coverage_delta, model_area_nm2


def _write_ceiling(g, axes, grid):
    g.attrs["has_ceiling"] = int(grid is not None)
    if grid is not None:
        ca = g.create_group("ceiling_axes")
        for d, axis_vals in enumerate(axes):
            ca.create_group(str(d)).create_dataset("values", data=np.asarray(axis_vals, dtype=np.float64))
        g.create_dataset("ceiling_grid", data=np.asarray(grid, dtype=np.float64))


def _read_ceiling(g):
    if not int(g.attrs["has_ceiling"]):
        return None, None
    ca = g["ceiling_axes"]
    return [ca[str(d)]["values"][:] for d in range(len(ca))], g["ceiling_grid"][:]


def load_model(path):
    """Load any model file written by this module (dispatch on /scheme)."""
    with h5py.File(path, "r") as f:
        scheme = _decode_fixed_str(f.attrs["scheme"])
    if scheme == "grid":
        return load_grid_model(path)
    if scheme == "nn_ensemble":
        return load_nn_model(path)
    raise ValueError(f"unknown scheme {scheme!r} in {path}")


# --------------------------------------------------------------------------- #
# grid scheme                                                                 #
# --------------------------------------------------------------------------- #

def _write_term_spec(group, terms):
    group.attrs["n_terms"] = len(terms)
    group.create_dataset("kind", data=_fixed_str_array(["spline" if isinstance(t, SplineTerm) else "linear"
                                                         for t in terms]))
    group.create_dataset("column", data=np.array([t.column for t in terms], dtype=np.int64))
    group.create_dataset("degree", data=np.array([t.degree if isinstance(t, SplineTerm) else -1 for t in terms],
                                                 dtype=np.int64))
    group.create_dataset("n_knots", data=np.array([t.n_knots if isinstance(t, SplineTerm) else -1 for t in terms],
                                                  dtype=np.int64))


def _read_term_spec(group):
    n_terms = int(group.attrs["n_terms"])
    kind = [_decode_fixed_str(k) for k in group["kind"][:]]
    column, degree, n_knots = group["column"][:], group["degree"][:], group["n_knots"][:]
    return [SplineTerm(int(column[i]), n_knots=int(n_knots[i]), degree=int(degree[i])) if kind[i] == "spline"
            else LinearTerm(int(column[i])) for i in range(n_terms)]


def _padded_present(clf: SplineClassifier, n_classes):
    """(n_present, present_padded[n_classes], coef_padded[n_classes, n_features],
    intercept_padded[n_classes]); see the module docstring for why the padding
    alone (without n_present / present_classes) would be wrong."""
    n_present = len(clf.present_classes)
    present_padded = np.full(n_classes, -1, dtype=np.int64)
    present_padded[:n_present] = clf.present_classes
    coef_padded = np.zeros((n_classes, clf.coef.shape[1]))
    coef_padded[:n_present] = clf.coef
    intercept_padded = np.zeros(n_classes)
    intercept_padded[:n_present] = clf.intercept
    return n_present, present_padded, coef_padded, intercept_padded


def export_grid_model(path, model: GridCoverageModel, coverage_delta=None, model_area_nm2=None):
    """coverage_delta / model_area_nm2 default to the model's own attributes."""
    grid = model.grid
    coverage_delta = model.coverage_delta if coverage_delta is None else coverage_delta
    model_area_nm2 = model.model_area_nm2 if model_area_nm2 is None else model_area_nm2
    raw_clf0 = next(iter(grid.cell_classifiers.values())).raw_classifier
    terms, n_classes, n_features = raw_clf0.terms, grid.n_classes, raw_clf0.coef.shape[1]
    axes_shape = tuple(len(a.values) for a in grid.axes)

    coverage_max_arr = np.zeros(axes_shape)
    ceiling_arr = np.full(axes_shape, np.nan)
    scaler_mean_arr = np.zeros(axes_shape + (n_features,))
    scaler_scale_arr = np.zeros(axes_shape + (n_features,))
    n_present_arr = np.zeros(axes_shape, dtype=np.int64)
    present_arr = np.full(axes_shape + (n_classes,), -1, dtype=np.int64)
    coef_arr = np.zeros(axes_shape + (n_classes, n_features))
    intercept_arr = np.zeros(axes_shape + (n_classes,))
    term_knot_arrs = {i: np.zeros(axes_shape + (len(knots),))
                      for i, knots in enumerate(raw_clf0.knot_vectors) if knots is not None}

    for idx in itertools.product(*(range(n) for n in axes_shape)):
        cell = grid.cell_classifiers[idx]
        raw_clf = cell.raw_classifier
        coverage_max_arr[idx] = cell.coverage_max
        if cell.ceiling is not None:
            ceiling_arr[idx] = cell.ceiling
        scaler_mean_arr[idx] = raw_clf.scaler_mean
        scaler_scale_arr[idx] = raw_clf.scaler_scale
        n_present_arr[idx], present_arr[idx], coef_arr[idx], intercept_arr[idx] = _padded_present(raw_clf, n_classes)
        for i, knots in enumerate(raw_clf.knot_vectors):
            if knots is not None:
                term_knot_arrs[i][idx] = knots

    with h5py.File(path, "w") as f:
        _write_physics_header(f, "grid", model.physics, n_classes, coverage_delta, model_area_nm2)
        _write_term_spec(f.create_group("terms"), terms)
        g = f.create_group("grid")
        g.attrs["n_axes"] = len(grid.axes)
        g.attrs["interp_method"] = _fixed_str_array(grid.method)
        axg = g.create_group("axes")
        for d, axis in enumerate(grid.axes):
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
        physics, coverage_delta, model_area_nm2 = _read_physics_header(f, "grid")
        n_classes = int(f.attrs["n_classes"])
        terms = _read_term_spec(f["terms"])
        g = f["grid"]
        method = _decode_fixed_str(g.attrs["interp_method"])
        axes = [AxisSpec(g["axes"][str(d)]["values"][:], edge_policy=_decode_fixed_str(g["axes"][str(d)].attrs["edge_policy"]))
                for d in range(int(g.attrs["n_axes"]))]
        axes_shape = tuple(len(a.values) for a in axes)
        cg = g["cells"]
        cov_max, ceil = cg["coverage_max"][:], cg["ceiling"][:]
        s_mean, s_scale = cg["scaler_mean"][:], cg["scaler_scale"][:]
        n_present, present, coef, intercept = (cg["n_present_classes"][:], cg["present_classes"][:],
                                               cg["coef"][:], cg["intercept"][:])
        knots = {i: cg["term_knots"][str(i)][:] for i, t in enumerate(terms) if isinstance(t, SplineTerm)}

        cells = {}
        for idx in itertools.product(*(range(n) for n in axes_shape)):
            k = int(n_present[idx])
            clf = SplineClassifier(terms, [knots[i][idx] if i in knots else None for i in range(len(terms))],
                                   s_mean[idx], s_scale[idx], coef[idx][:k], intercept[idx][:k],
                                   present[idx][:k], n_classes)
            cells[idx] = CoverageCell(clf, physics, float(cov_max[idx]),
                                      None if np.isnan(ceil[idx]) else float(ceil[idx]))
    return GridCoverageModel(GridClassifier(axes, cells, n_classes, method=method), physics,
                             coverage_delta=coverage_delta, model_area_nm2=model_area_nm2)


# --------------------------------------------------------------------------- #
# NN ensemble                                                                 #
# --------------------------------------------------------------------------- #

def export_nn_model(path, model: "nn.NNEnsembleCoverageModel", coverage_delta=None, model_area_nm2=None):
    """coverage_delta / model_area_nm2 default to the model's own attributes."""
    ens = model.ensemble
    coverage_delta = model.coverage_delta if coverage_delta is None else coverage_delta
    model_area_nm2 = model.model_area_nm2 if model_area_nm2 is None else model_area_nm2
    with h5py.File(path, "w") as f:
        _write_physics_header(f, "nn_ensemble", model.physics, ens.n_classes, coverage_delta, model_area_nm2)
        g = f.create_group("nn")
        g.attrs["n_members"] = len(ens.members)
        g.attrs["combination"] = _fixed_str_array("mean")
        g.attrs["coverage_max"] = float(model.coverage_max)
        g.attrs["offset_class"] = int(ens.offset_class)
        g.attrs["offset_floor"] = float(nn.OFFSET_FLOOR)
        g.create_dataset("inputs", data=_fixed_str_array(["velocity", "angle", "coverage_fraction"]))
        for key, value in model.metadata.items():   # informational only (the Fortran reader ignores them)
            g.attrs["training_" + key] = value
        for i, net in enumerate(ens.members):
            m = g.create_group(f"member_{i}")
            m.attrs["n_layers"] = len(net.layers)
            m.create_dataset("input_mean", data=net.input_mean)
            m.create_dataset("input_std", data=net.input_std)
            for j, (w, b, act) in enumerate(net.layers):
                lg = m.create_group(f"layer_{j}")
                lg.create_dataset("weights", data=w)
                lg.create_dataset("biases", data=b)
                lg.attrs["activation"] = _fixed_str_array(act)
        _write_ceiling(g, model.ceiling_axes, model.ceiling_grid)


def load_nn_model(path, name=None):
    with h5py.File(path, "r") as f:
        physics, coverage_delta, model_area_nm2 = _read_physics_header(f, "nn_ensemble")
        g = f["nn"]
        if _decode_fixed_str(g.attrs["combination"]) != "mean":
            raise ValueError("only combination 'mean' is supported")
        if float(g.attrs["offset_floor"]) != nn.OFFSET_FLOOR:
            raise ValueError("offset_floor differs from nn.OFFSET_FLOOR")
        members = []
        for i in range(int(g.attrs["n_members"])):
            m = g[f"member_{i}"]
            layers = [(m[f"layer_{j}"]["weights"][:], m[f"layer_{j}"]["biases"][:],
                       _decode_fixed_str(m[f"layer_{j}"].attrs["activation"])) for j in range(int(m.attrs["n_layers"]))]
            members.append(nn.MLP(layers, m["input_mean"][:], m["input_std"][:]))
        ensemble = nn.NNEnsemble(members, physics.n_sites, int(g.attrs["offset_class"]), physics.zeroed_zero_index)
        ceiling_axes, ceiling_grid = _read_ceiling(g)
        metadata = {k[len("training_"):]: (_decode_fixed_str(v) if isinstance(v, (bytes, np.bytes_)) else v)
                    for k, v in g.attrs.items() if k.startswith("training_")}
        coverage_max = float(g.attrs["coverage_max"])
    return nn.NNEnsembleCoverageModel(ensemble, physics, coverage_max, ceiling_axes, ceiling_grid,
                                      coverage_delta=coverage_delta, model_area_nm2=model_area_nm2,
                                      name=name or str(path), metadata=metadata)
