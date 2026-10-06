"""
Equivalence of two processing sets: one opened from an MSv2 with the
``xradio_msv2`` engine, and the converted processing set of the same MS opened
with ``open_processing_set`` (the reference).

Not exported by ``xradio.testing.measurement_set``: import it by module path.
"""

import copy
import json

import numpy as np
import xarray as xr

#: Attributes that hold the time a dataset or data group was built
DATE_ATTRS = ("creation_date", "date")


def mask_dates(attrs: dict) -> dict:
    """A copy of ``attrs`` with the build dates (``creation_date`` and the
    ``date`` of every data group) replaced by "<date>"."""
    attrs = copy.deepcopy(dict(attrs))
    if "creation_date" in attrs:
        attrs["creation_date"] = "<date>"
    for group in attrs.get("data_groups", {}).values():
        if isinstance(group, dict) and "date" in group:
            group["date"] = "<date>"
    return attrs


def _to_json(obj):
    """JSON-normal form (as a zarr attribute round trip gives)."""
    if isinstance(obj, dict):
        return {str(k): _to_json(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_to_json(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _to_json(obj.tolist())
    if isinstance(obj, np.generic):
        return _to_json(obj.item())
    if isinstance(obj, float) and np.isnan(obj):
        return "NaN"
    return obj


def main_data_variables(node: xr.DataTree) -> list[str]:
    """The data variables of the main xds of an MSv4 node along time."""
    return [str(name) for name, var in node.data_vars.items() if "time" in var.dims]


def _values_equal(a, b) -> bool:
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return False
    if a.dtype.kind in "OUST" or b.dtype.kind in "OUST":
        return bool(np.array_equal(a.astype(str), b.astype(str)))
    if a.dtype != b.dtype:
        return False
    return a.tobytes() == b.tobytes()


def assert_nodes_identical(engine: xr.DataTree, reference: xr.DataTree) -> None:
    """
    The same node paths and, per node, identical datasets
    (``xr.testing.assert_identical`` of ``to_dataset(inherit=False)``, values
    loaded) and the same dtypes (strings: any string dtype), the build dates
    masked.
    """
    paths = sorted(node.path for node in engine.subtree)
    assert paths == sorted(node.path for node in reference.subtree)
    for path in paths:
        ds_e = engine[path].to_dataset(inherit=False)
        ds_r = reference[path].to_dataset(inherit=False)
        xr.testing.assert_identical(
            ds_e.assign_attrs(mask_dates(ds_e.attrs)),
            ds_r.assign_attrs(mask_dates(ds_r.attrs)),
        )
        for name, var in ds_r.variables.items():
            if var.dtype.kind not in "OUST":  # (strings: any string dtype)
                assert ds_e.variables[name].dtype == var.dtype, f"{path}/{name}"


def assert_main_chunks_equal(engine: xr.DataTree, reference: xr.DataTree) -> None:
    """The dask chunks of the main data variables of every MSv4 are equal."""
    for name, node in reference.children.items():
        for var in main_data_variables(node):
            assert engine[name][var].chunks == node[var].chunks, f"{name}/{var}"


def assert_lazy_selections_equal(
    engine: xr.DataTree, reference: xr.DataTree, rng=None
) -> int:
    """
    Selections of the main data variables of every MSv4 (steps, reversed
    steps, scalars, random baselines, a frequency slice or one channel in the
    middle) give bit-identical values. Returns the number of selections
    compared.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    compared = 0
    for name, node in reference.children.items():
        ds_r, ds_e = node.dataset, engine[name].dataset
        nt = ds_r.sizes["time"]
        bdim = "baseline_id" if "baseline_id" in ds_r.dims else "antenna_name"
        nb = ds_r.sizes[bdim]
        baselines = sorted(rng.choice(nb, size=min(3, nb), replace=False).tolist())
        nf = ds_r.sizes.get("frequency", 0)
        # (one channel: a channel-sliced read where the MS's tiles hold fewer
        # channels than a cell)
        frequency_sels = (slice(1, None), slice(nf // 2, nf // 2 + 1))
        for var in main_data_variables(node):
            for k, time_sel in enumerate(
                (
                    slice(None),
                    slice(1, None, 3),
                    nt // 2,
                    slice(nt - 1, None, -2),
                )
            ):
                sel = {"time": time_sel}
                if bdim in ds_r[var].dims:
                    sel[bdim] = baselines
                if "frequency" in ds_r[var].dims and nf > 2:
                    sel["frequency"] = frequency_sels[k % 2]
                assert _values_equal(
                    ds_e[var].isel(sel).values, ds_r[var].isel(sel).values
                ), f"{name}/{var} {sel}"
                compared += 1
    return compared


def assert_accessors_equal(engine: xr.DataTree, reference: xr.DataTree) -> None:
    """The xr_ps / xr_ms accessors give the same results."""
    summary_e, summary_r = engine.xr_ps.summary(), reference.xr_ps.summary()
    assert summary_e.astype(str).equals(summary_r.astype(str))
    assert engine.xr_ps.get_max_dims() == reference.xr_ps.get_max_dims()
    xr.testing.assert_identical(
        engine.xr_ps.get_freq_axis(), reference.xr_ps.get_freq_axis()
    )
    xr.testing.assert_identical(
        engine.xr_ps.get_combined_antenna_xds().compute(),
        reference.xr_ps.get_combined_antenna_xds().compute(),
    )
    xr.testing.assert_identical(
        engine.xr_ps.get_combined_field_and_source_xds().compute(),
        reference.xr_ps.get_combined_field_and_source_xds().compute(),
    )
    for name, node in reference.children.items():
        info_e = engine[name].xr_ms.get_partition_info()
        info_r = node.xr_ms.get_partition_info()
        assert json.dumps(_to_json(info_e), sort_keys=True) == json.dumps(
            _to_json(info_r), sort_keys=True
        ), name
    first = sorted(reference.children)[0]
    pol = str(reference[first].polarization.values[0])
    sel_e = engine[first].xr_ms.sel(data_group_name="base", polarization=pol)
    sel_r = reference[first].xr_ms.sel(data_group_name="base", polarization=pol)
    data = sel_r.attrs["data_groups"]["base"]["correlated_data"]
    assert sorted(sel_e.data_vars) == sorted(sel_r.data_vars)
    assert _values_equal(sel_e[data].values, sel_r[data].values)
    intents = summary_r["scan_intents"].iloc[0]
    assert sorted(engine.xr_ps.query(scan_intents=intents).children) == sorted(
        reference.xr_ps.query(scan_intents=intents).children
    )


def assert_processing_sets_equivalent(
    engine: xr.DataTree,
    reference: xr.DataTree,
    *,
    chunks: bool = True,
    accessors: bool = True,
    rng=None,
) -> None:
    """
    Assert that a processing set opened with the ``xradio_msv2`` engine
    (``engine``) is equivalent to the converted processing set of the same MS
    (``reference``, opened with ``open_processing_set``):

    - the same nodes, root attributes and, per node, identical datasets
      (values, dtypes, attributes; build dates masked);
    - with ``chunks``, the same dask chunks of the main data variables;
    - bit-identical lazy selections of the main data variables;
    - with ``accessors``, the same xr_ps / xr_ms accessor results.
    """
    assert engine.attrs == reference.attrs
    assert_nodes_identical(engine, reference)
    if chunks:
        assert_main_chunks_equal(engine, reference)
    assert_lazy_selections_equal(engine, reference, rng)
    if accessors and reference.children:
        assert_accessors_equal(engine, reference)
