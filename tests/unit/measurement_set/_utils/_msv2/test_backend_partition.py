"""Tests of one MSv2 partition as a lazy MSv4 node (backend_partition.py)."""

import hashlib
import json
import os
import shutil

import numpy as np
import pytest
import xarray as xr

from xradio._utils.zarr.config import ZARR_FORMAT
from xradio.measurement_set._utils._msv2 import (
    backend_arrays,
    backend_partition,
    conversion,
)
from xradio.measurement_set._utils._msv2._tables import read_rows
from xradio.measurement_set._utils._msv2.backend_arrays import (
    MSv2MainColumnArray,
    OnesArray,
)
from xradio.measurement_set._utils._msv2.backend_partition import open_partition
from xradio.measurement_set._utils._msv2.partition_queries import (
    create_partitions_with_main_rows,
)

MAIN_DATA_VARIABLES = {
    "VISIBILITY",
    "VISIBILITY_CORRECTED",
    "VISIBILITY_MODEL",
    "SPECTRUM",
    "FLAG",
    "WEIGHT",
    "UVW",
    "TIME_CENTROID",
    "EFFECTIVE_INTEGRATION_TIME",
}


def _driver_options(msname, scheme=()) -> dict:
    """The options of open_partition that the driver (backend_open) passes
    besides the build's: the MS's keys_token before the partitions are
    computed, and the scheme."""
    return {
        "keys_token": backend_arrays.keys_token(os.path.abspath(msname)),
        "partition_scheme": tuple(scheme),
    }


def _open(msname, scheme=(), idx=0, **kw):
    options = _driver_options(msname, scheme)
    partitions, runs = create_partitions_with_main_rows(msname, list(scheme))
    return open_partition(
        os.path.abspath(msname),
        partitions[idx],
        runs[idx],
        node_name="node",
        **options,
        **kw,
    )


def _store_contents(path: str) -> tuple[dict, dict]:
    """sha256 of every chunk file, and every zarr.json without dates."""

    def strip(obj):
        if isinstance(obj, dict):
            return {
                k: strip(v)
                for k, v in obj.items()
                if k not in ("creation_date", "date")
            }
        if isinstance(obj, list):
            return [strip(v) for v in obj]
        return obj

    chunks, metadata = {}, {}
    for dirpath, _, filenames in os.walk(path):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            rel = os.path.relpath(file_path, path)
            with open(file_path, "rb") as f:
                content = f.read()
            if filename == "zarr.json":
                metadata[rel] = strip(json.loads(content))
            else:
                chunks[rel] = hashlib.sha256(content).hexdigest()
    return chunks, metadata


def _ms_fds(msname: str) -> list[str]:
    """Open file descriptors of this process under the MS directory."""
    root = os.path.realpath(msname)
    found = []
    for fd in os.listdir("/proc/self/fd"):
        try:
            target = os.readlink(f"/proc/self/fd/{fd}")
        except OSError:
            continue
        if target.startswith(root + os.sep):
            found.append(target)
    return found


WRITE_CASES = {
    "dense_reversed": ("dense", [], 2, {}),
    "rich_field_time3": ("rich", ["FIELD_ID"], 5, {"main_chunksize": {"time": 3}}),
    "rich_interpolated": (
        "rich",
        [],
        1,
        {
            "pointing_interpolate": True,
            "ephemeris_interpolate": True,
            "sys_cal_interpolate": True,
            "pointing_chunksize": 0.00001,
        },
    ),
    "single_dish_antenna1": ("single_dish", ["ANTENNA1"], 3, {}),
    "single_dish": ("single_dish", [], 2, {"main_chunksize": 0.0001}),
    "no_weight": ("no_weight", [], 0, {}),
    "wsp_partial_no_pointing": ("wsp_partial", [], 3, {"with_pointing": False}),
    "sparse_dup_frequency_chunks": (
        "sparse_dup",
        [],
        1,
        {"main_chunksize": {"time": 4, "frequency": 5}},
    ),
}


@pytest.mark.parametrize("case", list(WRITE_CASES))
def test_node_writes_the_converters_msv4(backend_ms, case, tmp_path):
    """
    The node (lazy reads) written by to_zarr is byte for byte the MSv4 the
    converter writes for the partition (chunk files and metadata, dates
    aside): the same variables, values, attributes, encoding and
    sub-datasets.
    """
    variant, scheme, idx, kw = WRITE_CASES[case]
    msname = backend_ms(variant)
    options = _driver_options(msname, scheme)
    partitions, runs = create_partitions_with_main_rows(msname, scheme)
    node = open_partition(
        os.path.abspath(msname),
        partitions[idx],
        runs[idx],
        node_name="node",
        **options,
        **kw,
    )
    node.to_zarr(str(tmp_path / "engine"), mode="w", zarr_format=ZARR_FORMAT)
    out = str(tmp_path / "converted")
    conversion.convert_and_write_partition(
        msname,
        out,
        "0",
        partition_info=partitions[idx],
        use_table_iter=False,
        main_row_runs=runs[idx],
        **kw,
    )
    converted = os.path.join(out, os.path.basename(msname).replace(".ms", "") + "_0")
    chunks_a, metadata_a = _store_contents(converted)
    chunks_b, metadata_b = _store_contents(str(tmp_path / "engine"))
    assert sorted(chunks_a) == sorted(chunks_b)
    assert [name for name in chunks_a if chunks_a[name] != chunks_b[name]] == []
    assert metadata_a == metadata_b


def test_every_main_data_variable_is_lazy_and_nothing_is_read(backend_ms, monkeypatch):
    """
    Opening reads no MAIN data column on the grid: every data variable from
    a MAIN column is a lazily indexed MSv2 array (no placeholder left), and
    no grid read happens until the values are asked for (the one cell an
    open reads per column: test_open_reads_the_first_cell_of_each_column).
    """
    reads = []
    read_grid = read_rows.read_rows_to_grid

    def spy(table, col, *args, **kwargs):
        reads.append(col)
        return read_grid(table, col, *args, **kwargs)

    monkeypatch.setattr(read_rows, "read_rows_to_grid", spy)
    node = _open(backend_ms("rich"), idx=5)
    assert reads == []
    xds = node.to_dataset(inherit=False)
    assert MAIN_DATA_VARIABLES - {"SPECTRUM"} == set(xds.data_vars)
    for name in xds.data_vars:
        data = xds.variables[name]._data
        assert isinstance(data, xr.core.indexing.LazilyIndexedArray), name
        assert isinstance(data.array, MSv2MainColumnArray), name
        assert data.array.node == "node"
        assert not backend_partition._is_placeholder(xds.variables[name])
    for sub in node.children.values():
        assert not any(map(backend_partition._is_placeholder, sub.variables.values()))
    xds.VISIBILITY.isel(time=0).values  # noqa: B018
    assert reads == ["DATA"]


# The python-casacore table methods that read values
VALUE_READ_METHODS = (
    "getcell",
    "getcellslice",
    "getcol",
    "getcolnp",
    "getcolslice",
    "getcolslicenp",
    "getvarcol",
)


def test_open_reads_the_first_cell_of_each_column(backend_ms, monkeypatch):
    """
    Of the MAIN columns of the main data variables, opening an MS reads one
    cell per partition, that of the partition's first row (the converter's
    build takes the dtype from it: _partition_cell_shape_and_dtype), with
    getcell, and no other value (api.rst, "What is read when").
    """
    from casacore import tables

    from _xradio_xarray_backends import MSv2BackendEntrypoint

    msname = os.path.realpath(backend_ms("rich"))
    columns = {
        "DATA",
        "CORRECTED_DATA",
        "MODEL_DATA",
        "FLAG",
        "WEIGHT",
        "WEIGHT_SPECTRUM",
        "UVW",
        "TIME_CENTROID",
        "EXPOSURE",
    }
    reads = []

    def spying(method):
        read = getattr(tables.table, method)

        def spy(self, col, *args, **kwargs):
            if col in columns and os.path.realpath(self.name()) == msname:
                reads.append((method, col, args[:1]))
            return read(self, col, *args, **kwargs)

        return spy

    for method in VALUE_READ_METHODS:
        monkeypatch.setattr(tables.table, method, spying(method))
    tree = xr.open_datatree(msname, engine=MSv2BackendEntrypoint, partition_cache="off")
    opened = list(reads)
    _, runs = create_partitions_with_main_rows(msname, [])
    n_partitions = len(runs.bounds) - 1
    assert len(tree.children) == n_partitions
    first_rows = sorted(int(runs.starts[runs.bounds[i]]) for i in range(n_partitions))
    assert {method for method, _, _ in opened} == {"getcell"}
    rows = {}
    for _, col, (row,) in opened:
        rows.setdefault(col, []).append(int(row))
    # (WEIGHT_SPECTRUM is read, so WEIGHT is not)
    assert set(rows) == columns - {"WEIGHT"}
    for col, found in rows.items():
        assert sorted(found) == first_rows, col


def test_placeholders_left_are_an_error():
    import dask.array as da

    from xradio.measurement_set._utils._msv2 import stream_write

    xds = xr.Dataset(
        {"A": (("x",), stream_write.deferred_placeholder("A", (3,), np.float32))}
    )
    with pytest.raises(RuntimeError, match=r"\['A'\]"):
        backend_partition._check_no_placeholder_left(xds)
    backend_partition._check_no_placeholder_left(
        xr.Dataset({"A": (("x",), da.zeros(3))})
    )


def test_preferred_chunks(backend_ms):
    """
    preferred_chunks of the data variables are their zarr chunks
    (encoding["chunks"]); the coordinates along time (scan_name,
    field_name) get the time chunk of the data; pointing variables their
    own encoding chunks.
    """
    node = _open(
        backend_ms("rich"),
        idx=1,
        main_chunksize={"time": 3},
        pointing_chunksize=0.000001,
    )
    xds = node.to_dataset(inherit=False)
    for name, var in xds.data_vars.items():
        chunks = var.encoding["chunks"]
        assert var.encoding["preferred_chunks"] == dict(
            zip(var.dims, chunks, strict=True)
        ), name
        assert var.encoding["preferred_chunks"]["time"] == 3
    for name in ("scan_name", "field_name"):
        assert xds[name].encoding["preferred_chunks"] == {"time": 3}
    for name in xds.indexes:
        assert "preferred_chunks" not in xds.variables[name].encoding
    pointing = node["pointing_xds"].to_dataset(inherit=False)
    assert pointing.data_vars
    for name, var in pointing.data_vars.items():
        assert var.encoding["preferred_chunks"] == dict(
            zip(var.dims, var.encoding["chunks"], strict=True)
        ), name
    assert any(  # (the pointing chunk size gives several chunks)
        c < pointing.sizes[d]
        for var in pointing.data_vars.values()
        for d, c in var.encoding["preferred_chunks"].items()
    )


def test_default_chunks_are_the_converters(backend_ms):
    """Without main_chunksize, the preferred chunks are those of
    default_main_chunksize (one chunk along time for these small MSs)."""
    xds = _open(backend_ms("dense")).to_dataset(inherit=False)
    assert xds.VISIBILITY.encoding["preferred_chunks"] == dict(xds.VISIBILITY.sizes)
    assert xds.scan_name.encoding["preferred_chunks"] == {"time": xds.sizes["time"]}


@pytest.mark.parametrize(
    "drop",
    [
        "WEIGHT",
        ["VISIBILITY_MODEL", "NOT_A_VARIABLE"],
        ("FLAG", "UVW", "ANTENNA_POSITION", "POINTING_BEAM"),
    ],
)
def test_drop_variables(backend_ms, drop):
    """drop_variables: a str is one name, unknown names are ignored, names
    are dropped from every node (sub-datasets too) and from data_groups."""
    full = _open(backend_ms("rich"), idx=1)
    node = _open(backend_ms("rich"), idx=1, drop_variables=drop)
    names = [drop] if isinstance(drop, str) else list(drop)
    for sub in node.subtree:
        ds = sub.to_dataset(inherit=False)
        assert not set(names) & set(ds.variables), sub.path
        full_ds = full[sub.path].to_dataset(inherit=False)
        assert set(ds.variables) == set(full_ds.variables) - set(names), sub.path
    groups = node.attrs["data_groups"]
    assert set(groups) == set(full.attrs["data_groups"])
    for group in groups.values():
        assert not set(names) & set(group.values())
    if drop == "WEIGHT":
        assert all("weight" not in group for group in groups.values())
    # the attrs of the full node are not modified
    assert full.attrs["data_groups"]["base"]["weight"] == "WEIGHT"


def test_single_dish_node(backend_ms):
    """Single dish: SPECTRUM by antenna_name, no UVW (its column is never
    made lazy)."""
    xds = _open(backend_ms("single_dish"), ["ANTENNA1"], 3).to_dataset(inherit=False)
    assert "UVW" not in xds and "uvw_label" not in xds.dims
    assert xds.SPECTRUM.dims == ("time", "antenna_name", "frequency", "polarization")
    assert xds.attrs["type"] == "spectrum"


def test_weight_fallback_is_lazy(backend_ms):
    xds = _open(backend_ms("no_weight")).to_dataset(inherit=False)
    data = xds.WEIGHT.variable._data
    assert isinstance(data.array, OnesArray)
    assert xds.WEIGHT.dtype == np.float64 and xds.WEIGHT.shape == xds.VISIBILITY.shape
    assert np.all(xds.WEIGHT.isel(time=slice(0, 2)).values == 1)


def test_partition_without_rows(backend_ms):
    """ANTENNA1 partitions of an interferometer MS hold autocorrelations
    only: none here, so every partition opens as None."""
    partitions, runs = create_partitions_with_main_rows(
        backend_ms("dense"), ["ANTENNA1"]
    )
    assert runs[0].lengths.sum() == 0
    assert open_partition(backend_ms("dense"), partitions[0], runs[0]) is None


@pytest.mark.skipif(not os.path.isdir("/proc/self/fd"), reason="needs /proc")
def test_no_table_left_open(backend_ms, tmp_path, monkeypatch):
    """No file of the MS is open after a partition opened, after a lazy read,
    and after a failed open."""
    msname = str(tmp_path / "copy.ms")
    shutil.copytree(backend_ms("rich"), msname)
    node = _open(msname, idx=1)
    assert _ms_fds(msname) == []
    node.ds.FLAG.values  # noqa: B018
    assert _ms_fds(msname) == []

    def failing(*args, **kwargs):
        raise ValueError("simulated build failure")

    monkeypatch.setattr(conversion, "create_field_and_source_xds", failing)
    with pytest.raises(ValueError, match="simulated"):
        _open(msname, idx=1)
    assert _ms_fds(msname) == []


def test_build_holds_the_casatools_lock(backend_ms, monkeypatch):
    """The build of a partition runs in casatools_serialized (the
    process-wide casatools lock with casatools, a no-op otherwise)."""
    from contextlib import contextmanager

    held = []
    build = backend_partition.build_partition

    @contextmanager
    def serialized():
        held.append("lock")
        try:
            yield
        finally:
            held.append("unlock")

    @contextmanager
    def spy(*args, **kwargs):
        held.append("build")
        with build(*args, **kwargs) as built:
            yield built

    monkeypatch.setattr(backend_partition, "casatools_serialized", serialized)
    monkeypatch.setattr(backend_partition, "build_partition", spy)
    _open(backend_ms("dense"))
    assert held == ["lock", "build", "unlock"]


def test_index_is_seeded(backend_ms):
    """Opening seeds the index memo: the first read needs no rebuild."""
    backend_arrays.clear_index_memo()
    node = _open(backend_ms("shuffled"), idx=1)
    node.ds.VISIBILITY.values  # noqa: B018
    assert backend_arrays.INDEX_MEMO.stats["rebuilds"] == 0
