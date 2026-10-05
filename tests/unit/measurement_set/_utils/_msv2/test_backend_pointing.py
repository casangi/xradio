"""
The lazy pointing_xds of the MSv2 xarray backend (``backend_pointing.py``)
and the converter's hook for it (``create_pointing_xds(generic_loader=)``):
values bit-identical to the converter's pointing_xds (duplicated rows,
missing cells and their fill values, every selection), no POINTING value
read or kept at open (only the index), whatever the size of the table, the
tables the index cannot describe (built again on read), staleness,
pickling, threads and fork.
"""

import contextlib
import dataclasses
import functools
import gc
import os
import pickle
import re
import shutil
import threading
import time
import tracemalloc

import dask
import numpy as np
import pytest
import xarray as xr

tables = pytest.importorskip("casacore.tables")

from _xradio_xarray_backends import MSv2BackendEntrypoint  # noqa: E402
from xradio.measurement_set._utils._msv2 import (  # noqa: E402
    backend_pointing as bpt,
)
from xradio.measurement_set._utils._msv2._tables import (  # noqa: E402
    read_pointing as rp,
)
from xradio.measurement_set._utils._msv2._tables.subtable_cache import (  # noqa: E402
    SubtableCache,
    activate_subtable_cache,
)
from xradio.measurement_set._utils._msv2._tables.table_query import (  # noqa: E402
    open_table_ro,
)
from xradio.measurement_set._utils._msv2.backend_errors import (  # noqa: E402
    MSv2ChangedError,
    MSv2ReadError,
)
from xradio.measurement_set._utils._msv2.msv4_sub_xdss import (  # noqa: E402
    create_pointing_xds,
)

NANTS = 5
NTIMES = 400
TIME0 = 5.2e9
DATA_COLUMNS = ("DIRECTION", "ENCODER", "OVER_THE_TOP")
# The variants of make_pointing_ms
POINTING_VARIANTS = (
    "regular",
    "int_encoder",
    "no_encoder",
    "varying_target",
    "empty",
    "empty_encoder",
    "float_over_the_top",
    "two_polynomials",
    "varying_direction",
)
# Variants whose index read_pointing_index reads (lazy PointingColumnArray
# variables); "empty" has no rows, and the pointing_xds of the others is
# built by the converter's code (PointingBuildArray variables)
LAZY_VARIANTS = (
    "regular",
    "int_encoder",
    "no_encoder",
    "two_polynomials",
    "varying_target",
)
REBUILT_VARIANTS = ("empty_encoder", "float_over_the_top")


def make_pointing_ms(
    path: str, variant: str = "regular", seed: int = 0, ntimes: int = NTIMES
) -> str:
    """
    A directory with only a POINTING sub-table (all create_pointing_xds
    reads), as in test_read_pointing.py.

    Every antenna is sampled at its own subset of ``ntimes`` times (missing
    (time, antenna) cells, antenna 0 at about 60% of them), rows are
    shuffled (not time-ordered) and a few rows repeat the (TIME, ANTENNA_ID)
    of another row with other values (duplicates: the lowest row wins).

    variant:
    - "regular": all cells defined, one shape per column; DIRECTION (1, 2)
      and ENCODER (2,) doubles, OVER_THE_TOP boolean
    - "int_encoder": ENCODER is an int array column (xarray's fill of missing
      cells promotes ints to float)
    - "no_encoder": no ENCODER and OVER_THE_TOP columns
    - "varying_target": TARGET (not converted) has cells of two shapes
    - "empty": no rows
    - "empty_encoder": zero-size ENCODER cells
    - "float_over_the_top": OVER_THE_TOP is a float column
    - "two_polynomials": DIRECTION and TARGET cells of shape (2, 2) (the
      converter cannot convert them)
    - "varying_direction": a few DIRECTION cells of shape (2, 2), the others
      (and the first row) (1, 2)
    """
    os.makedirs(path, exist_ok=True)
    rng = np.random.default_rng(seed)
    extra = []
    if variant != "no_encoder":
        if variant == "int_encoder":
            extra.append(tables.makearrcoldesc("ENCODER", 0, ndim=1, valuetype="int"))
        else:
            extra.append(tables.makearrcoldesc("ENCODER", 0.0, ndim=1))
        if variant == "float_over_the_top":
            extra.append(tables.makescacoldesc("OVER_THE_TOP", 0.0, valuetype="float"))
        else:
            extra.append(tables.makescacoldesc("OVER_THE_TOP", False))
    tb = tables.default_ms_subtable(
        "POINTING", os.path.join(path, "POINTING"), tables.maketabdesc(extra)
    )
    try:
        if variant == "empty":
            return path
        times = TIME0 + np.arange(ntimes) * 0.048
        rows_t, rows_a = [], []
        for ant in range(NANTS):
            keep = rng.random(ntimes) < (0.95 if ant else 0.6)
            rows_t.append(times[keep])
            rows_a.append(np.full(keep.sum(), ant))
        time = np.concatenate(rows_t)
        ant = np.concatenate(rows_a)
        dup = rng.choice(time.size, max(25, time.size // 50), replace=False)
        time = np.concatenate([time, time[dup]])
        ant = np.concatenate([ant, ant[dup]])
        order = rng.permutation(time.size)
        time, ant = time[order], ant[order]
        nrows = time.size
        tb.addrows(nrows)
        tb.putcol("TIME", time)
        tb.putcol("ANTENNA_ID", ant.astype(np.int32))
        tb.putcol("INTERVAL", np.full(nrows, 0.048))
        tb.putcol("NAME", np.array([f"src{r % 3}" for r in range(nrows)]))
        tb.putcol("NUM_POLY", np.zeros(nrows, np.int32))
        tb.putcol("TIME_ORIGIN", time)
        tb.putcol("TRACKING", rng.random(nrows) < 0.5)
        n_poly = 2 if variant == "two_polynomials" else 1
        tb.putcol("DIRECTION", rng.normal(size=(nrows, n_poly, 2)))
        if variant == "varying_direction":
            for row in range(1, nrows, 211):
                tb.putcell("DIRECTION", row, rng.normal(size=(2, 2)))
        if variant == "varying_target":
            for row in range(nrows):
                shape = (2, 2) if row % 97 == 5 else (1, 2)
                tb.putcell("TARGET", row, rng.normal(size=shape))
        else:
            tb.putcol("TARGET", rng.normal(size=(nrows, n_poly, 2)))
        if variant == "empty_encoder":
            for row in range(nrows):
                tb.putcell("ENCODER", row, np.zeros(0))
        elif variant == "int_encoder":
            tb.putcol("ENCODER", rng.integers(-(2**30), 2**30, size=(nrows, 2)))
        elif variant != "no_encoder":
            tb.putcol("ENCODER", rng.normal(size=(nrows, 2)))
        if variant == "float_over_the_top":
            tb.putcol("OVER_THE_TOP", rng.random(nrows).astype(np.float32))
        elif variant != "no_encoder":
            tb.putcol("OVER_THE_TOP", rng.random(nrows) < 0.3)
    finally:
        tb.close()
    return path


@pytest.fixture(scope="module")
def pointing_ms(tmp_path_factory):
    """The make_pointing_ms tables by variant (read only)."""
    base = tmp_path_factory.mktemp("backend_pointing")
    return {
        variant: make_pointing_ms(str(base / f"{variant}.ms"), variant)
        for variant in POINTING_VARIANTS
    }


def antenna_names(antenna_ids) -> xr.DataArray:
    """antenna_name with an antenna_id index, as build_partition passes it"""
    antenna_ids = np.asarray(antenna_ids)
    return xr.DataArray(
        np.array([f"ant{a}" for a in antenna_ids]),
        dims="antenna_name",
        coords={"antenna_id": ("antenna_name", antenna_ids)},
        name="antenna_name",
    ).set_xindex("antenna_id")


def assert_xds_bit_identical(a: xr.Dataset, b: xr.Dataset) -> None:
    """Same variables (and order), dims, dtypes, bits, indexes and attrs."""
    assert list(a.variables) == list(b.variables)
    assert list(a.data_vars) == list(b.data_vars)
    assert dict(a.sizes) == dict(b.sizes)
    assert set(a.xindexes) == set(b.xindexes)
    assert repr(a.attrs) == repr(b.attrs)
    for name, var_a in a.variables.items():
        var_b = b.variables[name]
        assert var_a.dims == var_b.dims, name
        assert var_a.dtype == var_b.dtype, name
        values_a, values_b = var_a.values, var_b.values
        if values_a.dtype == object:
            np.testing.assert_array_equal(values_a, values_b, err_msg=name)
        else:
            assert (
                np.ascontiguousarray(values_a).tobytes()
                == np.ascontiguousarray(values_b).tobytes()
            ), name
        assert repr(var_a.attrs) == repr(var_b.attrs), name


def whole_time_range(ms: str) -> tuple[np.float64, np.float64]:
    return (np.float64(TIME0 - 1), np.float64(TIME0 + NTIMES))


def test_read_pointing_index(pointing_ms, monkeypatch):
    """The index (TIME, ANTENNA_ID, row numbers sorted by TIME) equals the
    one of the converter's sub-table cache, and its selections give the
    converter's generic pointing dataset; it is read for tables of any size
    (the cache's limit of POINTING_MAX_CACHED_INDEX_BYTES does not apply);
    the tables it cannot describe give None."""
    table = os.path.join(pointing_ms["regular"], "POINTING")
    kept = rp.read_pointing_columns(table, DATA_COLUMNS)
    index = bpt.read_pointing_index(table, DATA_COLUMNS)
    columns = index.columns
    assert kept.data is not None and columns.data is None
    assert columns.data_columns == kept.data_columns == DATA_COLUMNS
    for name in ("time", "antenna_id", "row", "tolerance"):
        np.testing.assert_array_equal(getattr(columns, name), getattr(kept, name))
    assert columns.data_dims == kept.data_dims
    assert columns.nbytes == kept.time.nbytes + kept.antenna_id.nbytes + kept.row.nbytes
    assert index.cell_shapes == {
        "DIRECTION": (1, 2),
        "ENCODER": (2,),
        "OVER_THE_TOP": (),
    }
    assert index.odd_rows.size == 0
    ants = np.arange(NANTS)
    with_data = dataclasses.replace(columns, data=kept.data)
    assert_xds_bit_identical(
        rp.pointing_generic_xds(with_data, whole_time_range(table), ants),
        rp.pointing_generic_xds(kept, whole_time_range(table), ants),
    )
    monkeypatch.setattr(rp, "POINTING_MAX_CACHED_INDEX_BYTES", 1024)
    assert rp.read_pointing_columns(table, DATA_COLUMNS) is None
    large = bpt.read_pointing_index(table, DATA_COLUMNS)
    np.testing.assert_array_equal(large.columns.row, columns.row)
    for variant in LAZY_VARIANTS + ("varying_direction",):
        table = os.path.join(pointing_ms[variant], "POINTING")
        index = bpt.read_pointing_index(table, DATA_COLUMNS)
        assert index is not None, variant
        np.testing.assert_array_equal(index.odd_rows, odd_rows(table), variant)
    for variant in REBUILT_VARIANTS + ("empty",):
        table = os.path.join(pointing_ms[variant], "POINTING")
        assert bpt.read_pointing_index(table, DATA_COLUMNS) is None, variant


def odd_rows(table: str) -> np.ndarray:
    """The rows with a cell of another shape than most cells of its array
    column (read one by one)."""
    odd = set()
    with open_table_ro(table) as tb:
        for col in tb.colnames():
            if tb.isscalarcol(col) or tb.coldatatype(col) == "record":
                continue
            shapes = [tb.getcell(col, row).shape for row in range(tb.nrows())]
            values, counts = np.unique(
                np.array([str(shape) for shape in shapes]), return_counts=True
            )
            common = values[np.argmax(counts)]
            odd.update(row for row, shape in enumerate(shapes) if str(shape) != common)
    return np.array(sorted(odd), dtype=np.int64)


def test_cell_shape_scan(pointing_ms, tmp_path, monkeypatch):
    """The shapes of the cells come from getcolshapestring, in chunks (no
    values): the common shape of a column (also when the first row has
    another), and every row with another shape or without a value (a call
    that fails is split); too many failing calls give up (the table is then
    built by the converter's code)."""
    ms = shutil.copytree(pointing_ms["regular"], str(tmp_path / "copy.ms"))
    table = os.path.join(ms, "POINTING")
    with tables.table(table, readonly=False, ack=False) as tb:
        nrows = tb.nrows()
        tb.putcell("DIRECTION", 0, np.zeros((2, 2)))  # (the first row)
        tb.putcell("ENCODER", nrows - 1, np.zeros(3))
        tb.addrows(1)  # (cells without a value)
        tb.putcell("TIME", nrows, TIME0)
    monkeypatch.setattr(bpt, "SHAPE_SCAN_ROWS", 100)
    calls = []
    shape_strings = bpt._shape_strings

    def spy(*args):
        calls.append(args[1:])
        return shape_strings(*args)

    monkeypatch.setattr(bpt, "_shape_strings", spy)
    index = bpt.read_pointing_index(table, DATA_COLUMNS)
    assert index.cell_shapes["DIRECTION"] == (1, 2)
    np.testing.assert_array_equal(index.odd_rows, [0, nrows - 1, nrows])
    assert {(col, start) for col, start, _ in calls} >= {
        (col, start)
        for col in ("DIRECTION", "TARGET", "ENCODER")
        for start in range(0, nrows + 1, 100)
    }
    monkeypatch.setattr(bpt, "SHAPE_SCAN_MAX_ERRORS", 2)
    assert bpt.read_pointing_index(table, DATA_COLUMNS) is None


def test_create_pointing_xds_generic_loader(pointing_ms):
    """generic_loader replaces the POINTING reads without interp_time (its
    arguments: the MS, time range, antennas and the data variable of every
    data column), never with interp_time; None reads POINTING as usual."""
    ms = pointing_ms["regular"]
    time_min_max = whole_time_range(ms)
    ant_names = antenna_names(range(NANTS))
    expected = create_pointing_xds(ms, ant_names, time_min_max, None)
    calls = []

    def loader(*args):
        calls.append(args)
        return None

    assert_xds_bit_identical(
        create_pointing_xds(ms, ant_names, time_min_max, None, generic_loader=loader),
        expected,
    )
    ((in_file, loader_time, loader_ants, columns),) = calls
    assert in_file == ms and loader_time == time_min_max
    np.testing.assert_array_equal(loader_ants, np.arange(NANTS))
    assert columns == {
        "DIRECTION": "POINTING_BEAM",
        "ENCODER": "POINTING_DISH_MEASURED",
        "OVER_THE_TOP": "POINTING_OVER_THE_TOP",
    }

    main_time = np.linspace(*time_min_max, 11) - 3_506_716_800.0
    interp_time = xr.DataArray(main_time, dims="time", coords={"time": main_time})
    expected = create_pointing_xds(ms, ant_names, time_min_max, interp_time)
    calls.clear()
    assert_xds_bit_identical(
        create_pointing_xds(
            ms, ant_names, time_min_max, interp_time, generic_loader=loader
        ),
        expected,
    )
    assert not calls

    # what the loader gives is used (and with a cache active, POINTING is not read)
    generic = rp.pointing_generic_xds(
        rp.read_pointing_columns(os.path.join(ms, "POINTING"), DATA_COLUMNS),
        time_min_max,
        np.arange(NANTS),
    )
    cache = SubtableCache(n_partitions=2)
    with activate_subtable_cache(cache):
        actual = create_pointing_xds(
            ms,
            ant_names,
            time_min_max,
            None,
            generic_loader=lambda *args: generic.copy(),
        )
    assert not cache.stats
    assert_xds_bit_identical(
        actual, create_pointing_xds(ms, ant_names, time_min_max, None)
    )
    empty = create_pointing_xds(
        ms, ant_names, time_min_max, None, generic_loader=lambda *args: xr.Dataset()
    )
    assert not empty.data_vars and not empty.attrs


# --- the lazy pointing_xds ---------------------------------------------------

# (time range as positions in the POINTING times, antennas), as in
# test_read_pointing.py: from the whole table to single samples
SELECTIONS = [
    ((0.0, 1.0), [0, 1, 2, 3, 4]),
    ((0.0, 1.0), [1, 3]),
    ((0.1, 0.7), [0, 2, 4]),
    ((0.5, 0.5), [0, 1, 2, 3, 4]),  # one time: projected to its neighbours
    ((0.31, 0.33), [0]),  # antenna 0 has many missing cells
    ((-0.5, 0.2), [2, 3]),  # partly before the first POINTING time
    ((0.9, 1.5), [4]),  # partly after the last one
    ((0.0, 1.0), [7]),  # no POINTING rows for this antenna
    ((0.0, 1.0), [3, 9]),
]


@pytest.fixture(autouse=True)
def _fresh_pointing_memos():
    """Every test starts (and leaves) with empty per-process memos."""
    bpt.clear_pointing_memos()
    yield
    bpt.clear_pointing_memos()


def pointing_times(ms: str) -> np.ndarray:
    with open_table_ro(os.path.join(ms, "POINTING")) as tb:
        return np.unique(tb.getcol("TIME"))


def time_range(ms: str, positions: tuple[float, float]) -> tuple:
    utimes = pointing_times(ms)
    span = utimes[-1] - utimes[0]
    return (
        np.float64(utimes[0] + positions[0] * span),
        np.float64(utimes[0] + positions[1] * span),
    )


def lazy_pointing(ms: str, ant_names: xr.DataArray, time_min_max: tuple):
    """create_pointing_xds as the backend's build calls it, with the
    placeholders replaced by lazy arrays: (pointing_xds, specs)."""
    specs = {}
    xds = create_pointing_xds(
        ms,
        ant_names,
        time_min_max,
        None,
        generic_loader=functools.partial(
            bpt.deferred_pointing_generic_xds, specs=specs
        ),
    )
    return bpt.lazy_pointing_xds(xds, specs, node="test_node"), specs


def opened_pointing(ms: str, ant_names: xr.DataArray, time_min_max: tuple):
    """create_pointing_xds as open_partition calls it: lazy variables, or
    (tables the index cannot describe) variables that build it again."""
    specs, context = {}, {}
    xds = create_pointing_xds(
        ms,
        ant_names,
        time_min_max,
        None,
        generic_loader=functools.partial(
            bpt.deferred_pointing_generic_xds, specs=specs, context=context
        ),
    )
    if specs:
        return bpt.lazy_pointing_xds(xds, specs, node="test_node")
    build = bpt.PointingBuild(
        ms,
        context["time_min_max"],
        context["antenna_ids"],
        tuple(str(name) for name in ant_names.values),
    )
    return bpt.rebuilt_pointing_xds(xds, build, node="test_node")


def pointing_array(var) -> "bpt.PointingColumnArray | None":
    """The PointingColumnArray under the wrappers of a variable, if any."""
    data = getattr(var, "variable", var)._data
    for _ in range(10):
        if isinstance(data, bpt.PointingColumnArray):
            return data
        data = getattr(data, "array", None)
        if data is None:
            return None
    return None


def orthogonal_index(values: np.ndarray, key: tuple) -> np.ndarray:
    """values indexed with key as xarray's isel does (each dimension on its
    own: at most one list in key)."""
    for axis in reversed(range(len(key))):
        index = [slice(None)] * values.ndim
        index[axis] = key[axis]
        values = values[tuple(index)]
    return values


def random_key(rng, shape: tuple[int, ...]) -> tuple:
    """A key of ints, slices (any step, also negative) and at most one list."""
    key = []
    list_axis = int(rng.integers(-1, len(shape)))
    for axis, size in enumerate(shape):
        kind = rng.integers(0, 4)
        if axis == list_axis:
            key.append(sorted(rng.choice(size, int(rng.integers(1, size + 1)))))
        elif kind == 0:
            key.append(int(rng.integers(-size, size)))
        else:
            a, b = sorted(int(x) for x in rng.integers(0, size + 1, 2))
            step = int(rng.choice([1, 1, 2, 3, -1, -2]))
            key.append(slice(a, b, step) if step > 0 else slice(b, a, step))
    return tuple(key)


def assert_bits_equal(actual: np.ndarray, expected: np.ndarray, what="") -> None:
    assert actual.dtype == expected.dtype, what
    assert actual.shape == expected.shape, what
    assert (
        np.ascontiguousarray(actual).tobytes()
        == np.ascontiguousarray(expected).tobytes()
    ), what


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("selection", SELECTIONS)
@pytest.mark.parametrize("variant", ["regular", "int_encoder", "no_encoder"])
def test_lazy_pointing_xds_identical(pointing_ms, variant, selection, cached):
    """The lazy pointing_xds is bit-identical to the converter's (uncached
    TaQL reads), with missing cells (fill values: NaN, True, the int of a
    NaN), duplicated (TIME, ANTENNA_ID) rows (the lowest row wins) and
    shuffled rows; with or without an active sub-table cache."""
    ms = pointing_ms[variant]
    positions, ants = selection
    time_min_max = time_range(ms, positions)
    ant_names = antenna_names(ants)
    expected = create_pointing_xds(ms, ant_names, time_min_max, None)
    with activate_subtable_cache(SubtableCache(n_partitions=2) if cached else None):
        actual, specs = lazy_pointing(ms, ant_names, time_min_max)
    if ants == [7]:
        assert not actual.data_vars and not specs
    else:
        assert set(specs) == set(actual.data_vars) == set(expected.data_vars)
        for name, var in actual.data_vars.items():
            assert pointing_array(var) is not None, name
            assert var.dtype == specs[name].dtype
    assert_xds_bit_identical(actual, expected)


def test_lazy_pointing_fill_values(pointing_ms):
    """The test tables do exercise the fills of missing cells and the
    duplicates, also for the int column."""
    ms = pointing_ms["int_encoder"]
    actual, _ = lazy_pointing(ms, antenna_names(range(NANTS)), time_range(ms, (0, 1)))
    missing = np.isnan(actual.POINTING_BEAM.values[..., 0])
    assert missing.any() and not missing.all()
    assert actual.POINTING_OVER_THE_TOP.values[missing].all()
    encoder = actual.POINTING_DISH_MEASURED.values
    assert encoder.dtype == np.int32
    with np.errstate(invalid="ignore"):
        fill = np.array(np.nan).astype(np.int32)
    assert (encoder[missing] == fill).all()
    with open_table_ro(os.path.join(ms, "POINTING")) as tb:
        pairs = np.stack([tb.getcol("TIME"), tb.getcol("ANTENNA_ID")], axis=1)
    assert len(np.unique(pairs, axis=0)) < len(pairs)


@pytest.mark.parametrize("sub_block_bytes", [None, 1])
@pytest.mark.parametrize("variant", ["regular", "int_encoder"])
def test_lazy_pointing_blocks(pointing_ms, monkeypatch, variant, sub_block_bytes):
    """Any selection (ints, slices with steps, lists) of the lazy variables
    gives the converter's values; also when every time is a sub-block."""
    if sub_block_bytes is not None:
        monkeypatch.setattr(bpt, "POINTING_SUB_BLOCK_BYTES", sub_block_bytes)
    ms = pointing_ms[variant]
    ant_names = antenna_names(range(NANTS))
    time_min_max = time_range(ms, (0.05, 0.9))
    expected = create_pointing_xds(ms, ant_names, time_min_max, None)
    actual, _ = lazy_pointing(ms, ant_names, time_min_max)
    rng = np.random.default_rng(7)
    for name, var in expected.data_vars.items():
        full = var.values
        for _ in range(30):
            key = random_key(rng, full.shape)
            assert_bits_equal(
                actual[name][key].values,
                orthogonal_index(full, key),
                f"{name}[{key}]",
            )


def build_array(var) -> "bpt.PointingBuildArray | None":
    """The PointingBuildArray under the wrappers of a variable, if any."""
    data = getattr(var, "variable", var)._data
    for _ in range(10):
        if isinstance(data, bpt.PointingBuildArray):
            return data
        data = getattr(data, "array", None)
        if data is None:
            return None
    return None


@pytest.mark.parametrize("variant", ["varying_target", *REBUILT_VARIANTS, "empty"])
def test_tables_of_other_cells(pointing_ms, variant):
    """Tables that are no plain grid of cells give the converter's
    pointing_xds (or its error), with lazy variables built again by the
    converter's code on read (TARGET cells of two shapes among the rows of
    the partition, zero-size ENCODER cells, a float OVER_THE_TOP); none for
    a table without rows."""
    ms = pointing_ms[variant]
    time_min_max = (np.float64(TIME0 - 1), np.float64(TIME0 + NTIMES))
    ant_names = antenna_names(range(NANTS))

    def build(opened: bool):
        try:
            if opened:
                return opened_pointing(ms, ant_names, time_min_max)
            return create_pointing_xds(ms, ant_names, time_min_max, None)
        except Exception as exc:
            return f"{type(exc).__name__}: {exc}"

    expected, actual = build(False), build(True)
    if isinstance(expected, str):
        assert actual == expected
        return
    if variant == "empty":
        assert not expected.data_vars and not actual.data_vars
        return
    assert all(build_array(v) is not None for v in actual.data_vars.values())
    assert_xds_bit_identical(actual, expected)
    rng = np.random.default_rng(3)
    for name, var in expected.data_vars.items():
        for _ in range(5):
            key = random_key(rng, var.shape)
            assert_bits_equal(
                actual[name][key].values, orthogonal_index(var.values, key)
            )
    blob = pickle.dumps(build_array(actual[name]))
    assert len(blob) < 4096
    array = pickle.loads(blob)
    key = xr.core.indexing.BasicIndexer((slice(None),) * len(array.shape))
    assert_bits_equal(array[key], expected[name].values, name)


def test_rebuilt_pointing_changed(pointing_ms, tmp_path):
    """A pointing_xds built again on read whose POINTING table changed shape
    since the open: MSv2ChangedError; a table gone: MSv2ReadError."""
    ms = shutil.copytree(pointing_ms["float_over_the_top"], str(tmp_path / "copy.ms"))
    ant_names = antenna_names(range(NANTS))
    actual = opened_pointing(ms, ant_names, whole_time_range(ms))
    assert build_array(actual.POINTING_BEAM) is not None
    _add_pointing_row(ms, time_offset=0.001)  # (one more time in the range)
    with pytest.raises(MSv2ChangedError, match="POINTING_BEAM"):
        _ = actual.POINTING_BEAM.values
    shutil.rmtree(os.path.join(ms, "POINTING"))
    with pytest.raises(MSv2ReadError, match="test_node"):
        _ = actual.POINTING_BEAM.values


@pytest.mark.parametrize("variant", ["float_over_the_top", "varying_direction"])
def test_rebuilt_pointing_with_other_times_raises(pointing_ms, tmp_path, variant):
    """POINTING TIME rewritten in place after the open, keeping the shape of
    the grid (every time shifted by half a sample): the pointing_xds built
    again on read has other time coordinates than the one opened, so a read
    raises MSv2ChangedError (its values would be served under the opened
    coordinates), as for lazily read pointing_xds; also with the memos
    emptied. Values rewritten in place (the same times) are read as they
    are now. (A table the index cannot describe, and a partition with cells
    of another shape.)"""
    ms = shutil.copytree(pointing_ms[variant], str(tmp_path / "copy.ms"))
    ant_names = antenna_names(range(NANTS))
    time_min_max = whole_time_range(ms)
    actual = opened_pointing(ms, ant_names, time_min_max)
    name = next(iter(actual.data_vars))
    assert build_array(actual[name]) is not None
    with tables.table(os.path.join(ms, "POINTING"), readonly=False, ack=False) as tb:
        value = tb.getcell("OVER_THE_TOP", 0)
        boolean = isinstance(value, bool | np.bool_)
        tb.putcell("OVER_THE_TOP", 0, (not value) if boolean else value + 1)
    assert_bits_equal(
        actual[name].values,
        create_pointing_xds(ms, ant_names, time_min_max, None)[name].values,
    )
    with tables.table(os.path.join(ms, "POINTING"), readonly=False, ack=False) as tb:
        tb.putcol("TIME", tb.getcol("TIME") + 0.5)
    for _ in range(2):
        with pytest.raises(MSv2ChangedError, match="times or antennas"):
            _ = actual[name].values
        bpt.clear_pointing_memos()


def _window_without(ms: str, rows: np.ndarray) -> tuple:
    """The widest time range (inside, 3 samples from either end) between the
    times of POINTING rows ``rows``."""
    with open_table_ro(os.path.join(ms, "POINTING")) as tb:
        odd_times = np.unique(tb.getcol("TIME")[rows])
    utimes = pointing_times(ms)
    edges = np.concatenate([[utimes[0] - 1], odd_times, [utimes[-1] + 1]])
    gap = int(np.argmax(np.diff(edges)))
    return (np.float64(edges[gap] + 0.15), np.float64(edges[gap + 1] - 0.15))


@pytest.mark.parametrize("variant", ["varying_direction", "varying_target"])
def test_cells_of_another_shape_as_the_converter(pointing_ms, variant):
    """Cells of two shapes in an array column (DIRECTION, a data column;
    TARGET, not one), where the converter reads every partition on its own
    and leaves out (getcol, 1,000 rows or more) or pads (fewer) the column
    whose cells vary in its rows: a partition whose rows include cells of
    the other shape gets the converter's pointing_xds (built by its code:
    without POINTING_BEAM for DIRECTION, or its error), the others are read
    lazily; both bit-identical to the converter's."""
    ms = pointing_ms[variant]
    index = bpt.read_pointing_index(os.path.join(ms, "POINTING"), DATA_COLUMNS)
    assert index.odd_rows.size
    ant_names = antenna_names(range(NANTS))

    def build(opened: bool, time_min_max, names=ant_names):
        try:
            if opened:
                return opened_pointing(ms, names, time_min_max)
            return create_pointing_xds(ms, names, time_min_max, None)
        except Exception as exc:
            return f"{type(exc).__name__}: {exc}"

    for time_min_max, names, lazy in (
        (whole_time_range(ms), ant_names, False),  # (1,000 rows or more)
        (_window_without(ms, index.odd_rows), ant_names, True),
    ):
        expected, actual = (
            build(False, time_min_max, names),
            build(True, time_min_max, names),
        )
        assert not isinstance(expected, str)
        assert_xds_bit_identical(actual, expected)
        assert actual.data_vars
        find = pointing_array if lazy else build_array
        assert all(find(var) is not None for var in actual.data_vars.values())
    if variant == "varying_direction":
        assert "POINTING_BEAM" not in build(False, whole_time_range(ms))
    # fewer than 1,000 rows, with a cell of the other shape: padded by the
    # converter (and its error, here)
    with open_table_ro(os.path.join(ms, "POINTING")) as tb:
        row = int(index.odd_rows[0])
        time, ant = tb.getcell("TIME", row), tb.getcell("ANTENNA_ID", row)
    few = (np.float64(time - 1), np.float64(time + 1))
    names = antenna_names([ant])
    expected, actual = build(False, few, names), build(True, few, names)
    if isinstance(expected, str):
        assert actual == expected
    else:
        assert_xds_bit_identical(actual, expected)
        assert all(build_array(var) is not None for var in actual.data_vars.values())


@pytest.mark.parametrize("variant", ["rich", "single_dish"])
def test_engine_with_cells_of_another_shape(ms_copy, tmp_path, variant):
    """The engine with a POINTING DIRECTION cell of another shape in the
    time range of the partitions gives the converter's processing set, or
    its error (here: the partitions have fewer than 1,000 POINTING rows,
    which the converter pads and then fails on; the engine raises it, also
    for every partition with on_partition_error="skip")."""
    from xradio.measurement_set import (
        convert_msv2_to_processing_set,
        open_processing_set,
    )
    from xradio.testing.measurement_set.equivalence import assert_nodes_identical

    ms = ms_copy(variant)
    with tables.table(ms, ack=False) as main_tb:
        main_time = np.median(main_tb.getcol("TIME"))
    with tables.table(os.path.join(ms, "POINTING"), readonly=False, ack=False) as tb:
        # (a row in the time range of the partitions)
        row = int(np.argmin(np.abs(tb.getcol("TIME") - main_time)))
        cell = tb.getcell("DIRECTION", row)
        tb.putcell("DIRECTION", row, np.vstack([cell, cell * 0 + 1e-6]))
    out = str(tmp_path / "converted.ps.zarr")
    try:
        convert_msv2_to_processing_set(ms, out)
    except ValueError as exc:
        with pytest.raises(RuntimeError, match=re.escape(str(exc))):
            xr.open_datatree(
                ms,
                engine=MSv2BackendEntrypoint,
                partition_cache="off",
                on_partition_error="raise",
            )
        return
    tree = xr.open_datatree(
        ms, engine=MSv2BackendEntrypoint, chunks={}, partition_cache="off"
    )
    assert_nodes_identical(tree, open_processing_set(out))


def test_rebuilt_pointing_variables_share_a_build(pointing_ms, tmp_path, monkeypatch):
    """The variables of a pointing_xds built again on read share one build
    (POINTING_BUILD_MEMO), also when dask threads read them at once; a
    POINTING table written since (its fingerprint) is built again, with its
    new values; a build larger than the memo's bound is not kept."""
    ms = shutil.copytree(pointing_ms["float_over_the_top"], str(tmp_path / "copy.ms"))
    ant_names = antenna_names(range(NANTS))
    time_min_max = whole_time_range(ms)
    actual = opened_pointing(ms, ant_names, time_min_max)
    memo = bpt.POINTING_BUILD_MEMO
    assert len(actual.data_vars) == 3 and memo.stats["builds"] == 0
    with dask.config.set(scheduler="threads", num_workers=4):
        values = actual.chunk().compute()
    assert memo.stats["builds"] == 1
    assert_xds_bit_identical(
        values, create_pointing_xds(ms, ant_names, time_min_max, None)
    )
    with tables.table(os.path.join(ms, "POINTING"), readonly=False, ack=False) as tb:
        tb.putcell("OVER_THE_TOP", 0, 5.0)
    assert_xds_bit_identical(
        actual.compute(), create_pointing_xds(ms, ant_names, time_min_max, None)
    )
    assert memo.stats["builds"] == 2
    memo.clear()
    monkeypatch.setattr(memo, "max_bytes", 16)
    actual.compute()
    assert memo.stats["builds"] == 3 and len(memo) == 0


def test_lazy_pointing_same_errors(pointing_ms):
    """POINTING cells the converter cannot convert (two polynomial terms)
    make the lazy build fail where the converter fails, with its error."""
    ms = pointing_ms["two_polynomials"]
    time_min_max = time_range(ms, (0, 1))
    ant_names = antenna_names(range(NANTS))
    with pytest.raises(ValueError) as expected:
        create_pointing_xds(ms, ant_names, time_min_max, None)
    specs = {}
    with pytest.raises(ValueError) as actual:
        create_pointing_xds(
            ms,
            ant_names,
            time_min_max,
            None,
            generic_loader=functools.partial(
                bpt.deferred_pointing_generic_xds, specs=specs
            ),
        )
    assert str(actual.value) == str(expected.value)
    assert specs  # (the lazy path was taken)


def test_open_reads_no_pointing_values(pointing_ms, monkeypatch):
    """Building the lazy pointing_xds reads TIME and ANTENNA_ID and one cell
    of every data column (its dtype), nothing else of the data columns (no
    check of their cells), and pivots nothing. Computing reads the values."""
    ms = pointing_ms["regular"]
    reads, row_reads, pivots = [], [], []
    read_column_rows, read_rows = bpt.read_column_rows, rp.read_rows

    def spy_column_rows(table, col, rows, *args, **kwargs):
        reads.append((col, len(rows)))
        return read_column_rows(table, col, rows, *args, **kwargs)

    def spy_rows(table, col, rows, out, *args, **kwargs):
        row_reads.append(col)
        return read_rows(table, col, rows, out, *args, **kwargs)

    def spy_pivot(*args):
        pivots.append(args[-1])
        return rp.pivot_time_antenna(*args)

    monkeypatch.setattr(bpt, "read_column_rows", spy_column_rows)
    monkeypatch.setattr(rp, "read_rows", spy_rows)
    monkeypatch.setattr(bpt, "pivot_time_antenna", spy_pivot)
    ant_names = antenna_names(range(NANTS))
    actual, specs = lazy_pointing(ms, ant_names, time_range(ms, (0, 1)))
    assert set(specs) == set(actual.data_vars)
    with open_table_ro(os.path.join(ms, "POINTING")) as tb:
        nrows = tb.nrows()
    assert sorted(reads) == sorted(
        [(col, 1) for col in DATA_COLUMNS] + [("ANTENNA_ID", nrows), ("TIME", nrows)]
    )
    assert not row_reads and not pivots
    reads.clear()
    actual.compute()
    assert len(pivots) == len(DATA_COLUMNS)
    assert all(n > 1 for col, n in reads if col in DATA_COLUMNS)


def test_selections_read_only_their_rows(pointing_ms, monkeypatch):
    """Lists and steps along time and antenna read only the rows of the
    selected cells."""
    ms = pointing_ms["regular"]
    ant_names = antenna_names(range(NANTS))
    time_min_max = time_range(ms, (0, 1))
    expected = create_pointing_xds(ms, ant_names, time_min_max, None)
    actual, _ = lazy_pointing(ms, ant_names, time_min_max)
    grid = expected.POINTING_BEAM
    reads = []
    read_sorted_rows = bpt.PointingColumnArray._read_sorted_rows

    def spy(self, table, rows, col, *args):
        reads.append((col, rows.size))
        return read_sorted_rows(self, table, rows, col, *args)

    monkeypatch.setattr(bpt.PointingColumnArray, "_read_sorted_rows", spy)
    nt = grid.sizes["time_pointing"]
    for isel in (
        {"time_pointing": [0, nt - 1]},
        {"time_pointing": slice(None, None, 50), "antenna_name": [4, 1]},
        {"time_pointing": slice(nt - 1, None, -97), "antenna_name": 2},
    ):
        reads.clear()
        assert_bits_equal(
            actual.POINTING_BEAM.isel(isel).values, grid.isel(isel).values, str(isel)
        )
        cells = grid.isel(isel).values[..., 0]
        n_values = [n for col, n in reads if col == "DIRECTION"]
        assert n_values == [int(np.isfinite(cells).sum())], isel
        assert {col for col, _ in reads} == {"DIRECTION"}  # (unchanged: no checks)


def test_open_keeps_no_pointing_values(tmp_path):
    """What an open keeps grows with the index (TIME, ANTENNA_ID, row
    numbers: the memo) and the coordinates, not with the values."""
    ms = make_pointing_ms(str(tmp_path / "large.ms"), "regular", ntimes=40_000)
    ant_names = antenna_names(range(NANTS))
    time_min_max = (np.float64(TIME0 - 1), np.float64(TIME0 + 40_000))
    gc.collect()
    tracemalloc.start()
    try:
        before = tracemalloc.get_traced_memory()[0]
        actual, _ = lazy_pointing(ms, ant_names, time_min_max)
        gc.collect()
        kept = tracemalloc.get_traced_memory()[0] - before
    finally:
        tracemalloc.stop()
    expected = create_pointing_xds(ms, ant_names, time_min_max, None)
    values = sum(var.nbytes for var in expected.data_vars.values())
    coords = sum(var.nbytes for var in actual.coords.values())
    index = bpt.POINTING_INDEX_MEMO.nbytes
    assert values > 6 * 2**20
    assert kept - index <= coords + 2**20, (kept, index, coords)
    assert kept < values / 1.5, (kept, values)
    assert_xds_bit_identical(actual, expected)


def test_pickled_arrays_read_after_memo_loss(pointing_ms):
    """The arrays pickle to their description (a few kB); in a process
    without the memos (cleared here) the index is read again, once, and the
    partition's rows are selected and checked again."""
    ms = pointing_ms["regular"]
    ant_names = antenna_names([0, 2, 3])
    time_min_max = time_range(ms, (0.2, 0.8))
    expected = create_pointing_xds(ms, ant_names, time_min_max, None)
    actual, _ = lazy_pointing(ms, ant_names, time_min_max)
    blobs = {}
    for name, var in actual.data_vars.items():
        blobs[name] = pickle.dumps(pointing_array(var))
        assert len(blobs[name]) < 4096, name
    bpt.clear_pointing_memos()
    for name, blob in blobs.items():
        array = pickle.loads(blob)
        key = xr.core.indexing.BasicIndexer((slice(None),) * len(array.shape))
        assert_bits_equal(array[key], expected[name].values, name)
    assert bpt.POINTING_INDEX_MEMO.stats["reads"] == 1
    assert bpt.POINTING_SELECTION_MEMO.stats["selections"] == 1


class GetcolOnlyTable:
    """A python-casacore table with the read API of the casatools shim (no
    getcolnp, getcolslicenp or selectrows), recording its getcol calls."""

    def __init__(self, table):
        self._table = table
        self.getcol_calls = []

    def __getattr__(self, name):
        if name in ("getcolnp", "getcolslicenp", "selectrows"):
            raise AttributeError(name)
        return getattr(self._table, name)

    def getcol(self, col, startrow=0, nrow=-1, rowincr=1):
        values = self._table.getcol(col, startrow, nrow, rowincr)
        self.getcol_calls.append((col, startrow, len(values)))
        return values


@pytest.mark.parametrize(
    "max_gap, window_bytes", [(1024, 16 * 2**20), (1, 16 * 2**20), (1024, 64)]
)
def test_reads_without_in_place_reads(pointing_ms, monkeypatch, max_gap, window_bytes):
    """With the read API of the casatools shim (getcol only), the rows of a
    block that are fragmented (one antenna of shuffled rows) are read in
    windows of consecutive rows, not one getcol per run; same values; every
    read holds casatools_serialized."""
    monkeypatch.setattr(bpt, "_WINDOW_MAX_GAP", max_gap)
    monkeypatch.setattr(bpt, "_WINDOW_BYTES", window_bytes)
    ms = pointing_ms["regular"]
    ant_names = antenna_names(range(NANTS))
    time_min_max = time_range(ms, (0, 1))
    expected = create_pointing_xds(ms, ant_names, time_min_max, None)
    actual, _ = lazy_pointing(ms, ant_names, time_min_max)
    open_table_ro = bpt.open_table_ro
    used, entered = [], []

    @contextlib.contextmanager
    def getcol_only(path):
        with open_table_ro(path) as table:
            used.append(GetcolOnlyTable(table))
            yield used[-1]

    @contextlib.contextmanager
    def serialized():
        entered.append(True)
        yield

    monkeypatch.setattr(bpt, "open_table_ro", getcol_only)
    monkeypatch.setattr(bpt, "casatools_serialized", serialized)
    n_reads = 0
    for name in expected.data_vars:
        for ant in range(NANTS):
            assert_bits_equal(
                actual[name].isel(antenna_name=ant).values,
                expected[name].isel(antenna_name=ant).values,
                f"{name} antenna {ant}",
            )
            n_reads += 1
    assert len(entered) == len(used) == n_reads
    calls = [
        call
        for table in used
        for call in table.getcol_calls
        if call[0] not in ("TIME", "ANTENNA_ID")  # (the checks of the rows)
    ]
    if (max_gap, window_bytes) == (1024, 16 * 2**20):
        assert len(calls) <= 2 * n_reads  # (the whole column is never one call)
    else:
        assert len(calls) > 2 * n_reads


def _add_pointing_row(ms: str, time_offset: float = 0.0) -> None:
    with tables.table(os.path.join(ms, "POINTING"), readonly=False, ack=False) as tb:
        tb.addrows(1)
        row = tb.nrows() - 1
        for col in tb.colnames():
            if col not in ("NAME",):
                tb.putcell(col, row, tb.getcell(col, 0))
        tb.putcell("TIME", row, tb.getcell("TIME", 0) + time_offset)


def test_changed_pointing_rows_raise(pointing_ms, tmp_path):
    """POINTING rows added after the open: MSv2ChangedError on read, in this
    process (memo) and in another one (index read again)."""
    ms = shutil.copytree(pointing_ms["regular"], str(tmp_path / "copy.ms"))
    actual, _ = lazy_pointing(ms, antenna_names(range(NANTS)), time_range(ms, (0, 1)))
    _add_pointing_row(ms)
    with pytest.raises(MSv2ChangedError, match="rows"):
        _ = actual.POINTING_BEAM.values
    bpt.clear_pointing_memos()
    with pytest.raises(MSv2ChangedError, match="rows"):
        _ = actual.POINTING_BEAM.values


def test_changed_pointing_times_raise(pointing_ms, tmp_path):
    """A TIME changed in place (same number of rows): a read checks the TIME
    and ANTENNA_ID of its rows, reads the index again and finds another
    selection: MSv2ChangedError, with the index of the open in the memo as
    in a process that reads it again."""
    ms = shutil.copytree(pointing_ms["regular"], str(tmp_path / "copy.ms"))
    ant_names = antenna_names(range(NANTS))
    time_min_max = time_range(ms, (0, 1))
    actual, _ = lazy_pointing(ms, ant_names, time_min_max)
    with tables.table(os.path.join(ms, "POINTING"), readonly=False, ack=False) as tb:
        tb.putcell("TIME", 3, tb.getcell("TIME", 3) + 0.01)
    with pytest.raises(MSv2ChangedError, match="times, antennas or rows"):
        _ = actual.POINTING_BEAM.values
    bpt.clear_pointing_memos()
    with pytest.raises(MSv2ChangedError, match="times, antennas or rows"):
        _ = actual.POINTING_DISH_MEASURED.values


def test_swapped_pointing_antennas_are_read(pointing_ms, tmp_path):
    """The ANTENNA_ID of two rows of one time swapped in place (the same
    times, antennas and rows of the partition): with the index of the open in
    the memo or read again, the values are those of the changed table."""
    ms = shutil.copytree(pointing_ms["regular"], str(tmp_path / "copy.ms"))
    ant_names = antenna_names(range(NANTS))
    time_min_max = time_range(ms, (0, 1))
    with tables.table(os.path.join(ms, "POINTING"), readonly=False, ack=False) as tb:
        time, ant = tb.getcol("TIME"), tb.getcol("ANTENNA_ID")
    rows = next(
        np.flatnonzero(time == t)[:2]
        for t in np.unique(time)
        if np.unique(ant[time == t]).size >= 2
        and np.unique(ant[time == t]).size == (time == t).sum()
    )
    for clear in (False, True):
        bpt.clear_pointing_memos()
        actual, _ = lazy_pointing(ms, ant_names, time_min_max)
        with tables.table(
            os.path.join(ms, "POINTING"), readonly=False, ack=False
        ) as tb:
            values = tb.getcol("ANTENNA_ID")
            values[rows] = values[rows[::-1]]
            tb.putcol("ANTENNA_ID", values)
        if clear:
            bpt.clear_pointing_memos()
        assert_xds_bit_identical(
            actual, create_pointing_xds(ms, ant_names, time_min_max, None)
        )


def test_pointing_rows_are_checked_only_after_writes(
    pointing_ms, tmp_path, monkeypatch
):
    """The TIME and ANTENNA_ID of the rows read are checked one by one only
    when POINTING was written since the open (its fingerprint), or while a
    handle of this process has it open for writing."""
    ms = shutil.copytree(pointing_ms["regular"], str(tmp_path / "copy.ms"))
    ant_names = antenna_names(range(NANTS))
    time_min_max = time_range(ms, (0, 1))
    expected = create_pointing_xds(ms, ant_names, time_min_max, None)
    actual, _ = lazy_pointing(ms, ant_names, time_min_max)
    checks = []
    rows_moved = bpt.PointingColumnArray._rows_moved

    def spy(self, *args):
        checks.append(args[-1].size)
        return rows_moved(self, *args)

    monkeypatch.setattr(bpt.PointingColumnArray, "_rows_moved", spy)
    assert_bits_equal(actual.POINTING_BEAM.values, expected.POINTING_BEAM.values)
    assert checks == []
    table = os.path.join(ms, "POINTING")
    writer = tables.table(table, readonly=False, ack=False)
    try:
        assert_bits_equal(actual.POINTING_BEAM.values, expected.POINTING_BEAM.values)
        assert len(checks) == 1
    finally:
        writer.close()
    checks.clear()
    with tables.table(table, readonly=False, ack=False) as tb:
        tb.putcol("TRACKING", tb.getcol("TRACKING"))
    assert_bits_equal(actual.POINTING_BEAM.values, expected.POINTING_BEAM.values)
    assert len(checks) == 1


def test_changed_pointing_values_are_read(pointing_ms, tmp_path):
    """Values rewritten after the open are read as they are now (as for MAIN),
    and the next open reads the index again (its fingerprint changed)."""
    ms = shutil.copytree(pointing_ms["regular"], str(tmp_path / "copy.ms"))
    ant_names = antenna_names(range(NANTS))
    time_min_max = time_range(ms, (0, 1))
    actual, _ = lazy_pointing(ms, ant_names, time_min_max)
    lazy_pointing(ms, ant_names, time_min_max)
    assert bpt.POINTING_INDEX_MEMO.stats["reads"] == 1
    with tables.table(os.path.join(ms, "POINTING"), readonly=False, ack=False) as tb:
        tb.putcol("DIRECTION", tb.getcol("DIRECTION") * 2)
    assert_xds_bit_identical(
        actual, create_pointing_xds(ms, ant_names, time_min_max, None)
    )
    lazy_pointing(ms, ant_names, time_min_max)
    assert bpt.POINTING_INDEX_MEMO.stats["reads"] == 2


def test_unreadable_pointing_raises_read_error(pointing_ms, tmp_path):
    """A read that fails names the column, variable and node."""
    ms = shutil.copytree(pointing_ms["regular"], str(tmp_path / "copy.ms"))
    actual, _ = lazy_pointing(ms, antenna_names(range(NANTS)), time_range(ms, (0, 1)))
    shutil.rmtree(os.path.join(ms, "POINTING"))
    with pytest.raises(
        MSv2ReadError, match="ENCODER.*POINTING_DISH_MEASURED.*test_node"
    ):
        _ = actual.POINTING_DISH_MEASURED.values


def test_threads_read_the_index_once(pointing_ms):
    """Threads that miss the memos read the index and select the rows once."""
    ms = pointing_ms["regular"]
    ant_names = antenna_names(range(NANTS))
    time_min_max = time_range(ms, (0, 1))
    expected = create_pointing_xds(ms, ant_names, time_min_max, None)
    actual, _ = lazy_pointing(ms, ant_names, time_min_max)
    bpt.clear_pointing_memos()
    names = list(actual.data_vars) * 3
    results, errors = [None] * len(names), []
    barrier = threading.Barrier(len(names))

    def read(i):
        try:
            barrier.wait()
            results[i] = actual[names[i]].values
        except Exception as exc:  # pragma: no cover - reported below
            errors.append(exc)

    threads = [threading.Thread(target=read, args=(i,)) for i in range(len(names))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors
    for name, values in zip(names, results, strict=True):
        assert_bits_equal(values, expected[name].values, name)
    assert bpt.POINTING_INDEX_MEMO.stats["reads"] == 1
    assert bpt.POINTING_SELECTION_MEMO.stats["selections"] == 1


def wait_child(pid: int, seconds: float = 60.0) -> int | None:
    """The exit code of a forked child; a child still running after
    ``seconds`` (e.g. deadlocked) is killed: None."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        done, status = os.waitpid(pid, os.WNOHANG)
        if done:
            return os.waitstatus_to_exitcode(status)
        time.sleep(0.05)
    os.kill(pid, 9)
    os.waitpid(pid, 0)
    return None


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs os.fork")
@pytest.mark.filterwarnings("ignore:This process .* is multi-threaded")
def test_fork_child_gets_empty_memos(pointing_ms):
    """A child forked while another thread holds the memos' locks gets empty
    memos with free locks, and reads the values."""
    ms = pointing_ms["regular"]
    ant_names = antenna_names(range(NANTS))
    time_min_max = time_range(ms, (0, 1))
    expected = create_pointing_xds(ms, ant_names, time_min_max, None)
    actual, _ = lazy_pointing(ms, ant_names, time_min_max)
    actual.compute()
    assert len(bpt.POINTING_INDEX_MEMO) and len(bpt.POINTING_SELECTION_MEMO)
    memos = (bpt.POINTING_INDEX_MEMO, bpt.POINTING_SELECTION_MEMO)
    with memos[0]._lock, memos[1]._lock:
        pid = os.fork()
        if pid == 0:  # child
            code = 1
            try:
                if all(memo._lock.acquire(timeout=5) for memo in memos):
                    for memo in memos:
                        memo._lock.release()
                    empty = not any(len(memo) for memo in memos)
                    values = actual.POINTING_BEAM.values
                    same = values.tobytes() == expected.POINTING_BEAM.values.tobytes()
                    code = 0 if empty and same else 2
            finally:
                os._exit(code)
    assert wait_child(pid) == 0


# --- the engine ----------------------------------------------------------------


@pytest.mark.parametrize("variant", ["rich", "single_dish"])
def test_engine_pointing_is_lazy(backend_ms, variant, monkeypatch):
    """xr.open_datatree with the engine reads no pointing value; every
    pointing_xds data variable is a PointingColumnArray, read on access
    (with pointing_interpolate: eager, as the converter builds it)."""
    reads = []
    read_selection = bpt.PointingColumnArray._read_selection

    def spy(self, selections):
        reads.append(selections)
        return read_selection(self, selections)

    monkeypatch.setattr(bpt.PointingColumnArray, "_read_selection", spy)
    ms = backend_ms(variant)
    tree = xr.open_datatree(
        ms, engine=MSv2BackendEntrypoint, chunks=None, partition_cache="off"
    )
    pointing = [
        tree[name]["pointing_xds"].to_dataset(inherit=False)
        for name in tree.children
        if "pointing_xds" in tree[name].children
    ]
    assert pointing and not reads
    for xds in pointing:
        assert set(xds.data_vars) == {"POINTING_BEAM"}
        assert all(pointing_array(var) is not None for var in xds.data_vars.values())
        assert np.isfinite(xds.POINTING_BEAM.values).any()
    assert len(reads) == len(pointing)

    tree = xr.open_datatree(
        ms,
        engine=MSv2BackendEntrypoint,
        chunks=None,
        partition_cache="off",
        pointing_interpolate=True,
    )
    interpolated = [
        tree[name]["pointing_xds"].to_dataset(inherit=False)
        for name in tree.children
        if "pointing_xds" in tree[name].children
    ]
    assert interpolated
    for xds in interpolated:
        assert "time" in xds.dims
        assert all(pointing_array(var) is None for var in xds.data_vars.values())


def test_pointing_of_a_changed_partition_raises(ms_copy, monkeypatch):
    """The pointing_xds of a partition whose MAIN rows changed since the open
    (rows of its first time moved to another partition: its time range and
    antennas select its POINTING rows) raises MSv2ChangedError, like its main
    data variables, whichever kind of lazy array it has; the pointing_xds of
    a partition the change does not touch reads its values."""
    from xradio.measurement_set._utils._msv2.partition_queries import (
        create_partitions_with_main_rows,
    )

    ms = ms_copy("rich")
    partitions, _ = create_partitions_with_main_rows(ms, [])
    cal = "CALIBRATE_PHASE#ON_SOURCE"

    def node_of(ddi, obs_mode):
        (idx,) = (
            i
            for i, info in enumerate(partitions)
            if info["DATA_DESC_ID"] == [ddi] and info["OBS_MODE"] == [obs_mode]
        )
        return sorted(tree.children)[idx]

    for kind in ("lazy", "rebuilt"):
        if kind == "rebuilt":  # (built by the converter's code on read)
            monkeypatch.setattr(bpt, "read_pointing_index", lambda *a, **k: None)
        bpt.clear_pointing_memos()
        tree = xr.open_datatree(
            ms, engine=MSv2BackendEntrypoint, chunks=None, partition_cache="off"
        )
        changed = tree[node_of(0, cal)]["pointing_xds"].to_dataset(inherit=False)
        other = tree[node_of(3, cal)]["pointing_xds"].to_dataset(inherit=False)
        find = pointing_array if kind == "lazy" else build_array
        assert find(changed.POINTING_BEAM) is not None
        expected = other.POINTING_BEAM.values
        with tables.table(ms, readonly=False, ack=False) as main_tb:
            state = main_tb.getcol("STATE_ID")
            main_tb.putcol("STATE_ID", np.where(np.arange(state.size) < 10, 1, state))
        with pytest.raises(MSv2ChangedError, match="another partition now"):
            changed.POINTING_BEAM.values  # noqa: B018
        assert_bits_equal(other.POINTING_BEAM.values, expected)
        with tables.table(ms, readonly=False, ack=False) as main_tb:
            main_tb.putcol("STATE_ID", state)


def test_engine_pointing_is_lazy_over_the_cache_limit(backend_ms, monkeypatch):
    """POINTING tables larger than what the converter's sub-table cache holds
    (POINTING_MAX_CACHED_INDEX_BYTES, about 16.7 million rows) are read
    lazily too: the backend reads its own index, whatever its size."""
    monkeypatch.setattr(rp, "POINTING_MAX_CACHED_INDEX_BYTES", 16)
    ms = backend_ms("rich")
    tree = xr.open_datatree(
        ms, engine=MSv2BackendEntrypoint, chunks=None, partition_cache="off"
    )
    pointing = [
        tree[name]["pointing_xds"].to_dataset(inherit=False)
        for name in tree.children
        if "pointing_xds" in tree[name].children
    ]
    assert pointing
    for xds in pointing:
        assert all(pointing_array(var) is not None for var in xds.data_vars.values())
        assert np.isfinite(xds.POINTING_BEAM.values).any()
