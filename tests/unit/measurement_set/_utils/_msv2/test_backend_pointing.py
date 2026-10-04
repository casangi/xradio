"""
The lazy pointing_xds of the MSv2 xarray backend (``backend_pointing.py``)
and the converter's hooks for it (``create_pointing_xds(generic_loader=)``,
``read_pointing_columns(keep_data=)``): values bit-identical to the
converter's pointing_xds (duplicated rows, missing cells and their fill
values, every block), no POINTING value read or kept at open, staleness,
pickling, threads and fork.
"""

import contextlib
import functools
import gc
import os
import pickle
import shutil
import threading
import time
import tracemalloc

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
)
# Variants whose pointing_xds the converter's sub-table cache holds (the
# backend reads them lazily); the others are read eagerly
LAZY_VARIANTS = ("regular", "int_encoder", "no_encoder", "two_polynomials")


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


def test_read_pointing_columns_keep_data(pointing_ms):
    """keep_data=False keeps no data column (they are still checked), and the
    selections of the index give the pointing_xds of the cached columns."""
    table = os.path.join(pointing_ms["regular"], "POINTING")
    kept = rp.read_pointing_columns(table, DATA_COLUMNS)
    index = rp.read_pointing_columns(table, DATA_COLUMNS, keep_data=False)
    assert kept.data is not None and index.data is None
    assert index.data_columns == kept.data_columns == DATA_COLUMNS
    for name in ("time", "antenna_id", "row"):
        np.testing.assert_array_equal(getattr(index, name), getattr(kept, name))
    assert index.nbytes == kept.time.nbytes + kept.antenna_id.nbytes + kept.row.nbytes
    ants = np.arange(NANTS)
    assert_xds_bit_identical(
        rp.pointing_generic_xds(index, whole_time_range(table), ants),
        rp.pointing_generic_xds(kept, whole_time_range(table), ants),
    )
    for variant in ("varying_target", "empty", "empty_encoder", "float_over_the_top"):
        table = os.path.join(pointing_ms[variant], "POINTING")
        assert rp.read_pointing_columns(table, DATA_COLUMNS, keep_data=False) is None


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


@pytest.mark.parametrize(
    "variant", ["varying_target", "empty", "empty_encoder", "float_over_the_top"]
)
def test_eager_pointing_tables(pointing_ms, variant):
    """Tables that the converter's cache does not hold (the backend could not
    reproduce them exactly) are read eagerly, as by the converter: same
    result or same error, no lazy variable."""
    ms = pointing_ms[variant]
    time_min_max = (np.float64(TIME0 - 1), np.float64(TIME0 + NTIMES))
    ant_names = antenna_names(range(NANTS))

    def build(lazy: bool):
        try:
            if lazy:
                return lazy_pointing(ms, ant_names, time_min_max)
            return create_pointing_xds(ms, ant_names, time_min_max, None), {}
        except Exception as exc:
            return f"{type(exc).__name__}: {exc}", {}

    (expected, _), (actual, specs) = build(False), build(True)
    assert not specs
    if isinstance(expected, str):
        assert actual == expected
    else:
        assert_xds_bit_identical(actual, expected)
        assert all(pointing_array(v) is None for v in actual.data_vars.values())
    assert bpt.POINTING_INDEX_MEMO.stats["reads"] == 1


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
    """Building the lazy pointing_xds reads one cell of every data column
    (its dtype and shape) and checks the cells in bounded reads (as the
    converter's cache does); it pivots nothing. Computing reads the values."""
    ms = pointing_ms["regular"]
    reads, checks, pivots = [], [], []
    read_column_rows, read_rows = rp.read_column_rows, rp.read_rows

    def spy_column_rows(table, col, rows, *args, **kwargs):
        reads.append((col, len(rows)))
        return read_column_rows(table, col, rows, *args, **kwargs)

    def spy_rows(table, col, rows, out, *args, **kwargs):
        checks.append((col, out.size))
        return read_rows(table, col, rows, out, *args, **kwargs)

    def spy_pivot(*args):
        pivots.append(args[-1])
        return rp.pivot_time_antenna(*args)

    monkeypatch.setattr(rp, "read_column_rows", spy_column_rows)
    monkeypatch.setattr(rp, "read_rows", spy_rows)
    monkeypatch.setattr(bpt, "read_column_rows", spy_column_rows)
    monkeypatch.setattr(bpt, "pivot_time_antenna", spy_pivot)
    ant_names = antenna_names(range(NANTS))
    actual, specs = lazy_pointing(ms, ant_names, time_range(ms, (0, 1)))
    assert set(specs) == set(actual.data_vars)
    data_reads = [(col, n) for col, n in reads if col in DATA_COLUMNS]
    assert sorted(data_reads) == [(col, 1) for col in DATA_COLUMNS]
    assert all(n <= rp._CHECK_CHUNK_ELEMS for _, n in checks)
    assert not pivots
    reads.clear()
    actual.compute()
    assert len(pivots) == len(DATA_COLUMNS)
    assert all(n > 1 for col, n in reads if col in DATA_COLUMNS)


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
        self.getcol_calls.append((startrow, len(values)))
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
    calls = [call for table in used for call in table.getcol_calls]
    if (max_gap, window_bytes) == (1024, 16 * 2**20):
        assert len(calls) <= 2 * n_reads  # (the whole column is never one call)
    else:
        assert len(calls) > 2 * n_reads


def _add_pointing_row(ms: str) -> None:
    with tables.table(os.path.join(ms, "POINTING"), readonly=False, ack=False) as tb:
        tb.addrows(1)
        row = tb.nrows() - 1
        for col in tb.colnames():
            if col not in ("NAME",):
                tb.putcell(col, row, tb.getcell(col, 0))


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
    """A TIME changed in place (same number of rows): this process reads with
    the index of the open (the values of its rows); a process that reads the
    index again finds another selection: MSv2ChangedError."""
    ms = shutil.copytree(pointing_ms["regular"], str(tmp_path / "copy.ms"))
    ant_names = antenna_names(range(NANTS))
    time_min_max = time_range(ms, (0, 1))
    expected = create_pointing_xds(ms, ant_names, time_min_max, None)
    actual, _ = lazy_pointing(ms, ant_names, time_min_max)
    with tables.table(os.path.join(ms, "POINTING"), readonly=False, ack=False) as tb:
        tb.putcell("TIME", 3, tb.getcell("TIME", 3) + 0.01)
    assert_xds_bit_identical(actual, expected)
    bpt.clear_pointing_memos()
    with pytest.raises(MSv2ChangedError, match="times, antennas or rows"):
        _ = actual.POINTING_DISH_MEASURED.values


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
    raw_indexing_method = bpt.PointingColumnArray._raw_indexing_method

    def spy(self, key):
        reads.append(key)
        return raw_indexing_method(self, key)

    monkeypatch.setattr(bpt.PointingColumnArray, "_raw_indexing_method", spy)
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
