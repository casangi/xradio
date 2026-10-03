import os
import threading

import numpy as np
import pytest
import xarray as xr

tables = pytest.importorskip("casacore.tables")

from xradio.measurement_set._utils._msv2._tables import (  # noqa: E402
    read_pointing as rp,
)
from xradio.measurement_set._utils._msv2._tables.read import (  # noqa: E402
    load_generic_table,
    redimension_ms_subtable,
)
from xradio.measurement_set._utils._msv2._tables.subtable_cache import (  # noqa: E402
    SubtableCache,
    activate_subtable_cache,
)
from xradio.measurement_set._utils._msv2._tables.table_query import (  # noqa: E402
    open_table_ro,
)
from xradio.measurement_set._utils._msv2.msv4_sub_xdss import (  # noqa: E402
    create_pointing_xds,
)
from xradio.measurement_set._utils._msv2.subtables import (  # noqa: E402
    subt_rename_ids,
)

NANTS = 5
NTIMES = 400
TIME0 = 5.2e9
DATA_COLUMNS = ("DIRECTION", "ENCODER", "OVER_THE_TOP")


def make_pointing_ms(path: str, variant: str = "regular", seed: int = 0) -> str:
    """
    A directory with only a POINTING sub-table (all create_pointing_xds reads).

    Every antenna is sampled at its own subset of NTIMES times (missing (time,
    antenna) cells), rows are shuffled (not time-ordered) and a few rows repeat
    the (TIME, ANTENNA_ID) of another row with other values (duplicates).

    variant:
    - "regular": all cells defined, one shape per column
    - "varying_target": TARGET (not converted) has cells of two shapes
    - "no_direction": no DIRECTION column
    - "empty": no rows
    - "empty_encoder": zero-size ENCODER cells
    - "float_over_the_top": OVER_THE_TOP is a float column
    """
    os.makedirs(path, exist_ok=True)
    rng = np.random.default_rng(seed)
    extra = [tables.makearrcoldesc("ENCODER", 0.0, ndim=1)]
    if variant == "float_over_the_top":
        extra.append(tables.makescacoldesc("OVER_THE_TOP", 0.0, valuetype="float"))
    else:
        extra.append(tables.makescacoldesc("OVER_THE_TOP", False))
    tb = tables.default_ms_subtable(
        "POINTING", os.path.join(path, "POINTING"), tables.maketabdesc(extra)
    )
    try:
        if variant == "no_direction":
            tb.removecols(["DIRECTION"])
        if variant == "empty":
            return path
        times = TIME0 + np.arange(NTIMES) * 0.048
        rows_t, rows_a = [], []
        for ant in range(NANTS):
            keep = rng.random(NTIMES) < (0.95 if ant else 0.6)
            rows_t.append(times[keep])
            rows_a.append(np.full(keep.sum(), ant))
        time = np.concatenate(rows_t)
        ant = np.concatenate(rows_a)
        dup = rng.choice(time.size, 25, replace=False)
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
        if variant != "no_direction":
            tb.putcol("DIRECTION", rng.normal(size=(nrows, 1, 2)))
        if variant == "varying_target":
            for row in range(nrows):
                shape = (2, 2) if row % 97 == 5 else (1, 2)
                tb.putcell("TARGET", row, rng.normal(size=shape))
        else:
            tb.putcol("TARGET", rng.normal(size=(nrows, 1, 2)))
        if variant == "empty_encoder":
            for row in range(nrows):
                tb.putcell("ENCODER", row, np.zeros(0))
        else:
            tb.putcol("ENCODER", rng.normal(size=(nrows, 2)))
        if variant == "float_over_the_top":
            tb.putcol("OVER_THE_TOP", rng.random(nrows).astype(np.float32))
        else:
            tb.putcol("OVER_THE_TOP", rng.random(nrows) < 0.3)
    finally:
        tb.close()
    return path


@pytest.fixture(scope="module")
def pointing_ms(tmp_path_factory):
    base = tmp_path_factory.mktemp("read_pointing")
    return {
        variant: make_pointing_ms(str(base / f"{variant}.ms"), variant)
        for variant in (
            "regular",
            "varying_target",
            "no_direction",
            "empty",
            "empty_encoder",
            "float_over_the_top",
        )
    }


def antenna_names(antenna_ids) -> xr.DataArray:
    """antenna_name with an antenna_id index, as convert_and_write_partition passes it"""
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


def pointing_times(ms):
    with open_table_ro(os.path.join(ms, "POINTING")) as tb:
        return np.unique(tb.getcol("TIME"))


# (time range as positions in the POINTING times, antennas): from the whole table
# (more than 1000 rows: the uncached read uses getcol) to single samples (row())
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


@pytest.mark.parametrize("data_in_memory", [True, False])
@pytest.mark.parametrize("interpolate", [False, True])
@pytest.mark.parametrize("selection", SELECTIONS)
def test_create_pointing_xds_cached_identical(
    pointing_ms, selection, interpolate, data_in_memory, monkeypatch
):
    """Cached POINTING columns + numpy pivot give exactly the uncached pointing_xds,
    with missing cells, duplicated (TIME, ANTENNA_ID) rows and shuffled rows, with
    the data columns in memory or read per partition (over the memory budget)."""
    if not data_in_memory:
        monkeypatch.setattr(rp, "POINTING_MAX_CACHED_DATA_BYTES", 0)
    ms = pointing_ms["regular"]
    utimes = pointing_times(ms)
    span = utimes[-1] - utimes[0]
    (pos_min, pos_max), ants = selection
    time_min_max = (
        np.float64(utimes[0] + pos_min * span),
        np.float64(utimes[0] + pos_max * span),
    )
    ant_names = antenna_names(ants)
    interp_time = None
    if interpolate:
        main_time = np.linspace(*time_min_max, 11) - 3_506_716_800.0
        interp_time = xr.DataArray(main_time, dims="time", coords={"time": main_time})

    expected = create_pointing_xds(ms, ant_names, time_min_max, interp_time)
    cache = SubtableCache()
    with activate_subtable_cache(cache):
        actual = create_pointing_xds(ms, ant_names, time_min_max, interp_time)

    assert cache.stats["pointing_cached"] == 1
    (pointing_columns,) = [
        value for key, value in cache._state.values.items() if "pointing" in key[0]
    ]
    assert (pointing_columns.data is not None) == data_in_memory
    assert_xds_bit_identical(expected, actual)
    if ants == [7]:
        assert not actual.data_vars
    else:
        assert {"POINTING_BEAM", "POINTING_DISH_MEASURED"} <= set(actual.data_vars)


def test_create_pointing_xds_cached_missing_cells_and_duplicates(pointing_ms):
    """The test table does exercise missing cells (bool fill: True) and duplicates
    (the first row wins), and the cached path reproduces both."""
    ms = pointing_ms["regular"]
    with open_table_ro(os.path.join(ms, "POINTING")) as tb:
        time, ant = tb.getcol("TIME"), tb.getcol("ANTENNA_ID")
        direction = tb.getcol("DIRECTION")
    pairs = np.stack([time, ant.astype(np.float64)], axis=1)
    _, first, counts = np.unique(pairs, axis=0, return_index=True, return_counts=True)
    assert (counts > 1).any()
    dup_row = first[np.flatnonzero(counts > 1)[0]]

    cache = SubtableCache()
    with activate_subtable_cache(cache):
        xds = create_pointing_xds(
            ms, antenna_names(range(NANTS)), (time.min() - 1, time.max() + 1), None
        )
    assert xds.sizes["time_pointing"] * xds.sizes["antenna_name"] > first.size
    beam = xds.POINTING_BEAM.sel(
        time_pointing=time[dup_row] - 3_506_716_800.0,
        antenna_name=f"ant{ant[dup_row]}",
    )
    np.testing.assert_array_equal(beam.values, direction[dup_row, 0])
    assert xds.POINTING_OVER_THE_TOP.dtype == bool
    missing = np.isnan(xds.POINTING_BEAM.values[..., 0])
    assert missing.any()
    assert xds.POINTING_OVER_THE_TOP.values[missing].all()


@pytest.mark.parametrize(
    "variant",
    [
        "varying_target",
        "no_direction",
        "empty",
        "empty_encoder",
        "float_over_the_top",
    ],
)
def test_create_pointing_xds_not_cacheable(pointing_ms, variant):
    """Tables the cache cannot reproduce exactly are read per partition, as before."""
    ms = pointing_ms[variant]
    assert rp.read_pointing_columns(os.path.join(ms, "POINTING"), DATA_COLUMNS) is None
    time_min_max = (np.float64(TIME0 - 1), np.float64(TIME0 + NTIMES))
    ant_names = antenna_names(range(NANTS))
    cache = SubtableCache()

    def convert(subtable_cache):
        with activate_subtable_cache(subtable_cache):
            try:
                return create_pointing_xds(ms, ant_names, time_min_max, None)
            except Exception as exc:
                return f"{type(exc).__name__}: {exc}"

    expected, actual = convert(None), convert(cache)
    assert cache.stats["pointing_uncached"] == 1
    if isinstance(expected, str):
        assert actual == expected
    else:
        assert_xds_bit_identical(expected, actual)


def test_select_pointing_rows_matches_taql(pointing_ms):
    """The numpy selection picks the rows of the TaQL WHERE of the uncached read."""
    from xradio.measurement_set._utils._msv2._tables.read import (
        make_taql_where_between_min_max,
    )

    ms = pointing_ms["regular"]
    pointing_columns = rp.read_pointing_columns(
        os.path.join(ms, "POINTING"), DATA_COLUMNS
    )
    utimes = pointing_times(ms)
    rng = np.random.default_rng(3)
    # tb is used by the TaQL query ($tb); opened (no read lock) as xradio opens it
    with open_table_ro(os.path.join(ms, "POINTING")) as tb:  # noqa: F841
        for _ in range(20):
            lo, hi = np.sort(rng.uniform(utimes[0] - 1, utimes[-1] + 1, 2))
            ants = np.unique(rng.integers(0, NANTS + 2, 3))
            where = make_taql_where_between_min_max((lo, hi), ms, "POINTING", "TIME")
            where += f" AND (ANTENNA_ID IN [{','.join(map(str, ants))}])"
            with tables.taql(f"select * from $tb {where}") as sel:
                expected = np.asarray(sel.rownumbers())
            idx = rp.select_pointing_rows(pointing_columns, (lo, hi), ants)
            np.testing.assert_array_equal(np.sort(pointing_columns.row[idx]), expected)


def test_generic_dims_match_load_generic_table(pointing_ms):
    """generic_dims names the dimensions as load_generic_table does."""
    ms = pointing_ms["regular"]
    generic = load_generic_table(
        ms, "POINTING", timecols=["TIME"], rename_ids=subt_rename_ids["POINTING"]
    )
    with open_table_ro(os.path.join(ms, "POINTING")) as tb:
        columns = [
            (
                col,
                col.endswith("_ID") or col == "TIME",
                () if tb.isscalarcol(col) else tb.getcell(col, 0).shape,
            )
            for col in tb.colnames()
        ]
        nrows = tb.nrows()
    var_dims, sizes = rp.generic_dims(columns, nrows)
    # load_generic_table pivots to (TIME, ANTENNA_ID): compare the cell dimensions
    for name, dims in var_dims.items():
        if name in generic.data_vars:
            assert generic[name].dims[2:] == dims[1:], name
    assert var_dims["DIRECTION"] == ("row", "n_polynomial", "dim_2")
    assert var_dims["ENCODER"] == ("row", "dir")
    assert sizes == {"row": nrows, "n_polynomial": 1, "dim_2": 2, "dir": 2}


@pytest.mark.parametrize("dtype", [np.float64, np.float32, np.int32, np.bool_])
@pytest.mark.parametrize("missing", [False, True])
def test_pivot_time_antenna_matches_redimension(dtype, missing):
    """pivot_time_antenna fills and casts like redimension_ms_subtable (xarray
    unstack + cast back), also with duplicated rows (first one wins)."""
    rng = np.random.default_rng(int(missing))
    times = np.repeat(np.arange(6.0), 3)
    ants = np.tile(np.arange(3), 6).astype(np.int32)
    if missing:
        keep = rng.random(times.size) < 0.6
        times, ants = times[keep], ants[keep]
    # duplicate two rows with other values
    times = np.concatenate([times, times[:2]])
    ants = np.concatenate([ants, ants[:2]])
    values = (rng.normal(size=(times.size, 2)) * 10).astype(dtype)
    xds = xr.Dataset(
        {"V": (("row", "dim_1"), values)},
        coords={"TIME": ("row", times), "ANTENNA_ID": ("row", ants)},
    )
    expected = redimension_ms_subtable(xds, "POINTING")

    utime, tcode = np.unique(times, return_inverse=True)
    uant, acode = np.unique(ants, return_inverse=True)
    key = tcode * uant.size + acode
    _, first = np.unique(key, return_index=True)
    grid = rp.pivot_time_antenna(
        values[first], tcode[first], acode[first], (utime.size, uant.size)
    )
    np.testing.assert_array_equal(expected.TIME.values, utime)
    np.testing.assert_array_equal(expected.ANTENNA_ID.values, uant)
    assert grid.dtype == expected.V.dtype
    assert grid.tobytes() == np.ascontiguousarray(expected.V.values).tobytes()


@pytest.mark.parametrize("data_in_memory", [True, False])
def test_pointing_columns_built_once_across_threads(
    pointing_ms, data_in_memory, monkeypatch
):
    """Partitions converted in threads (parallel_mode="partition") share one read."""
    if not data_in_memory:
        monkeypatch.setattr(rp, "POINTING_MAX_CACHED_DATA_BYTES", 0)
    ms = pointing_ms["regular"]
    cache = SubtableCache()
    results, errors = [], []
    utimes = pointing_times(ms)

    def convert(ants):
        try:
            with activate_subtable_cache(cache):
                results.append(
                    create_pointing_xds(
                        ms, antenna_names(ants), (utimes[0], utimes[-1]), None
                    )
                )
        except Exception as exc:  # pragma: no cover - reported below
            errors.append(exc)

    threads = [threading.Thread(target=convert, args=([a],)) for a in range(NANTS)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors
    assert len(results) == NANTS
    for xds in results:
        (name,) = xds.antenna_name.values
        expected = create_pointing_xds(
            ms, antenna_names([int(name[3:])]), (utimes[0], utimes[-1]), None
        )
        assert_xds_bit_identical(expected, xds)
    assert cache.stats["value_builds"] == 1
    assert cache.stats["pointing_cached"] == NANTS


def test_load_cached_pointing_generic_xds_without_cache(pointing_ms):
    """No active cache (or no time range): the caller reads POINTING itself."""
    ms = pointing_ms["regular"]
    assert (
        rp.load_cached_pointing_generic_xds(ms, (0.0, 1e10), np.arange(3), DATA_COLUMNS)
        is None
    )
    with activate_subtable_cache(SubtableCache()):
        assert (
            rp.load_cached_pointing_generic_xds(ms, None, np.arange(3), DATA_COLUMNS)
            is None
        )
        assert (
            rp.load_cached_pointing_generic_xds(
                ms, (0.0, 1e10), np.arange(0), DATA_COLUMNS
            )
            is None
        )
