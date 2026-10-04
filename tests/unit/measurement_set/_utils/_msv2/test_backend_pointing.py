"""
The pointing_xds of the MSv2 xarray backend: the converter's hooks for it
(``create_pointing_xds(generic_loader=)``, ``read_pointing_columns(
keep_data=)``).
"""

import os

import numpy as np
import pytest
import xarray as xr

tables = pytest.importorskip("casacore.tables")

from xradio.measurement_set._utils._msv2._tables import (  # noqa: E402
    read_pointing as rp,
)
from xradio.measurement_set._utils._msv2._tables.subtable_cache import (  # noqa: E402
    SubtableCache,
    activate_subtable_cache,
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
