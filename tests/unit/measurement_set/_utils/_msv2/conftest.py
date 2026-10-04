"""
Fixtures of the MSv2 tests: generated MSs for the MSv2 xarray backend
(engine ``xradio_msv2``), shared by its test modules.

The MSs are generated once per session; tests only read them (a test that
writes into an MS works on a copy).
"""

import os
import shutil

import numpy as np
import pytest

# The variants of backend_ms (see make_backend_ms)
BACKEND_MS_VARIANTS = (
    "dense",
    "sparse_dup",
    "baseline_major",
    "time_descending",
    "shuffled",
    "rich",
    "wsp_partial",
    "no_weight",
    "single_dish",
)
# Rows per time and times per DDI of the generated MSs (5 antennas, 10
# baselines, 300 rows per DDI)
N_ANTENNAS = 5
N_BASELINES = 10
N_TIMES = 30


def _add_tiled_shape_column(main_tb, col, values, tile_rows, rows=None):
    """Add a TiledShapeStMan column (2-D cells) and write ``values`` into it;
    only ``rows`` (all by default) are written, the other cells stay
    undefined."""
    from casacore import tables

    valuetype = {"b": "boolean", "f": "float", "c": "complex"}[values.dtype.kind]
    desc = tables.makearrcoldesc(
        col,
        values.flat[0],
        ndim=2,
        valuetype=valuetype,
        datamanagertype="TiledShapeStMan",
        datamanagergroup=f"TSM_{col}",
    )
    cell = values.shape[1:]
    main_tb.addcols(
        tables.maketabdesc([desc]),
        dminfo={
            "TYPE": "TiledShapeStMan",
            "NAME": f"TSM_{col}",
            "SPEC": {
                "DEFAULTTILESHAPE": np.array([cell[1], cell[0], tile_rows], np.int32)
            },
        },
    )
    if rows is None:
        main_tb.putcol(col, values)
        return
    for row in rows:
        main_tb.putcell(col, int(row), values[row])


def _grid_indices(variant: str, pos: np.ndarray, ddi: np.ndarray, rng) -> tuple:
    """Time and baseline index of every row (``pos``: row index in its DDI)."""
    if variant == "single_dish":
        return pos // N_ANTENNAS, pos % N_ANTENNAS
    if variant == "baseline_major":
        return pos % N_TIMES, pos // N_TIMES
    if variant == "time_descending":
        return N_TIMES - 1 - pos // N_BASELINES, pos % N_BASELINES
    if variant == "shuffled":
        shuffled = pos.copy()
        for d in np.unique(ddi):
            in_ddi = ddi == d
            shuffled[in_ddi] = rng.permutation(pos[in_ddi])
        return shuffled // N_BASELINES, shuffled % N_BASELINES
    tidx, bidx = pos // N_BASELINES, pos % N_BASELINES
    if variant == "sparse_dup":
        # 10% of the rows to 3 extra, sparsely filled times (empty cells), and
        # some rows on the (time, baseline) of the row before (duplicated cells)
        nrows = pos.size
        moved = rng.choice(nrows, nrows // 10, replace=False)
        tidx[moved] = N_TIMES + rng.integers(0, 3, moved.size)
        dup = np.concatenate(
            [
                rng.choice(np.flatnonzero((pos > 0) & (ddi == d)), 4, replace=False)
                for d in np.unique(ddi)
            ]
        )
        tidx[dup], bidx[dup] = tidx[dup - 1], bidx[dup - 1]
    return tidx, bidx


def _rewrite_pointing(msname: str, time0: float, rng) -> None:
    """A POINTING table with rows of every antenna at times around those of
    MAIN (one row per antenna and second, plus a few outside)."""
    from casacore import tables

    times = time0 - 3.0 + np.arange(N_TIMES + 40, dtype=float)
    antennas = np.arange(N_ANTENNAS, dtype=np.int32)
    ant_col = np.tile(antennas, times.size)
    time_col = np.repeat(times, antennas.size)
    nrows = ant_col.size
    with tables.table(
        os.path.join(msname, "POINTING"), readonly=False, ack=False
    ) as tbl:
        tbl.removerows(np.arange(tbl.nrows()))
        tbl.addrows(nrows)
        tbl.putcol("ANTENNA_ID", ant_col)
        tbl.putcol("TIME", time_col)
        tbl.putcol("INTERVAL", np.ones(nrows))
        tbl.putcol("NAME", np.repeat("test_pointing", nrows))
        tbl.putcol("NUM_POLY", np.zeros(nrows, np.int32))
        tbl.putcol("TIME_ORIGIN", time_col)
        direction = np.deg2rad([28.98, 34.03]) + 0.01 * rng.random((nrows, 1, 2))
        tbl.putcol("DIRECTION", direction)
        tbl.putcol("TARGET", direction + 0.001)
        tbl.putcol("TRACKING", np.ones(nrows, bool))


def make_backend_ms(msname: str, variant: str, seed: int = 0) -> str:
    """
    Generate an MS (gen_test_ms: 2 SPWs x 2 polarization setups = 4 DDIs of
    300 rows, 16 channels, 2 correlations, 5 antennas) and rewrite its MAIN
    rows with random values for one of BACKEND_MS_VARIANTS. All variants
    have increasing frequencies in SPW 0 and decreasing ones in SPW 1, and
    DATA, CORRECTED_DATA (TiledColumnStMan), FLAG, WEIGHT, UVW,
    TIME_CENTROID and EXPOSURE.

    - "dense": every (time, baseline) once, time-major rows (30 times x 10
      baselines per DDI).
    - "sparse_dup": as "dense" with 10% of the rows on 3 extra sparse times
      and some duplicated (time, baseline) cells.
    - "baseline_major", "time_descending", "shuffled": other row orders.
    - "rich": several fields, scans and states (STATE_ID=-1 in scan 3, two
      OBS_MODEs), DATA and CORRECTED_DATA in TiledShapeStMan, MODEL_DATA in
      StandardStMan (three data groups), WEIGHT_SPECTRUM (TiledShapeStMan)
      and a POINTING row per antenna and second.
    - "wsp_partial": WEIGHT_SPECTRUM with undefined cells in the middle of
      every DDI (the converter falls back to WEIGHT).
    - "no_weight": undefined WEIGHT cells (WEIGHT=1 fallback).
    - "single_dish": FLOAT_DATA (TiledShapeStMan), autocorrelations only, a
      POINTING row per antenna and second.

    Returns
    -------
    str
        ``msname``.
    """
    from casacore import tables

    from xradio.testing.measurement_set.msv2_io import default_ms_descr, gen_test_ms

    if variant not in BACKEND_MS_VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}")
    gen_test_ms(
        msname,
        descr=dict(default_ms_descr, data_cols=["DATA", "CORRECTED_DATA"]),
        opt_tables=True,
        vlbi_tables=False,
        required_only=True,
        misbehave=False,
    )
    rng = np.random.default_rng(seed + BACKEND_MS_VARIANTS.index(variant))
    with tables.table(msname, readonly=False, ack=False) as main_tb:
        nrows = main_tb.nrows()
        ddi = main_tb.getcol("DATA_DESC_ID")
        assert np.all(np.diff(ddi) >= 0)
        pos = np.arange(nrows) - np.searchsorted(ddi, ddi)  # row index in its DDI
        tidx, bidx = _grid_indices(variant, pos, ddi, rng)
        if variant == "single_dish":
            ant1 = ant2 = np.arange(N_ANTENNAS)
        else:
            ant1, ant2 = np.triu_indices(N_ANTENNAS, 1)
        time0 = main_tb.getcell("TIME", 0)
        main_tb.putcol("TIME", time0 + tidx.astype(float))
        main_tb.putcol("TIME_CENTROID", time0 + tidx + rng.random(nrows) * 0.1)
        main_tb.putcol("ANTENNA1", ant1[bidx].astype(np.int32))
        main_tb.putcol("ANTENNA2", ant2[bidx].astype(np.int32))
        main_tb.putcol("EXPOSURE", rng.random(nrows))
        main_tb.putcol("UVW", rng.normal(size=(nrows, 3)))
        cell = main_tb.getcell("DATA", 0).shape
        visibilities = {
            col: (
                rng.normal(size=(nrows,) + cell) + 1j * rng.normal(size=(nrows,) + cell)
            ).astype(np.complex64)
            for col in ("DATA", "CORRECTED_DATA")
        }
        main_tb.putcol("FLAG", rng.random((nrows,) + cell) < 0.3)
        if variant != "no_weight":
            main_tb.putcol("WEIGHT", rng.random((nrows, cell[1])).astype(np.float32))
        weights = rng.random((nrows,) + cell).astype(np.float32)
        if variant == "single_dish":
            main_tb.removecols(["DATA", "CORRECTED_DATA"])
            _add_tiled_shape_column(
                main_tb, "FLOAT_DATA", visibilities["DATA"].real.copy(), 7
            )
        elif variant == "rich":
            # (columns of a TiledColumnStMan cannot be removed one by one)
            main_tb.removecols(["DATA", "CORRECTED_DATA"])
            _add_tiled_shape_column(main_tb, "DATA", visibilities["DATA"], 7)
            _add_tiled_shape_column(
                main_tb, "CORRECTED_DATA", visibilities["CORRECTED_DATA"], 5
            )
            model = (weights + 1j * weights[:, ::-1]).astype(np.complex64)
            desc = tables.makearrcoldesc("MODEL_DATA", 0j, ndim=2, valuetype="complex")
            main_tb.addcols(tables.maketabdesc([desc]))  # StandardStMan
            main_tb.putcol("MODEL_DATA", model)
            _add_tiled_shape_column(main_tb, "WEIGHT_SPECTRUM", weights, 7)
            main_tb.putcol("FIELD_ID", ((tidx // 5) % 2).astype(np.int32))
            scan = 1 + np.minimum(tidx // 10, 2)
            main_tb.putcol("SCAN_NUMBER", scan.astype(np.int32))
            main_tb.putcol(
                "STATE_ID",
                np.select([scan == 1, scan == 2], [0, 1], -1).astype(np.int32),
            )
        else:
            for col, values in visibilities.items():
                main_tb.putcol(col, values)
            if variant == "wsp_partial":
                defined = np.flatnonzero((pos < 100) | (pos >= 110))
                _add_tiled_shape_column(main_tb, "WEIGHT_SPECTRUM", weights, 7, defined)
    with tables.table(
        os.path.join(msname, "SPECTRAL_WINDOW"), readonly=False, ack=False
    ) as spw_tb:
        n_chan = spw_tb.getcol("CHAN_FREQ").shape[1]
        increasing = 1.0e9 + 1.0e6 * np.arange(n_chan)
        spw_tb.putcell("CHAN_FREQ", 0, increasing)
        spw_tb.putcell("CHAN_FREQ", 1, increasing[::-1] + 1.0e8)
    if variant == "rich":
        with tables.table(
            os.path.join(msname, "STATE"), readonly=False, ack=False
        ) as st:
            st.putcol(
                "OBS_MODE",
                np.array(["CALIBRATE_PHASE#ON_SOURCE", "OBSERVE_TARGET#ON_SOURCE"]),
            )
    if variant in ("rich", "single_dish"):
        _rewrite_pointing(msname, time0, rng)
    return msname


@pytest.fixture(scope="session")
def backend_ms(tmp_path_factory):
    """
    ``backend_ms(variant)``: the path of a generated MS of one of
    BACKEND_MS_VARIANTS (made on first use, see make_backend_ms). Read only:
    tests that write into an MS copy it first.
    """
    base = tmp_path_factory.mktemp("msv2_backend_ms")
    paths: dict[str, str] = {}

    def get(variant: str) -> str:
        if variant not in paths:
            paths[variant] = make_backend_ms(str(base / f"{variant}.ms"), variant)
        return paths[variant]

    yield get
    shutil.rmtree(base, ignore_errors=True)


@pytest.fixture(autouse=True)
def _msv2_partition_cache_off(monkeypatch):
    """The partition cache of the MSv2 backend is off in these tests (the
    default of partition_cache=None): opening an MS never uses partitions of
    an earlier open. Tests of the cache pass partition_cache explicitly."""
    monkeypatch.setenv("XRADIO_MSV2_PARTITION_CACHE", "off")
