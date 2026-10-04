import os
import pathlib
import shutil
import types
from collections import namedtuple
from contextlib import nullcontext as no_raises

import numpy as np
import pytest
import xarray as xr

import xradio.measurement_set._utils._msv2.conversion as conversion
from xradio.measurement_set._utils._msv2 import stream_write
from xradio.measurement_set.schema import VisibilityXds
from xradio.schema.check import check_dataset, check_datatree
from xradio.testing.measurement_set.checker import check_msv4_matches_descr

minxds = namedtuple("minxds", "data_vars coords sizes")
xds_main = minxds(
    {"VISIBILITY": np.empty((1), dtype=np.complex64)},
    {"baseline_id"},
    {
        "time": 220,
        "baseline_id": 55,
        "frequency": 3890,
        "polarization": 2,
    },
)
xds_main_sd = minxds(
    {"SPECTRUM": np.empty((1), dtype=np.complex64)},
    {"antenna_name"},
    {
        "time": 1200,
        "antenna_name": 11,
        "frequency": 1800,
        "polarization": 2,
    },
)
xds_main_sd_bogus = minxds(
    {"SPECTRUM": np.empty((1), dtype=np.complex64)},
    {"antenna_name"},
    {
        "time": 1200,
        "bogus": 11,
        "frequency": 1800,
        "polarization": 2,
    },
)
xds_pointing = minxds(
    {"BEAM_POINTING": np.empty((1), dtype=np.complex64)},
    {"antenna_name"},
    {
        "time": 10220,
        "antenna_name": 55,
        "direction": 2,
    },
)
xds_pointing_small = minxds(
    {"BEAM_POINTING": np.empty((1), dtype=np.complex64)},
    {"antenna_name"},
    {
        "time": 102,
        "antenna_name": 12,
        "direction": 2,
    },
)


@pytest.mark.parametrize(
    "input_chunksize, xds_type, xds, expected_chunksize, expected_error",
    [
        ({}, "main", None, {}, no_raises()),
        ({}, "pointing", None, {}, no_raises()),
        (
            {"time": 10, "baseline_id": 5, "frequency": 100, "polarization": 2},
            "main",
            xds_main,
            {"time": 10, "baseline_id": 5, "frequency": 100, "polarization": 2},
            no_raises(),
        ),
        (
            {"time": 10, "foo": 3},
            "main",
            xds_main,
            {},
            pytest.raises(ValueError, match="foo"),
        ),
        (
            0.02,
            "main",
            xds_main,
            {"time": 70, "baseline_id": 16, "frequency": 1198, "polarization": 2},
            no_raises(),
        ),
        (
            "wrong_input_chunksize",
            "main",
            xds_main,
            None,
            pytest.raises(ValueError, match="expected as a dict"),
        ),
        (
            0.01,
            "main",
            xds_main_sd,
            {"time": 408, "antenna_name": 3, "frequency": 548, "polarization": 2},
            no_raises(),
        ),
        (
            0.01,
            "main",
            xds_main_sd_bogus,
            None,
            pytest.raises(KeyError, match="antenna_name"),
        ),
        (
            0.0002,
            "pointing",
            xds_pointing,
            {"time": 1677, "antenna_name": 8, "direction": 2},
            no_raises(),
        ),
    ],
)
def test_parse_chunksize(
    input_chunksize, xds_type, xds, expected_chunksize, expected_error
):
    # parse_chunksize checks the input chunksize (if dict), or auto-calculates the
    # sizes (if given as numeric memory value). The calculations are better tested for the
    # lower level functions
    with expected_error:
        assert (
            conversion.parse_chunksize(input_chunksize, xds_type, xds)
            == expected_chunksize
        )


@pytest.mark.parametrize(
    "chunksize, xds_type, expectation",
    [
        ({}, "main", no_raises()),
        ({}, "pointing", no_raises()),
        (
            {"baseline_id": 1, "frequency": 2, "polarization": 3, "time": 4},
            "main",
            no_raises(),
        ),
        (
            {"baseline_id": 1, "frequency": 2, "polarization": 3, "time": 4},
            "pointing",
            pytest.raises(ValueError, match="baseline_id"),
        ),
        ({"foo": "a"}, "main", pytest.raises(ValueError, match="foo")),
        ({"foo": 1}, "pointing", pytest.raises(ValueError, match="foo")),
    ],
)
def test_check_chunksize(chunksize, xds_type, expectation):
    with expectation:
        conversion.check_chunksize(chunksize, xds_type)


@pytest.mark.parametrize(
    "pseudo_xds, xds_type, expected_chunksize, expected_error",
    [
        (
            xds_main,
            "main",
            {"baseline_id": 28, "frequency": 2048, "polarization": 2, "time": 117},
            no_raises(),
        ),
        (
            xds_main,
            "bar_wrong",
            {"baseline_id": 28, "frequency": 2048, "polarization": 2, "time": 117},
            pytest.raises(RuntimeError),
        ),
    ],
)
def test_mem_chunksize_to_dict(
    pseudo_xds, xds_type, expected_chunksize, expected_error
):
    # mem_chunksize_to_dict relies on mem_chunksize_to_dict_main_*,
    # mem_chunksize_to_dict_pointing*, etc. which are better tested below
    with expected_error:
        assert (
            conversion.mem_chunksize_to_dict(0.1, xds_type, pseudo_xds)
            == expected_chunksize
        )


@pytest.mark.parametrize(
    "mem_size, pseudo_xds, expected_chunksize, expected_error",
    [
        (
            1e-9,  # not enough even for one data point / all pols
            xds_main,
            {"baseline_id": 28, "frequency": 2048, "polarization": 2, "time": 117},
            pytest.raises(RuntimeError, match="memory bound"),
        ),
        (
            0.9,  # enough to hold all in mem
            xds_main,
            {"time": 220, "baseline_id": 55, "frequency": 3890, "polarization": 2},
            no_raises(),
        ),
        (
            0.01,
            xds_main_sd,
            {"antenna_name": 3, "frequency": 548, "polarization": 2, "time": 408},
            no_raises(),
        ),
        (
            0.01,
            xds_main_sd_bogus,
            None,
            pytest.raises(KeyError, match="antenna_name"),
        ),
    ],
)
def test_mem_chunksize_to_dict_main(
    mem_size, pseudo_xds, expected_chunksize, expected_error
):
    with expected_error:
        assert (
            conversion.mem_chunksize_to_dict_main(mem_size, pseudo_xds)
            == expected_chunksize
        )


@pytest.mark.parametrize(
    "mem_size, dim_sizes, expected_chunksize",
    [
        (
            0.001,
            {"time": 200, "baseline_id": 21, "frequency": 1000, "polarization": 3},
            {"time": 50, "baseline_id": 4, "frequency": 223, "polarization": 3},
        ),
        (
            0.02,
            {"time": 200, "baseline_id": 21, "frequency": 1000, "polarization": 3},
            {"time": 124, "baseline_id": 12, "frequency": 601, "polarization": 3},
        ),
        (
            0.03,
            {"time": 200, "baseline_id": 21, "frequency": 1000, "polarization": 3},
            {"time": 140, "baseline_id": 14, "frequency": 684, "polarization": 3},
        ),
        (
            0.05,
            {"time": 200, "baseline_id": 21, "frequency": 1000, "polarization": 4},
            {"time": 151, "baseline_id": 15, "frequency": 740, "polarization": 4},
        ),
    ],
)
def test_mem_chunksize_to_dict_main_balanced(mem_size, dim_sizes, expected_chunksize):
    res = conversion.mem_chunksize_to_dict_main_balanced(
        mem_size, dim_sizes, "baseline_id", 8
    )
    assert res == pytest.approx(expected_chunksize)


@pytest.mark.parametrize(
    "mem_size, dim_sizes, expected_chunksize",
    [
        (
            0.1,
            xds_pointing,
            {"time": 10220, "antenna_name": 55, "direction": 2},
        ),
        (
            0.5,
            xds_pointing_small,
            xds_pointing_small.sizes,
        ),
        (0.5, minxds({}, {}, {}), {}),
    ],
)
def test_mem_chunksize_to_dict_pointing(mem_size, dim_sizes, expected_chunksize):
    res = conversion.mem_chunksize_to_dict_pointing(mem_size, dim_sizes)
    assert res == pytest.approx(expected_chunksize)


def _placeholder_xds(sizes: dict, variables: dict) -> xr.Dataset:
    """A main xds whose data variables (name -> (dims, dtype)) hold no memory
    (zero-strided)."""
    data_vars = {}
    for name, (dims, dtype) in variables.items():
        shape = tuple(sizes[dim] for dim in dims)
        data_vars[name] = (dims, np.broadcast_to(np.zeros((), dtype=dtype), shape))
    return xr.Dataset(data_vars)


VIS_DIMS = ("time", "baseline_id", "frequency", "polarization")
SD_DIMS = ("time", "antenna_name", "frequency", "polarization")
MAIN_VARS = {
    "VISIBILITY": (VIS_DIMS, np.complex64),
    "FLAG": (VIS_DIMS, np.bool_),
    "WEIGHT": (VIS_DIMS, np.float32),
    "UVW": (("time", "baseline_id", "uvw_label"), np.float64),
    "TIME_CENTROID": (("time", "baseline_id"), np.float64),
}


def _sizes(time, baselines, frequency, polarization):
    return {
        "time": time,
        "baseline_id": baselines,
        "antenna_name": baselines,
        "frequency": frequency,
        "polarization": polarization,
        "uvw_label": 3,
    }


@pytest.mark.parametrize(
    "sizes, variables, names, expected",
    [
        # small partition: one time chunk (as one chunk per variable before)
        (_sizes(30, 10, 16, 2), MAIN_VARS, None, {"time": 30}),
        # 3c391-like: 718848 bytes of VISIBILITY per time step
        (_sizes(4000, 351, 64, 4), MAIN_VARS, None, {"time": 186}),
        # double precision visibilities: half the time steps
        (
            _sizes(4000, 351, 64, 4),
            MAIN_VARS | {"VISIBILITY": (VIS_DIMS, np.complex128)},
            None,
            {"time": 93},
        ),
        # the WEIGHT=1 fallback (float64) as large as complex64 VISIBILITY
        (
            _sizes(4000, 351, 64, 4),
            MAIN_VARS | {"WEIGHT": (VIS_DIMS, np.float64)},
            None,
            {"time": 186},
        ),
        # one channel and polarization: UVW (24 bytes per baseline) is largest
        (_sizes(10**6, 351, 1, 1), MAIN_VARS, None, {"time": 2**27 // (351 * 24)}),
        # single dish without UVW (dropped)
        (
            _sizes(10**5, 12, 4096, 2),
            {
                "SPECTRUM": (SD_DIMS, np.float32),
                "FLAG": (SD_DIMS, np.bool_),
                "UVW": (("time", "antenna_name", "uvw_label"), np.float64),
            },
            ["SPECTRUM", "FLAG"],
            {"time": 2**27 // (12 * 4096 * 2 * 4)},
        ),
        # one time step of 252 MiB (under the Blosc limit): time only
        (_sizes(5, 2016, 4096, 4), MAIN_VARS, None, {"time": 1}),
        # one time step of 2.35 GiB (over the Blosc limit): frequency too
        (
            _sizes(3, 131328, 600, 4),
            MAIN_VARS,
            None,
            {"time": 1, "frequency": 2**27 // (131328 * 4 * 8)},
        ),
        # no data variable along time
        (_sizes(5, 3, 2, 1), {}, None, {}),
    ],
)
def test_default_main_chunksize(sizes, variables, names, expected, monkeypatch):
    logged = []
    logger = types.SimpleNamespace(
        warning=lambda msg: logged.append(("warning", msg)),
        debug=lambda msg: logged.append(("debug", msg)),
    )
    monkeypatch.setattr(conversion, "xradio_logger", lambda: logger)
    xds = _placeholder_xds(sizes, variables)
    chunks = conversion.default_main_chunksize(xds, names)
    assert chunks == expected
    # a warning only if a time step is over the Blosc limit (other axes split):
    # the streamed write holds one time step per batch
    warnings = [msg for level, msg in logged if level == "warning"]
    if set(expected) - {"time"}:
        assert len(warnings) == 1
        assert "frequency" in warnings[0] and "one time step" in warnings[0]
    else:
        assert warnings == []
    for name in names or list(xds.data_vars):
        var = xds[name].variable
        nbytes = conversion._chunk_nbytes(var, chunks)
        assert nbytes <= conversion.BLOSC_MAX_BUFFER_BYTES
        if chunks.get("time", 0) > 1:
            assert nbytes <= conversion.DEFAULT_MAIN_CHUNK_BYTES


def test_itemsize_spec():
    assert conversion.itemsize_spec(xds_main) == 8


def test_itemsize_pointing_spec():
    assert conversion.itemsize_spec(xds_pointing) == 8


def test_calc_used_gb():
    res = conversion.calc_used_gb(
        {"time": 200, "baseline_id": 21, "frequency": 1000, "polarization": 3},
        "baseline_id",
        8,
    )
    assert res == pytest.approx(0.0938773)


@pytest.mark.parametrize(
    "input_name, partitions, expected_estimate",
    [
        ("test_ms_minimal_required.ms", {}, (0.0, 0, 0)),
    ],
)
def test_estimate_memory_and_cores_for_partitions(
    input_name, partitions, expected_estimate
):
    # partitions = {}

    res = conversion.estimate_memory_and_cores_for_partitions(input_name, partitions)

    assert res[0] == pytest.approx(expected_estimate[0])
    assert res[1:] == expected_estimate[1:]


def test_convert_and_write_partition_empty_complete(ms_empty_complete):
    conversion.convert_and_write_partition(
        in_file=ms_empty_complete.fname,
        out_file="out_file_test_empty_complete.zarr",
        ms_v4_id="msv4_id",
        partition_info={
            "DATA_DESC_ID": [0],
            "OBS_MODE": ["scan_intent#subscan_intent"],
        },
        use_table_iter=False,
    )


def test_convert_and_write_partition_empty_required(ms_empty_required):
    conversion.convert_and_write_partition(
        in_file=ms_empty_required.fname,
        out_file="out_file_test_empty.zarr",
        ms_v4_id="msv4_id",
        partition_info={
            "DATA_DESC_ID": [0],
            "OBS_MODE": ["scan_intent#subscan_intent"],
        },
        use_table_iter=False,
    )


def test_convert_and_write_partition_min(ms_minimal_required):
    out_name = "out_file_test_convert_write.zarr"
    msv4_id = "msv4_id"
    try:
        conversion.convert_and_write_partition(
            in_file=ms_minimal_required.fname,
            out_file=out_name,
            ms_v4_id=msv4_id,
            partition_info={
                "DATA_DESC_ID": [0],
                "OBS_MODE": ["scan_intent#subscan_intent"],
            },
            use_table_iter=False,
            pointing_interpolate=True,
            ephemeris_interpolate=True,
            phase_cal_interpolate=True,
            sys_cal_interpolate=True,
        )

        # msv4_xds = xr.open_dataset(
        #     out_name
        #     + "/"
        #     + ms_minimal_required.fname.rsplit(".")[0]
        #     + "_"
        #     + msv4_id,
        #     engine="zarr",
        # )
        msv4_xdt = xr.open_datatree(
            out_name + "/" + ms_minimal_required.fname.rsplit(".")[0] + "_" + msv4_id,
            engine="zarr",
        )
        check_dataset(msv4_xdt.ds, VisibilityXds)
        check_datatree(msv4_xdt)
        check_msv4_matches_descr(msv4_xdt, ms_minimal_required.descr)
    finally:
        shutil.rmtree(out_name)


def test_convert_and_write_partition_misbehaved(ms_minimal_misbehaved):
    out_name = "out_file_test_convert_write_misbehaving_ms.zarr"
    msv4_id = "msv4_id"
    try:
        conversion.convert_and_write_partition(
            in_file=ms_minimal_misbehaved.fname,
            out_file=out_name,
            ms_v4_id=msv4_id,
            partition_info={
                "DATA_DESC_ID": [0],
                "OBS_MODE": ["scan_intent#subscan_intent"],
                "__OBS_MODE": [0],
            },
            use_table_iter=False,
            pointing_interpolate=True,
        )

        msv4_xdt = xr.open_datatree(
            out_name + "/" + ms_minimal_misbehaved.fname.rsplit(".")[0] + "_" + msv4_id,
            engine="zarr",
        )
        check_dataset(msv4_xdt.ds, VisibilityXds)
        check_datatree(msv4_xdt).expect()
        check_msv4_matches_descr(msv4_xdt, ms_minimal_misbehaved.descr)
    finally:
        shutil.rmtree(out_name)


def test_convert_and_write_partition_with_antenna1(ms_minimal_required):
    out_name = "out_file_test_convert_write_with_antenna1.zarr"
    msv4_id = "msv4_id"
    try:
        conversion.convert_and_write_partition(
            in_file=ms_minimal_required.fname,
            out_file=out_name,
            ms_v4_id=msv4_id,
            partition_info={
                "DATA_DESC_ID": [0],
                "OBS_MODE": ["scan_intent#subscan_intent"],
                "ANTENNA1": [0],
            },
            use_table_iter=False,
            pointing_interpolate=True,
        )

        # Will need a SD-like test ms. Otherwise the partition is empty:
        with pytest.raises(FileNotFoundError, match="does not exist"):
            msv4_xds = xr.open_dataset(
                out_name
                + "/"
                + ms_minimal_required.fname.rsplit(".")[0]
                + "_"
                + msv4_id,
                engine="zarr",
            )
            msv4_xdt = xr.open_datatree(
                out_name
                + "/"
                + ms_minimal_required.fname.rsplit(".")[0]
                + "_"
                + msv4_id,
                engine="zarr",
            )
            check_dataset(msv4_xds, VisibilityXds)
            check_dataset(msv4_xdt.ds, VisibilityXds)
            check_datatree(msv4_xdt).expect()
            check_msv4_matches_descr(msv4_xdt, ms_minimal_required.descr)
    finally:
        # shutil.rmtree(out_name)
        pass


def test_convert_and_write_partition_without_opt(ms_minimal_without_opt):
    out_name = "out_file_test_convert_write_ms_min_without_optional_subtables.zarr"
    msv4_id = "msv4_id"
    try:
        conversion.convert_and_write_partition(
            in_file=ms_minimal_without_opt.fname,
            out_file=out_name,
            ms_v4_id=msv4_id,
            partition_info={
                "DATA_DESC_ID": [0],
                "OBS_MODE": ["scan_intent#subscan_intent"],
                "STATE_ID": [0],
            },
            use_table_iter=False,
            ephemeris_interpolate=True,
            pointing_interpolate=True,
        )

        msv4_xdt = xr.open_datatree(
            out_name
            + "/"
            + ms_minimal_without_opt.fname.rsplit(".")[0]
            + "_"
            + msv4_id,
            engine="zarr",
        )
        check_dataset(msv4_xdt.ds, VisibilityXds)
        check_datatree(msv4_xdt).expect()
        check_msv4_matches_descr(msv4_xdt, ms_minimal_without_opt.descr)
    finally:
        shutil.rmtree(out_name)


ms_custom_description = {
    "nrows_per_ddi": 300,
    "nchans": 4,
    "npols": 1,
    "data_cols": ["DATA", "MODEL_DATA", "CORRECTED_DATA"],
    "SPECTRAL_WINDOW": {"0": 0},
    "POLARIZATION": {"0": 0},
    "ANTENNA": {"0": 0, "1": 1, "2": 2},
    "FIELD": {"0": 0},
    "SCAN": {"1": {"0": {"intent": "intent#subintent"}}},
    "STATE": {"0": {"id": 0, "intent": None}},
    "OBSERVATION": {"0": 0},
    "FEED": {"0": 0},
    "PROCESSOR": {"0": 0},
    "SOURCE": {},
}


@pytest.mark.parametrize("ms_custom_spec", [ms_custom_description], indirect=True)
def test_convert_and_write_partition_custom(ms_custom_spec):
    out_name = "out_file_test_convert_write.zarr"
    msv4_id = "msv4_id"
    try:
        # use_table_iter: deprecated, no effect
        with pytest.warns(DeprecationWarning, match="use_table_iter") as record:
            conversion.convert_and_write_partition(
                in_file=ms_custom_spec.fname,
                out_file=out_name,
                ms_v4_id=msv4_id,
                partition_info={
                    "DATA_DESC_ID": [0],
                    "OBS_MODE": ["scan_intent#subscan_intent"],
                },
                use_table_iter=True,
                pointing_interpolate=True,
                ephemeris_interpolate=True,
                phase_cal_interpolate=False,
                sys_cal_interpolate=False,
            )
        assert record[0].filename == __file__
        assert len([w for w in record if "use_table_iter" in str(w.message)]) == 1
        msv4_xdt = xr.open_datatree(
            out_name + "/" + ms_custom_spec.fname.rsplit(".")[0] + "_" + msv4_id,
            engine="zarr",
        )
        check_dataset(msv4_xdt.ds, VisibilityXds)
        check_datatree(msv4_xdt).expect()
        check_msv4_matches_descr(msv4_xdt, ms_custom_spec.descr)

    finally:
        shutil.rmtree(out_name)


# --- MAIN reads: row reads vs a TaQL reference ----------------------------------

MAIN_LAYOUTS = ("dense", "sparse_dup", "baseline_major")


def _rewrite_main_rows(
    msname: str, layout: str, seed: int, weight: bool = True
) -> None:
    """
    Rewrite the MAIN rows of a generated MS: every DDI gets 30 times x 10
    baselines (time-major or baseline-major rows), random data, flags, weights
    (unless ``weight`` is False: the WEIGHT cells stay undefined) and UVW.
    "sparse_dup" moves 10% of the rows to 3 extra, sparsely filled times
    (leaving their cells empty) and gives some rows the (time, baseline) of the
    row before (duplicated cells). The generated MS uses TiledColumnStMan,
    which cannot remove rows.
    """
    from casacore import tables

    rng = np.random.default_rng(seed)
    with tables.table(msname, readonly=False, ack=False) as main_tb:
        nrows = main_tb.nrows()
        ddi = main_tb.getcol("DATA_DESC_ID")
        assert np.all(np.diff(ddi) >= 0)
        pos = np.arange(nrows) - np.searchsorted(ddi, ddi)  # row index in its DDI
        ant1, ant2 = np.triu_indices(5, 1)
        nbl = ant1.size
        ntimes = int(pos.max() + 1) // nbl
        if layout == "baseline_major":
            tidx, bidx = pos % ntimes, pos // ntimes
        else:
            tidx, bidx = pos // nbl, pos % nbl
        if layout == "sparse_dup":
            moved = rng.choice(nrows, nrows // 10, replace=False)
            tidx[moved] = ntimes + rng.integers(0, 3, moved.size)
            dup = np.concatenate(
                [
                    rng.choice(np.flatnonzero((pos > 0) & (ddi == d)), 4, replace=False)
                    for d in np.unique(ddi)
                ]
            )
            tidx[dup], bidx[dup] = tidx[dup - 1], bidx[dup - 1]
            for d in np.unique(ddi):  # every DDI has duplicated (time, baseline) cells
                cells = tidx[ddi == d] * nbl + bidx[ddi == d]
                assert np.unique(cells).size < cells.size
        time0 = main_tb.getcell("TIME", 0)
        main_tb.putcol("TIME", time0 + tidx.astype(float))
        main_tb.putcol("TIME_CENTROID", time0 + tidx + rng.random(nrows) * 0.1)
        main_tb.putcol("ANTENNA1", ant1[bidx].astype(np.int32))
        main_tb.putcol("ANTENNA2", ant2[bidx].astype(np.int32))
        main_tb.putcol("EXPOSURE", rng.random(nrows))
        main_tb.putcol("UVW", rng.normal(size=(nrows, 3)))
        cell = main_tb.getcell("DATA", 0).shape
        for col in ("DATA", "CORRECTED_DATA"):
            values = rng.normal(size=(nrows,) + cell) + 1j * rng.normal(
                size=(nrows,) + cell
            )
            main_tb.putcol(col, values.astype(np.complex64))
        main_tb.putcol("FLAG", rng.random((nrows,) + cell) < 0.3)
        if weight:
            main_tb.putcol("WEIGHT", rng.random((nrows, cell[1])).astype(np.float32))


@pytest.fixture(scope="module")
def ms_main_layouts(tmp_path_factory):
    """Generated MSs whose MAIN rows have the layouts of MAIN_LAYOUTS."""
    from xradio.testing.measurement_set.msv2_io import default_ms_descr, gen_test_ms

    base = tmp_path_factory.mktemp("ms_main_layouts")
    descr = dict(default_ms_descr, data_cols=["DATA", "CORRECTED_DATA"])
    paths = {}
    for seed, layout in enumerate(MAIN_LAYOUTS):
        msname = str(base / f"layout_{layout}.ms")
        gen_test_ms(
            msname,
            descr=descr,
            opt_tables=True,
            vlbi_tables=False,
            required_only=True,
            misbehave=False,
        )
        _rewrite_main_rows(msname, layout, seed)
        paths[layout] = msname
    yield paths
    shutil.rmtree(base, ignore_errors=True)


def _without_dates(attrs):
    """attrs as comparable JSON, without the creation dates (they differ per run)."""
    import json

    def strip(obj):
        if isinstance(obj, dict):
            return {
                k: strip(v)
                for k, v in obj.items()
                if k not in ("creation_date", "date")
            }
        if isinstance(obj, list | tuple):
            return [strip(v) for v in obj]
        return obj

    return json.dumps(strip(attrs), sort_keys=True, default=str)


def assert_msv4_bit_identical(xdt_a: xr.DataTree, xdt_b: xr.DataTree) -> None:
    """Same nodes, variables (bitwise values, dtypes, dims), attrs and chunks."""
    assert {node.path for node in xdt_a.subtree} == {
        node.path for node in xdt_b.subtree
    }
    for node in xdt_a.subtree:
        ds_a = node.to_dataset(inherit=False)
        ds_b = xdt_b[node.path].to_dataset(inherit=False)
        assert list(ds_a.variables) == list(ds_b.variables), node.path
        assert _without_dates(ds_a.attrs) == _without_dates(ds_b.attrs), node.path
        for name, var_a in ds_a.variables.items():
            var_b = ds_b.variables[name]
            where = f"{node.path}/{name}"
            assert var_a.dims == var_b.dims, where
            assert var_a.dtype == var_b.dtype, where
            assert var_a.encoding.get("chunks") == var_b.encoding.get("chunks"), where
            values_a, values_b = var_a.values, var_b.values
            if values_a.dtype == object:
                np.testing.assert_array_equal(values_a, values_b, err_msg=where)
            else:
                assert values_a.tobytes() == values_b.tobytes(), where
            assert _without_dates(var_a.attrs) == _without_dates(var_b.attrs), where


def _convert_partition(
    monkeypatch, msname, out_file, partition_info, stream=True, **kw
):
    """Convert one partition; with stream=False without the streamed write
    (the data variables read whole and written by one to_zarr)."""
    kw.setdefault("use_table_iter", False)
    if not stream:
        kw["allow_stream_write"] = False
    convert = (
        conversion.convert_and_write_partition
        if stream
        else conversion._convert_and_write_partition
    )
    convert(
        in_file=msname,
        out_file=out_file,
        ms_v4_id="0",
        partition_info=partition_info,
        persistence_mode="w",
        **kw,
    )
    msv4_name = pathlib.Path(msname).name.replace(".ms", "") + "_0"
    return xr.open_datatree(os.path.join(out_file, msv4_name), engine="zarr")


def _convert_partition_reference(
    monkeypatch, reference, msname, out_file, partition_info, **kw
):
    """
    Convert a partition with the MAIN data columns read by a TaQL selection
    of the partition (``reference``, the taql_main_reference fixture: the
    read path of earlier versions) instead of the row reads, read whole and
    written by one to_zarr (no streamed write).
    """

    def taql_read(main_rows, col, cshape, tidxs, bidxs, *args):
        grid = reference(msname, partition_info, col)
        assert grid.shape[:2] == tuple(cshape)
        return grid

    with monkeypatch.context() as m:
        m.setattr(conversion, "read_col_conversion_numpy", taql_read)
        kw.setdefault("use_table_iter", False)
        conversion._convert_and_write_partition(
            in_file=msname,
            out_file=out_file,
            ms_v4_id="0",
            partition_info=partition_info,
            persistence_mode="w",
            allow_stream_write=False,
            **kw,
        )
    msv4_name = pathlib.Path(msname).name.replace(".ms", "") + "_0"
    return xr.open_datatree(os.path.join(out_file, msv4_name), engine="zarr")


@pytest.mark.parametrize("layout", MAIN_LAYOUTS)
@pytest.mark.parametrize(
    "partition_source", ["create_partitions", "hand_built", "mismatched_runs"]
)
def test_convert_and_write_partition_matches_taql_reference(
    ms_main_layouts,
    layout,
    partition_source,
    tmp_path,
    monkeypatch,
    taql_main_reference,
):
    """
    The row reads (default: streamed write) give exactly the MSv4 of the TaQL
    selection's reads (read whole, one to_zarr), for the rows of the
    partition descriptions (with or without row runs, or with runs of another
    description).
    """
    from xradio.measurement_set._utils._msv2.partition_queries import (
        create_partitions_with_main_rows,
    )

    msname = ms_main_layouts[layout]
    partitions, runs = create_partitions_with_main_rows(msname, [])
    if partition_source == "create_partitions":
        partition, kw = partitions[1], {"main_row_runs": runs[1]}
    elif partition_source == "hand_built":
        # without row runs: rows from the numpy twin of the TaQL selection
        partition = {"DATA_DESC_ID": [1], "OBS_MODE": ["scan_intent#subscan_intent"]}
        kw = {}
    else:  # runs of another description: not used (the rows follow the dict)
        partition, kw = partitions[1], {"main_row_runs": runs[0]}

    taql = _convert_partition_reference(
        monkeypatch, taql_main_reference, msname, str(tmp_path / "t"), partition, **kw
    )
    rows = _convert_partition(monkeypatch, msname, str(tmp_path / "r"), partition, **kw)

    assert {"VISIBILITY", "VISIBILITY_CORRECTED", "FLAG", "WEIGHT", "UVW"} <= set(
        rows.ds.data_vars
    )
    assert rows.ds.WEIGHT.dtype == np.float32  # read from WEIGHT, not the ones fallback
    if layout == "sparse_dup":
        assert np.isnan(rows.ds.VISIBILITY.values).any()  # padded missing cells
    assert_msv4_bit_identical(taql, rows)


@pytest.mark.parametrize("layout", MAIN_LAYOUTS)
def test_convert_and_write_partition_time_mode(
    ms_main_layouts, layout, tmp_path, monkeypatch
):
    """parallel_mode="time": same output as parallel_mode="none", also for
    sparse, duplicated and baseline-major rows."""
    from xradio.measurement_set._utils._msv2.partition_queries import (
        create_partitions_with_main_rows,
    )

    msname = ms_main_layouts[layout]
    partitions, runs = create_partitions_with_main_rows(msname, [])
    partition = partitions[0]
    chunks = {"time": 4}
    none = _convert_partition(
        monkeypatch,
        msname,
        str(tmp_path / "n"),
        partition,
        main_chunksize=chunks,
        main_row_runs=runs[0],
    )
    timed = _convert_partition(
        monkeypatch,
        msname,
        str(tmp_path / "t"),
        partition,
        main_chunksize=chunks,
        parallel_mode="time",
        main_row_runs=runs[0],
    )
    assert_msv4_bit_identical(none, timed)


def _to_tiled_shape_columns(main_tb, columns: dict, cell: tuple, tile_rows: int):
    """Replace MAIN columns by TiledShapeStMan columns (tile_rows rows per tile)
    holding the given values."""
    from casacore import tables

    main_tb.removecols([col for col in columns if col in main_tb.colnames()])
    for col, values in columns.items():
        group = f"TSM_{col}"
        desc = tables.makearrcoldesc(
            col,
            values.flat[0],
            ndim=2,
            valuetype={"b": "boolean", "f": "float", "c": "complex"}[values.dtype.kind],
            datamanagertype="TiledShapeStMan",
            datamanagergroup=group,
        )
        tile = np.array([cell[1], cell[0], tile_rows], dtype=np.int32)
        main_tb.addcols(
            tables.maketabdesc([desc]),
            dminfo={
                "TYPE": "TiledShapeStMan",
                "NAME": group,
                "SPEC": {"DEFAULTTILESHAPE": tile},
            },
        )
        main_tb.putcol(col, values)


@pytest.fixture(scope="module")
def ms_tiled_shape_main(tmp_path_factory):
    """
    Generated MSs whose MAIN data columns are TiledShapeStMan columns with
    7-row tiles (tiles hold rows of several partitions), with interleaved
    FIELD_IDs:

    - "interferometer": DATA, CORRECTED_DATA, FLAG; every 5th row an
      autocorrelation.
    - "single_dish": FLOAT_DATA, FLAG; autocorrelations only (ANTENNA1
      partitions).
    """
    from casacore import tables

    from xradio.testing.measurement_set.msv2_io import default_ms_descr, gen_test_ms

    base = tmp_path_factory.mktemp("ms_tiled_shape_main")
    paths = {}
    for variant in ("interferometer", "single_dish"):
        msname = str(base / f"tsm_{variant}.ms")
        gen_test_ms(
            msname,
            descr=dict(default_ms_descr, data_cols=["DATA", "CORRECTED_DATA"]),
            opt_tables=True,
            vlbi_tables=False,
            required_only=True,
            misbehave=False,
        )
        rng = np.random.default_rng(11)
        with tables.table(msname, readonly=False, ack=False) as main_tb:
            nrows = main_tb.nrows()
            rows = np.arange(nrows)
            cell = main_tb.getcell("DATA", 0).shape
            main_tb.putcol("FIELD_ID", ((rows // 13) % 2).astype(np.int32))
            ant1, ant2 = main_tb.getcol("ANTENNA1"), main_tb.getcol("ANTENNA2")
            auto = rows % 5 == 0 if variant == "interferometer" else rows >= 0
            ant2[auto] = ant1[auto]
            main_tb.putcol("ANTENNA2", ant2)
            visibilities = (
                rng.normal(size=(nrows,) + cell) + 1j * rng.normal(size=(nrows,) + cell)
            ).astype(np.complex64)
            columns = {"FLAG": rng.random((nrows,) + cell) < 0.3}
            if variant == "interferometer":
                columns["DATA"] = visibilities
                columns["CORRECTED_DATA"] = (visibilities * 2).astype(np.complex64)
            else:
                main_tb.removecols(["DATA", "CORRECTED_DATA"])
                columns["FLOAT_DATA"] = visibilities.real.astype(np.float32)
            _to_tiled_shape_columns(main_tb, columns, cell, tile_rows=7)
        paths[variant] = msname
    yield paths
    shutil.rmtree(base, ignore_errors=True)


@pytest.mark.parametrize(
    "variant, scheme",
    [
        ("interferometer", []),
        ("interferometer", ["FIELD_ID"]),
        ("interferometer", ["FIELD_ID", "SCAN_NUMBER"]),
        ("single_dish", ["ANTENNA1"]),
        ("single_dish", ["FIELD_ID", "ANTENNA1"]),
    ],
)
def test_convert_and_write_partition_tiled_shape_main_matches_taql_reference(
    ms_tiled_shape_main, variant, scheme, tmp_path, monkeypatch, taql_main_reference
):
    """
    TiledShapeStMan MAIN columns whose tiles hold rows of several partitions,
    with the FIELD_ID / ANTENNA1 partition schemes: the row reads give exactly
    the output of the TaQL selection's reads.
    """
    from casacore import tables

    from xradio.measurement_set._utils._msv2.partition_queries import (
        create_partitions_with_main_rows,
    )

    msname = ms_tiled_shape_main[variant]
    with tables.table(msname, ack=False) as main_tb:
        data_col = "DATA" if variant == "interferometer" else "FLOAT_DATA"
        assert main_tb.getdminfo(data_col)["TYPE"] == "TiledShapeStMan"
    partitions, runs = create_partitions_with_main_rows(msname, scheme)
    converted = 0
    for idx in range(min(len(partitions), 3)):
        kw = {"main_row_runs": runs[idx], "with_pointing": False}
        out = {}
        for main_read in ("taql", "rows"):
            try:
                if main_read == "taql":
                    out[main_read] = _convert_partition_reference(
                        monkeypatch,
                        taql_main_reference,
                        msname,
                        str(tmp_path / f"{main_read}{idx}"),
                        partitions[idx],
                        **kw,
                    )
                else:
                    out[main_read] = _convert_partition(
                        monkeypatch,
                        msname,
                        str(tmp_path / f"{main_read}{idx}"),
                        partitions[idx],
                        **kw,
                    )
            except Exception as exc:
                out[main_read] = f"{type(exc).__name__}: {exc}"
        if isinstance(out["taql"], str) or isinstance(out["rows"], str):
            assert out["taql"] == out["rows"]
            continue
        assert_msv4_bit_identical(out["taql"], out["rows"])
        converted += 1
    assert converted >= 2
    if scheme == []:  # time mode on tiled-shape columns too
        timed = _convert_partition(
            monkeypatch,
            msname,
            str(tmp_path / "time"),
            partitions[0],
            main_chunksize={"time": 4},
            parallel_mode="time",
            **kw | {"main_row_runs": runs[0]},
        )
        none = _convert_partition(
            monkeypatch,
            msname,
            str(tmp_path / "none"),
            partitions[0],
            main_chunksize={"time": 4},
            **kw | {"main_row_runs": runs[0]},
        )
        assert_msv4_bit_identical(none, timed)


def test_convert_and_write_partition_no_taql_on_main(
    ms_main_layouts, tmp_path, monkeypatch
):
    import sys

    from casacore import tables

    from xradio.measurement_set._utils._msv2.partition_queries import (
        create_partitions_with_main_rows,
    )

    msname = ms_main_layouts["dense"]
    partitions, runs = create_partitions_with_main_rows(msname, [])
    partition = partitions[0]
    queries = []
    taql = tables.taql

    def spy_taql(query, *args, **kwargs):
        queries.append(query)
        if not args and "locals" not in kwargs:
            # $name substitution looks at the caller's local variables
            kwargs["locals"] = sys._getframe(1).f_locals
        return taql(query, *args, **kwargs)

    monkeypatch.setattr(tables, "taql", spy_taql)
    for parallel_mode in ("none", "time"):
        queries.clear()
        _convert_partition(
            monkeypatch,
            msname,
            str(tmp_path / parallel_mode),
            partition,
            main_row_runs=runs[0],
            main_chunksize={"time": 4},
            parallel_mode=parallel_mode,
        )
        assert queries  # sub-tables are still read with TaQL
        assert not [q for q in queries if "$mtable" in q]
    # nor in the memory estimate of the partitions
    queries.clear()
    conversion.estimate_memory_and_cores_for_partitions(msname, partitions, runs)
    conversion.estimate_memory_and_cores_for_partitions(msname, partitions)
    assert queries == []


@pytest.mark.parametrize(
    "col_name, parallel_mode, expected",
    [
        ("DATA", "none", "read_col_conversion_numpy"),
        ("DATA", "time", "read_col_conversion_dask"),
        ("FLAG", "time", "read_col_conversion_dask"),
        ("UVW", "time", "read_col_conversion_numpy"),
        ("TIME_CENTROID", "time", "read_col_conversion_numpy"),
        ("WEIGHT", "partition", "read_col_conversion_numpy"),
    ],
)
def test_get_read_col_conversion_function(col_name, parallel_mode, expected):
    func = conversion.get_read_col_conversion_function(col_name, parallel_mode)
    assert func.__name__ == expected


@pytest.mark.parametrize("stream", ["0", "1"])
def test_create_data_variables_reads_columns_in_sorted_order(
    ms_main_layouts, stream, tmp_path, monkeypatch
):
    """The read order (and with it the memory peak and the order of the data
    variables) does not depend on the hash seed."""
    from xradio.measurement_set._utils._msv2.partition_queries import (
        create_partitions,
    )

    read_cols = []
    if stream == "0":
        get_function = conversion.get_read_col_conversion_function

        def spy(col_name, *args, **kwargs):
            read_cols.append(col_name)
            return get_function(col_name, *args, **kwargs)

        monkeypatch.setattr(conversion, "get_read_col_conversion_function", spy)
    else:  # the columns are checked (and later written) in the same order
        deferred_column = conversion.deferred_main_column

        def spy(main_rows, col, *args, **kwargs):
            read_cols.append(col)
            return deferred_column(main_rows, col, *args, **kwargs)

        monkeypatch.setattr(conversion, "deferred_main_column", spy)
    msname = ms_main_layouts["dense"]
    partition = create_partitions(msname, [])[0]
    _convert_partition(
        monkeypatch, msname, str(tmp_path / "r"), partition, stream=stream == "1"
    )
    assert read_cols == sorted(read_cols)
    assert "DATA" in read_cols and "WEIGHT" in read_cols


@pytest.mark.parametrize("parallel_mode", ["none", "time"])
def test_create_data_variables_releases_the_row_plans(
    ms_main_layouts, parallel_mode, tmp_path, monkeypatch
):
    """The grid plan / time-chunk rows (8-24 bytes per row) are not kept
    alive through to_zarr."""
    from xradio.measurement_set._utils._msv2._tables.read_rows import MainTableRows
    from xradio.measurement_set._utils._msv2.partition_queries import (
        create_partitions,
    )

    released = []
    create = conversion.create_data_variables

    def spy(in_file, xds, table_manager, *args, **kwargs):
        create(in_file, xds, table_manager, *args, **kwargs)
        assert isinstance(table_manager, MainTableRows)
        released.append(
            table_manager._grid_plan is None and table_manager._time_chunk_rows is None
        )

    monkeypatch.setattr(conversion, "create_data_variables", spy)
    msname = ms_main_layouts["baseline_major"]
    partition = create_partitions(msname, [])[0]
    _convert_partition(
        monkeypatch,
        msname,
        str(tmp_path / "r"),
        partition,
        main_chunksize={"time": 4},
        parallel_mode=parallel_mode,
    )
    assert released == [True]


# --- sub-table cache --------------------------------------------------------------


@pytest.mark.parametrize("ms_fixture", ["ms_minimal_required", "ms_minimal_misbehaved"])
@pytest.mark.parametrize("interpolate", [False, True])
def test_convert_and_write_partition_subtable_cache_bit_identical(
    ms_fixture, interpolate, tmp_path, monkeypatch, request
):
    """Every partition converts to exactly the same MSv4 (or fails the same way)
    with the sub-table cache shared by the partitions (POINTING, SYSCAL, WEATHER,
    PHASE_CAL, GAIN_CURVE, ephemerides, ...) as with per-partition reads."""
    from xradio.measurement_set._utils._msv2._tables.subtable_cache import (
        SubtableCache,
    )
    from xradio.measurement_set._utils._msv2.partition_queries import (
        create_partitions,
    )

    msname = request.getfixturevalue(ms_fixture).fname
    partitions = create_partitions(msname, ["FIELD_ID"])
    cache = SubtableCache(n_partitions=len(partitions))
    resolve = conversion.resolve_subtable_cache

    def no_cache(subtable_cache):
        return None

    msv4_name = pathlib.Path(msname).name.replace(".ms", "") + "_0"
    n_converted = 0
    for idx, partition in enumerate(partitions):
        results = {}
        for mode in ("0", "1"):
            if mode == "0":  # no cache: every sub-table read per partition
                monkeypatch.setattr(conversion, "resolve_subtable_cache", no_cache)
            else:
                monkeypatch.setattr(conversion, "resolve_subtable_cache", resolve)
            out_file = str(tmp_path / f"p{idx}_cache{mode}")
            try:
                conversion.convert_and_write_partition(
                    in_file=msname,
                    out_file=out_file,
                    ms_v4_id="0",
                    partition_info=partition,
                    use_table_iter=False,
                    pointing_interpolate=interpolate,
                    ephemeris_interpolate=interpolate,
                    phase_cal_interpolate=interpolate,
                    sys_cal_interpolate=interpolate,
                    persistence_mode="w",
                    subtable_cache=cache,  # unused without cache (mode "0")
                )
            except Exception as exc:  # e.g. PHASE_CAL rows missing for a SPW
                results[mode] = f"{type(exc).__name__}: {exc}"
                continue
            results[mode] = xr.open_datatree(
                os.path.join(out_file, msv4_name), engine="zarr"
            )
        if isinstance(results["0"], str):
            assert results["1"] == results["0"]
            continue
        n_converted += 1
        assert "/pointing_xds" in {node.path for node in results["1"].subtree}
        assert_msv4_bit_identical(results["0"], results["1"])
    assert n_converted > 1
    assert cache.stats["pointing_cached"] == n_converted
    assert cache.stats["memo_hits"] > 0


# --- streamed write of the MAIN data variables -------------------------------------


def _store_contents(path: str) -> tuple[dict, dict]:
    """
    The files of a zarr store: the sha256 of every chunk file, and every
    zarr.json (array / group metadata, consolidated metadata) as JSON without
    the creation dates (they differ per run).
    """
    import hashlib
    import json

    chunks, metadata = {}, {}
    for dirpath, _, filenames in os.walk(path):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            rel = os.path.relpath(file_path, path)
            with open(file_path, "rb") as f:
                content = f.read()
            if filename == "zarr.json":
                metadata[rel] = _without_dates(json.loads(content))
            else:
                chunks[rel] = hashlib.sha256(content).hexdigest()
    return chunks, metadata


def assert_stores_identical(path_a: str, path_b: str) -> None:
    """Byte-identical chunk files, same metadata except dates."""
    chunks_a, metadata_a = _store_contents(path_a)
    chunks_b, metadata_b = _store_contents(path_b)
    assert sorted(chunks_a) == sorted(chunks_b)
    assert [name for name in chunks_a if chunks_a[name] != chunks_b[name]] == []
    assert sorted(metadata_a) == sorted(metadata_b)
    for name in metadata_a:
        assert metadata_a[name] == metadata_b[name], name


DEFAULT_STREAM_BATCH_BYTES = stream_write.STREAM_BATCH_BYTES


def _set_stream_batch_mb(monkeypatch, batch_mb):
    """Set the target batch size of the streamed write (MiB; None: the
    default)."""
    value = (
        DEFAULT_STREAM_BATCH_BYTES
        if batch_mb is None
        else max(1, int(batch_mb * 2**20))
    )
    monkeypatch.setattr(stream_write, "STREAM_BATCH_BYTES", value)


def _convert_streamed(
    monkeypatch, msname, out_file, partition_info, stream, batch_mb=None, **kw
):
    """Convert one partition with ("1") or without ("0") the streamed write;
    returns the MSv4 and its store path."""
    _set_stream_batch_mb(monkeypatch, batch_mb)
    xdt = _convert_partition(
        monkeypatch, msname, out_file, partition_info, stream=stream == "1", **kw
    )
    msv4_name = pathlib.Path(msname).name.replace(".ms", "") + "_0"
    return xdt, os.path.join(out_file, msv4_name)


@pytest.fixture
def stream_stats(monkeypatch):
    """The statistics returned by every write_deferred_variables (streamed)
    and read_deferred_variables (partition read whole) call."""
    recorded = []

    def spy_on(name):
        function = getattr(conversion, name)

        def spy(*args, **kwargs):
            stats = function(*args, **kwargs)
            recorded.append(stats)
            return stats

        monkeypatch.setattr(conversion, name, spy)

    spy_on("write_deferred_variables")
    spy_on("read_deferred_variables")
    return recorded


STREAM_CHUNKS = {
    "one_chunk": None,  # the default: one chunk per variable
    "time4": {"time": 4},  # uneven last chunk
    "time1": {"time": 1},
    "balanced": 2e-6,  # GiB: chunks along time, baseline and frequency
}
# one chunk per batch, a few chunks per batch, the default (one batch here)
STREAM_BATCH_MB = (1e-9, 0.02, None)


@pytest.mark.parametrize("chunks", list(STREAM_CHUNKS))
@pytest.mark.parametrize("layout", MAIN_LAYOUTS)
def test_stream_write_bit_identical(
    ms_main_layouts, layout, chunks, tmp_path, monkeypatch, stream_stats
):
    """
    The streamed write gives byte-identical chunk files and the same metadata
    (except dates) as reading everything and one to_zarr: dense, sparse with
    duplicated (time, baseline) rows and baseline-major rows, several chunk
    layouts and batch sizes.
    """
    from xradio.measurement_set._utils._msv2 import stream_write
    from xradio.measurement_set._utils._msv2.partition_queries import (
        create_partitions_with_main_rows,
    )

    msname = ms_main_layouts[layout]
    partitions, runs = create_partitions_with_main_rows(msname, [])
    kw = {"main_chunksize": STREAM_CHUNKS[chunks], "main_row_runs": runs[1]}
    old_xdt, old = _convert_streamed(
        monkeypatch, msname, str(tmp_path / "old"), partitions[1], "0", **kw
    )
    for idx, batch_mb in enumerate(STREAM_BATCH_MB):
        _, new = _convert_streamed(
            monkeypatch,
            msname,
            str(tmp_path / f"new{idx}"),
            partitions[1],
            "1",
            batch_mb,
            **kw,
        )
        assert_stores_identical(old, new)
    assert len(stream_stats) == len(STREAM_BATCH_MB)

    n_times = old_xdt.ds.sizes["time"]
    time_chunk = old_xdt.ds.VISIBILITY.encoding["chunks"][0]
    expected = {
        "VISIBILITY",
        "VISIBILITY_CORRECTED",
        "FLAG",
        "WEIGHT",
        "UVW",
        "TIME_CENTROID",
        "EFFECTIVE_INTEGRATION_TIME",
    }
    for stats, batch_mb in zip(stream_stats, STREAM_BATCH_MB, strict=True):
        assert set(stats["variables"]) == expected
        vis = stats["variables"]["VISIBILITY"]
        assert vis["calls"] >= 1 and vis["direct_rows"] + vis["scatter_rows"] > 0
        target = (
            DEFAULT_STREAM_BATCH_BYTES if batch_mb is None else int(batch_mb * 2**20)
        )
        batches = stream_write.time_batches(
            n_times, time_chunk, vis["bytes"] // n_times, max(1, target)
        )
        fits_budget = vis["bytes"] <= stream_write.FRAGMENTED_BATCH_FACTOR * target
        if len(batches) == 1:
            assert vis["batches"] == 1 and vis["guard"] == "one batch"
        elif layout == "dense":  # time-ordered: one row run per batch
            assert vis["guard"] == "time" and vis["batches"] == len(batches)
            assert vis["runs_batched"] == vis["runs_whole"] + len(batches) - 1
        elif layout == "baseline_major":  # the small-read guard
            assert vis["runs_whole"] == 1
            if fits_budget:
                assert vis["guard"] == "fragmented: one pass" and vis["batches"] == 1
            else:
                assert vis["guard"] == "fragmented: large batches"
        else:  # sparse_dup: the rows of the extra times are spread over the partition
            assert vis["guard"] in (
                "time",
                "fragmented: one pass",
                "fragmented: large batches",
            )
    # default batch size: the partition is a fraction of a batch, read whole
    assert stream_stats[-1]["summary"]["in_memory"]
    assert [s["summary"]["in_memory"] for s in stream_stats[:-1]] == [False, False]
    for stats in stream_stats[-1]["variables"].values():
        assert stats["batches"] == 1
    # one chunk per batch on time-ordered rows
    if layout == "dense":
        vis = stream_stats[0]["variables"]["VISIBILITY"]
        assert vis["batches"] == -(-n_times // time_chunk)


@pytest.mark.parametrize(
    "layout, guard",
    [
        ("baseline_major", "forced_time_batches"),
        ("sparse_dup", "forced_time_batches"),
        ("baseline_major", "large_batches"),
    ],
)
def test_stream_write_fragmented_rows_bit_identical(
    ms_main_layouts, layout, guard, tmp_path, monkeypatch, stream_stats
):
    """Rows scattered over time batches (baseline-major, sparse with duplicated
    cells) read in time batches (guard disabled) or in the larger batches of
    the guard: still byte-identical."""
    from xradio.measurement_set._utils._msv2 import stream_write
    from xradio.measurement_set._utils._msv2.partition_queries import (
        create_partitions_with_main_rows,
    )

    msname = ms_main_layouts[layout]
    partitions, runs = create_partitions_with_main_rows(msname, [])
    kw = {"main_chunksize": {"time": 4}, "main_row_runs": runs[2]}
    old_xdt, old = _convert_streamed(
        monkeypatch, msname, str(tmp_path / "old"), partitions[2], "0", **kw
    )
    n_chunks = -(-old_xdt.ds.sizes["time"] // 4)
    if guard == "forced_time_batches":
        monkeypatch.setattr(stream_write, "FRAGMENTED_RUNS_RATIO", np.inf)
        monkeypatch.setattr(stream_write, "FRAGMENTED_EXTRA_READS", np.inf)
    else:
        monkeypatch.setattr(stream_write, "FRAGMENTED_BATCH_FACTOR", 3)
    # 0.01 MiB: one 10240-byte VISIBILITY chunk (4 times x 10 baselines x 16 x 2)
    _, new = _convert_streamed(
        monkeypatch, msname, str(tmp_path / "new"), partitions[2], "1", 0.01, **kw
    )
    assert_stores_identical(old, new)
    vis = stream_stats[-1]["variables"]["VISIBILITY"]
    if guard == "forced_time_batches":
        assert vis["guard"] == "time" and vis["batches"] == n_chunks
        assert vis["runs_batched"] > 2 * vis["runs_whole"]
    else:  # 3 chunks per batch
        assert vis["guard"] == "fragmented: large batches"
        assert vis["batches"] == -(-n_chunks // 3)


@pytest.mark.parametrize(
    "variant, scheme",
    [
        ("interferometer", []),
        ("interferometer", ["FIELD_ID"]),
        ("single_dish", ["ANTENNA1"]),
    ],
)
def test_stream_write_tiled_shape_main_bit_identical(
    ms_tiled_shape_main, variant, scheme, tmp_path, monkeypatch
):
    """TiledShapeStMan MAIN columns (tiles shared by partitions, interleaved
    FIELD_IDs) and single dish (antenna_name, no UVW)."""
    from xradio.measurement_set._utils._msv2.partition_queries import (
        create_partitions_with_main_rows,
    )

    msname = ms_tiled_shape_main[variant]
    partitions, runs = create_partitions_with_main_rows(msname, scheme)
    converted = 0
    for idx in range(min(len(partitions), 2)):
        kw = {
            "main_row_runs": runs[idx],
            "with_pointing": False,
            "main_chunksize": {"time": 3},
        }
        out = {}
        for stream in ("0", "1"):
            try:
                xdt, out[stream] = _convert_streamed(
                    monkeypatch,
                    msname,
                    str(tmp_path / f"s{stream}_{idx}"),
                    partitions[idx],
                    stream,
                    1e-9,
                    **kw,
                )
            except Exception as exc:
                out[stream] = f"{type(exc).__name__}: {exc}"
        if out["0"].startswith(str(tmp_path)) and out["1"].startswith(str(tmp_path)):
            assert_stores_identical(out["0"], out["1"])
            converted += 1
            if variant == "single_dish":
                assert "antenna_name" in xdt.ds.SPECTRUM.dims and "UVW" not in xdt.ds
        else:
            assert out["0"] == out["1"]
    assert converted >= 1


STREAM_EDGE_VARIANTS = (
    "reversed_freq",
    "wsp_partial",
    "no_weight",
    "varying_shape",
    "interleaved_fields",
)


def _add_tiled_shape_column(main_tb, col, values, tile_rows, rows=None):
    """Add a TiledShapeStMan column; only ``rows`` (all by default) are written,
    the other cells stay undefined."""
    from casacore import tables

    desc = tables.makearrcoldesc(
        col,
        values.flat[0],
        ndim=2,
        valuetype={"f": "float", "c": "complex"}[values.dtype.kind],
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


@pytest.fixture(scope="module")
def ms_stream_edges(tmp_path_factory):
    """
    Generated MSs (dense, time-ordered rows, see _rewrite_main_rows) for the
    column decisions of the streamed write:

    - "reversed_freq": decreasing CHAN_FREQ, WEIGHT_SPECTRUM (TiledShapeStMan)
    - "wsp_partial": WEIGHT_SPECTRUM with undefined cells in the middle of
      every partition (falls back to WEIGHT)
    - "no_weight": no WEIGHT_SPECTRUM and undefined WEIGHT cells (WEIGHT=1)
    - "varying_shape": MODEL_DATA (StandardStMan) with cells of another shape
      in every partition (VISIBILITY_MODEL is dropped); in "reversed_freq"
      all its cells have one shape
    - "interleaved_fields": baseline-major rows whose FIELD_ID alternates time
      by time (the FIELD_ID partitions interleave row by row inside every
      baseline), WEIGHT_SPECTRUM in TiledShapeStMan tiles of 7 rows

    "wsp_partial" also has decreasing CHAN_FREQ (WEIGHT from the WEIGHT column,
    the same along frequency, is not reversed). "reftable_wsp_partial" is a
    persistent reference table (selection) of the DDI 1 rows of
    "wsp_partial", with copies of its sub-tables.
    """
    from casacore import tables

    from xradio.testing.measurement_set.msv2_io import default_ms_descr, gen_test_ms

    base = tmp_path_factory.mktemp("ms_stream_edges")
    paths = {}
    for seed, variant in enumerate(STREAM_EDGE_VARIANTS):
        msname = str(base / f"edge_{variant}.ms")
        gen_test_ms(
            msname,
            descr=dict(default_ms_descr, data_cols=["DATA", "CORRECTED_DATA"]),
            opt_tables=True,
            vlbi_tables=False,
            required_only=True,
            misbehave=False,
        )
        layout = "baseline_major" if variant == "interleaved_fields" else "dense"
        _rewrite_main_rows(msname, layout, seed, weight=variant != "no_weight")
        rng = np.random.default_rng(100 + seed)
        with tables.table(msname, readonly=False, ack=False) as main_tb:
            nrows = main_tb.nrows()
            cell = main_tb.getcell("DATA", 0).shape
            ddi = main_tb.getcol("DATA_DESC_ID")
            pos = np.arange(nrows) - np.searchsorted(ddi, ddi)  # row index in its DDI
            weights = rng.random((nrows,) + cell).astype(np.float32)
            if variant in ("reversed_freq", "interleaved_fields"):
                _add_tiled_shape_column(main_tb, "WEIGHT_SPECTRUM", weights, 7)
            if variant == "interleaved_fields":
                tidx = np.unique(main_tb.getcol("TIME"), return_inverse=True)[1]
                main_tb.putcol("FIELD_ID", (tidx % 2).astype(np.int32))
            elif variant == "wsp_partial":
                defined = np.flatnonzero((pos < 100) | (pos >= 110))
                _add_tiled_shape_column(main_tb, "WEIGHT_SPECTRUM", weights, 7, defined)
            if variant in ("reversed_freq", "varying_shape"):
                # MODEL_DATA in StandardStMan (indirect arrays: cells compared)
                model = (weights + 1j * weights[::-1]).astype(np.complex64)
                desc = tables.makearrcoldesc(
                    "MODEL_DATA", 0j, ndim=2, valuetype="complex"
                )
                main_tb.addcols(tables.maketabdesc([desc]))
                main_tb.putcol("MODEL_DATA", model)
                if variant == "varying_shape":
                    for row in np.flatnonzero((pos >= 50) & (pos < 53)):
                        main_tb.putcell("MODEL_DATA", int(row), model[row][:, :1])
        if variant in ("reversed_freq", "wsp_partial"):
            with tables.table(
                os.path.join(msname, "SPECTRAL_WINDOW"), readonly=False, ack=False
            ) as spw_tb:
                n_spw, n_chan = spw_tb.getcol("CHAN_FREQ").shape
                decreasing = 1.0e9 + 1.0e6 * np.arange(n_chan)[::-1]
                spw_tb.putcol("CHAN_FREQ", np.tile(decreasing, (n_spw, 1)))
        paths[variant] = msname
    # a persistent reference table: getdminfo describes its root table
    root = paths["wsp_partial"]
    ref_path = str(base / "edge_reftable_wsp_partial.ms")
    with tables.table(root, ack=False) as root_tb:
        ddi_rows = np.flatnonzero(root_tb.getcol("DATA_DESC_ID") == 1)
        selection = root_tb.selectrows(ddi_rows)
        selection.copy(ref_path, deep=False).close()
        selection.close()
    for name in os.listdir(root):
        sub = os.path.join(root, name)
        if os.path.isdir(sub) and not os.path.exists(os.path.join(ref_path, name)):
            shutil.copytree(sub, os.path.join(ref_path, name))
    paths["reftable_wsp_partial"] = ref_path
    yield paths
    shutil.rmtree(base, ignore_errors=True)


@pytest.fixture
def partition_attempts(monkeypatch):
    """The unreadable_columns of every attempt of _convert_and_write_partition."""
    attempts = []
    convert = conversion._convert_and_write_partition

    def spy(*args, unreadable_columns=frozenset(), **kwargs):
        attempts.append(set(unreadable_columns))
        return convert(*args, unreadable_columns=unreadable_columns, **kwargs)

    monkeypatch.setattr(conversion, "_convert_and_write_partition", spy)
    return attempts


@pytest.mark.parametrize("variant", STREAM_EDGE_VARIANTS + ("reftable_wsp_partial",))
def test_stream_write_column_decisions_bit_identical(
    ms_stream_edges, variant, tmp_path, monkeypatch, stream_stats, partition_attempts
):
    """
    The columns that the non-streamed path skips (undefined cells, other cell
    shapes) or replaces (WEIGHT_SPECTRUM -> WEIGHT -> WEIGHT=1) are skipped or
    replaced the same way: decided before writing where the storage manager
    tells (TiledShapeStMan of a plain table), otherwise by the read, after
    which the partition is converted again without the column. Reversed
    frequencies are reversed batch by batch.
    """
    from xradio.measurement_set._utils._msv2.partition_queries import (
        create_partitions_with_main_rows,
    )

    msname = ms_stream_edges[variant]
    partitions, runs = create_partitions_with_main_rows(msname, [])
    idx = 0 if variant == "reftable_wsp_partial" else 1  # (DDI 1 only)
    kw = {"main_chunksize": {"time": 4}, "main_row_runs": runs[idx]}
    old_xdt, old = _convert_streamed(
        monkeypatch, msname, str(tmp_path / "old"), partitions[idx], "0", **kw
    )
    partition_attempts.clear()
    new_xdt, new = _convert_streamed(
        monkeypatch, msname, str(tmp_path / "new"), partitions[idx], "1", 1e-9, **kw
    )
    assert_stores_identical(old, new)
    assert_msv4_bit_identical(old_xdt, new_xdt)
    xds, stats = new_xdt.ds, stream_stats[-1]["variables"]
    assert stats["VISIBILITY"]["batches"] > 1
    # FLAG (StandardStMan indirect arrays in the generated MSs): left to the read
    assert not stats["FLAG"]["verified"] and "read decides" in stats["FLAG"]["readable"]
    if variant == "reversed_freq":
        assert np.all(np.diff(xds.frequency.values) > 0)
        assert stats["WEIGHT"]["col"] == "WEIGHT_SPECTRUM"
        assert "TiledShapeStMan" in stats["WEIGHT"]["readable"]
        assert "read decides" in stats["VISIBILITY_MODEL"]["readable"]
        assert partition_attempts == [set()]
    elif variant == "wsp_partial":  # decided from the TiledShapeStMan index
        assert np.all(np.diff(xds.frequency.values) > 0)
        assert stats["WEIGHT"]["col"] == "WEIGHT"
        assert xds.WEIGHT.dtype == np.float32
        assert partition_attempts == [set()]
    elif variant == "reftable_wsp_partial":  # decided by the read
        assert stats["WEIGHT"]["col"] == "WEIGHT"
        assert "reference" in stats["VISIBILITY"]["readable"]
        assert partition_attempts == [set(), {"WEIGHT_SPECTRUM"}]
    elif variant == "no_weight":
        assert stats["WEIGHT"]["col"] is None and xds.WEIGHT.dtype == np.float64
        assert np.all(xds.WEIGHT.values == 1)
        # every WEIGHT cell undefined: the first-cell probe decides, before writing
        assert partition_attempts == [set()]
    elif variant == "varying_shape":
        assert "VISIBILITY_MODEL" not in xds and "VISIBILITY_MODEL" not in stats
        assert sorted(xds.attrs["data_groups"]) == ["base", "corrected"]
        assert partition_attempts == [set(), {"MODEL_DATA"}]
    else:
        assert variant == "interleaved_fields"
        assert stats["WEIGHT"]["col"] == "WEIGHT_SPECTRUM"


def test_stream_write_guard_interleaved_partitions(
    ms_stream_edges, tmp_path, monkeypatch, stream_stats
):
    """
    FIELD_ID partitions interleaved row by row inside every baseline: time
    batches have the row runs of one pass (one per row), but every batch
    would load every 7-row WEIGHT_SPECTRUM tile again. The guard reads WEIGHT
    in one pass; VISIBILITY (one 1024-row tile for the whole DDI) keeps its
    time batches. Byte-identical either way.
    """
    from xradio.measurement_set._utils._msv2.partition_queries import (
        create_partitions_with_main_rows,
    )

    msname = ms_stream_edges["interleaved_fields"]
    partitions, runs = create_partitions_with_main_rows(msname, ["FIELD_ID"])
    idx = next(
        i
        for i, part in enumerate(partitions)
        if list(part["FIELD_ID"]) == [0] and list(part["DATA_DESC_ID"]) == [1]
    )
    kw = {"main_chunksize": {"time": 2}, "main_row_runs": runs[idx]}
    _, old = _convert_streamed(
        monkeypatch, msname, str(tmp_path / "old"), partitions[idx], "0", **kw
    )
    # WEIGHT chunk (2 times x 10 baselines x 16 x 2 float32) 2560 bytes: one
    # chunk per batch, the variable (15 times) fits 8 batches
    _, new = _convert_streamed(
        monkeypatch, msname, str(tmp_path / "new"), partitions[idx], "1", 0.003, **kw
    )
    assert_stores_identical(old, new)
    stats = stream_stats[-1]["variables"]
    weight, vis = stats["WEIGHT"], stats["VISIBILITY"]
    assert weight["window"] == "tile" and weight["window_rows"] == 7
    assert weight["guard"] == "fragmented: one pass" and weight["batches"] == 1
    assert weight["runs_whole"] == 150  # every row its own run
    assert "tiles" in weight["guard_reason"] and "runs" not in weight["guard_reason"]
    assert vis["window_rows"] == 1024 and vis["guard"] == "time"
    assert vis["batches"] == 8 and vis["runs_batched"] == vis["runs_whole"] == 150


def test_stream_write_read_failure_converts_the_partition_again(
    ms_main_layouts, tmp_path, monkeypatch, stream_stats, partition_attempts
):
    """
    A read failing after part of a variable was written (I/O error, ...): the
    MSv4 is removed and the partition converted again without the column,
    which is what the non-streamed path gives when that read fails (it skips
    the column).
    """
    from xradio.measurement_set._utils._msv2._tables import read_rows
    from xradio.measurement_set._utils._msv2.partition_queries import (
        create_partitions_with_main_rows,
    )

    msname = ms_main_layouts["dense"]
    partitions, runs = create_partitions_with_main_rows(msname, [])
    kw = {"main_chunksize": {"time": 4}, "main_row_runs": runs[0]}
    read = read_rows.read_rows_to_grid
    reads = []

    def failing_read(table, col, *args, **kwargs):
        if col == "CORRECTED_DATA":
            reads.append(col)
            if len(reads) == fail_at:
                raise OSError("simulated read failure")
        return read(table, col, *args, **kwargs)

    monkeypatch.setattr(read_rows, "read_rows_to_grid", failing_read)
    fail_at = 1  # the non-streamed path: its one read of the column fails
    old_xdt, old = _convert_streamed(
        monkeypatch, msname, str(tmp_path / "old"), partitions[0], "0", **kw
    )
    assert "VISIBILITY_CORRECTED" not in old_xdt.ds
    reads.clear()
    partition_attempts.clear()
    fail_at = 2  # the streamed write: its 2nd batch fails
    new_xdt, new = _convert_streamed(
        monkeypatch, msname, str(tmp_path / "new"), partitions[0], "1", 1e-9, **kw
    )
    assert partition_attempts == [set(), {"CORRECTED_DATA"}]
    assert len(reads) == 2  # not read again
    assert_stores_identical(old, new)
    assert_msv4_bit_identical(old_xdt, new_xdt)
    assert "VISIBILITY_CORRECTED" not in stream_stats[-1]["variables"]


def test_stream_write_write_failure_removes_the_msv4(
    ms_main_layouts, tmp_path, monkeypatch
):
    """A failure of the zarr writes after some chunks were written: the error
    is raised and the MSv4 removed (no MSv4 whose unwritten data variables read
    as fill values); the processing set directory is kept."""
    import zarr

    from xradio.measurement_set._utils._msv2.partition_queries import (
        create_partitions_with_main_rows,
    )

    msname = ms_main_layouts["dense"]
    partitions, runs = create_partitions_with_main_rows(msname, [])
    kw = {"main_chunksize": {"time": 4}, "main_row_runs": runs[0]}
    setitem = zarr.Array.__setitem__
    writes = []

    def failing_setitem(self, key, value):
        writes.append(key)
        if len(writes) == 3:
            raise OSError("simulated write failure")
        return setitem(self, key, value)

    write = conversion.write_deferred_variables

    def spy(*args, **kwargs):
        with monkeypatch.context() as m:
            m.setattr(zarr.Array, "__setitem__", failing_setitem)
            return write(*args, **kwargs)

    monkeypatch.setattr(conversion, "write_deferred_variables", spy)
    out = str(tmp_path / "new")
    with pytest.raises(OSError, match="simulated write failure"):
        _convert_streamed(monkeypatch, msname, out, partitions[0], "1", 1e-9, **kw)
    assert len(writes) == 3
    assert os.path.isdir(out) and os.listdir(out) == []


def assert_msv4_opens_as_incomplete(store_path: str) -> None:
    """
    An MSv4 whose streamed write was interrupted has no consolidated metadata,
    as one whose to_zarr was interrupted: opening it warns (xarray reads the
    non-consolidated metadata) and fails with consolidated=True, instead of
    silently reading its unwritten chunks as fill values.
    """
    import json

    with open(os.path.join(store_path, "zarr.json")) as f:
        assert "consolidated_metadata" not in json.load(f)
    with pytest.warns(RuntimeWarning, match="consolidated metadata"):
        xr.open_datatree(store_path, engine="zarr")
    with pytest.raises(ValueError, match="onsolidated"):
        xr.open_datatree(store_path, engine="zarr", consolidated=True)


def _msv4_stores(ps_path: str) -> list[str]:
    return sorted(
        os.path.join(ps_path, name)
        for name in os.listdir(ps_path)
        if os.path.isdir(os.path.join(ps_path, name))
    )


def test_stream_write_interrupted_msv4_opens_as_incomplete(
    ms_main_layouts, tmp_path, monkeypatch
):
    """
    A streamed write interrupted after some chunks were written, with no
    clean up (as by a hard kill: here a zarr write raises and discard_msv4
    does nothing): the MSv4 opens as incomplete (no consolidated metadata,
    see assert_msv4_opens_as_incomplete) and the processing set, whose root
    is consolidated after the last partition, does not list it. A completed
    streamed write has the consolidated metadata of to_zarr (the store
    comparisons of test_stream_write_bit_identical).
    """
    import zarr

    from xradio.measurement_set import (
        convert_msv2_to_processing_set,
        open_processing_set,
    )

    _set_stream_batch_mb(monkeypatch, 1e-9)
    setitem = zarr.Array.__setitem__
    writes = []

    def failing_setitem(self, key, value):
        writes.append(key)
        if len(writes) == 3:
            raise OSError("simulated interruption")
        return setitem(self, key, value)

    write = conversion.write_deferred_variables

    def spy(*args, **kwargs):
        with monkeypatch.context() as m:
            m.setattr(zarr.Array, "__setitem__", failing_setitem)
            return write(*args, **kwargs)

    monkeypatch.setattr(conversion, "write_deferred_variables", spy)
    monkeypatch.setattr(conversion, "discard_msv4", lambda *args, **kwargs: "")
    out = str(tmp_path / "interrupted.ps.zarr")
    with pytest.raises(OSError, match="simulated interruption"):
        convert_msv2_to_processing_set(
            ms_main_layouts["dense"],
            out,
            main_chunksize={"time": 4},
            persistence_mode="w",
        )
    assert len(writes) == 3
    (msv4,) = _msv4_stores(out)
    assert_msv4_opens_as_incomplete(msv4)
    assert list(open_processing_set(out).children) == []


def test_stream_write_killed_msv4_opens_as_incomplete(ms_main_layouts, tmp_path):
    """
    The same after a real hard kill: the converting process is killed
    (SIGKILL) during the streamed write of its first partition.
    """
    import signal
    import subprocess
    import sys
    import textwrap

    from xradio.measurement_set import open_processing_set

    out = str(tmp_path / "killed.ps.zarr")
    code = textwrap.dedent(
        f"""
        import os
        import signal

        import zarr

        from xradio.measurement_set import convert_msv2_to_processing_set
        from xradio.measurement_set._utils._msv2 import conversion, stream_write

        stream_write.STREAM_BATCH_BYTES = 1  # one chunk per batch
        setitem = zarr.Array.__setitem__
        writes = []

        def killing_setitem(self, key, value):
            writes.append(key)
            if len(writes) == 3:
                os.kill(os.getpid(), signal.SIGKILL)
            return setitem(self, key, value)

        write = conversion.write_deferred_variables

        def spy(*args, **kwargs):
            zarr.Array.__setitem__ = killing_setitem
            return write(*args, **kwargs)

        conversion.write_deferred_variables = spy
        convert_msv2_to_processing_set(
            {ms_main_layouts["dense"]!r},
            {out!r},
            main_chunksize={{"time": 4}},
            persistence_mode="w",
        )
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=300
    )
    assert proc.returncode == -signal.SIGKILL, proc.stderr[-2000:]
    (msv4,) = _msv4_stores(out)
    assert_msv4_opens_as_incomplete(msv4)
    assert list(open_processing_set(out).children) == []


@pytest.mark.parametrize("discard", ["removes", "does_nothing"])
def test_stream_write_retry_on_a_url_store(
    ms_main_layouts, discard, tmp_path, monkeypatch, partition_attempts
):
    """
    A read failure of the streamed write into a store given by URL (file://,
    written through fsspec) with persistence mode "w-": the partition is
    converted again without the column (as the non-streamed path, which skips
    it), not failing because the MSv4 of the failed attempt exists:
    discard_msv4 removes it through fsspec and, if that did nothing, the
    retry overwrites it (mode "w").
    """
    from xradio.measurement_set._utils._msv2._tables import read_rows
    from xradio.measurement_set._utils._msv2.partition_queries import (
        create_partitions_with_main_rows,
    )

    msname = ms_main_layouts["dense"]
    partitions, runs = create_partitions_with_main_rows(msname, [])
    kw = {"main_chunksize": {"time": 4}, "main_row_runs": runs[0]}
    read = read_rows.read_rows_to_grid
    reads = []

    def failing_read(table, col, *args, **kwargs):
        if col == "CORRECTED_DATA":
            reads.append(col)
            if len(reads) == fail_at:
                raise OSError("simulated read failure")
        return read(table, col, *args, **kwargs)

    monkeypatch.setattr(read_rows, "read_rows_to_grid", failing_read)
    fail_at = 1  # the non-streamed path: its one read of the column fails
    _, old = _convert_streamed(
        monkeypatch, msname, str(tmp_path / "old"), partitions[0], "0", **kw
    )
    discarded = []
    discard_msv4 = conversion.discard_msv4

    def discard_spy(*args, **kwargs):
        discarded.append(
            "" if discard == "does_nothing" else discard_msv4(*args, **kwargs)
        )
        return discarded[-1]

    monkeypatch.setattr(conversion, "discard_msv4", discard_spy)
    modes = []
    convert = conversion._convert_and_write_partition

    def spy(*args, persistence_mode="w-", **kwargs):
        modes.append(persistence_mode)
        return convert(*args, persistence_mode=persistence_mode, **kwargs)

    monkeypatch.setattr(conversion, "_convert_and_write_partition", spy)
    reads.clear()
    partition_attempts.clear()
    fail_at = 2  # the streamed write: its 2nd batch fails
    _set_stream_batch_mb(monkeypatch, 1e-9)
    new = tmp_path / "new"
    conversion.convert_and_write_partition(
        in_file=msname,
        out_file="file://" + str(new),
        ms_v4_id="0",
        partition_info=partitions[0],
        use_table_iter=False,
        persistence_mode="w-",
        **kw,
    )
    assert modes == ["w-", "w"]
    assert discarded == ["" if discard == "does_nothing" else "MSv4 removed"]
    assert partition_attempts == [set(), {"CORRECTED_DATA"}]
    assert_stores_identical(old, os.path.join(new, os.path.basename(old)))


def test_convert_and_write_partition_signature():
    """convert_and_write_partition shows (and binds its arguments with) the
    parameters of _convert_and_write_partition, without the internal ones it
    sets on every attempt."""
    import inspect

    public = inspect.signature(conversion.convert_and_write_partition)
    inner = inspect.signature(conversion._convert_and_write_partition)
    internal = ["unreadable_columns", "allow_stream_write"]
    assert list(public.parameters) == [
        name for name in inner.parameters if name not in internal
    ]
    for name, param in public.parameters.items():
        assert param == inner.parameters[name], name
    for name in internal:
        with pytest.raises(TypeError, match=name):
            conversion.convert_and_write_partition(
                "in.ms", "out", "0", {}, False, **{name: None}
            )
    with pytest.raises(TypeError, match="use_table_iter"):
        conversion.convert_and_write_partition("in.ms", "out", "0", {})


@pytest.mark.parametrize("declared", [False, True])
def test_stream_write_encoding_that_changes_values_is_not_streamed(
    ms_main_layouts, declared, tmp_path, monkeypatch, stream_stats
):
    """
    A deferred data variable whose encoding makes to_zarr write other values
    (here WEIGHT stored as float64) is not streamed: told by the encoding
    before writing (the partition is read whole and written by to_zarr) or,
    if only the zarr metadata declares it, before any value is written (the
    partition is converted again without the streamed write). The MSv4 is
    the non-streamed one.
    """
    from xradio.measurement_set._utils._msv2.partition_queries import (
        create_partitions_with_main_rows,
    )

    add_encoding = conversion.add_encoding

    def float64_weight(xds, *args, **kwargs):
        add_encoding(xds, *args, **kwargs)
        if "WEIGHT" in xds:
            xds["WEIGHT"].encoding["dtype"] = "float64"

    monkeypatch.setattr(conversion, "add_encoding", float64_weight)
    if declared:  # only the zarr metadata tells
        monkeypatch.setattr(conversion, "deferred_encoding_problems", lambda *a: [])
    attempts = []
    convert = conversion._convert_and_write_partition

    def spy(*args, allow_stream_write=True, **kwargs):
        attempts.append(allow_stream_write)
        return convert(*args, allow_stream_write=allow_stream_write, **kwargs)

    monkeypatch.setattr(conversion, "_convert_and_write_partition", spy)
    msname = ms_main_layouts["dense"]
    partitions, runs = create_partitions_with_main_rows(msname, [])
    kw = {"main_chunksize": {"time": 4}, "main_row_runs": runs[0]}
    old_xdt, old = _convert_streamed(
        monkeypatch, msname, str(tmp_path / "old"), partitions[0], "0", **kw
    )
    assert old_xdt.ds["WEIGHT"].encoding["dtype"] == np.float64
    attempts.clear()
    stream_stats.clear()
    new_xdt, new = _convert_streamed(
        monkeypatch, msname, str(tmp_path / "new"), partitions[0], "1", 1e-9, **kw
    )
    assert_stores_identical(old, new)
    assert_msv4_bit_identical(old_xdt, new_xdt)
    if declared:
        assert attempts == [True, False]
        assert stream_stats == []  # the streamed write raised before writing
    else:
        assert attempts == [True]
        assert [stats["summary"]["in_memory"] for stats in stream_stats] == [True]


@pytest.mark.parametrize(
    "parallel_mode, stream, streamed",
    [
        ("none", "1", True),
        ("partition", "1", True),
        ("none", "0", False),
        ("time", "1", False),  # already lazy (dask)
        ("time_without_chunk", "1", True),  # read like "none"
    ],
)
@pytest.mark.parametrize("batch_mb", [1e-9, None])
def test_stream_write_selection(
    ms_main_layouts,
    parallel_mode,
    stream,
    streamed,
    batch_mb,
    tmp_path,
    monkeypatch,
):
    """Which configurations stream; a partition whose data variables are a
    small fraction of a batch (here with the default batch size) is read whole
    instead. parallel_mode "time" without a time chunk size says how to get
    dask parallelism along time."""
    from xradio.measurement_set._utils._msv2.partition_queries import (
        create_partitions,
    )

    logged = []
    logger = types.SimpleNamespace(
        **{
            level: lambda msg, level=level: logged.append((level, str(msg)))
            for level in ("debug", "info", "warning", "error")
        }
    )
    monkeypatch.setattr(conversion, "xradio_logger", lambda: logger)
    calls = []
    monkeypatch.setattr(
        conversion,
        "write_deferred_variables",
        lambda *args, **kwargs: calls.append(args) or {},
    )
    read_whole = []
    read_deferred = conversion.read_deferred_variables

    def spy(*args, **kwargs):
        read_whole.append(args)
        return read_deferred(*args, **kwargs)

    monkeypatch.setattr(conversion, "read_deferred_variables", spy)
    _set_stream_batch_mb(monkeypatch, batch_mb)
    msname = ms_main_layouts["dense"]
    partition = create_partitions(msname, [])[0]
    if streamed:
        # placeholders only: the spy writes nothing, so do not read the result
        conversion.convert_and_write_partition(
            in_file=msname,
            out_file=str(tmp_path / "r"),
            ms_v4_id="0",
            partition_info=partition,
            use_table_iter=False,
            # without a time chunk size (main_chunksize None) "time" reads as "none"
            parallel_mode=parallel_mode.removesuffix("_without_chunk"),
            persistence_mode="w",
        )
    else:
        _convert_partition(
            monkeypatch,
            msname,
            str(tmp_path / "r"),
            partition,
            stream=stream == "1",
            main_chunksize={"time": 4},
            parallel_mode=parallel_mode,
        )
    in_memory = streamed and batch_mb is None
    assert len(calls) == (1 if streamed and not in_memory else 0)
    assert len(read_whole) == (1 if in_memory else 0)
    time_warnings = [
        msg for level, msg in logged if level == "warning" and "parallel_mode" in msg
    ]
    if parallel_mode == "time_without_chunk":
        (msg,) = time_warnings
        assert "default main_chunksize=None" in msg and "(streamed write)" in msg
        assert "main_chunksize={'time': n}" in msg
    else:
        assert time_warnings == []


# --- the casatools shim: reads without getcolnp, getcolslicenp, selectrows -------


def _assert_casatools_like_backend():
    from xradio.measurement_set._utils._msv2._tables import read_rows

    assert not read_rows.backend_has_in_place_reads()
    assert conversion.resolve_subtable_cache(None) is None  # no sub-table cache


@pytest.mark.parametrize("layout", MAIN_LAYOUTS)
@pytest.mark.parametrize(
    "parallel_mode, chunks, batch_mb",
    [
        ("none", None, None),  # read whole, one to_zarr
        ("none", {"time": 4}, 1e-9),  # streamed, one chunk per batch
        ("time", {"time": 4}, None),  # lazy (dask) reads
    ],
)
def test_convert_and_write_partition_casatools_reads_identical(
    ms_main_layouts,
    layout,
    parallel_mode,
    chunks,
    batch_mb,
    tmp_path,
    monkeypatch,
    request,
):
    """
    With the table API of the casatools shim (getcol only, no sub-table
    cache) a partition converts to a byte-identical MSv4: dense, sparse with
    duplicated rows and baseline-major rows, all MAIN read modes.
    """
    from xradio.measurement_set._utils._msv2.partition_queries import (
        create_partitions_with_main_rows,
    )

    msname = ms_main_layouts[layout]
    partitions, runs = create_partitions_with_main_rows(msname, [])
    kw = {
        "main_chunksize": chunks,
        "main_row_runs": runs[1],
        "parallel_mode": parallel_mode,
    }
    _, old = _convert_streamed(
        monkeypatch, msname, str(tmp_path / "old"), partitions[1], "1", batch_mb, **kw
    )
    request.getfixturevalue("casatools_like_tables")
    _assert_casatools_like_backend()
    for name, extra in (("runs", {}), ("no_runs", {"main_row_runs": None})):
        _, new = _convert_streamed(
            monkeypatch,
            msname,
            str(tmp_path / name),
            partitions[1],
            "1",
            batch_mb,
            **(kw | extra),
        )
        assert_stores_identical(old, new)


@pytest.mark.parametrize(
    "variant", ["reversed_freq", "wsp_partial", "varying_shape", "reftable_wsp_partial"]
)
def test_convert_and_write_partition_casatools_column_decisions(
    ms_stream_edges, variant, tmp_path, monkeypatch, request, partition_attempts
):
    """Undefined cells, cells of another shape, a reference table and
    reversed frequencies with the casatools table API: the same MSv4."""
    from xradio.measurement_set._utils._msv2.partition_queries import (
        create_partitions_with_main_rows,
    )

    msname = ms_stream_edges[variant]
    partitions, runs = create_partitions_with_main_rows(msname, [])
    idx = 0 if variant == "reftable_wsp_partial" else 1
    kw = {"main_chunksize": {"time": 4}, "main_row_runs": runs[idx]}
    _, old = _convert_streamed(
        monkeypatch, msname, str(tmp_path / "old"), partitions[idx], "1", 1e-9, **kw
    )
    attempts_old = list(partition_attempts)
    partition_attempts.clear()
    request.getfixturevalue("casatools_like_tables")
    _assert_casatools_like_backend()
    _, new = _convert_streamed(
        monkeypatch, msname, str(tmp_path / "new"), partitions[idx], "1", 1e-9, **kw
    )
    assert_stores_identical(old, new)
    assert partition_attempts == attempts_old


@pytest.mark.parametrize(
    "variant, scheme",
    [("interferometer", ["FIELD_ID"]), ("single_dish", ["ANTENNA1"])],
)
def test_convert_and_write_partition_casatools_tiled_shape_main(
    ms_tiled_shape_main, variant, scheme, tmp_path, monkeypatch, request
):
    """TiledShapeStMan MAIN columns whose tiles hold rows of several
    partitions, with the casatools table API."""
    from xradio.measurement_set._utils._msv2.partition_queries import (
        create_partitions_with_main_rows,
    )

    msname = ms_tiled_shape_main[variant]
    partitions, runs = create_partitions_with_main_rows(msname, scheme)
    kw = {"with_pointing": False, "main_chunksize": {"time": 3}}
    old = []
    for idx in range(2):
        old.append(
            _convert_streamed(
                monkeypatch,
                msname,
                str(tmp_path / f"old{idx}"),
                partitions[idx],
                "1",
                1e-9,
                main_row_runs=runs[idx],
                **kw,
            )[1]
        )
    request.getfixturevalue("casatools_like_tables")
    _assert_casatools_like_backend()
    for idx in range(2):
        _, new = _convert_streamed(
            monkeypatch,
            msname,
            str(tmp_path / f"new{idx}"),
            partitions[idx],
            "1",
            1e-9,
            main_row_runs=runs[idx],
            **kw,
        )
        assert_stores_identical(old[idx], new)


# --- default chunks of the main xds (main_chunksize=None) ---------------------


def assert_msv4_values_identical(xdt_a: xr.DataTree, xdt_b: xr.DataTree) -> None:
    """As assert_msv4_bit_identical, but the chunks may differ."""
    assert {node.path for node in xdt_a.subtree} == {
        node.path for node in xdt_b.subtree
    }
    for node in xdt_a.subtree:
        ds_a = node.to_dataset(inherit=False)
        ds_b = xdt_b[node.path].to_dataset(inherit=False)
        assert list(ds_a.variables) == list(ds_b.variables), node.path
        assert _without_dates(ds_a.attrs) == _without_dates(ds_b.attrs), node.path
        for name, var_a in ds_a.variables.items():
            var_b = ds_b.variables[name]
            where = f"{node.path}/{name}"
            assert var_a.dims == var_b.dims and var_a.dtype == var_b.dtype, where
            values_a, values_b = var_a.values, var_b.values
            if values_a.dtype == object:
                np.testing.assert_array_equal(values_a, values_b, err_msg=where)
            else:
                assert values_a.tobytes() == values_b.tobytes(), where
            assert _without_dates(var_a.attrs) == _without_dates(var_b.attrs), where
            encoding = {
                key: repr(var_a.encoding.get(key))
                for key in ("dtype", "compressors", "filters", "_FillValue")
            }
            assert encoding == {
                key: repr(var_b.encoding.get(key)) for key in encoding
            }, where


@pytest.mark.parametrize("layout", MAIN_LAYOUTS)
@pytest.mark.parametrize("batch_mb", [None, 1e-9])
def test_default_main_chunksize_time_chunks(
    ms_main_layouts, layout, batch_mb, tmp_path, monkeypatch
):
    """
    main_chunksize=None chunks the data variables along time only, about
    DEFAULT_MAIN_CHUNK_BYTES of the largest one (here made small): the MSv4 is
    byte-identical to one with these time chunks given explicitly, and holds
    the values of one with other chunks. With the real default (small
    partitions) it is one chunk per variable, as before.
    """
    from xradio.measurement_set._utils._msv2.partition_queries import (
        create_partitions_with_main_rows,
    )

    msname = ms_main_layouts[layout]
    partitions, runs = create_partitions_with_main_rows(msname, [])
    kw = {"main_row_runs": runs[1]}
    # one chunk per variable (main_chunksize=None before this version)
    one_chunk_xdt, one_chunk = _convert_streamed(
        monkeypatch,
        msname,
        str(tmp_path / "one"),
        partitions[1],
        "1",
        None,
        main_chunksize={},
        **kw,
    )
    _, default = _convert_streamed(
        monkeypatch, msname, str(tmp_path / "d0"), partitions[1], "1", None, **kw
    )
    assert_stores_identical(one_chunk, default)
    n_times = one_chunk_xdt.ds.sizes["time"]
    assert one_chunk_xdt.ds.VISIBILITY.encoding["chunks"][0] == n_times

    # 3 time steps of VISIBILITY (10 baselines x 16 channels x 2 x 8 bytes)
    monkeypatch.setattr(conversion, "DEFAULT_MAIN_CHUNK_BYTES", 3 * 2560 + 100)
    default_xdt, default = _convert_streamed(
        monkeypatch, msname, str(tmp_path / "d"), partitions[1], "1", batch_mb, **kw
    )
    _, explicit = _convert_streamed(
        monkeypatch,
        msname,
        str(tmp_path / "e"),
        partitions[1],
        "1",
        batch_mb,
        main_chunksize={"time": 3},
        **kw,
    )
    assert_stores_identical(default, explicit)
    for name, var in default_xdt.ds.data_vars.items():
        assert var.encoding["chunks"] == (3,) + var.shape[1:], name
    assert_msv4_values_identical(one_chunk_xdt, default_xdt)


def test_default_main_chunksize_beyond_the_blosc_limit(
    ms_main_layouts, tmp_path, monkeypatch, stream_stats
):
    """
    A data variable whose time steps are larger than the Blosc limit (here
    made small) is also chunked along frequency, and converts (streamed in
    batches of one chunk) to the values of a one-chunk conversion.
    """
    from xradio.measurement_set._utils._msv2.partition_queries import (
        create_partitions_with_main_rows,
    )

    msname = ms_main_layouts["dense"]
    partitions, runs = create_partitions_with_main_rows(msname, [])
    kw = {"main_row_runs": runs[0]}
    one_chunk_xdt, _ = _convert_streamed(
        monkeypatch,
        msname,
        str(tmp_path / "one"),
        partitions[0],
        "0",
        main_chunksize={},
        **kw,
    )
    # a time step of VISIBILITY is 2560 bytes, of one channel 160 bytes
    monkeypatch.setattr(conversion, "BLOSC_MAX_BUFFER_BYTES", 1000)
    monkeypatch.setattr(conversion, "DEFAULT_MAIN_CHUNK_BYTES", 500)
    xdt, store = _convert_streamed(
        monkeypatch, msname, str(tmp_path / "d"), partitions[0], "1", 1e-9, **kw
    )
    assert xdt.ds.VISIBILITY.encoding["chunks"] == (1, 10, 3, 2)
    assert xdt.ds.FLAG.encoding["chunks"] == (1, 10, 3, 2)
    assert xdt.ds.UVW.encoding["chunks"] == (1, 10, 3)
    for name, var in xdt.ds.data_vars.items():
        chunk = np.prod(var.encoding["chunks"]) * var.dtype.itemsize
        assert chunk <= 1000, name
    assert not stream_stats[-1]["summary"]["in_memory"]
    assert_msv4_values_identical(one_chunk_xdt, xdt)
