import os
import pathlib
import shutil
from collections import namedtuple
from contextlib import nullcontext as no_raises

import numpy as np
import pytest
import xarray as xr

import xradio.measurement_set._utils._msv2.conversion as conversion
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
        msv4_xdt = xr.open_datatree(
            out_name + "/" + ms_custom_spec.fname.rsplit(".")[0] + "_" + msv4_id,
            engine="zarr",
        )
        check_dataset(msv4_xdt.ds, VisibilityXds)
        check_datatree(msv4_xdt).expect()
        check_msv4_matches_descr(msv4_xdt, ms_custom_spec.descr)

    finally:
        shutil.rmtree(out_name)


# --- MAIN read paths: TaQL vs rows (TEMPORARY XRADIO_MSV2_MAIN_READ switch) ------

MAIN_LAYOUTS = ("dense", "sparse_dup", "baseline_major")


def _rewrite_main_rows(msname: str, layout: str, seed: int) -> None:
    """
    Rewrite the MAIN rows of a generated MS: every DDI gets 30 times x 10
    baselines (time-major or baseline-major rows), random data, flags, weights
    and UVW. "sparse_dup" moves 10% of the rows to 3 extra, sparsely filled
    times (leaving their cells empty) and gives some rows the (time, baseline)
    of the row before (duplicated cells). The generated MS uses
    TiledColumnStMan, which cannot remove rows.
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


def _convert_partition(monkeypatch, msname, out_file, partition_info, main_read, **kw):
    monkeypatch.setenv(conversion.MAIN_READ_ENV_VAR, main_read)
    kw.setdefault("use_table_iter", False)
    conversion.convert_and_write_partition(
        in_file=msname,
        out_file=out_file,
        ms_v4_id="0",
        partition_info=partition_info,
        persistence_mode="w",
        **kw,
    )
    msv4_name = pathlib.Path(msname).name.replace(".ms", "") + "_0"
    return xr.open_datatree(os.path.join(out_file, msv4_name), engine="zarr")


@pytest.mark.parametrize("layout", MAIN_LAYOUTS)
@pytest.mark.parametrize(
    "partition_source", ["create_partitions", "hand_built", "mismatched_runs"]
)
def test_convert_and_write_partition_rows_vs_taql_bit_identical(
    ms_main_layouts, layout, partition_source, tmp_path, monkeypatch
):
    """The rows read path gives exactly the output of the TaQL path."""
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

    taql = _convert_partition(
        monkeypatch, msname, str(tmp_path / "t"), partition, "taql", **kw
    )
    taql_iter = _convert_partition(
        monkeypatch,
        msname,
        str(tmp_path / "ti"),
        partition,
        "taql",
        use_table_iter=True,
        **kw,
    )
    rows = _convert_partition(
        monkeypatch, msname, str(tmp_path / "r"), partition, "rows", **kw
    )

    assert {"VISIBILITY", "VISIBILITY_CORRECTED", "FLAG", "WEIGHT", "UVW"} <= set(
        rows.ds.data_vars
    )
    assert rows.ds.WEIGHT.dtype == np.float32  # read from WEIGHT, not the ones fallback
    if layout == "sparse_dup":
        assert np.isnan(rows.ds.VISIBILITY.values).any()  # padded missing cells
    assert_msv4_bit_identical(taql, rows)
    assert_msv4_bit_identical(taql_iter, rows)


@pytest.mark.parametrize("layout", MAIN_LAYOUTS)
def test_convert_and_write_partition_rows_time_mode(
    ms_main_layouts, layout, tmp_path, monkeypatch
):
    """parallel_mode="time" on the rows path: same output as the numpy path,
    also for sparse, duplicated and baseline-major rows."""
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
        "rows",
        main_chunksize=chunks,
        main_row_runs=runs[0],
    )
    timed = _convert_partition(
        monkeypatch,
        msname,
        str(tmp_path / "t"),
        partition,
        "rows",
        main_chunksize=chunks,
        parallel_mode="time",
        main_row_runs=runs[0],
    )
    assert_msv4_bit_identical(none, timed)
    if layout == "dense":  # the TaQL time path needs dense, time-ordered rows
        taql_timed = _convert_partition(
            monkeypatch,
            msname,
            str(tmp_path / "tt"),
            partition,
            "taql",
            main_chunksize=chunks,
            parallel_mode="time",
        )
        assert_msv4_bit_identical(taql_timed, timed)


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
def test_convert_and_write_partition_rows_vs_taql_tiled_shape_main(
    ms_tiled_shape_main, variant, scheme, tmp_path, monkeypatch
):
    """
    TiledShapeStMan MAIN columns whose tiles hold rows of several partitions,
    with the FIELD_ID / ANTENNA1 partition schemes: the rows path gives exactly
    the output of the TaQL path.
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
                out[main_read] = _convert_partition(
                    monkeypatch,
                    msname,
                    str(tmp_path / f"{main_read}{idx}"),
                    partitions[idx],
                    main_read,
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
            "rows",
            main_chunksize={"time": 4},
            parallel_mode="time",
            **kw | {"main_row_runs": runs[0]},
        )
        none = _convert_partition(
            monkeypatch,
            msname,
            str(tmp_path / "none"),
            partitions[0],
            "rows",
            main_chunksize={"time": 4},
            **kw | {"main_row_runs": runs[0]},
        )
        assert_msv4_bit_identical(none, timed)


def test_convert_and_write_partition_rows_runs_no_taql_on_main(
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
    _convert_partition(
        monkeypatch,
        msname,
        str(tmp_path / "r"),
        partition,
        "rows",
        main_row_runs=runs[0],
    )
    assert queries  # sub-tables are still read with TaQL
    assert not [q for q in queries if "$mtable" in q]
    queries.clear()
    _convert_partition(monkeypatch, msname, str(tmp_path / "t"), partition, "taql")
    assert [q for q in queries if "$mtable" in q]


def test_get_main_read_mode(monkeypatch):
    monkeypatch.delenv(conversion.MAIN_READ_ENV_VAR, raising=False)
    assert conversion.get_main_read_mode() == "rows"
    for value, expected in (("taql", "taql"), (" ROWS ", "rows"), ("", "rows")):
        monkeypatch.setenv(conversion.MAIN_READ_ENV_VAR, value)
        assert conversion.get_main_read_mode() == expected
    monkeypatch.setenv(conversion.MAIN_READ_ENV_VAR, "bogus")
    with pytest.raises(ValueError, match="XRADIO_MSV2_MAIN_READ"):
        conversion.get_main_read_mode()


def test_get_main_read_mode_without_python_casacore(monkeypatch):
    # the casatools fallback module has no getcolnp/selectrows
    monkeypatch.setattr(conversion, "ROWS_READ_SUPPORTED", False)
    monkeypatch.delenv(conversion.MAIN_READ_ENV_VAR, raising=False)
    assert conversion.get_main_read_mode() == "taql"
    monkeypatch.setenv(conversion.MAIN_READ_ENV_VAR, "rows")
    with pytest.raises(ValueError, match="python-casacore"):
        conversion.get_main_read_mode()


@pytest.mark.parametrize(
    "col_name, parallel_mode, read_rows, expected",
    [
        ("DATA", "none", False, "read_col_conversion_numpy"),
        ("DATA", "time", False, "read_col_conversion_dask"),
        ("UVW", "time", False, "read_col_conversion_numpy"),
        ("DATA", "none", True, "read_col_conversion_rows"),
        ("FLAG", "time", True, "read_col_conversion_dask_rows"),
        ("TIME_CENTROID", "time", True, "read_col_conversion_rows"),
    ],
)
def test_get_read_col_conversion_function(col_name, parallel_mode, read_rows, expected):
    func = conversion.get_read_col_conversion_function(
        col_name, parallel_mode, read_rows=read_rows
    )
    assert func.__name__ == expected


def test_create_data_variables_reads_columns_in_sorted_order(
    ms_main_layouts, tmp_path, monkeypatch
):
    """The read order (and with it the memory peak) does not depend on the hash seed."""
    from xradio.measurement_set._utils._msv2.partition_queries import (
        create_partitions,
    )

    read_cols = []
    get_function = conversion.get_read_col_conversion_function

    def spy(col_name, *args, **kwargs):
        read_cols.append(col_name)
        return get_function(col_name, *args, **kwargs)

    monkeypatch.setattr(conversion, "get_read_col_conversion_function", spy)
    msname = ms_main_layouts["dense"]
    partition = create_partitions(msname, [])[0]
    _convert_partition(monkeypatch, msname, str(tmp_path / "r"), partition, "rows")
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
        "rows",
        main_chunksize={"time": 4},
        parallel_mode=parallel_mode,
    )
    assert released == [True]


# --- sub-table cache (TEMPORARY XRADIO_MSV2_SUBTABLE_CACHE switch) ----------------


@pytest.mark.parametrize("ms_fixture", ["ms_minimal_required", "ms_minimal_misbehaved"])
@pytest.mark.parametrize("interpolate", [False, True])
def test_convert_and_write_partition_subtable_cache_bit_identical(
    ms_fixture, interpolate, tmp_path, monkeypatch, request
):
    """Every partition converts to exactly the same MSv4 (or fails the same way)
    with the sub-table cache shared by the partitions (POINTING, SYSCAL, WEATHER,
    PHASE_CAL, GAIN_CURVE, ephemerides, ...) as with per-partition reads."""
    from xradio.measurement_set._utils._msv2._tables.subtable_cache import (
        SUBTABLE_CACHE_ENV_VAR,
        SubtableCache,
    )
    from xradio.measurement_set._utils._msv2.partition_queries import (
        create_partitions,
    )

    msname = request.getfixturevalue(ms_fixture).fname
    partitions = create_partitions(msname, ["FIELD_ID"])
    cache = SubtableCache(n_partitions=len(partitions))
    msv4_name = pathlib.Path(msname).name.replace(".ms", "") + "_0"
    n_converted = 0
    for idx, partition in enumerate(partitions):
        results = {}
        for mode in ("0", "1"):
            monkeypatch.setenv(SUBTABLE_CACHE_ENV_VAR, mode)
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
                    subtable_cache=cache,  # unused with XRADIO_MSV2_SUBTABLE_CACHE=0
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
