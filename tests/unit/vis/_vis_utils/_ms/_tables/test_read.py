import pytest

import numpy as np
from pathlib import Path


@pytest.mark.parametrize(
    "tab_name, expected_result",
    [
        ("ms_minimal_required", True),
        ("ms_tab_nonexistent", False),
        ("ddi_xds_min", False),
    ],
)
def test_table_exists(tab_name, expected_result, request):
    from xradio.vis._vis_utils._ms._tables.read import table_exists

    fixture = request.getfixturevalue(tab_name)
    fname = fixture.fname if isinstance(fixture, tuple) else fixture
    assert table_exists(fname) == expected_result


@pytest.mark.parametrize(
    "times, expected_result",
    [
        (
            np.array([0, 1_900_000_000.36]),
            np.array(
                ["1858-11-17T00:00:00.0", "1919-02-01T17:46:40.359999895"],
                dtype="datetime64[ns]",
            ),
        ),
        (np.array([10]), np.array([], dtype="datetime64[ns]")),
        (
            np.array([5_000_000_000.1234]),
            np.array(["2017-04-27T08:53:20.123399734"], dtype="datetime64[ns]"),
        ),
        (
            np.array([10_000_000_000]),
            np.array(["2175-10-06T17:46:40.000000000"], dtype="datetime64[ns]"),
        ),
    ],
)
def test_convert_casacore_time(times, expected_result):
    from xradio.vis._vis_utils._ms._tables.read import convert_casacore_time

    assert all(convert_casacore_time(times) == expected_result)


@pytest.mark.parametrize(
    "times, expected_result",
    [
        (
            np.array([0, 50000]),
            np.array(
                ["1858-11-17T00:00:00.0", "1995-10-10T00:00:00.0"],
                dtype="datetime64[ns]",
            ),
        ),
        (
            np.array([58000.123]),
            np.array(["2017-09-04T02:57:07.200000048"], dtype="datetime64[ns]"),
        ),
        (
            np.array([70000.34]),
            np.array(["2050-07-13T08:09:35.999999523"], dtype="datetime64[ns]"),
        ),
    ],
)
def test_convert_mjd_time(times, expected_result, request):
    from xradio.vis._vis_utils._ms._tables.read import convert_mjd_time
    assert all(convert_mjd_time(times) == expected_result)


def test_extract_table_attributes_main(ms_minimal_required):
    from xradio.vis._vis_utils._ms._tables.read import extract_table_attributes

    res = extract_table_attributes(ms_minimal_required.fname)
    assert all(entry in res for entry in ["MS_VERSION", "column_descriptions", "info"])
    assert all(sub in res["info"] for sub in ["type", "subType", "readme"])
    assert len(res["column_descriptions"]) == 22
    columns = ["TIME", "DATA_DESC_ID", "ANTENNA1", "ANTENNA2", "WEIGHT"]
    assert all([col in res["column_descriptions"] for col in columns])


def test_extract_table_attributes_ant(ms_minimal_required):
    from xradio.vis._vis_utils._ms._tables.read import extract_table_attributes

    res = extract_table_attributes(str(Path(ms_minimal_required.fname) / "ANTENNA"))
    assert len(res) == 2
    assert "column_descriptions" in res
    assert "info" in res
    assert len(res["column_descriptions"]) == 8
    cols = ["DISH_DIAMETER", "FLAG_ROW", "MOUNT", "NAME", "OFFSET", "POSITION"]
    assert all([col in res["column_descriptions"] for col in cols])
    assert all(sub in res["info"] for sub in ["type", "subType", "readme"])


def test_add_units_measures(main_xds_min):
    from xradio.vis._vis_utils._ms._tables.read import add_units_measures

    col_descr = {"column_descriptions": {}}
    xds_vars = {"uvw": main_xds_min.uvw, "time": main_xds_min.time}
    res = add_units_measures(xds_vars, col_descr)
    assert xds_vars["uvw"].attrs
    assert xds_vars["time"].attrs


def test_make_freq_attrs_uvw(spw_xds_min):
    from xradio.vis._vis_utils._ms._tables.read import make_freq_attrs

    res = make_freq_attrs(spw_xds_min, 0)
    expected = {"measure": {"ref_frame": "REST", "type": "frequency"}, "units": "Hz"}
    assert res == expected


def test_get_pad_nan_uvw(main_xds_min):
    from xradio.vis._vis_utils._ms._tables.read import get_pad_nan

    res = get_pad_nan(main_xds_min.data_vars["uvw"])
    assert np.isnan(res)


def test_get_pad_nan_feed1(main_xds_min):
    from xradio.vis._vis_utils._ms._tables.read import get_pad_nan

    res = get_pad_nan(main_xds_min.data_vars["feed1_id"])
    # Beware, with integer types this can be different integer values
    # depending on platform (https://github.com/numpy/numpy/issues/21166)
    with np.errstate(invalid="ignore"):
        expected_nan = np.array([np.nan]).astype(np.int32)
    assert res == expected_nan


def test_redimension_ms_subtable_source(source_xds_min):
    from xradio.vis._vis_utils._ms._tables.read import redimension_ms_subtable
    import xarray as xr

    res = redimension_ms_subtable(source_xds_min, "SOURCE")
    assert isinstance(res, xr.Dataset)
    src_coords = ["source_id", "time", "spectral_window_id", "pulsar_id"]
    assert all([coord in res.coords for coord in src_coords])


def test_is_ephem_subtable_ms(ms_minimal_required):
    from xradio.vis._vis_utils._ms._tables.read import is_ephem_subtable

    res = is_ephem_subtable(ms_minimal_required.fname)
    assert res == False


def test_add_ephemeris_vars(ms_minimal_required):
    from xradio.vis._vis_utils._ms._tables.read import add_ephemeris_vars
    import xarray as xr

    # would need an ephem_xds fixture
    ephem_xds = xr.Dataset(data_vars={"mjd": ("row", np.array([]))})
    res = add_ephemeris_vars(
        Path(ms_minimal_required.fname) / "FIELD" / "EPHEM0_f0.tab", ephem_xds
    )
    assert res
    assert all(
        [xvar in res.data_vars for xvar in ["ephemeris_row_id", "ephemeris_id", "time"]]
    )


def test_is_nested_ms_empty():
    from xradio.vis._vis_utils._ms._tables.read import is_nested_ms

    with pytest.raises(KeyError, match="other"):
        res = is_nested_ms({})
        assert res == False


def test_is_nested_ms_ant(ms_minimal_required):
    from xradio.vis._vis_utils._ms._tables.read import is_nested_ms
    from xradio.vis._vis_utils._ms._tables.read import extract_table_attributes

    ant_subt = str(Path(ms_minimal_required.fname) / "ANTENNA")
    ctds_attrs = extract_table_attributes(ms_minimal_required.fname)
    attrs = {"other": {"msv2": {"ctds_attrs": ctds_attrs}}}

    res = is_nested_ms(attrs)
    assert res == True


def test_is_nested_ms_ms_min(ms_minimal_required):
    from xradio.vis._vis_utils._ms._tables.read import is_nested_ms
    from xradio.vis._vis_utils._ms._tables.read import extract_table_attributes

    ctds_attrs = extract_table_attributes(ms_minimal_required.fname)
    attrs = {"other": {"msv2": {"ctds_attrs": ctds_attrs}}}

    res = is_nested_ms(attrs)
    assert res == True


def test_read_generic_table_ant(ms_minimal_required):
    from xradio.vis._vis_utils._ms._tables.read import read_generic_table
    import xarray as xr

    res = read_generic_table(ms_minimal_required.fname, "ANTENNA")
    assert res
    assert type(res) == xr.Dataset
    assert all([dim in res.dims for dim in ["row", "dim_1"]])


def test_read_generic_table_state(ms_minimal_required):
    from xradio.vis._vis_utils._ms._tables.read import read_generic_table
    import xarray as xr

    res = read_generic_table(ms_minimal_required.fname, "STATE")
    assert res
    assert type(res) == xr.Dataset
    assert all([dim in res.dims for dim in ["row"]])
    assert all(
        [
            xvar in res.data_vars
            for xvar in ["cal", "load", "sig", "sub_scan", "obs_mode"]
        ]
    )


def test_read_generic_table_ephem(ms_minimal_required):
    from xradio.vis._vis_utils._ms._tables.read import read_generic_table
    import xarray as xr

    res = read_generic_table(ms_minimal_required.fname, "FIELD/EPHEM0_FIELDNAME.tab")
    exp_attrs = {
        "other": {
            "msv2": {
                "bad_cols": ["MJD"],
                "ctds_attrs": {
                    "column_descriptions": {
                        "MJD": {
                            "valueType": "double",
                            "dataManagerType": "StandardStMan",
                            "dataManagerGroup": "StandardStMan",
                            "option": 0,
                            "maxlen": 0,
                            "comment": "comment...",
                            "keywords": {
                                "QuantumUnits": ["s"],
                                "MEASINFO": {"type": "epoch", "Ref": "bogus MJD"},
                            },
                        }
                    },
                    "info": {"readme": "", "subType": "", "type": ""},
                },
            }
        }
    }
    assert isinstance(res, xr.Dataset)
    assert all([dim in res.dims for dim in ["ephemeris_row_id", "ephemeris_id"]])
    assert "time" in res.data_vars
    assert res.data_vars["time"].size == 1
    assert res.attrs == exp_attrs


def test_load_generic_cols_state(ms_minimal_required):
    from xradio.vis._vis_utils._ms._tables.read import load_generic_cols
    from xradio.vis._vis_utils._ms._tables.table_query import open_table_ro
    import xarray as xr

    subt_state = str(Path(ms_minimal_required.fname) / "STATE")
    with open_table_ro(subt_state) as tb_tool:
        ignore_cols = ["FLAG_ROW"]
        res = load_generic_cols(
            ms_minimal_required.fname, tb_tool, timecols=["TIME"], ignore=ignore_cols
        )
        assert res
        assert isinstance(res, tuple)
        assert res[0] == {}
        assert all([col.lower() not in res[1] for col in ignore_cols])
        assert all(
            [var in res[1] for var in ["load", "obs_mode", "ref", "sig", "sub_scan"]]
        )
        assert all([isinstance(val, xr.DataArray) for val in res[1].values()])


def test_load_generic_cols_spw(ms_minimal_required):
    from xradio.vis._vis_utils._ms._tables.read import load_generic_cols
    from xradio.vis._vis_utils._ms._tables.table_query import open_table_ro
    import xarray as xr

    subt_state = str(Path(ms_minimal_required.fname) / "SPECTRAL_WINDOW")
    with open_table_ro(subt_state) as tb_tool:
        ignore_cols = ["FLAG_ROW"]
        res = load_generic_cols(
            ms_minimal_required.fname, tb_tool, timecols=["TIME"], ignore=ignore_cols
        )
        assert res
        assert isinstance(res, tuple)
        assert res[0] == {}
        assert all([col not in res[1] for col in ignore_cols])
        expected_vars = [
            "chan_freq",
            "ref_frequency",
            "effective_bw",
            "resolution",
            "freq_group",
            "freq_group_name",
            "if_conv_chain",
            "name",
            "net_sideband",
            "num_chan",
            "total_bandwidth",
        ]
        assert all([var in res[1] for var in expected_vars])
        assert all([isinstance(val, xr.DataArray) for val in res[1].values()])


def test_read_flat_col_chunk_time(ms_minimal_required):
    from xradio.vis._vis_utils._ms._tables.read import read_flat_col_chunk

    res = read_flat_col_chunk(ms_minimal_required.fname, "TIME", (10,), [0, 1, 5], 0, 0)
    assert isinstance(res, np.ndarray)
    assert res.shape == (3,)
    assert np.all(res >= 1e9)


def test_read_flat_col_chunk_sigma(ms_minimal_required):
    from xradio.vis._vis_utils._ms._tables.read import read_flat_col_chunk

    npols = ms_minimal_required.descr["npols"]
    res = read_flat_col_chunk(
        ms_minimal_required.fname, "SIGMA", (10, npols), [4, 5, 8, 9], 0, 0
    )
    assert isinstance(res, np.ndarray)
    assert res.shape == (4, npols)
    assert np.all(res == 1)


def test_read_flat_col_chunk_flag(ms_minimal_required):
    from xradio.vis._vis_utils._ms._tables.read import read_flat_col_chunk

    npols = ms_minimal_required.descr["npols"]
    nchans = ms_minimal_required.descr["nchans"]
    res = read_flat_col_chunk(
        ms_minimal_required.fname, "FLAG", (10, 32, npols), [0, 1, 2], 0, 0
    )
    assert isinstance(res, np.ndarray)
    assert res.shape == (3, nchans, npols)
    assert np.all(res == False)
