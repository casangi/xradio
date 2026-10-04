from pathlib import Path

import numpy as np
import pytest


@pytest.mark.parametrize(
    "tab_name, expected_result",
    [
        ("ms_minimal_required", True),
        ("ms_tab_nonexistent", False),
        ("generic_antenna_xds_min", False),
    ],
)
def test_table_exists(tab_name, expected_result, request):
    from xradio.measurement_set._utils._msv2._tables.read import table_exists

    fixture = request.getfixturevalue(tab_name)
    fname = fixture.fname if isinstance(fixture, tuple) else fixture
    assert table_exists(fname) == expected_result


@pytest.mark.parametrize(
    "tab_name, col_name, expected_result",
    [
        ("ms_minimal_required", "TIME", True),
        ("ms_minimal_required", "WEIGHT_SPECTRUM", False),
        ("ms_minimal_required", "WEIGHT", True),
        ("ms_minimal_required", "DATA", True),
        ("ms_minimal_required", "TIME_CENTROID", True),
        ("ms_minimal_required", "FOO_INEXISTENT", False),
    ],
)
def test_table_has_column(tab_name, col_name, expected_result, request):
    from xradio.measurement_set._utils._msv2._tables.read import table_has_column

    fixture = request.getfixturevalue(tab_name)
    fname = fixture.fname if isinstance(fixture, tuple) else fixture
    assert table_has_column(fname, col_name) == expected_result


@pytest.mark.parametrize(
    "tab_name, col_name, expected_raises",
    [
        ("ms_tab_nonexistent", "DATA", pytest.raises(RuntimeError)),
        ("ms_tab_nonexistent", "ANY", pytest.raises(RuntimeError)),
        ("generic_antenna_xds_min", "TIME", pytest.raises(RuntimeError)),
        ("generic_source_xds_min", "ANTENNA1", pytest.raises(RuntimeError)),
    ],
)
def test_table_has_column_raises(tab_name, col_name, expected_raises, request):
    from xradio.measurement_set._utils._msv2._tables.read import table_has_column

    fixture = request.getfixturevalue(tab_name)
    fname = fixture.fname if isinstance(fixture, tuple) else fixture

    with expected_raises:
        _res = table_has_column(fname, col_name)


@pytest.mark.parametrize(
    "times, to_datetime, expected_result",
    [
        (
            np.array([0, 1_900_000_000.36]),
            True,
            np.array(
                ["1858-11-17T00:00:00.0", "1919-02-01T17:46:40.359999895"],
                dtype="datetime64[ns]",
            ),
        ),
        (np.array([10]), True, np.array([], dtype="datetime64[ns]")),
        (
            np.array([5_000_000_000.1234]),
            True,
            np.array(["2017-04-27T08:53:20.123399734"], dtype="datetime64[ns]"),
        ),
        (
            np.array([10_000_000_000]),
            True,
            np.array(["2175-10-06T17:46:40.000000000"], dtype="datetime64[ns]"),
        ),
        (
            np.array([0, 1_900_000_000.36]),
            False,
            np.array(
                [-3.5067168e09, -1.60671679964e09],
                dtype="float64",
            ),
        ),
        (np.array([10]), False, np.array([], dtype="float64")),
        (
            np.array([5_000_000_000.12345]),
            False,
            np.array([1.4932832001234502e09], dtype="float64"),
        ),
        (
            np.array([10_000_000_000]),
            False,
            np.array([6.4932832e09], dtype="float64"),
        ),
    ],
)
def test_convert_casacore_time(times, to_datetime, expected_result):
    from xradio.measurement_set._utils._msv2._tables.read import convert_casacore_time

    assert all(convert_casacore_time(times, to_datetime) == expected_result)


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
    from xradio.measurement_set._utils._msv2._tables.read import convert_mjd_time

    assert all(convert_mjd_time(times) == expected_result)


@pytest.mark.parametrize(
    "times, expected_result",
    [
        (np.array([5.123456789e09]), np.array([59299.268391])),
        (
            np.array([5.05366344e09, 5.05366345e09, 5.05366369e09, 5.05366426e09]),
            np.array([58491.475, 58491.47511574, 58491.47789352, 58491.48449074]),
        ),
    ],
)
def test_convert_casacore_time_to_mjd(times, expected_result):
    from xradio.measurement_set._utils._msv2._tables.read import (
        convert_casacore_time_to_mjd,
    )

    np.testing.assert_array_almost_equal(
        convert_casacore_time_to_mjd(times), expected_result
    )


def test_extract_table_attributes_main(ms_minimal_required):
    from xradio.measurement_set._utils._msv2._tables.read import (
        extract_table_attributes,
    )

    res = extract_table_attributes(ms_minimal_required.fname)
    assert all(entry in res for entry in ["MS_VERSION", "column_descriptions", "info"])
    assert all(sub in res["info"] for sub in ["type", "subType", "readme"])
    assert len(res["column_descriptions"]) == 22
    columns = ["TIME", "DATA_DESC_ID", "ANTENNA1", "ANTENNA2", "WEIGHT"]
    assert all([col in res["column_descriptions"] for col in columns])


# Tolerances used in min/max TaQL queries
tol_eps = np.finfo(np.float64).eps * 4
tol_interval = 0.1e9 / 4


@pytest.mark.parametrize(
    "in_min_max, expected_result",
    [
        ((0, 1.6e9), (0, 2.0e9)),
        ((0, 2.3e9), (0, 2.3e9)),
    ],
)
def test_find_projected_min_max_table(ms_minimal_required, in_min_max, expected_result):
    from xradio.measurement_set._utils._msv2._tables.read import (
        find_projected_min_max_table,
    )

    res = find_projected_min_max_table(
        in_min_max, ms_minimal_required.fname, "POINTING", "TIME"
    )
    np.testing.assert_array_almost_equal(res, expected_result)


# example from casatestdata/.../alma_ephemobj_icrs.ms, largely truncated
mjd_ephemeris_uranus = np.array(
    [57362.97916667, 57362.99305556, 57363.00694444, 57363.02083333, 57363.03472222]
)
# example from casatestdata/.../titan-one-baseline-one-timestamp.ms, largely truncated
mjd_ephemeris_titan = np.array(
    [56090.0, 56091.0, 56092.0, 56093.0, 56094.0, 56095.0, 56096.0]
)
# example from casatestdata/.../uid___A002_X1c6e54_X223-thinned.ms
time_pointing_ngc3256 = np.array(
    [
        4808345570.024,
        4808345606.024,
        4808345606.024,
        4808345642.024,
        4808345675.024,
        4808345702.024,
        4808345705.024,
        4808345720.024,
        4808345735.024,
        4808345735.024,
        4808345765.024,
        4808345765.024,
        4808345777.024,
        4808345777.024,
    ]
)


@pytest.mark.parametrize(
    "in_min_max, in_array, expected_result",
    [
        ((0, 4.6e9), (np.array([5e9])), (0, 5.0e9)),
        ((0, 4.3e9), (np.array([4e9])), (0, 4.3e9)),
        (
            (4.95e9, 5.3e9),
            np.array([4.9e9, 5.0e9]),
            (4.9e9 - tol_interval, 5.3e9 + tol_interval),
        ),
        (
            (3.88e9, 4.1e9),
            np.array([3.8e9, 3.9e9, 4.0e9]),
            (3.8e9 - tol_interval, 4.1e9 + tol_interval),
        ),
        (
            (3.88e9, 4.0e9),
            np.array([3.8e9, 3.9e9, 4.0e9]),
            (3.8e9 - tol_interval, 4.0e9 + tol_interval),
        ),
        (
            (58491.475746944445, 58491.4852313889),
            np.array([58491.47222222, 58491.48611111]),
            (58491.46875, 58491.489583),
        ),
        (
            (57363.0023961111, 57363.00245944445),
            mjd_ephemeris_uranus,
            (57362.989583, 57363.010417),
        ),
        (
            (57363.0023961111, 57363.00811111),
            mjd_ephemeris_uranus,
            (57362.989583, 57363.024306),
        ),
        (
            (56093.97593833323, 56093.97593833345),
            mjd_ephemeris_titan,
            (56092.75, 56094.25),
        ),
        (
            (56092.97, 56095.99),
            mjd_ephemeris_titan,
            (56091.75, 56096.25),
        ),
        (
            (4808345861.784, 4808346243.816),
            time_pointing_ngc3256,
            (4808345777.024, 4808346243.816),
        ),
        (
            (4808345661.784, 4808345843.816),
            time_pointing_ngc3256,
            (4808345642.024, 4808345843.816),
        ),
        (
            (4808345699.064, 4808345744.424),
            time_pointing_ngc3256,
            (4808345675.024, 4808345765.024),
        ),
        (
            (4808345000.064, 4808345744.424),
            time_pointing_ngc3256,
            (4808345000.064, 4808345765.024),
        ),
        (
            (4808345699.064, 4808345744.424),
            time_pointing_ngc3256,
            (4808345675.024, 4808345765.024),
        ),
    ],
)
def test_find_projected_min_max_array(in_min_max, in_array, expected_result):
    from xradio.measurement_set._utils._msv2._tables.read import (
        find_projected_min_max_array,
    )

    res = find_projected_min_max_array(in_min_max, in_array)
    np.set_printoptions(precision=13)
    np.set_printoptions(formatter={"float": "{: 0.3f}".format})
    np.testing.assert_array_almost_equal(res, expected_result)


def test_make_taql_where_between_min_max_empty(ms_empty_required):
    from xradio.measurement_set._utils._msv2._tables.read import (
        make_taql_where_between_min_max,
    )

    res = make_taql_where_between_min_max(
        (0, 10), ms_empty_required.fname, "POINTING", "TIME"
    )

    assert res is None


@pytest.mark.parametrize(
    "in_min_max, expected_result",
    [
        ((0, 1.0e9), f"where TIME >= {-tol_eps} AND TIME <= {2e9 + tol_eps}"),
        ((0, 3.0e9), f"where TIME >= {-tol_eps} AND TIME <= {3e9 + tol_eps}"),
    ],
)
def test_make_taql_where_between_min_max(
    ms_minimal_required, in_min_max, expected_result
):
    from xradio.measurement_set._utils._msv2._tables.read import (
        make_taql_where_between_min_max,
    )

    res = make_taql_where_between_min_max(
        in_min_max, ms_minimal_required.fname, "POINTING", "TIME"
    )

    assert res == expected_result


def test_extract_table_attributes_ant(ms_minimal_required):
    from xradio.measurement_set._utils._msv2._tables.read import (
        extract_table_attributes,
    )

    res = extract_table_attributes(str(Path(ms_minimal_required.fname) / "ANTENNA"))
    assert len(res) == 2
    assert "column_descriptions" in res
    assert "info" in res
    assert len(res["column_descriptions"]) == 8
    cols = ["DISH_DIAMETER", "FLAG_ROW", "MOUNT", "NAME", "OFFSET", "POSITION"]
    assert all([col in res["column_descriptions"] for col in cols])
    assert all(sub in res["info"] for sub in ["type", "subType", "readme"])


def test_add_units_measures(msv4_xds_min):
    from xradio.measurement_set._utils._msv2._tables.read import add_units_measures

    col_descr = {"column_descriptions": {}}
    xds_vars = {
        "UVW": msv4_xds_min.UVW,
        "time": msv4_xds_min.time,
    }
    res = add_units_measures(xds_vars, col_descr)

    assert res["UVW"].attrs == xds_vars["UVW"].attrs
    assert res["time"].attrs == xds_vars["time"].attrs


def test_add_units_measures_dubious_units(msv4_xds_min):
    from xradio.measurement_set._utils._msv2._tables.read import add_units_measures

    col_descr = {
        "column_descriptions": {
            "UVW": {"keywords": {}},
            "TIME": {"keywords": {"QuantumUnits": "dubious/units"}},
            "DATA": {"keywords": {"QuantumUnits": (None, None)}},
            "TIME_CENTROID": {"keywords": {"QuantumUnits": (3.1,)}},
        },
    }
    xds_vars = {
        "time": msv4_xds_min.time,
        "DATA": msv4_xds_min.VISIBILITY,
        "TIME_CENTROID": msv4_xds_min.TIME_CENTROID,
    }

    res = add_units_measures(xds_vars, col_descr)
    assert res["time"].attrs == xds_vars["time"].attrs
    assert res["DATA"].attrs == xds_vars["DATA"].attrs
    assert res["TIME_CENTROID"].attrs == xds_vars["TIME_CENTROID"].attrs


def test_get_pad_value_in_tablerow_column(ms_minimal_required):
    from xradio.measurement_set._utils._msv2._tables.read import (
        get_pad_value,
        get_pad_value_in_tablerow_column,
    )
    from xradio.measurement_set._utils._msv2._tables.table_query import open_table_ro

    with open_table_ro(ms_minimal_required.fname + "/POLARIZATION") as tb_tool:
        trows = tb_tool.row([], exclude=True)[0:12]
        val_corr_type = get_pad_value_in_tablerow_column(trows, "CORR_TYPE")
        assert val_corr_type == get_pad_value(np.int32)

        val_corr_prod = get_pad_value_in_tablerow_column(trows, "CORR_PRODUCT")
        assert val_corr_prod == get_pad_value(np.int32)

        with pytest.raises(RuntimeError, match="unexpected type"):
            val_num_corr = get_pad_value_in_tablerow_column(trows, "NUM_CORR")
            assert val_num_corr == get_pad_value(np.int32)
        with pytest.raises(RuntimeError, match="unexpected type"):
            val_flag_row = get_pad_value_in_tablerow_column(trows, "FLAG_ROW")
            assert val_flag_row == get_pad_value(np.int32)


def test_get_pad_value_uvw(msv4_xds_min):
    from xradio.measurement_set._utils._msv2._tables.read import get_pad_value

    res = get_pad_value(msv4_xds_min.data_vars["UVW"].dtype)

    assert np.isnan(res)
    assert np.isnan(get_pad_value(np.float64))


def test_get_pad_value_n_polynomial(pointing_xds_min):
    from xradio._utils.list_and_array import get_pad_value

    res = get_pad_value(pointing_xds_min.coords["antenna_name"].dtype)

    assert res == get_pad_value(str)


def test_get_pad_value_baseline_id(msv4_xds_min):
    from xradio._utils.list_and_array import get_pad_value

    res = get_pad_value(msv4_xds_min.coords["baseline_id"].dtype)

    assert res == get_pad_value(np.int64)


def test_redimension_ms_subtable_source(generic_source_xds_min):
    import xarray as xr

    from xradio.measurement_set._utils._msv2._tables.read import redimension_ms_subtable

    res = redimension_ms_subtable(generic_source_xds_min, "SOURCE")
    assert isinstance(res, xr.Dataset)
    src_coords = ["SOURCE_ID", "TIME", "SPECTRAL_WINDOW_ID", "PULSAR_ID"]
    assert all([coord in res.coords for coord in src_coords])


def test_redimension_ms_subtable_phase_cal(generic_phase_cal_xds_min):
    import xarray as xr

    from xradio.measurement_set._utils._msv2._tables.read import redimension_ms_subtable

    res = redimension_ms_subtable(generic_phase_cal_xds_min, "PHASE_CAL")
    assert isinstance(res, xr.Dataset)
    src_coords = ["ANTENNA_ID", "TIME", "SPECTRAL_WINDOW_ID"]
    assert all([coord in res.coords for coord in src_coords])


def test_is_ephem_subtable_ms(ms_minimal_required):
    from xradio.measurement_set._utils._msv2._tables.read import is_ephem_subtable

    res = is_ephem_subtable(ms_minimal_required.fname)
    assert res is False


def test_add_ephemeris_vars(ms_minimal_required):
    import xarray as xr

    from xradio.measurement_set._utils._msv2._tables.read import add_ephemeris_vars

    # would need an ephem_xds fixture
    ephem_xds = xr.Dataset(data_vars={"MJD": ("row", np.array([]))})
    res = add_ephemeris_vars(
        Path(ms_minimal_required.fname) / "FIELD" / "EPHEM0_f0.tab", ephem_xds
    )
    assert res
    assert all(
        [xvar in res.data_vars for xvar in ["ephemeris_row_id", "ephemeris_id", "time"]]
    )


def test_is_nested_ms_empty():
    from xradio.measurement_set._utils._msv2._tables.read import is_nested_ms

    with pytest.raises(KeyError, match="other"):
        res = is_nested_ms({})
        assert res is False


def test_is_nested_ms_ant(ms_minimal_required):
    from xradio.measurement_set._utils._msv2._tables.read import (
        extract_table_attributes,
        is_nested_ms,
    )

    ctds_attrs = extract_table_attributes(ms_minimal_required.fname)
    attrs = {"other": {"msv2": {"ctds_attrs": ctds_attrs}}}

    res = is_nested_ms(attrs)
    assert res is True


def test_is_nested_ms_ms_min(ms_minimal_required):
    from xradio.measurement_set._utils._msv2._tables.read import (
        extract_table_attributes,
        is_nested_ms,
    )

    ctds_attrs = extract_table_attributes(ms_minimal_required.fname)
    attrs = {"other": {"msv2": {"ctds_attrs": ctds_attrs}}}

    res = is_nested_ms(attrs)
    assert res is True


def test_load_generic_table_antenna(ms_minimal_required):
    import xarray as xr

    from xradio.measurement_set._utils._msv2._tables.read import load_generic_table

    res = load_generic_table(ms_minimal_required.fname, "ANTENNA")
    assert res
    assert type(res) is xr.Dataset
    assert all([dim in res.dims for dim in ["row", "dim_1"]])


def test_load_generic_table_feed(ms_minimal_required):
    import xarray as xr

    from xradio.measurement_set._utils._msv2._tables.read import load_generic_table

    res = load_generic_table(ms_minimal_required.fname, "FEED")
    assert res
    assert type(res) is xr.Dataset
    assert all(
        [
            dim in res.dims
            for dim in ["ANTENNA_ID", "SPECTRAL_WINDOW_ID", "dim_1", "dim_2", "dim_3"]
        ]
    )


def test_load_generic_table_polarization(ms_minimal_required):
    import xarray as xr

    from xradio.measurement_set._utils._msv2._tables.read import load_generic_table

    res = load_generic_table(ms_minimal_required.fname, "POLARIZATION")
    assert res
    assert type(res) is xr.Dataset
    assert all([dim in res.dims for dim in ["row", "dim_1", "dim_2"]])


@pytest.mark.parametrize(
    "input_name, expected_additional_columns",
    [
        (
            "ms_minimal_required",
            ["NUM_LINES", "TRANSITION", "REST_FREQUENCY", "SYSVEL"],
        ),
        ("ms_minimal_misbehaved", []),
    ],
)
def test_load_generic_table_source(input_name, expected_additional_columns, request):
    import xarray as xr

    from xradio.measurement_set._utils._msv2._tables.read import load_generic_table

    fixture = request.getfixturevalue(input_name)
    input_path = fixture.fname

    res = load_generic_table(input_path, "SOURCE")
    assert res
    assert type(res) is xr.Dataset
    assert all(
        [
            dim in res.dims
            for dim in [
                "SOURCE_ID",
                "TIME",
                "SPECTRAL_WINDOW_ID",
                "dim_1",
                "dim_2",
                "dim_3",
            ]
        ]
    )
    assert all(
        [
            xvar in res.data_vars
            for xvar in ["NAME", "CALIBRATION_GROUP", "DIRECTION", "PROPER_MOTION"]
            + expected_additional_columns
        ]
    )


def test_load_generic_table_state(ms_minimal_required):
    import xarray as xr

    from xradio.measurement_set._utils._msv2._tables.read import load_generic_table

    res = load_generic_table(ms_minimal_required.fname, "STATE")
    assert res
    assert type(res) is xr.Dataset
    assert all([dim in res.dims for dim in ["row"]])
    assert all(
        [
            xvar in res.data_vars
            for xvar in ["CAL", "LOAD", "SIG", "SUB_SCAN", "OBS_MODE"]
        ]
    )


def test_load_generic_table_pointing(ms_minimal_required):
    import xarray as xr

    from xradio.measurement_set._utils._msv2._tables.read import load_generic_table

    res = load_generic_table(ms_minimal_required.fname, "POINTING")
    assert res
    assert type(res) is xr.Dataset
    assert all([dim in res.dims for dim in ["TIME", "ANTENNA_ID", "dim_1", "dim_2"]])
    assert all(
        [
            xvar in res.data_vars
            for xvar in [
                "DIRECTION",
                "INTERVAL",
                "NAME",
                "NUM_POLY",
                "TARGET",
                "TIME_ORIGIN",
                "TRACKING",
            ]
        ]
    )


def test_load_generic_table_ephem(ms_minimal_required):
    import xarray as xr

    from xradio.measurement_set._utils._msv2._tables.read import load_generic_table

    res = load_generic_table(ms_minimal_required.fname, "FIELD/EPHEM0_FIELDNAME.tab")
    exp_attrs = {
        "other": {
            "msv2": {
                "bad_cols": [],
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
    for key, val in exp_attrs["other"]["msv2"].items():
        key in res.attrs["other"]["msv2"] and val == res.attrs["other"]["msv2"][key]


def test_load_generic_table_weather(ms_minimal_required):
    import xarray as xr

    from xradio.measurement_set._utils._msv2._tables.read import load_generic_table

    res = load_generic_table(ms_minimal_required.fname, "WEATHER")
    assert res
    assert type(res) is xr.Dataset
    assert all([dim in res.dims for dim in ["ANTENNA_ID", "TIME"]])
    assert all([var in res.data_vars for var in ["H2O"]])


def test_load_generic_cols_state(ms_minimal_required):
    import xarray as xr

    from xradio.measurement_set._utils._msv2._tables.read import load_generic_cols
    from xradio.measurement_set._utils._msv2._tables.table_query import open_table_ro

    subt_state = str(Path(ms_minimal_required.fname) / "STATE")
    with open_table_ro(subt_state) as tb_tool:
        ignore_cols = ["FLAG_ROW"]
        res = load_generic_cols(
            ms_minimal_required.fname, tb_tool, timecols=["TIME"], ignore=ignore_cols
        )
        assert res
        assert isinstance(res, tuple)
        assert res[0] == {}
        assert all([col not in res[1] for col in ignore_cols])
        assert all(
            [var in res[1] for var in ["LOAD", "OBS_MODE", "REF", "SIG", "SUB_SCAN"]]
        )
        assert all([isinstance(val, xr.DataArray) for val in res[1].values()])


def test_load_generic_cols_spw(ms_minimal_required):
    import xarray as xr

    from xradio.measurement_set._utils._msv2._tables.read import load_generic_cols
    from xradio.measurement_set._utils._msv2._tables.table_query import open_table_ro

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
            "CHAN_FREQ",
            "REF_FREQUENCY",
            "EFFECTIVE_BW",
            "RESOLUTION",
            "FREQ_GROUP",
            "FREQ_GROUP_NAME",
            "IF_CONV_CHAIN",
            "NAME",
            "NET_SIDEBAND",
            "NUM_CHAN",
            "TOTAL_BANDWIDTH",
        ]
        assert all([var in res[1] for var in expected_vars])
        assert all([isinstance(val, xr.DataArray) for val in res[1].values()])


def test_read_flat_col_chunk_time(ms_minimal_required):
    from xradio.measurement_set._utils._msv2._tables.read import read_flat_col_chunk

    res = read_flat_col_chunk(ms_minimal_required.fname, "TIME", (10,), [0, 1, 5], 0, 0)
    assert isinstance(res, np.ndarray)
    assert res.shape == (3,)
    assert np.all(res >= 1e9)


def test_read_flat_col_chunk_sigma(ms_minimal_required):
    from xradio.measurement_set._utils._msv2._tables.read import read_flat_col_chunk

    npols = ms_minimal_required.descr["npols"]
    res = read_flat_col_chunk(
        ms_minimal_required.fname, "SIGMA", (10, npols), [4, 5, 8, 9], 0, 0
    )
    assert isinstance(res, np.ndarray)
    assert res.shape == (4, npols)
    assert np.all(res == 1)


def test_read_flat_col_chunk_flag(ms_minimal_required):
    from xradio.measurement_set._utils._msv2._tables.read import read_flat_col_chunk

    npols = ms_minimal_required.descr["npols"]
    nchans = ms_minimal_required.descr["nchans"]
    res = read_flat_col_chunk(
        ms_minimal_required.fname, "FLAG", (10, nchans, npols), [0, 1, 2], 0, 0
    )
    assert isinstance(res, np.ndarray)
    assert res.shape == (3, nchans, npols)
    assert np.all(~res)


DDI0_PARTITION = {"DATA_DESC_ID": [0], "OBS_MODE": ["scan_intent#subscan_intent"]}


def _ddi0_main_rows(main_tb, max_elems=None):
    """The DDI 0 partition rows of an MS and their time / baseline indices."""
    from xradio.measurement_set._utils._msv2._tables.read_rows import MainTableRows
    from xradio.measurement_set._utils._msv2.conversion import calc_indx_for_row_split
    from xradio.measurement_set._utils._msv2.partition_queries import (
        partition_main_rows,
    )

    kwargs = {} if max_elems is None else {"max_elems": max_elems}
    main_rows = MainTableRows(
        main_tb, partition_main_rows(main_tb, DDI0_PARTITION), **kwargs
    )
    tidxs, bidxs, ant1, _ant2, utime = calc_indx_for_row_split(main_rows)
    return main_rows, tidxs, bidxs, (len(utime), len(ant1))


@pytest.mark.parametrize(
    "col", ["DATA", "FLAG", "UVW", "TIME_CENTROID", "EXPOSURE", "WEIGHT", "SIGMA"]
)
def test_read_col_conversion_numpy_matches_taql_reference(
    ms_minimal_required, col, taql_main_reference
):
    """The row reads give exactly the values of a TaQL selection's getcol
    (same dtype, pads and grid), also with tiny read calls."""
    from xradio.measurement_set._utils._msv2._tables.read import (
        read_col_conversion_numpy,
    )
    from xradio.measurement_set._utils._msv2._tables.table_query import open_table_ro

    fname = ms_minimal_required.fname
    try:
        expected = taql_main_reference(fname, DDI0_PARTITION, col)
    except RuntimeError:
        expected = None  # e.g. undefined WEIGHT cells: the column is skipped

    with open_table_ro(fname) as main_tb:
        for max_elems in (2**26, 5):
            main_rows, tidxs, bidxs, cshape = _ddi0_main_rows(main_tb, max_elems)
            if expected is None:
                with pytest.raises(RuntimeError):
                    read_col_conversion_numpy(main_rows, col, cshape, tidxs, bidxs)
                continue
            data = read_col_conversion_numpy(main_rows, col, cshape, tidxs, bidxs)
            assert data.dtype == expected.dtype
            assert data.shape == expected.shape
            # bit-identical, including the NaN pads of the missing cells
            assert data.tobytes() == expected.tobytes()


@pytest.mark.parametrize("time_chunksize", [1, 7, 1000])
@pytest.mark.parametrize("col", ["DATA", "FLAG"])
def test_read_col_conversion_dask_matches_numpy(
    ms_minimal_required, col, time_chunksize
):
    import dask.array as da

    from xradio.measurement_set._utils._msv2._tables.read import (
        read_col_conversion_dask,
        read_col_conversion_numpy,
    )
    from xradio.measurement_set._utils._msv2._tables.table_query import open_table_ro

    fname = ms_minimal_required.fname
    with open_table_ro(fname) as main_tb:
        main_rows, tidxs, bidxs, cshape = _ddi0_main_rows(main_tb)
        expected = read_col_conversion_numpy(main_rows, col, cshape, tidxs, bidxs)
        lazy = read_col_conversion_dask(
            main_rows, col, cshape, tidxs, bidxs, time_chunksize
        )
        assert isinstance(lazy, da.Array)
        assert (
            lazy.chunks[0] == da.core.normalize_chunks(time_chunksize, (cshape[0],))[0]
        )
        assert lazy.chunks[1:] == tuple((n,) for n in expected.shape[1:])
        data = lazy.compute(scheduler="synchronous")
    assert data.dtype == expected.dtype
    # missing cells padded as in the numpy path (FLAG False, NaN visibilities)
    assert data.tobytes() == expected.tobytes()


def _graph_arrays(obj, found=None, visited=None, depth=0):
    """numpy arrays (by base array id) reachable from a dask graph's values."""
    found = {} if found is None else found
    visited = set() if visited is None else visited
    if depth > 12 or id(obj) in visited:
        return found
    visited.add(id(obj))
    if isinstance(obj, np.ndarray):
        base = obj
        while isinstance(base.base, np.ndarray):
            base = base.base
        found[id(base)] = base
        return found
    if isinstance(obj, dict):
        children = list(obj.values())
    elif isinstance(obj, list | tuple | set):
        children = list(obj)
    elif isinstance(obj, str | bytes | int | float | type(None)):
        return found
    else:
        children = [
            getattr(obj, slot)
            for slot in getattr(type(obj), "__slots__", ())
            if hasattr(obj, slot)
        ] + [
            getattr(obj, attr)
            for attr in ("args", "kwargs", "value")
            if hasattr(obj, attr)
        ]
    for child in children:
        _graph_arrays(child, found, visited, depth + 1)
    return found


def test_read_col_conversion_dask_shares_row_indices(ms_minimal_required):
    """
    parallel_mode="time": the blocks of all large columns share one graph key
    with the partition's row / time / baseline indices; the graphs hold no
    per-block or per-column copies of them (that was 16 bytes per row and
    column, kept until to_zarr returns).
    """
    import dask

    from xradio.measurement_set._utils._msv2._tables.read import (
        read_col_conversion_dask,
    )
    from xradio.measurement_set._utils._msv2._tables.read_rows import TimeChunkRows
    from xradio.measurement_set._utils._msv2._tables.table_query import open_table_ro

    fname = ms_minimal_required.fname
    with open_table_ro(fname) as main_tb:
        main_rows, tidxs, bidxs, cshape = _ddi0_main_rows(main_tb)
        lazy = {
            col: read_col_conversion_dask(main_rows, col, cshape, tidxs, bidxs, 2)
            for col in ("DATA", "FLAG")
        }
        # the partition's own index arrays (by base array, as _graph_arrays)
        held = set(_graph_arrays([main_rows.rows, tidxs, bidxs]))
        found = {}
        for array in lazy.values():
            graph = dict(array.__dask_graph__())
            shared = [
                value
                for key, value in graph.items()
                if isinstance(key, str) and key.startswith("TimeChunkRows")
            ]
            assert len(shared) == 1
            _graph_arrays(graph, found)
        keys = [set(dict(array.__dask_graph__())) for array in lazy.values()]
        shared_keys = {
            k for k in keys[0] & keys[1] if str(k).startswith("TimeChunkRows")
        }
        assert len(shared_keys) == 1  # one key for both columns
        extra = sum(a.nbytes for k, a in found.items() if k not in held)
        n_chunks = len(lazy["DATA"].chunks[0])
        # chunk bounds only (+ 8 bytes per row for rows not in time order)
        assert extra <= 16 * (n_chunks + 1) + 8 * main_rows.nrows()
        chunk_rows = main_rows.time_chunk_rows(
            tidxs, bidxs, lazy["DATA"].chunks[0], cshape[1]
        )
        assert isinstance(chunk_rows, TimeChunkRows)
        if chunk_rows.order is None:
            assert extra <= 16 * (n_chunks + 1)
        computed = dask.compute(*lazy.values(), scheduler="threads", num_workers=2)
    assert all(isinstance(value, np.ndarray) for value in computed)


# --- sub-table cache (TEMPORARY XRADIO_MSV2_SUBTABLE_CACHE switch) ----------------


def _assert_xds_bit_identical(a, b):
    """Same variables (and order), dims, dtypes, bits, indexes and attrs."""
    assert list(a.variables) == list(b.variables)
    assert dict(a.sizes) == dict(b.sizes)
    assert set(a.xindexes) == set(b.xindexes)
    assert repr(a.attrs) == repr(b.attrs)
    for name, var_a in a.variables.items():
        var_b = b.variables[name]
        assert var_a.dims == var_b.dims, name
        assert var_a.dtype == var_b.dtype, name
        if var_a.dtype == object:
            np.testing.assert_array_equal(var_a.values, var_b.values, err_msg=name)
        else:
            assert (
                np.ascontiguousarray(var_a.values).tobytes()
                == np.ascontiguousarray(var_b.values).tobytes()
            ), name
        assert repr(var_a.attrs) == repr(var_b.attrs), name


def _load_both_ways(path, tname, **kwargs):
    """load_generic_table without and with an active sub-table cache (twice: the
    second call of a memoized table is a memo hit)."""
    from xradio.measurement_set._utils._msv2._tables.read import load_generic_table
    from xradio.measurement_set._utils._msv2._tables.subtable_cache import (
        SubtableCache,
        activate_subtable_cache,
    )

    def load():
        try:
            return load_generic_table(path, tname, **kwargs)
        except Exception as exc:
            return f"{type(exc).__name__}: {exc}"

    expected = load()
    cache = SubtableCache()
    with activate_subtable_cache(cache):
        actual = [load(), load()]
    return expected, actual, cache


@pytest.mark.parametrize(
    "ms_fixture", ["ms_minimal_required", "ms_minimal_misbehaved", "ms_empty_complete"]
)
def test_load_generic_table_cached_identical(ms_fixture, request):
    """Every sub-table of the generated MSs (incl. misbehaving, variable-shape
    columns) loads identically with the vectorized loader and the memo."""
    from xradio.measurement_set._utils._msv2.subtables import subt_rename_ids

    fname = request.getfixturevalue(ms_fixture).fname
    tnames = sorted(
        str(p.parent.relative_to(fname)) for p in Path(fname).glob("*/table.dat")
    ) + sorted(
        str(p.parent.relative_to(fname))
        for p in Path(fname).glob("FIELD/EPHEM*/table.dat")
    )
    assert tnames
    for tname in tnames:
        kwargs = {}
        if tname in subt_rename_ids:
            kwargs["rename_ids"] = subt_rename_ids[tname]
        if tname in ("POINTING", "SYSCAL", "WEATHER", "PHASE_CAL"):
            kwargs["timecols"] = ["TIME"]
        if "EPHEM" in tname:
            kwargs["timecols"] = ["MJD"]
        expected, actual, _ = _load_both_ways(fname, tname, **kwargs)
        for result in actual:
            if isinstance(expected, str):
                assert result == expected, tname
            else:
                _assert_xds_bit_identical(expected, result)


@pytest.fixture(scope="module")
def generic_cols_table(tmp_path_factory):
    """
    A table with the column kinds load_generic_cols handles: scalars of every
    value type (tables.row() gives Python scalars), fixed and variable-shape
    arrays, undefined cells, string arrays, an IncrementalStMan and a
    TiledShapeStMan array.
    """
    tables = pytest.importorskip("casacore.tables")
    path = str(tmp_path_factory.mktemp("generic_cols") / "GENERIC")
    nrows = 12
    desc = tables.maketabdesc(
        [
            tables.makescacoldesc("ANTENNA_ID", 0),
            tables.makescacoldesc("TIME", 0.0),
            tables.makescacoldesc("S_FLOAT", 0.0, valuetype="float"),
            tables.makescacoldesc("S_SHORT", 0, valuetype="short"),
            tables.makescacoldesc("S_UCHAR", 0, valuetype="uchar"),
            tables.makescacoldesc("S_BOOL", False),
            tables.makescacoldesc("S_COMPLEX", 0j, valuetype="complex"),
            tables.makescacoldesc("S_STRING", ""),
            tables.makearrcoldesc("A_FIXED", 0.0, shape=[2, 3], valuetype="float"),
            tables.makearrcoldesc("A_VAR", 0.0, ndim=1),
            tables.makearrcoldesc("A_UNDEF", 0.0, ndim=1),
            tables.makearrcoldesc("A_STRING", "", ndim=1),
            tables.makearrcoldesc(
                "A_ISM",
                0,
                ndim=1,
                valuetype="int",
                datamanagertype="IncrementalStMan",
                datamanagergroup="ISM",
            ),
            tables.makearrcoldesc(
                "A_TSM",
                0j,
                ndim=2,
                valuetype="complex",
                datamanagertype="TiledShapeStMan",
                datamanagergroup="TSM",
            ),
        ]
    )
    rng = np.random.default_rng(7)
    with tables.table(path, desc, nrow=nrows, readonly=False, ack=False) as tb:
        tb.putcol("ANTENNA_ID", np.arange(nrows, dtype=np.int32) % 3)
        tb.putcol("TIME", 5e9 + np.arange(nrows) * 1.5)
        tb.putcol("S_FLOAT", rng.normal(size=nrows).astype(np.float32))
        tb.putcol("S_SHORT", np.arange(nrows, dtype=np.int16) - 5)
        tb.putcol("S_UCHAR", np.arange(nrows, dtype=np.uint8) + 200)
        tb.putcol("S_BOOL", rng.random(nrows) < 0.5)
        tb.putcol("S_COMPLEX", (rng.normal(size=nrows) + 1j).astype(np.complex64))
        tb.putcol("S_STRING", np.array([f"name{'x' * (r % 4)}" for r in range(nrows)]))
        tb.putcol("A_FIXED", rng.normal(size=(nrows, 2, 3)).astype(np.float32))
        for row in range(nrows):
            tb.putcell("A_VAR", row, rng.normal(size=2 + row % 3))
            if row % 4:
                tb.putcell("A_UNDEF", row, rng.normal(size=3))
            tb.putcell("A_STRING", row, np.array(["a", f"b{row}"]))
            tb.putcell("A_ISM", row, np.full(2, row // 4, dtype=np.int32))
            tb.putcell("A_TSM", row, np.full((2, 2), row + 1j, dtype=np.complex64))
    return path


@pytest.mark.parametrize(
    "where", [None, "where ANTENNA_ID IN [1, 2]", "where ROWID() IN [0, 3, 7]"]
)
def test_load_generic_cols_vectorized_matches_row_loader(generic_cols_table, where):
    from xradio.measurement_set._utils._msv2._tables.read import (
        load_generic_cols,
        load_generic_cols_vectorized,
    )
    from xradio.measurement_set._utils._msv2._tables.table_query import (
        open_query,
        open_table_ro,
    )

    with open_table_ro(generic_cols_table) as gtable:
        with open_query(gtable, f"select * from $gtable {where or ''}") as tb_tool:
            expected = load_generic_cols(
                generic_cols_table, tb_tool, ["TIME"], ["S_UCHAR"]
            )
            actual = load_generic_cols_vectorized(
                generic_cols_table, tb_tool, ["TIME"], ["S_UCHAR"]
            )
    for exp, act in zip(expected, actual, strict=True):
        assert list(exp) == list(act)  # same variables, same order
        for name in exp:
            assert exp[name].dims == act[name].dims, name
            assert exp[name].dtype == act[name].dtype, name
            np.testing.assert_array_equal(exp[name].values, act[name].values)
    assert "S_UCHAR" not in actual[1]
    assert actual[1]["S_FLOAT"].dtype == np.float64  # as tables.row() gives it
    assert actual[1]["S_SHORT"].dtype == np.int64
    assert actual[1]["A_FIXED"].dtype == np.float32


def test_getcol_as_tablerow_stack_falls_back_to_rows(generic_cols_table):
    """Columns that getcol cannot give as row() would (varying shape, undefined
    cells, string arrays, tiled storage) are left to tables.row()."""
    from xradio.measurement_set._utils._msv2._tables.read import (
        find_loadable_cols,
        getcol_as_tablerow_stack,
    )
    from xradio.measurement_set._utils._msv2._tables.table_query import open_table_ro

    with open_table_ro(generic_cols_table) as tb_tool:
        storage = {
            col: dm["TYPE"]
            for dm in tb_tool.getdminfo().values()
            for col in dm["COLUMNS"]
        }
        vectorized = {
            col: getcol_as_tablerow_stack(tb_tool, col, col_type, storage.get(col))
            is not None
            for col, col_type in find_loadable_cols(tb_tool, []).items()
        }
    assert {col for col, ok in vectorized.items() if not ok} == {
        "A_VAR",
        "A_UNDEF",
        "A_STRING",
        "A_TSM",
    }


class _ReadCallSpy:
    """Table proxy recording the elements read by every getcol / getcolnp call."""

    def __init__(self, tb):
        self._tb = tb
        self.calls = []

    def __getattr__(self, name):
        return getattr(self._tb, name)

    def getcol(self, col, *args, **kwargs):
        values = self._tb.getcol(col, *args, **kwargs)
        size = len(values) if isinstance(values, list) else np.asarray(values).size
        self.calls.append(("getcol", col, args, size))
        return values

    def getcolnp(self, col, out, *args):
        self.calls.append(("getcolnp", col, args, out.size))
        return self._tb.getcolnp(col, out, *args)


@pytest.mark.parametrize("max_elems", [None, 7, 1])
@pytest.mark.parametrize("where", [None, "where ANTENNA_ID IN [1, 2]"])
def test_load_generic_cols_vectorized_reads_are_bounded(
    generic_cols_table, max_elems, where, monkeypatch
):
    """
    Every column read of the vectorized loader covers at most
    SUBTABLE_READ_MAX_ELEMS elements (python-casacore #130; at least one row),
    with a range (never one unbounded whole-column getcol), and the result
    is still that of the row() loader.
    """
    from xradio.measurement_set._utils._msv2._tables import read as read_module
    from xradio.measurement_set._utils._msv2._tables.table_query import (
        open_query,
        open_table_ro,
    )

    if max_elems is not None:
        monkeypatch.setattr(read_module, "SUBTABLE_READ_MAX_ELEMS", max_elems)
    limit = read_module.SUBTABLE_READ_MAX_ELEMS
    with open_table_ro(generic_cols_table) as gtable:
        with open_query(gtable, f"select * from $gtable {where or ''}") as tb_tool:
            expected = read_module.load_generic_cols(
                generic_cols_table, tb_tool, ["TIME"], ["S_UCHAR"]
            )
            spy = _ReadCallSpy(tb_tool)
            actual = read_module.load_generic_cols_vectorized(
                generic_cols_table, spy, ["TIME"], ["S_UCHAR"]
            )
            nrows = tb_tool.nrows()
    for exp, act in zip(expected, actual, strict=True):
        assert list(exp) == list(act)
        for name in exp:
            assert exp[name].dtype == act[name].dtype, name
            np.testing.assert_array_equal(exp[name].values, act[name].values)
    assert spy.calls
    for kind, col, args, size in spy.calls:
        assert len(args) >= 2, (kind, col, args)  # startrow, nrow given
        assert args[1] < nrows or nrows == 1, (kind, col, args)  # not all rows
        # at most max_elems elements, or one row
        assert size <= limit or args[1] == 1, (kind, col, args, size)


def test_getcol_as_tablerow_stack_raises_memory_error(generic_cols_table, monkeypatch):
    """A MemoryError is not turned into a row() read (which needs more memory)."""
    from xradio.measurement_set._utils._msv2._tables import read as read_module
    from xradio.measurement_set._utils._msv2._tables.table_query import open_table_ro

    def fail(*args, **kwargs):
        raise MemoryError("no memory")

    monkeypatch.setattr(read_module, "read_column_rows", fail)
    with open_table_ro(generic_cols_table) as tb_tool:
        for col, col_type, manager in (
            ("TIME", "double", "StandardStMan"),
            ("A_FIXED", "float", "StandardStMan"),
        ):
            with pytest.raises(MemoryError):
                read_module.getcol_as_tablerow_stack(tb_tool, col, col_type, manager)
    with pytest.raises(MemoryError):
        read_module.load_sorted_column(generic_cols_table, "TIME")

    def other_error(*args, **kwargs):
        raise RuntimeError("cannot read")

    monkeypatch.setattr(read_module, "read_column_rows", other_error)
    assert read_module.load_sorted_column(generic_cols_table, "TIME") is None
    with open_table_ro(generic_cols_table) as tb_tool:
        assert (
            read_module.getcol_as_tablerow_stack(
                tb_tool, "TIME", "double", "StandardStMan"
            )
            is None
        )


def test_load_generic_table_memoized(ms_minimal_required):
    """Memoized sub-tables are loaded once per cache, other tables every time."""
    expected, actual, cache = _load_both_ways(ms_minimal_required.fname, "ANTENNA")
    for result in actual:
        _assert_xds_bit_identical(expected, result)
    assert cache.stats["memo_misses"] == 1 and cache.stats["memo_hits"] == 1
    actual[0]["POSITION"].values[:] = 0  # callers own their copies
    _assert_xds_bit_identical(expected, actual[1])

    _, _, cache = _load_both_ways(ms_minimal_required.fname, "FIELD")
    assert cache.stats["memo_misses"] == 0 and cache.stats["memo_hits"] == 0


@pytest.mark.parametrize(
    "min_max",
    [(0, 1e10), (4.8e9, 4.8e9), (1e9, 2e9), (6e9, 7e9), (5e9 + 2, 5e9 + 6.1)],
)
def test_find_projected_min_max_table_cached(generic_cols_table, min_max):
    import os

    from xradio.measurement_set._utils._msv2._tables.read import (
        find_projected_min_max_table,
    )
    from xradio.measurement_set._utils._msv2._tables.subtable_cache import (
        SubtableCache,
        activate_subtable_cache,
    )

    path, name = os.path.split(generic_cols_table)
    expected = find_projected_min_max_table(min_max, path, name, "TIME")
    for n_partitions in (2, 1):
        cache = SubtableCache(n_partitions=n_partitions)
        with activate_subtable_cache(cache):
            for _ in range(3):
                assert (
                    find_projected_min_max_table(min_max, path, name, "TIME")
                    == expected
                )
        assert cache.stats["value_builds"] == 1
        # a cache of one partition reads the first request per partition
        assert cache.stats["value_deferred"] == (n_partitions == 1)


def test_find_projected_min_max_table_cached_errors(
    generic_cols_table, ms_empty_required
):
    """Columns that cannot be projected fail as without cache (S_BOOL: fewer than
    two non-zero values... here: an empty table and a missing column)."""
    import os

    from xradio.measurement_set._utils._msv2._tables.read import (
        find_projected_min_max_table,
    )
    from xradio.measurement_set._utils._msv2._tables.subtable_cache import (
        SubtableCache,
        activate_subtable_cache,
    )

    def project(cache, *args):
        with activate_subtable_cache(cache):
            try:
                return find_projected_min_max_table(*args)
            except Exception as exc:
                return type(exc).__name__

    path, name = os.path.split(generic_cols_table)
    for args in [
        ((0, 1e10), ms_empty_required.fname, "POINTING", "TIME"),  # no rows
        ((0, 1e10), path, name, "NO_SUCH_COLUMN"),
        (None, path, name, "TIME"),
    ]:
        assert project(SubtableCache(n_partitions=2), *args) == project(None, *args)
