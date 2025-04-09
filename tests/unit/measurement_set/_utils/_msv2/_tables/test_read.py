from contextlib import nullcontext as does_not_raise
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
        ("ddi_xds_min", "TIME", pytest.raises(RuntimeError)),
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


def test_add_units_measures(msv4_min_correlated_xds):
    from xradio.measurement_set._utils._msv2._tables.read import add_units_measures

    col_descr = {"column_descriptions": {}}
    xds_vars = {
        "UVW": msv4_min_correlated_xds.UVW,
        "time": msv4_min_correlated_xds.time,
    }
    res = add_units_measures(xds_vars, col_descr)
    assert xds_vars["UVW"].attrs
    assert xds_vars["time"].attrs


def test_add_units_measures_dubious_units(msv4_min_correlated_xds):
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
        "time": msv4_min_correlated_xds.time,
        "DATA": msv4_min_correlated_xds.VISIBILITY,
        "TIME_CENTROID": msv4_min_correlated_xds.TIME_CENTROID,
    }

    res = add_units_measures(xds_vars, col_descr)
    assert xds_vars["time"].attrs
    assert xds_vars["DATA"].attrs
    assert xds_vars["TIME_CENTROID"].attrs


def test_get_pad_value_in_tablerow_column(ms_minimal_required):
    from xradio.measurement_set._utils._msv2._tables.table_query import open_table_ro
    from xradio.measurement_set._utils._msv2._tables.read import (
        get_pad_value,
        get_pad_value_in_tablerow_column,
    )

    with open_table_ro(ms_minimal_required.fname + "/POLARIZATION") as tb_tool:
        trows = tb_tool.row([], exclude=True)[0:12]
        val_corr_type = get_pad_value_in_tablerow_column(trows, "CORR_TYPE")
        assert val_corr_type == get_pad_value(np.int32)

        val_corr_prod = get_pad_value_in_tablerow_column(trows, "CORR_PRODUCT")
        assert val_corr_prod == get_pad_value(np.int32)

        with pytest.raises(RuntimeError, match="unexpected type"):
            val_proc_id = get_pad_value_in_tablerow_column(trows, "NUM_CORR")
        with pytest.raises(RuntimeError, match="unexpected type"):
            val_proc_id = get_pad_value_in_tablerow_column(trows, "FLAG_ROW")


def test_get_pad_value_uvw(msv4_min_correlated_xds):
    from xradio.measurement_set._utils._msv2._tables.read import get_pad_value

    res = get_pad_value(msv4_min_correlated_xds.data_vars["UVW"].dtype)

    assert np.isnan(res)
    assert np.isnan(get_pad_value(np.float64))


def test_get_pad_value_n_polynomial(pointing_xds_min):
    from xradio._utils.list_and_array import get_pad_value

    res = get_pad_value(pointing_xds_min.coords["antenna_name"].dtype)

    assert res == get_pad_value(str)


def test_get_pad_value_baseline_id(msv4_min_correlated_xds):
    from xradio._utils.list_and_array import get_pad_value

    res = get_pad_value(msv4_min_correlated_xds.coords["baseline_id"].dtype)

    assert res == get_pad_value(np.int64)


def test_redimension_ms_subtable_source(generic_source_xds_min):
    from xradio.measurement_set._utils._msv2._tables.read import redimension_ms_subtable
    import xarray as xr

    res = redimension_ms_subtable(generic_source_xds_min, "SOURCE")
    assert isinstance(res, xr.Dataset)
    src_coords = ["SOURCE_ID", "TIME", "SPECTRAL_WINDOW_ID", "PULSAR_ID"]
    assert all([coord in res.coords for coord in src_coords])


def test_is_ephem_subtable_ms(ms_minimal_required):
    from xradio.measurement_set._utils._msv2._tables.read import is_ephem_subtable

    res = is_ephem_subtable(ms_minimal_required.fname)
    assert res == False


def test_add_ephemeris_vars(ms_minimal_required):
    from xradio.measurement_set._utils._msv2._tables.read import add_ephemeris_vars
    import xarray as xr

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
        assert res == False


def test_is_nested_ms_ant(ms_minimal_required):
    from xradio.measurement_set._utils._msv2._tables.read import is_nested_ms
    from xradio.measurement_set._utils._msv2._tables.read import (
        extract_table_attributes,
    )

    ant_subt = str(Path(ms_minimal_required.fname) / "ANTENNA")
    ctds_attrs = extract_table_attributes(ms_minimal_required.fname)
    attrs = {"other": {"msv2": {"ctds_attrs": ctds_attrs}}}

    res = is_nested_ms(attrs)
    assert res == True


def test_is_nested_ms_ms_min(ms_minimal_required):
    from xradio.measurement_set._utils._msv2._tables.read import is_nested_ms
    from xradio.measurement_set._utils._msv2._tables.read import (
        extract_table_attributes,
    )

    ctds_attrs = extract_table_attributes(ms_minimal_required.fname)
    attrs = {"other": {"msv2": {"ctds_attrs": ctds_attrs}}}

    res = is_nested_ms(attrs)
    assert res == True


def test_load_generic_table_ant(ms_minimal_required):
    from xradio.measurement_set._utils._msv2._tables.read import load_generic_table
    import xarray as xr

    res = load_generic_table(ms_minimal_required.fname, "ANTENNA")
    assert res
    assert type(res) == xr.Dataset
    assert all([dim in res.dims for dim in ["row", "dim_1"]])


def test_load_generic_table_state(ms_minimal_required):
    from xradio.measurement_set._utils._msv2._tables.read import load_generic_table
    import xarray as xr

    res = load_generic_table(ms_minimal_required.fname, "STATE")
    assert res
    assert type(res) == xr.Dataset
    assert all([dim in res.dims for dim in ["row"]])
    assert all(
        [
            xvar in res.data_vars
            for xvar in ["CAL", "LOAD", "SIG", "SUB_SCAN", "OBS_MODE"]
        ]
    )


def test_load_generic_table_ephem(ms_minimal_required):
    from xradio.measurement_set._utils._msv2._tables.read import load_generic_table
    import xarray as xr

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
    assert res.attrs == exp_attrs


def test_load_generic_cols_state(ms_minimal_required):
    from xradio.measurement_set._utils._msv2._tables.read import load_generic_cols
    from xradio.measurement_set._utils._msv2._tables.table_query import open_table_ro
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
        assert all([col not in res[1] for col in ignore_cols])
        assert all(
            [var in res[1] for var in ["LOAD", "OBS_MODE", "REF", "SIG", "SUB_SCAN"]]
        )
        assert all([isinstance(val, xr.DataArray) for val in res[1].values()])


def test_load_generic_cols_spw(ms_minimal_required):
    from xradio.measurement_set._utils._msv2._tables.read import load_generic_cols
    from xradio.measurement_set._utils._msv2._tables.table_query import open_table_ro
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
        ms_minimal_required.fname, "FLAG", (10, 32, npols), [0, 1, 2], 0, 0
    )
    assert isinstance(res, np.ndarray)
    assert res.shape == (3, nchans, npols)
    assert np.all(res == False)
