import numpy as np
import pytest
import xarray as xr

from xradio.measurement_set.schema import PointingXds, WeatherXds
from xradio.schema.check import check_dataset


def test_rename_and_interpolate_to_time_with_none_time(pointing_xds_min):
    from xradio.measurement_set._utils._msv2.msv4_sub_xdss import (
        rename_and_interpolate_to_time,
    )

    out_xds = rename_and_interpolate_to_time(
        pointing_xds_min, "time_bogus", None, message_prefix="test_call"
    )
    assert out_xds == pointing_xds_min


def test_rename_and_interpolate_to_time_bogus(
    pointing_xds_min, msv4_min_correlated_xds
):
    from xradio.measurement_set._utils._msv2.msv4_sub_xdss import (
        rename_and_interpolate_to_time,
    )

    input_time = msv4_min_correlated_xds.time
    with pytest.raises(KeyError, match="No variable named 'time_bogus'."):
        out_xds = rename_and_interpolate_to_time(
            pointing_xds_min, "time_bogus", input_time, message_prefix="test_call"
        )


def test_rename_and_interpoalte_to_time_main(msv4_min_correlated_xds):
    from xradio.measurement_set._utils._msv2.msv4_sub_xdss import (
        rename_and_interpolate_to_time,
    )

    input_time = msv4_min_correlated_xds.time
    out_xds = rename_and_interpolate_to_time(
        msv4_min_correlated_xds, "time", input_time, message_prefix="test_call"
    )

    xr.testing.assert_equal(out_xds.time, input_time)


def test_interpolate_to_time_bogus(antenna_xds_min, msv4_min_correlated_xds):
    from xradio.measurement_set._utils._msv2.msv4_sub_xdss import interpolate_to_time

    input_time = msv4_min_correlated_xds.time
    with pytest.raises(KeyError, match="No variable named 'time'."):
        out_xds = interpolate_to_time(
            antenna_xds_min, interp_time=input_time, message_prefix="test_call"
        )


def test_interpolate_to_time_main(msv4_min_correlated_xds):
    from xradio.measurement_set._utils._msv2.msv4_sub_xdss import interpolate_to_time

    input_time = msv4_min_correlated_xds.time
    out_xds = interpolate_to_time(
        msv4_min_correlated_xds, interp_time=input_time, message_prefix="test_call"
    )

    xr.testing.assert_equal(out_xds.time, input_time)


def test_create_weather_xds_empty_ant_ids(ms_empty_required):
    from xradio.measurement_set._utils._msv2.msv4_sub_xdss import create_weather_xds

    with pytest.raises(AttributeError, match="has no attribute"):
        weather_xds = create_weather_xds(ms_empty_required.fname, xr.DataSet())


def test_create_weather_xds_empty(ms_empty_required):
    from xradio.measurement_set._utils._msv2.msv4_sub_xdss import create_weather_xds

    ant_ids_xds = xr.DataArray(
        ["test_station0_ant0", "test_station0_ant1"],
        dims=["antenna_id"],
        coords={
            "antenna_id": [0, 1],
            "antenna_name": ("antenna_id", ["test_ant0", "test_ant1"]),
        },
        name="ant_xds_station_name_ids",
    )
    weather_xds = create_weather_xds(ms_empty_required.fname, ant_ids_xds)
    assert not weather_xds


def test_create_weather_xds_min(ms_minimal_required):
    from xradio.measurement_set._utils._msv2.msv4_sub_xdss import create_weather_xds

    ant_ids_xds = xr.DataArray(
        ["test_station0_ant0", "test_station0_ant1"],
        dims=["antenna_id"],
        coords={
            "antenna_id": [0, 1],
            "antenna_name": ("antenna_id", ["test_ant0", "test_ant1"]),
        },
        name="ant_xds_station_name_ids",
    )
    weather_xds = create_weather_xds(ms_minimal_required.fname, ant_ids_xds)
    check_dataset(weather_xds, WeatherXds)


def test_create_pointing_xds_empty_ant_ids(ms_empty_required):
    from xradio.measurement_set._utils._msv2.msv4_sub_xdss import create_pointing_xds

    with pytest.raises(AttributeError, match="has no attribute"):
        pointing_xds = create_pointing_xds(ms_empty_required.fname, xr.DataSet())


def test_create_pointing_xds_empty(ms_empty_required):
    from xradio.measurement_set._utils._msv2.msv4_sub_xdss import create_pointing_xds

    ant_ids_xds = xr.Dataset(
        data_vars={"antenna_name": ("antenna_id", ["test_ant"])},
        coords={"antenna_id": [0]},
    )
    pointing_xds = create_pointing_xds(
        ms_empty_required.fname, ant_ids_xds, (0, 2e10), None
    )
    check_dataset(pointing_xds, PointingXds)


def test_create_pointing_xds_empty_time_interp(ms_empty_required):
    from xradio.measurement_set._utils._msv2.msv4_sub_xdss import create_pointing_xds

    ant_ids_xds = xr.Dataset(
        data_vars={"antenna_name": ("antenna_id", ["test_ant"])},
        coords={"antenna_id": [0]},
    )
    time_interp = np.arange(0, 100)
    pointing_xds = create_pointing_xds(
        ms_empty_required.fname, ant_ids_xds, (0, 2e10), time_interp
    )
    check_dataset(pointing_xds, PointingXds)


# More TODO: several create_xxx_xds being defined / developed
