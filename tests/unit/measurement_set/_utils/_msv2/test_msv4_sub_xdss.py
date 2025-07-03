import numpy as np
import pytest
import xarray as xr

from xradio.measurement_set.schema import PointingXds, SystemCalibrationXds, WeatherXds
from xradio.schema.check import check_dataset


def test_rename_and_interpolate_to_time_with_none_time(pointing_xds_min):
    from xradio.measurement_set._utils._msv2.msv4_sub_xdss import (
        rename_and_interpolate_to_time,
    )

    out_xds = rename_and_interpolate_to_time(
        pointing_xds_min, "time_bogus", None, message_prefix="test_call"
    )
    assert out_xds == pointing_xds_min


def test_rename_and_interpolate_to_time_with_syscal_none_time(sys_cal_xds_min):
    from xradio.measurement_set._utils._msv2.msv4_sub_xdss import (
        rename_and_interpolate_to_time,
    )

    out_xds = rename_and_interpolate_to_time(
        sys_cal_xds_min, "time_bogus", None, message_prefix="test_call"
    )
    assert out_xds == sys_cal_xds_min


def test_rename_and_interpolate_to_time_bogus(pointing_xds_min, msv4_xds_min):
    from xradio.measurement_set._utils._msv2.msv4_sub_xdss import (
        rename_and_interpolate_to_time,
    )

    input_time = msv4_xds_min.time
    time_bogus_name = "time_bogus"
    with pytest.raises(KeyError, match=f"Could not find node at {time_bogus_name}"):
        out_xds = rename_and_interpolate_to_time(
            pointing_xds_min, time_bogus_name, input_time, message_prefix="test_call"
        )
        assert out_xds


def test_rename_and_interpoalte_to_time_main(msv4_xds_min):
    from xradio.measurement_set._utils._msv2.msv4_sub_xdss import (
        rename_and_interpolate_to_time,
    )

    input_time = msv4_xds_min.time
    out_xds = rename_and_interpolate_to_time(
        msv4_xds_min, "time", input_time, message_prefix="test_call"
    )

    xr.testing.assert_equal(out_xds.time, input_time)


def test_interpolate_to_time_bogus(antenna_xds_min, msv4_xds_min):
    from xradio.measurement_set._utils._msv2.msv4_sub_xdss import interpolate_to_time

    input_time = msv4_xds_min.time
    out_xds = interpolate_to_time(
        antenna_xds_min, interp_time=input_time, message_prefix="test_call"
    )
    assert out_xds == input_time


def test_interpolate_to_time_main(msv4_xds_min):
    from xradio.measurement_set._utils._msv2.msv4_sub_xdss import interpolate_to_time

    input_time = msv4_xds_min.time
    out_xds = interpolate_to_time(
        msv4_xds_min, interp_time=input_time, message_prefix="test_call"
    )

    xr.testing.assert_equal(out_xds.time, input_time)


@pytest.fixture(scope="session")
def ant_xds_station_name_ids():
    """Makes a ant_xds_station_name_ids as the one used in conversion (it still has antenna_id coord)"""
    attrs = {
        "type": "location",
        "units": ["m", "m", "m"],
        "frame": "ITRS",
        "coordinate_system": "geocentric",
        "origin_object_name": "earth",
    }
    nants = 5
    ant_ids_xds = xr.DataArray(
        np.broadcast_to([0.0, 0.0, 0.0], (nants, 3)),
        dims=["antenna_name", "cartesian_pos_label"],
        coords={
            "antenna_id": ("antenna_name", np.arange(0, nants)),
            "antenna_name": (
                "antenna_name",
                [f"test_ant{idx}" for idx in np.arange(0, nants)],
            ),
            "cartesian_pos_label": (
                "cartesian_pos_label",
                ["x", "y", "z"],
            ),
            "station_name": (
                "antenna_name",
                [f"test_station{idx}" for idx in np.arange(0, nants)],
            ),
        },
        attrs=attrs,
        name="ANTENNA_POSITION",
    )
    ant_ids_xds = ant_ids_xds.set_xindex("antenna_id")

    yield ant_ids_xds


def test_create_weather_xds_empty_ant_ids(ms_empty_required):
    from xradio.measurement_set._utils._msv2.msv4_sub_xdss import create_weather_xds

    with pytest.raises(AttributeError, match="has no attribute"):
        weather_xds = create_weather_xds(ms_empty_required.fname, xr.DataSet())
        assert weather_xds


def test_create_weather_xds_empty(ms_empty_complete, ant_xds_station_name_ids):
    from xradio.measurement_set._utils._msv2.msv4_sub_xdss import create_weather_xds

    weather_xds = create_weather_xds(ms_empty_complete.fname, ant_xds_station_name_ids)
    assert not weather_xds


def test_create_weather_xds_min(ms_minimal_required, ant_xds_station_name_ids):
    from xradio.measurement_set._utils._msv2.msv4_sub_xdss import create_weather_xds

    weather_xds = create_weather_xds(
        ms_minimal_required.fname, ant_xds_station_name_ids
    )
    check_dataset(weather_xds, WeatherXds)


def test_create_weather_xds_misbehaved(ms_minimal_misbehaved, ant_xds_station_name_ids):
    from xradio.measurement_set._utils._msv2.msv4_sub_xdss import create_weather_xds

    weather_xds = create_weather_xds(
        ms_minimal_misbehaved.fname, ant_xds_station_name_ids
    )
    check_dataset(weather_xds, WeatherXds)


def test_create_weather_xds_ms_without_opt_subtables(
    ms_minimal_without_opt, ant_xds_station_name_ids
):
    from xradio.measurement_set._utils._msv2.msv4_sub_xdss import create_weather_xds

    weather_xds = create_weather_xds(
        ms_minimal_without_opt.fname, ant_xds_station_name_ids
    )
    assert weather_xds is None


def test_create_pointing_xds_empty_ant_ids(ms_empty_required):
    from xradio.measurement_set._utils._msv2.msv4_sub_xdss import create_pointing_xds

    with pytest.raises(AttributeError, match="has no attribute"):
        pointing_xds = create_pointing_xds(ms_empty_required.fname, xr.DataSet())
        assert pointing_xds


def test_create_pointing_xds_empty(ms_empty_required):
    from xradio.measurement_set._utils._msv2.msv4_sub_xdss import create_pointing_xds

    ant_ids_da = xr.DataArray(
        [f"test_ant{idx}" for idx in np.arange(0, 5)],
        name="antenna_name",
        coords={"antenna_id": np.arange(0, 5)},
    )
    pointing_xds = create_pointing_xds(
        ms_empty_required.fname, ant_ids_da, (0, 2e10), None
    )
    check_dataset(pointing_xds, PointingXds)


def test_create_pointing_xds_empty_time_interp(ms_empty_required):
    from xradio.measurement_set._utils._msv2.msv4_sub_xdss import create_pointing_xds

    ant_ids_da = xr.DataArray(
        [f"test_ant{idx}" for idx in np.arange(0, 5)],
        name="antenna_name",
        coords={"antenna_id": np.arange(0, 5)},
    )
    time_interp = np.arange(0, 100)
    pointing_xds = create_pointing_xds(
        ms_empty_required.fname, ant_ids_da, (0, 2e10), time_interp
    )
    check_dataset(pointing_xds, PointingXds)


def test_create_pointing_xds_min_time_interp(ms_minimal_required):
    from xradio.measurement_set._utils._msv2.msv4_sub_xdss import create_pointing_xds

    ant_ids_da = xr.DataArray(
        [f"test_ant{idx}" for idx in np.arange(0, 5)],
        name="antenna_name",
        coords={"antenna_id": np.arange(0, 5)},
    )
    time_interp = np.arange(0, 100)
    pointing_xds = create_pointing_xds(
        ms_minimal_required.fname, ant_ids_da, (0, 2e10), time_interp
    )
    check_dataset(pointing_xds, PointingXds)


def test_create_pointing_xds_without_opt(ms_minimal_without_opt):
    from xradio.measurement_set._utils._msv2.msv4_sub_xdss import create_pointing_xds

    ant_ids_da = xr.DataArray(
        [f"test_ant{idx}" for idx in np.arange(0, 5)],
        name="antenna_name",
        coords={"antenna_id": np.arange(0, 5)},
    )
    time_interp = np.arange(0, 100)
    pointing_xds = create_pointing_xds(
        ms_minimal_without_opt.fname, ant_ids_da, (0, 2e10), time_interp
    )
    check_dataset(pointing_xds, PointingXds)


def test_create_system_calibration_xds_empty(ms_empty_required, msv4_xdt_min):
    from xradio.measurement_set._utils._msv2.msv4_sub_xdss import (
        create_system_calibration_xds,
    )

    # Need to make the antenna_id coord as in the middle of conversion, before it is removed
    ant_xds = msv4_xdt_min["antenna_xds"].ds.copy()
    nants = len(ant_xds.antenna_name)
    ant_xds_with_ids = ant_xds.assign_coords(
        {"antenna_id": ("antenna_name", np.arange(0, nants))}
    )

    sys_cal_xds = create_system_calibration_xds(
        ms_empty_required.fname, 0, msv4_xdt_min.frequency, ant_xds_with_ids, None
    )
    assert sys_cal_xds is None


def test_create_system_calibration_xds_min(ms_minimal_required, msv4_xdt_min):
    from xradio.measurement_set._utils._msv2.msv4_sub_xdss import (
        create_system_calibration_xds,
    )

    # Need to make the antenna_id coord as in the middle of conversion, before it is removed
    ant_xds = msv4_xdt_min["antenna_xds"].ds.copy()
    nants = len(ant_xds.antenna_name)
    ant_xds_with_ids = ant_xds.assign_coords(
        {"antenna_id": ("antenna_name", np.arange(0, nants))}
    )

    sys_cal_xds = create_system_calibration_xds(
        ms_minimal_required.fname, 0, msv4_xdt_min.frequency, ant_xds_with_ids, None
    )
    check_dataset(sys_cal_xds, SystemCalibrationXds)


def test_create_system_calibration_xds_without_opt(
    ms_minimal_without_opt, msv4_xdt_min
):
    from xradio.measurement_set._utils._msv2.msv4_sub_xdss import (
        create_system_calibration_xds,
    )

    # Need to make the antenna_id coord as in the middle of conversion, before it is removed
    ant_xds = msv4_xdt_min["antenna_xds"].ds.copy()
    nants = len(ant_xds.antenna_name)
    ant_xds_with_ids = ant_xds.assign_coords(
        {"antenna_id": ("antenna_name", np.arange(0, nants))}
    )

    sys_cal_xds = create_system_calibration_xds(
        ms_minimal_without_opt.fname, 0, msv4_xdt_min.frequency, ant_xds_with_ids, None
    )
    assert sys_cal_xds is None
