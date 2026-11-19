import shutil

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


# SYSCAL of ms_minimal_required has 10 channels
SYSCAL_CHANNELS = {
    "in_order": 1.0e9 + np.arange(10),
    "reversed": 1.0e9 - np.arange(10),
    "reordered_by_casa": 1.0e9 + np.array([5, 6, 7, 8, 9, 0, 1, 2, 3, 4]),
    "selected_or_averaged_by_casa": 1.0e9 + np.arange(16),
}


@pytest.mark.parametrize("window", list(SYSCAL_CHANNELS))
def test_create_system_calibration_xds_channel_order(
    ms_minimal_required, msv4_xdt_min, tmp_path, window
):
    from xradio._utils._casacore.backend import tables
    from xradio.measurement_set._utils._msv2.msv4_sub_xdss import (
        create_system_calibration_xds,
    )

    msname = str(tmp_path / "syscal.ms")
    shutil.copytree(ms_minimal_required.fname, msname)
    with tables.table(f"{msname}/SYSCAL", readonly=False, ack=False) as syscal:
        shape = syscal.getcol("TCAL_SPECTRUM").shape
        syscal.putcol(
            "TCAL_SPECTRUM",
            np.broadcast_to(np.arange(10.0)[:, np.newaxis], shape).copy(),
        )

    ant_xds = msv4_xdt_min["antenna_xds"].ds.copy()
    nants = len(ant_xds.antenna_name)
    ant_xds_with_ids = ant_xds.assign_coords(
        {"antenna_id": ("antenna_name", np.arange(0, nants))}
    )
    frequencies = SYSCAL_CHANNELS[window]
    frequency = xr.DataArray(
        frequencies, dims="frequency", attrs=msv4_xdt_min.frequency.attrs
    )

    sys_cal_xds = create_system_calibration_xds(
        msname, 0, frequency, ant_xds_with_ids, None
    )

    assert not check_dataset(sys_cal_xds, SystemCalibrationXds).issues
    channels = sys_cal_xds.TCAL.isel(
        antenna_name=0, time_system_cal=0, receptor_label=0
    ).values
    if window in ("in_order", "reversed"):
        coord = sys_cal_xds.frequency_system_cal
        np.testing.assert_array_equal(coord.values, np.sort(frequencies))
        assert coord.attrs["units"] == "Hz"
        np.testing.assert_array_equal(channels, np.argsort(frequencies))
    else:
        assert "frequency_system_cal" not in sys_cal_xds.coords
        np.testing.assert_array_equal(channels, np.arange(10))


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
