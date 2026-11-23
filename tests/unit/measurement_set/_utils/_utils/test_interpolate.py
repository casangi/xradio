import pytest
import xarray as xr


def test_rename_and_interpolate_to_time_with_none_time(pointing_xds_min):
    from xradio.measurement_set._utils._utils.interpolate import (
        rename_and_interpolate_to_time,
    )

    out_xds = rename_and_interpolate_to_time(
        pointing_xds_min, "time_bogus", None, message_prefix="test_call"
    )
    xr.testing.assert_identical(out_xds, pointing_xds_min)


def test_rename_and_interpolate_to_time_with_syscal_none_time(sys_cal_xds_min):
    from xradio.measurement_set._utils._utils.interpolate import (
        rename_and_interpolate_to_time,
    )

    out_xds = rename_and_interpolate_to_time(
        sys_cal_xds_min, "time_bogus", None, message_prefix="test_call"
    )
    xr.testing.assert_identical(out_xds, sys_cal_xds_min)


def test_rename_and_interpolate_to_time_bogus(pointing_xds_min, msv4_xds_min):
    from xradio.measurement_set._utils._utils.interpolate import (
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
    from xradio.measurement_set._utils._utils.interpolate import (
        rename_and_interpolate_to_time,
    )

    input_time = msv4_xds_min.time
    out_xds = rename_and_interpolate_to_time(
        msv4_xds_min, "time", input_time, message_prefix="test_call"
    )

    xr.testing.assert_equal(out_xds.time, input_time)


def test_interpolate_to_time_bogus(antenna_xds_min, msv4_xds_min):
    from xradio.measurement_set._utils._utils.interpolate import interpolate_to_time

    input_time = msv4_xds_min.time
    out_xds = interpolate_to_time(
        antenna_xds_min, interp_time=input_time, message_prefix="test_call"
    )
    assert out_xds == input_time


def test_interpolate_to_time_main(msv4_xds_min):
    from xradio.measurement_set._utils._utils.interpolate import interpolate_to_time

    input_time = msv4_xds_min.time
    out_xds = interpolate_to_time(
        msv4_xds_min, interp_time=input_time, message_prefix="test_call"
    )

    xr.testing.assert_equal(out_xds.time, input_time)
