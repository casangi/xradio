import pytest

import xarray as xr


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
