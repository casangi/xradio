import xarray as xr
import pytest


def test_rename_and_interpolate_to_time_with_none_time(ddi_xds_min, main_xds_min):
    from xradio.measurement_set._utils._msv2.msv4_sub_xdss import (
        rename_and_interpolate_to_time,
    )

    out_xds = rename_and_interpolate_to_time(
        ddi_xds_min, "time_bogus", None, message_prefix="test_call"
    )
    assert out_xds == ddi_xds_min


def test_rename_and_interpolate_to_time_bogus(ddi_xds_min, main_xds_min):
    from xradio.measurement_set._utils._msv2.msv4_sub_xdss import (
        rename_and_interpolate_to_time,
    )

    input_time = main_xds_min.time
    with pytest.raises(KeyError, match="No variable named 'time_bogus'."):
        out_xds = rename_and_interpolate_to_time(
            ddi_xds_min, "time_bogus", input_time, message_prefix="test_call"
        )


def test_rename_and_interpoalte_to_time_main(main_xds_min):
    from xradio.measurement_set._utils._msv2.msv4_sub_xdss import (
        rename_and_interpolate_to_time,
    )

    input_time = main_xds_min.time
    out_xds = rename_and_interpolate_to_time(
        main_xds_min, "time", input_time, message_prefix="test_call"
    )

    xr.testing.assert_equal(out_xds.time, input_time)


def test_interpolate_to_time_bogus(ddi_xds_min, main_xds_min):
    from xradio.measurement_set._utils._msv2.msv4_sub_xdss import interpolate_to_time

    input_time = main_xds_min.time
    with pytest.raises(KeyError, match="No variable named 'time'."):
        out_xds = interpolate_to_time(
            ddi_xds_min, interp_time=input_time, message_prefix="test_call"
        )


def test_interpolate_to_time_main(main_xds_min):
    from xradio.measurement_set._utils._msv2.msv4_sub_xdss import interpolate_to_time

    input_time = main_xds_min.time
    out_xds = interpolate_to_time(
        main_xds_min, interp_time=input_time, message_prefix="test_call"
    )

    xr.testing.assert_equal(out_xds.time, input_time)


# TODO: several create_xxx_xds being defined / developed
