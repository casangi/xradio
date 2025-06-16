import pytest
import xarray as xr

from xradio.measurement_set.measurement_set_xdt import (
    MeasurementSetXdt,
    InvalidAccessorLocation,
)

# starting point for measurement_set_xdt unit tests


def test_sel():
    ms_xdt = MeasurementSetXdt(xr.DataTree())

    with pytest.raises(InvalidAccessorLocation, match="not a MSv4 node"):
        assert ms_xdt.sel()


def test_get_field_and_source_xds():
    ms_xdt = MeasurementSetXdt(xr.DataTree())

    with pytest.raises(InvalidAccessorLocation, match="not a MSv4 node"):
        assert ms_xdt.get_field_and_source_xds()


def test_get_partition_info():
    ms_xdt = MeasurementSetXdt(xr.DataTree())

    with pytest.raises(InvalidAccessorLocation, match="not a MSv4 node"):
        partition_info = ms_xdt.get_partition_info()
        assert isinstance(partition_info, dict)
        assert all(
            [
                key in partition_info
                for key in [
                    "spectral_window_name",
                    "spectral_window_intent",
                    "field_name",
                    "polarization_setup",
                    "scan_name",
                    "source_name",
                    "intents",
                    "line_name",
                ]
            ]
        )
