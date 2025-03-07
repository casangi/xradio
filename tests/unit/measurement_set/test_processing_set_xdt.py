import pytest
import xarray as xr

from xradio.measurement_set.processing_set_xdt import (
    ProcessingSetXdt,
    InvalidAccessorLocation,
)

# starting point for processing_set_xdt unit tests


def test_summary():
    ps_xdt = ProcessingSetXdt(xr.DataTree())

    with pytest.raises(InvalidAccessorLocation, match="not a processing set node"):
        summary = ps_xdt.summary()
        assert summary


def test_get_max_dims():
    ps_xdt = ProcessingSetXdt(xr.DataTree())

    with pytest.raises(InvalidAccessorLocation, match="not a processing set node"):
        dims = ps_xdt.get_max_dims()
        assert dims


def test_get_freq_axis():
    ps_xdt = ProcessingSetXdt(xr.DataTree())

    with pytest.raises(InvalidAccessorLocation, match="not a processing set node"):
        freq = ps_xdt.get_freq_axis()
        assert freq


def test_query():
    ps_xdt = ProcessingSetXdt(xr.DataTree())

    with pytest.raises(InvalidAccessorLocation, match="not a processing set node"):
        empty_query = ps_xdt.query()
        assert empty_query


def test_get_combined_field_and_source_xds():
    ps_xdt = ProcessingSetXdt(xr.DataTree())

    with pytest.raises(InvalidAccessorLocation, match="not a processing set node"):
        field_and_source_xds = ps_xdt.get_combined_field_and_source_xds()
        assert field_and_source_xds


def test_get_combined_field_and_source_xds_ephemeris():
    ps_xdt = ProcessingSetXdt(xr.DataTree())

    with pytest.raises(InvalidAccessorLocation, match="not a processing set node"):
        field_and_source_xds = ps_xdt.get_combined_field_and_source_xds_ephemeris()
        assert field_and_source_xds


def test_get_combined_antenna():
    ps_xdt = ProcessingSetXdt(xr.DataTree())

    with pytest.raises(InvalidAccessorLocation, match="not a processing set node"):
        antenna_xds = ps_xdt.get_combined_antenna_xds()
        assert antenna_xds
