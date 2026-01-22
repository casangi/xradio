import pytest
import xarray as xr

from xradio.measurement_set.measurement_set_xdt import (
    MeasurementSetXdt,
    InvalidAccessorLocation,
)
from xradio.measurement_set.schema import FieldSourceXds, VisibilityXds
from xradio.schema.check import check_dataset, check_datatree

# starting point for measurement_set_xdt unit tests. Some additional test cases added for clear
# coverage gaps, but several more still missing for better systematic coverage w/o relying on stk tests


def test_sel_invalid():
    ms_xdt = MeasurementSetXdt(xr.DataTree())

    with pytest.raises(InvalidAccessorLocation, match="not a MSv4 node"):
        assert ms_xdt.sel()


def test_get_field_and_source_xds_invalid():
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
                    "spectral_window_intents",
                    "field_name",
                    "polarization_setup",
                    "scan_name",
                    "source_name",
                    "scan_intents",
                    "line_name",
                ]
            ]
        )


def test_add_data_group_with_defaults(msv4_xdt_min):

    ms_xdt = msv4_xdt_min.xr_ms
    result_xdt = ms_xdt.add_data_group("test_added_data_group_with_defaults")

    check_dataset(result_xdt.ds, VisibilityXds)
    check_datatree(result_xdt)


def test_add_data_group_with_values(msv4_xdt_min):
    ms_xdt = msv4_xdt_min.xr_ms
    result_xdt = ms_xdt.add_data_group(
        "test_added_data_group_with_param_values",
        {
            "correlated_data": "VISIBILITY",
            "weight": "EFFECTIVE_INTEGRATION_TIME",  # no check for this (coords mismatch, etc.)
            "flag": "FLAG",
            "uvw": "UVW",
            "field_and_source_xds": "field_and_source_base_xds",
            "date_time": "today, now",
            "description": "a test data group",
        },
        data_group_dv_shared_with="base",
    )

    check_dataset(result_xdt.ds, VisibilityXds)
    check_datatree(result_xdt)


def test_get_field_and_source_xds(msv4_xdt_min):

    result_xdt = msv4_xdt_min.xr_ms.get_field_and_source_xds()
    check_dataset(result_xdt, FieldSourceXds)


def test_get_field_and_source_xds_with_group(msv4_xdt_min):

    result_xdt = msv4_xdt_min.xr_ms.get_field_and_source_xds(data_group_name="base")
    check_dataset(result_xdt, FieldSourceXds)


expected_partition_info_fields = [
    "spectral_window_name",
    "field_name",
    "polarization_setup",
    "scan_name",
    "source_name",
    "scan_intents",
    "line_name",
    "data_group_name",
]


def test_get_partition_info_default(msv4_xdt_min):
    partition_info = msv4_xdt_min.xr_ms.get_partition_info()
    assert isinstance(partition_info, dict)
    for field in expected_partition_info_fields:
        assert field in partition_info


def test_get_partition_info_with_group_wrong(msv4_xdt_min):

    wrong_group_name = "missing"
    with pytest.raises(KeyError, match=wrong_group_name):
        _partition_info = msv4_xdt_min.xr_ms.get_partition_info(
            data_group_name=wrong_group_name
        )


def test_get_partition_info_with_group(msv4_xdt_min):

    partition_info = msv4_xdt_min.xr_ms.get_partition_info(data_group_name="base")
    assert isinstance(partition_info, dict)
    for field in expected_partition_info_fields:
        assert field in partition_info


def test_sel_with_data_group_missing(msv4_xdt_min):
    wrong_group_name = "corrected"
    with pytest.raises(KeyError, match=wrong_group_name):
        _result_xds = msv4_xdt_min.xr_ms.sel(data_group_name=wrong_group_name)


def test_sel_with_data_group(msv4_xdt_min):
    result_xdt = msv4_xdt_min.xr_ms.sel(data_group_name="base")
    check_dataset(result_xdt.ds, VisibilityXds)


def test_sel_polarization(msv4_xdt_min):
    result_xdt = msv4_xdt_min.xr_ms.sel(polarization="XX")
    check_dataset(result_xdt.ds, VisibilityXds)


if __name__ == "__main__":
    pytest.main(["-v", "-s", __file__])
