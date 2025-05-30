import pytest
import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path
from toolviper.utils.data import download

from xradio.measurement_set.processing_set_xdt import (
    ProcessingSetXdt,
    InvalidAccessorLocation,
)
from xradio.measurement_set import (
    load_processing_set,
    convert_msv2_to_processing_set,
    open_processing_set,
)
from xradio.schema.check import check_datatree

# Define input MS path for testing
input_ms = "Antennae_North.cal.lsrk.split.ms"
# input_ephemeris_ms = "venus_ephem_test.ms"
input_ephemeris_ms = "ALMA_uid___A002_X1003af4_X75a3.split.avg.ms"


# Fixtures for test data setup
@pytest.fixture
def test_data_path():
    """Returns path to test MeasurementSet v2"""
    # Download MS
    download(file=input_ms, folder="/tmp")
    return Path("/tmp/" + input_ms)


@pytest.fixture
def test_ephemeris_data_path():
    """Returns path to test ephemeris MeasurementSet v2"""
    # Download Ephemeris MS
    download(file=input_ephemeris_ms, folder="/tmp")
    return Path("/tmp/" + input_ephemeris_ms)


@pytest.fixture
def test_ps_path(test_data_path, tmp_path):
    """Create a processing set from test MS for testing"""
    ps_path = tmp_path / "test_processing_set.ps.zarr"

    # Convert MS to processing set
    convert_msv2_to_processing_set(
        in_file=str(test_data_path),
        out_file=str(ps_path),
        partition_scheme=[],
        main_chunksize=0.01,
        pointing_chunksize=0.00001,
        pointing_interpolate=True,
        ephemeris_interpolate=True,
        use_table_iter=False,
        overwrite=True,
        parallel_mode="none",
    )
    return ps_path


@pytest.fixture
def test_ephemeris_ps_path(test_ephemeris_data_path, tmp_path):
    """Create a processing set from ephemeris MS for testing"""
    ps_path = tmp_path / "test_ephemeris_processing_set.ps.zarr"

    # Convert Ephemeris MS to processing set
    convert_msv2_to_processing_set(
        in_file=str(test_ephemeris_data_path),
        out_file=str(ps_path),
        partition_scheme=[],
        main_chunksize=0.01,
        pointing_chunksize=0.00001,
        pointing_interpolate=True,
        ephemeris_interpolate=True,  # Important for ephemeris data
        use_table_iter=False,
        overwrite=True,
        parallel_mode="none",
    )
    return ps_path


# Tests with empty DataTree (testing error handling)
class TestProcessingSetXdtErrors:
    """Tests for ProcessingSetXdt error handling with empty DataTree"""

    def test_summary_empty(self):
        """Test that summary raises an exception on empty DataTree"""
        ps_xdt = ProcessingSetXdt(xr.DataTree())

        with pytest.raises(InvalidAccessorLocation, match="not a processing set node"):
            summary = ps_xdt.summary()
            assert summary

    def test_get_max_dims_empty(self):
        """Test that get_max_dims raises an exception on empty DataTree"""
        ps_xdt = ProcessingSetXdt(xr.DataTree())

        with pytest.raises(InvalidAccessorLocation, match="not a processing set node"):
            dims = ps_xdt.get_max_dims()
            assert dims

    def test_get_freq_axis_empty(self):
        """Test that get_freq_axis raises an exception on empty DataTree"""
        ps_xdt = ProcessingSetXdt(xr.DataTree())

        with pytest.raises(InvalidAccessorLocation, match="not a processing set node"):
            freq = ps_xdt.get_freq_axis()
            assert freq

    def test_query_empty(self):
        """Test that query raises an exception on empty DataTree"""
        ps_xdt = ProcessingSetXdt(xr.DataTree())

        with pytest.raises(InvalidAccessorLocation, match="not a processing set node"):
            empty_query = ps_xdt.query()
            assert empty_query

    def test_get_combined_antenna(self):
        """Test that get_combined_antenna_xds raises an exception on empty DataTree"""
        ps_xdt = ProcessingSetXdt(xr.DataTree())

        with pytest.raises(InvalidAccessorLocation, match="not a processing set node"):
            antenna_xds = ps_xdt.get_combined_antenna_xds()
            assert antenna_xds

    def test_get_combined_field_and_source_xds_empty(self):
        """Test that get_combined_field_and_source_xds raises an exception on empty DataTree"""
        ps_xdt = ProcessingSetXdt(xr.DataTree())

        with pytest.raises(InvalidAccessorLocation, match="not a processing set node"):
            field_and_source_xds = ps_xdt.get_combined_field_and_source_xds()
            assert field_and_source_xds

    def test_get_combined_field_and_source_xds_ephemeris_empty(self):
        """Test that get_combined_field_and_source_xds with ephemeris raises an exception on empty DataTree"""
        ps_xdt = ProcessingSetXdt(xr.DataTree())

        with pytest.raises(InvalidAccessorLocation, match="not a processing set node"):
            field_and_source_xds = ps_xdt.get_combined_field_and_source_xds_ephemeris()
            assert field_and_source_xds


# Tests with actual data loaded from disk
class TestProcessingSetXdtWithData:
    """Tests for ProcessingSetXdt using real data loaded from disk"""

    def test_summary(self, test_ps_path):
        """Test the summary method on a real processing set"""
        ps_xdt = load_processing_set(str(test_ps_path))

        # Get summary
        summary = ps_xdt.xr_ps.summary()
        print(summary)

        # Verify it returns a pandas DataFrame
        assert isinstance(summary, pd.DataFrame)

        # Verify the DataFrame is not empty
        assert not summary.empty

        # Verify expected columns are present
        expected_columns = [
            "name",
            "intents",
            "shape",
            "polarization",
            "scan_name",
            "spw_name",
        ]
        for col in expected_columns:
            assert col in summary.columns

    def test_get_max_dims(self, test_ps_path):
        """Test getting maximum dimensions from a processing set"""
        ps_xdt = load_processing_set(str(test_ps_path))

        # Get max dimensions
        max_dims = ps_xdt.xr_ps.get_max_dims()

        # Verify it returns a dictionary with dimension names as keys
        assert isinstance(max_dims, dict)
        assert "time" in max_dims
        assert "frequency" in max_dims
        assert isinstance(max_dims["time"], int)
        assert isinstance(max_dims["frequency"], int)
        assert max_dims["time"] == 50
        assert max_dims["frequency"] == 8
        assert max_dims["baseline_id"] == 77
        assert max_dims["polarization"] == 2

    def test_get_freq_axis(self, test_ps_path):
        """Test getting frequency axis from a processing set"""
        ps_xdt = load_processing_set(str(test_ps_path))

        # Get frequency axis
        freq_axis = ps_xdt.xr_ps.get_freq_axis()

        # Verify it returns an xarray DataArray
        assert isinstance(freq_axis, xr.DataArray)

        # Check the DataArray properties
        assert "frequency" in freq_axis.dims  # Should have a frequency dimension
        assert freq_axis.size > 0  # Should have values
        assert np.all(freq_axis.values > 0)  # Frequencies should be positive

    def test_query(self, test_ps_path):
        """Test querying a processing set"""
        ps_xdt = load_processing_set(str(test_ps_path))

        # Get list of MS names
        ms_names = list(ps_xdt.children.keys())
        assert len(ms_names) > 0

        # Query for specific MS
        result = ps_xdt.xr_ps.query(name=ms_names[0])

        # Verify the result is a DataTree containing only the requested MS
        assert isinstance(result, xr.DataTree)
        assert len(result.children) == 1
        assert ms_names[0] in result.children

        # Test querying with data_group_name
        result_with_dg = ps_xdt.xr_ps.query(data_group_name="base")

        # Verify the result includes the data group
        assert isinstance(result_with_dg, xr.DataTree)
        for ms_xdt in result_with_dg.children.values():
            assert "base" in ms_xdt.attrs.get("data_groups", {})

    def test_get_combined_field_and_source_xds(self, test_ps_path):
        """Test getting combined field and source dataset from a processing set"""
        ps_xdt = load_processing_set(str(test_ps_path))

        # Get combined field and source dataset
        combined_field_source_xds = ps_xdt.xr_ps.get_combined_field_and_source_xds()

        # Verify it returns an xarray Dataset
        assert isinstance(combined_field_source_xds, xr.Dataset)

        # Check required fields are present
        assert "line_name" not in combined_field_source_xds.coords
        assert "field_name" in combined_field_source_xds.coords
        assert "time" in combined_field_source_xds.dims
        assert "FIELD_PHASE_CENTER" in combined_field_source_xds.data_vars
        assert "SOURCE_LOCATION" in combined_field_source_xds.data_vars


# Tests with ephemeris data loaded from disk
class TestProcessingSetXdtWithEphemerisData:
    """Tests for ProcessingSetXdt using real ephemeris data loaded from disk"""

    def test_check_ephemeris_datatree(self, test_ephemeris_ps_path):
        """Test that the converted MS to PS complies with the datatree schema checker"""
        ps_xdt = load_processing_set(str(test_ephemeris_ps_path))

        issues = check_datatree(ps_xdt)
        # The check_datatree function returns a SchemaIssues object, not a string
        assert (
            str(issues) == "No schema issues found"
        ), f"Schema validation failed: {issues}"

    def test_get_combined_field_and_source_xds_ephemeris(self, test_ephemeris_ps_path):
        """Test getting combined field and source dataset with ephemeris from a processing set"""
        ps_xdt = load_processing_set(str(test_ephemeris_ps_path))

        # Get combined field and source dataset with ephemeris
        combined_ephemeris_field_source_xds = (
            ps_xdt.xr_ps.get_combined_field_and_source_xds_ephemeris()
        )

        # Verify it returns an xarray Dataset
        assert isinstance(combined_ephemeris_field_source_xds, xr.Dataset)

        # Check required fields are present
        assert (
            combined_ephemeris_field_source_xds.attrs["type"]
            == "field_and_source_ephemeris"
        )
        assert "time" in combined_ephemeris_field_source_xds.dims
        assert "field_name" in combined_ephemeris_field_source_xds.coords
        assert "time" in combined_ephemeris_field_source_xds.coords
        print(combined_ephemeris_field_source_xds.coords)

        # Check ephemeris-specific fields
        assert "SOURCE_LOCATION" in combined_ephemeris_field_source_xds.data_vars
        assert "FIELD_PHASE_CENTER" in combined_ephemeris_field_source_xds.data_vars
        assert "FIELD_OFFSET" in combined_ephemeris_field_source_xds.data_vars
        assert "SOURCE_RADIAL_VELOCITY" in combined_ephemeris_field_source_xds.data_vars

        # Check center field calculation
        assert "center_field_name" in combined_ephemeris_field_source_xds.attrs

        # Center field name could be either a string or a numpy array containing a string
        center_field = combined_ephemeris_field_source_xds.attrs["center_field_name"]
        if isinstance(center_field, np.ndarray):
            # If it's a numpy array, check that it contains a string value
            assert center_field.dtype.kind in ["U", "S"]  # Unicode or byte string
            assert center_field.size == 1  # Should be a single value
        else:
            # Otherwise it should be a regular string
            assert isinstance(center_field, (str, np.str_))

    def test_field_offset_calculation(self, test_ephemeris_ps_path):
        """Test that field offsets are correctly calculated"""
        ps_xdt = load_processing_set(str(test_ephemeris_ps_path))

        field_source_xds = ps_xdt.xr_ps.get_combined_field_and_source_xds_ephemeris()

        # Verify field offset calculation
        field_offset = field_source_xds["FIELD_OFFSET"]

        # The field offset should only include ra and dec components
        assert "ra" in field_offset.sky_dir_label.values
        assert "dec" in field_offset.sky_dir_label.values

        # Check that offsets have reasonable values (should be in radians)
        assert np.all(np.abs(field_offset) < np.pi)  # Should be wrapped to [-π, π]

    def test_time_interpolation(self, test_ephemeris_ps_path):
        """Test that time interpolation works correctly for ephemeris data"""
        ps_xdt = load_processing_set(str(test_ephemeris_ps_path))

        field_source_xds = ps_xdt.xr_ps.get_combined_field_and_source_xds_ephemeris()

        # Check that time is a dimension
        assert "time" in field_source_xds.dims

        # Time should have multiple points for ephemeris data
        assert field_source_xds.sizes["time"] > 1


if __name__ == "__main__":
    pytest.main(["-v", "-s", __file__])
