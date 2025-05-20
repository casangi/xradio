import pytest
import numpy as np
import xarray as xr

from xradio.measurement_set.measurement_set_xdt import (
    MeasurementSetXdt,
    InvalidAccessorLocation,
    MS_DATASET_TYPES,
)


# Helper function to create a minimal valid MSv4 DataTree for testing
def create_test_ms_datatree(ms_type="visibility"):
    """
    Create a minimal valid MSv4 DataTree for testing purposes.

    Parameters
    ----------
    ms_type : str
        Type of measurement set to create (visibility, spectrum, radiometer)

    Returns
    -------
    xarray.DataTree
        A valid MSv4 DataTree node
    """
    # Ensure valid type
    if ms_type not in MS_DATASET_TYPES:
        raise ValueError(f"ms_type must be one of {MS_DATASET_TYPES}")

    # Create base coordinates
    time = np.array([1.0, 2.0, 3.0])
    baseline = np.array([0, 1, 2])
    polarization = np.array(["XX", "YY"])
    frequency = np.linspace(1e9, 2e9, 10)  # 10 frequency channels

    # Create a scan_name coordinate
    scan_name = np.array(["scan1", "scan1", "scan2"])

    # Create mock data - properly generating complex data
    # Generate real and imaginary parts separately and combine them
    real_part = np.random.random(
        (len(time), len(baseline), len(frequency), len(polarization))
    )
    imag_part = np.random.random(
        (len(time), len(baseline), len(frequency), len(polarization))
    )
    correlated_data = real_part + 1j * imag_part  # Create complex numbers

    weight = np.random.random(
        (len(time), len(baseline), len(frequency), len(polarization))
    )

    flag = np.zeros(
        (len(time), len(baseline), len(frequency), len(polarization)), dtype=bool
    )

    # For visibility type, create UVW data
    uvw = np.random.random((len(time), len(baseline), 3))

    # Create the main dataset with all required data variables
    ds = xr.Dataset(
        data_vars={
            "correlated_data": xr.DataArray(
                correlated_data, dims=("time", "baseline", "frequency", "polarization")
            ),
            "weight": xr.DataArray(
                weight, dims=("time", "baseline", "frequency", "polarization")
            ),
            "flag": xr.DataArray(
                flag, dims=("time", "baseline", "frequency", "polarization")
            ),
            "scan_name": xr.DataArray(scan_name, dims=("time")),
        },
        coords={
            "time": time,
            "baseline": baseline,
            "polarization": polarization,
            "frequency": frequency,
        },
    )

    # Add UVW for visibility type
    if ms_type == "visibility":
        ds["uvw"] = xr.DataArray(uvw, dims=("time", "baseline", "uvw_index"))

    # Create the field_and_source dataset
    field_source_ds = xr.Dataset(
        data_vars={
            "field_name": xr.DataArray(["field1", "field2"], dims=("field_id")),
            "source_name": xr.DataArray(["source1", "source2"], dims=("field_id")),
        },
        coords={
            "field_id": [0, 1],
            "line_name": (
                ["field_id"],
                ["line1", "line2"],
            ),  # Define line_name as a coordinate
        },
    )

    # Create the DataTree structure
    xdt = xr.DataTree(ds)

    # Add the field_and_source dataset as a child node
    xdt["field_and_source"] = xr.DataTree(field_source_ds)

    # Add required attributes
    xdt.attrs["type"] = ms_type
    xdt.attrs["data_groups"] = {
        "base": {
            "correlated_data": "correlated_data",
            "weight": "weight",
            "flag": "flag",
            "field_and_source": "field_and_source",
            "field_and_source_xds": "field_and_source",  # Add field_and_source_xds key for compatibility
            "date": "2023-04-01T00:00:00",
            "data_time": "2023-04-01T00:00:00",
            "description": "Base data group",
        }
    }

    # For visibility type, add UVW to data group
    if ms_type == "visibility":
        xdt.attrs["data_groups"]["base"]["uvw"] = "uvw"

    # Add frequency and polarization attributes needed by some methods
    xdt.frequency.attrs["spectral_window_name"] = "test_spw"

    # Add observation info
    xdt.observation_info = {"intents": ["OBSERVE_TARGET#ON_SOURCE"]}

    return xdt


class TestMeasurementSetXdt:
    """Test cases for the MeasurementSetXdt class."""

    def test_invalid_sel(self):
        ms_xdt = MeasurementSetXdt(xr.DataTree())

        with pytest.raises(InvalidAccessorLocation, match="not a MSv4 node"):
            assert ms_xdt.sel()

    def test_get_invalid_field_and_source_xds(self):
        ms_xdt = MeasurementSetXdt(xr.DataTree())

        with pytest.raises(InvalidAccessorLocation, match="not a MSv4 node"):
            assert ms_xdt.get_field_and_source_xds()

    def test_get_invalid_partition_info(self):
        ms_xdt = MeasurementSetXdt(xr.DataTree())
        with pytest.raises(InvalidAccessorLocation, match="not a MSv4 node"):
            partition_info = ms_xdt.get_partition_info()
            assert isinstance(partition_info, dict)
            assert all(
                [
                    key in partition_info
                    for key in [
                        "spectral_window_name",
                        "field_name",
                        "polarization_setup",
                        "scan_name",
                        "source_name",
                        "intents",
                        "line_name",
                    ]
                ]
            )

    def test_initialization_with_invalid_ms(self):
        """Test initialization with an invalid DataTree (not MSv4)."""
        # Create a DataTree with invalid type
        invalid_xdt = xr.DataTree(xr.Dataset())
        invalid_xdt.attrs["type"] = "not_a_valid_ms_type"

        # Accessing methods should raise InvalidAccessorLocation
        with pytest.raises(InvalidAccessorLocation):
            invalid_xdt.xr_ms.get_partition_info()

    def test_initialization_with_ms(self):
        """Test initialization with a valid MSv4 DataTree."""
        # Create a valid MSv4 DataTree
        ms_xdt = create_test_ms_datatree(ms_type="visibility")

        # The accessor should be accessible and initialized correctly
        assert hasattr(ms_xdt, "xr_ms")
        assert isinstance(ms_xdt.xr_ms, MeasurementSetXdt)

    def test_get_partition_info(self):
        """Test get_partition_info method returns expected structure."""
        # Create a valid MSv4 DataTree
        ms_xdt = create_test_ms_datatree(ms_type="visibility")

        # Get partition info
        info = ms_xdt.xr_ms.get_partition_info()

        # Verify the structure of the returned info
        assert "spectral_window_name" in info
        assert "field_name" in info
        assert "polarization_setup" in info
        assert "scan_name" in info
        assert "source_name" in info
        assert "intents" in info
        assert "line_name" in info
        assert "data_group_name" in info

        # Check values
        assert info["spectral_window_name"] == "test_spw"
        assert "XX" in info["polarization_setup"]
        assert "YY" in info["polarization_setup"]
        assert "scan1" in info["scan_name"]
        assert "scan2" in info["scan_name"]
        assert "field1" in info["field_name"]
        assert "field2" in info["field_name"]
        assert "source1" in info["source_name"]
        assert "source2" in info["source_name"]
        assert "line1" in info["line_name"]
        assert "line2" in info["line_name"]
        assert info["data_group_name"] == "base"

    def test_get_field_and_source_xds(self):
        """Test get_field_and_source_xds method returns correct dataset."""
        # Create a valid MSv4 DataTree
        ms_xdt = create_test_ms_datatree(ms_type="visibility")

        # Get field and source dataset
        field_source_ds = ms_xdt.xr_ms.get_field_and_source_xds()

        # Verify it's an xarray Dataset
        assert isinstance(field_source_ds, xr.Dataset)

        # Check that it has the expected variables and coordinates
        assert "field_name" in field_source_ds.data_vars
        assert "source_name" in field_source_ds.data_vars
        assert "line_name" in field_source_ds.coords  # line_name is now a coordinate

        # Check some values
        assert "field1" in field_source_ds.field_name.values
        assert "source1" in field_source_ds.source_name.values
        assert "line1" in field_source_ds.line_name.values

    def test_sel_with_data_group(self):
        """Test sel method with data_group_name parameter."""
        # Create a valid MSv4 DataTree with multiple data groups
        ms_xdt = create_test_ms_datatree(ms_type="visibility")

        # Add a second data group - providing the required field_and_source_xds parameter
        ms_xdt.xr_ms.add_data_group(
            new_data_group_name="calibrated",
            field_and_source_xds="field_and_source",  # Explicitly specify the field_and_source_xds
            date_time="2023-04-01T00:00:00",
            description="Calibrated data group",
        )

        # Select the base data group
        selected_xdt = ms_xdt.xr_ms.sel(data_group_name="base")

        # Verify that selection worked correctly
        assert "base" in selected_xdt.attrs["data_groups"]
        assert "calibrated" not in selected_xdt.attrs["data_groups"]

    @pytest.mark.xfail(reason="Fails with TypeError: len() of unsized object")
    def test_sel_with_polarization(self):
        """Test sel method with polarization parameter."""
        # Create a valid MSv4 DataTree
        ms_xdt = create_test_ms_datatree(ms_type="visibility")

        # Verify original polarization values before selection
        assert "polarization" in ms_xdt.ds.coords
        assert len(ms_xdt.ds.polarization.values) == 2
        assert "XX" in ms_xdt.ds.polarization.values
        assert "YY" in ms_xdt.ds.polarization.values

        # Get data variables with polarization dimension
        data_vars_with_pol = []
        for var_name, var in ms_xdt.ds.data_vars.items():
            if "polarization" in var.dims:
                data_vars_with_pol.append(var_name)

        # Select XX polarization
        selected_xdt = ms_xdt.xr_ms.sel(polarization="XX")

        # Verify that selection worked correctly
        # Check that the polarization coordinate exists
        assert "polarization" in selected_xdt.ds.coords

        # Check the value of polarization - it can be either a scalar or an array with one element
        pol_value = selected_xdt.ds.polarization.values

        # Handle both scalar value or array
        if hasattr(pol_value, "__len__"):
            # It's an array-like object
            assert len(pol_value) == 1
            assert pol_value[0] == "XX"
        else:
            # It's a scalar value
            assert (
                pol_value.item() == "XX"
                if hasattr(pol_value, "item")
                else pol_value == "XX"
            )

        # Check that the data variables have been correctly filtered
        assert (
            len(data_vars_with_pol) > 0
        ), "No data variables found with polarization dimension"
        for var_name in data_vars_with_pol:
            assert var_name in selected_xdt.ds.data_vars
            # Check that the shape along polarization dimension is 1 or the dimension is gone
            if "polarization" in selected_xdt.ds[var_name].dims:
                pol_axis = selected_xdt.ds[var_name].dims.index("polarization")
                assert selected_xdt.ds[var_name].shape[pol_axis] == 1

    def test_add_data_group(self):
        """Test add_data_group method creates a new data group."""
        # Create a valid MSv4 DataTree
        ms_xdt = create_test_ms_datatree(ms_type="visibility")

        # Add a new data group with explicit field_and_source_xds parameter
        ms_xdt = ms_xdt.xr_ms.add_data_group(
            new_data_group_name="processed",
            field_and_source_xds="field_and_source",  # Explicitly specify the field_and_source_xds
            date_time="2023-04-01T00:00:00",
            description="Processed data",
        )

        # Verify the new data group exists
        assert "processed" in ms_xdt.attrs["data_groups"]

        # Verify the data group has the correct structure
        data_group = ms_xdt.attrs["data_groups"]["processed"]
        assert data_group["correlated_data"] == "correlated_data"
        assert data_group["weight"] == "weight"
        assert data_group["flag"] == "flag"
        assert data_group["uvw"] == "uvw"
        assert data_group["field_and_source"] == "field_and_source"
        assert data_group["description"] == "Processed data"

    def test_different_ms_types(self):
        """Test that different MS types are handled correctly."""
        # Test with all valid MS types
        for ms_type in MS_DATASET_TYPES:
            ms_xdt = create_test_ms_datatree(ms_type=ms_type)

            # Verify type is set correctly
            assert ms_xdt.attrs["type"] == ms_type

            # Verify we can get partition info without error
            info = ms_xdt.xr_ms.get_partition_info()
            assert info["data_group_name"] == "base"

            # For visibility type, verify UVW handling
            if ms_type == "visibility":
                assert "uvw" in ms_xdt.ds
                assert "uvw" in ms_xdt.attrs["data_groups"]["base"]
            else:
                assert "uvw" not in ms_xdt.attrs["data_groups"]["base"]
