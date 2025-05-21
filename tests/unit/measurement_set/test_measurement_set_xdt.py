import pytest
import numpy as np
import xarray as xr
from xradio.schema.check import check_datatree

from xradio.measurement_set.measurement_set_xdt import (
    MeasurementSetXdt,
    InvalidAccessorLocation,
    MS_DATASET_TYPES,
)


# Helper function to create a minimal valid MSv4 DataTree for testing
def create_test_ms_datatree(ms_type="visibility", validate=True):
    """
    Create a minimal valid MSv4 DataTree for testing purposes.

    Parameters
    ----------
    ms_type : str
        Type of measurement set to create (visibility, spectrum, radiometer)
    validate : bool, optional
        Whether to validate the created datatree against schema, by default True

    Returns
    -------
    xarray.DataTree
        A valid MSv4 DataTree node according to
        https://xradio.readthedocs.io/en/latest/measurement_set/schema_and_api/measurement_set_schema.html

    Raises
    ------
    ValueError
        If ms_type is invalid or if validation fails
    """
    # Ensure valid type
    # MS_DATASET_TYPES ={"visibility", "spectrum", "radiometer"}
    # This example is setup for visibility type only
    if ms_type not in MS_DATASET_TYPES:
        raise ValueError(f"ms_type must be one of {MS_DATASET_TYPES}")

    # Correlated DataSet
    # Create base coordinates
    time = np.array([1.0, 2.0, 3.0])
    baseline_id = np.array([0, 1, 2])  # Changed from baseline to baseline_id
    frequency = np.linspace(1e9, 2e9, 10)  # 10 frequency channels
    polarization = np.array(["XX", "YY"])
    # Create field_name coordinate
    field_name = np.array(["field1", "field1", "field2"])
    # Create baseline antenna name coordinates
    baseline_antenna1_name = np.array(["ant1", "ant1", "ant2"])
    baseline_antenna2_name = np.array(["ant2", "ant3", "ant3"])

    # Create a scan_name coordinate, which is optional
    scan_name = np.array(["scan1", "scan1", "scan2"])

    # Create mock data - properly generating complex data
    # Generate real and imaginary parts separately and combine them
    real_part = np.random.random(
        (len(time), len(baseline_id), len(frequency), len(polarization))
    )
    imag_part = np.random.random(
        (len(time), len(baseline_id), len(frequency), len(polarization))
    )
    visibility = real_part + 1j * imag_part  # Create complex numbers

    weight = np.random.random(
        (len(time), len(baseline_id), len(frequency), len(polarization))
    )

    flag = np.zeros(
        (len(time), len(baseline_id), len(frequency), len(polarization)), dtype=bool
    )

    # For visibility type, create UVW data
    uvw = np.random.random((len(time), len(baseline_id), 3))
    uvw_label = ["u", "v", "w"]  # Added uvw_label coordinate

    # Create the main dataset with all required data variables
    # PS: check if scan_name is correct here!!!
    ds = xr.Dataset(
        data_vars={
            "VISIBILITY": xr.DataArray(  # Changed from correlated_data to VISIBILITY
                visibility, dims=("time", "baseline_id", "frequency", "polarization")
            ),
            "WEIGHT": xr.DataArray(  # Changed from weight to WEIGHT
                weight, dims=("time", "baseline_id", "frequency", "polarization")
            ),
            "FLAG": xr.DataArray(  # Changed from flag to FLAG
                flag, dims=("time", "baseline_id", "frequency", "polarization")
            ),
            "scan_name": xr.DataArray(scan_name, dims=("time")),
        },
        coords={
            "time": time,
            "baseline_id": baseline_id,  # Changed from baseline to baseline_id
            "polarization": polarization,
            "frequency": frequency,
            "field_name": ("time", field_name),  # Added field_name coordinate
            "baseline_antenna1_name": (
                "baseline_id",
                baseline_antenna1_name,
            ),  # Added baseline_antenna1_name
            "baseline_antenna2_name": (
                "baseline_id",
                baseline_antenna2_name,
            ),  # Added baseline_antenna2_name
        },
    )

    # Add UVW for visibility type
    if ms_type == "visibility":
        ds["UVW"] = xr.DataArray(uvw, dims=("time", "baseline_id", "uvw_label"))
        ds.coords["uvw_label"] = uvw_label  # Added uvw_label coordinate
        ds["UVW"].attrs["units"] = ["m", "m", "m"]
        ds["UVW"].attrs["type"] = "uvw"
        ds["UVW"].attrs["frame"] = "icrs"
    ds["VISIBILITY"].attrs["units"] = ["Jy", "Jy"]

    # Add UVW for visibility type
    if ms_type == "visibility":
        ds["UVW"] = xr.DataArray(  # Changed from uvw to UVW
            uvw,
            dims=("time", "baseline_id", "uvw_label"),  # Changed uvw_index to uvw_label
        )
        ds.coords["uvw_label"] = uvw_label  # Added uvw_label coordinate
        ds["UVW"].attrs["units"] = ["m", "m", "m"]
        ds["UVW"].attrs["type"] = "uvw"
        ds["UVW"].attrs[
            "frame"
        ] = "icrs"  # Fixed frame attribute spelling and added required value

    # Add required attributes to coordinates
    # Time attributes
    ds.time.attrs["type"] = "time"
    ds.time.attrs["units"] = ["s"]
    ds.time.attrs["scale"] = "utc"
    ds.time.attrs["format"] = "unix"
    ds.time.attrs["integration_time"] = xr.DataArray(
        float(np.ones_like(time)[0] * 0.1), attrs={"units": ["s"], "type": "quantity"}
    )  # 0.1 seconds integration time

    # Frequency attributes
    ds.frequency.attrs["type"] = "spectral_coord"
    ds.frequency.attrs["units"] = ["Hz"]
    ds.frequency.attrs["observer"] = "TOPO"
    ds.frequency.attrs["reference_frequency"] = xr.DataArray(
        1.5e9, attrs={"units": ["Hz"], "type": "spectral_coord", "observer": "TOPO"}
    )  # Reference frequency in Hz
    ds.frequency.attrs["channel_width"] = xr.DataArray(
        1e6, attrs={"units": ["Hz"], "type": "quantity"}  # 1 MHz channel width
    )  # 1 MHz channel width

    # Create the field_and_source dataset with proper schema
    field_source_ds = xr.Dataset(
        coords={
            "field_name": ["field1", "field2"],
            "source_name": ("field_name", ["source1", "source2"]),
            "sky_dir_label": ("sky_dir_label", ["ra", "dec"]),
            "line_name": ("field_name", ["line1", "line2"]),
        },
        attrs={"type": "field_and_source"},
    )

    # Add schema type to field_and_source dataset

    # Create the DataTree structure
    xdt = xr.DataTree(ds)

    # Add the field_and_source dataset as a child node
    xdt["field_and_source"] = xr.DataTree(field_source_ds)

    # Add required attributes
    xdt.attrs["type"] = ms_type
    xdt.attrs["data_groups"] = {
        "base": {
            "VISIBILITY": "VISIBILITY",  # Changed from correlated_data to VISIBILITY
            "WEIGHT": "WEIGHT",  # Changed from weight to WEIGHT
            "FLAG": "FLAG",  # Changed from flag to FLAG
            "field_and_source": "field_and_source",
            "field_and_source_xds": "field_and_source",
            "date": "2023-04-01T00:00:00",
            "data_time": "2023-04-01T00:00:00",
            "description": "Base data group",
        }
    }

    # For visibility type, add UVW to data group
    if ms_type == "visibility":
        xdt.attrs["data_groups"]["base"]["UVW"] = "UVW"  # Changed from uvw to UVW

    # Add frequency and polarization attributes needed by some methods
    xdt.frequency.attrs["spectral_window_name"] = "test_spw"

    # Add required attributes that were missing
    xdt.attrs["schema_version"] = "1.0.0"
    xdt.attrs["creator"] = {
        "name": "test_creator",
        "version": "1.0.0",
    }
    xdt.attrs["creation_date"] = "2023-04-01T00:00:00"

    # Add observation_info
    xdt.attrs["observation_info"] = {
        "observer": ["observer1", "observer2"],
        "project": "test_project",
        "release_date": "2023-04-01T00:00:00",
        "intents": ["OBSERVE_TARGET#ON_SOURCE"],
    }

    # Add processor_info
    xdt.attrs["processor_info"] = {
        "type": "CORRELATOR",
        "sub_type": "JIVE",
    }

    xdt.attrs["data_groups"] = {
        "base": {
            "correlated_data": "VISIBILITY",
            "weight": "WEIGHT",
            "flag": "FLAG",
            "uvw": "UVW",
            "field_and_source": "field_and_source",
            "description": "Base data group",
            "date": "2023-04-01T00:00:00",
        }
    }

    # Add processor_info
    xdt.attrs["creator"] = {
        "software_name": "XRADIO",
        "version": "0.0.56",
    }

    # Validate the created datatree if requested
    if validate:
        # Validate the entire datatree
        datatree_issues = check_datatree(xdt)
        if str(datatree_issues) != "No schema issues found":
            raise ValueError(f"DataTree validation failed: {datatree_issues}")

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

    def test_check_valid_datatree(self):
        """Test that the created datatree complies with the datatree schema checker"""
        ms_xdt = create_test_ms_datatree(ms_type="visibility")

        issues = check_datatree(ms_xdt)
        # The check_datatree function returns a SchemaIssues object, not a string
        assert (
            str(issues) == "No schema issues found"
        ), f"Schema validation failed: {issues}"

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

        # Check that it has the expected coordinates
        assert "field_name" in field_source_ds.coords
        assert "source_name" in field_source_ds.coords
        assert "line_name" in field_source_ds.coords
        assert "sky_dir_label" in field_source_ds.coords

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

        # Get the polarization value and handle both scalar and array cases
        pol_value = selected_xdt.ds.polarization.values

        # Convert to list if it's not already a list
        if not isinstance(pol_value, (list, np.ndarray)):
            pol_value = [pol_value]

        # Verify the polarization value

        assert str(pol_value) == "XX"

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
                # Verify the selected polarization value in the data variable
                assert str(selected_xdt.ds[var_name].coords["polarization"][0]) == "XX"

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
        assert data_group["correlated_data"] == "VISIBILITY"
        assert data_group["weight"] == "WEIGHT"
        assert data_group["flag"] == "FLAG"
        assert data_group["uvw"] == "UVW"
        assert data_group["field_and_source"] == "field_and_source"
        assert data_group["description"] == "Processed data"
