import pytest
import numpy as np
import xarray as xr
from xradio.measurement_set.load_processing_set import ProcessingSetIterator
from xradio.measurement_set import load_processing_set

from xradio.schema.check import check_datatree


@pytest.mark.parametrize(
    "convert_measurement_set_to_processing_set",
    ["Antennae_North.cal.lsrk.split.ms"],
    indirect=True,
)
class TestLoadProcessingSet:
    """Tests for load_processing_set using real data"""

    # @pytest.mark.parametrize("convert_measurement_set_to_processing_set", ["Antennae_North.cal.lsrk.split.ms"], indirect=True)
    def test_check_datatree(self, convert_measurement_set_to_processing_set):
        """Test that the converted MS to PS complies with the datatree schema checker"""
        ps_xdt = load_processing_set(str(convert_measurement_set_to_processing_set))
        issues = check_datatree(ps_xdt)
        # The check_datatree function returns a SchemaIssues object, not a string
        assert (
            str(issues) == "No schema issues found"
        ), f"Schema validation failed: {issues}"

    def test_basic_load(self, convert_measurement_set_to_processing_set):
        """Test basic loading of processing set without parameters"""
        ps_xdt = load_processing_set(str(convert_measurement_set_to_processing_set))

        # Verify basic structure
        assert isinstance(ps_xdt, xr.DataTree)
        assert ps_xdt.attrs.get("type") == "processing_set"
        # Should have at least one measurement set. Partitioning gives the number of children
        assert len(ps_xdt.children) > 0

    def test_selective_loading(self, convert_measurement_set_to_processing_set):
        """Test loading with selection parameters"""
        # First load normally to get MS names
        full_ps = load_processing_set(str(convert_measurement_set_to_processing_set))

        # Check MS names are the expected ones
        ms_basename = "Antennae_North.cal.lsrk.split"
        expected_names = [f"{ms_basename}_{i}" for i in range(4)]  # 0 to 3
        ms_names = list(full_ps.children.keys())
        assert len(ms_names) == len(
            expected_names
        ), "Number of measurement sets doesn't match"
        for ms_name, expected_name in zip(sorted(ms_names), sorted(expected_names)):
            assert (
                ms_name == expected_name
            ), f"Expected {expected_name} but got {ms_name}"

        # Test loading with selection parameters
        sel_parms = {ms_name: {"time": slice(0, 10)}}
        ps_xdt = load_processing_set(
            str(convert_measurement_set_to_processing_set), sel_parms=sel_parms
        )

        assert isinstance(ps_xdt, xr.DataTree)
        assert ms_name in ps_xdt.children
        assert ps_xdt[ms_name].dims["time"] <= 10

    def test_data_group_selection(self, convert_measurement_set_to_processing_set):
        """Test loading with specific data group"""
        ps_xdt = load_processing_set(
            str(convert_measurement_set_to_processing_set), data_group_name="base"
        )

        assert isinstance(ps_xdt, xr.DataTree)
        for ms_xdt in ps_xdt.children.values():
            assert "base" in ms_xdt.attrs.get("data_groups", {})

    def test_variable_selection(self, convert_measurement_set_to_processing_set):
        """Test loading with specific variables included/excluded"""
        # Test including specific variables
        include_vars = ["VISIBILITY"]
        ps_xdt = load_processing_set(
            str(convert_measurement_set_to_processing_set),
            include_variables=include_vars,
        )

        for ms_xdt in ps_xdt.children.values():
            assert "VISIBILITY" in ms_xdt.data_vars
            assert len(ms_xdt.data_vars) == 1

        # Test dropping specific variables
        drop_vars = ["WEIGHT"]
        ps_xdt = load_processing_set(
            str(convert_measurement_set_to_processing_set), drop_variables=drop_vars
        )

        for ms_xdt in ps_xdt.children.values():
            assert "WEIGHT" not in ms_xdt.data_vars

    def test_sub_datasets(self, convert_measurement_set_to_processing_set):
        """Test loading with and without sub-datasets"""
        # Test with sub-datasets
        ps_with_subs = load_processing_set(
            str(convert_measurement_set_to_processing_set), load_sub_datasets=True
        )

        # Test without sub-datasets
        ps_without_subs = load_processing_set(
            str(convert_measurement_set_to_processing_set), load_sub_datasets=False
        )

        for ms_xdt in ps_without_subs.children.values():
            assert not any("xds" in name for name in ms_xdt.keys())


@pytest.mark.parametrize(
    "convert_measurement_set_to_processing_set",
    ["Antennae_North.cal.lsrk.split.ms"],
    indirect=True,
)
class TestProcessingSetIterator:
    """Integration tests for ProcessingSetIterator using real data"""

    def test_iterator_with_store(self, convert_measurement_set_to_processing_set):
        """Test iterator loading from store"""
        # First load normally to get MS names
        full_ps = load_processing_set(str(convert_measurement_set_to_processing_set))
        ms_name = list(full_ps.children.keys())[0]

        sel_parms = {ms_name: {"time": slice(0, 10)}}

        iterator = ProcessingSetIterator(
            sel_parms=sel_parms,
            input_data_store=str(convert_measurement_set_to_processing_set),
        )

        # Test iteration
        item = next(iterator)
        # The item should be the first measurement set from the processing set
        assert isinstance(item, xr.DataTree)
        assert "time" in item.dims
        assert item.dims["time"] <= 10

        # Test StopIteration
        with pytest.raises(StopIteration):
            next(iterator)

    def test_iterator_with_memory(self, convert_measurement_set_to_processing_set):
        """Test iterator with in-memory data"""
        # Load data into memory
        full_ps = load_processing_set(str(convert_measurement_set_to_processing_set))
        ms_name = list(full_ps.children.keys())[0]

        sel_parms = {ms_name: {"time": slice(0, 10)}}

        iterator = ProcessingSetIterator(
            sel_parms=sel_parms,
            input_data_store=str(convert_measurement_set_to_processing_set),
            input_data=full_ps,
        )

        # Test iteration
        item = next(iterator)
        assert isinstance(item, xr.DataTree)

        # Test StopIteration
        with pytest.raises(StopIteration):
            next(iterator)

    def test_iterator_with_data_groups(self, convert_measurement_set_to_processing_set):
        """Test iterator with data group selection"""
        # First load normally to get MS names
        full_ps = load_processing_set(str(convert_measurement_set_to_processing_set))
        ms_name = list(full_ps.children.keys())[0]

        sel_parms = {ms_name: {"time": slice(0, 10)}}

        iterator = ProcessingSetIterator(
            sel_parms=sel_parms,
            input_data_store=str(convert_measurement_set_to_processing_set),
            data_group_name="base",
        )

        item = next(iterator)
        assert isinstance(item, xr.DataTree)
        assert "base" in item.attrs.get("data_groups", {})


if __name__ == "__main__":
    pytest.main(["-v", "-s", __file__])
