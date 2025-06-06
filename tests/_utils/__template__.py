import pytest
from pathlib import Path
from xradio.measurement_set import load_processing_set
from xradio.schema.check import check_datatree


def test_simple():
    assert True


def test_simple_string(sample_fixture):
    print(f"Sample fixture value: {sample_fixture}")
    assert sample_fixture == "sample_data"


class TestLoadProcessingSet:
    """Tests for load_processing_set using real data"""

    # This is a simple test to ensure that pytest is working correctly.
    # It should be run with pytest to verify that the testing framework is set up properly.
    def test_pytest_setup(self):
        """A simple test to ensure pytest is set up correctly."""
        assert True, "Pytest setup is working correctly."

    @pytest.mark.parametrize(
        "test_ps_path", ["Antennae_North.cal.lsrk.split.ms"], indirect=True
    )
    def test_check_datatree(self, test_ps_path):
        """Test that the converted MS to PS complies with the datatree schema checker"""
        ps_xdt = load_processing_set(str(test_ps_path))
        issues = check_datatree(ps_xdt)
        # The check_datatree function returns a SchemaIssues object, not a string
        assert (
            str(issues) == "No schema issues found"
        ), f"Schema validation failed: {issues}"


if __name__ == "__main__":
    pytest.main(["-v", "-s", __file__])
# This is a simple test to ensure that pytest is working correctly.
# It should be run with pytest to verify that the testing framework is set up properly.
# To run this test, use the command: pytest -v __test_example__.py
# This will execute the test and show the results in verbose mode.
