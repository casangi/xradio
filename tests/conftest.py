from pathlib import Path
import pytest
from toolviper.utils.data import download
from xradio.measurement_set import convert_msv2_to_processing_set

input_ms = "Antennae_North.cal.lsrk.split.ms"


@pytest.fixture(scope="module")
def sample_fixture():
    return "sample_data"


@pytest.fixture
def test_data_path():
    """Returns path to test MeasurementSet v2"""
    # Download MS
    download(file=input_ms, folder="/tmp")
    return Path("/tmp/" + input_ms)


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
