from pathlib import Path
import shutil
import os
import pytest
from toolviper.utils.data import download
from xradio.measurement_set import convert_msv2_to_processing_set


@pytest.fixture(scope="module")
def sample_fixture():
    return "sample_data"


def download_measurement_set(input_ms, directory="/tmp"):
    """Returns path to test MeasurementSet v2"""
    # Download MS
    download(file=input_ms, folder=directory)
    return Path(os.path.join(directory, input_ms))


@pytest.fixture
def convert_measurement_set_to_processing_set(request, tmp_path):
    """Create a processing set from test MS for testing"""
    ps_path = tmp_path / "test_processing_set.ps.zarr"
    # Convert MS to processing set
    convert_msv2_to_processing_set(
        in_file=str(download_measurement_set(request.param, tmp_path)),
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
    yield ps_path
    shutil.rmtree(ps_path)
    shutil.rmtree(download_measurement_set(request.param, tmp_path))
