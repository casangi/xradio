from pathlib import Path
import os
import pytest
from toolviper.utils.data import download
from xradio.measurement_set import convert_msv2_to_processing_set


import shutil


@pytest.fixture(scope="module")
def sample_fixture():
    return "sample_data"


def test_data_path(input_ms, folder="/tmp"):
    """Returns path to test MeasurementSet v2"""
    # Download MS
    download(file=input_ms, folder=folder)
    return Path(os.path.join(folder, input_ms))


@pytest.fixture
def test_ps_path(request, tmp_path):
    """Create a processing set from test MS for testing"""
    ps_path = tmp_path / "test_processing_set.ps.zarr"
    # Convert MS to processing set
    convert_msv2_to_processing_set(
        in_file=str(test_data_path(request.param, tmp_path)),
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
    shutil.rmtree(test_data_path(request.param, tmp_path))
