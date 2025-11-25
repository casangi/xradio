from pathlib import Path
import shutil
import os
import pytest
from xradio.testing.measurement_set.io import (
    convert_msv2_to_processing_set,
    download_measurement_set,
)


@pytest.fixture(scope="module")
def sample_fixture():
    return "sample_data"


@pytest.fixture
def convert_measurement_set_to_processing_set(request, tmp_path):
    """Create a processing set from test MS for testing"""
    ps_path = tmp_path / "test_processing_set.ps.zarr"
    ms_path = download_measurement_set(request.param, tmp_path)
    # Convert MS to processing set
    convert_msv2_to_processing_set(
        ms_path,
        ps_path,
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
    shutil.rmtree(ms_path)
