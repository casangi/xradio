from collections import namedtuple
import os
from pathlib import Path
import pytest
import shutil


import xarray as xr


from toolviper.utils.data import download
from xradio.measurement_set import convert_msv2_to_processing_set


from tests.unit.measurement_set.ms_test_utils.gen_test_ms import (
    gen_test_ms,
    make_ms_empty,
)

# Ensure pytest assert introspection in vis data checks
pytest.register_assert_rewrite(
    "tests.unit.measurement_set.ms_test_utils.check_msv4_matches_msv2_description"
)


"""
A tuple with an MS filename (as str) and a description of its expected structure and contents (as a dict).
"""
MSWithSpec = namedtuple("MSWithSpec", "fname descr")


# def pytest_configure(config):
#     config.addinivalue_line(
#         "markers",
#         "uses_download: marks tests that use the function to download test MSs (require medium-size downloads and "
#         "tend to be slower than others)",
#     )


# Generated test MS fixtures


@pytest.fixture(scope="session")
def ms_empty_required():
    """
    An MS that has the required tables/columns definitions and is empty
    (0 rows)
    """
    name = "test_ms_empty_def_required.ms"
    make_ms_empty(name)
    yield MSWithSpec(name, {})
    shutil.rmtree(name)


@pytest.fixture
def ms_empty_complete(scope="session"):
    """
    An MS that has the complete tables/columns definitions and is empty
    (0 rows)
    """
    name = "test_ms_empty_def_complete.ms"
    make_ms_empty(name, complete=True)
    yield MSWithSpec(name, {})
    shutil.rmtree(name)


@pytest.fixture(scope="session")
def ms_minimal_required():
    """
    Small MS with the required set of tables and columns definitions (according to python-casacore
    standard MS definitions)
    """
    name = "test_msv2_minimal_required.ms"
    spec = gen_test_ms(name, required_only=True)
    yield MSWithSpec(name, spec)
    shutil.rmtree(name)


@pytest.fixture(scope="session")
def ms_minimal_misbehaved():
    """
    Small MS with a number of misbehaviors as observed in different MS from various observatories
    and projects
    """
    name = "test_msv2_minimal_required_misbehaved.ms"
    spec = gen_test_ms(
        name, opt_tables=True, vlbi_tables=False, required_only=True, misbehave=True
    )
    yield MSWithSpec(name, spec)
    shutil.rmtree(name)


@pytest.fixture(scope="session")
def ms_minimal_without_opt():
    """
    Small MS with a number of misbehaviors as observed in different MS from various observatories
    and projects
    """
    name = "test_msv2_minimal_required_without_opt_subtables.ms"
    spec = gen_test_ms(
        name, opt_tables=False, vlbi_tables=False, required_only=True, misbehave=False
    )
    yield MSWithSpec(name, spec)
    shutil.rmtree(name)


# Besides the few test MSs from above,  one can generate custom MSs passing different descr dicts.
@pytest.fixture(scope="session")
def ms_custom_spec(request):
    """
    Expects in request.param an MS description / description of the visibilities dataset used in
    gen_test_ms to produce it
    """

    name_appendix = "session"
    if hasattr(request, "cls") and request.cls:
        name_appendix = request.cls.__name__

    name = f"test_ms_custom_spec_for_{name_appendix}.ms"
    msv2_custom_description = gen_test_ms(
        name,
        request.param,
        opt_tables=True,
        vlbi_tables=False,
        required_only=True,
        misbehave=False,
    )
    yield MSWithSpec(name, msv2_custom_description)
    shutil.rmtree(name)


@pytest.fixture(scope="session")
def ms_tab_nonexistent():
    name = "test_nonexistent_table_from_test_table_exists.foo.bar.tab"
    yield MSWithSpec(name, {})


# Generic xds fixtures (generic: loaded from MSv2 mostly as is, not yet in MSv4 format)


@pytest.fixture(scope="session")
def generic_antenna_xds_min(ms_minimal_required):
    """A generic antenna xds (loaded form MSv2 mostly as is), loaded from the MS/ANTENNA subtable"""
    from xradio.measurement_set._utils._msv2._tables.read import load_generic_table

    generic_antenna_xds = load_generic_table(ms_minimal_required.fname, "ANTENNA")
    return generic_antenna_xds


@pytest.fixture(scope="session")
def generic_phase_cal_xds_min(ms_minimal_required):
    """A generic phase_cal xds (loaded form MSv2 mostly as is), loaded from the minimal MS/PHASE_CAL subtable"""

    from xradio.measurement_set._utils._msv2._tables.read import load_generic_table

    subt = load_generic_table(ms_minimal_required.fname, "PHASE_CAL")
    return subt


@pytest.fixture(scope="session")
def generic_source_xds_min(ms_minimal_required):
    """A generic source xds (loaded form MSv2 mostly as is), loaded from the minimal MS/SOURCE subtable"""

    from xradio.measurement_set._utils._msv2._tables.read import load_generic_table

    subt = load_generic_table(ms_minimal_required.fname, "SOURCE")
    return subt


# MSv4 xds and xdt fixtures


@pytest.fixture(scope="session")
def processing_set_min_path():
    """path to the 'mininal_required' processing set"""
    out_name = "test_converted_msv2_to_msv4_minimal_required.zarr"
    yield out_name


@pytest.fixture(scope="session")
def msv4_min_path(processing_set_min_path, ms_minimal_required):
    """path to the MSv4 of the 'mininal_required' processing set"""
    msv4_id = "msv4id"
    msv4_path = (
        processing_set_min_path
        + "/"
        + ms_minimal_required.fname.rsplit(".")[0]
        + "_"
        + msv4_id
    )
    yield msv4_path


@pytest.fixture(scope="session")
def msv4_xdt_min(ms_minimal_required, processing_set_min_path, msv4_min_path):
    """An MSv4 xdt (one single MSv4)"""
    from xradio.measurement_set._utils._msv2.conversion import (
        convert_and_write_partition,
    )

    convert_and_write_partition(
        ms_minimal_required.fname,
        processing_set_min_path,
        "msv4id",
        {"DATA_DESC_ID": [0], "OBS_MODE": ["CAL_ATMOSPHERE#ON_SOURCE"]},
        use_table_iter=False,
        overwrite=True,
    )

    msv4_xdt = xr.open_datatree(
        msv4_min_path,
        engine="zarr",
    )

    yield msv4_xdt
    shutil.rmtree(processing_set_min_path)


@pytest.fixture(scope="session")
def msv4_xds_min(msv4_xdt_min):
    """An MSv4 main xds (correlated data, one partition) xds"""

    msv4_xds = msv4_xdt_min.ds
    yield msv4_xds


@pytest.fixture(scope="session")
def antenna_xds_min(msv4_xdt_min, msv4_min_path):
    """An MSv4 secondary antenna dataset ('antenna_xds')"""

    antenna_xds = msv4_xdt_min["antenna_xds"].ds

    yield antenna_xds


@pytest.fixture(scope="session")
def pointing_xds_min(msv4_xdt_min, msv4_min_path):
    """An MSv4 secondary pointing dataset ('pointing_xds')"""

    pointing_xds = msv4_xdt_min["pointing_xds"]

    yield pointing_xds


@pytest.fixture(scope="session")
def sys_cal_xds_min(msv4_xdt_min, msv4_min_path):
    """An MSv4 secondary sys cal dataset ('system_calibration_xds')"""

    syscal_xds = msv4_xdt_min["system_calibration_xds"]

    yield syscal_xds


# Used in test_processing_set_xdt / test_load_processing_set


@pytest.fixture(scope="session")
def processing_set_from_custom_ms(request):
    """
    Expects in request.param an MS description / description of the visibilities dataset used in
    gen_test_ms to produce it. After that, it is converted to a processing set.
    """

    name_appendix = "session"
    if hasattr(request, "cls") and request.cls:
        name_appendix = request.cls.__name__
    msv2_name = f"test_ms_custom_spec_for_{name_appendix}.ms"
    msv2_custom_description = gen_test_ms(
        msv2_name,
        request.param,
        opt_tables=True,
        vlbi_tables=False,
        required_only=True,
        misbehave=False,
    )

    ps_name = f"test_proc_set_from_custom_ms_for_{name_appendix}.ps.zarr"
    convert_msv2_to_processing_set(
        in_file=msv2_name,
        out_file=ps_name,
        partition_scheme=[],
        overwrite=False,
        parallel_mode="partition",
    )
    shutil.rmtree(msv2_name)

    yield ps_name
    shutil.rmtree(ps_name)


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
