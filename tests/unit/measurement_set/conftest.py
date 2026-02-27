from collections import namedtuple
import os
from pathlib import Path
import pytest
import shutil

# Ensure pytest assert introspection in vis data checks
# Must be imported before any other imports
pytest.register_assert_rewrite("xradio.testing.measurement_set.checker")

import xarray as xr

# Ensure pytest assert introspection in vis data checks
pytest.register_assert_rewrite("xradio.testing.measurement_set.checker")

from xradio.measurement_set import open_processing_set
from xradio.testing.measurement_set.msv2_io import (
    gen_test_ms,
    gen_minimal_ms,
    make_ms_empty,
    build_minimal_msv4_xdt,
    build_processing_set_from_msv2,
)
from xradio.testing.measurement_set.io import download_measurement_set

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
    fname, spec = gen_minimal_ms(name)
    yield MSWithSpec(fname, spec)
    shutil.rmtree(fname)


@pytest.fixture(scope="session")
def ms_minimal_misbehaved():
    """
    Small MS with a number of misbehaviors as observed in different MS from various observatories
    and projects
    """
    name = "test_msv2_minimal_required_misbehaved.ms"
    fname, spec = gen_test_ms(
        name,
        opt_tables=True,
        vlbi_tables=False,
        required_only=True,
        misbehave=True,
    )
    yield MSWithSpec(fname, spec)
    shutil.rmtree(fname)


@pytest.fixture(scope="session")
def ms_minimal_without_opt():
    """
    Small MS with a number of misbehaviors as observed in different MS from various observatories
    and projects
    """
    name = "test_msv2_minimal_required_without_opt_subtables.ms"
    fname, spec = gen_test_ms(
        name,
        opt_tables=False,
        vlbi_tables=False,
        required_only=True,
        misbehave=False,
    )
    yield MSWithSpec(fname, spec)
    shutil.rmtree(fname)


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
    fname, msv2_custom_description = gen_test_ms(
        name,
        descr=request.param,
        opt_tables=True,
        vlbi_tables=False,
        required_only=True,
        misbehave=False,
    )
    yield MSWithSpec(fname, msv2_custom_description)
    shutil.rmtree(fname)


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
def msv4_xdt_min(ms_minimal_required, tmp_path_factory):
    """An MSv4 xdt (one single MSv4)"""
    processing_set_root = tmp_path_factory.mktemp(
        "test_converted_msv2_to_msv4_minimal_required"
    )
    msv4_path = build_minimal_msv4_xdt(
        ms_minimal_required.fname,
        out_root=processing_set_root,
        msv4_id="msv4id",
        partition_kwargs={
            "DATA_DESC_ID": [0],
            "OBS_MODE": ["CAL_ATMOSPHERE#ON_SOURCE"],
        },
    )

    msv4_xdt = xr.open_datatree(msv4_path, engine="zarr")

    yield msv4_xdt
    shutil.rmtree(processing_set_root, ignore_errors=True)


@pytest.fixture(scope="session")
def msv4_xds_min(msv4_xdt_min):
    """An MSv4 main xds (correlated data, one partition) xds"""

    msv4_xds = msv4_xdt_min.ds
    yield msv4_xds


@pytest.fixture(scope="session")
def antenna_xds_min(msv4_xdt_min):
    """An MSv4 secondary antenna dataset ('antenna_xds')"""

    antenna_xds = msv4_xdt_min["antenna_xds"].ds

    yield antenna_xds


@pytest.fixture(scope="session")
def pointing_xds_min(msv4_xdt_min):
    """An MSv4 secondary pointing dataset ('pointing_xds')"""

    pointing_xds = msv4_xdt_min["pointing_xds"]

    yield pointing_xds


@pytest.fixture(scope="session")
def sys_cal_xds_min(msv4_xdt_min):
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
    gen_test_ms(
        msv2_name,
        descr=request.param,
        opt_tables=True,
        vlbi_tables=False,
        required_only=True,
        misbehave=False,
    )

    ps_name = Path(f"test_proc_set_from_custom_ms_for_{name_appendix}.ps.zarr")
    build_processing_set_from_msv2(
        msv2_name,
        ps_name,
        partition_scheme=[],
        persistence_mode="w",
        parallel_mode="partition",
    )
    open_processing_set(str(ps_name))  # check it opens
    shutil.rmtree(msv2_name)

    yield ps_name
    shutil.rmtree(ps_name)


@pytest.fixture
def convert_measurement_set_to_processing_set(request, tmp_path):
    """Create a processing set from test MS for testing"""
    ps_path = tmp_path / "test_processing_set.ps.zarr"
    ms_path = download_measurement_set(request.param, tmp_path)
    # Convert MS to processing set
    build_processing_set_from_msv2(
        ms_path,
        ps_path,
        partition_scheme=[],
        main_chunksize=0.01,
        pointing_chunksize=0.00001,
        pointing_interpolate=True,
        ephemeris_interpolate=True,
        use_table_iter=False,
        persistence_mode="w",
        parallel_mode="none",
    )
    open_processing_set(str(ps_path))  # check it opens
    yield ps_path
    shutil.rmtree(ps_path)
    shutil.rmtree(ms_path)
