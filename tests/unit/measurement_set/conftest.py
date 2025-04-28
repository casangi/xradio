import pytest

# Ensure pytest assert introspection in vis data checks
pytest.register_assert_rewrite("tests.unit.measurement_set.ms_test_utils.cds_checks")

from collections import namedtuple
import shutil

import xarray as xr

from tests.unit.measurement_set.ms_test_utils.gen_test_ms import (
    gen_test_ms,
    make_ms_empty,
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


@pytest.fixture(scope="session")
def ms_empty_required():
    """
    An MS that has all the required tables/columns definitions and is empty
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
    name = "test_msv2_minimal_required.ms"
    spec = gen_test_ms(name, required_only=True)
    yield MSWithSpec(name, spec)
    shutil.rmtree(name)


@pytest.fixture(scope="session")
def ms_minimal_misbehaved():
    name = "test_msv2_minimal_required_misbehaved.ms"
    spec = gen_test_ms(
        name, opt_tables=True, vlbi_tables=False, required_only=True, misbehave=True
    )
    yield MSWithSpec(name, spec)
    shutil.rmtree(name)


@pytest.fixture(scope="session")
def ms_minimal_for_writes():
    """MS to be used to write subtables inside"""
    name = "test_msv2_minimal_required_for_writes.ms"
    spec = gen_test_ms(name, required_only=True)
    yield MSWithSpec(name, spec)
    shutil.rmtree(name)


@pytest.fixture(scope="session")
def ms_tab_nonexistent():
    name = "test_nonexistent_table_from_test_table_exists.foo.bar.tab"
    yield MSWithSpec(name, {})


@pytest.fixture(scope="session")
def ddi_xds_min(ms_minimal_required):
    """A DATA_DESCRIPTION/DDI xds, loaded from the minimal MS"""
    from xradio.measurement_set._utils._msv2._tables.read import load_generic_table

    # not available:
    # subt = cds_minimal_required.metainfo["ddi"]

    subt = load_generic_table(ms_minimal_required.fname, "DATA_DESCRIPTION")
    return subt


@pytest.fixture(scope="session")
def generic_antenna_xds_min(ms_minimal_required):
    """A generic antenna xds (loaded form MSv2 mostly as is), loaded from the MS/ANTENNA subtable"""
    from xradio.measurement_set._utils._msv2._tables.read import load_generic_table

    generic_antenna_xds = load_generic_table(ms_minimal_required.fname, "ANTENNA")
    return generic_antenna_xds


@pytest.fixture(scope="session")
def field_xds_min(cds_minimal_required):
    """A field xds, loaded from the minimal MS/FIELD subtable"""

    subt = cds_minimal_required["metainfo"]["field"]
    # Or alternatively, from ms_minimal_required read subtable
    # subt = load_generic_table(ms_minimal_required.fname, "FIELD")
    return subt


@pytest.fixture(scope="session")
def feed_xds_min(cds_minimal_required):
    """A feed xds, loaded from the minimal MS/FEED subtable"""

    subt = cds_minimal_required["metainfo"]["feed"]
    # Or alternatively, from ms_minimal_required read subtable
    # subt = load_generic_table(ms_minimal_required.fname, "FEED")
    return subt


@pytest.fixture(scope="session")
def generic_feed_xds_min(ms_minimal_required):
    """A generic feed xds (loaded form MSv2 mostly as is), loaded from the MS/FEED subtable"""
    from xradio.measurement_set._utils._msv2._tables.read import load_generic_table

    generic_feed_xds = load_generic_table(ms_minimal_required.fname, "FEED")
    return generic_feed_xds


@pytest.fixture(scope="session")
def observation_xds_min(cds_minimal_required):
    """An observation xds, loaded from the minimal MS/OBSERVATION subtable"""

    subt = cds_minimal_required["metainfo"]["observation"]
    # Or alternatively, from ms_minimal_required read subtable
    # subt = load_generic_table(ms_minimal_required.fname, "OBSERVATION")
    return subt


@pytest.fixture(scope="session")
def generic_observation_xds_min(ms_minimal_required):
    """A generic observation xds (loaded form MSv2 mostly as is), loaded from the minimal MS/OBSERVATION subtable"""

    from xradio.measurement_set._utils._msv2._tables.read import load_generic_table

    subt = load_generic_table(ms_minimal_required.fname, "OBSERVATION")
    return subt


@pytest.fixture(scope="session")
def generic_polarization_xds_min(ms_minimal_required):
    """A pol xds, loaded from the minimal MS/POLAIZATION subtable"""
    from xradio.measurement_set._utils._msv2._tables.read import load_generic_table

    subt = load_generic_table(ms_minimal_required.fname, "POLARIZATION")
    return subt


@pytest.fixture(scope="session")
def generic_source_xds_min(ms_minimal_required):
    """A generic source xds (loaded form MSv2 mostly as is), loaded from the minimal MS/SOURCE subtable"""

    from xradio.measurement_set._utils._msv2._tables.read import load_generic_table

    subt = load_generic_table(ms_minimal_required.fname, "SOURCE")
    return subt


@pytest.fixture(scope="session")
def field_and_source_xds_min(ms_minimal_required):
    """A field_and_source_xds (no ephemeris), loaded from the minimal MS/FIELD+SOURCE subtables"""

    field_and_source_xds, source_id, num_lines = create_field_and_source_xds(
        ms_minimal_required.fname,
        np.arange(0, 1),
        0,
        np.arange(0, 1),
        False,
        (0, 1e10),
        xr.DataArray(),
    )

    return field_and_source_xds


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
def field_and_source_xds_min(msv4_xdt_min, msv4_min_path):
    """An MSv4 secondary field and source dataset ('field_and_source_xds')"""

    field_and_source_xds = msv4_xdt_min["field_and_source_base_xds"]

    yield field_and_source_xds


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


@pytest.fixture(scope="session")
def weather_xds_min(msv4_xdt_min, msv4_min_path):
    """An MSv4 secondary weather dataset ('weather_xds')"""

    weather_xds = msv4_xdt_min["weather"]

    yield weather_xds


# TODO: more differentiated custom MSs, consider @pytest.mark.ms_custom_spec({...})
@pytest.fixture(scope="session")
def ms_custom(spec):
    name = "test_ms_custom.ms"  # + rnd
    gen_test_ms(name, spec)
    yield name
    shutil.rmtree(name)


@pytest.fixture(scope="session")
def main_xds_min(ms_minimal_required):
    """A main xds (one partition, when partitioning by intent"""
    from xradio.measurement_set._utils._msv2._tables.read_main_table import (
        read_expanded_main_table,
    )

    # Alternatively:
    # cds = read_ms(ms_minimal_required.fname, partition_scheme="intent")
    # part_key = (0, 0, "scan_intent#subscan_intent")
    # part = cds.partitions[part_key]
    part, _part_ids, attrs = read_expanded_main_table(
        ms_minimal_required.fname, 0, (1, 0)
    )
    part.attrs = attrs

    quick_fix_ndarray_shape_attrs(part)

    yield part


@pytest.fixture(scope="session")
def main_xds_flat_min(ms_minimal_required):
    """A "flat" (row dim) main xds (one partition, when partitioning by ddi)"""
    from xradio.measurement_set._utils._msv2._tables.read_main_table import (
        read_flat_main_table,
    )
    from xradio.measurement_set._utils._msv2.msv2_msv3 import ignore_msv2_cols

    xds, _part_ids, _attrs = read_flat_main_table(
        ms_minimal_required.fname, 0, ignore_msv2_cols=ignore_msv2_cols
    )

    yield xds


@pytest.fixture(scope="session")
def cds_minimal_required(
    ms_minimal_required, main_xds_flat_min, generic_antenna_xds_min
):
    """a simple cds data structure (should be read from an MS)"""
    from xradio.measurement_set._utils._msv2._tables.write_exp_api import (
        vis_xds_packager_mxds,
    )

    subts = [
        ("antenna", generic_antenna_xds_min),
        ("field", xr.Dataset()),
        ("spectral_window", xr.Dataset()),
        ("polarization", xr.Dataset()),
        ("feed", xr.Dataset()),
        ("source", xr.Dataset()),
    ]
    parts = {"0": main_xds_flat_min}
    cds = vis_xds_packager_mxds(parts, subts, True)

    for _key, part in cds.attrs["partitions"].items():
        quick_fix_ndarray_shape_attrs(part)

    yield cds


def quick_fix_ndarray_shape_attrs(part):
    """
    Crude fix for unsupported attrs => update/extend attrs dict filters.
    Shape attributes which take ndarray type values, added through python-casacore,
    "fix" for the experimental write MS , "UVW"]: (but UVW was expected, from CASA
    tests MSs)
    """
    for col in ["DATA", "CORRECTED_DATA", "MODEL_DATA"]:
        if (
            "other" in part.attrs
            and col in part.attrs["other"]["msv2"]["ctds_attrs"]["column_descriptions"]
            and "shape"
            in part.attrs["other"]["msv2"]["ctds_attrs"]["column_descriptions"][col]
        ):
            part.attrs["other"]["msv2"]["ctds_attrs"]["column_descriptions"][col].pop(
                "shape"
            )
