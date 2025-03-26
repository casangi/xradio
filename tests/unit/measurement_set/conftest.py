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


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "uses_download: marks tests that use the function to download test MSs (require medium-size downloads and "
        "tend to be slower than others)",
    )


@pytest.fixture(scope="session")
def essential_subtables():
    """
    The set of MS subtables (loaded as sub-xdss) without which we cannot read
    an MS.
    """
    return {"antenna", "spectral_window", "polarization"}


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
    make_ms_empty(name)
    yield MSWithSpec(name, {})
    shutil.rmtree(name)


@pytest.fixture(scope="session")
def ms_minimal_required():
    name = "test_msv2_minimal_required.ms"
    spec = gen_test_ms(name, required_only=True)
    yield MSWithSpec(name, spec)
    shutil.rmtree(name)


@pytest.fixture(scope="session")
def ms_minimal_dims1_required():
    """
    An MS populated minimally, with size one for several relevant dimensions:
    observation, field, scan, spw, etc.
    """
    name = "test_msv2_minimal_dims1_required.ms"
    spec = gen_test_ms(name)
    yield MSWithSpec(name, spec)
    shutil.rmtree(name)


@pytest.fixture(scope="session")
def ms_tab_nonexistent():
    name = "test_nonexistent_table_from_test_table_exists.foo.bar.tab"
    yield MSWithSpec(name, {})


@pytest.fixture(scope="session")
def ms_minimal_for_writes():
    """MS to be used to write subtables inside"""
    name = "test_msv2_minimal_required_for_writes.ms"
    spec = gen_test_ms(name, required_only=True)
    yield MSWithSpec(name, spec)
    shutil.rmtree(name)


@pytest.fixture(scope="session")
def vis_zarr_empty():
    """
    An empty zarr dataset
    """
    name = "test_cor_zarr_empty.zarr"
    xds = xr.Dataset()
    xds.to_zarr(name)
    yield name
    shutil.rmtree(name)


@pytest.fixture(scope="session")
def ddi_xds_min(ms_minimal_required):
    """A DATA_DESCRIPTION/DDI xds, loaded from the minimal MS"""
    from xradio.measurement_set._utils._msv2._tables.read import load_generic_table

    # not available:
    # subt = cds_minimal_required.metainfo["ddi"]

    subt = load_generic_table(ms_minimal_required.fname, "DATA_DESCRIPTION")
    return subt


@pytest.fixture(scope="session")
def spw_xds_min(cds_minimal_required):  # ms_minimal_required):
    """An SPW xds, loaded from the minimal MS/SPECTRAL_WINDOW subtable"""

    subt = cds_minimal_required.metainfo["spectral_window"]
    # Or alternatively, from ms_minimal_required read subtable
    # subt = load_generic_table(ms_minimal_required.fname, "SPECTRAL_WINDOW")
    return subt


@pytest.fixture(scope="session")
def pol_xds_min(cds_minimal_required):
    """A pol xds, loaded from the minimal MS/POLAIZATION subtable"""

    subt = cds_minimal_required.metainfo["polarization"]
    # Or alternatively, from ms_minimal_required read subtable
    # subt = load_generic_table(ms_minimal_required.fname, "POLARIZATION")
    return subt


@pytest.fixture(scope="session")
def ant_xds_min(cds_minimal_required):
    """An antenna xds, loaded from the minimal MS/ANTENNA subtable"""

    subt = cds_minimal_required.metainfo["antenna"]
    # Or alternatively, from ms_minimal_required read subtable
    # subt = load_generic_table(ms_minimal_required.fname, "ANTENNA")
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

    subt = cds_minimal_required.metainfo["field"]
    # Or alternatively, from ms_minimal_required read subtable
    # subt = load_generic_table(ms_minimal_required.fname, "FIELD")
    return subt


@pytest.fixture(scope="session")
def feed_xds_min(cds_minimal_required):
    """A feed xds, loaded from the minimal MS/FEED subtable"""

    subt = cds_minimal_required.metainfo["feed"]
    # Or alternatively, from ms_minimal_required read subtable
    # subt = load_generic_table(ms_minimal_required.fname, "FEED")
    return subt


@pytest.fixture(scope="session")
def observation_xds_min(cds_minimal_required):
    """An observation xds, loaded from the minimal MS/OBSERVATION subtable"""

    subt = cds_minimal_required.metainfo["observation"]
    # Or alternatively, from ms_minimal_required read subtable
    # subt = load_generic_table(ms_minimal_required.fname, "OBSERVATION")
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


def quick_fix_ndarray_shape_attrs(part):
    """
    Crude fix for unsupported attrs => update/extend attrs dict filters.
    Shape attributes which take ndarray type values, added through python-casacore,
    "fix" for the experimental write MS , "UVW"]: (but UVW was expected, from CASA
    tests MSs)
    """
    for col in ["DATA", "CORRECTED_DATA", "MODEL_DATA"]:
        if (
            col in part.attrs["other"]["msv2"]["ctds_attrs"]["column_descriptions"]
            and "shape"
            in part.attrs["other"]["msv2"]["ctds_attrs"]["column_descriptions"][col]
        ):
            part.attrs["other"]["msv2"]["ctds_attrs"]["column_descriptions"][col].pop(
                "shape"
            )


@pytest.fixture(scope="session")
def main_xds_min(ms_minimal_required):
    """A main xds (one partition, when partitioning by intent"""
    # from xradio.measurement_set._utils.msv2 import read_ms

    # cds = read_ms(ms_minimal_required.fname, partition_scheme="intent")
    # part_key = (0, 0, "scan_intent#subscan_intent")
    # part = cds.partitions[part_key]

    # quick_fix_ndarray_shape_attrs(part)

    part = xr.Dataset()

    yield part


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
def msv4_min_correlated_xds(
    ms_minimal_required, processing_set_min_path, msv4_min_path
):
    """An MSv4 main xds (correlated data, one partition) xds"""
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

    xds = xr.open_dataset(
        msv4_min_path + "/correlated_xds",
        engine="zarr",
    )

    yield xds
    shutil.rmtree(processing_set_min_path)


@pytest.fixture(scope="session")
def antenna_xds_min(msv4_min_correlated_xds, msv4_min_path):
    """An MSv4 secondary antenna dataset ('antenna_xds')"""

    antenna_xds = xr.open_dataset(
        msv4_min_path + "/antenna_xds",
        engine="zarr",
    )

    yield antenna_xds


@pytest.fixture(scope="session")
def field_and_source_xds_min(msv4_min_correlated_xds, msv4_min_path):
    """An MSv4 secondary field and source dataset ('field_and_source_xds')"""

    field_and_source_xds = xr.open_dataset(
        msv4_min_path + "/field_and_source_xds_base",
        engine="zarr",
    )

    yield field_and_source_xds


@pytest.fixture(scope="session")
def pointing_xds_min(msv4_min_correlated_xds, msv4_min_path):
    """An MSv4 secondary pointing dataset ('pointing_xds')"""

    pointing_xds = xr.open_dataset(
        msv4_min_path + "/pointing_xds",
        engine="zarr",
    )

    yield pointing_xds


@pytest.fixture(scope="session")
def sys_cal_xds_min(msv4_min_correlated_xds, msv4_min_path):
    """An MSv4 secondary sys cal dataset ('system_calibration_xds')"""

    syscal_xds = xr.open_dataset(
        msv4_min_path + "/system_calibration_xds",
        engine="zarr",
    )

    yield syscal_xds


@pytest.fixture(scope="session")
def weather_xds_min(msv4_xds_min, msv4_min_path):
    """An MSv4 secondary weather dataset ('weather_xds')"""

    weather_xds = xr.open_dataset(
        msv4_min_path + "/system_calibration_xds",
        engine="zarr",
    )

    yield weather_xds


@pytest.fixture(scope="session")
def cds_minimal_required(ms_minimal_required):
    """a simple cds data structure read from an MS (also a fixture defined here)"""
    # from xradio.measurement_set._utils.msv2 import read_ms
    from xradio.measurement_set._utils._utils.cds import CASAVisSet

    # cds = read_ms(ms_minimal_required.fname)
    cds = CASAVisSet(
        {
            "spectral_window": xr.Dataset(),
            "polarization": xr.Dataset(),
            "antenna": xr.Dataset(),
            "field": xr.Dataset(),
            "feed": xr.Dataset(),
            "source": xr.Dataset(),
        },
        {},
        {},
    )

    for _key, part in cds.partitions.items():
        quick_fix_ndarray_shape_attrs(part)

    yield cds


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


# TODO: more differentiated custom MSs, consider @pytest.mark.ms_custom_spec({...})
@pytest.fixture(scope="session")
def ms_custom(spec):
    name = "test_ms_custom.ms"  # + rnd
    gen_test_ms(name, spec)
    yield name
    shutil.rmtree(name)


@pytest.fixture(scope="session")
def ms_alma_antennae_north_split():
    """
    An MS that is downloaded (one of the smallest), with multiple fields (3)
    + with SOURCE and STATE populated
    """
    from toolviper.utils.data import download

    name = "Antennae_North.cal.lsrk.split.ms"
    # name = "small_meerkat.ms"
    download(file=name)
    # TODO: extend with more attrs
    descr = {"nchans": 8, "npols": 2}
    yield MSWithSpec(name, descr)


@pytest.fixture(scope="session")
def ms_as_zarr_min():
    """An MS loaded and then saved to zarr format"""
    name = "xds_saved_as_zarr_bogus_for_now.zarr"
    yield name
