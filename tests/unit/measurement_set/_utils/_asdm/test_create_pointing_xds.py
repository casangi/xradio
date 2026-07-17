import pytest

from xradio.measurement_set._utils._asdm.create_pointing_xds import create_pointing_xds


def test_create_pointing_xds_empty():
    with pytest.raises(AttributeError, match="has no attribute"):
        create_pointing_xds(None)


def test_create_pointing_xds_with_asdm_empty(asdm_empty):
    with pytest.raises(IndexError, match="out of bounds"):
        create_pointing_xds(asdm_empty)


def test_create_pointing_xds_with_asdm_default(asdm_with_spw_default):
    with pytest.raises(IndexError, match="out of bounds"):
        create_pointing_xds(asdm_with_spw_default)


def test_create_pointing_xds_with_asdm_simple(asdm_with_spw_simple):
    with pytest.raises(IndexError, match="out of bounds"):
        create_pointing_xds(asdm_with_spw_simple)


def test_create_pointing_xds_with_asdm_simple_pointing(
    # asdm_with_spw_simple_pointing
):

    assert False
