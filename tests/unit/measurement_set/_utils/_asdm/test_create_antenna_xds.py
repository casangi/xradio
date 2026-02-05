import pandas as pd
import xarray as xr

import pytest

from xradio.measurement_set._utils._asdm.create_antenna_xds import (
    create_antenna_xds,
    create_feed_xds,
    get_telescope_name,
)


def test_create_antenna_xds_empty():
    with pytest.raises(AttributeError, match="has no attribute"):
        create_antenna_xds(None, 2, 0, xr.DataArray([0]))


def test_create_antenna_xds_with_asdm_empty(asdm_empty):
    with pytest.raises(RuntimeError, match="Issue with telescopeName"):
        create_antenna_xds(asdm_empty, 0, 0, xr.DataArray([0]))


def test_create_antenna_xds_with_asdm_default(asdm_with_spw_default):
    with pytest.raises(RuntimeError, match="antennas found"):
        create_antenna_xds(asdm_with_spw_default, 3, 0, xr.DataArray([0]))


def test_create_antenna_xds_with_asdm_simple(asdm_with_spw_simple):
    with pytest.raises(RuntimeError, match="antennas found"):
        create_antenna_xds(asdm_with_spw_simple, 4, 0, xr.DataArray([0]))


def test_create_feed_xds_empty():
    with pytest.raises(AttributeError, match="has no attribute"):
        create_feed_xds(None, pd.DataFrame(), 0, xr.DataArray([[0]]))


def test_create_feed_xds_with_asdm_empty(asdm_empty):
    with pytest.raises(AttributeError, match="has no attribute"):
        create_feed_xds(asdm_empty, 0, 0, xr.DataArray([[0]]))


def test_create_feed_xds_with_asdm_default(asdm_with_spw_default):
    with pytest.raises(AttributeError, match="has no attribute"):
        create_feed_xds(asdm_with_spw_default, 3, 0, xr.DataArray([[0]]))


def test_create_feed_xds_with_asdm_simple(asdm_with_spw_simple):
    with pytest.raises(AttributeError, match="has no attribute"):
        create_feed_xds(asdm_with_spw_simple, 4, 0, xr.DataArray([[0]]))
