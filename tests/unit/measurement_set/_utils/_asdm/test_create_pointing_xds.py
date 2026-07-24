import pytest

import pyasdm

import xarray as xr

from xradio.measurement_set._utils._asdm.create_pointing_xds import (
    create_pointing_xds,
)
from xradio.measurement_set.schema import PointingXds
from xradio.schema.check import check_dataset


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


def test_create_pointing_xds_with_asdm_antenna_pointing(
    asdm_with_antenna_station_pointing,
):
    pointing_xds = create_pointing_xds(asdm_with_antenna_station_pointing)
    check_dataset(pointing_xds, PointingXds)
    # Make sure for this non-interpolated xds, there is time_pointing
    for coord in ["time_pointing", "antenna_name", "local_sky_dir_label"]:
        assert coord in pointing_xds.coords
    # The set of variables expected in this example
    for data_var in ["DIRECTION", "POINTING_DISH_MEASURED"]:
        assert data_var in pointing_xds.data_vars
