import xarray as xr

import pytest

from xradio.measurement_set._utils._asdm.create_info_dicts import (
    create_info_dicts,
    create_processor_info,
    create_observation_info,
)


def test_create_info_dicts_with_asdm_empty(asdm_empty):
    with pytest.raises(IndexError, match="out of range"):
        create_info_dicts(asdm_empty, xr.Dataset(), {"fieldId": [0]})


def test_create_info_dicts_with_asdm_default(asdm_with_spw_default):
    with pytest.raises(IndexError, match="out of range"):
        create_info_dicts(asdm_with_spw_default, xr.Dataset(), {"fieldId": [0]})


def test_create_info_dicts_with_asdm_simple(asdm_with_spw_simple):
    with pytest.raises(IndexError, match="out of range"):
        create_info_dicts(asdm_with_spw_simple, xr.Dataset(), {"fieldId": [0]})
