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


def test_create_info_dicts_with_asdm_simple_extended(
    asdm_with_main_execblock_config_processor_sbsummary,
):

    # Only field from the partition dict needed here is configDescriptionId
    info_dicts = create_info_dicts(
        asdm_with_main_execblock_config_processor_sbsummary,
        xr.Dataset(),
        {"configDescriptionId": [0]},
    )
    assert isinstance(info_dicts, dict)
    assert "observation_info" in info_dicts
    assert info_dicts["observation_info"] == {
        "observer": ["riechers"],
        "release_date": "",
        "project_UID": "uid://A001/X35fd/X21f",
        "execution_block_UID": "uid://A002/X11b94a6/X119b",
        "session_reference_UID": "uid://A002/X11b94a6/X119a",
        "observing_log": "[]",
        "scheduling_block_UID": "u",
    }
    assert "processor_info" in info_dicts
    assert info_dicts["processor_info"] == {
        "type": "RADIOMETER",
        "sub_type": "SQUARE_LAW_DETECTOR",
    }


def test_create_observation_info_with_asdm_simple_extended(
    asdm_with_main_execblock_config_processor_sbsummary,
):

    # Only field from the partition dict needed here is configDescriptionId
    observation_info = create_observation_info(
        asdm_with_main_execblock_config_processor_sbsummary,
        {"configDescriptionId": [0]},
    )
    assert isinstance(observation_info, dict)
    assert observation_info == {
        "observer": ["riechers"],
        "release_date": "",
        "project_UID": "uid://A001/X35fd/X21f",
        "execution_block_UID": "uid://A002/X11b94a6/X119b",
        "session_reference_UID": "uid://A002/X11b94a6/X119a",
        "observing_log": "[]",
        "scheduling_block_UID": "u",
    }


def test_create_processor_info_with_asdm_simple_extended(
    asdm_with_main_execblock_config_processor_sbsummary,
):

    # Only field from the partition dict needed here is configDescriptionId
    processor_info = create_processor_info(
        asdm_with_main_execblock_config_processor_sbsummary,
        {"configDescriptionId": [0]},
    )
    assert isinstance(processor_info, dict)
    assert processor_info == {
        "type": "RADIOMETER",
        "sub_type": "SQUARE_LAW_DETECTOR",
    }
