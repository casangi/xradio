import numpy as np
import pytest
import shutil

import xarray as xr


def test_partition_when_empty_state(ms_empty_required):
    from xradio.measurement_set._utils._msv2.partition_queries import (
        partition_when_empty_state,
    )

    res = partition_when_empty_state(ms_empty_required.fname)

    exp = (np.array([], dtype=np.int32), [], [])
    assert res
    assert np.allclose(res[0], exp[0])
    assert res[1] == exp[1]
    assert res[2] == exp[2]


def test_find_distinct_obs_mode(ms_minimal_required):
    from xradio.measurement_set._utils._msv2.partition_queries import (
        find_distinct_obs_mode,
    )

    with pytest.raises(RuntimeError, match="Error in TaQL command"):
        res = find_distinct_obs_mode(ms_minimal_required.fname, "name::STATE")
        assert res


@pytest.mark.parametrize(
    "mixed_intents_no_wvr, expected_results",
    [
        (
            "OBSERVE_TARGET#ON_SOURCE,POSITION_SWITCH",
            ["OBSERVE_TARGET#ON_SOURCE,POSITION_SWITCH"],
        ),
        ("OBSERVE_TARGET#ON_SOURCE", ["OBSERVE_TARGET#ON_SOURCE"]),
    ],
)
def test_filter_intents_per_ddi(mixed_intents_no_wvr, expected_results):
    from xradio.measurement_set._utils._msv2.partition_queries import (
        filter_intents_per_ddi,
    )

    spw_name_by_ddi = {
        0: "WVR#NOMINAL",
        1: "ALMA_R_B03#BB_1#SW-01#FULL_RES",
    }

    res = filter_intents_per_ddi([0], "WVR", mixed_intents_no_wvr, spw_name_by_ddi)
    assert res == expected_results


@pytest.mark.parametrize(
    "mixed_intents_wvr, expected_results",
    [
        ("CALIBRATE#BANDPASS,CALIBRATE#WVR", ["CALIBRATE#WVR"]),
        (
            "CALIBRATE_ATMOSPHERE#OFF_SOURCE,CALIBRATE_WVR#OFF_SOURCE",
            ["CALIBRATE_WVR#OFF_SOURCE"],
        ),
    ],
)
def test_filter_intents_per_ddi_wvr(mixed_intents_wvr, expected_results):
    from xradio.measurement_set._utils._msv2.partition_queries import (
        filter_intents_per_ddi,
    )

    spw_name_by_ddi = {
        0: "WVR#NOMINAL",
        1: "ALMA_R_B03#BB_1#SW-01#FULL_RES",
        2: "ALMA_R_B03#BB_1#SW-01#CH_AVG",
        3: "ALMA_R_B03#BB_2#SW-01#FULL_RES",
        4: "ALMA_R_B03#BB_2#SW-01#CH_AVG",
        5: "ALMA_R_B03#BB_3#SW-01#FULL_RES",
        6: "ALMA_R_B03#BB_3#SW-01#CH_AVG",
        7: "ALMA_R_B03#BB_4#SW-01#FULL_RES",
        8: "ALMA_R_B03#BB_4#SW-01#CH_AVG",
    }

    res = filter_intents_per_ddi([0], "WVR", mixed_intents_wvr, {})
    assert res == [mixed_intents_wvr]

    res = filter_intents_per_ddi([0], "WVR", mixed_intents_wvr, spw_name_by_ddi)
    assert res == expected_results


def test_make_ddi_state_intent_lists():
    from xradio.measurement_set._utils._msv2.partition_queries import (
        make_ddi_state_intent_lists,
    )

    res = make_ddi_state_intent_lists(None, None, np.empty(0), {})

    exp = ([], [], [])
    assert res
    assert res == exp


def test_make_partition_ids_by_ddi_intent_empty(ms_empty_required):
    from xradio.measurement_set._utils._msv2.partition_queries import (
        make_partition_ids_by_ddi_intent,
    )

    res = make_partition_ids_by_ddi_intent(ms_empty_required.fname, xr.Dataset())

    exp = (np.array([], dtype=np.int32), [], [], [])
    assert res
    assert np.allclose(res[0], exp[0])
    assert res[1] == exp[1]
    assert res[2] == exp[2]
    assert res[3] == exp[3]


def test_make_partition_ids_by_ddi_intent_min(ms_minimal_required):
    from xradio.measurement_set._utils._msv2.partition_queries import (
        make_partition_ids_by_ddi_intent,
    )

    res = make_partition_ids_by_ddi_intent(ms_minimal_required.fname, xr.Dataset())

    nddis = ms_minimal_required.descr["nddis"]
    exp = (
        np.arange(nddis, dtype=np.int32),
        nddis * [None],
        nddis * [np.array([0, 1])],
        nddis * ["scan_intent#subscan_intent"],
    )

    assert res
    assert np.allclose(res[0], exp[0])
    assert res[1] == exp[1]
    assert len(res[2]) == nddis
    # assert not (res[2][0] - exp[2][0]).any()
    assert np.allclose(res[2][0], exp[2][0])
    assert res[3] == exp[3]


@pytest.mark.uses_download
def test_make_partition_ids_by_ddi_intent_antennae(ms_alma_antennae_north_split):
    from xradio.measurement_set._utils._msv2.partition_queries import (
        make_partition_ids_by_ddi_intent,
    )

    res = make_partition_ids_by_ddi_intent(
        ms_alma_antennae_north_split.fname, xr.Dataset()
    )

    nddis = 1
    exp = (
        np.arange(nddis),
        2 * nddis * [None],
        np.stack((np.arange(22, 38), np.arange(38, 54)), axis=0),
        [
            "OBSERVE_TARGET#ON_SOURCE",
            "OBSERVE_TARGET#ON_SOURCE,CALIBRATE_WVR#ON_SOURCE",
        ],
    )
    assert res
    assert np.allclose(res[0], exp[0])
    assert res[1] == exp[1]
    assert np.allclose(res[2], exp[2])
    assert res[3] == exp[3]
