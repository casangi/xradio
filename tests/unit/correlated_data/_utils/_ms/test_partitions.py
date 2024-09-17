import numpy as np
import pytest


def test_make_spw_names_by_ddi():
    from xradio.correlated_data._utils._ms.partitions import make_spw_names_by_ddi
    import xarray

    with pytest.raises(
        AttributeError, match="object has no attribute 'SPECTRAL_WINDOW_ID'"
    ):
        res = make_spw_names_by_ddi(xarray.Dataset(), xarray.Dataset())
        assert res


def test_make_spw_names_by_ddi_min(ddi_xds_min, spw_xds_min):
    from xradio.correlated_data._utils._ms.partitions import make_spw_names_by_ddi

    res = make_spw_names_by_ddi(ddi_xds_min, spw_xds_min)
    assert res
    nddis = 4
    assert np.all(np.arange(nddis) == sorted(res.keys()))
    # exp_names = {0: 'unspecified_test#0', 1: 'unspecified_test#0', 2: 'unspecified_test#0', 3: 'unspecified_test#0', 4: 'unspecified_test#1', 5: 'unspecified_test#1', 6: 'unspecified_test#1', 7: 'unspecified_test#1'}
    exp_names = {
        0: "unspecified_test#0",
        1: "unspecified_test#0",
        2: "unspecified_test#1",
        3: "unspecified_test#1",
    }
    assert res == exp_names


def test_make_spw_names_by_ddi_min_only_ddi(ddi_xds_min):
    from xradio.correlated_data._utils._ms.partitions import make_spw_names_by_ddi
    import xarray

    with pytest.raises(AttributeError, match="object has no attribute 'NAME'"):
        res = make_spw_names_by_ddi(ddi_xds_min, xarray.Dataset())
        assert res


def test_make_spw_names_by_ddi_min_only_spw(spw_xds_min):
    from xradio.correlated_data._utils._ms.partitions import make_spw_names_by_ddi
    import xarray

    with pytest.raises(
        AttributeError, match="object has no attribute 'SPECTRAL_WINDOW_ID'"
    ):
        res = make_spw_names_by_ddi(xarray.Dataset(), spw_xds_min)
        assert res


@pytest.mark.parametrize(
    "intents, expected_results",
    [
        (
            "OBSERVE_TARGET#ON_SOURCE,POSITION_SWITCH",
            {"OBSERVE_TARGET": ["ON_SOURCE"], "POSITION_SWITCH": [""]},
        ),
        ("OBSERVE_TARGET#UNSPECIFIED", {"OBSERVE_TARGET": ["UNSPECIFIED"]}),
        (
            "CALIBRATE_DELAY#ON_SOURCE,CALIBRATE_PHASE#ON_SOURCE",
            {"CALIBRATE_DELAY": ["ON_SOURCE"], "CALIBRATE_PHASE": ["ON_SOURCE"]},
        ),
        (
            "CALIBRATE_ATMOSPHERE#OFF_SOURCE,CALIBRATE_ATMOSPHERE#ON_SOURCE,CALIBRATE_WVR#OFF_SOURCE,CALIBRATE_WVR#ON_SOURCE",
            {
                "CALIBRATE_ATMOSPHERE": ["OFF_SOURCE", "ON_SOURCE"],
                "CALIBRATE_WVR": ["OFF_SOURCE", "ON_SOURCE"],
            },
        ),
        ("OBSERVE_TARGET.UNSPECIFIED", {"OBSERVE_TARGET": ["UNSPECIFIED"]}),
        (
            "CALIBRATE_DELAY.ON_SOURCE,CALIBRATE_PHASE.ON_SOURCE",
            {"CALIBRATE_DELAY": ["ON_SOURCE"], "CALIBRATE_PHASE": ["ON_SOURCE"]},
        ),
    ],
)
def test_split_intents(intents, expected_results):
    from xradio.correlated_data._utils._ms.partitions import split_intents

    split_intents = split_intents(intents)
    assert split_intents == expected_results


def test_make_part_key():
    from xradio.correlated_data._utils._ms.partitions import make_part_key
    import xarray

    with pytest.raises(KeyError, match="partition_ids"):
        res = make_part_key(xarray.Dataset(), partition_scheme="intent")
        assert res


def test_read_ms_scan_subscan_partitions(ms_empty_required):
    from xradio.correlated_data._utils._ms.partitions import (
        read_ms_scan_subscan_partitions,
    )

    with pytest.raises(
        AttributeError, match="object has no attribute 'SPECTRAL_WINDOW_ID'"
    ):
        res = read_ms_scan_subscan_partitions(ms_empty_required.fname, "intent")
        assert res


def test_read_ms_ddi_partitions(ms_empty_required):
    from xradio.correlated_data._utils._ms.partitions import read_ms_ddi_partitions

    with pytest.raises(AttributeError, match="Dataset' object has no attribute 'row'"):
        res = read_ms_ddi_partitions(ms_empty_required.fname)
        assert res


def test_finalize_partitions():
    from xradio.correlated_data._utils._ms.partitions import finalize_partitions

    res = finalize_partitions({}, {})
    assert res == {}
