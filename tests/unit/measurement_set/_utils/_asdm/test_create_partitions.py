import numpy as np
import pandas as pd

import pytest

from xradio.measurement_set._utils._asdm.create_partitions import (
    create_partitions,
    finalize_partitions_groupby,
)


def test_create_partitions_empty():
    with pytest.raises(AttributeError, match="no attribute"):
        create_partitions(None, ["fieldId"])


def test_create_partitions_asdm_empty(asdm_empty):
    partitions = create_partitions(asdm_empty, ["fieldId"])
    assert len(partitions) == 0


def test_create_partitions_asdm_with_spw_default(asdm_with_spw_default):
    partitions = create_partitions(asdm_with_spw_default, ["fieldId"])
    assert len(partitions) == 0


def test_create_partitions_asdm_with_spw_simple(asdm_with_spw_simple, monkeypatch):
    monkeypatch.setattr(
        "pyasdm.MainRow.getBDFPath", lambda bdf_paths: "/monkypatched_path/foo"
    )
    partitions = create_partitions(asdm_with_spw_simple, ["fieldId"])
    assert len(partitions) == 0


def test_create_partitions_with_includes_asdm_with_spw_simple(
    asdm_with_main_config, monkeypatch
):

    monkeypatch.setattr(
        "pyasdm.MainRow.getBDFPath", lambda bdf_paths: "/monkypatched_path/foo"
    )
    create_partitions(
        asdm_with_main_config,
        ["ExecBlockId"],
        include_processor_types=["CORRELATOR", "SPECTROMETER", "RADIOMETER"],
        include_spectral_resolution_types=[
            "CHANNEL_AVERAGE",
            "BASEBAND_WIDE",
            "FULL_RESOLUTION",
        ],
    )


def test_create_partitions_with_filter_on_processor_type_asdm_with_spw_simple(
    asdm_with_main, monkeypatch
):

    monkeypatch.setattr(
        "pyasdm.MainRow.getBDFPath", lambda bdf_paths: "/monkypatched_path/foo"
    )
    with pytest.raises(RuntimeError, match="left after filtering processor types"):
        create_partitions(
            asdm_with_main,
            ["fieldId"],
            include_processor_types=["SPECTROMETER"],
        )


def test_create_partitions_with_filter_on_spectral_resolution_type_asdm_with_spw_simple(
    asdm_with_main, monkeypatch
):

    monkeypatch.setattr(
        "pyasdm.MainRow.getBDFPath", lambda bdf_paths: "/monkypatched_path/foo"
    )
    with pytest.raises(
        RuntimeError, match="left after filtering spectral resolution types"
    ):
        create_partitions(
            asdm_with_main,
            ["fieldId"],
            include_spectral_resolution_types=["FULL_RESOLUTION"],
        )


def test_finalize_partitions_groupby_empty():
    with pytest.raises(TypeError, match="scalar index"):
        finalize_partitions_groupby(
            pd.DataFrame([[0, 0]], columns=["fieldId", "scanIntent"]),
            ["fieldId"],
            [0],
        )


@pytest.mark.parametrize(
    "partitioning_dict, expected_parts_len",
    [
        (
            {
                "time": {0: pd.Timestamp("2024-08-10 09:57:31.200000048")},
                "fieldId": {0: 0},
                "configDescriptionId": {0: 0},
                "scanNumber": {0: 1},
                "subscanNumber": {0: 1},
                "stateId": {0: 0},
                "dataUID": {0: "uid://A002/X11b94a6/X119f"},
                "BDFPath": {0: "/monkypatched_path/foo"},
                "execBlockId": {0: 0},
                "dataDescriptionId": {0: 0},
                "processorType": {0: "RADIOMETER"},
                "spectralType": {0: "BASEBAND_WIDE"},
                "spectralWindowId": {0: 0},
                "polOrHoloId": {0: 0},
                "sourceId": {0: 0},
                "scanIntent": {0: 0},
            },
            1,
        ),
        (
            {
                "time": {
                    0: pd.Timestamp("2024-08-10 09:57:31.200000048"),
                    1: pd.Timestamp("2024-08-10 09:57:32.200000048"),
                },
                "fieldId": {0: 0, 1: 0},
                "configDescriptionId": {0: 0, 1: 0},
                "scanNumber": {0: 1, 1: 1},
                "subscanNumber": {0: 1, 1: 1},
                "stateId": {0: 0, 1: 0},
                "dataUID": {
                    0: "uid://A002/X11b94a6/X119f",
                    1: "uid://A002/X11b94a6/X11a0",
                },
                "BDFPath": {0: "/monkypatched_path/foo", 1: "/monkypatched_path/bar"},
                "execBlockId": {0: 0, 1: 0},
                "dataDescriptionId": {0: 0, 1: 0},
                "processorType": {0: "RADIOMETER", 1: "RADIOMETER"},
                "spectralType": {0: "BASEBAND_WIDE", 1: "BASEBAND_WIDE"},
                "spectralWindowId": {0: 0, 1: 1},
                "polOrHoloId": {0: 0, 1: 0},
                "sourceId": {0: 0, 1: 0},
                "scanIntent": {0: 0, 1: 0},
            },
            1,
        ),
        (
            {
                "time": {
                    0: pd.Timestamp("2024-08-10 09:57:31.200000048"),
                    1: pd.Timestamp("2024-08-10 09:57:32.200000048"),
                    2: pd.Timestamp("2024-08-10 09:57:33.200000048"),
                },
                "fieldId": {0: 0, 1: 0, 2: 0},
                "configDescriptionId": {0: 0, 1: 0, 2: 0},
                "scanNumber": {0: 1, 1: 1, 2: 1},
                "subscanNumber": {0: 1, 1: 2, 2: 3},
                "stateId": {0: 0, 1: 0, 2: 0},
                "dataUID": {
                    0: "uid://A002/X11b94a6/X119f",
                    1: "uid://A002/X11b94a6/X11a0",
                    2: "uid://A002/X11b94a6/X11a1",
                },
                "BDFPath": {
                    0: "/monkypatched_path/foo",
                    1: "/monkypatched_path/bar",
                    2: "/monkypatched_path/baz",
                },
                "execBlockId": {0: 0, 1: 0, 2: 0},
                "dataDescriptionId": {0: 0, 1: 0, 2: 0},
                "processorType": {0: "RADIOMETER", 1: "RADIOMETER", 2: "CORRELATOR"},
                "spectralType": {
                    0: "BASEBAND_WIDE",
                    1: "BASEBAND_WIDE",
                    2: "FULL_RESOLUTION",
                },
                "spectralWindowId": {0: 0, 1: 1, 2: 2},
                "polOrHoloId": {0: 0, 1: 0, 2: 1},
                "sourceId": {0: 0, 1: 0, 2: 0},
                "scanIntent": {0: 0, 1: 0, 2: 1},
            },
            3,
        ),
    ],
)
def test_finalize_partitions_groupby(partitioning_dict, expected_parts_len):
    partitioning_df = pd.DataFrame.from_dict(partitioning_dict)

    parts = finalize_partitions_groupby(
        partitioning_df,
        ["fieldId", "scanNumber", "subscanNumber"],
        np.array(
            [
                ["CALIBRATE_POINTING", "CALIBRATE_WVR"],
                ["CALIBRATE_ATMOSPHERE", "CALIBRATE_WVR"],
            ]
        ),
    )
    assert isinstance(parts, list)
    assert len(parts) == expected_parts_len
