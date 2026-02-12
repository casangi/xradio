import numpy as np
import pandas as pd

import pytest

import pyasdm

from xradio.measurement_set._utils._asdm.create_partitions import (
    create_partitions,
    finalize_partitions_groupby,
)


def add_main_table(asdm: pyasdm.ASDM):
    main_row_0_xml = """
  <row>
    <time> 5230000651200000000 </time>
    <numAntenna> 2 </numAntenna>
    <timeSampling>INTEGRATION</timeSampling>
    <interval> 24192000000 </interval>
    <numIntegration> 1512 </numIntegration>
    <scanNumber> 1 </scanNumber>
    <subscanNumber> 1 </subscanNumber>
    <dataSize> 1681962 </dataSize>
    <dataUID>
      <EntityRef entityId="uid://A002/X11b94a6/X119f" partId="X00000000" entityTypeName="Main" documentVersion="1"/>
    </dataUID>
    <configDescriptionId> ConfigDescription_0 </configDescriptionId>
    <execBlockId> ExecBlock_0 </execBlockId>
    <fieldId> Field_0 </fieldId>
    <stateId> 1 12 State_0 State_0 State_0 State_0 State_0 State_0 State_0 State_0 State_0 State_0 State_0 State_0  </stateId>
  </row>
    """
    main_table = asdm.getMain()
    main_row_0 = pyasdm.MainRow(main_table)
    main_row_0.setFromXML(main_row_0_xml)
    main_table.add(main_row_0)


def add_config_description_table(asdm: pyasdm.ASDM):
    config_description_row_0_xml = """
  <row>
    <configDescriptionId> ConfigDescription_0 </configDescriptionId>
    <numAntenna> 2 </numAntenna>
    <numDataDescription> 4 </numDataDescription>
    <numFeed> 1 </numFeed>
    <correlationMode>AUTO_ONLY</correlationMode>
    <numAtmPhaseCorrection> 1 </numAtmPhaseCorrection>
    <atmPhaseCorrection> 1 1 AP_UNCORRECTED</atmPhaseCorrection>
    <processorType>RADIOMETER</processorType>
    <spectralType>BASEBAND_WIDE</spectralType>
    <antennaId> 1 12 Antenna_0 Antenna_1 Antenna_2 Antenna_3 Antenna_4 Antenna_5 Antenna_6 Antenna_7 Antenna_8 Antenna_9 Antenna_10 Antenna_11  </antennaId>
    <dataDescriptionId> 1 4 DataDescription_0 DataDescription_1 DataDescription_2 DataDescription_3  </dataDescriptionId>
    <feedId> 1 12 0 0 0 0 0 0 0 0 0 0 0 0  </feedId>
    <processorId> Processor_0 </processorId>
    <switchCycleId> 1 4 SwitchCycle_0 SwitchCycle_0 SwitchCycle_0 SwitchCycle_0  </switchCycleId>
  </row>
    """
    config_description_table = asdm.getConfigDescription()
    config_description_row_0 = pyasdm.ConfigDescriptionRow(config_description_table)
    config_description_row_0.setFromXML(config_description_row_0_xml)
    config_description_table.add(config_description_row_0)


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
    asdm_with_spw_simple, monkeypatch
):

    add_main_table(asdm_with_spw_simple)
    add_config_description_table(asdm_with_spw_simple)

    monkeypatch.setattr(
        "pyasdm.MainRow.getBDFPath", lambda bdf_paths: "/monkypatched_path/foo"
    )
    create_partitions(
        asdm_with_spw_simple,
        ["ExecBlockId"],
        include_processor_types=["CORRELATOR", "SPECTROMETER", "RADIOMETER"],
        include_spectral_resolution_types=[
            "CHANNEL_AVERAGE",
            "BASEBAND_WIDE",
            "FULL_RESOLUTION",
        ],
    )


def test_create_partitions_with_filter_on_processor_type_asdm_with_spw_simple(
    asdm_with_spw_simple, monkeypatch
):

    add_main_table(asdm_with_spw_simple)
    monkeypatch.setattr(
        "pyasdm.MainRow.getBDFPath", lambda bdf_paths: "/monkypatched_path/foo"
    )
    with pytest.raises(RuntimeError, match="left after filtering processor types"):
        create_partitions(
            asdm_with_spw_simple,
            ["fieldId"],
            include_processor_types=["SPECTROMETER"],
        )


def test_create_partitions_with_filter_on_spectral_resolution_type_asdm_with_spw_simple(
    asdm_with_spw_simple, monkeypatch
):

    add_main_table(asdm_with_spw_simple)
    monkeypatch.setattr(
        "pyasdm.MainRow.getBDFPath", lambda bdf_paths: "/monkypatched_path/foo"
    )
    with pytest.raises(
        RuntimeError, match="left after filtering spectral resolution types"
    ):
        create_partitions(
            asdm_with_spw_simple,
            ["fieldId"],
            include_spectral_resolution_types=["FULL_RESOLUTION"],
        )


def test_finalize_partitions_groupby():
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
                "processorType": {0: "RADIOMETER", 1: "RADIOMETER", 1: "CORRELATOR"},
                "spectralType": {
                    0: "BASEBAND_WIDE",
                    1: "BASEBAND_WIDE",
                    1: "FULL_RESOLUTION",
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
