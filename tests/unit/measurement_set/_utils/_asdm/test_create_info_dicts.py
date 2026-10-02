import importlib.util
import sys
from pathlib import Path

import numpy as np
import pyasdm
import pytest
import xarray as xr

from xradio.measurement_set._utils._asdm.create_info_dicts import (
    create_info_dicts,
    create_observation_info,
    create_processor_info,
)
from xradio.measurement_set.schema import ObservationInfoDict, ProcessorInfoDict
from xradio.schema.check import check_dict

# 2024-07-21T04:26:40 UTC as ASDM ArrayTime ns (unix 1721536000 s)
RELEASE_DATE_NS = (1_721_536_000 + 3_506_716_800) * 10**9


def _execblock_xml(
    eb_id,
    eb_uid,
    project_uid,
    observer,
    sb_summary_id=0,
    release_date_ns=None,
    observing_log=(),
):
    release_date = (
        f"<releaseDate> {release_date_ns} </releaseDate>"
        if release_date_ns is not None
        else ""
    )
    log = " ".join(observing_log)
    return f"""<row><execBlockId> ExecBlock_{eb_id} </execBlockId>
<startTime> 5230000552242000000 </startTime><endTime> 5230001040525000000 </endTime>
<execBlockNum> {eb_id + 1} </execBlockNum>
<execBlockUID><EntityRef entityId="{eb_uid}" partId="X00000000" entityTypeName="ASDM" documentVersion="1"/></execBlockUID>
<projectUID><EntityRef entityId="{project_uid}" partId="X00000000" entityTypeName="ObsProject" documentVersion="1"/></projectUID>
<configName> 7M </configName><telescopeName> ALMA </telescopeName><observerName> {observer} </observerName>
<numObservingLog> {len(observing_log)} </numObservingLog><observingLog> 1 {len(observing_log)} {log} </observingLog>
<sessionReference><EntityRef entityId="{eb_uid}0" partId="X00000000" entityTypeName="Session" documentVersion="1"/></sessionReference>
<baseRangeMin> 0.0 </baseRangeMin><baseRangeMax> 0.0 </baseRangeMax><baseRmsMinor> 0.0 </baseRmsMinor>
<baseRmsMajor> 0.0 </baseRmsMajor><basePa> 0.0 </basePa><aborted> false </aborted>
<numAntenna> 2 </numAntenna>{release_date}<observingScript> StandardInterferometry.py </observingScript>
<antennaId> 1 2 Antenna_0 Antenna_1 </antennaId><sBSummaryId> SBSummary_{sb_summary_id} </sBSummaryId></row>"""


def _sbsummary_xml(sb_summary_id, sb_uid):
    return f"""<row><sBSummaryId> SBSummary_{sb_summary_id} </sBSummaryId>
<sbSummaryUID><EntityRef entityId="{sb_uid}" partId="X00000000" entityTypeName="SchedBlock" documentVersion="1"/></sbSummaryUID>
<projectUID><EntityRef entityId="uid://A001/X35fd/X21f" partId="X00000000" entityTypeName="ObsProject" documentVersion="1"/></projectUID>
<obsUnitSetUID><EntityRef entityId="uid://A001/X35fd/X21f" partId="X00000000" entityTypeName="ObsProject" documentVersion="1"/></obsUnitSetUID>
<frequency> {100.0 + sb_summary_id} </frequency><frequencyBand>ALMA_RB_03</frequencyBand><sbType>OBSERVER</sbType>
<sbDuration> 7200000000000 </sbDuration><numObservingMode> 1 </numObservingMode>
<observingMode> 1 1 &quot;Standard Interferometry&quot; </observingMode><numberRepeats> 1 </numberRepeats>
<numScienceGoal> 0 </numScienceGoal><scienceGoal> 1 0 </scienceGoal>
<numWeatherConstraint> 0 </numWeatherConstraint><weatherConstraint> 1 0 </weatherConstraint></row>"""


def _set_row_from_xml(row, xml):
    """
    ``row.setFromXML(xml)`` with boolean attributes (e.g. ExecBlock ``aborted``)
    set to their XML value: pyasdm parses XML booleans with ``bool(text)``, so
    "false" reads as True (see synthetic_asdm.set_row_from_xml).
    """
    name = "xradio_tests_asdm_synthetic_asdm"  # same module object as the conftest
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            name, Path(__file__).with_name("synthetic_asdm.py")
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name].set_row_from_xml(row, xml)


def _add_row(table, row_cls, xml):
    row = row_cls(table)
    _set_row_from_xml(row, xml)
    table.add(row)


@pytest.fixture(scope="module")
def asdm_two_exec_blocks():
    """
    Two ExecBlocks with different UIDs/observers/SBs; EB 1 has a releaseDate,
    an observing log, and refers to a missing SBSummary row.
    """
    asdm = pyasdm.ASDM()
    _add_row(
        asdm.getExecBlock(),
        pyasdm.ExecBlockRow,
        _execblock_xml(0, "uid://A002/X1/X1", "uid://A001/X1/X1", "first_observer"),
    )
    _add_row(
        asdm.getExecBlock(),
        pyasdm.ExecBlockRow,
        _execblock_xml(
            1,
            "uid://A002/X2/X2",
            "uid://A001/X2/X2",
            "second_observer",
            sb_summary_id=5,
            release_date_ns=RELEASE_DATE_NS,
            observing_log=("DA41_out", "weather_ok"),
        ),
    )
    _add_row(
        asdm.getSBSummary(), pyasdm.SBSummaryRow, _sbsummary_xml(0, "uid://A001/X3/X3")
    )
    return asdm


def test_create_info_dicts_with_asdm_empty(asdm_empty):
    with pytest.raises(ValueError, match="execBlockId=0"):
        create_info_dicts(
            asdm_empty,
            xr.Dataset(),
            {"execBlockId": np.array([0]), "configDescriptionId": np.array([0])},
        )


def test_create_info_dicts_with_asdm_default(asdm_with_spw_default):
    with pytest.raises(KeyError, match="execBlockId"):
        create_info_dicts(asdm_with_spw_default, xr.Dataset(), {"fieldId": [0]})


def test_create_info_dicts_with_asdm_simple(asdm_with_execblock_spw_simple):
    """ExecBlock but no ConfigDescription table."""
    with pytest.raises(ValueError, match="configDescriptionId=0"):
        create_info_dicts(
            asdm_with_execblock_spw_simple,
            xr.Dataset(),
            {"execBlockId": [0], "configDescriptionId": [0]},
        )


def test_create_info_dicts_with_asdm_simple_extended(
    asdm_with_main_execblock_config_processor_sbsummary,
):
    info_dicts = create_info_dicts(
        asdm_with_main_execblock_config_processor_sbsummary,
        xr.Dataset(),
        {"execBlockId": np.array([0]), "configDescriptionId": np.array([0])},
    )
    assert isinstance(info_dicts, dict)
    assert "observation_info" in info_dicts
    assert info_dicts["observation_info"] == {
        "observer": ["riechers"],
        "release_date": "",
        "project_UID": "uid://A001/X35fd/X21f",
        "execution_block_UID": "uid://A002/X11b94a6/X119b",
        "session_reference_UID": "uid://A002/X11b94a6/X119a",
        "observing_log": None,
        "scheduling_block_UID": "uid://A001/X362e/X332",
    }
    assert not check_dict(info_dicts["observation_info"], ObservationInfoDict)
    assert "processor_info" in info_dicts
    assert info_dicts["processor_info"] == {
        "type": "RADIOMETER",
        "sub_type": "SQUARE_LAW_DETECTOR",
    }
    assert not check_dict(info_dicts["processor_info"], ProcessorInfoDict)


def test_create_info_dicts_with_asdm_sd_extended(
    asdm_sd_with_main_execblock_config_processor_sbsummary,
):
    info_dicts = create_info_dicts(
        asdm_sd_with_main_execblock_config_processor_sbsummary,
        xr.Dataset(),
        {"execBlockId": [0], "configDescriptionId": [1]},
    )
    assert isinstance(info_dicts, dict)
    assert "observation_info" in info_dicts
    assert info_dicts["observation_info"] == {
        "observer": ["lknee"],
        "release_date": "",
        "project_UID": "uid://A002/X5ca254/X1",
        "execution_block_UID": "uid://A002/Xac5575/X4086",
        "session_reference_UID": "uid://A002/X5ca254/Xb",
        "observing_log": None,
        "scheduling_block_UID": "uid://A002/X5ca254/X3",
    }
    assert "processor_info" in info_dicts
    assert info_dicts["processor_info"] == {
        "type": "CORRELATOR",
        "sub_type": "ALMA_CORRELATOR_MODE",
    }


def test_create_observation_info_with_asdm_simple_extended(
    asdm_with_main_execblock_config_processor_sbsummary,
):
    observation_info = create_observation_info(
        asdm_with_main_execblock_config_processor_sbsummary,
        {"execBlockId": [0]},
    )
    assert isinstance(observation_info, dict)
    assert observation_info == {
        "observer": ["riechers"],
        "release_date": "",
        "project_UID": "uid://A001/X35fd/X21f",
        "execution_block_UID": "uid://A002/X11b94a6/X119b",
        "session_reference_UID": "uid://A002/X11b94a6/X119a",
        "observing_log": None,
        "scheduling_block_UID": "uid://A001/X362e/X332",
    }


def test_create_observation_info_uses_partition_exec_block(asdm_two_exec_blocks):
    """Every partition gets the info of its own ExecBlock (F72)."""
    first = create_observation_info(asdm_two_exec_blocks, {"execBlockId": [0]})
    assert first == {
        "observer": ["first_observer"],
        "release_date": "",
        "project_UID": "uid://A001/X1/X1",
        "execution_block_UID": "uid://A002/X1/X1",
        "session_reference_UID": "uid://A002/X1/X10",
        "observing_log": None,
        "scheduling_block_UID": "uid://A001/X3/X3",
    }

    second = create_observation_info(
        asdm_two_exec_blocks, {"execBlockId": np.array([1])}
    )
    assert second == {
        "observer": ["second_observer"],
        "release_date": "2024-07-21T04:26:40.000000000",
        "project_UID": "uid://A001/X2/X2",
        "execution_block_UID": "uid://A002/X2/X2",
        "session_reference_UID": "uid://A002/X2/X20",
        "observing_log": "DA41_out\nweather_ok",
        # SBSummary_5 does not exist
        "scheduling_block_UID": None,
    }
    for info in (first, second):
        assert not check_dict(info, ObservationInfoDict)


def test_create_observation_info_release_date_is_iso_string(asdm_two_exec_blocks):
    """release_date is an ISO 8601 str, consistent with ArrayTime.toFITS (F33)."""
    info = create_observation_info(asdm_two_exec_blocks, {"execBlockId": [1]})
    release_date = info["release_date"]
    assert isinstance(release_date, str)
    eb_row = asdm_two_exec_blocks.getExecBlock().get()[1]
    assert release_date == eb_row.getReleaseDate().toFITS()
    assert np.datetime64(release_date, "ns") == np.datetime64(
        "2024-07-21T04:26:40", "ns"
    )
    # the observation_info can be stored as attribute in zarr (json)
    xds = xr.Dataset(attrs={"observation_info": info})
    assert xds.attrs["observation_info"]["release_date"] == release_date


@pytest.mark.parametrize(
    "partition_descr, error, match",
    [
        ({"execBlockId": [0, 1]}, ValueError, r"exactly one execBlockId.*\[0, 1\]"),
        ({"execBlockId": []}, ValueError, "exactly one execBlockId"),
        ({"execBlockId": [7]}, ValueError, "execBlockId=7"),
        ({"fieldId": [0]}, KeyError, "execBlockId"),
    ],
)
def test_create_observation_info_errors(
    asdm_two_exec_blocks, partition_descr, error, match
):
    with pytest.raises(error, match=match):
        create_observation_info(asdm_two_exec_blocks, partition_descr)


def test_create_processor_info_with_asdm_simple_extended(
    asdm_with_main_execblock_config_processor_sbsummary,
):
    processor_info = create_processor_info(
        asdm_with_main_execblock_config_processor_sbsummary,
        {"configDescriptionId": [0]},
    )
    assert isinstance(processor_info, dict)
    assert processor_info == {
        "type": "RADIOMETER",
        "sub_type": "SQUARE_LAW_DETECTOR",
    }


@pytest.mark.parametrize(
    "partition_descr, match",
    [
        ({"configDescriptionId": [0, 1]}, "exactly one configDescriptionId"),
        ({"configDescriptionId": [9]}, "configDescriptionId=9"),
        # ConfigDescription_1 refers to Processor_1, not in the Processor table
        ({"configDescriptionId": [1]}, "No Processor row"),
    ],
)
def test_create_processor_info_errors(
    asdm_with_main_execblock_config_processor_sbsummary, partition_descr, match
):
    with pytest.raises(ValueError, match=match):
        create_processor_info(
            asdm_with_main_execblock_config_processor_sbsummary, partition_descr
        )


def test_create_info_dicts_synthetic_release_date(synth_full_pol):
    """End to end from an on-disk ASDM with ExecBlock.releaseDate (F33)."""
    asdm = pyasdm.ASDM()
    asdm.setFromFile(synth_full_pol.path)
    info_dicts = create_info_dicts(
        asdm, xr.Dataset(), {"execBlockId": [0], "configDescriptionId": [0]}
    )
    observation_info = info_dicts["observation_info"]
    assert observation_info["release_date"].startswith("2024-07-21T04:26:40")
    assert observation_info["observer"] == ["synthetic"]
    assert observation_info["execution_block_UID"] == "uid://A002/X1234/X5678"
    assert observation_info["scheduling_block_UID"] == "uid://A001/X362e/X332"
    assert observation_info["observing_log"] is None
    assert not check_dict(observation_info, ObservationInfoDict)
    assert info_dicts["processor_info"]["type"] == "CORRELATOR"
    assert not check_dict(info_dicts["processor_info"], ProcessorInfoDict)
