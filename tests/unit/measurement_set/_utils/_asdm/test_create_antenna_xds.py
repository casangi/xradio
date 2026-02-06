import pandas as pd
import xarray as xr

import pytest

import pyasdm

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


def test_get_telescope_name_asdm_default(asdm_with_spw_default):
    with pytest.raises(RuntimeError, match="Issue with telescopeName"):
        get_telescope_name(asdm_with_spw_default)


def test_get_telescope_name_asdm_with_spw_simple(asdm_with_spw_simple):
    execblock_row_0_xml = """
  <row>
    <execBlockId> ExecBlock_0 </execBlockId>
    <startTime> 5230000552242000000 </startTime>
    <endTime> 5230001040525000000 </endTime>
    <execBlockNum> 999 </execBlockNum>
    <execBlockUID>
      <EntityRef entityId="uid://A002/X11b94a6/X119b" partId="X00000000" entityTypeName="ASDM" documentVersion="1"/>
    </execBlockUID>
    <projectUID>
      <EntityRef entityId="uid://A001/X35fd/X21f" partId="X00000000" entityTypeName="ObsProject" documentVersion="1"/>
    </projectUID>
    <configName> 7M </configName>
    <telescopeName> ALMA </telescopeName>
    <observerName> riechers </observerName>
    <numObservingLog> 0 </numObservingLog>
    <observingLog> 1 0  </observingLog>
    <sessionReference>
      <EntityRef entityId="uid://A002/X11b94a6/X119a" partId="X00000000" entityTypeName="Session" documentVersion="1"/>
    </sessionReference>
    <baseRangeMin> 0.0 </baseRangeMin>
    <baseRangeMax> 0.0 </baseRangeMax>
    <baseRmsMinor> 0.0 </baseRmsMinor>
    <baseRmsMajor> 0.0 </baseRmsMajor>
    <basePa> 0.0 </basePa>
    <aborted> false </aborted>
    <numAntenna> 12 </numAntenna>
    <siteAltitude> 0.0 </siteAltitude>
    <siteLongitude> 0.0 </siteLongitude>
    <siteLatitude> 0.0 </siteLatitude>
    <observingScript> StandardInterferometry.py </observingScript>
    <antennaId> 1 12 Antenna_0 Antenna_1 Antenna_2 Antenna_3 Antenna_4 Antenna_5 Antenna_6 Antenna_7 Antenna_8 Antenna_9 Antenna_10 Antenna_11  </antennaId>
    <sBSummaryId> SBSummary_0 </sBSummaryId>
  </row>
    """
    execblock_table = asdm_with_spw_simple.getExecBlock()
    execblock_row_0 = pyasdm.ExecBlockRow(execblock_table)
    execblock_row_0.setFromXML(execblock_row_0_xml)
    execblock_table.add(execblock_row_0)

    name = get_telescope_name(asdm_with_spw_simple)
    assert name == "ALMA"
