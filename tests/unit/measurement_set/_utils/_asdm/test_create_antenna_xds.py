import copy

import pandas as pd
import xarray as xr

import pytest

import pyasdm

from xradio.measurement_set._utils._asdm.create_antenna_xds import (
    create_antenna_xds,
    create_feed_xds,
    get_telescope_name,
)


def add_execblock_table(asdm: pyasdm.ASDM):
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
    <numAntenna> 2 </numAntenna>
    <siteAltitude> 0.0 </siteAltitude>
    <siteLongitude> 0.0 </siteLongitude>
    <siteLatitude> 0.0 </siteLatitude>
    <observingScript> StandardInterferometry.py </observingScript>
    <antennaId> 1 12 Antenna_0 Antenna_1 Antenna_2 Antenna_3 Antenna_4 Antenna_5 Antenna_6 Antenna_7 Antenna_8 Antenna_9 Antenna_10 Antenna_11  </antennaId>
    <sBSummaryId> SBSummary_0 </sBSummaryId>
  </row>
    """
    execblock_table = asdm.getExecBlock()
    execblock_row_0 = pyasdm.ExecBlockRow(execblock_table)
    execblock_row_0.setFromXML(execblock_row_0_xml)
    execblock_table.add(execblock_row_0)


def add_feed_table(asdm: pyasdm.ASDM):
    feed_row_0_xml = """
  <row>
    <feedId> 0 </feedId>
    <timeInterval> 7226686294548387903 3993371484612775807 </timeInterval>
    <numReceptor> 2 </numReceptor>
    <beamOffset> 2 2 2 0.0 0.0 0.0 0.0  </beamOffset>
    <focusReference> 2 2 3 -99999.0 -99999.0 -99999.0 -99999.0 -99999.0 -99999.0  </focusReference>
    <polarizationTypes> 1 2 X Y</polarizationTypes>
    <polResponse> 2 2 2 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0  </polResponse>
    <receptorAngle> 1 2 -0.9346238144 0.6361725124  </receptorAngle>
    <antennaId> Antenna_0 </antennaId>
    <receiverId> 1 2 0 0  </receiverId>
    <spectralWindowId> SpectralWindow_0 </spectralWindowId>
  </row>
    """
    feed_row_1_xml = """
  <row>
    <feedId> 0 </feedId>
    <timeInterval> 7226686294548387903 3993371484612775807 </timeInterval>
    <numReceptor> 2 </numReceptor>
    <beamOffset> 2 2 2 0.0 0.0 0.0 0.0  </beamOffset>
    <focusReference> 2 2 3 -99999.0 -99999.0 -99999.0 -99999.0 -99999.0 -99999.0  </focusReference>
    <polarizationTypes> 1 2 X Y</polarizationTypes>
    <polResponse> 2 2 2 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0  </polResponse>
    <receptorAngle> 1 2 -0.9346238144 0.6361725124  </receptorAngle>
    <antennaId> Antenna_1 </antennaId>
    <receiverId> 1 2 0 0  </receiverId>
    <spectralWindowId> SpectralWindow_0 </spectralWindowId>
  </row>
    """
    feed_table = asdm.getFeed()
    feed_row_0 = pyasdm.FeedRow(feed_table)
    feed_row_0.setFromXML(feed_row_0_xml)
    feed_table.add(feed_row_0)
    feed_row_1 = pyasdm.FeedRow(feed_table)
    feed_row_1.setFromXML(feed_row_1_xml)
    feed_table.add(feed_row_1)


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


def test_create_antenna_xds_with_asdm_simple_7m_antennas(asdm_with_spw_simple):

    add_execblock_table(asdm_with_spw_simple)

    antenna_row_0_xml = """
  <row>
    <antennaId> Antenna_0 </antennaId>
    <name> CM01 </name>
    <antennaMake>MITSUBISHI_7</antennaMake>
    <antennaType>GROUND_BASED</antennaType>
    <dishDiameter> 7.0 </dishDiameter>
    <position> 1 3 -0.002052 -2.32E-4 7.502983  </position>
    <offset> 1 3 0.0 0.0 0.0  </offset>
    <time> 5230000552242000000 </time>
    <stationId> Station_0 </stationId>
  </row>
    """
    antenna_row_1_xml = """
  <row>
    <antennaId> Antenna_1 </antennaId>
    <name> CM03 </name>
    <antennaMake>MITSUBISHI_7</antennaMake>
    <antennaType>GROUND_BASED</antennaType>
    <dishDiameter> 7.0 </dishDiameter>
    <position> 1 3 -0.001862 0.001218 7.49941  </position>
    <offset> 1 3 0.0 0.0 0.0  </offset>
    <time> 5230000552242000000 </time>
    <stationId> Station_1 </stationId>
  </row>
    """
    antenna_table = asdm_with_spw_simple.getAntenna()
    antenna_row_0 = pyasdm.AntennaRow(antenna_table)
    antenna_row_0.setFromXML(antenna_row_0_xml)
    antenna_table.add(antenna_row_0)
    antenna_row_1 = pyasdm.AntennaRow(antenna_table)
    antenna_row_1.setFromXML(antenna_row_1_xml)
    antenna_table.add(antenna_row_1)

    station_row_0_xml = """
  <row>
    <stationId> Station_0 </stationId>
    <name> N602 </name>
    <position> 1 3 2225077.740418 -5440126.559719 -2481521.871323  </position>
    <type>ANTENNA_PAD</type>
  </row>
"""
    station_row_1_xml = """
  <row>
    <stationId> Station_1 </stationId>
    <name> J503 </name>
    <position> 1 3 2225074.121158 -5440116.537775 -2481546.908647  </position>
    <type>ANTENNA_PAD</type>
  </row>

    """
    station_table = asdm_with_spw_simple.getStation()
    station_row_0 = pyasdm.StationRow(station_table)
    station_row_0.setFromXML(station_row_0_xml)
    station_table.add(station_row_0)
    station_row_1 = pyasdm.StationRow(station_table)
    station_row_1.setFromXML(station_row_1_xml)
    station_table.add(station_row_1)

    # w/o Feed table
    create_antenna_xds(asdm_with_spw_simple, 2, 0, xr.DataArray([[0]]))

    # w/ Feed table (no need for polarization xarray)
    asdm_with_feed = copy.deepcopy(asdm_with_spw_simple)
    add_feed_table(asdm_with_feed)
    create_antenna_xds(asdm_with_feed, 2, 0, None)


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
    name = get_telescope_name(asdm_with_spw_simple)
    assert name == "ALMA"
