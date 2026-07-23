import pytest

import pyasdm

import xarray as xr

from xradio.measurement_set._utils._asdm.create_pointing_xds import create_pointing_xds

from xradio.measurement_set.schema import PointingXds
from xradio.schema.check import check_dataset

# Examples mix from uid___A002_X94f2b3_Xb79 (TelCal unit tests) and uid___A002_Xf002b5_X3233 (PL Bencharmk2025)
# Not including <atmosphericCorrection>
pointing_row_xml_0 = """
<row>
<timeInterval> 2568597435384000000 24240000000 </timeInterval> <numSample> 4 </numSample>
<encoder> 2 4 2 -1.86 1.14 -1.86 1.14 -1.86 1.14 -1.86 1.14 </encoder>
<pointingTracking> True </pointingTracking> <usePolynomials> False </usePolynomials>
<timeOrigin> 5137194858648000000 </timeOrigin> <numTerm> 1 </numTerm>
<pointingDirection> 2 4 2 -1.86 1.14 -1.86 1.14 -1.86 1.14 -1.86 1.14 </pointingDirection>
<target> 2 4 2 -1.86 1.14 -1.86 1.14 -1.86 1.14 -1.86 1.14 </target>
<offset> 2 4 2 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 </offset>
<sourceOffset> 2 4 2 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 </sourceOffset>
<antennaId> Antenna_0 </antennaId> <pointingModelId> 2 </pointingModelId>
</row>
"""
pointing_row_xml_1 = """
<row>
<timeInterval> 2568597447744000000 24240000000 </timeInterval> <numSample> 4 </numSample>
<encoder> 2 4 2 -1.86 1.14 -1.86 1.14 -1.86 1.14 -1.86 1.14 </encoder>
<pointingTracking> True </pointingTracking> <usePolynomials> False </usePolynomials>
<timeOrigin> 5137194858648000000 </timeOrigin> <numTerm> 1 </numTerm>
<pointingDirection> 2 4 2 -1.86 1.14 -1.86 1.14 -1.86 1.14 -1.86 1.14 </pointingDirection>
<target> 2 4 2 -1.86 1.14 -1.86 1.14 -1.86 1.14 -1.86 1.14 </target>
<offset> 2 4 2 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 </offset>
<sourceOffset> 2 4 2 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 </sourceOffset>
<antennaId> Antenna_0 </antennaId> <pointingModelId> 2 </pointingModelId>
</row>
"""
pointing_row_xml_2 = """
<row>
<timeInterval> 2568597435384000000 24240000000 </timeInterval> <numSample> 4 </numSample>
<encoder> 2 4 2 -1.86 1.14 -1.86 1.14 -1.86 1.14 -1.86 1.14 </encoder>
<pointingTracking> True </pointingTracking> <usePolynomials> False </usePolynomials>
<timeOrigin> 5137194858648000000 </timeOrigin> <numTerm> 1 </numTerm>
<pointingDirection> 2 4 2 -1.86 1.14 -1.86 1.14 -1.86 1.14 -1.86 1.14 </pointingDirection>
<target> 2 4 2 -1.86 1.14 -1.86 1.14 -1.86 1.14 -1.86 1.14 </target>
<offset> 2 4 2 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 </offset>
<sourceOffset> 2 4 2 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 </sourceOffset>
<antennaId> Antenna_1 </antennaId> <pointingModelId> 2 </pointingModelId>
</row>
"""


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
    asdm_with_execblock_antenna_station_feed,
):

    # TODO: turn into specific fixture, asdm_with_antenna_pointing():
    pointing_table = asdm_with_execblock_antenna_station_feed.getPointing()
    for pointing_row_xml in [
        pointing_row_xml_0,
        pointing_row_xml_1,
        # pointing_row_xml_2,
    ]:
        pointing_row = pyasdm.PointingRow(pointing_table)
        pointing_row.setFromXML(pointing_row_xml)
        pointing_table.add(pointing_row)

    pointing_xds = create_pointing_xds(asdm_with_execblock_antenna_station_feed)
    check_dataset(pointing_xds, PointingXds)
    # Make sure for this non-interpolated xds, there is time_pointing
    for coord in ["time_pointing", "antenna_name", "local_sky_dir_label"]:
        assert coord in pointing_xds.coords
    # The set of variables expected in this example
    for data_var in ["DIRECTION", "POINTING_DISH_MEASURED"]:
        assert data_var in pointing_xds.data_vars
