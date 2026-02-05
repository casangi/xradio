import pytest

import pyasdm

from xradio.measurement_set._utils._asdm._utils.field_source import get_direction_codes


def test_get_direction_codes_empty(asdm_empty):

    with pytest.raises(AttributeError, match="has no attribute"):
        get_direction_codes(asdm_empty, (0, 0, 1))


def test_get_direction_codes_asdm_with_spw_default(asdm_with_spw_default):

    with pytest.raises(AttributeError, match="has no attribute"):
        get_direction_codes(asdm_with_spw_default, (0, 0, 1))


def test_get_direction_codes_asdm_with_spw_simple(asdm_with_spw_simple):

    # directionCode was ICRS - chanced for testing purposes
    source_row_0_xml = """
  <row>
    <sourceId> 0 </sourceId>
    <timeInterval> 7090683272335387903 4265377529038775807 </timeInterval>
    <code> none </code>
    <direction> 1 2 1.3528024488371877 0.31436086058385826  </direction>
    <properMotion> 1 2 0.0 0.0  </properMotion>
    <sourceName> J0510+1800 </sourceName>
    <directionCode>J2000</directionCode>
    <numFreq> 4 </numFreq>
    <numStokes> 4 </numStokes>
    <frequency> 1 4 2.1998305541101968E11 2.1800401136221878E11 2.3300428785048865E11 2.350043230298097E11  </frequency>
    <stokesParameter> 1 4 I Q U V</stokesParameter>
    <flux> 2 4 4 3.855274498565509 0.0 0.0 0.0 3.8656094704537534 0.0 0.0 0.0 3.7901533014049034 0.0 0.0 0.0 3.7805688135422617 0.0 0.0 0.0  </flux>
    <size> 2 4 2 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0  </size>
    <spectralWindowId> SpectralWindow_0 </spectralWindowId>
  </row>
"""
    source_table = asdm_with_spw_simple.getSource()
    source_row_0 = pyasdm.SourceRow(source_table)
    source_row_0.setFromXML(source_row_0_xml)
    source_table.add(source_row_0)

    get_direction_codes(
        asdm_with_spw_simple,
        (
            0,
            pyasdm.types.ArrayTimeInterval(
                "7090683272335387903 4265377529038775807", None
            ),
            0,
        ),
    )

    source_row_1_xml = """
  <row>
    <sourceId> 1 </sourceId>
    <timeInterval> 7090683272335387903 4265377529038775807 </timeInterval>
    <code> none </code>
    <direction> 1 2 1.3528024488371877 0.31436086058385826  </direction>
    <properMotion> 1 2 0.0 0.0  </properMotion>
    <sourceName> Test_Source </sourceName>
    <spectralWindowId> SpectralWindow_0 </spectralWindowId>
  </row>
"""
    source_row_1 = pyasdm.SourceRow(source_table)
    source_row_1.setFromXML(source_row_1_xml)
    source_table.add(source_row_1)

    get_direction_codes(
        asdm_with_spw_simple,
        (
            1,
            pyasdm.types.ArrayTimeInterval(
                "7090683272335387903 4265377529038775807", None
            ),
            0,
        ),
    )
