import copy

import pytest

import pyasdm

from xradio.measurement_set._utils._asdm.create_field_and_source_xds import (
    create_field_and_source_xds,
    make_sky_coord_measure_attrs,
)


def test_create_field_and_source_xds_empty():
    with pytest.raises(AttributeError, match="has no attribute"):
        create_field_and_source_xds(None, {}, 0, False)


def test_create_field_and_source_xds_with_asdm_empty(asdm_empty):
    with pytest.raises(IndexError, match="out of bounds"):
        create_field_and_source_xds(asdm_empty, {"fieldId": [0]}, 0, False)


def test_create_field_and_source_xds_with_asdm_default(asdm_with_spw_default):
    with pytest.raises(IndexError, match="out of bounds"):
        create_field_and_source_xds(asdm_with_spw_default, {"fieldId": [0]}, 0, False)


def test_create_field_and_source_xds_with_asdm_simple(asdm_with_spw_simple):
    with pytest.raises(IndexError, match="out of bounds"):
        create_field_and_source_xds(asdm_with_spw_simple, {"fieldId": [0]}, 0, False)


def test_create_field_and_source_xds_with_field_source(asdm_with_spw_simple):

    asdm_with_source_field = copy.deepcopy(asdm_with_spw_simple)

    # numLines, etc. optional attributes (line info) borrowed from a different row
    source_row_0_xml_main_chunk = """
    <sourceId> 0 </sourceId>
    <timeInterval> 7226686294548387903 3993371484612775807 </timeInterval>
    <code> none </code>
    <direction> 1 2 1.148703043969797 -0.02343136276090743  </direction>
    <properMotion> 1 2 0.0 0.0  </properMotion>
    <sourceName> J0423-0120 </sourceName>
    <directionCode>ICRS</directionCode>
    <numFreq> 8 </numFreq>
    <numStokes> 4 </numStokes>
    <frequency> 1 8 6.246009851630612E11 6.351401391376388E11 6.227258305018281E11 6.370152937988718E11 6.208506844426849E11 6.38890439858015E11 6.189755340824967E11 6.407655902182032E11  </frequency>
    <stokesParameter> 1 4 I Q U V</stokesParameter>
    <flux> 2 8 4 2.3005181262897874 0.0 0.0 0.0 2.287221128582244 0.0 0.0 0.0 2.3029156368538692 0.0 0.0 0.0 2.284886406783615 0.0 0.0 0.0 2.305322876552315 0.0 0.0 0.0 2.282560930875987 0.0 0.0 0.0 2.307739931084516 0.0 0.0 0.0 2.28024462134997 0.0 0.0 0.0  </flux>
    <size> 2 8 2 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0  </size>
    <dopplerVelocity> 1 1 0.0  </dopplerVelocity>
    <dopplerReferenceSystem>LSRK</dopplerReferenceSystem>
    <dopplerCalcType>OPTICAL</dopplerCalcType>
    <parallax> 1 1 0.0  </parallax>
    <spectralWindowId> SpectralWindow_0 </spectralWindowId>
    """
    source_row_0_xml_without_lineinfo = f"""
  <row>
    {source_row_0_xml_main_chunk}
  </row>
    """
    source_table = asdm_with_source_field.getSource()
    source_row_0 = pyasdm.SourceRow(source_table)
    source_row_0.setFromXML(source_row_0_xml_without_lineinfo)
    source_table.add(source_row_0)

    field_row_0_xml = """
  <row>
    <fieldId> Field_0 </fieldId>
    <fieldName> J0423-0120 </fieldName>
    <numPoly> 1 </numPoly>
    <delayDir> 2 1 2 1.1487030439690096 -0.023431362760917465  </delayDir>
    <phaseDir> 2 1 2 1.1487030439690096 -0.023431362760917465  </phaseDir>
    <referenceDir> 2 1 2 1.148703043969797 -0.02343136276090743  </referenceDir>
    <time> 5230000639104000000 </time>
    <code> none </code>
    <directionCode>ICRS</directionCode>
    <sourceId> 0 </sourceId>
  </row>
    """
    field_table = asdm_with_source_field.getField()
    field_row_0 = pyasdm.FieldRow(field_table)
    field_row_0.setFromXML(field_row_0_xml)
    field_table.add(field_row_0)

    # IF
    create_field_and_source_xds(asdm_with_source_field, {"fieldId": [0]}, 0, False)

    # SD
    create_field_and_source_xds(asdm_with_source_field, {"fieldId": [0]}, 0, True)

    # IF with lineinfo in Source
    asdm_with_source_field_with_lineinfo = copy.deepcopy(asdm_with_spw_simple)
    source_row_0_xml_with_lineinfo = f"""
  <row>
    {source_row_0_xml_main_chunk}
    <numLines> 1 </numLines>
    <transition> 1 1 &quot;cii_blue(ID=0)&quot;  </transition>
    <restFrequency> 1 1 1.8949423708552605E12  </restFrequency>
    <sysVel> 1 1 0.0  </sysVel>
  </row>
    """
    source_table = asdm_with_source_field_with_lineinfo.getSource()
    source_row_0 = pyasdm.SourceRow(source_table)
    source_row_0.setFromXML(source_row_0_xml_with_lineinfo)
    source_table.add(source_row_0)

    field_table = asdm_with_source_field_with_lineinfo.getField()
    field_row_0 = pyasdm.FieldRow(field_table)
    field_row_0.setFromXML(field_row_0_xml)
    field_table.add(field_row_0)
    create_field_and_source_xds(
        asdm_with_source_field_with_lineinfo, {"fieldId": [0]}, 0, False
    )


@pytest.mark.parametrize(
    "units, frame, expected_output",
    [
        ("any", "ICRS", {"units": "any", "frame": "ICRS", "type": "sky_coord"}),
        ("rad", "ICRS", {"units": "rad", "frame": "ICRS", "type": "sky_coord"}),
        ("m", "altaz", {"units": "m", "frame": "altaz", "type": "sky_coord"}),
    ],
)
def test_make_sky_coord_measure_attrs(units, frame, expected_output):
    result = make_sky_coord_measure_attrs(units, frame)
    assert result == expected_output
