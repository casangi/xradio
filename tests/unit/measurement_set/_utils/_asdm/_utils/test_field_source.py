import copy

import pandas as pd
import pyasdm
import pytest

from xradio.measurement_set._utils._asdm._utils.field_source import (
    get_source_direction_code,
    retrieve_field_direction_frame,
    retrieve_source_frequency_frame,
)

input_field_rows = [
    {"fieldId": 0, "directionCode": "J2000"},
    {"fieldId": 1, "directionCode": "ICRS"},
    {"fieldId": 2, "directionCode": "AZELSWGEO"},
    {"fieldId": 3, "directionCode": None},
]


@pytest.mark.parametrize(
    "input_field_id, expected_frame",
    [(0, "fk5"), (1, "icrs"), (2, "azelswgeo"), (3, "fk5"), (55, "fk5")],
)
def test_retrieve_field_direction_frame(input_field_id, expected_frame):
    input_field_df = pd.DataFrame.from_records(input_field_rows)
    frame = retrieve_field_direction_frame(input_field_df, input_field_id)
    assert frame == expected_frame


def test_get_source_direction_code_empty(asdm_empty):
    with pytest.raises(AttributeError, match="has no attribute"):
        get_source_direction_code(asdm_empty, (0, 0, 1))


def test_get_source_direction_code_asdm_with_spw_default(asdm_with_spw_default):
    with pytest.raises(AttributeError, match="has no attribute"):
        get_source_direction_code(asdm_with_spw_default, (0, 0, 1))


def test_get_source_direction_code_asdm_with_spw_simple(asdm_with_spw_simple):
    asdm_with_source = copy.deepcopy(asdm_with_spw_simple)

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
    source_table = asdm_with_source.getSource()
    source_row_0 = pyasdm.SourceRow(source_table)
    source_row_0.setFromXML(source_row_0_xml)
    source_table.add(source_row_0)

    get_source_direction_code(
        asdm_with_source,
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

    get_source_direction_code(
        asdm_with_source,
        (
            1,
            pyasdm.types.ArrayTimeInterval(
                "7090683272335387903 4265377529038775807", None
            ),
            0,
        ),
    )


@pytest.mark.parametrize(
    "input_source_row, expected_frame",
    [
        ({"sourceId": 0, "frequencyRefCode": "LSRD"}, "lsrd"),
        ({"sourceId": 1, "frequencyRefCode": "LSRK"}, "lsrk"),
        ({"sourceId": 2, "frequencyRefCode": "BARY"}, "BARY"),
        ({"sourceId": 5, "frequencyRefCode": "REST"}, "REST"),
        ({"sourceId": 4, "frequencyRefCode": "GEO"}, "gcrs"),
        ({"sourceId": 3, "frequencyRefCode": "TOPO"}, "TOPO"),
        ({"sourceId": 3, "frequencyRefCode": None}, "lsrk"),
    ],
)
def test_retrieve_source_frequency_frame(input_source_row, expected_frame):
    input_source_df = pd.DataFrame.from_records([input_source_row])
    frame = retrieve_source_frequency_frame(input_source_df)
    assert frame == expected_frame
