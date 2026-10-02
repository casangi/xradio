import typing

import pyasdm
import pytest
from pyasdm.enumerations.DirectionReferenceCode import DirectionReferenceCode
from pyasdm.enumerations.FrequencyReferenceCode import FrequencyReferenceCode

from xradio.measurement_set._utils._asdm._utils.field_source import (
    ASDM_DIRECTION_CODE_TO_FRAME,
    ASDM_FREQUENCY_CODE_TO_OBSERVER,
    direction_code_name,
    direction_code_to_frame,
    frequency_code_to_observer,
    get_optional_row_attr,
    make_field_name,
)
from xradio.measurement_set.schema import (
    AllowedSkyCoordFrames,
    AllowedSpectralCoordFrames,
)

SOURCE_TIME_INTERVAL = "7090683272335387903 4265377529038775807"


def _source_row_xml(source_id: int, name: str, direction_code: str | None) -> str:
    code_xml = (
        f"<directionCode>{direction_code}</directionCode>" if direction_code else ""
    )
    return f"""
  <row>
    <sourceId> {source_id} </sourceId>
    <timeInterval> {SOURCE_TIME_INTERVAL} </timeInterval>
    <code> none </code>
    <direction> 1 2 1.3528024488371877 0.31436086058385826  </direction>
    <properMotion> 1 2 0.0 0.0  </properMotion>
    <sourceName> {name} </sourceName>
    {code_xml}
    <spectralWindowId> SpectralWindow_0 </spectralWindowId>
  </row>
"""


def _add_source_row(asdm: pyasdm.ASDM, xml: str):
    source_table = asdm.getSource()
    row = pyasdm.SourceRow(source_table)
    row.setFromXML(xml)
    return source_table.add(row)


@pytest.mark.parametrize(
    "name, field_id, expected",
    [
        ("M100", 3, "M100_3"),
        (" J0423-0120 ", 0, "J0423-0120_0"),
        ("M100", 12, "M100_12"),
    ],
)
def test_make_field_name(name, field_id, expected):
    assert make_field_name(name, field_id) == expected


def test_make_field_name_disambiguates_shared_names():
    names = [make_field_name("M100", field_id) for field_id in range(3)]
    assert names == ["M100_0", "M100_1", "M100_2"]
    assert len(set(names)) == 3


@pytest.mark.parametrize(
    "code, expected",
    [
        ("J2000", "fk5"),
        ("ICRS", "icrs"),
        ("B1950", "fk4"),
        ("GALACTIC", "galactic"),
        ("SUPERGAL", "supergalactic"),
        ("HADEC", "hadec"),
        ("AZELNE", "altaz"),
        ("AZELNEGEO", "altaz"),
        ("ITRF", "itrs"),
        ("icrs", "icrs"),
        (None, "fk5"),
    ],
)
def test_direction_code_to_frame(code, expected):
    assert direction_code_to_frame(code) == expected


@pytest.mark.parametrize("code", ["J2000", "ICRS", "B1950", "GALACTIC"])
def test_direction_code_to_frame_from_enumeration(code):
    enum_value = DirectionReferenceCode(code)
    assert direction_code_to_frame(enum_value) == ASDM_DIRECTION_CODE_TO_FRAME[code]


@pytest.mark.parametrize(
    "code", ["APP", "JMEAN", "JTRUE", "B1950_VLA", "AZELSW", "TOPO", "MOON"]
)
def test_direction_code_to_frame_unsupported(code):
    with pytest.raises(ValueError, match=f"direction reference code {code} is not"):
        direction_code_to_frame(code)


@pytest.mark.parametrize(
    "code, expected",
    [
        ("ICRS", "ICRS"),
        (" azelgeo ", "AZELGEO"),
        (DirectionReferenceCode("HADEC"), "HADEC"),
        (None, "J2000"),
    ],
)
def test_direction_code_name(code, expected):
    """Code name used in messages: enumeration or string, None -> J2000."""
    assert direction_code_name(code) == expected


def test_direction_frames_are_allowed_by_schema():
    allowed = set(typing.get_args(AllowedSkyCoordFrames))
    assert set(ASDM_DIRECTION_CODE_TO_FRAME.values()) <= allowed


@pytest.mark.parametrize(
    "code, expected",
    [
        ("LSRK", "lsrk"),
        ("LSRD", "lsrd"),
        ("TOPO", "TOPO"),
        ("BARY", "BARY"),
        ("GEO", "gcrs"),
        ("REST", "REST"),
        ("LABREST", "REST"),
        (FrequencyReferenceCode("LSRK"), "lsrk"),
        (None, "lsrk"),
    ],
)
def test_frequency_code_to_observer(code, expected):
    assert frequency_code_to_observer(code) == expected


def test_frequency_code_to_observer_default_and_unsupported():
    assert frequency_code_to_observer(None, default="REST") == "REST"
    with pytest.raises(ValueError, match="frequency reference code GALACTO"):
        frequency_code_to_observer("GALACTO")


def test_frequency_observers_are_allowed_by_schema():
    allowed = set(typing.get_args(AllowedSpectralCoordFrames))
    assert set(ASDM_FREQUENCY_CODE_TO_OBSERVER.values()) <= allowed


def test_get_optional_row_attr():
    asdm = pyasdm.ASDM()
    with_code = _add_source_row(asdm, _source_row_xml(0, "A", "ICRS"))
    without_code = _add_source_row(asdm, _source_row_xml(1, "B", None))

    assert get_optional_row_attr(with_code, "directionCode").getName() == "ICRS"
    assert get_optional_row_attr(without_code, "directionCode") is None
    assert get_optional_row_attr(without_code, "numLines", default=0) == 0
    # required attributes have no is*Exists method
    with pytest.raises(AttributeError):
        get_optional_row_attr(with_code, "sourceName")
