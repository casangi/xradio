import pytest

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
