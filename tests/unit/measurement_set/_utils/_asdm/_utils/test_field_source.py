import pytest


def test_get_direction_codes_empty(asdm_empty):
    from xradio.measurement_set._utils._asdm._utils.field_source import get_direction_codes

    with pytest.raises(AttributeError, match="has no attribute"):
        get_direction_codes(asdm_empty, (0, 0, 1))

