import pytest

from xradio.measurement_set._utils._asdm.open_asdm import open_asdm


def test_open_asdm_empty():
    with pytest.raises(AttributeError, match="no attribute"):
        open_asdm(None, ["fieldId"])
