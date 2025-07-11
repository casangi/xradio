import pytest

from xradio.measurement_set._utils._asdm.open_partition import open_partition


def test_open_partition_empty():
    with pytest.raises(AttributeError, match="no attribute"):
        open_partition(None, ["fieldId"])
