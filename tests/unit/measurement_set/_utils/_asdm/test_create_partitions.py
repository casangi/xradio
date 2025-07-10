import pytest

from xradio.measurement_set._utils._asdm.create_partitions import create_partitions


def test_create_partitions_empty():
    with pytest.raises(AttributeError, match="no attribute"):
        create_partitions(None, ["fieldId"])
