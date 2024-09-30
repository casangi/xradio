import pytest


## Tests for _utils/_ms/subtables functions. Move to its own file
def test_subtables_read_ms_subtables_required(ms_empty_required):
    from xradio.measurement_set._utils._msv2.subtables import read_ms_subtables

    res = read_ms_subtables(ms_empty_required.fname, done_subt=[])
    assert res == {}


def test_subtables_read_ms_subtables_complete(ms_empty_complete):
    from xradio.measurement_set._utils._msv2.subtables import read_ms_subtables

    res = read_ms_subtables(ms_empty_complete.fname, done_subt=[])
    assert res == {}


def test_subtables_pointing_to_partition():
    from xradio.measurement_set._utils._msv2.subtables import add_pointing_to_partition
    import xarray

    with pytest.raises(
        AttributeError, match="'Dataset' object has no attribute 'time'"
    ):
        res = add_pointing_to_partition(xarray.Dataset(), xarray.Dataset())
        assert res
