import pytest

import pandas as pd

def test_normalize_time_slice_invalid():
    from xradio.vis._vis_utils._ms._tables.read_subtables import normalize_time_slice

    with pytest.raises(TypeError, match="argument must be"):
        res = normalize_time_slice(None, slice({}, {}))


def test_normalize_time_slice_invalid_content():
    from xradio.vis._vis_utils._ms._tables.read_subtables import normalize_time_slice

    with pytest.raises(ValueError, match="Invalid time type"):
        res = normalize_time_slice(None, slice(3.3, 0.1))


def test_normalize_time_slice_pd_timestamp():
    from xradio.vis._vis_utils._ms._tables.read_subtables import normalize_time_slice

    tstamp = pd.Timestamp(1000)
    with pytest.raises(AttributeError, match="has no attribute"):
        res = normalize_time_slice(None, slice(tstamp, tstamp))


def test_normalize_time_slice():
    from xradio.vis._vis_utils._ms._tables.read_subtables import normalize_time_slice

    with pytest.raises(RuntimeError, match="Error in TaQL"):
        res = normalize_time_slice(None, slice(0, 33))
        assert res == 33
