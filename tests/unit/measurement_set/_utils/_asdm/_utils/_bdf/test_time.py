import pandas as pd
import pytest


def test_get_times_from_bdfs_empty():
    from xradio.measurement_set._utils._asdm._utils._bdf.time import get_times_from_bdfs

    with pytest.raises(ValueError, match="at least one"):
        get_times_from_bdfs([], pd.DataFrame())


def test_get_times_from_bdfs_non_existent():
    from xradio.measurement_set._utils._asdm._utils._bdf.time import get_times_from_bdfs
    from pyasdm.exceptions import BDFReaderException

    with pytest.raises(BDFReaderException, match="No such file or directory"):
        get_times_from_bdfs(["empty-non-existent"], pd.DataFrame())


def test_make_blob_info_empty():
    from xradio.measurement_set._utils._asdm._utils._bdf.time import make_blob_info
    from pyasdm.bdf import BDFHeader

    info = make_blob_info(BDFHeader())
    assert isinstance(info, pd.DataFrame)
    assert info.shape == (1, 22)


def test_load_times_from_bdfs_empty():
    from xradio.measurement_set._utils._asdm._utils._bdf.time import (
        load_times_from_bdfs,
    )

    from pyasdm.exceptions import BDFReaderException

    with pytest.raises(BDFReaderException, match="No such file or directory"):
        load_times_from_bdfs(["/path/nonexistant/foo", "/path/nonexistant/bar"])


def test_load_times_bdf_empty():
    from xradio.measurement_set._utils._asdm._utils._bdf.time import load_times_bdf

    from pyasdm.exceptions import BDFReaderException

    with pytest.raises(BDFReaderException, match="No such file or directory"):
        load_times_bdf("/path/nonexistant/foo")
