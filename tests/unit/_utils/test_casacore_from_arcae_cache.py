import numpy as np
import pytest

pytest.importorskip("arcae")

from xradio._utils._casacore import casacore_from_arcae as tables  # noqa: E402


def test_getcolshapestring_of_runs_of_cells_of_different_shapes(tmp_path):
    desc = tables.maketabdesc(
        [tables.makearrcoldesc("V", 0.0, ndim=2, valuetype="double")]
    )
    path = str(tmp_path / "runs.tbl")
    shapes = [(2, 3), (2, 3), (4, 1), (2, 3), (2, 3), (2, 3), (10, 12)]
    with tables.table(path, desc, nrow=len(shapes), readonly=False) as tb:
        for row, shape in enumerate(shapes):
            tb.putcell("V", row, np.zeros(shape))
    expected = [str(list(shape)) for shape in shapes]
    with tables.table(path) as tb:
        assert tb.getcolshapestring("V") == expected
        assert tb.getcolshapestring("V", 1, 4) == expected[1:5]
        assert tb.getcolshapestring("V", 6, 1) == expected[6:]
