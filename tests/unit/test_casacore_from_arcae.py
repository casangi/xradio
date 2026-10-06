import numpy as np
import pytest

pytest.importorskip("arcae")

from xradio._utils._casacore import casacore_from_arcae as tables  # noqa: E402

NROWS = 3
SHAPE = (6, 4)

# (blc, trc, inc) in python-casacore convention: C order, trc inclusive
CELL_SLICES = [
    ([0, 0], [5, 3], []),
    ([1, 1], [4, 2], []),
    ([0, 1], [4, 3], [2, 2]),
    ([1, 0], [5, 3], [3, 1]),
    ([2, 1], [2, 3], []),
    ([0, 0], [4, 3], [5, 3]),
]


def _numpy_index(blc, trc, inc):
    inc = inc or [1] * len(blc)
    return tuple(slice(b, t + 1, s) for b, t, s in zip(blc, trc, inc, strict=True))


@pytest.fixture
def tiled_cell_table(tmp_path):
    desc = tables.maketabdesc(
        [
            tables.makearrcoldesc(
                "DATA",
                np.complex64(0),
                shape=list(SHAPE),
                datamanagertype="TiledCellStMan",
                datamanagergroup="TCS",
            )
        ]
    )
    dminfo = {
        "*1": {
            "TYPE": "TiledCellStMan",
            "NAME": "TCS",
            "COLUMNS": ["DATA"],
            "SPEC": {"DEFAULTTILESHAPE": [2, 3]},
        }
    }
    data = (np.arange(NROWS * np.prod(SHAPE)).reshape(NROWS, *SHAPE) * (1 + 1j)).astype(
        np.complex64
    )
    path = str(tmp_path / "tiled.tbl")
    with tables.table(path, desc, nrow=NROWS, dminfo=dminfo, readonly=False) as tb:
        tb.putcol("DATA", data)
        assert tb.getdminfo("DATA")["TYPE"] == "TiledCellStMan"
    return path, data


@pytest.mark.parametrize("blc, trc, inc", CELL_SLICES)
def test_getcellslice_tiled_cell_column(tiled_cell_table, blc, trc, inc):
    path, data = tiled_cell_table
    with tables.table(path) as tb:
        got = tb.getcellslice("DATA", 1, blc, trc, inc)
    np.testing.assert_array_equal(
        got, data[1][_numpy_index(blc, trc, inc)], strict=True
    )


@pytest.mark.parametrize("blc, trc, inc", CELL_SLICES)
def test_putcellslice_tiled_cell_column(tiled_cell_table, blc, trc, inc):
    path, data = tiled_cell_table
    index = _numpy_index(blc, trc, inc)
    value = -data[1][index]
    with tables.table(path, readonly=False) as tb:
        tb.putcellslice("DATA", 1, value, blc, trc, inc)
    expected = data.copy()
    expected[1][index] = value
    with tables.table(path) as tb:
        for row in range(NROWS):
            np.testing.assert_array_equal(
                tb.getcell("DATA", row), expected[row], strict=True
            )
