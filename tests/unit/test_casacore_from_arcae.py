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


def test_string_column_descriptions():
    """A string default makes a string column, as in python-casacore."""
    assert tables.makescacoldesc("S", "none")["desc"]["valueType"] == "string"
    assert tables.makearrcoldesc("A", "", ndim=2)["desc"]["valueType"] == "string"


# Undefined cells (array cells never written), expected as python-casacore
# 3.8.1 gives them: rows 1 and 3 are defined, rows 0, 2 and 4 are not.
UNDEFINED_ROWS = [0, 2, 4]
VAR_COLUMNS = {
    # name: (valuetype, ndim, value of the defined cells, undefined row value)
    "VARF": ("double", -1, np.arange(3.0), np.empty((0,))),
    "VARF2": ("double", 2, np.ones((2, 3)), np.empty((0, 0))),
    "VARI": ("int", 1, np.arange(2, dtype=np.int32), np.empty((0,), np.int32)),
    "VARS": ("string", -1, np.array(["a", "b"]), {"shape": [], "array": []}),
    "VARS1": ("string", 1, np.array(["a", "b"]), []),
    "VARS2": ("string", 2, np.array([["a", "b"]]), {"shape": [0, 0], "array": []}),
}


@pytest.fixture(scope="module")
def undefined_cells_table(tmp_path_factory):
    desc = tables.maketabdesc(
        [
            tables.makearrcoldesc(col, 0.0, ndim=ndim, valuetype=valuetype)
            for col, (valuetype, ndim, _, _) in VAR_COLUMNS.items()
        ]
    )
    path = str(tmp_path_factory.mktemp("undefined") / "undefined.tbl")
    with tables.table(path, desc, nrow=5, readonly=False) as tb:
        for col, (_, _, value, _) in VAR_COLUMNS.items():
            tb.putcell(col, 1, value)
            tb.putcell(col, 3, value)
    return path


@pytest.mark.parametrize("col", VAR_COLUMNS)
def test_iscelldefined_undefined_cells(undefined_cells_table, col):
    with tables.table(undefined_cells_table) as tb:
        defined = [tb.iscelldefined(col, row) for row in range(5)]
    assert defined == [False, True, False, True, False]


@pytest.mark.parametrize("col", VAR_COLUMNS)
@pytest.mark.parametrize("startrow, nrow", [(0, 1), (2, 1), (1, 3), (0, 2), (0, -1)])
def test_getcolshapestring_raises_for_undefined_cells(
    undefined_cells_table, col, startrow, nrow
):
    with tables.table(undefined_cells_table) as tb:
        with pytest.raises(RuntimeError, match="no array in row"):
            tb.getcolshapestring(col, startrow, nrow)


@pytest.mark.parametrize("col", VAR_COLUMNS)
def test_getcolshapestring_of_defined_cells(undefined_cells_table, col):
    shape = str(list(VAR_COLUMNS[col][2].shape))
    with tables.table(undefined_cells_table) as tb:
        assert tb.getcolshapestring(col, 1, 1) == [shape]
        assert tb.getcolshapestring(col, 3, 1) == [shape]


@pytest.mark.parametrize("col", VAR_COLUMNS)
@pytest.mark.parametrize(
    "read",
    [
        lambda tb, col: tb.getcell(col, 0),
        lambda tb, col: tb.col(col)[2],
        lambda tb, col: tb.getcol(col, 0, 1),
        lambda tb, col: tb.getcol(col, 1, 3),
        lambda tb, col: tb.getcol(col),
    ],
    ids=["getcell", "col", "getcol-undefined", "getcol-mixed", "getcol-all"],
)
def test_reads_of_undefined_cells_raise(undefined_cells_table, col, read):
    with tables.table(undefined_cells_table) as tb:
        with pytest.raises(RuntimeError):
            read(tb, col)


def _assert_tablerow_value(got, expected):
    if isinstance(expected, np.ndarray):
        assert isinstance(got, np.ndarray)
        np.testing.assert_array_equal(got, expected, strict=True)
    else:
        assert got == expected
        assert type(got) is type(expected)


@pytest.mark.parametrize("col", VAR_COLUMNS)
@pytest.mark.parametrize("rows", [slice(0, 5), slice(0, 1), slice(4, 5)])
def test_tablerow_undefined_cells(undefined_cells_table, col, rows):
    _, _, value, undefined = VAR_COLUMNS[col]
    with tables.table(undefined_cells_table) as tb:
        got = [row[col] for row in tb.row([col])[rows]]
        single = tb.row([col])[rows.start][col]
    assert len(got) == rows.stop - rows.start
    for rownr, cell in zip(range(rows.start, rows.stop), got, strict=True):
        if rownr in UNDEFINED_ROWS:
            _assert_tablerow_value(cell, undefined)
        else:
            # the defined cells keep their values next to undefined ones
            if isinstance(cell, dict):  # python-casacore's n-d string arrays
                cell = np.reshape(cell["array"], cell["shape"])
            np.testing.assert_array_equal(cell, value)
    _assert_tablerow_value(single, undefined)


@pytest.mark.parametrize("col", VAR_COLUMNS)
def test_never_written_column(tmp_path, col):
    valuetype, ndim, _, undefined = VAR_COLUMNS[col]
    desc = tables.maketabdesc(
        [tables.makearrcoldesc(col, 0.0, ndim=ndim, valuetype=valuetype)]
    )
    path = str(tmp_path / "unwritten.tbl")
    with tables.table(path, desc, nrow=3, readonly=False):
        pass
    with tables.table(path) as tb:
        assert not tb.iscelldefined(col, 0)
        with pytest.raises(RuntimeError, match="no array in row 0"):
            tb.getcolshapestring(col)
        with pytest.raises(RuntimeError):
            tb.getcol(col)
        for row in tb.row([col])[0:3]:
            _assert_tablerow_value(row[col], undefined)


def test_undefined_cells_of_tiled_shape_column(tmp_path):
    desc = tables.maketabdesc(
        [
            tables.makearrcoldesc(
                "TSM",
                0.0,
                ndim=2,
                valuetype="float",
                datamanagertype="TiledShapeStMan",
                datamanagergroup="TSM",
            )
        ]
    )
    dminfo = tables.makedminfo(desc, {"TSM": {"DEFAULTTILESHAPE": [2, 4, 2]}})
    value = np.ones((2, 4), dtype=np.float32)
    path = str(tmp_path / "tsm.tbl")
    with tables.table(path, desc, nrow=5, dminfo=dminfo, readonly=False) as tb:
        tb.putcell("TSM", 1, value)
        tb.putcell("TSM", 3, value)
    with tables.table(path) as tb:
        assert tb.getdminfo("TSM")["TYPE"] == "TiledShapeStMan"
        assert [tb.iscelldefined("TSM", row) for row in range(5)] == [
            False,
            True,
            False,
            True,
            False,
        ]
        assert tb.getcolshapestring("TSM", 1, 1) == ["[2, 4]"]
        with pytest.raises(RuntimeError, match="no array in row 2"):
            tb.getcolshapestring("TSM", 1, 2)
        with pytest.raises(RuntimeError):
            tb.getcell("TSM", 0)
        got = [row["TSM"] for row in tb.row(["TSM"])[0:5]]
    for rownr, cell in enumerate(got):
        expected = np.empty((0, 0), np.float32) if rownr in UNDEFINED_ROWS else value
        np.testing.assert_array_equal(cell, expected, strict=True)


def test_cells_of_different_ndim_next_to_undefined_cells(tmp_path):
    desc = tables.maketabdesc(
        [tables.makearrcoldesc("VAR", 0.0, ndim=-1, valuetype="double")]
    )
    path = str(tmp_path / "ndim.tbl")
    with tables.table(path, desc, nrow=4, readonly=False) as tb:
        tb.putcell("VAR", 1, np.arange(3.0))
        tb.putcell("VAR", 2, np.ones((2, 2)))
    with tables.table(path) as tb:
        assert [tb.iscelldefined("VAR", row) for row in range(4)] == [
            False,
            True,
            True,
            False,
        ]
        assert tb.getcolshapestring("VAR", 1, 2) == ["[3]", "[2, 2]"]
        with pytest.raises(RuntimeError, match="no array in row 3"):
            tb.getcolshapestring("VAR", 1, 3)
        got = [row["VAR"] for row in tb.row(["VAR"])[0:4]]
    for cell, expected in zip(
        got,
        [np.empty((0,)), np.arange(3.0), np.ones((2, 2)), np.empty((0,))],
        strict=True,
    ):
        np.testing.assert_array_equal(cell, expected, strict=True)


def test_tablerow_defined_cells_without_values(tmp_path):
    # defined cells of a shape with a zero dimension (row 0) next to defined
    # (row 1) and undefined (row 2) cells, written with Glish-style TaQL
    # (shapes in Fortran order)
    columns = {
        # name: (valuetype, ndim, value of row 1, TaQL value of row 0,
        #        shape string and tablerow value of row 0, of row 2)
        "I1": (
            "int",
            1,
            np.arange(2, dtype=np.int32),
            "array(0, [0])",
            "[0]",
            np.empty((0,), np.int32),
            np.empty((0,), np.int32),
        ),
        "S1": ("string", 1, np.array(["a"]), "array('', [0])", "[0]", [], []),
        "D2": (
            "double",
            2,
            np.ones((2, 3)),
            "array(0., [3, 0])",
            "[0, 3]",
            np.empty((0, 3)),
            np.empty((0, 0)),
        ),
        "S2": (
            "string",
            2,
            np.array([["a", "b"]]),
            "array('', [2, 0])",
            "[0, 2]",
            {"shape": [0, 2], "array": []},
            {"shape": [0, 0], "array": []},
        ),
        "C2": (
            "complex",
            2,
            np.full((2, 3), 1 + 2j, np.complex64),
            "array(0+0i, [0, 2])",
            "[2, 0]",
            np.empty((2, 0), np.complex64),
            np.empty((0, 0), np.complex64),
        ),
        "B1": (
            "boolean",
            1,
            np.array([True, False]),
            "array(F, [0])",
            "[0]",
            np.empty((0,), bool),
            np.empty((0,), bool),
        ),
    }
    desc = tables.maketabdesc(
        [
            tables.makearrcoldesc(col, 0.0, ndim=spec[1], valuetype=spec[0])
            for col, spec in columns.items()
        ]
    )
    path = str(tmp_path / "empty.tbl")
    with tables.table(path, desc, nrow=3, readonly=False) as tb:
        for col, (_, _, value, taql_value, _, _, _) in columns.items():
            tb.putcell(col, 1, value)
            tables.taql(
                f"UPDATE $tb SET {col} = {taql_value} WHERE ROWID() == 0",
                style="Glish",
            )
    with tables.table(path) as tb:
        for col, (_, _, value, _, shape, empty, undefined) in columns.items():
            assert [tb.iscelldefined(col, row) for row in range(3)] == [
                True,
                True,
                False,
            ]
            assert tb.getcolshapestring(col, 0, 1) == [shape]
            assert tb.getcolshapestring(col, 0, 2) == [shape, str(list(value.shape))]
            with pytest.raises(RuntimeError, match="no array in row 2"):
                tb.getcolshapestring(col, 0, 3)
            got = [row[col] for row in tb.row([col])[0:3]]
            _assert_tablerow_value(got[0], empty)
            _assert_tablerow_value(tb.row([col])[0][col], empty)
            # (rows without an undefined cell)
            _assert_tablerow_value(tb.row([col])[0:2][0][col], empty)
            if isinstance(got[1], dict):  # python-casacore's n-d string arrays
                got[1] = np.reshape(got[1]["array"], got[1]["shape"])
            np.testing.assert_array_equal(got[1], value)
            _assert_tablerow_value(got[2], undefined)


def test_writes_of_another_shape_reshape_variable_shaped_cells(tmp_path):
    """Like python-casacore, putcell and putcol reshape the defined cells of
    a variable-shaped column (arcae writes into the cell's shape)."""
    desc = tables.maketabdesc(
        [tables.makearrcoldesc("V", 0.0, ndim=1, valuetype="double")]
    )
    with tables.table(str(tmp_path / "var.tbl"), desc, nrow=3, readonly=False) as tb:
        tb.putcol("V", np.zeros((3, 16)))
        for n in (24, 8):
            tb.putcell("V", 1, np.arange(n, dtype=float))
            np.testing.assert_array_equal(tb.getcell("V", 1), np.arange(n))
        tb.putcol("V", np.ones((2, 5)), startrow=1)
        assert [tb.getcell("V", row).shape for row in range(3)] == [(16,), (5,), (5,)]
        np.testing.assert_array_equal(tb.getcol("V", startrow=1), np.ones((2, 5)))


def test_iswritable(tmp_path):
    path = str(tmp_path / "writable.tbl")
    desc = tables.maketabdesc([tables.makescacoldesc("S", 0)])
    tables.table(path, desc, nrow=1, readonly=False).close()
    with tables.table(path, readonly=False) as tb:
        assert tb.iswritable()
    with tables.table(path) as tb:
        assert not tb.iswritable()
