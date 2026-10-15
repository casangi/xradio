import os

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


def test_putcell_tiled_cell_column(tiled_cell_table, monkeypatch):
    path, data = tiled_cell_table
    new_value = (np.ones(SHAPE) * 42.0 * (1 - 1j)).astype(np.complex64)

    with tables.table(path, readonly=False) as tb:

        def _fail_taql(*args, **kwargs):
            raise AssertionError(
                "_put_rows_via_taql should not be called for fixed-shape tiled cell write"
            )

        monkeypatch.setattr(tb, "_put_rows_via_taql", _fail_taql)
        tb.putcell("DATA", 1, new_value)

    with tables.table(path) as tb:
        np.testing.assert_array_equal(tb.getcell("DATA", 1), new_value, strict=True)
        np.testing.assert_array_equal(tb.getcell("DATA", 0), data[0], strict=True)
        np.testing.assert_array_equal(tb.getcell("DATA", 2), data[2], strict=True)


@pytest.fixture
def tiled_shape_table(tmp_path):
    """A variable-shaped TiledShapeStMan column (as MS DATA), cells of SHAPE."""
    desc = tables.maketabdesc(
        [
            tables.makearrcoldesc(
                "DATA",
                np.complex64(0),
                ndim=len(SHAPE),
                datamanagertype="TiledShapeStMan",
                datamanagergroup="TSS",
            )
        ]
    )
    dminfo = {
        "*1": {
            "TYPE": "TiledShapeStMan",
            "NAME": "TSS",
            "COLUMNS": ["DATA"],
            "SPEC": {"DEFAULTTILESHAPE": [2, 3, 2]},
        }
    }
    data = (np.arange(NROWS * np.prod(SHAPE)).reshape(NROWS, *SHAPE) * (1 - 1j)).astype(
        np.complex64
    )
    path = str(tmp_path / "tiled_shape.tbl")
    with tables.table(path, desc, nrow=NROWS, dminfo=dminfo, readonly=False) as tb:
        tb.putcol("DATA", data)
        assert tb.getdminfo("DATA")["TYPE"] == "TiledShapeStMan"
    return path, data


@pytest.mark.parametrize("shape", [(3, 4), (6, 2), (7, 4)])
def test_putcell_tiled_cell_column_other_shape(tiled_cell_table, shape):
    """A value of another shape than the cell raises python-casacore's error
    and leaves the cell unchanged (a cell-sliced write would fill part of it)."""
    path, data = tiled_cell_table
    with tables.table(path, readonly=False) as tb:
        with pytest.raises(RuntimeError, match="Table array conformance error"):
            tb.putcell("DATA", 1, np.ones(shape, np.complex64))
    with tables.table(path) as tb:
        np.testing.assert_array_equal(tb.getcell("DATA", 1), data[1], strict=True)


def test_putcell_tiled_shape_column(tiled_shape_table):
    """casacore cannot reshape a defined TiledShapeStMan cell: putcell raises
    python-casacore's error (a TaQL update would abort the process); a value
    of the cell's shape is written in place."""
    path, data = tiled_shape_table
    with tables.table(path, readonly=False) as tb:
        with pytest.raises(RuntimeError, match="Table array conformance error"):
            tb.putcell("DATA", 1, np.ones((2, 3), np.complex64))
        tb.putcell("DATA", 2, data[2] * 2)
    with tables.table(path) as tb:
        np.testing.assert_array_equal(tb.getcell("DATA", 1), data[1], strict=True)
        np.testing.assert_array_equal(tb.getcell("DATA", 2), data[2] * 2, strict=True)


def test_putcell_tiled_shape_column_undefined_cell(tmp_path):
    """putcell defines an undefined TiledShapeStMan cell with the value's shape."""
    desc = tables.maketabdesc(
        [
            tables.makearrcoldesc(
                "DATA",
                np.complex64(0),
                ndim=2,
                datamanagertype="TiledShapeStMan",
                datamanagergroup="TSS",
            )
        ]
    )
    dminfo = {"*1": {"TYPE": "TiledShapeStMan", "NAME": "TSS", "COLUMNS": ["DATA"]}}
    path = str(tmp_path / "undefined.tbl")
    value = (np.arange(6).reshape(2, 3) * (1 + 1j)).astype(np.complex64)
    with tables.table(path, desc, nrow=NROWS, dminfo=dminfo, readonly=False) as tb:
        tb.putcell("DATA", 1, value)
    with tables.table(path) as tb:
        assert not tb.iscelldefined("DATA", 0)
        np.testing.assert_array_equal(tb.getcell("DATA", 1), value, strict=True)


def test_putcol_tiled_columns_keep_cell_shapes(tiled_cell_table, tiled_shape_table):
    """putcol raises python-casacore's error instead of aborting the process
    when it would change the shape of tiled cells, and writes nothing."""
    for path, data in (tiled_cell_table, tiled_shape_table):
        with tables.table(path, readonly=False) as tb:
            with pytest.raises(RuntimeError, match="Table array conformance error"):
                tb.putcol("DATA", np.ones((NROWS, 2, 3), np.complex64))
        with tables.table(path) as tb:
            for row in range(NROWS):
                np.testing.assert_array_equal(
                    tb.getcell("DATA", row), data[row], strict=True
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


# Defined cells of a shape with a zero dimension (rows 2 and 3) after defined
# (row 0) and undefined (row 1) cells, read as python-casacore 3.8.1 reads
# them (string cells as numpy arrays, like the shim's other string cells)
ZERO_SIZE_COLUMNS = {
    # name: (valuetype, ndim, value of row 0, Glish TaQL value of rows 2 and 3,
    #        their value)
    "D1": ("double", 1, np.arange(2.0), "array(0., [0])", np.empty((0,))),
    "I2": (
        "int",
        2,
        np.ones((2, 3), np.int32),
        "array(0, [3, 0])",
        np.empty((0, 3), np.int32),
    ),
    "C2": (
        "complex",
        2,
        np.full((2, 2), 1 + 2j, np.complex64),
        "array(0+0i, [0, 2])",
        np.empty((2, 0), np.complex64),
    ),
    "B1": ("boolean", 1, np.array([True, False]), "array(F, [0])", np.empty(0, bool)),
    "DV": ("double", -1, np.arange(3.0), "array(0., [2, 0])", np.empty((0, 2))),
    "S1": ("string", 1, np.array(["a", "b"]), "array('', [0])", np.empty(0, str)),
}


@pytest.mark.parametrize("col", ZERO_SIZE_COLUMNS)
def test_reads_of_defined_cells_without_values(tmp_path, col):
    valuetype, ndim, value, taql_value, empty = ZERO_SIZE_COLUMNS[col]
    desc = tables.maketabdesc(
        [tables.makearrcoldesc(col, 0.0, ndim=ndim, valuetype=valuetype)]
    )
    path = str(tmp_path / "zero_size.tbl")
    with tables.table(path, desc, nrow=4, readonly=False) as tb:
        tb.putcell(col, 0, value)
        tables.taql(
            f"UPDATE $tb SET {col} = {taql_value} WHERE ROWID() >= 2", style="Glish"
        )
    with tables.table(path) as tb:
        np.testing.assert_array_equal(tb.getcell(col, 2), empty, strict=True)
        np.testing.assert_array_equal(tb.col(col)[3], empty, strict=True)
        np.testing.assert_array_equal(tb.getcol(col, 3, 1), empty[None], strict=True)
        np.testing.assert_array_equal(
            tb.getcol(col, 2), np.stack([empty, empty]), strict=True
        )
        np.testing.assert_array_equal(tb.getcell(col, 0), value, strict=True)
        with pytest.raises(RuntimeError):
            tb.getcell(col, 1)
        with pytest.raises(RuntimeError):
            tb.getcol(col)


@pytest.mark.parametrize("col", VAR_COLUMNS)
def test_failed_reads_of_undefined_cells_do_not_read_cell_shapes(
    undefined_cells_table, col, monkeypatch
):
    """Only arcae's error for cells with a zero dimension makes getcol and
    getcell read the cell shapes."""

    def cell_shapes(*args):
        raise AssertionError("cell shapes read")

    with tables.table(undefined_cells_table) as tb:
        monkeypatch.setattr(tb, "_cell_shapes", cell_shapes)
        with pytest.raises(RuntimeError, match="not defined"):
            tb.getcell(col, 0)
        with pytest.raises(RuntimeError, match="not defined"):
            tb.getcol(col)


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


def test_query_name_writes_a_reference_table(tmp_path):
    path, ref = str(tmp_path / "plain.tbl"), str(tmp_path / "ref.tbl")
    desc = tables.maketabdesc([tables.makescacoldesc("S", 0)])
    with tables.table(path, desc, nrow=4, readonly=False) as tb:
        tb.putcol("S", np.arange(4, dtype=np.int32))
        tb.query("S > 1", name=ref).close()
    with tables.table(ref) as tb:
        np.testing.assert_array_equal(tb.getcol("S"), [2, 3])


def test_addcols_of_one_column_and_removerows(tmp_path):
    desc = tables.maketabdesc([tables.makescacoldesc("S", 0)])
    with tables.table(str(tmp_path / "t.tbl"), desc, nrow=4, readonly=False) as tb:
        tb.putcol("S", np.arange(4, dtype=np.int32))
        tb.addcols(tables.makearrcoldesc("A", 0.0, ndim=1))
        assert tb.colnames() == ["S", "A"]
        tb.removerows([])
        tb.removerows([0, 2])
        np.testing.assert_array_equal(tb.getcol("S"), [1, 3])


def test_default_ms_subtable_without_an_ms(tmp_path):
    path = str(tmp_path / "not_an_ms" / "POINTING")
    os.makedirs(os.path.dirname(path))
    extra = tables.maketabdesc([tables.makescacoldesc("OVER_THE_TOP", False)])
    with tables.default_ms_subtable("POINTING", path, extra) as tb:
        assert {"ANTENNA_ID", "TIME", "DIRECTION", "OVER_THE_TOP"} <= set(tb.colnames())
    assert not os.path.exists(os.path.join(os.path.dirname(path), "table.dat"))


# Columns of new tables, and the data managers python-casacore 3.8.1 binds
# them to from their descriptions: {NAME: (TYPE, COLUMNS)}
def _dm_test_desc():
    def tiled(name, dmtype, group, **kw):
        return tables.makearrcoldesc(
            name,
            0j,
            valuetype="complex",
            datamanagertype=dmtype,
            datamanagergroup=group,
            **kw,
        )

    return tables.maketabdesc(
        [
            tables.makescacoldesc("SCALAR", 0.0),
            tables.makescacoldesc(
                "ISM",
                0,
                valuetype="int",
                datamanagertype="IncrementalStMan",
                datamanagergroup="ISMData",
            ),
            tiled("TCSM", "TiledColumnStMan", "TCSMData", shape=[4, 2], options=4),
            tiled("TSM1", "TiledShapeStMan", "TSMData", ndim=2),
            tiled("TSM2", "TiledShapeStMan", "TSMData", ndim=2),
            tiled("TSMNOGROUP", "TiledShapeStMan", "", ndim=2),
        ]
    )


DESC_DATA_MANAGERS = {
    "StandardStMan": ("StandardStMan", ["SCALAR"]),
    "ISMData": ("IncrementalStMan", ["ISM"]),
    "TCSMData": ("TiledColumnStMan", ["TCSM"]),
    "TSMData": ("TiledShapeStMan", ["TSM1", "TSM2"]),
    "TiledShapeStMan": ("TiledShapeStMan", ["TSMNOGROUP"]),
}
DM_TEST_CELLS = (np.arange(10 * 8) * (1 - 1j)).astype(np.complex64).reshape(10, 4, 2)


def _data_managers(tb):
    return {
        dm["NAME"]: (dm["TYPE"], sorted(dm["COLUMNS"]))
        for dm in tb.getdminfo().values()
    }


def _tile_shapes(tb):
    return {
        dm["NAME"]: [list(hc["TileShape"]) for hc in dm["SPEC"]["HYPERCUBES"].values()]
        for dm in tb.getdminfo().values()
        if dm["TYPE"].startswith("Tiled")
    }


def _write_and_check_cells(path):
    with tables.table(path, readonly=False) as tb:
        for col in ("TCSM", "TSM1", "TSM2", "TSMNOGROUP"):
            tb.putcol(col, DM_TEST_CELLS)
    with tables.table(path) as tb:
        for col in ("TCSM", "TSM1", "TSM2", "TSMNOGROUP"):
            np.testing.assert_array_equal(tb.getcol(col), DM_TEST_CELLS, strict=True)


def test_new_table_columns_use_the_data_managers_of_their_descriptions(tmp_path):
    path = str(tmp_path / "desc.tbl")
    tables.table(path, _dm_test_desc(), nrow=10, readonly=False).close()
    _write_and_check_cells(path)
    with tables.table(path) as tb:
        assert _data_managers(tb) == DESC_DATA_MANAGERS
        assert tb.getcoldesc("TSM1")["dataManagerType"] == "TiledShapeStMan"


def test_new_table_dminfo_with_tile_shapes(tmp_path):
    desc = _dm_test_desc()
    dminfo = tables.makedminfo(
        {col: desc[col] for col in desc if col != "TSMNOGROUP"},
        {
            "TSMData": {"DEFAULTTILESHAPE": np.array([2, 4, 5], np.int32)},
            "TCSMData": {"DEFAULTTILESHAPE": np.array([2, 4, 8], np.int32)},
        },
    )
    path = str(tmp_path / "dminfo.tbl")
    tables.table(path, desc, nrow=10, dminfo=dminfo, readonly=False).close()
    _write_and_check_cells(path)
    with tables.table(path) as tb:
        assert _data_managers(tb) == DESC_DATA_MANAGERS
        assert _tile_shapes(tb) == {
            "TCSMData": [[2, 4, 8]],
            "TSMData": [[2, 4, 5]],
            "TiledShapeStMan": [[2, 4, 4096]],
        }


def test_new_table_dminfo_binds_columns_and_their_descriptions_the_rest(tmp_path):
    dminfo = {
        "*1": {
            "TYPE": "IncrementalStMan",
            "NAME": "ISM2",
            "SPEC": {},
            "COLUMNS": ["SCALAR"],
        },
        "*2": {
            "TYPE": "TiledShapeStMan",
            "NAME": "TSM1Data",
            "SPEC": {"DEFAULTTILESHAPE": np.array([2, 4, 3], np.int32)},
            "COLUMNS": ["TSM1"],
        },
    }
    path = str(tmp_path / "bound.tbl")
    tables.table(path, _dm_test_desc(), nrow=10, dminfo=dminfo, readonly=False).close()
    _write_and_check_cells(path)
    with tables.table(path) as tb:
        assert _data_managers(tb) == {
            "ISM2": ("IncrementalStMan", ["SCALAR"]),
            "ISMData": ("IncrementalStMan", ["ISM"]),
            "TCSMData": ("TiledColumnStMan", ["TCSM"]),
            "TSM1Data": ("TiledShapeStMan", ["TSM1"]),
            "TSMData": ("TiledShapeStMan", ["TSM2"]),
            "TiledShapeStMan": ("TiledShapeStMan", ["TSMNOGROUP"]),
        }
        assert _tile_shapes(tb)["TSM1Data"] == [[2, 4, 3]]


def test_table_recreated_from_getdesc_and_getdminfo(tmp_path):
    """getdminfo has empty arrays (DEFAULTTILESHAPE of TSMNOGROUP)."""
    path, again = str(tmp_path / "first.tbl"), str(tmp_path / "again.tbl")
    tables.table(path, _dm_test_desc(), nrow=10, readonly=False).close()
    _write_and_check_cells(path)
    with tables.table(path) as tb:
        desc, dminfo = tb.getdesc(), tb.getdminfo()
        tiles = _tile_shapes(tb)
    tables.table(again, desc, nrow=10, dminfo=dminfo, readonly=False).close()
    _write_and_check_cells(again)
    with tables.table(again) as tb:
        assert _data_managers(tb) == DESC_DATA_MANAGERS
        assert _tile_shapes(tb) == tiles


def test_new_table_dminfo_of_an_unknown_column_raises(tmp_path):
    dminfo = {
        "*1": {"TYPE": "StandardStMan", "NAME": "S", "SPEC": {}, "COLUMNS": ["NO"]}
    }
    with pytest.raises(RuntimeError, match="Table column NO is unknown"):
        tables.table(str(tmp_path / "t.tbl"), _dm_test_desc(), dminfo=dminfo)


def test_addcols_data_managers(tmp_path):
    """Without dminfo a column joins a storage manager that can add columns,
    as in python-casacore."""

    def column(name):
        return tables.makearrcoldesc(
            name,
            0.0,
            ndim=2,
            valuetype="float",
            datamanagertype="TiledShapeStMan",
            datamanagergroup=name,
        )

    path = str(tmp_path / "addcols.tbl")
    with tables.table(path, _dm_test_desc(), nrow=10, readonly=False) as tb:
        tb.addcols(column("NEW1"))
        tb.addcols(
            column("NEW2"),
            {"TYPE": "TiledShapeStMan", "NAME": "NEW2", "SPEC": {}},
        )
        tb.addcols(
            column("NEW3"),
            {
                "TYPE": "TiledShapeStMan",
                "NAME": "NEW3",
                "SPEC": {"DEFAULTTILESHAPE": np.array([2, 4, 5], np.int32)},
            },
        )
        # getdminfo gives an empty DEFAULTTILESHAPE for the default tile shape
        tb.addcols(
            column("NEW4"),
            {
                "TYPE": "TiledShapeStMan",
                "NAME": "NEW4",
                "SPEC": {"DEFAULTTILESHAPE": []},
            },
        )
        for col in ("NEW2", "NEW3", "NEW4"):
            tb.putcol(col, DM_TEST_CELLS.real)
        assert _data_managers(tb) == dict(
            DESC_DATA_MANAGERS,
            StandardStMan=("StandardStMan", ["NEW1", "SCALAR"]),
            NEW2=("TiledShapeStMan", ["NEW2"]),
            NEW3=("TiledShapeStMan", ["NEW3"]),
            NEW4=("TiledShapeStMan", ["NEW4"]),
        )
        tiles = _tile_shapes(tb)
        assert tiles["NEW3"] == [[2, 4, 5]]
        assert tiles["NEW4"] == tiles["NEW2"]


def test_default_ms_subtable_without_an_ms_data_managers(tmp_path):
    path = str(tmp_path / "not_an_ms" / "SOURCE")
    os.makedirs(os.path.dirname(path))
    extra = tables.maketabdesc(
        tables.makescacoldesc(
            "EXTRA",
            0,
            valuetype="int",
            datamanagertype="IncrementalStMan",
            datamanagergroup="ExtraISM",
        )
    )
    with tables.default_ms_subtable("SOURCE", path, extra) as tb:
        assert _data_managers(tb)["ExtraISM"] == ("IncrementalStMan", ["EXTRA"])


def test_iter_over_the_rows_of_each_value(tmp_path):
    desc = tables.maketabdesc(
        [tables.makescacoldesc("K", 0), tables.makescacoldesc("R", 0)]
    )
    with tables.table(str(tmp_path / "iter.tbl"), desc, nrow=6, readonly=False) as tb:
        tb.putcol("K", np.array([2, 1, 2, 2, 3, 1], dtype=np.int32))
        tb.putcol("R", np.arange(6, dtype=np.int32))
        groups = [(set(sub.getcol("K")), set(sub.getcol("R"))) for sub in tb.iter("K")]
        assert groups == [({1}, {1, 5}), ({2}, {0, 2, 3}), ({3}, {4})]
        groups = [list(sub.getcol("R")) for sub in tb.iter(["K"], sort=False)]
        assert groups == [[0], [1], [2, 3], [4], [5]]
        with pytest.raises(NotImplementedError, match="multiple columns"):
            next(tb.iter(["K", "R"]))
        np.testing.assert_array_equal(tb.rownumbers(), np.arange(6))
    with tables.table(str(tmp_path / "empty.tbl"), desc, readonly=False) as tb:
        assert list(tb.iter("K")) == []


def test_column_keywords(tmp_path):
    desc = tables.maketabdesc([tables.makescacoldesc("T", 0.0)])
    with tables.table(str(tmp_path / "kw.tbl"), desc, nrow=1, readonly=False) as tb:
        tb.putcolkeyword("T", "UNIT", "s")
        tb.putcolkeyword("T", "REF", np.array([1.0, 2.0]))
        keywords = tb.getcolkeywords("T")
        assert keywords["UNIT"] == "s"
        np.testing.assert_array_equal(keywords["REF"], [1.0, 2.0])
        tb.removecolkeyword("T", "UNIT")
        assert list(tb.getcolkeywords("T")) == ["REF"]


def test_tables_that_cannot_be_opened_or_created(tmp_path):
    path = str(tmp_path / "t.tbl")
    with pytest.raises(RuntimeError):
        tables.table(str(tmp_path / "missing.tbl"))
    desc = tables.maketabdesc([tables.makescacoldesc("S", 0)])
    with pytest.raises(ValueError, match="ninstances=1"):
        tables.table(path, desc, readonly=False, ninstances=2)
    tables.table(path, desc, nrow=1, readonly=False).close()
    with pytest.raises(RuntimeError, match="already exists"):
        tables.table(path, desc, readonly=False)


def test_unsupported_row_and_cell_increments(tmp_path):
    desc = tables.maketabdesc([tables.makearrcoldesc("A", 0.0, shape=[3])])
    with tables.table(str(tmp_path / "t.tbl"), desc, nrow=2, readonly=False) as tb:
        with pytest.raises(NotImplementedError):
            tb.getcol("A", rowincr=2)
        with pytest.raises(NotImplementedError):
            tb.putcol("A", np.zeros((1, 3)), rowincr=2)
        with pytest.raises(NotImplementedError):
            tb.getcolslice("A", [0], [1], rowincr=2)
        with pytest.raises(NotImplementedError):
            tb.getcolslice("A", [0], [2], inc=[2])


def test_keywords_dminfo_and_info(tmp_path):
    desc = tables.maketabdesc([tables.makescacoldesc("S", 0)])
    with tables.table(str(tmp_path / "t.tbl"), desc, nrow=1, readonly=False) as tb:
        with pytest.raises(RuntimeError, match="keyword NOPE does not exist"):
            tb.getkeyword("NOPE")
        tb.putkeyword("K", 1)
        assert "K" in tb.keywordnames()
        tb.removekeyword("K")
        assert "K" not in tb.keywordnames()
        group = tb.getdminfo("S")
        assert "S" in group["COLUMNS"]
        with pytest.raises(RuntimeError, match="not found"):
            tb.getdminfo("NOPE")
        assert tb.info() == {"type": "", "subType": "", "readme": ""}
        tb.putinfo({"type": "Measurement Set", "readme": "line one"})
        assert tb.info() == {
            "type": "Measurement Set",
            "subType": "",
            "readme": "line one\n",
        }


def test_taql_with_python_values_and_table_variables(tmp_path):
    desc = tables.maketabdesc(
        [tables.makescacoldesc("S", 0), tables.makescacoldesc("N", "")]
    )
    path = str(tmp_path / "t.tbl")
    with tables.table(path, desc, nrow=4, readonly=False) as tb:
        tb.putcol("S", np.arange(4, dtype=np.int32))
        tb.putcol("N", ["a", "b", "d", "e"])
    t = tables.table(path)
    values = {"low": 1, "name": "b", "rows": [0, 3]}
    for query, expected in (
        ("SELECT FROM $t WHERE S > $low", [2, 3]),
        ("SELECT FROM $t WHERE N == $name", [1]),
        ("SELECT FROM $t WHERE S IN $rows", [0, 3]),
    ):
        with tables.taql(query, locals=values) as result:
            np.testing.assert_array_equal(result.getcol("S"), expected)
    u = tables.table(path)
    with pytest.raises(NotImplementedError, match="more than one table"):
        tables.taql("SELECT FROM $t, $u")
    with pytest.raises(RuntimeError, match="empty IN list"):
        tables.taql("SELECT FROM $t WHERE S IN []")
    t.close()
    u.close()
