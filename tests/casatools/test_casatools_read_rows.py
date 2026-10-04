"""
The row reads of xradio on real casatools tables of downloaded test MSs:
read_rows, read_row_range, read_column_rows, read_rows_to_grid, MainTableRows
and the cell shape scan of check_partition_cells, through the casatools shim,
whose tables have no in-place reads (no getcolnp, getcolslicenp or
selectrows): the cells are read with getcol (getcell for a one-row table) and
copied. The python-casacore unit tests
(tests/unit/measurement_set/_utils/_msv2/_tables/test_read_rows.py) check the
in-place reads the same way.

The calls the reads make (getcol, getcell, getcolshapestring) are counted by
wrapping these methods of the opened table (``count_calls``): instrumentation
only, casatools reads the cells. The expected values are the cells read one by
one with getcell (``cells_by_getcell``), as the column dtype.

Checked:

- the values of row selections from one row to the whole table (runs of rows,
  fragmented rows, random rows) of scalar columns and of StandardStMan,
  IncrementalStMan, TiledColumnStMan and TiledShapeStMan array columns, and of
  channel / polarization ranges (whole cells read, then sliced);
- one getcol call per run of rows (no selectrows, also for fragmented rows)
  when a run fits in a call; every call reads rows after those of the previous
  call, at most max_elems elements and GETCOL_MAX_BYTES of cells (at least one
  row), and never the whole column (casacore's whole-column read does not
  check for undefined cells); a one-row table is read with getcell;
- the values are of the column dtype (casatools returns Float, Complex and Int
  cells as float64, complex128 and int64);
- undefined cells and cells of another shape raise RuntimeError;
- read_rows_to_grid fills the grid as numpy's fancy assignment of all the rows
  (of duplicated cells the last row wins, also around the direct segments
  copied as slices), reads fragmented rows with one getcol per run, bridges
  small gaps, and reads the partition rows again without the gap when a
  bridged gap holds cells of another shape;
- check_partition_cells compares the cell shapes with one getcolshapestring
  call per run of rows.

Not covered (no downloadable test MS has a MAIN column with both defined and
undefined cells, see tests/README.md): reads across undefined cells between
defined ones, a bridged gap over undefined cells, the shape scan finding an
undefined cell after a defined first cell.

Skipped where casatools is not installed or python-casacore is (the Linux and
macOS workflows), see reference.skip_unless_casatools_backend.
"""

import contextlib

import numpy as np
import pytest

from tests.casatools import reference as ref

ref.skip_unless_casatools_backend()

from xradio._utils._casacore.tables import open_table_ro  # noqa: E402
from xradio.measurement_set._utils._msv2._tables import read_rows as rr  # noqa: E402

# MAIN columns read by test_read_rows: (MS, column)
READ_COLUMNS = [
    (ref.LOFAR, "ANTENNA1"),  # Int scalar
    (ref.LOFAR, "TIME"),  # Double scalar
    (ref.LOFAR, "UVW"),  # Double [3], TiledColumnStMan
    (ref.LOFAR, "WEIGHT"),  # Float [4], IncrementalStMan
    (ref.LOFAR, "DATA"),  # Complex [15, 4], TiledColumnStMan
    (ref.LOFAR, "FLAG"),  # Bool [15, 4], TiledColumnStMan
    # Complex [32, 4], TiledShapeStMan; the whole column is over GETCOL_MAX_BYTES
    (ref.VLBI, "DATA"),
    (ref.VLBI, "WEIGHT_SPECTRUM"),  # Float [32, 4], TiledShapeStMan
    (ref.SD_STANDARD, "FLOAT_DATA"),  # Float [1024, 2], StandardStMan
]
SELECTIONS = ("all", "one_row", "few_runs", "fragmented", "random")


def row_selection(name: str, nrows: int) -> np.ndarray:
    """Rows of a table of ``nrows`` rows."""
    if name == "all":
        return np.arange(nrows)
    if name == "one_row":
        return np.array([17])
    if name == "few_runs":
        return np.r_[3:40, 50:51, 60:120, nrows - 1 : nrows]
    if name == "fragmented":
        # 100 runs of one row: more than FRAGMENTED_RUNS (selectrows with
        # in-place reads)
        return np.arange(0, 200, 2)
    if name == "random":
        return np.sort(np.random.default_rng(1).choice(nrows, 77, replace=False))
    raise ValueError(name)


@pytest.fixture(scope="module")
def tables():
    """
    ``tables(ms_name, subtable=None)``: the MAIN table (or a sub-table) of a
    test MS, opened read-only as the converter opens it (open_table_ro), once
    per module.
    """
    opened = {}
    with contextlib.ExitStack() as stack:

        def get(ms_name: str, subtable: str | None = None):
            key = (ms_name, subtable)
            if key not in opened:
                path = str(ref.ms_path(ms_name))
                if subtable:
                    path = f"{path}/{subtable}"
                opened[key] = stack.enter_context(open_table_ro(path))
            return opened[key]

        yield get


def cells_by_getcell(table, col: str, rows) -> np.ndarray:
    """The cells of ``rows`` read one by one with getcell, as the column
    dtype (exact: casatools' wider values hold the stored ones)."""
    dtype = rr.column_dtype(table, col)
    cells = [np.asarray(table.getcell(col, int(row))) for row in rows]
    return np.array(cells).astype(dtype, copy=False)


def unlike(values: np.ndarray) -> np.ndarray:
    """A C-contiguous array of the shape and dtype of ``values`` that differs
    from it in every element (a read must overwrite all of it)."""
    if values.dtype == np.bool_:
        return ~values
    return np.where(values == 1, 2, 1).astype(values.dtype)


class TableCalls:
    """The calls made to one table while counted (``count_calls``)."""

    def __init__(self) -> None:
        # (start row, number of rows) of every getcol call that succeeded
        self.getcol: list[tuple[int, int]] = []
        # (start row, number of rows) of every getcol call that raised
        self.getcol_failed: list[tuple[int, int]] = []
        self.getcell = 0
        # the arguments of every getcolshapestring call
        self.getcolshapestring: list[tuple] = []

    def rows_read(self) -> np.ndarray:
        """The rows of the getcol calls that succeeded, in call order."""
        ranges = [np.arange(start, start + n) for start, n in self.getcol]
        return np.concatenate(ranges) if ranges else np.empty(0, dtype=np.int64)


@contextlib.contextmanager
def count_calls(table):
    """
    Count the getcol, getcell and getcolshapestring calls made to ``table``
    (a casatools shim table) inside the ``with`` block: the methods are
    wrapped on the instance and restored after it.
    """
    calls = TableCalls()
    getcol, getcell = table.getcol, table.getcell
    getcolshapestring = table.getcolshapestring

    def counted_getcol(col, startrow=0, nrow=-1, rowincr=1):
        try:
            values = getcol(col, startrow, nrow, rowincr)
        except Exception:
            calls.getcol_failed.append((int(startrow), int(nrow)))
            raise
        calls.getcol.append((int(startrow), len(values)))
        return values

    def counted_getcell(*args, **kwargs):
        calls.getcell += 1
        return getcell(*args, **kwargs)

    def counted_getcolshapestring(*args, **kwargs):
        calls.getcolshapestring.append(args)
        return getcolshapestring(*args, **kwargs)

    wrapped = {
        "getcol": counted_getcol,
        "getcell": counted_getcell,
        "getcolshapestring": counted_getcolshapestring,
    }
    for name, method in wrapped.items():
        setattr(table, name, method)
    try:
        yield calls
    finally:
        for name in wrapped:
            delattr(table, name)


def assert_bounded_calls(
    calls: TableCalls, table, rows_per_call: int, row_bytes: int
) -> None:
    """
    Every getcol call reads rows after those of the previous call, at most
    ``rows_per_call`` rows (max_elems) and GETCOL_MAX_BYTES of cells of
    ``row_bytes`` (at least one row), and never the whole column.
    """
    nrows = table.nrows()
    end = 0
    for start, n in calls.getcol:
        assert start >= end, calls.getcol
        assert 1 <= n <= rows_per_call, (start, n)
        assert n * row_bytes <= max(row_bytes, rr.GETCOL_MAX_BYTES), (start, n)
        assert (start, n) != (0, nrows), "a getcol call over the whole column"
        end = start + n


# --- read_rows, read_row_range, read_column_rows -----------------------------


@pytest.mark.parametrize("ms, col", READ_COLUMNS)
@pytest.mark.parametrize("selection", SELECTIONS)
@pytest.mark.parametrize("cells_per_call", [None, 3])
def test_read_rows(tables, ms, col, selection, cells_per_call):
    """
    read_rows reads the cells of the rows with one getcol call per run of
    rows (also for fragmented rows: no selectrows), in calls bounded by
    max_elems (here the default, or 3 cells) and GETCOL_MAX_BYTES, never over
    the whole column, into the buffer of the column dtype.
    """
    tb = tables(ms)
    rows = row_selection(selection, tb.nrows())
    expected = cells_by_getcell(tb, col, rows)
    cell_elems = int(np.prod(expected.shape[1:], dtype=np.int64)) or 1
    if cells_per_call is None:
        max_elems = rr.DEFAULT_MAX_ELEMS
    else:
        max_elems = cells_per_call * cell_elems
    out = unlike(expected)
    with count_calls(tb) as calls:
        stats = rr.read_rows(tb, col, rows, out, max_elems=max_elems)
    np.testing.assert_array_equal(out, expected)
    assert stats == {"calls": len(calls.getcol)}  # no selectrows
    assert calls.getcell == 0 and not calls.getcol_failed
    np.testing.assert_array_equal(calls.rows_read(), rows)
    row_bytes = cell_elems * out.itemsize
    assert_bounded_calls(calls, tb, max(1, max_elems // cell_elems), row_bytes)
    longest_run = int(rr.rows_to_runs(rows)[1].max())
    if (
        longest_run * cell_elems <= max_elems
        and longest_run * row_bytes <= rr.GETCOL_MAX_BYTES
    ):
        # one call per run (two for all the rows: never the whole column)
        whole_column = rows.size == tb.nrows()
        assert len(calls.getcol) == rr.count_row_runs(rows) + whole_column


def test_read_rows_bounded_getcol_calls(tables, monkeypatch):
    """A run of rows is read with getcol calls of at most GETCOL_MAX_BYTES of
    cells (here 7 cells), also within a call of max_elems elements."""
    tb = tables(ref.LOFAR)
    rows = np.arange(150)
    expected = cells_by_getcell(tb, "DATA", rows)
    monkeypatch.setattr(rr, "GETCOL_MAX_BYTES", 7 * expected[0].nbytes)
    out = unlike(expected)
    with count_calls(tb) as calls:
        stats = rr.read_rows(tb, "DATA", rows, out)
    np.testing.assert_array_equal(out, expected)
    assert calls.getcol == [(start, 7) for start in range(0, 147, 7)] + [(147, 3)]
    assert stats == {"calls": 22}


@pytest.mark.parametrize(
    "ms, col", [(ref.LOFAR, "DATA"), (ref.VLBI, "WEIGHT_SPECTRUM")]
)
@pytest.mark.parametrize("selection", ["all", "few_runs", "fragmented", "one_row"])
@pytest.mark.parametrize(
    "chan, pol",
    [(slice(1, 4), None), (None, slice(1, 2)), (slice(5, 6), slice(0, 1))],
)
@pytest.mark.parametrize("max_elems", [7, rr.DEFAULT_MAX_ELEMS])
def test_read_rows_cell_slice(tables, ms, col, selection, chan, pol, max_elems):
    """A channel / polarization range: whole cells are read with getcol (at
    most max_elems elements of the ranges per call) and sliced."""
    tb = tables(ms)
    rows = row_selection(selection, tb.nrows())
    cells = cells_by_getcell(tb, col, rows)
    expected = np.ascontiguousarray(
        cells[:, chan if chan else slice(None), pol if pol else slice(None)]
    )
    out = unlike(expected)
    with count_calls(tb) as calls:
        stats = rr.read_rows(
            tb, col, rows, out, chan=chan, pol=pol, max_elems=max_elems
        )
    np.testing.assert_array_equal(out, expected)
    assert stats == {"calls": len(calls.getcol)} and calls.getcell == 0
    np.testing.assert_array_equal(calls.rows_read(), rows)
    range_elems = int(np.prod(expected.shape[1:]))
    assert_bounded_calls(
        calls, tb, max(1, max_elems // range_elems), cells[0].size * out.itemsize
    )


@pytest.mark.parametrize("col", ["WEIGHT", "SIGMA", "FLAG_CATEGORY"])
def test_read_rows_undefined_cells_raise(tables, col):
    """
    Undefined cells raise RuntimeError (casatools' getcol raises): no cell of
    WEIGHT, SIGMA and FLAG_CATEGORY of uid___A002_Xe3a5fd_Xe38e is defined,
    whose other columns read.
    """
    tb = tables(ref.SD_NO_WEIGHT)
    assert not tb.iscelldefined(col, 0)
    for rows in (
        np.array([5]),
        np.arange(10, 20),
        row_selection("fragmented", tb.nrows()),
        np.arange(tb.nrows()),
    ):
        out = np.zeros((rows.size, 2), rr.column_dtype(tb, col))
        with pytest.raises(RuntimeError):
            rr.read_rows(tb, col, rows, out)
    rows = np.arange(10, 20)
    expected = cells_by_getcell(tb, "FLOAT_DATA", rows)
    out = unlike(expected)
    rr.read_rows(tb, "FLOAT_DATA", rows, out)
    np.testing.assert_array_equal(out, expected)


def test_read_rows_other_cell_shapes_raise(tables):
    """
    Cells of another shape than the buffer's raise RuntimeError: rows of the
    TiledShapeStMan DATA column of the ALMA MS whose cell shape changes (rows
    of another DATA_DESC_ID), and a buffer of another cell shape.
    """
    tb = tables(ref.ALMA)
    shapes = tb.getcolshapestring("DATA", 0, 1000)
    change = next(row for row, shape in enumerate(shapes) if shape != shapes[0])
    cell_shape = rr.parse_shape_string(shapes[0])
    rows = np.arange(change - 3, change)
    expected = cells_by_getcell(tb, "DATA", rows)
    assert expected.shape[1:] == cell_shape
    out = unlike(expected)
    rr.read_rows(tb, "DATA", rows, out)
    np.testing.assert_array_equal(out, expected)
    rows = np.arange(change - 3, change + 3)
    with pytest.raises(RuntimeError):
        rr.read_rows(tb, "DATA", rows, np.zeros((rows.size,) + cell_shape, out.dtype))
    with pytest.raises(RuntimeError, match="shape"):
        rr.read_rows(
            tables(ref.LOFAR), "DATA", np.arange(5), np.zeros((5, 15, 1), out.dtype)
        )


def test_read_rows_one_row_table(tables):
    """
    The cell of a one-row table is read with getcell (a getcol of its row
    would be casacore's whole-column read): the POLARIZATION and FIELD tables
    of small_lofar.ms have one row.
    """
    pol_tb = tables(ref.LOFAR, "POLARIZATION")
    assert pol_tb.nrows() == 1
    cell = cells_by_getcell(pol_tb, "CORR_PRODUCT", [0])[0]  # Int [4, 2]
    num_corr = pol_tb.getcell("NUM_CORR", 0)
    with count_calls(pol_tb) as calls:
        out = unlike(cell[np.newaxis])
        stats = rr.read_rows(pol_tb, "CORR_PRODUCT", np.array([0]), out)
        np.testing.assert_array_equal(out[0], cell)
        assert stats == {"calls": 1}
        out = unlike(cell[np.newaxis, 1:3, 1:2])
        rr.read_rows(
            pol_tb, "CORR_PRODUCT", [0], out, chan=slice(1, 3), pol=slice(1, 2)
        )
        np.testing.assert_array_equal(out[0], cell[1:3, 1:2])
        values = rr.read_column_rows(pol_tb, "NUM_CORR", np.array([0]))
        assert values.dtype == np.int32 and values.tolist() == [num_corr]
    assert calls.getcol == [] and calls.getcell == 3

    field_tb = tables(ref.LOFAR, "FIELD")
    assert field_tb.nrows() == 1
    name = field_tb.getcell("NAME", 0)
    with count_calls(field_tb) as calls:
        assert list(rr.read_column_rows(field_tb, "NAME", np.array([0]))) == [name]
    assert calls.getcol == [] and calls.getcell == 1


def test_read_column_rows(tables):
    """
    read_column_rows gives the values in the column dtype (that of
    python-casacore's getcol, not casatools' wider one), with one getcol call
    per run of rows; string columns are read with getcol calls of at most
    max_elems cells.
    """
    tb = tables(ref.LOFAR)
    rows = row_selection("few_runs", tb.nrows())
    for col in ("ANTENNA1", "TIME", "UVW", "WEIGHT", "DATA", "FLAG"):
        expected = cells_by_getcell(tb, col, rows)
        with count_calls(tb) as calls:
            values = rr.read_column_rows(tb, col, rows)
        np.testing.assert_array_equal(values, expected)
        assert values.dtype == rr.column_dtype(tb, col), col
        assert len(calls.getcol) == rr.count_row_runs(rows), col

    ant_tb = tables(ref.LOFAR, "ANTENNA")
    last = ant_tb.nrows() - 1
    rows = np.r_[0:5, 10:12, last]
    expected = [ant_tb.getcell("NAME", int(row)) for row in rows]
    with count_calls(ant_tb) as calls:
        names = rr.read_column_rows(ant_tb, "NAME", rows, max_elems=4)
    assert list(names) == expected
    assert calls.getcol == [(0, 4), (4, 1), (10, 2), (last, 1)]


def test_read_row_range(tables):
    """read_row_range reads consecutive rows in getcol calls of at most
    max_elems elements, never over the whole column."""
    tb = tables(ref.LOFAR)
    expected = cells_by_getcell(tb, "ANTENNA1", np.arange(200))
    out = unlike(expected)
    with count_calls(tb) as calls:
        stats = rr.read_row_range(tb, "ANTENNA1", 0, 200, out, max_elems=64)
    np.testing.assert_array_equal(out, expected)
    assert calls.getcol == [(0, 64), (64, 64), (128, 64), (192, 8)]
    assert stats == {"calls": 4}

    nrows = tb.nrows()
    expected = cells_by_getcell(tb, "ANTENNA1", np.arange(nrows))
    out = unlike(expected)
    with count_calls(tb) as calls:
        rr.read_row_range(tb, "ANTENNA1", 0, nrows, out)
    np.testing.assert_array_equal(out, expected)
    assert calls.getcol == [(0, nrows - 1), (nrows - 1, 1)]

    expected = cells_by_getcell(tb, "DATA", np.arange(30, 35))
    out = unlike(expected)
    rr.read_row_range(tb, "DATA", 30, 5, out)
    np.testing.assert_array_equal(out, expected)


def test_main_table_rows(tables):
    """MainTableRows (a partition of the MAIN table) reads its rows in the
    column dtype."""
    tb = tables(ref.LOFAR)
    rows = row_selection("random", tb.nrows())
    main_rows = rr.MainTableRows(tb, rows)
    values = main_rows.getcol("ANTENNA1")
    np.testing.assert_array_equal(values, cells_by_getcell(tb, "ANTENNA1", rows))
    assert values.dtype == np.int32
    values = main_rows.getcol("DATA", 3, 4)
    np.testing.assert_array_equal(values, cells_by_getcell(tb, "DATA", rows[3:7]))
    assert values.dtype == np.complex64
    assert main_rows.getcell("TIME", 5) == tb.getcell("TIME", int(rows[5]))


# --- read_rows_to_grid ------------------------------------------------------------


def make_plan_case(seed, first_row, nt=9, nb=6, keep=0.8, n_dup=0, shuffle_cells=False):
    """
    Rows (ascending from about ``first_row``, runs separated by gaps) and
    their grid cells: time-major cells (or shuffled ones, as for a
    baseline-major MS), some cells missing, optionally duplicated cells.
    """
    rng = np.random.default_rng(seed)
    cells = np.arange(nt * nb)
    if shuffle_cells:
        cells = rng.permutation(cells)
    cells = cells[rng.random(cells.size) < keep]
    if n_dup:
        src = rng.integers(0, cells.size, n_dup)
        cells = np.insert(cells, rng.integers(0, cells.size, n_dup), cells[src])
    n = cells.size
    gaps = rng.random(n) < 0.15
    rows = first_row + np.cumsum(1 + gaps * rng.integers(1, 4, n)) - 1
    return rows, cells, nt, nb


def grid_case(table, col, rows, cells, nt, nb):
    """
    (grid to read into, expected grid): the grid differs from the expected
    one in every cell that receives a row; the expected one is numpy's fancy
    assignment of all the rows (of duplicated cells the last row wins).
    """
    values = cells_by_getcell(table, col, rows)
    cell_shape = values.shape[1:]
    assigned = np.zeros((nt * nb,) + cell_shape, values.dtype)
    assigned[cells] = values
    grid = unlike(assigned).reshape((nt, nb) + cell_shape)
    expected = grid.copy()
    expected.reshape((nt * nb,) + cell_shape)[cells] = values
    return grid, expected


def assert_one_ascending_pass(calls: TableCalls, rows: np.ndarray) -> int:
    """The getcol calls read ascending rows (each at most once), all the
    partition rows among them; returns the number of other (bridged) rows."""
    read = calls.rows_read()
    assert np.all(np.diff(read) > 0)
    assert np.isin(rows, read).all()
    return read.size - rows.size


@pytest.mark.parametrize("seed", range(2))
@pytest.mark.parametrize("n_dup", [0, 5])
@pytest.mark.parametrize("shuffle_cells", [False, True])
@pytest.mark.parametrize(
    "ms, col",
    [
        (ref.LOFAR, "DATA"),
        (ref.LOFAR, "FLAG"),
        (ref.LOFAR, "WEIGHT"),
        (ref.LOFAR, "TIME"),
        (ref.VLBI, "DATA"),
    ],
)
@pytest.mark.parametrize("min_read_bytes", [0, None])
def test_read_rows_to_grid(tables, seed, n_dup, shuffle_cells, ms, col, min_read_bytes):
    """
    The grid is the fancy assignment of all the rows (direct segments read
    straight into it with min_read_bytes 0, the other rows through the
    temporary, duplicated cells), read in one ascending pass of bounded getcol
    calls (no selectrows).
    """
    tb = tables(ms)
    rows, cells, nt, nb = make_plan_case(
        seed, 1000 * seed + 7, n_dup=n_dup, shuffle_cells=shuffle_cells
    )
    grid, expected = grid_case(tb, col, rows, cells, nt, nb)
    plan = rr.make_row_grid_plan(rows, cells, nt * nb, min_direct_rows=4)
    kw = {} if min_read_bytes is None else {"min_read_bytes": min_read_bytes}
    with count_calls(tb) as calls:
        stats = rr.read_rows_to_grid(tb, col, plan, grid, **kw)
    np.testing.assert_array_equal(grid, expected)
    assert "selectrows_calls" not in stats
    assert stats["calls"] == len(calls.getcol) and not calls.getcol_failed
    assert stats.get("direct_rows", 0) + stats.get("scatter_rows", 0) == rows.size
    if min_read_bytes == 0:
        assert stats.get("direct_rows", 0) == plan.direct_lengths.sum()
    assert assert_one_ascending_pass(calls, rows) == stats.get("gap_rows", 0)
    cell_elems = int(np.prod(grid.shape[2:], dtype=np.int64)) or 1
    assert_bounded_calls(
        calls,
        tb,
        max(1, rr.DEFAULT_MAX_ELEMS // cell_elems),
        cell_elems * grid.itemsize,
    )


def read_reversed_cells(table, col, rows, **kwargs):
    """read_rows_to_grid of ``rows`` into cells in the reverse row order (no
    direct segment: all rows go through the temporary); returns the stats and
    the calls."""
    cells = np.arange(rows.size)[::-1]
    grid, expected = grid_case(table, col, rows, cells, rows.size, 1)
    plan = rr.make_row_grid_plan(rows, cells, rows.size)
    with count_calls(table) as calls:
        stats = rr.read_rows_to_grid(table, col, plan, grid, **kwargs)
    np.testing.assert_array_equal(grid, expected)
    return stats, calls


def test_read_rows_to_grid_fragmented_and_bridged(tables):
    """
    A fragmented temporary is read with one getcol call per run (one
    selectrows call with in-place reads); runs separated by small gaps of
    other rows are read with one call, the gap rows discarded.
    """
    tb = tables(ref.LOFAR)
    # 19 runs of 8 rows, gaps of 2 rows (over MAX_GAP_FRACTION of a run)
    rows = rr.runs_to_rows(np.arange(0, 190, 10), np.full(19, 8))
    stats, calls = read_reversed_cells(tb, "DATA", rows)
    assert calls.getcol == [(start, 8) for start in range(0, 190, 10)]
    assert "selectrows_calls" not in stats and "gap_rows" not in stats
    # gaps of 2 and 1 rows between runs of 60+ rows: one call over rows 0-198
    rows = np.r_[0:60, 62:130, 131:199]
    stats, calls = read_reversed_cells(tb, "DATA", rows)
    assert calls.getcol == [(0, 199)] and stats["gap_rows"] == 3


def test_read_rows_to_grid_bridge_fallback_other_shape(tables, monkeypatch):
    """
    A bridged gap of rows whose cells have another shape (in the
    TiledShapeStMan DATA column of the ALMA MS, runs of 2652 rows of one
    DATA_DESC_ID separated by 255 rows of two others) makes the getcol call
    fail: the partition rows are read again without the gap, and no other gap
    of the column is bridged (the temporary holds 4000 rows: every temporary
    spans a gap).
    """
    monkeypatch.setattr(rr, "MAX_GAP_FRACTION", 1.0)
    tb = tables(ref.ALMA)
    shapes = np.array(tb.getcolshapestring("DATA", 0, 9000))
    values, counts = np.unique(shapes, return_counts=True)
    rows = np.flatnonzero(shapes == values[np.argmax(counts)])
    assert rr.count_row_runs(rows) >= 3
    row_bytes = cells_by_getcell(tb, "DATA", rows[:1])[0].nbytes
    stats, calls = read_reversed_cells(tb, "DATA", rows, max_tmp_bytes=4000 * row_bytes)
    assert stats["bridge_fallbacks"] == 1 and "gap_rows" not in stats
    # one failed call, over the first gap; the other calls read partition rows
    [(start, n)] = calls.getcol_failed
    assert start == rows[0] and not np.isin(np.arange(start, start + n), rows).all()
    np.testing.assert_array_equal(calls.rows_read(), rows)


def test_read_rows_to_grid_last_row_wins_around_slice_copies(tables, monkeypatch):
    """
    In one temporary, the rows scattered before, between and after the direct
    segments copied as slices keep the order of the rows: of duplicated
    (time, baseline) cells the last row wins (rows 0 and 2 both scattered
    before the first slice copy, rows 1 and 17, 3 and 31 on either side of a
    slice copy).
    """
    tb = tables(ref.LOFAR)
    row_bytes = cells_by_getcell(tb, "DATA", [0])[0].nbytes
    monkeypatch.setattr(rr, "MIN_SLICE_COPY_BYTES", 8 * row_bytes)
    nt, nb = 6, 10
    cells = np.r_[[3, 7, 3, 9], 20:32, [40, 7], 45:58, [9, 58]]
    rows = 50 + np.arange(cells.size)
    plan = rr.make_row_grid_plan(rows, cells, nt * nb, min_direct_rows=4)
    np.testing.assert_array_equal(plan.direct_offsets, [4, 18])
    np.testing.assert_array_equal(plan.direct_lengths, [12, 13])
    assert plan.n_duplicate_rows == 6
    grid, expected = grid_case(tb, "DATA", rows, cells, nt, nb)
    with count_calls(tb) as calls:
        stats = rr.read_rows_to_grid(
            tb, "DATA", plan, grid, min_read_bytes=100 * row_bytes
        )
    # one temporary (one getcol call): both direct segments copied from it
    assert calls.getcol == [(50, rows.size)]
    assert stats.get("direct_rows", 0) == 0 and stats["scatter_rows"] == rows.size
    np.testing.assert_array_equal(grid, expected)
    values = cells_by_getcell(tb, "DATA", rows)
    for cell, row in ((3, 2), (7, 17), (9, 31)):
        np.testing.assert_array_equal(grid[cell // nb, cell % nb], values[row])


# --- check_partition_cells --------------------------------------------------------


def test_check_partition_cells_shape_scan(tables):
    """
    check_partition_cells of a TiledShapeStMan column whose hypercubes have
    several cell shapes (DATA of the ALMA MS) compares the cell shapes of the
    partition rows with one getcolshapestring call per run (no selectrows,
    also for fragmented rows), and finds a cell of another shape. The
    converter cannot describe the storage of casatools tables (no
    ``partnames``: column_storage describes the error, the read decides), so
    the storage is described here from casatools' getdminfo.
    """
    tb = tables(ref.ALMA)
    assert rr.column_storage(tb, "DATA").error
    storage = rr.column_storage(tb, "DATA", table_dminfo=tb.getdminfo(), plain=True)
    assert storage.dm_type == "TiledShapeStMan" and not storage.error
    assert len(storage.hypercubes) > 1
    shapes = np.array(tb.getcolshapestring("DATA", 0, 2000))
    same = np.flatnonzero(shapes == shapes[0])
    fragmented = same[::2]  # runs of one row
    assert rr.count_row_runs(fragmented) > rr.FRAGMENTED_RUNS
    with count_calls(tb) as calls:
        check = rr.check_partition_cells(tb, "DATA", fragmented, storage)
    assert check.verified and "compared" in check.how
    # the first cell, then one call per run
    assert len(calls.getcolshapestring) == 1 + rr.count_row_runs(fragmented)
    other = np.r_[same[:10], np.flatnonzero(shapes != shapes[0])[:1]]
    with pytest.raises(rr.ColumnNotReadableError, match="shape other than"):
        rr.check_partition_cells(tb, "DATA", np.sort(other), storage)
