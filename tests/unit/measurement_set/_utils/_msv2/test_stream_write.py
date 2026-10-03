import types

import numpy as np
import pytest

tables = pytest.importorskip("casacore.tables")

from xradio.measurement_set._utils._msv2 import stream_write as sw  # noqa: E402
from xradio.measurement_set._utils._msv2._tables import read_rows as rr  # noqa: E402

NROWS = 300
NCHAN = 4
NPOL = 2
# rows of the *_UNDEF columns that are written (rows 40-49 stay undefined)
UNDEF_DEFINED_ROWS = np.r_[0:40, 50:NROWS]
# rows of the TSM_TWO_SHAPES / SSM_VAR columns with a second cell shape
OTHER_SHAPE_ROWS = np.r_[250:NROWS]


def test_get_stream_write_mode(monkeypatch):
    monkeypatch.delenv(sw.STREAM_WRITE_ENV_VAR, raising=False)
    assert sw.get_stream_write_mode() is True
    for value, expected in (("0", False), ("1", True), (" 0 ", False), ("", True)):
        monkeypatch.setenv(sw.STREAM_WRITE_ENV_VAR, value)
        assert sw.get_stream_write_mode() is expected
    monkeypatch.setenv(sw.STREAM_WRITE_ENV_VAR, "yes")
    with pytest.raises(ValueError, match=sw.STREAM_WRITE_ENV_VAR):
        sw.get_stream_write_mode()


def test_get_stream_batch_bytes(monkeypatch):
    monkeypatch.delenv(sw.STREAM_BATCH_MB_ENV_VAR, raising=False)
    assert sw.get_stream_batch_bytes() == sw.DEFAULT_STREAM_BATCH_MB * 2**20
    monkeypatch.setenv(sw.STREAM_BATCH_MB_ENV_VAR, "16")
    assert sw.get_stream_batch_bytes() == 16 * 2**20
    monkeypatch.setenv(sw.STREAM_BATCH_MB_ENV_VAR, "1e-9")
    assert sw.get_stream_batch_bytes() == 1  # at least one byte (one chunk)
    for value in ("0", "-3", "abc", "nan"):
        monkeypatch.setenv(sw.STREAM_BATCH_MB_ENV_VAR, value)
        with pytest.raises(ValueError, match=sw.STREAM_BATCH_MB_ENV_VAR):
            sw.get_stream_batch_bytes()


@pytest.mark.parametrize(
    "n_times, time_chunk, bytes_per_time, target, expected",
    [
        # variable smaller than the target: one batch
        (23, 4, 10, 10**6, (23,)),
        # exactly one chunk per batch (target below a chunk), uneven last chunk
        (23, 4, 10, 1, (4, 4, 4, 4, 4, 3)),
        # two chunks per batch, last batch the remaining (partial) chunks
        (23, 4, 10, 80, (8, 8, 7)),
        (24, 4, 10, 80, (8, 8, 8)),
        # one chunk per variable (main_chunksize=None) or a chunk longer than the
        # time axis: one batch whatever the target
        (23, 23, 10, 1, (23,)),
        (10, 1000, 10, 1, (10,)),
        # one time per chunk
        (5, 1, 10, 25, (2, 2, 1)),
        (0, 4, 10, 80, ()),
    ],
)
def test_time_batches(n_times, time_chunk, bytes_per_time, target, expected):
    batches = sw.time_batches(n_times, time_chunk, bytes_per_time, target)
    assert batches == expected
    assert sum(batches) == n_times
    # batch boundaries are zarr chunk boundaries
    bounds = np.cumsum((0,) + batches)[:-1]
    assert np.all(bounds % min(time_chunk, max(n_times, 1)) == 0)


def test_count_row_runs():
    assert sw.count_row_runs(np.array([], dtype=np.int64)) == 0
    assert sw.count_row_runs(np.array([5])) == 1
    assert sw.count_row_runs(np.array([0, 1, 2, 7, 8, 20])) == 3


def _layout(nt, nb, baseline_major, first_row=0):
    """Partition rows (all cells) of a time-major or baseline-major MAIN table."""
    tidx, bidx = np.divmod(np.arange(nt * nb), nb)
    if baseline_major:
        bidx, tidx = np.divmod(np.arange(nt * nb), nt)
    return np.arange(nt * nb) + first_row, tidx, bidx


@pytest.mark.parametrize("baseline_major", [False, True])
def test_choose_time_batches_guard(baseline_major, monkeypatch):
    nt, nb, time_chunk, bytes_per_time = 40, 6, 4, 100
    rows, tidx, bidx = _layout(nt, nb, baseline_major)
    main_rows = types.SimpleNamespace(rows=rows)
    runs_whole = sw.count_row_runs(rows)

    def choose(target):
        return sw.choose_time_batches(
            main_rows,
            tidx,
            bidx,
            nb,
            nt,
            time_chunk,
            bytes_per_time,
            target,
            runs_whole,
        )

    one = choose(10**9)
    assert one.batches == (nt,) and one.guard == "one batch"
    # one chunk (400 bytes) per batch of 500 bytes: 10 batches
    choice = choose(500)
    if not baseline_major:
        assert choice.guard == "time" and choice.batches == (4,) * 10
        assert choice.runs_batched == 10  # one run per batch
    else:
        # 10 batches x 6 baselines = 60 runs instead of 1: the variable (4000
        # bytes) fits FRAGMENTED_BATCH_FACTOR (8) batches, read in one pass
        assert choice.guard == "fragmented: one pass"
        assert choice.batches == (nt,) and choice.runs_batched == runs_whole == 1
    # a variable larger than the budget (2 batches): batches of the budget size
    monkeypatch.setattr(sw, "FRAGMENTED_BATCH_FACTOR", 2)
    choice = choose(500)
    if baseline_major:
        assert choice.guard == "fragmented: large batches"
        assert choice.batches == (8,) * 5
        assert choice.runs_batched == 5 * nb
    else:
        assert choice.guard == "time"
    # the batches cover the rows: every row in exactly one batch, ascending
    seen = []
    for k in range(len(choice.batches)):
        rows_k, gidx_k = choice.chunk_rows.chunk(k)
        assert np.all(np.diff(rows_k) > 0)
        assert np.all((gidx_k >= 0) & (gidx_k < choice.batches[k] * nb))
        seen.append(rows_k)
    np.testing.assert_array_equal(np.sort(np.concatenate(seen)), rows)


def test_choose_time_batches_reuses_the_last_batching():
    nt, nb = 12, 3
    rows, tidx, bidx = _layout(nt, nb, baseline_major=False)
    main_rows = types.SimpleNamespace(rows=rows)
    cache = {}
    args = (main_rows, tidx, bidx, nb, nt, 2)
    first = sw.choose_time_batches(*args, 10, 20, 1, cache)
    again = sw.choose_time_batches(*args, 10, 20, 1, cache)
    assert again.chunk_rows is first.chunk_rows
    other = sw.choose_time_batches(*args, 10, 40, 1, cache)
    assert other.chunk_rows is not first.chunk_rows
    assert cache["batches"] == other.batches


def test_deferred_placeholder_raises_if_computed():
    placeholder = sw.deferred_placeholder("VISIBILITY", (3, 2, 4), np.complex64)
    assert placeholder.shape == (3, 2, 4) and placeholder.dtype == np.complex64
    reversed_ = placeholder[:, :, ::-1]  # lazy operations are fine
    with pytest.raises(RuntimeError, match="VISIBILITY was computed"):
        reversed_.compute()


def test_reverse_axis_in_place():
    values = np.arange(5 * 3 * 7 * 2, dtype=np.float32).reshape(5, 3, 7, 2)
    expected = np.flip(values, axis=2).copy()
    sw._reverse_axis_in_place(values, 2, max_tmp_bytes=1)  # one time step per slab
    np.testing.assert_array_equal(values, expected)
    sw._reverse_axis_in_place(values, 2)
    np.testing.assert_array_equal(values, np.flip(expected, axis=2))


@pytest.fixture(scope="module")
def cells_table(tmp_path_factory):
    """
    A casacore table with array columns whose cells are all defined with one
    shape (TiledColumnStMan, TiledShapeStMan, direct StandardStMan), defined
    with two shapes (TiledShapeStMan, indirect StandardStMan) or partly
    undefined (TiledShapeStMan, indirect StandardStMan).
    """
    path = str(tmp_path_factory.mktemp("stream_cells") / "cells.tab")

    def tiled(name, dm_type, **kw):
        return tables.makearrcoldesc(
            name,
            0j,
            valuetype="complex",
            datamanagertype=dm_type,
            datamanagergroup=f"group_{name}",
            **kw,
        )

    desc = tables.maketabdesc(
        [
            tables.makescacoldesc("SCALAR", 0.0),
            tiled("TCSM", "TiledColumnStMan", shape=[NCHAN, NPOL], options=4),
            tiled("TSM", "TiledShapeStMan", ndim=2),
            tiled("TSM_TWO_SHAPES", "TiledShapeStMan", ndim=2),
            tiled("TSM_UNDEF", "TiledShapeStMan", ndim=2),
            tables.makearrcoldesc(
                "SSM_DIRECT", 0.0, shape=[3], valuetype="double", options=5
            ),
            tables.makearrcoldesc("SSM_VAR", 0.0, ndim=2, valuetype="float"),
            tables.makearrcoldesc("SSM_UNDEF", 0.0, ndim=1, valuetype="float"),
        ]
    )
    rng = np.random.default_rng(3)
    cell = (NCHAN, NPOL)
    other = (NCHAN - 1, NPOL)
    tb = tables.table(path, desc, nrow=NROWS, readonly=False, ack=False)
    try:
        tb.putcol("SCALAR", rng.random(NROWS))
        data = rng.random((NROWS,) + cell).astype(np.complex64)
        tb.putcol("TCSM", data)
        tb.putcol("TSM", data)
        tb.putcol("SSM_DIRECT", rng.random((NROWS, 3)))
        for row in range(NROWS):
            shape = other if row in OTHER_SHAPE_ROWS else cell
            tb.putcell("TSM_TWO_SHAPES", row, data[row][: shape[0]])
            tb.putcell("SSM_VAR", row, rng.random(shape).astype(np.float32))
        for row in UNDEF_DEFINED_ROWS:
            tb.putcell("TSM_UNDEF", int(row), data[row])
            tb.putcell("SSM_UNDEF", int(row), rng.random(NPOL).astype(np.float32))
    finally:
        tb.close()
    yield path


def _rows_path_reads(tb, col, rows):
    """Whether the row read path (first-cell probe + read_rows) reads the cells."""
    try:
        first = int(rows[0])
        if tb.isscalarcol(col):
            shape = ()
        else:
            shape = rr.parse_shape_string(tb.getcolshapestring(col, first, 1)[0])
        out = np.empty((rows.size,) + shape, dtype=rr.column_dtype(tb, col))
        rr.read_rows(tb, col, rows, out, max_elems=37)
    except Exception:
        return False
    return True


# columns whose cells are compared row by row (the others are decided in O(1))
SCANNED_COLUMNS = ("TSM_TWO_SHAPES", "TSM_UNDEF", "SSM_VAR", "SSM_UNDEF")
SELECTIONS = {
    "all": np.arange(NROWS),
    "head": np.arange(0, 40),
    "around_undefined": np.r_[30:60],
    # 150 runs: more than FRAGMENTED_RUNS (selectrows) when scanned in one batch
    "fragmented": np.arange(0, NROWS, 2),
    "fragmented_defined": np.arange(50, 250, 2),
    "tail": np.arange(260, NROWS),
    "one_row": np.array([45]),
}


@pytest.mark.parametrize("scan_rows", [16, sw.SHAPE_SCAN_ROWS])
@pytest.mark.parametrize("selection", list(SELECTIONS))
@pytest.mark.parametrize(
    "col",
    [
        "SCALAR",
        "TCSM",
        "TSM",
        "TSM_TWO_SHAPES",
        "TSM_UNDEF",
        "SSM_DIRECT",
        "SSM_VAR",
        "SSM_UNDEF",
    ],
)
def test_check_partition_cells_matches_the_read_path(
    cells_table, col, selection, scan_rows, monkeypatch
):
    """check_partition_cells accepts exactly the cells the row read path reads."""
    rows = SELECTIONS[selection]
    monkeypatch.setattr(sw, "SHAPE_SCAN_ROWS", scan_rows)
    with tables.table(cells_table, ack=False) as tb:
        readable = _rows_path_reads(tb, col, rows)
        if readable:
            how = sw.check_partition_cells(tb, col, rows)
            # decided without a scan where the storage manager tells
            scanned = "compared" in how
            assert scanned == (col in SCANNED_COLUMNS)
        else:
            with pytest.raises(sw.ColumnNotReadableError):
                sw.check_partition_cells(tb, col, rows)
    expected_unreadable = {
        "TSM_TWO_SHAPES": {"all", "fragmented"},
        "SSM_VAR": {"all", "fragmented"},
        "TSM_UNDEF": {"all", "around_undefined", "fragmented", "one_row"},
        "SSM_UNDEF": {"all", "around_undefined", "fragmented", "one_row"},
    }
    assert readable == (selection not in expected_unreadable.get(col, set()))


def test_check_partition_cells_never_scans_the_whole_column(cells_table, monkeypatch):
    """The shape scan never asks for all rows of the table in one call."""
    calls = []
    with tables.table(cells_table, ack=False) as tb:

        class Spy:
            def __getattr__(self, name):
                return getattr(tb, name)

            def getcolshapestring(self, col, startrow=0, nrow=-1, *args):
                calls.append((startrow, nrow))
                return tb.getcolshapestring(col, startrow, nrow, *args)

        monkeypatch.setattr(sw, "SHAPE_SCAN_ROWS", 10**6)
        sw.check_partition_cells(Spy(), "SSM_VAR", np.arange(80))
        with pytest.raises(sw.ColumnNotReadableError):
            sw.check_partition_cells(Spy(), "SSM_VAR", np.arange(NROWS))
    assert calls and all(nrow != NROWS for _, nrow in calls)
