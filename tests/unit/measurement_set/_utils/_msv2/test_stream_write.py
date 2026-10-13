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
    assert rr.count_row_runs(np.array([], dtype=np.int64)) == 0
    assert rr.count_row_runs(np.array([5])) == 1
    assert rr.count_row_runs(np.array([0, 1, 2, 7, 8, 20])) == 3


def test_count_row_windows():
    assert rr.count_row_windows(np.array([], dtype=np.int64), 4) == 0
    rows = np.array([0, 1, 2, 7, 8, 20])
    assert rr.count_row_windows(rows, 1) == rows.size
    assert rr.count_row_windows(rows, 4) == 4  # windows 0, 1, 2, 5
    assert rr.count_row_windows(rows, 100) == 1
    with pytest.raises(ValueError):
        rr.count_row_windows(rows, 0)


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
    runs_whole = rr.count_row_runs(rows)

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
            window_rows=5,
        )

    one = choose(10**9)
    assert one.batches == (nt,) and one.guard == "one batch"
    assert one.chunk_rows is None  # read with the partition's grid plan
    # one chunk (400 bytes) per batch of 500 bytes: 10 batches
    choice = choose(500)
    if not baseline_major:
        assert choice.guard == "time" and choice.batches == (4,) * 10
        assert choice.runs_batched == 10  # one run per batch
        # 24 rows per batch, tiles of 5 rows: only the tiles across a batch
        # boundary are loaded twice
        assert choice.windows_whole == 48
        loaded = sum(
            np.unique(choice.chunk_rows.chunk(k)[0] // 5).size for k in range(10)
        )
        assert choice.windows_batched == loaded <= 48 + 9
    else:
        # 10 batches x 6 baselines = 60 runs instead of 1: the variable (4000
        # bytes) fits FRAGMENTED_BATCH_FACTOR (8) batches, read in one pass
        assert choice.guard == "fragmented: one pass"
        assert choice.batches == (nt,) and choice.runs_batched == runs_whole == 1
        assert choice.chunk_rows is None and "row runs" in choice.reason
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
    assert choice.chunk_rows.n_chunks == len(choice.batches)
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
    first = sw.choose_time_batches(*args, 10, 20, 1, cache=cache)
    again = sw.choose_time_batches(*args, 10, 20, 1, cache=cache)
    assert again.chunk_rows is first.chunk_rows
    other = sw.choose_time_batches(*args, 10, 40, 1, cache=cache)
    assert other.chunk_rows is not first.chunk_rows
    assert cache["batching"][0] == other.batches


def _interleaved_layout(nt, nb, n_parts, part):
    """
    Partition ``part`` of a baseline-major table in which ``n_parts``
    partitions alternate time by time inside every baseline (e.g. the SPWs or
    fields of a multi-SPW MS sorted by baseline): every partition row is its
    own row run, and every time batch spans the rows of every baseline.
    """
    bidx_all, tidx_all = np.divmod(np.arange(nt * nb), nt)
    sel = tidx_all % n_parts == part
    return np.flatnonzero(sel), tidx_all[sel] // n_parts, bidx_all[sel]


@pytest.mark.parametrize(
    "window_rows, fragmented",
    [
        (1, False),  # each row its own tile: nothing is loaded twice
        (16, False),  # a batch spans 32 rows of a baseline: whole tiles
        (64, True),  # every tile holds the rows of 2 batches
        (128, True),  # ... of 4 batches
    ],
)
def test_choose_time_batches_guard_interleaved_partitions(window_rows, fragmented):
    """
    The row runs cannot see this layout (time batches have the runs of one
    pass, one per row); the tile windows can: when a tile holds rows of
    several batches, every batch loads it again.
    """
    nt, nb, time_chunk, bytes_per_time = 256, 20, 8, 1000
    rows, tidx, bidx = _interleaved_layout(nt, nb, 2, 0)
    main_rows = types.SimpleNamespace(rows=rows)
    runs_whole = rr.count_row_runs(rows)
    assert runs_whole == rows.size
    n_times = nt // 2
    choice = sw.choose_time_batches(
        main_rows,
        tidx,
        bidx,
        nb,
        n_times,
        time_chunk,
        bytes_per_time,
        2 * time_chunk * bytes_per_time,  # 2 chunks per batch: 8 batches
        runs_whole,
        window_rows=window_rows,
    )
    # what the time batches load: 8 batches of 16 times
    chunk_rows = rr.TimeChunkRows(rows, tidx, bidx, (16,) * 8, nb)
    loaded = chunk_rows.n_windows(window_rows)
    assert choice.windows_whole == rr.count_row_windows(rows, window_rows)
    if not fragmented:
        assert choice.guard == "time" and len(choice.batches) == 8
        assert choice.runs_batched == runs_whole  # one run per row either way
        assert choice.windows_batched == loaded == choice.windows_whole
    else:
        assert loaded >= 2 * choice.windows_whole  # each tile loaded 2-4 times
        assert choice.guard == "fragmented: one pass", choice
        assert choice.batches == (n_times,) and "tiles" in choice.reason
        assert "row runs" not in choice.reason  # the runs alone would accept
        assert choice.windows_batched == choice.windows_whole


def test_choose_time_batches_guard_two_time_ordered_streams():
    """Two time-ordered row streams covering the same times (e.g. two
    concatenated observations of one partition): each batch reads one run of
    each stream, about one extra tile per stream and batch boundary, so the
    time batches are kept."""
    nt, nb, time_chunk, bytes_per_time = 64, 10, 4, 1000
    tidx_one, bidx_one = np.divmod(np.arange(nt * nb), nb)
    rows = np.arange(2 * nt * nb)
    tidx = np.concatenate([tidx_one, tidx_one])
    bidx = np.concatenate([bidx_one, bidx_one])
    main_rows = types.SimpleNamespace(rows=rows)
    choice = sw.choose_time_batches(
        main_rows,
        tidx,
        bidx,
        nb,
        nt,
        time_chunk,
        2 * bytes_per_time,  # a time holds two rows per baseline
        time_chunk * 2 * bytes_per_time,  # one chunk per batch: 16 batches
        rr.count_row_runs(rows),
        window_rows=32,
    )
    assert choice.guard == "time" and len(choice.batches) == 16
    assert choice.windows_batched - choice.windows_whole <= 2 * 15


def test_time_chunk_rows_n_windows_brute_force():
    rng = np.random.default_rng(5)
    for _ in range(12):
        nrows = int(rng.integers(1, 200))
        rows = np.sort(rng.choice(1000, nrows, replace=False))
        tidx = rng.integers(0, 17, nrows)
        bidx = rng.integers(0, 5, nrows)
        chunks = (3, 5, 1, 8)
        window = int(rng.integers(1, 40))
        chunk_rows = rr.TimeChunkRows(rows, tidx, bidx, chunks, 5)
        expected = sum(
            np.unique(chunk_rows.chunk(k)[0] // window).size for k in range(len(chunks))
        )
        assert chunk_rows.n_windows(window) == expected
        assert sum(chunk_rows.chunk_n_rows(k) for k in range(4)) == nrows
        assert [chunk_rows.chunk_n_times(k) for k in range(4)] == list(chunks)


def test_deferred_placeholder_raises_if_computed():
    placeholder = sw.deferred_placeholder("VISIBILITY", (3, 2, 4), np.complex64)
    assert placeholder.shape == (3, 2, 4) and placeholder.dtype == np.complex64
    reversed_ = placeholder[:, :, ::-1]  # lazy operations are fine
    with pytest.raises(RuntimeError, match="VISIBILITY was computed"):
        reversed_.compute()


def test_reverse_axis_in_place():
    values = np.arange(5 * 3 * 7 * 2, dtype=np.float32).reshape(5, 3, 7, 2)
    expected = np.flip(values, axis=2).copy()
    rr.reverse_axis_in_place(values, 2, max_tmp_bytes=1)  # one time step per slab
    np.testing.assert_array_equal(values, expected)
    rr.reverse_axis_in_place(values, 2)
    np.testing.assert_array_equal(values, np.flip(expected, axis=2))
    with pytest.raises(ValueError):
        rr.reverse_axis_in_place(values, 0)


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


# columns whose cell shapes are compared row by row (TiledShapeStMan index)
SCANNED_COLUMNS = ("TSM_TWO_SHAPES", "TSM_UNDEF")
# columns whose cells only the read can verify (shapes stored with the data)
READ_DECIDES_COLUMNS = ("SSM_VAR", "SSM_UNDEF")
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


@pytest.mark.parametrize("scan_rows", [16, rr.SHAPE_SCAN_ROWS])
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
    """
    check_partition_cells never vouches for cells the row read path cannot
    read and never rejects cells it reads: tiled and direct columns are
    decided by the storage manager, indirect StandardStMan arrays are left to
    the read (their shapes are stored with the data).
    """
    rows = SELECTIONS[selection]
    monkeypatch.setattr(rr, "SHAPE_SCAN_ROWS", scan_rows)
    with tables.table(cells_table, ack=False) as tb:
        readable = _rows_path_reads(tb, col, rows)
        first_cell_defined = tb.iscelldefined(col, int(rows[0]))
        if col in READ_DECIDES_COLUMNS and first_cell_defined:
            check = rr.check_partition_cells(tb, col, rows)
            assert not check.verified and "read decides" in check.how
        elif readable:
            check = rr.check_partition_cells(tb, col, rows)
            assert check.verified
            # decided without a scan where the storage manager tells
            assert ("compared" in check.how) == (col in SCANNED_COLUMNS)
        else:
            with pytest.raises(rr.ColumnNotReadableError):
                rr.check_partition_cells(tb, col, rows)
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

        monkeypatch.setattr(rr, "SHAPE_SCAN_ROWS", 10**6)
        assert rr.check_partition_cells(Spy(), "TSM_TWO_SHAPES", np.arange(80)).verified
        with pytest.raises(rr.ColumnNotReadableError):
            rr.check_partition_cells(Spy(), "TSM_TWO_SHAPES", np.arange(NROWS))
    assert calls and all(nrow != NROWS for _, nrow in calls)


def test_check_partition_cells_does_not_scan_indirect_arrays(cells_table):
    """An indirect StandardStMan column is not scanned (its shapes are read
    with its data, so a scan would read the column twice)."""
    calls = []
    with tables.table(cells_table, ack=False) as tb:

        class Spy:
            def __getattr__(self, name):
                return getattr(tb, name)

            def getcolshapestring(self, col, startrow=0, nrow=-1, *args):
                calls.append((startrow, nrow))
                return tb.getcolshapestring(col, startrow, nrow, *args)

            def selectrows(self, rows):
                raise AssertionError("no scan expected")

        for col in READ_DECIDES_COLUMNS:
            check = rr.check_partition_cells(Spy(), col, np.arange(0, NROWS, 2))
            assert not check.verified
    assert calls == [(0, 1), (0, 1)]  # the first cell only


@pytest.mark.parametrize("kind", ["selectrows", "persistent_ref", "concat"])
def test_check_partition_cells_reference_and_concat_tables(cells_table, kind, tmp_path):
    """
    getdminfo of a reference table describes the root table (and of a
    concatenation its first part), not its rows: e.g. a selection of 30 rows
    with undefined cells is shorter than the root's 290 TSM_UNDEF cube rows.
    Nothing is vouched for from the storage manager there: the read decides.
    """
    with tables.table(cells_table, ack=False) as tb:
        parts = []
        if kind == "concat":
            other = str(tmp_path / "other.tab")
            tb.copy(other, deep=True).close()
            parts.append(tables.table(other, ack=False))
            table = tables.table([tb, parts[0]], ack=False)
        else:
            table = tb.selectrows(np.r_[30:60])  # holds the undefined rows 40-49
            if kind == "persistent_ref":
                path = str(tmp_path / "ref.tab")
                table.copy(path, deep=False).close()
                table.close()
                table = tables.table(path, ack=False)
        try:
            rows = np.arange(min(table.nrows(), 60))
            assert (
                table.nrows()
                <= sum(
                    int(cube["CubeShape"][-1])
                    for cube in table.getdminfo("TSM_UNDEF")["SPEC"][
                        "HYPERCUBES"
                    ].values()
                )
                or kind == "concat"
            )
            for col in ("TCSM", "TSM", "TSM_UNDEF", "SSM_DIRECT", "SSM_UNDEF"):
                storage = rr.column_storage(table, col)
                assert not storage.plain and not storage.error
                check = rr.check_partition_cells(table, col, rows, storage)
                assert not check.verified and "reference" in check.how
                window, how = rr.column_row_window(storage, table.nrows(), (4, 2), 64)
                assert how == "nominal"
            assert rr.check_partition_cells(table, "SCALAR", rows).verified
        finally:
            table.close()
            for part in parts:
                part.close()


def test_check_partition_cells_errors_leave_the_decision_to_the_read(
    cells_table, monkeypatch
):
    """An unexpected error while deciding never skips a column."""
    with tables.table(cells_table, ack=False) as tb:

        class Broken:
            def __getattr__(self, name):
                return getattr(tb, name)

            def getdminfo(self, col):
                raise KeyError("SPEC")

        storage = rr.column_storage(Broken(), "TSM")
        assert storage.error.startswith("KeyError") and not storage.plain
        check = rr.check_partition_cells(Broken(), "TSM", np.arange(10))
        assert not check.verified and "read decides" in check.how

        def broken_scan(*args):
            raise ValueError("unexpected")

        monkeypatch.setattr(rr, "_scan_cell_shapes", broken_scan)
        check = rr.check_partition_cells(tb, "TSM_TWO_SHAPES", np.arange(10))
        assert not check.verified and "unexpected" in check.how


def test_column_row_window(cells_table):
    """Tile rows of row-tiled columns, a nominal window otherwise."""
    with tables.table(cells_table, ack=False) as tb:
        for col in ("TCSM", "TSM"):
            storage = rr.column_storage(tb, col)
            cube = storage.hypercubes[0]
            assert rr.column_row_window(storage, NROWS, (NCHAN, NPOL), 64) == (
                int(cube["TileShape"][-1]),
                "tile",
            )
        # two cubes (cell shapes): the cube of the partition's cell shape, its
        # tile scaled by table rows per cube row
        storage = rr.column_storage(tb, "TSM_TWO_SHAPES")
        cubes = {
            tuple(int(n) for n in cube["CellShape"]): cube
            for cube in storage.hypercubes
        }
        cube = cubes[(NPOL, NCHAN - 1)]
        window, how = rr.column_row_window(storage, NROWS, (NCHAN - 1, NPOL), 64)
        expected = int(cube["TileShape"][-1]) * NROWS / int(cube["CubeShape"][-1])
        assert how == "tile" and window == max(1, round(expected))
        storage = rr.column_storage(tb, "SSM_VAR")
        assert rr.column_row_window(storage, NROWS, (4,), 1000) == (
            rr.NOMINAL_WINDOW_BYTES // 1000,
            "nominal",
        )


@pytest.mark.parametrize("n_chunks", [1, 3])
def test_read_time_chunk_matches_the_whole_grid(cells_table, n_chunks):
    """read_time_chunk gives the time slices of the grid read in one pass
    (padding, duplicated cells: last row wins), with the transform and the
    reversal applied."""
    nt, nb = 12, 5
    rng = np.random.default_rng(11)
    rows = np.sort(rng.choice(NROWS, 50, replace=False))
    tidx, bidx = rng.integers(0, nt, rows.size), rng.integers(0, nb, rows.size)
    with tables.table(cells_table, ack=False) as tb:
        plan = rr.make_row_grid_plan(rows, tidx * nb + bidx, nt * nb)
        shape = (nt, nb, NCHAN, NPOL)
        whole = rr.read_grid(tb, "TSM", plan, shape, np.complex64)
        assert whole.shape == shape and np.isnan(whole).any()  # padded cells
        expected = np.flip(whole * 2, axis=2)
        chunks = np.array_split(np.arange(nt), n_chunks)
        chunk_rows = rr.TimeChunkRows(rows, tidx, bidx, [c.size for c in chunks], nb)
        stats = {}
        parts = [
            rr.read_time_chunk(
                tb,
                "TSM",
                chunk_rows,
                k,
                (NCHAN, NPOL),
                np.complex64,
                transform=lambda grid: grid * 2,
                reverse_axis=2,
                stats=stats,
            )
            for k in range(n_chunks)
        ]
    np.testing.assert_array_equal(np.concatenate(parts), expected)
    assert stats["calls"] >= n_chunks
    # a chunk without rows needs no table
    empty = rr.TimeChunkRows(rows[:0], tidx[:0], bidx[:0], (nt,), nb)
    padded = rr.read_time_chunk(None, "TSM", empty, 0, (NCHAN, NPOL), np.complex64)
    assert padded.shape == shape and np.isnan(padded).all()


def _placeholder_xds():
    """A main xds with deferred placeholders (and their DeferredVariables) as
    the converter writes it: the encoding of add_encoding, xradio's attrs."""
    import xarray as xr

    dims = ("time", "baseline_id", "frequency")

    def placeholder(dtype):
        return xr.DataArray(sw.deferred_placeholder("X", (3, 2, 4), dtype), dims=dims)

    dtypes = {"VIS": np.complex64, "FLAG": bool, "W": np.float32, "N": np.int32}
    specs = {
        name: sw.DeferredVariable(name, "COL", np.dtype(dt))
        for name, dt in dtypes.items()
    }
    xds = xr.Dataset({name: placeholder(dt) for name, dt in dtypes.items()})
    xds["SMALL"] = xr.DataArray(np.zeros(3), dims=("time",))
    for name in specs:
        xds[name].encoding = {"chunks": [1, 2, 4], "compressors": ()}
        xds[name].attrs = {"units": "Jy", "type": "quantity"}
    return xds, specs


def test_check_deferred_variables():
    """Every lazy data variable must be deferred and every deferred variable
    still a placeholder."""
    xds, specs = _placeholder_xds()
    sw.check_deferred_variables(xds, specs)
    sw.check_deferred_variables(xds, {**specs, "UVW": specs["VIS"]})  # dropped
    # a renamed placeholder would be left unwritten
    with pytest.raises(RuntimeError, match="no deferred description"):
        sw.check_deferred_variables(xds.rename({"VIS": "VIS2"}), specs)
    with pytest.raises(RuntimeError, match="holds values"):
        sw.check_deferred_variables(xds, {**specs, "SMALL": specs["W"]})


# encodings with which to_zarr writes other values (or another dtype) than the
# values of the variable
VALUE_CODINGS = (
    {"scale_factor": 2.0},
    {"add_offset": 1.0},
    {"_FillValue": -1.0},  # NaN would be written as -1
    {"dtype": "float64"},
)


def test_deferred_encoding_problems():
    """Only xradio's encoding (chunks, compressors) on bool / int / float /
    complex values is streamed: any CF coding key in the encoding or the
    attributes, another encoding key or dtype is reported."""
    xds, specs = _placeholder_xds()
    assert sw.deferred_encoding_problems(xds, specs) == []
    assert sw.deferred_encoding_problems(xds, {**specs, "UVW": specs["W"]}) == []
    for coding in VALUE_CODINGS:
        bad = xds.copy()
        bad["W"].encoding = {**bad["W"].encoding, **coding}
        assert sw.deferred_encoding_problems(bad, specs) == [
            f"W: encoding {sorted(coding)}"
        ]
    # as attributes they are not applied by to_zarr, but are not told apart from
    # an applied coding in the stored metadata: not streamed either
    bad = xds.copy()
    bad["W"].attrs = {**bad["W"].attrs, "scale_factor": 2.0, "_FillValue": -1.0}
    assert sw.deferred_encoding_problems(bad, specs) == [
        "W: attributes ['_FillValue', 'scale_factor']"
    ]
    bad = xds.copy()
    bad["W"].encoding = {**bad["W"].encoding, "filters": None}
    assert sw.deferred_encoding_problems(bad, specs) == ["W: encoding ['filters']"]
    times = xds.copy()
    times["W"] = times["W"].astype("datetime64[ns]")
    times["W"].encoding = dict(xds["W"].encoding)
    assert sw.deferred_encoding_problems(times, specs) == ["W: dtype datetime64[ns]"]


def _declared_problems(xds, specs, store):
    """deferred_array_problems of the metadata to_zarr(compute=False) declares
    (the placeholders are never computed)."""
    import zarr

    xds.to_zarr(store, mode="w", zarr_format=3, compute=False)
    group = zarr.open_group(store, mode="r", use_consolidated=False)
    return sw.deferred_array_problems(group, xds, specs)


def test_deferred_array_problems(tmp_path):
    """The zarr metadata that to_zarr declares for the deferred variables
    matches what the batch writer writes (the NaN _FillValue xarray adds to
    floats masks nothing); an encoding that changes the values is reported
    from the metadata alone."""
    xds, specs = _placeholder_xds()
    store = str(tmp_path / "ok")
    assert _declared_problems(xds, specs, store) == []
    import zarr

    attrs = dict(zarr.open_group(store, mode="r")["W"].attrs)
    assert attrs["_FillValue"] == "AAAAAAAA+H8="  # NaN, as xarray stores it
    # (packing a float32 with a python float scale / offset also stores float64)
    expected = {
        "scale_factor": "W: attributes {scale_factor: 2.0}",
        "add_offset": "W: attributes {add_offset: 1.0}",
        "_FillValue": "W: attributes {_FillValue: 'AAAAAAAA8L8='}",  # -1.0
        "dtype": "W: zarr dtype float64, values float32",
    }
    for idx, coding in enumerate(VALUE_CODINGS):
        bad = xds.copy()
        bad["W"].encoding = {**bad["W"].encoding, **coding}
        problems = _declared_problems(bad, specs, str(tmp_path / f"bad{idx}"))
        assert expected[next(iter(coding))] in problems, problems
        assert all(problem.startswith("W: ") for problem in problems)
    # attributes written as such are not told apart from an applied coding
    bad = xds.copy()
    bad["W"].attrs = {**bad["W"].attrs, "scale_factor": 2.0}
    problems = _declared_problems(bad, specs, str(tmp_path / "attrs"))
    assert problems == ["W: attributes {scale_factor: 2.0}"]


def test_deferred_array_problems_from_zarr_metadata(tmp_path):
    """Arrays declared otherwise than the xds (chunks, filters, a missing
    array) are reported."""
    import zarr

    xds, specs = _placeholder_xds()
    specs = {"W": specs["W"]}
    group = zarr.open_group(str(tmp_path / "g"), mode="w", zarr_format=3)
    group.create_array("W", shape=(3, 2, 4), chunks=(3, 2, 4), dtype="float32")
    assert sw.deferred_array_problems(group, xds, specs) == [
        "W: zarr chunks (3, 2, 4), encoding (1, 2, 4)"
    ]
    del group["W"]
    group.create_array(
        "W",
        shape=(3, 2, 4),
        chunks=(1, 2, 4),
        dtype="float32",
        filters=[zarr.codecs.TransposeCodec(order=(0, 1, 2))],
    )
    problems = sw.deferred_array_problems(group, xds, specs)
    assert len(problems) == 1 and problems[0].startswith("W: filters")
    del group["W"]
    problems = sw.deferred_array_problems(group, xds, specs)
    assert len(problems) == 1 and problems[0].startswith("W: KeyError")


@pytest.mark.parametrize(
    "value, is_nan",
    [
        ("AAAAAAAA+H8=", True),  # float64 NaN (xarray, zarr format 3)
        ("AADAfw==", True),  # float32 NaN
        ("AAAAAAAA8L8=", False),  # -1.0
        ("NaN", True),
        (float("nan"), True),
        (np.float32("nan"), True),
        (0.0, False),
        (-1, False),
        (True, False),
        ("not base64!", False),
        ("AAAA", False),  # 3 bytes
        (None, False),
    ],
)
def test_stored_fill_value_is_nan(value, is_nan):
    assert sw._stored_fill_value_is_nan(value) is is_nan


FILE_URL_SKIP = "this zarr version cannot write file:// URLs into new directories"


def _store_location(tmp_path, location: str) -> str:
    """An MSv4 store path: local, a file:// URL or a memory:// URL (both
    written through fsspec)."""
    import uuid

    if location == "local":
        return str(tmp_path / "msv4")
    if location == "file":
        return "file://" + str(tmp_path / "msv4")
    return f"memory://{uuid.uuid4().hex}/msv4"


def _has_consolidated_metadata(store: str) -> bool:
    import zarr

    try:
        zarr.open_group(store, mode="r", zarr_format=3, use_consolidated=True)
    except ValueError:
        return False
    return True


@pytest.mark.parametrize("location", ["local", "file", "memory"])
@pytest.mark.parametrize("remove_store", [True, False])
def test_discard_msv4(tmp_path, remove_store, location, zarr_writes_file_urls):
    """A failed fill leaves no MSv4 with unwritten data variables: the whole
    store is removed (also a URL, through fsspec), or (an MSv4 that existed
    before, mode "a") the deferred arrays and the members added, the MSv4 left
    without consolidated metadata (as incomplete)."""
    import xarray as xr
    import zarr

    if location == "file" and not zarr_writes_file_urls:
        pytest.skip(FILE_URL_SKIP)
    store = _store_location(tmp_path, location)
    assert sw.msv4_members(store) is None
    old = xr.Dataset({"OLD": ("x", np.arange(3.0)), "VIS": ("x", np.zeros(3))})
    old.to_zarr(store, mode="w", zarr_format=3)
    members_before = sw.msv4_members(store)
    assert members_before == {"zarr.json", "OLD", "VIS"}
    new = xr.Dataset({"NEW": ("x", np.arange(3.0)), "VIS": ("x", np.ones(3))})
    new.to_zarr(store, mode="a", zarr_format=3)
    assert _has_consolidated_metadata(store)
    done = sw.discard_msv4(store, {"VIS"}, remove_store, members_before)
    if remove_store:
        assert done == "MSv4 removed" and sw.msv4_members(store) is None
        assert sw.discard_msv4(store, {"VIS"}, True) == "MSv4 not written"
        return
    group = zarr.open_group(store, mode="r", zarr_format=3, use_consolidated=False)
    assert sorted(group.array_keys()) == ["OLD"]
    assert not _has_consolidated_metadata(store)
    with pytest.warns(RuntimeWarning, match="consolidated metadata"):
        reopened = xr.open_zarr(store)
    assert sorted(reopened.data_vars) == ["OLD"]


@pytest.mark.parametrize("location", ["local", "file"])
def test_drop_consolidated_metadata_and_consolidate_msv4(
    tmp_path, location, zarr_writes_file_urls
):
    """Between drop_consolidated_metadata and consolidate_msv4 (the streamed
    write) an MSv4 opens as incomplete: a warning (the non-consolidated
    metadata is read), an error with consolidated=True. Afterwards it opens as
    written by to_zarr, with the same consolidated metadata."""
    import xarray as xr
    import zarr

    if location == "file" and not zarr_writes_file_urls:
        pytest.skip(FILE_URL_SKIP)
    store = _store_location(tmp_path, location)
    xds = xr.Dataset({"VIS": ("x", np.arange(3.0))}, attrs={"type": "visibility"})
    xr.DataTree(xds).to_zarr(store, mode="w", zarr_format=3)
    written = zarr.open_group(store, mode="r", use_consolidated=True).metadata
    sw.drop_consolidated_metadata(store)
    assert not _has_consolidated_metadata(store)
    with pytest.warns(RuntimeWarning, match="consolidated metadata"):
        assert xr.open_datatree(store, engine="zarr").attrs == xds.attrs
    with pytest.raises(ValueError, match="onsolidated"):
        xr.open_datatree(store, engine="zarr", consolidated=True)
    sw.consolidate_msv4(store)
    reopened = zarr.open_group(store, mode="r", use_consolidated=True).metadata
    assert reopened.to_dict() == written.to_dict()
    xr.testing.assert_identical(
        xr.open_datatree(store, engine="zarr", consolidated=True).to_dataset(), xds
    )


def test_main_table_rows_column_storage(cells_table):
    """The storage described from the table's data manager info read once is
    the per-column description."""

    def comparable(storage):
        cubes = [
            {key: np.asarray(value).tolist() for key, value in cube.items()}
            for cube in storage.hypercubes
        ]
        return (storage.plain, storage.dm_type, storage.option, cubes, storage.error)

    with tables.table(cells_table, ack=False) as tb:
        main_rows = rr.MainTableRows(tb, np.arange(10))
        for col in tb.colnames():
            storage = main_rows.column_storage(col)
            assert comparable(storage) == comparable(rr.column_storage(tb, col))
            assert storage.plain and not storage.error
        assert main_rows.column_storage("NO_SUCH_COLUMN").error


def test_fits_in_memory():
    import xarray as xr

    placeholder = sw.deferred_placeholder("V", (4, 8), np.float64)  # 256 bytes
    xds = xr.Dataset({"V": (("time", "b"), placeholder), "U": ("time", np.zeros(4))})
    specs = {"V": sw.DeferredVariable("V", "DATA", np.dtype(np.float64))}
    assert sw.deferred_bytes(xds, specs) == 256
    assert sw.fits_in_memory(xds, specs, int(256 / sw.IN_MEMORY_BATCH_FRACTION))
    assert not sw.fits_in_memory(xds, specs, int(256 / sw.IN_MEMORY_BATCH_FRACTION) - 1)
    assert sw.deferred_bytes(xds, {"UVW": specs["V"]}) == 0  # dropped variables
