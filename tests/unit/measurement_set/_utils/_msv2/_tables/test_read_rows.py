import subprocess
import sys
import textwrap

import numpy as np
import pytest

tables = pytest.importorskip("casacore.tables")

from xradio.measurement_set._utils._msv2._tables import read_rows as rr  # noqa: E402

NROWS = 200
NCHAN = 6
NPOL = 2
# rows of the TSM_UNDEF column that are written (the others stay undefined)
UNDEF_DEFINED_ROWS = np.r_[0:10, 20:NROWS]
# sentinel values that no row of the test table holds
SENTINEL = {
    np.dtype(np.complex64): np.complex64(-123456 - 654321j),
    np.dtype(np.bool_): True,
    np.dtype(np.float32): np.float32(-1.5e30),
    np.dtype(np.float64): -1.5e300,
    np.dtype(np.int32): np.int32(-987654),
}


@pytest.fixture(scope="module")
def rows_table(tmp_path_factory):
    """
    A small casacore table with row-dependent values in scalar, fixed-shape
    (StandardStMan) and variable-shape (TiledShapeStMan) columns, plus a
    TiledShapeStMan column with undefined cells. Yields (path, reference values).
    """
    path = str(tmp_path_factory.mktemp("read_rows") / "rows.tab")
    desc = tables.maketabdesc(
        [
            tables.makescacoldesc("SCALAR_INT", 0),
            tables.makescacoldesc("SCALAR_DOUBLE", 0.0),
            tables.makescacoldesc("NAME", "none"),
            tables.makearrcoldesc("SSM_WEIGHT", 0.0, shape=[NPOL], valuetype="float"),
            tables.makearrcoldesc(
                "TSM_DATA",
                0j,
                ndim=2,
                valuetype="complex",
                datamanagertype="TiledShapeStMan",
                datamanagergroup="TSMData",
            ),
            tables.makearrcoldesc(
                "TSM_FLAG",
                False,
                ndim=2,
                valuetype="boolean",
                datamanagertype="TiledShapeStMan",
                datamanagergroup="TSMFlag",
            ),
            tables.makearrcoldesc(
                "TSM_UNDEF",
                0j,
                ndim=2,
                valuetype="complex",
                datamanagertype="TiledShapeStMan",
                datamanagergroup="TSMUndef",
            ),
        ]
    )
    rng = np.random.default_rng(42)
    rows = np.arange(NROWS)
    chan_pol = np.arange(NCHAN)[:, None] * 10 + np.arange(NPOL)[None, :]
    ref = {
        "SCALAR_INT": (rows * 3 - 7).astype(np.int32),
        "SCALAR_DOUBLE": rows * 0.25 + 1e9,
        "NAME": np.array([f"row{r}" for r in rows]),
        "SSM_WEIGHT": rng.random((NROWS, NPOL)).astype(np.float32),
        "TSM_DATA": (
            (rows[:, None, None] + 1) * 1000 + chan_pol[None] + 1j * rows[:, None, None]
        ).astype(np.complex64),
        "TSM_FLAG": rng.random((NROWS, NCHAN, NPOL)) < 0.5,
    }
    tb = tables.table(path, desc, nrow=NROWS, readonly=False, ack=False)
    try:
        for col, values in ref.items():
            tb.putcol(col, values)
        for row in UNDEF_DEFINED_ROWS:
            tb.putcell("TSM_UNDEF", int(row), ref["TSM_DATA"][row])
    finally:
        tb.close()
    yield path, ref


@pytest.fixture
def rows_tb(rows_table):
    path, ref = rows_table
    tb = tables.table(path, readonly=True, ack=False)
    yield tb, ref
    tb.close()


def sentinel_buffer(shape, dtype):
    dtype = np.dtype(dtype)
    return np.full(shape, SENTINEL[dtype], dtype=dtype)


ROW_SELECTIONS = {
    "all": np.arange(NROWS),
    "one_row": np.array([17]),
    "few_runs": np.r_[3:40, 50:51, 60:120, 199:200],
    # 100 runs of one row: more than FRAGMENTED_RUNS, read with selectrows
    "fragmented": np.arange(0, NROWS, 2),
    "random": np.sort(np.random.default_rng(1).choice(NROWS, 77, replace=False)),
}


@pytest.mark.parametrize(
    "rows",
    [
        np.array([], dtype=np.int64),
        np.array([5]),
        np.arange(10),
        np.array([0, 1, 2, 7, 8, 20]),
        np.array([1, 3, 5, 7]),
    ],
)
def test_rows_to_runs_round_trip(rows):
    starts, lengths = rr.rows_to_runs(rows)
    assert starts.dtype == np.int64 and lengths.dtype == np.int64
    assert np.all(lengths > 0)
    # runs are maximal: consecutive runs are separated by a gap
    assert np.all(starts[1:] > starts[:-1] + lengths[:-1])
    np.testing.assert_array_equal(rr.runs_to_rows(starts, lengths), rows)


def test_runs_to_rows_shape_mismatch():
    with pytest.raises(ValueError, match="differ in shape"):
        rr.runs_to_rows(np.array([0, 5]), np.array([3]))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_group_row_runs_matches_per_group_selection(seed):
    rng = np.random.default_rng(seed)
    n_groups = 7
    # runs of equal groups with gaps (-1 = no group), like MAIN rows of partitions
    row_group = np.repeat(rng.integers(-1, n_groups, 60), rng.integers(1, 9, 60))
    runs = rr.group_row_runs(row_group, n_groups)
    assert len(runs) == n_groups
    for group, (starts, lengths) in enumerate(runs):
        np.testing.assert_array_equal(
            rr.runs_to_rows(starts, lengths), np.flatnonzero(row_group == group)
        )
        np.testing.assert_array_equal(
            (starts, lengths), rr.rows_to_runs(np.flatnonzero(row_group == group))
        )


def test_group_row_runs_empty():
    assert rr.group_row_runs(np.array([], dtype=np.int64), 0) == []
    runs = rr.group_row_runs(np.full(5, -1), 2)
    assert [r[0].size for r in runs] == [0, 0]


@pytest.mark.parametrize("selection", ROW_SELECTIONS)
@pytest.mark.parametrize(
    "col", ["SCALAR_INT", "SCALAR_DOUBLE", "SSM_WEIGHT", "TSM_DATA", "TSM_FLAG"]
)
@pytest.mark.parametrize("max_elems", [rr.DEFAULT_MAX_ELEMS, 25, 1])
def test_read_rows_matches_reference(rows_tb, selection, col, max_elems):
    tb, ref = rows_tb
    rows = ROW_SELECTIONS[selection]
    expected = ref[col][rows]
    out = sentinel_buffer(expected.shape, rr.column_dtype(tb, col))
    stats = rr.read_rows(tb, col, rows, out, max_elems=max_elems)
    np.testing.assert_array_equal(out, expected)

    cell_elems = int(np.prod(expected.shape[1:])) or 1
    rows_per_call = max(1, max_elems // cell_elems)
    starts, lengths = rr.rows_to_runs(rows)
    if starts.size <= rr.FRAGMENTED_RUNS and rows_per_call >= rows.size:
        # one call per run, but never one call over the whole column
        whole_column = rows.size == NROWS
        assert stats["calls"] == starts.size + whole_column
        assert stats.get("selectrows_calls", 0) == 0
    if selection == "fragmented" and rows_per_call > rr.FRAGMENTED_RUNS:
        assert stats["selectrows_calls"] >= 1
    if max_elems < rr.DEFAULT_MAX_ELEMS and starts.size <= rr.FRAGMENTED_RUNS:
        # max_elems caps every call
        assert stats["calls"] >= int(np.ceil(rows.size / rows_per_call))


def test_read_rows_max_elems_bounds_every_call(rows_tb, monkeypatch):
    tb, ref = rows_tb
    calls = []

    class SpyTable:
        """Records the number of rows of every get*np call."""

        def __getattr__(self, name):
            return getattr(tb, name)

        def getcolnp(self, col, out, *args):
            calls.append(out.size)
            return tb.getcolnp(col, out, *args)

        def selectrows(self, rows):
            ref_tb = tb.selectrows(rows)
            calls.append(("selectrows", len(rows)))
            return ref_tb

    max_elems = 5 * NCHAN * NPOL
    rows = np.r_[0:150]
    out = sentinel_buffer((rows.size, NCHAN, NPOL), np.complex64)
    rr.read_rows(SpyTable(), "TSM_DATA", rows, out, max_elems=max_elems)
    np.testing.assert_array_equal(out, ref["TSM_DATA"][rows])
    assert calls and all(c <= max_elems for c in calls)
    assert len(calls) == 30


@pytest.mark.parametrize("selection", ["all", "few_runs", "fragmented", "one_row"])
@pytest.mark.parametrize(
    "chan, pol",
    [(slice(1, 4), None), (None, slice(1, 2)), (slice(5, 6), slice(0, 1))],
)
@pytest.mark.parametrize("max_elems", [7, rr.DEFAULT_MAX_ELEMS])
def test_read_rows_cell_slice(rows_tb, selection, chan, pol, max_elems):
    tb, ref = rows_tb
    rows = ROW_SELECTIONS[selection]
    expected = ref["TSM_DATA"][rows][
        :, chan if chan else slice(None), pol if pol else slice(None)
    ]
    out = sentinel_buffer(expected.shape, np.complex64)
    stats = rr.read_rows(
        tb, "TSM_DATA", rows, out, chan=chan, pol=pol, max_elems=max_elems
    )
    np.testing.assert_array_equal(out, expected)
    if selection == "fragmented" and max_elems == rr.DEFAULT_MAX_ELEMS:
        assert stats["selectrows_calls"] >= 1


@pytest.mark.parametrize(
    "chan", [slice(-1, 2), slice(2, 2), slice(0, 4, 2), slice(None, 3)]
)
def test_read_rows_rejects_bad_slices(rows_tb, chan):
    tb, _ = rows_tb
    with pytest.raises(ValueError, match="slice"):
        rr.read_rows(
            tb,
            "TSM_DATA",
            np.arange(2),
            np.empty((2, 2, NPOL), np.complex64),
            chan=chan,
        )


@pytest.mark.parametrize(
    "rows, out_kind, error",
    [
        (np.array([3, 2]), "ok", ValueError),  # not sorted
        (np.array([2, 2]), "ok", ValueError),  # not unique
        (np.array([1, 2]), "complex128", TypeError),  # hidden-temporary dtype
        (np.array([1, 2]), "fortran", ValueError),
        (np.array([1, 2]), "unaligned", ValueError),  # silently not filled
        (np.array([1, 2]), "too_short", ValueError),
        (np.array([1, NROWS]), "ok", IndexError),
    ],
)
def test_read_rows_rejects_bad_arguments(rows_tb, rows, out_kind, error):
    tb, _ = rows_tb
    shape = (rows.size, NCHAN, NPOL)
    if out_kind == "ok":
        out = np.empty(shape, np.complex64)
    elif out_kind == "complex128":
        out = np.empty(shape, np.complex128)
    elif out_kind == "fortran":
        out = np.empty(shape, np.complex64, order="F")
    elif out_kind == "unaligned":
        nbytes = int(np.prod(shape)) * 8
        out = np.frombuffer(bytearray(nbytes + 1), dtype=np.complex64, offset=1)
        out = out.reshape(shape)
        assert not out.flags.aligned
    elif out_kind == "too_short":
        out = np.empty((rows.size - 1, NCHAN, NPOL), np.complex64)
    with pytest.raises(error):
        rr.read_rows(tb, "TSM_DATA", rows, out)


@pytest.mark.parametrize("max_elems", [0, rr.MAX_ELEMS_LIMIT + 1])
def test_read_rows_rejects_max_elems(rows_tb, max_elems):
    tb, _ = rows_tb
    with pytest.raises(ValueError, match="max_elems"):
        rr.read_rows(
            tb,
            "TSM_DATA",
            np.arange(2),
            np.empty((2, NCHAN, NPOL), np.complex64),
            max_elems=max_elems,
        )


def test_read_rows_slice_needs_2d_cells(rows_tb):
    tb, _ = rows_tb
    with pytest.raises(ValueError, match="2-D"):
        rr.read_rows(
            tb,
            "SSM_WEIGHT",
            np.arange(2),
            np.empty((2, 1), np.float32),
            chan=slice(0, 1),
        )


def test_read_rows_undefined_cells(rows_tb):
    tb, ref = rows_tb
    defined = np.r_[0:10]
    out = sentinel_buffer((defined.size, NCHAN, NPOL), np.complex64)
    rr.read_rows(tb, "TSM_UNDEF", defined, out)
    np.testing.assert_array_equal(out, ref["TSM_DATA"][defined])
    for rows in (np.r_[5:15], np.r_[12:13], np.r_[0:NROWS:3]):
        out = np.empty((rows.size, NCHAN, NPOL), np.complex64)
        with pytest.raises(RuntimeError):
            rr.read_rows(tb, "TSM_UNDEF", rows, out)


def test_read_rows_whole_column_with_undefined_cells_raises(rows_table, tmp_path):
    """
    casacore's whole-column read path crashes (SIGSEGV) on a TiledShapeStMan
    column with undefined cells. read_rows must never take it: run in a
    subprocess so that a regression does not take the test session down.
    """
    path, _ = rows_table
    one_row = str(tmp_path / "one_row.tab")
    code = textwrap.dedent(
        f"""
        import numpy as np
        from casacore import tables
        from xradio.measurement_set._utils._msv2._tables.read_rows import read_rows

        def check(tb, nrows):
            out = np.empty((nrows, {NCHAN}, {NPOL}), np.complex64)
            try:
                read_rows(tb, "TSM_UNDEF", np.arange(nrows), out)
            except RuntimeError:
                print("raised")

        tb = tables.table({path!r}, ack=False)
        check(tb, tb.nrows())
        tb.close()
        desc = tables.maketabdesc([tables.makearrcoldesc(
            "TSM_UNDEF", 0j, ndim=2, valuetype="complex",
            datamanagertype="TiledShapeStMan", datamanagergroup="G")])
        tb = tables.table({one_row!r}, desc, nrow=1, readonly=False, ack=False)
        check(tb, 1)
        tb.close()
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=300
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert proc.stdout.split() == ["raised", "raised"]


def test_read_column_rows(rows_tb):
    tb, ref = rows_tb
    rows = ROW_SELECTIONS["few_runs"]
    for col in ("SCALAR_INT", "SCALAR_DOUBLE", "SSM_WEIGHT", "TSM_DATA", "NAME"):
        values = rr.read_column_rows(tb, col, rows)
        np.testing.assert_array_equal(values, ref[col][rows])
        if col != "NAME":  # strings: per-run getcol fallback
            # the dtype python-casacore's getcol returns
            assert values.dtype == tb.getcol(col, 0, 2).dtype


def make_plan_case(seed, nt=9, nb=6, keep=0.8, n_dup=0, shuffle_cells=False):
    """
    Rows (ascending, runs separated by gaps) and their grid cells for a
    partition of the test table: time-major cells (or shuffled ones, as for a
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
    rows = np.cumsum(1 + gaps * rng.integers(1, 4, n)) - 1 + rng.integers(0, 5)
    assert rows[-1] < NROWS
    return rows, cells, nt, nb


def test_make_row_grid_plan_direct_and_scatter():
    rows = np.r_[10:40, 50:52, 60:80]
    gidx = np.r_[0:30, 100:102, 30:50]
    plan = rr.make_row_grid_plan(rows, gidx, 120, min_direct_rows=16)
    np.testing.assert_array_equal(plan.direct_offsets, [0, 32])
    np.testing.assert_array_equal(plan.direct_lengths, [30, 20])
    np.testing.assert_array_equal(plan.scatter_idx, [30, 31])
    assert not plan.grid_is_full
    assert plan.n_duplicate_rows == 0
    # a jump in grid index (with consecutive rows) also ends a direct segment
    plan = rr.make_row_grid_plan(np.arange(40), np.r_[0:20, 40:60], 60)
    np.testing.assert_array_equal(plan.direct_lengths, [20, 20])
    full = rr.make_row_grid_plan(np.arange(20), np.arange(20), 20)
    assert full.grid_is_full and full.scatter_idx.size == 0
    # segments shorter than min_direct_rows are batched through the scatter path
    short = rr.make_row_grid_plan(np.arange(12), np.arange(12), 12)
    assert short.grid_is_full and short.scatter_idx.size == 12


def test_make_row_grid_plan_duplicates_are_scattered():
    rows = np.arange(40)
    gidx = np.r_[0:20, 19:38]  # row 20 duplicates the cell of row 19
    gidx = np.r_[gidx, 5]  # row 39 duplicates the cell of row 5
    plan = rr.make_row_grid_plan(rows, gidx, 38, min_direct_rows=1)
    assert plan.n_duplicate_rows == 4
    scattered = set(plan.scatter_idx.tolist())
    assert {5, 19, 20, 39} <= scattered
    direct = set(
        np.concatenate(
            [
                np.arange(o, o + n)
                for o, n in zip(plan.direct_offsets, plan.direct_lengths, strict=True)
            ]
        ).tolist()
    )
    assert not direct & {5, 19, 20, 39}
    assert direct | scattered == set(range(40))
    assert plan.grid_is_full


def test_make_row_grid_plan_rejects():
    with pytest.raises(IndexError):
        rr.make_row_grid_plan(np.arange(3), np.array([0, 1, 9]), 9)
    with pytest.raises(ValueError):
        rr.make_row_grid_plan(np.arange(3), np.arange(2), 9)
    with pytest.raises(ValueError):
        rr.make_row_grid_plan(np.array([2, 1, 3]), np.arange(3), 9)


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("n_dup", [0, 5])
@pytest.mark.parametrize("shuffle_cells", [False, True])
@pytest.mark.parametrize("col", ["TSM_DATA", "TSM_FLAG", "SSM_WEIGHT", "SCALAR_DOUBLE"])
@pytest.mark.parametrize(
    "max_elems, max_tmp_bytes",
    [(rr.DEFAULT_MAX_ELEMS, rr.DEFAULT_MAX_TMP_BYTES), (40, 200)],
)
def test_read_rows_to_grid_matches_fancy_assignment(
    rows_tb, seed, n_dup, shuffle_cells, col, max_elems, max_tmp_bytes
):
    tb, ref = rows_tb
    rows, gidx, nt, nb = make_plan_case(seed, n_dup=n_dup, shuffle_cells=shuffle_cells)
    values = ref[col][rows]
    dtype = rr.column_dtype(tb, col)
    # reference: the TaQL path's numpy fancy assignment of all rows (last wins)
    expected = sentinel_buffer((nt, nb) + values.shape[1:], dtype)
    expected[gidx // nb, gidx % nb] = values

    plan = rr.make_row_grid_plan(rows, gidx, nt * nb, min_direct_rows=4)
    grid = sentinel_buffer(expected.shape, dtype)
    stats = rr.read_rows_to_grid(
        tb, col, plan, grid, max_elems=max_elems, max_tmp_bytes=max_tmp_bytes
    )
    np.testing.assert_array_equal(grid, expected)
    assert stats.get("direct_rows", 0) + stats.get("scatter_rows", 0) == rows.size
    assert stats.get("max_tmp_bytes", 0) <= max(
        max_tmp_bytes, int(np.prod(values.shape[1:])) * dtype.itemsize
    )
    if n_dup:
        assert plan.n_duplicate_rows > 0


class RowOrderSpy:
    """
    Table proxy that records the base-table rows of every get*np call, in call
    order (selectrows reference tables included).
    """

    def __init__(self, tb):
        self._tb = tb
        self.rows_read: list[np.ndarray] = []

    def __getattr__(self, name):
        return getattr(self._tb, name)

    def _record(self, args, nrow_total):
        startrow, nrow = (args + (0, nrow_total))[:2]
        self.rows_read.append(np.arange(startrow, startrow + nrow))

    def getcolnp(self, col, out, *args):
        self._record(args, out.shape[0])
        return self._tb.getcolnp(col, out, *args)

    def getcolslicenp(self, col, out, blc, trc, inc, *args):
        self._record(args, out.shape[0])
        return self._tb.getcolslicenp(col, out, blc, trc, inc, *args)

    def selectrows(self, rows):
        spy = self

        class RefSpy:
            def __init__(self, ref):
                self._ref = ref

            def __getattr__(self, name):
                return getattr(self._ref, name)

            def getcolnp(self, col, out, *args):
                spy.rows_read.append(np.asarray(rows, dtype=np.int64))
                return self._ref.getcolnp(col, out, *args)

            def getcolslicenp(self, col, out, *args):
                spy.rows_read.append(np.asarray(rows, dtype=np.int64))
                return self._ref.getcolslicenp(col, out, *args)

        return RefSpy(self._tb.selectrows(rows))


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("n_dup", [0, 5])
@pytest.mark.parametrize("max_tmp_bytes", [rr.DEFAULT_MAX_TMP_BYTES, 200])
def test_read_rows_to_grid_reads_rows_in_one_ascending_pass(
    rows_tb, seed, n_dup, max_tmp_bytes
):
    """
    Direct segments and scattered rows are read interleaved, in row order: a
    pass over the direct segments followed by a second pass over the scattered
    rows reads the tiles that hold both twice (one row-slab tile cache).
    """
    tb, ref = rows_tb
    rows, gidx, nt, nb = make_plan_case(seed, n_dup=n_dup)
    plan = rr.make_row_grid_plan(rows, gidx, nt * nb, min_direct_rows=4)
    assert plan.direct_lengths.size and plan.scatter_idx.size  # both kinds
    # scattered rows lie between direct segments
    first_direct = plan.direct_offsets[0]
    assert (plan.scatter_idx > first_direct).any()
    spy = RowOrderSpy(tb)
    grid = sentinel_buffer((nt, nb, NCHAN, NPOL), np.complex64)
    rr.read_rows_to_grid(spy, "TSM_DATA", plan, grid, max_tmp_bytes=max_tmp_bytes)
    read_order = np.concatenate(spy.rows_read)
    np.testing.assert_array_equal(read_order, rows)  # every row once, ascending
    expected = sentinel_buffer(grid.shape, np.complex64)
    expected[gidx // nb, gidx % nb] = ref["TSM_DATA"][rows]
    np.testing.assert_array_equal(grid, expected)


@pytest.fixture(scope="module")
def small_tiles_table(tmp_path_factory):
    """A TiledShapeStMan column with 16-row tiles (one tile per row-slab)."""
    path = str(tmp_path_factory.mktemp("small_tiles") / "tiles.tab")
    nrows = 2048
    desc = tables.maketabdesc(
        [
            tables.makearrcoldesc(
                "DATA",
                0j,
                ndim=2,
                valuetype="complex",
                datamanagertype="TiledShapeStMan",
                datamanagergroup="TiledData",
            )
        ]
    )
    dminfo = {
        "*1": {
            "NAME": "TiledData",
            "TYPE": "TiledShapeStMan",
            "SPEC": {"DEFAULTTILESHAPE": np.array([NPOL, NCHAN, 16], dtype=np.int32)},
            "COLUMNS": ["DATA"],
        }
    }
    tb = tables.table(path, desc, dminfo=dminfo, nrow=nrows, readonly=False, ack=False)
    try:
        values = np.arange(nrows * NCHAN * NPOL, dtype=np.float32).reshape(
            nrows, NCHAN, NPOL
        )
        tb.putcol("DATA", values.astype(np.complex64))
    finally:
        tb.close()
    return path, nrows


def _rchar() -> int:
    with open("/proc/self/io") as io_file:
        for line in io_file:
            if line.startswith("rchar:"):
                return int(line.split()[1])
    raise RuntimeError("no rchar in /proc/self/io")


@pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="needs /proc/self/io (Linux)"
)
def test_read_rows_to_grid_reads_each_tile_once(small_tiles_table):
    """
    Bytes read for a partition that mixes direct and scattered rows inside the
    same tiles equal those of one ascending read of the same rows (they were
    about 2x with a direct pass followed by a scatter pass).
    """
    path, nrows = small_tiles_table
    # In every block of 24 rows, rows 0-19 map to consecutive cells (a direct
    # segment) and rows 20-23 to their cells in reverse order (scattered), so
    # the 16-row tiles hold both kinds of rows.
    rows = np.arange(nrows)
    gidx = rows.copy()
    tail = rows % 24 >= 20
    gidx[tail] = (rows[tail] // 24) * 24 + 43 - rows[tail] % 24
    plan = rr.make_row_grid_plan(rows, gidx, nrows, min_direct_rows=8)
    assert plan.direct_lengths.size > 10 and plan.scatter_idx.size > 10

    def measure(read):
        # usernoread, as the converter opens MAIN: with the default (auto)
        # locking every call also reads the lock file (about 325 bytes)
        lock = {"option": "usernoread"}
        with tables.table(path, readonly=True, ack=False, lockoptions=lock) as tb:
            before = _rchar()
            read(tb)
            return _rchar() - before

    grid = np.empty((nrows, 1, NCHAN, NPOL), np.complex64)
    rows_path = measure(lambda tb: rr.read_rows_to_grid(tb, "DATA", plan, grid))
    out = np.empty((nrows, NCHAN, NPOL), np.complex64)
    one_pass = measure(lambda tb: rr.read_rows(tb, "DATA", rows, out))
    # a direct pass followed by a scatter pass read 1.3x here
    assert rows_path <= one_pass + 64
    np.testing.assert_array_equal(grid.reshape(out.shape)[gidx], out)


def test_read_rows_to_grid_dtype_differs_scatters(rows_tb):
    tb, ref = rows_tb
    rows, gidx, nt, nb = make_plan_case(3, keep=1.0)
    plan = rr.make_row_grid_plan(rows, gidx, nt * nb)
    grid = np.zeros((nt, nb, NCHAN, NPOL), np.complex128)
    stats = rr.read_rows_to_grid(tb, "TSM_DATA", plan, grid)
    expected = np.zeros_like(grid)
    expected[gidx // nb, gidx % nb] = ref["TSM_DATA"][rows]
    np.testing.assert_array_equal(grid, expected)
    assert stats.get("direct_rows", 0) == 0


def test_read_rows_to_grid_channel_slice(rows_tb):
    tb, ref = rows_tb
    rows, gidx, nt, nb = make_plan_case(1)
    plan = rr.make_row_grid_plan(rows, gidx, nt * nb)
    grid = sentinel_buffer((nt, nb, 2, NPOL), np.complex64)
    rr.read_rows_to_grid(tb, "TSM_DATA", plan, grid, chan=slice(2, 4), max_elems=8)
    expected = sentinel_buffer((nt, nb, 2, NPOL), np.complex64)
    expected[gidx // nb, gidx % nb] = ref["TSM_DATA"][rows][:, 2:4, :]
    np.testing.assert_array_equal(grid, expected)


def test_read_rows_to_grid_rejects_wrong_grid(rows_tb):
    tb, _ = rows_tb
    plan = rr.make_row_grid_plan(np.arange(4), np.arange(4), 4)
    with pytest.raises(ValueError):
        rr.read_rows_to_grid(
            tb, "TSM_DATA", plan, np.empty((3, 1, NCHAN, NPOL), np.complex64)
        )


def test_main_table_rows(rows_tb):
    tb, ref = rows_tb
    rows = ROW_SELECTIONS["few_runs"]
    part = rr.MainTableRows(tb, rows)
    assert part.nrows() == rows.size
    assert part.colnames() == tb.colnames()
    assert part.rownumbers() == rows.tolist()
    assert part.isscalarcol("SCALAR_INT") and not part.isscalarcol("TSM_DATA")
    np.testing.assert_array_equal(part.getcol("TSM_DATA"), ref["TSM_DATA"][rows])
    np.testing.assert_array_equal(
        part.getcol("SCALAR_INT", 0, -1), ref["SCALAR_INT"][rows]
    )
    np.testing.assert_array_equal(
        part.getcol("SCALAR_DOUBLE", 5, 7), ref["SCALAR_DOUBLE"][rows[5:12]]
    )
    assert part.getcell("SCALAR_DOUBLE", 3) == tb.getcell("SCALAR_DOUBLE", int(rows[3]))
    assert part.iscelldefined("TSM_DATA", 0)
    assert part.name() == tb.name()

    tidxs = np.arange(rows.size) // 4
    bidxs = np.arange(rows.size) % 4
    shape = (int(tidxs.max()) + 1, 4)
    plan = part.grid_plan(tidxs, bidxs, shape)
    assert part.grid_plan(tidxs, bidxs, shape) is plan
    with pytest.raises(ValueError):
        part.grid_plan(tidxs[:-1], bidxs[:-1], shape)
    part.close()
    with pytest.raises(IndexError):
        rr.MainTableRows(tb, np.array([0, NROWS]))
    with pytest.raises(ValueError):
        rr.MainTableRows(tb, np.array([3, 1]))


def test_main_table_rows_plans_follow_new_index_arrays(rows_tb):
    """
    The cached plans are keyed by the index arrays themselves, not by their
    id(): a new array that reuses a freed array's id must not get the stale
    plan (which would scatter the data into the wrong cells).
    """
    tb, _ = rows_tb
    nt, nb = 4, 10
    part = rr.MainTableRows(tb, np.arange(nt * nb))
    times = [np.repeat(np.arange(nt), nb), np.repeat(np.arange(nt)[::-1], nb)]
    baselines = np.tile(np.arange(nb), nt)
    n_stale = 0
    for trial in range(100):
        tidxs = times[trial % 2].copy()
        bidxs = baselines.copy()
        plan = part.grid_plan(tidxs, bidxs, (nt, nb))
        n_stale += not np.array_equal(plan.gidx, tidxs * nb + bidxs)
        assert part.grid_plan(tidxs, bidxs, (nt, nb)) is plan
        # freed in reverse order: the next copies typically get the same ids
        del plan
        del bidxs
        del tidxs
    assert n_stale == 0
    for trial in range(50):
        tidxs = np.repeat(np.roll(np.arange(nt), trial), nb)
        bidxs = np.tile(np.arange(nb), nt)
        chunk_rows = part.time_chunk_rows(tidxs, bidxs, (3, 1), nb)
        assert part.time_chunk_rows(tidxs, bidxs, (3, 1), nb) is chunk_rows
        assert chunk_rows.tidxs is tidxs
        del tidxs, bidxs, chunk_rows
    # equal values in a new array: a new plan (arrays may change in between)
    tidxs, bidxs = np.repeat(np.arange(nt), nb), np.tile(np.arange(nb), nt)
    plan = part.grid_plan(tidxs, bidxs, (nt, nb))
    assert part.grid_plan(tidxs.copy(), bidxs, (nt, nb)) is not plan
    assert part.grid_plan(tidxs.tolist(), bidxs, (nt, nb)).gidx.size == nt * nb
    part.release_plans()
    assert part._grid_plan is None and part._time_chunk_rows is None


@pytest.mark.parametrize("time_ordered", [True, False])
@pytest.mark.parametrize("time_chunks", [(1,) * 9, (4, 4, 1), (9,)])
def test_time_chunk_rows_matches_per_chunk_selection(time_ordered, time_chunks):
    rng = np.random.default_rng(7)
    nt, nb = 9, 5
    cells = np.flatnonzero(rng.random(nt * nb) < 0.7)
    if not time_ordered:  # baseline-major rows
        cells = cells[np.lexsort((cells // nb, cells % nb))]
    rows = np.sort(rng.choice(1000, cells.size, replace=False))
    tidxs, bidxs = cells // nb, cells % nb
    chunk_rows = rr.TimeChunkRows(rows, tidxs, bidxs, time_chunks, nb)
    assert chunk_rows.n_chunks == len(time_chunks)
    assert (chunk_rows.order is None) == (time_ordered or len(time_chunks) == 1)
    # the per-chunk arrays the time reader used to build for every column
    bounds = np.cumsum((0,) + time_chunks)
    chunk_of_row = np.searchsorted(bounds, tidxs, side="right") - 1
    for k in range(len(time_chunks)):
        idx = np.flatnonzero(chunk_of_row == k)
        rows_k, gidx_k = chunk_rows.chunk(k)
        np.testing.assert_array_equal(rows_k, rows[idx])
        np.testing.assert_array_equal(
            gidx_k, (tidxs[idx] - bounds[k]) * nb + bidxs[idx]
        )
        assert np.all(np.diff(rows_k) > 0)
    with pytest.raises(ValueError):
        rr.TimeChunkRows(rows, tidxs[:-1], bidxs, time_chunks, nb)
