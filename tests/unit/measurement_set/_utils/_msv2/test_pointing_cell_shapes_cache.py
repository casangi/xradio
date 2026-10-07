"""
The cell shapes of the POINTING table stored in the MS (the XRADIO_PARTITIONS
sub-table) by the ``xradio_msv2`` engine: the scan of the shapes of the
POINTING array cells (``backend_pointing._scan_cell_shapes``) is made by the
first open of an MS and stored with the fingerprint of the POINTING table;
later opens, in any process, take it from there while that fingerprint is
unchanged (the partition_cache modes "auto" and "read"; "rebuild" scans and
stores again, "off" neither uses nor stores it), and an MS that cannot be
written keeps the per-process memo. Sub-tables without it (stored before)
are read as before; a torn or foreign row is ignored.
"""

import contextlib
import json
import os
import stat
import subprocess
import sys
import textwrap

import numpy as np
import pytest
import xarray as xr

tables = pytest.importorskip("casacore.tables")

from _xradio_xarray_backends import MSv2BackendEntrypoint  # noqa: E402
from xradio.measurement_set._utils._msv2 import (  # noqa: E402
    backend_pointing as bpt,
)
from xradio.measurement_set._utils._msv2 import partition_cache  # noqa: E402
from xradio.measurement_set._utils._msv2._tables.table_lock_file import (  # noqa: E402
    table_fingerprint,
)
from xradio.measurement_set._utils._msv2.partition_cache import (  # noqa: E402
    POINTING_SHAPES_KEY,
    SUBTABLE_COLUMNS,
    SUBTABLE_NAME,
    normalise_row,
    row_checksum,
)

ENGINE = MSv2BackendEntrypoint
# The array columns of the generated MSs' POINTING whose description does
# not fix the cell shape (scanned)
SCANNED = {"DIRECTION", "TARGET"}


@pytest.fixture(autouse=True)
def _fresh_memos():
    """Every test starts (and leaves) with empty per-process memos."""
    bpt.clear_pointing_memos()
    partition_cache.clear_partition_memo()
    yield
    bpt.clear_pointing_memos()
    partition_cache.clear_partition_memo()


@pytest.fixture
def scans(monkeypatch):
    """The columns of the cell-shape scans (_scan_cell_shapes calls)."""
    found = []
    scan = bpt._scan_cell_shapes

    def spy(table, col, nrows):
        found.append(col)
        return scan(table, col, nrows)

    monkeypatch.setattr(bpt, "_scan_cell_shapes", spy)
    return found


def new_process() -> None:
    """Forget what this process keeps (as a new process would)."""
    bpt.clear_pointing_memos()
    partition_cache.clear_partition_memo()


def open_tree(msname: str, mode: str, **options) -> xr.DataTree:
    return xr.open_datatree(msname, engine=ENGINE, partition_cache=mode, **options)


def subtable_rows(msname: str) -> list[dict]:
    """The normalised rows of the sub-table."""
    with tables.table(os.path.join(msname, SUBTABLE_NAME), ack=False) as table:
        return [
            normalise_row({name: table.getcell(name, row) for name in SUBTABLE_COLUMNS})
            for row in range(table.nrows())
        ]


def shapes_rows(msname: str) -> list[dict]:
    """The rows of the POINTING cell shapes."""
    return [
        row for row in subtable_rows(msname) if row["SCHEME_KEY"] == POINTING_SHAPES_KEY
    ]


def identities(rows: list[dict]) -> list[tuple]:
    """(SCHEME_KEY, CACHE_ID, CHECKSUM) of rows: whether they were written
    again."""
    return [(row["SCHEME_KEY"], row["CACHE_ID"], row["CHECKSUM"]) for row in rows]


def partition_rows(msname: str) -> list[dict]:
    """The rows of partitions."""
    return [
        row for row in subtable_rows(msname) if row["SCHEME_KEY"] != POINTING_SHAPES_KEY
    ]


def history_nrows(msname: str) -> int:
    with tables.table(os.path.join(msname, "HISTORY"), ack=False) as table:
        return table.nrows()


def pointing_values(tree: xr.DataTree) -> dict:
    """The values of every pointing_xds of a tree, by MSv4."""
    return {
        name: node["pointing_xds"].to_dataset(inherit=False).compute()
        for name, node in tree.children.items()
        if "pointing_xds" in node.children
    }


def assert_same_pointing(a: xr.DataTree, b: xr.DataTree) -> None:
    values_a, values_b = pointing_values(a), pointing_values(b)
    assert values_a.keys() == values_b.keys() and values_a
    for name, xds in values_a.items():
        xr.testing.assert_identical(xds, values_b[name])


def file_digests(path: str) -> dict:
    """(size, mtime_ns) of every file under ``path`` but the lock files."""
    found = {}
    for dirpath, _, filenames in os.walk(path):
        for name in filenames:
            if name == "table.lock":
                continue
            full = os.path.join(dirpath, name)
            info = os.stat(full)
            found[full] = (info.st_size, info.st_mtime_ns)
    return found


def test_first_open_stores_later_opens_reuse(ms_copy, scans):
    """The first "auto" open scans the cell shapes of the POINTING array
    columns and stores them (a row of their own: no HISTORY row, the
    partitions untouched); later opens ("auto", "read") take them from the
    MS (no scan) and open the same pointing_xds; "off" scans and stores
    nothing; "rebuild" scans and stores again (here the same row)."""
    msname = ms_copy("rich")
    first = open_tree(msname, "auto")
    assert sorted(scans) == sorted(SCANNED)
    assert bpt.POINTING_INDEX_MEMO.stats["shapes stored"] == 1
    (row,) = shapes_rows(msname)
    assert row["ALGORITHM_VERSION"] == bpt.POINTING_SHAPES_VERSION
    assert json.loads(row["FINGERPRINT"]) == table_fingerprint(
        os.path.join(msname, "POINTING")
    )
    assert row["CHECKSUM"] == row_checksum(row)
    n_history = history_nrows(msname)
    partitions = identities(partition_rows(msname))
    for mode in ("auto", "read"):
        scans.clear()
        new_process()
        later = open_tree(msname, mode)
        assert scans == []
        assert bpt.POINTING_INDEX_MEMO.stats["stored shapes"] == 1
        assert_same_pointing(later, first)
    scans.clear()
    open_tree(msname, "auto")  # (the memo of this process)
    assert scans == [] and bpt.POINTING_INDEX_MEMO.stats["reads"] == 1
    assert history_nrows(msname) == n_history
    assert identities(partition_rows(msname)) == partitions
    for mode in ("off", "rebuild"):
        scans.clear()
        new_process()
        assert_same_pointing(open_tree(msname, mode), first)
        assert sorted(scans) == sorted(SCANNED)
    assert bpt.POINTING_INDEX_MEMO.stats["shapes hit-race"] == 1  # (rebuild)
    assert "stored shapes" not in bpt.POINTING_INDEX_MEMO.stats
    scans.clear()
    open_tree(msname, "rebuild")  # (scans again, also with the memo)
    assert sorted(scans) == sorted(SCANNED)
    assert identities(shapes_rows(msname)) == identities([row])


def test_stored_cell_shapes_follow_the_pointing_table(ms_copy, scans):
    """The stored cell shapes are used only while the POINTING table is the
    one they were scanned from (its fingerprint): after a write of POINTING
    (a cell of another shape, or values only) the next open scans again,
    finds the new cell, and replaces the stored shapes, which the open
    after it uses."""
    msname = ms_copy("rich")
    open_tree(msname, "auto")
    pointing = os.path.join(msname, "POINTING")
    with tables.table(pointing, readonly=False, ack=False) as tb:
        tb.putcell("TARGET", 5, np.zeros((2, 2)))
    for expected_odd in ([5], [5]):
        scans.clear()
        new_process()
        tree = open_tree(msname, "auto")
        assert sorted(scans) == sorted(SCANNED)
        (row,) = shapes_rows(msname)
        assert json.loads(row["FINGERPRINT"]) == table_fingerprint(pointing)
        shapes = bpt.decode_pointing_shapes(
            row["PARTITIONS"], table_fingerprint(pointing)["nrows"]
        )
        assert shapes["TARGET"].rows.tolist() == expected_odd
        assert shapes["TARGET"].others == ((2, 2),)
        assert shapes["DIRECTION"].rows.size == 0
        scans.clear()
        new_process()
        assert_same_pointing(open_tree(msname, "auto"), tree)
        assert scans == []
        with tables.table(pointing, readonly=False, ack=False) as tb:
            tb.putcol("TRACKING", ~tb.getcol("TRACKING"))


def test_another_process_reuses_the_stored_cell_shapes(ms_copy):
    """An open in another process takes the cell shapes that this one
    stored: it scans nothing."""
    msname = ms_copy("rich")
    open_tree(msname, "auto")
    script = textwrap.dedent(
        f"""
        import json
        import xarray as xr
        from _xradio_xarray_backends import MSv2BackendEntrypoint
        from xradio.measurement_set._utils._msv2 import backend_pointing as bpt
        scans = []
        scan = bpt._scan_cell_shapes
        bpt._scan_cell_shapes = lambda *args: scans.append(args[1]) or scan(*args)
        tree = xr.open_datatree(
            {msname!r}, engine=MSv2BackendEntrypoint, partition_cache="auto"
        )
        print(json.dumps([scans, dict(bpt.POINTING_INDEX_MEMO.stats)]))
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=True,
        env=dict(os.environ),
    )
    scans, stats = json.loads(result.stdout.strip().splitlines()[-1])
    assert scans == [] and stats["stored shapes"] == 1


def test_caches_without_cell_shapes(ms_copy, scans):
    """A sub-table stored without the cell shapes (partitions only, as
    before them): "read" uses its partitions and scans, writing nothing;
    "auto" adds the cell shapes and keeps the partitions and HISTORY."""
    msname = ms_copy("rich")
    partition_cache.load_or_create_partitions(msname, [], "auto")
    assert shapes_rows(msname) == []
    stored = identities(subtable_rows(msname))
    n_history = history_nrows(msname)
    before = file_digests(msname)
    new_process()
    open_tree(msname, "read")
    assert sorted(scans) == sorted(SCANNED)
    assert partition_cache.PARTITIONS_MEMO.stats["stored hits"] == 1
    assert file_digests(msname) == before
    new_process()
    open_tree(msname, "auto")
    assert partition_cache.PARTITIONS_MEMO.stats["stored hits"] == 1
    assert len(shapes_rows(msname)) == 1
    assert identities(partition_rows(msname)) == stored
    assert history_nrows(msname) == n_history


def _rewrite_shapes_row(msname: str, change, checksum: bool) -> None:
    """Rewrite the row of the cell shapes with ``change(values)`` (its
    CHECKSUM renewed or not)."""
    with tables.table(
        os.path.join(msname, SUBTABLE_NAME), readonly=False, ack=False
    ) as table:
        keys = table.getcol("SCHEME_KEY")
        index = keys.index(POINTING_SHAPES_KEY)
        values = {name: table.getcell(name, index) for name in SUBTABLE_COLUMNS}
        change(values)
        if checksum:
            values["CHECKSUM"] = row_checksum(normalise_row(values))
        for name in SUBTABLE_COLUMNS:
            table.putcell(name, index, values[name])


def _other_rows(values: dict) -> None:
    content = json.loads(values["PARTITIONS"])
    content["columns"]["TARGET"]["rows"] = [10**9]
    content["columns"]["TARGET"]["codes"] = [0]
    content["columns"]["TARGET"]["others"] = [[2, 2]]
    values["PARTITIONS"] = json.dumps(content)


@pytest.mark.parametrize(
    "change, checksum",
    [
        (_other_rows, False),  # (a torn row: CHECKSUM mismatch)
        (_other_rows, True),  # (rows beyond the table)
        (lambda values: values.update(PARTITIONS="[1, 2"), True),  # (no JSON)
        (lambda values: values.update(PARTITIONS="{}"), True),  # (no version)
        (lambda values: values.update(FINGERPRINT="{}"), True),  # (another table)
    ],
)
def test_unusable_stored_cell_shapes_are_replaced(ms_copy, scans, change, checksum):
    """Stored cell shapes that are torn, corrupt or of another table are not
    used (the open scans, with the same result) and are replaced by "auto";
    the next open uses those."""
    msname = ms_copy("rich")
    first = open_tree(msname, "auto")
    (row,) = shapes_rows(msname)
    _rewrite_shapes_row(msname, change, checksum)
    scans.clear()
    new_process()
    assert_same_pointing(open_tree(msname, "auto"), first)
    assert sorted(scans) == sorted(SCANNED)
    assert "stored shapes" not in bpt.POINTING_INDEX_MEMO.stats
    (replaced,) = shapes_rows(msname)
    assert replaced["PARTITIONS"] == row["PARTITIONS"]
    assert replaced["FINGERPRINT"] == row["FINGERPRINT"]
    scans.clear()
    new_process()
    open_tree(msname, "auto")
    assert scans == []


def test_cell_shapes_row_is_not_evicted(ms_copy):
    """The row of the cell shapes is neither counted nor evicted with the
    rows of partitions (at most MAX_STORED_ROWS of those)."""
    msname = ms_copy("rich")
    open_tree(msname, "auto")
    schemes = [
        [],
        ["FIELD_ID"],
        ["SCAN_NUMBER"],
        ["STATE_ID"],
        ["SOURCE_ID"],
        ["SUB_SCAN_NUMBER"],
        ["FIELD_ID", "SCAN_NUMBER"],
        ["FIELD_ID", "STATE_ID"],
        ["SCAN_NUMBER", "STATE_ID"],
        ["ANTENNA1"],
    ]
    for scheme in schemes:
        partition_cache.load_or_create_partitions(msname, scheme, "auto")
    assert len(partition_rows(msname)) == partition_cache.MAX_STORED_ROWS
    assert len(shapes_rows(msname)) == 1


@pytest.mark.skipif(os.geteuid() == 0, reason="root writes read-only files")
def test_read_only_ms_uses_the_stored_cell_shapes(ms_copy, scans):
    """An MS that cannot be written: its stored cell shapes are used (no
    scan) and nothing is written; without them, every new process scans
    (the per-process memo serves the opens of a process)."""
    msname = ms_copy("rich")
    open_tree(msname, "auto")
    paths = [msname] + [
        os.path.join(dirpath, name)
        for dirpath, dirnames, filenames in os.walk(msname)
        for name in dirnames + filenames
    ]
    modes = {path: stat.S_IMODE(os.lstat(path).st_mode) for path in paths}
    before = file_digests(msname)
    for path in paths:
        os.chmod(path, modes[path] & ~0o222)
    try:
        scans.clear()
        new_process()
        open_tree(msname, "auto")
        assert scans == []
        assert file_digests(msname) == before
    finally:
        for path in paths:
            os.chmod(path, modes[path])
    partition_cache.remove_partition_cache(msname)
    for path in [p for p in paths if os.path.exists(p)]:
        os.chmod(path, stat.S_IMODE(os.lstat(path).st_mode) & ~0o222)
    new_process()
    try:
        with pytest.warns(partition_cache.PartitionCacheWarning):
            for expected in (sorted(SCANNED), []):
                scans.clear()
                partition_cache.clear_partition_memo()
                open_tree(msname, "auto")
                assert sorted(scans) == expected
    finally:
        for dirpath, dirnames, filenames in os.walk(msname):
            for name in [dirpath] + [
                os.path.join(dirpath, n) for n in dirnames + filenames
            ]:
                os.chmod(name, stat.S_IMODE(os.lstat(name).st_mode) | 0o200)


def test_reference_pointing_table_is_not_stored(ms_copy, scans):
    """A POINTING table whose rows live in another table (a reference
    table: its fingerprint does not tell their writes): its cell shapes are
    neither stored nor taken from the MS."""
    msname = ms_copy("rich")
    pointing = os.path.join(msname, "POINTING")
    os.rename(pointing, pointing + "_ROWS")
    with tables.table(pointing + "_ROWS", ack=False) as tb:
        rows = tb.selectrows(np.arange(tb.nrows()))
        rows.copy(pointing, deep=False).close()
        rows.close()
    open_tree(msname, "auto")
    assert sorted(scans) == sorted(SCANNED)
    assert shapes_rows(msname) == []
    stats = bpt.POINTING_INDEX_MEMO.stats
    assert stats["shapes memory:POINTING writes not followed"] == 1
    scans.clear()
    new_process()
    open_tree(msname, "auto")
    assert sorted(scans) == sorted(SCANNED)


@contextlib.contextmanager
def held(table_path: str):
    """Another process that holds a write lock on a table."""
    script = textwrap.dedent(
        f"""
        import time
        from casacore import tables
        t = tables.table({table_path!r}, readonly=False, lockoptions="user", ack=False)
        t.lock(True)
        print("ready", flush=True)
        time.sleep(120)
        """
    )
    holder = subprocess.Popen(
        [sys.executable, "-c", script], stdout=subprocess.PIPE, text=True
    )
    try:
        assert holder.stdout.readline().strip() == "ready"
        yield holder
    finally:
        holder.kill()
        holder.wait()


def test_store_pointing_shapes_statuses(ms_copy):
    """store_pointing_shapes never raises: not linked (no partitions
    stored), the sub-table locked by another process, stored, the same
    again (hit-race); lookup_pointing_shapes gives what was stored."""
    msname = ms_copy("dense")
    version = bpt.POINTING_SHAPES_VERSION
    store = partition_cache.store_pointing_shapes
    assert store(msname, "{}", '{"a": 1}', version) == "memory:not linked"
    partition_cache.load_or_create_partitions(msname, [], "auto")
    with held(os.path.join(msname, SUBTABLE_NAME)):
        assert store(msname, "{}", '{"a": 1}', version) == "memory:locked"
    assert partition_cache.lookup_pointing_shapes(msname, version) is None
    assert store(msname, "{}", '{"a": 1}', version) == "stored"
    assert store(msname, "{}", '{"a":1}', version) == "hit-race"
    assert partition_cache.lookup_pointing_shapes(msname, version) == (
        "{}",
        '{"a": 1}',
    )
    assert store(msname, "{}", '{"a": 2}', version) == "stored"
    assert partition_cache.lookup_pointing_shapes(msname, version + 1) is None
    assert len(shapes_rows(msname)) == 1
