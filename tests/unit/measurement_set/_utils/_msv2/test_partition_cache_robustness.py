"""
Robustness of the partition cache of the MSv2 xarray backend
(partition_cache.py, Linux): racing writer processes, readers of a row being
rewritten, writers killed or failing at every step of the write protocol,
tables locked by other processes, and read-only MSs. On copies of the
generated MSs; trees are compared with the converted processing set.
"""

import contextlib
import json
import os
import shutil
import signal
import stat
import subprocess
import sys
import textwrap
import threading
import time
import warnings

import numpy as np
import pytest
import xarray as xr
from casacore import tables

from _xradio_xarray_backends import MSv2BackendEntrypoint
from xradio.measurement_set import convert_msv2_to_processing_set, open_processing_set
from xradio.measurement_set._utils._msv2 import partition_cache
from xradio.measurement_set._utils._msv2.backend_errors import PartitionCacheWarning
from xradio.measurement_set._utils._msv2.partition_cache import (
    SUBTABLE_COLUMNS,
    SUBTABLE_NAME,
    TMP_PREFIX,
    lookup_stored_row,
    normalise_row,
    row_checksum,
)
from xradio.testing.measurement_set.equivalence import assert_nodes_identical

ENGINE = MSv2BackendEntrypoint

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="Linux (fcntl locks, /proc)"
)


@pytest.fixture(autouse=True)
def _clear_memo():
    partition_cache.clear_partition_memo()
    yield
    partition_cache.clear_partition_memo()


@pytest.fixture(scope="module")
def oracle(backend_ms, tmp_path_factory):
    """``oracle(variant, scheme)``: the processing set the converter writes
    for a generated MS (written on first use)."""
    base = tmp_path_factory.mktemp("cache_oracle")
    paths: dict[tuple, str] = {}

    def get(variant: str, scheme: tuple[str, ...] = ()) -> str:
        key = (variant, tuple(scheme))
        if key not in paths:
            out = str(base / f"{variant}_{len(paths)}.ps.zarr")
            convert_msv2_to_processing_set(
                backend_ms(variant), out, partition_scheme=list(scheme)
            )
            paths[key] = out
        return paths[key]

    yield get
    shutil.rmtree(base, ignore_errors=True)


def same_name_copy(ms_copy, variant: str, layout: str = "shared") -> str:
    """A copy of a generated MS with its name (the MSv4 names derive from it,
    as in its oracle)."""
    return ms_copy(variant, layout, name=f"{variant}.ms")


def file_digests(path: str) -> dict[str, tuple[int, str]]:
    import hashlib

    digests = {}
    for dirpath, _, filenames in os.walk(path):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            with open(file_path, "rb") as f:
                content = f.read()
            digests[os.path.relpath(file_path, path)] = (
                len(content),
                hashlib.sha256(content).hexdigest(),
            )
    return digests


def history_nrows(msname: str) -> int:
    with tables.table(os.path.join(msname, "HISTORY"), ack=False) as history:
        return history.nrows()


def stored_keys(msname: str) -> list[str]:
    subtable = os.path.join(msname, SUBTABLE_NAME)
    if not os.path.isdir(subtable):
        return []
    with tables.table(subtable, ack=False) as table:
        return [str(key) for key in table.getcol("SCHEME_KEY")] if table.nrows() else []


def tmp_tables(msname: str) -> list[str]:
    return [name for name in os.listdir(msname) if name.startswith(TMP_PREFIX)]


def dangling(msname: str) -> bool:
    """Whether MAIN has the keyword without the sub-table."""
    return partition_cache.link_state(msname) == "dangling"


def status_of(msname: str, scheme=(), mode: str = "auto") -> str:
    partition_cache.clear_partition_memo()
    return partition_cache.load_or_create_partitions(
        os.path.abspath(msname), list(scheme), mode
    ).status


def check_tree(msname: str, reference: str, scheme=(), mode: str = "auto") -> None:
    tree = xr.open_datatree(
        msname,
        engine=ENGINE,
        chunks={},
        partition_cache=mode,
        partition_scheme=list(scheme),
    )
    assert_nodes_identical(tree, open_processing_set(reference))


# --- racing writer processes -----------------------------------------------------------

RACER = textwrap.dedent(
    """
    import json, os, sys, time
    import xarray as xr
    from _xradio_xarray_backends import MSv2BackendEntrypoint
    from xradio.measurement_set import open_processing_set
    from xradio.measurement_set._utils._msv2 import partition_cache
    from xradio.testing.measurement_set.equivalence import assert_nodes_identical

    msname, scheme, reference, ready, go = sys.argv[1:]
    scheme = json.loads(scheme)
    open(ready, "w").close()
    while not os.path.exists(go):
        time.sleep(0.002)
    result = partition_cache.load_or_create_partitions(msname, scheme, "auto")
    # ("read": the open writes nothing more)
    tree = xr.open_datatree(msname, engine=MSv2BackendEntrypoint, chunks={},
                            partition_cache="read", partition_scheme=scheme)
    assert_nodes_identical(tree, open_processing_set(reference))
    print(json.dumps({"status": result.status}))
    """
)


def race(msname: str, schemes: list[list[str]], references: list[str], tmp_path):
    """Run one process per scheme, released at once; returns their statuses."""
    go = str(tmp_path / f"go-{time.monotonic_ns()}")
    readies, processes = [], []
    for i, (scheme, reference) in enumerate(zip(schemes, references, strict=True)):
        ready = f"{go}-ready-{i}"
        readies.append(ready)
        processes.append(
            subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    RACER,
                    msname,
                    json.dumps(scheme),
                    reference,
                    ready,
                    go,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        )
    deadline = time.monotonic() + 120
    while not all(os.path.exists(r) for r in readies):
        assert time.monotonic() < deadline, "racers did not start"
        time.sleep(0.01)
    open(go, "w").close()
    statuses = []
    for process in processes:
        out, err = process.communicate(timeout=300)
        assert process.returncode == 0, err[-3000:]
        statuses.append(json.loads(out.strip().splitlines()[-1])["status"])
    return statuses


def test_racing_processes(ms_copy, oracle, tmp_path):
    """4 processes open one cold MS at once (one scheme, then 3 schemes, one
    of them twice in another key order): never two rows of a scheme, one
    HISTORY row per stored content, no temporary sub-table left, and every
    tree equals the converted processing set. (On "dense", with one field
    and scan, every scheme here gives the partitions of [].)"""
    msname = same_name_copy(ms_copy, "dense")
    n_history = history_nrows(msname)
    reference = oracle("dense")
    statuses = race(msname, [["FIELD_ID"]] * 4, [reference] * 4, tmp_path)
    assert stored_keys(msname) == ['["FIELD_ID"]']
    stored = statuses.count("stored")
    # (a racer that loses a lock keeps its partitions in memory; one that
    # finds the row in the store's re-check is a hit-race; one that reads the
    # stored row is a hit)
    assert stored <= 1 and history_nrows(msname) == n_history + stored
    assert set(statuses) <= {"stored", "hit", "hit-race", "memory:locked"}
    schemes = [
        [],
        ["SCAN_NUMBER"],
        ["FIELD_ID", "SCAN_NUMBER"],
        ["SCAN_NUMBER", "FIELD_ID"],
    ]
    statuses = race(msname, schemes, [reference] * 4, tmp_path)
    keys = stored_keys(msname)
    assert len(keys) == len(set(keys)) <= 4
    assert set(keys) <= {
        '["FIELD_ID"]',
        "[]",
        '["SCAN_NUMBER"]',
        '["FIELD_ID", "SCAN_NUMBER"]',
    }
    assert set(statuses) <= {"stored", "hit", "hit-race", "memory:locked"}
    assert history_nrows(msname) == n_history + stored + statuses.count("stored")
    assert tmp_tables(msname) == [] and not dangling(msname)
    # whatever the racers left, the next opens store the rest
    for scheme in schemes:
        assert status_of(msname, scheme) in ("hit", "stored")
    assert len(stored_keys(msname)) == 4


# --- readers of a row being rewritten ---------------------------------------------------

REWRITER = textwrap.dedent(
    """
    import json, sys, time
    import numpy as np
    from casacore import tables

    subtable, rows_file, seconds = sys.argv[1], sys.argv[2], float(sys.argv[3])
    with open(rows_file) as f:
        rows = json.load(f)
    for row in rows:
        for name in ("ROW_STARTS", "ROW_LENGTHS", "PARTITION_BOUNDS"):
            row[name] = np.asarray(row[name], dtype=float)
    table = tables.table(subtable, readonly=False, lockoptions={"option": "usernoread"},
                         ack=False)
    end, count = time.time() + seconds, 0
    print("ready", flush=True)
    while time.time() < end:
        row = rows[count % 2]
        table.lock(write=True, nattempts=10)
        for name in row:
            table.putcell(name, 0, row[name])
        table.flush()
        table.unlock()
        count += 1
    table.close()
    print(count, flush=True)
    """
)


def _row_json(row: dict) -> dict:
    return {
        name: (value.tolist() if isinstance(value, np.ndarray) else value)
        for name, value in row.items()
    }


def test_readers_never_accept_a_torn_row(ms_copy, tmp_path):
    """A process rewrites the stored row back to back, alternating two valid
    rows, while 2 threads read it: reads are torn (exceptions, CHECKSUM
    mismatches) but no accepted row is torn."""
    msname = ms_copy("rich")
    assert status_of(msname, ["FIELD_ID", "SCAN_NUMBER"]) == "stored"
    subtable = os.path.join(msname, SUBTABLE_NAME)
    with tables.table(subtable, ack=False) as table:
        first = normalise_row({n: table.getcell(n, 0) for n in SUBTABLE_COLUMNS})
    second = dict(first, CACHE_ID="f" * 32, TIME=first["TIME"] + 1.0)
    second["PARTITIONS"] = first["PARTITIONS"] + " "
    second["CHECKSUM"] = row_checksum(second)
    rows_file = tmp_path / "rows.json"
    rows_file.write_text(json.dumps([_row_json(first), _row_json(second)]))
    writer = subprocess.Popen(
        [sys.executable, "-c", REWRITER, subtable, str(rows_file), "4"],
        stdout=subprocess.PIPE,
        text=True,
    )
    assert writer.stdout.readline().strip() == "ready"
    key = '["FIELD_ID", "SCAN_NUMBER"]'
    counts = {"row": 0, "accepted torn": 0, "rejected": 0}
    valid = {first["CHECKSUM"]: first, second["CHECKSUM"]: second}
    lock = threading.Lock()

    def read(stop):
        while time.monotonic() < stop:
            lookup = lookup_stored_row(msname, key)
            with lock:
                if lookup.state != "row":
                    counts["rejected"] += 1
                    continue
                counts["row"] += 1
                expected = valid.get(lookup.row["CHECKSUM"])
                same = expected is not None and all(
                    np.array_equal(lookup.row[n], expected[n]) for n in SUBTABLE_COLUMNS
                )
                if not same:
                    counts["accepted torn"] += 1

    stop = time.monotonic() + 3.5
    readers = [threading.Thread(target=read, args=(stop,)) for _ in range(2)]
    for reader in readers:
        reader.start()
    for reader in readers:
        reader.join()
    out, _ = writer.communicate(timeout=60)
    assert writer.returncode == 0
    assert int(out.strip()) > 10  # (rewrites)
    assert counts["row"] > 10
    assert counts["accepted torn"] == 0, counts


KILLED_WRITER = textwrap.dedent(
    """
    import sys, time
    from casacore import tables

    table = tables.table(sys.argv[1], readonly=False, lockoptions={"option": "usernoread"},
                         ack=False)
    table.lock(write=True, nattempts=10)
    table.putcell("TIME", 0, table.getcell("TIME", 0) + 5.0)
    table.putcell("PARTITIONS", 0, table.getcell("PARTITIONS", 0) + " ")
    table.flush()
    print("half written", flush=True)
    time.sleep(120)
    """
)


def test_writer_killed_mid_row(ms_copy, oracle):
    """A writer killed (SIGKILL) in the middle of a row: the next open finds
    a CHECKSUM mismatch, computes the partitions and stores them again."""
    msname = same_name_copy(ms_copy, "rich")
    assert status_of(msname, ["FIELD_ID"]) == "stored"
    writer = subprocess.Popen(
        [sys.executable, "-c", KILLED_WRITER, os.path.join(msname, SUBTABLE_NAME)],
        stdout=subprocess.PIPE,
        text=True,
    )
    assert writer.stdout.readline().strip() == "half written"
    writer.send_signal(signal.SIGKILL)
    writer.wait()
    assert lookup_stored_row(msname, '["FIELD_ID"]').state == "torn"
    n_history = history_nrows(msname)
    assert status_of(msname, ["FIELD_ID"]) == "stored"
    assert history_nrows(msname) == n_history + 1
    assert stored_keys(msname) == ['["FIELD_ID"]']
    assert status_of(msname, ["FIELD_ID"]) == "hit"
    check_tree(msname, oracle("rich", ("FIELD_ID",)), ("FIELD_ID",))


# --- tables locked by other processes ---------------------------------------------------


def hold_table(msname: str, lockoptions: dict | None = None, write_lock: bool = False):
    """A subprocess that opens a table with python-casacore (by default with
    its default, auto, locking), reads a column (with ``write_lock``: opens
    it for update with user locking and takes its write lock) and waits
    until killed."""
    if write_lock:
        options, take = ', readonly=False, lockoptions="user"', "t.lock(True)"
    else:
        options = "" if lockoptions is None else f", lockoptions={lockoptions!r}"
        take = "t.getcol(t.colnames()[0])"
    script = textwrap.dedent(
        f"""
        import time
        from casacore import tables
        t = tables.table({msname!r}, ack=False{options})
        {take}
        print("ready", flush=True)
        time.sleep(120)
        """
    )
    holder = subprocess.Popen(
        [sys.executable, "-c", script], stdout=subprocess.PIPE, text=True
    )
    assert holder.stdout.readline().strip() == "ready"
    return holder


@contextlib.contextmanager
def held(msname: str, lockoptions: dict | None = None, write_lock: bool = False):
    holder = hold_table(msname, lockoptions, write_lock)
    try:
        yield holder
    finally:
        holder.kill()
        holder.wait()


def test_main_held_by_another_process(ms_copy, oracle):
    """Another process has MAIN open with default (auto) locking: the open
    does not wait (one lock attempt), warns 'locked' once, writes nothing,
    and gives the converter's tree. A holder without read locks (usernoread)
    does not block the store."""
    msname = same_name_copy(ms_copy, "dense")
    reference = oracle("dense")
    with held(msname):
        before = file_digests(msname)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            start = time.monotonic()
            assert status_of(msname) == "memory:locked"
            assert time.monotonic() - start < 1.0
            assert status_of(msname) == "memory:locked"
            check_tree(msname, reference)
        notices = [w for w in caught if w.category is PartitionCacheWarning]
        assert len(notices) == 1
        assert "locked: another process has the MS open" in str(notices[0].message)
        assert file_digests(msname) == before
        assert SUBTABLE_NAME not in os.listdir(msname)
    with held(msname, {"option": "usernoread"}):
        assert status_of(msname) == "stored"
    assert status_of(msname) == "hit"


@pytest.mark.parametrize("table", ["", "HISTORY", SUBTABLE_NAME])
def test_tables_write_locked_by_another_process(ms_copy, table):
    """Another process holds the write lock of MAIN, HISTORY or the
    sub-table (user locking, as a writer that keeps it): the open does not
    wait for it (the checks before a store open the tables without locks,
    then one lock attempt) and keeps the partitions in memory; once the
    lock is released, they are stored."""
    msname = ms_copy("dense")
    if table == SUBTABLE_NAME:
        assert status_of(msname, ["FIELD_ID"]) == "stored"
    with held(os.path.join(msname, table), write_lock=True):
        start = time.monotonic()
        with pytest.warns(PartitionCacheWarning, match="locked"):
            assert status_of(msname) == "memory:locked"
        assert time.monotonic() - start < 5.0
    assert status_of(msname) == "stored"


def test_history_held_by_another_process(ms_copy):
    """HISTORY locked elsewhere: nothing but the (linked, empty) sub-table is
    written; the next open stores."""
    msname = ms_copy("dense")
    with held(os.path.join(msname, "HISTORY")):
        n_history = history_nrows(msname)
        with pytest.warns(PartitionCacheWarning, match="locked"):
            assert status_of(msname) == "memory:locked"
        assert history_nrows(msname) == n_history
        assert stored_keys(msname) == []
        assert not dangling(msname)
    assert status_of(msname) == "stored"


# --- failures at every step of the write protocol --------------------------------------


def _fail(*args, **kwargs):
    raise OSError("simulated failure")


def _after(function):
    """``function``, then a failure."""

    def run(*args, **kwargs):
        function(*args, **kwargs)
        raise OSError("simulated failure")

    return run


def _half_row(table, index, values):
    for name in SUBTABLE_COLUMNS[: len(SUBTABLE_COLUMNS) // 2]:
        table.putcell(name, index, values[name])
    raise OSError("simulated failure")


CRASH_POINTS = {
    # step: (function of partition_cache replaced, its replacement, state
    # left: sub-table, keyword, rows, HISTORY rows added)
    "temporary table made": ("_rename_subtable", _fail, (False, False, 0, 0)),
    "sub-table renamed": ("_put_main_keyword", _fail, (False, False, 0, 0)),
    "keyword written": ("_write_row", _fail, (True, True, 0, 0)),
    "HISTORY row written": (
        "_append_history_row",
        _after(partition_cache._append_history_row),
        (True, True, 0, 1),
    ),
    "row half written": ("_put_row_cells", _half_row, (True, True, 1, 1)),
}


@pytest.mark.parametrize("step", list(CRASH_POINTS))
def test_failure_after_each_step(step, ms_copy, monkeypatch, oracle):
    """A failure after each step of the write protocol: the open gives the
    converter's tree (partitions in memory, a warning); no keyword without
    its sub-table, no temporary or unlinked sub-table; the next open stores
    the partitions."""
    msname = same_name_copy(ms_copy, "dense")
    n_history = history_nrows(msname)
    name, replacement, (subtable, keyword, rows, added) = CRASH_POINTS[step]
    with monkeypatch.context() as m:
        m.setattr(partition_cache, name, replacement)
        with pytest.warns(PartitionCacheWarning, match="write failed"):
            assert status_of(msname) == "memory:write failed"
    assert os.path.isdir(os.path.join(msname, SUBTABLE_NAME)) == subtable
    assert (SUBTABLE_NAME in _keywords(msname)) == keyword
    assert len(stored_keys(msname)) == rows
    assert history_nrows(msname) == n_history + added
    assert tmp_tables(msname) == [] and not dangling(msname)
    assert partition_cache.link_state(msname) in ("absent", "linked")
    if rows:  # (a half written row reads torn, or raises: undefined cells)
        assert lookup_stored_row(msname, "[]").state in ("torn", "unreadable")
    assert status_of(msname) == "stored"
    assert stored_keys(msname) == ["[]"]
    assert status_of(msname) == "hit"
    check_tree(msname, oracle("dense"))


def _keywords(msname: str) -> list[str]:
    with tables.table(msname, ack=False) as main_tb:
        return list(main_tb.keywordnames())


def test_states_left_by_a_killed_writer(ms_copy):
    """The states a writer killed at each step leaves (crash consistency):
    the next open recovers from each."""
    msname = ms_copy("dense")
    # a temporary table of a process that is gone: removed by the next store
    done = subprocess.run(
        [sys.executable, "-c", "import os; print(os.getpid())"],
        capture_output=True,
        text=True,
        check=True,
    )
    import socket

    orphan = f"{TMP_PREFIX}{socket.gethostname()}-{int(done.stdout)}-0000eeee"
    partition_cache._create_subtable(msname)
    os.rename(os.path.join(msname, SUBTABLE_NAME), os.path.join(msname, orphan))
    assert status_of(msname) == "stored"
    assert tmp_tables(msname) == []
    # a sub-table without the keyword: linked by the next store
    with tables.table(msname, readonly=False, ack=False) as main_tb:
        main_tb.removekeyword(SUBTABLE_NAME)
    assert partition_cache.link_state(msname) == "unlinked"
    assert status_of(msname, mode="read") == "memory:mode-read"
    assert status_of(msname) in ("stored", "revalidated", "hit-race")
    assert partition_cache.link_state(msname) == "linked"
    assert status_of(msname) == "hit"


# --- read-only MSs ---------------------------------------------------------------------


def _read_only(path: str, recursive: bool) -> None:
    targets = [path]
    if recursive:
        for dirpath, dirnames, filenames in os.walk(path):
            targets += [os.path.join(dirpath, n) for n in dirnames + filenames]
    for target in targets:
        mode = os.lstat(target).st_mode
        os.chmod(target, stat.S_IMODE(mode) & ~0o222)


def _writable_again(path: str) -> None:
    for dirpath, dirnames, filenames in os.walk(path):
        for name in [dirpath] + [
            os.path.join(dirpath, n) for n in dirnames + filenames
        ]:
            os.chmod(name, stat.S_IMODE(os.lstat(name).st_mode) | 0o200)


READ_ONLY_VARIANTS = {
    # variant: (path made read-only, recursively, reason)
    "whole MS": ("", True, "MS directory not writable"),
    "MS directory": ("", False, "MS directory not writable"),
    "MAIN table.dat": ("table.dat", False, "MAIN table not writable"),
    "MAIN table.lock": ("table.lock", False, "MAIN lock file not writable"),
    "HISTORY": ("HISTORY", True, "HISTORY not writable"),
}


@pytest.mark.skipif(os.geteuid() == 0, reason="root writes read-only files")
@pytest.mark.parametrize("variant", list(READ_ONLY_VARIANTS))
def test_read_only_ms(variant, ms_copy, oracle):
    """An MS that cannot be written: three opens give the converter's tree
    with one PartitionCacheWarning, and change no file (table.lock
    included): no temporary table, keyword or sub-table."""
    msname = same_name_copy(ms_copy, "dense")
    target, recursive, reason = READ_ONLY_VARIANTS[variant]
    before = file_digests(msname)
    _read_only(os.path.join(msname, target), recursive)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for _ in range(3):
                partition_cache.clear_partition_memo()
                check_tree(msname, oracle("dense"))
        notices = [w for w in caught if w.category is PartitionCacheWarning]
        assert [f"({reason})" in str(w.message) for w in notices] == [True]
        assert file_digests(msname) == before
        assert tmp_tables(msname) == [] and SUBTABLE_NAME not in os.listdir(msname)
        assert SUBTABLE_NAME not in _keywords(msname)
    finally:
        _writable_again(msname)


@pytest.mark.skipif(os.geteuid() == 0, reason="root writes read-only files")
def test_read_only_ms_with_a_stored_cache(ms_copy, oracle):
    """A read-only copy of an MS that holds a valid cache: a hit, no warning,
    no file changed."""
    msname = same_name_copy(ms_copy, "dense")
    assert status_of(msname) == "stored"
    before = file_digests(msname)
    _read_only(msname, True)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", PartitionCacheWarning)
            assert status_of(msname) == "hit"
            check_tree(msname, oracle("dense"))
        assert file_digests(msname) == before
    finally:
        _writable_again(msname)


@pytest.mark.skipif(not os.path.isdir("/proc/self/fd"), reason="needs /proc")
def test_no_ms_file_left_open_by_the_cache(ms_copy):
    """No file of the MS is open after opens that store, hit, revalidate,
    or fail to store."""
    msname = ms_copy("rich")
    root = os.path.realpath(msname)

    def open_files():
        found = []
        for fd in os.listdir("/proc/self/fd"):
            with contextlib.suppress(OSError):
                target = os.readlink(f"/proc/self/fd/{fd}")
                if target.startswith(root + os.sep):
                    found.append(target)
        return found

    for scheme in ([], ["FIELD_ID"]):
        xr.open_datatree(
            msname, engine=ENGINE, partition_cache="auto", partition_scheme=scheme
        )
        assert open_files() == []
    partition_cache.clear_partition_memo()
    xr.open_datatree(msname, engine=ENGINE, partition_cache="auto")
    assert open_files() == []
    with tables.table(os.path.join(msname, "HISTORY"), readonly=False, ack=False) as h:
        h.addrows(1)
    assert status_of(msname) == "revalidated"
    assert open_files() == []
    with held(msname):
        shutil.rmtree(os.path.join(msname, SUBTABLE_NAME))
        with pytest.warns(PartitionCacheWarning):
            assert status_of(msname, ["SCAN_NUMBER"]) == "memory:locked"
    assert open_files() == []
