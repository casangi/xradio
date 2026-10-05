"""Tests of the table.lock parser and the MS fingerprint (_tables/table_lock_file.py)."""

import json
import os
import struct
import subprocess
import sys

import pytest
from casacore import tables

from xradio.measurement_set._utils._msv2._tables import table_lock_file
from xradio.measurement_set._utils._msv2._tables.table_lock_file import (
    TableLockInfo,
    data_manager_files,
    data_managers,
    history_nrows,
    ms_fingerprint,
    parse_sync_data,
    read_table_lock,
    table_fingerprint,
    table_nrows,
)

KEY_COLUMNS = (
    "DATA_DESC_ID",
    "FIELD_ID",
    "SCAN_NUMBER",
    "STATE_ID",
    "OBSERVATION_ID",
    "ANTENNA1",
    "ANTENNA2",
)
VARIANTS = ("dense", "rich", "single_dish", "no_weight")


def sync_block(
    version=1, nrrow=10, nrcolumn=3, modify=4, table_change=2, counters=(1, 5)
) -> bytes:
    """A table.lock content with synchronization data, as casacore writes it
    (TableSyncData::write, AipsIO canonical format)."""

    def u32(value):
        return struct.pack(">I", value)

    def string(value):
        return u32(len(value)) + value.encode()

    block_body = string("Block") + u32(1) + u32(len(counters))
    block_body += b"".join(u32(c) for c in counters)
    block = u32(4 + len(block_body)) + block_body
    body = string("sync") + u32(version)
    body += u32(nrrow) if version == 1 else struct.pack(">Q", nrrow)
    body += struct.pack(">i", nrcolumn) + u32(modify)
    if nrcolumn >= 0:
        body += u32(table_change) + block
    info = u32(0xBEBEBEBE) + u32(4 + len(body)) + body
    request_ids = struct.pack(">65i", *([1, 1234, 5678] + [0] * 62))
    return request_ids + u32(len(info)) + info


# --- the parser ---------------------------------------------------------------


def test_parse_version_1_and_2():
    info = parse_sync_data(sync_block())
    assert info == TableLockInfo(
        True,
        version=1,
        nrrow=10,
        nrcolumn=3,
        modify_counter=4,
        table_change_counter=2,
        dm_change_counters=(1, 5),
    )
    # version 2: a 64-bit number of rows (written above 2**32 - 1 rows)
    info = parse_sync_data(sync_block(version=2, nrrow=2**33 + 7, counters=(3,)))
    assert info.lock_ok and info.version == 2 and info.nrrow == 2**33 + 7
    assert info.dm_change_counters == (3,)
    assert parse_sync_data(sync_block(counters=())).dm_change_counters == ()


@pytest.mark.parametrize(
    "content, error",
    [
        (b"", "short"),
        (b"\0" * 100, "short"),
        (b"\0" * 264, "no synchronization data"),
        (sync_block()[:300], "truncated"),
        (sync_block().replace(b"sync", b"synd"), "AipsIO object"),
        (sync_block().replace(b"Block", b"Brick"), "AipsIO object"),
        (sync_block(version=3), "version 3"),
        (sync_block(nrcolumn=-1), "no change counters"),
        (
            sync_block()[:260] + struct.pack(">I", 64) + b"\xbe" * 4 + b"\xff" * 60,
            "beyond the end",
        ),
    ],
    ids=[
        "empty",
        "100 bytes",
        "no data",
        "truncated",
        "not sync",
        "not Block",
        "version 3",
        "no counters",
        "garbage",
    ],
)
def test_parse_short_or_garbage(content, error):
    info = parse_sync_data(content)
    assert not info.lock_ok and info.nrrow is None
    assert error in info.error


def test_parse_garbage_with_a_valid_magic():
    content = bytearray(sync_block())
    # the magic value kept, the object length and type garbled
    content[268:280] = b"\xff" * 12
    info = parse_sync_data(bytes(content))
    assert not info.lock_ok and info.error


def test_read_missing_lock_file(tmp_path):
    assert read_table_lock(str(tmp_path)) is None
    (tmp_path / "table.lock").write_bytes(sync_block(nrrow=12))
    assert read_table_lock(str(tmp_path)).nrrow == 12


@pytest.mark.parametrize("variant", VARIANTS)
def test_lock_files_agree_with_casacore(variant, backend_ms):
    """The number of rows and columns and the data manager counters of every
    table of a generated MS (MAIN and sub-tables) agree with python-casacore."""
    msname = backend_ms(variant)
    paths = [msname] + [
        os.path.join(msname, name)
        for name in sorted(os.listdir(msname))
        if os.path.isfile(os.path.join(msname, name, "table.dat"))
    ]
    assert len(paths) > 10
    for path in paths:
        info = read_table_lock(path)
        with tables.table(path, ack=False) as table:
            assert info.lock_ok, path
            assert info.nrrow == table.nrows() == table_nrows(path), path
            assert info.nrcolumn == len(table.colnames()), path
            assert len(info.dm_change_counters) == len(table.getdminfo()), path
            assert len(data_managers(table)) == len(table.getdminfo())


@pytest.mark.parametrize("layout", ["shared", "per_column"])
def test_counter_and_files_of_the_written_data_manager(layout, ms_copy):
    """A write into a column changes the counter (at the data manager's index
    in getdminfo order) and the files (table.f<SEQNR>*) of the data manager
    holding it, and of no other data manager."""
    msname = ms_copy("dense", layout)
    for column in ("FIELD_ID", "UVW", "ANTENNA2"):
        with tables.table(msname, ack=False) as table:
            dms = data_managers(table)
        index = next(i for i, dm in enumerate(dms) if column in dm[3])
        before = read_table_lock(msname).dm_change_counters
        files_before = [data_manager_files(msname, dm[2]) for dm in dms]
        assert files_before[index]
        with tables.table(msname, readonly=False, ack=False) as table:
            # (other values: an IncrementalStMan does not write equal ones)
            table.putcol(column, table.getcol(column) + 1)
        after = read_table_lock(msname).dm_change_counters
        files_after = [data_manager_files(msname, dm[2]) for dm in dms]
        changed = [i for i in range(len(dms)) if before[i] != after[i]]
        assert changed == [index], column
        changed_files = [
            i for i in range(len(dms)) if files_before[i] != files_after[i]
        ]
        assert changed_files == [index], column


def test_data_manager_files_names(tmp_path):
    for name in ("table.f1", "table.f1i", "table.f1_TSM0", "table.f10", "table.f11i"):
        (tmp_path / name).write_bytes(b"x" * 3)
    assert sorted(data_manager_files(str(tmp_path), 1)) == [
        "table.f1",
        "table.f1_TSM0",
        "table.f1i",
    ]
    assert list(data_manager_files(str(tmp_path), 10)) == ["table.f10"]
    assert data_manager_files(str(tmp_path), None) == {}
    assert data_manager_files(str(tmp_path), 1)["table.f1"][0] == 3


# --- rows of a table held open in this process ---------------------------------


def test_rows_of_a_table_held_open_with_rows_added_by_another_process(ms_copy):
    """casacore shares one table object per path in a process: with MAIN held
    open here, a new open does not see rows another process added (the
    'stale handle' trap). The lock file has them, and so have table_nrows and
    the fingerprint."""
    msname = ms_copy("dense")
    before = ms_fingerprint(msname, KEY_COLUMNS)
    held = tables.table(
        msname, readonly=True, lockoptions={"option": "usernoread"}, ack=False
    )
    try:
        nrows = held.nrows()
        script = (
            "from casacore import tables\n"
            f"with tables.table({msname!r}, readonly=False, ack=False) as t:\n"
            "    t.copyrows(t, startrowin=0, nrow=7)\n"
        )
        subprocess.run([sys.executable, "-c", script], check=True)
        with tables.table(
            msname, readonly=True, lockoptions={"option": "usernoread"}, ack=False
        ) as table:
            assert table.nrows() == nrows  # stale
        assert read_table_lock(msname).nrrow == nrows + 7
        assert table_nrows(msname) == nrows + 7
        after = ms_fingerprint(msname, KEY_COLUMNS)
        assert after["main"]["nrows"] == nrows + 7 and after != before
    finally:
        held.close()


def test_table_nrows_without_lock_file_data(ms_copy):
    """An empty lock file (no synchronization data): the rows come from the
    table, re-synchronized."""
    msname = ms_copy("dense")
    field = os.path.join(msname, "FIELD")
    with open(os.path.join(field, "table.lock"), "wb"):
        pass
    assert not read_table_lock(field).lock_ok
    with tables.table(field, ack=False) as table:
        expected = table.nrows()
    assert table_nrows(field) == expected
    fingerprint = table_fingerprint(field)
    assert fingerprint["nrows"] == expected and not fingerprint["lock_ok"]
    assert fingerprint["counters"] is None and fingerprint["ncolumns"] is None


def test_rows_not_flushed_by_this_process_are_kept(ms_copy, monkeypatch):
    """A table whose write lock this process holds, with rows added and not
    flushed (a user's writable handle): table_nrows and the fingerprint
    without lock file data, which re-synchronize the table otherwise, give
    the rows of the process and drop none of them (a resync rolls them
    back). (Closing a handle releases the lock and flushes them: every case
    adds rows with a new handle.)"""
    msname = ms_copy("dense")

    def disk_rows():
        with tables.table(msname, ack=False) as table:
            return table.nrows()

    nrows = disk_rows()
    writer = tables.table(msname, readonly=False, ack=False)
    try:
        writer.addrows(3)
        assert read_table_lock(msname).nrrow == nrows  # (not flushed)
        assert table_nrows(msname, TableLockInfo(False)) == nrows + 3
        assert writer.nrows() == nrows + 3
    finally:
        writer.close()
    assert disk_rows() == nrows + 3
    monkeypatch.setattr(
        table_lock_file, "read_table_lock", lambda path: TableLockInfo(False)
    )
    writer = tables.table(msname, readonly=False, ack=False)
    try:
        writer.addrows(2)
        assert table_fingerprint(msname)["nrows"] == nrows + 5
        assert writer.nrows() == nrows + 5
    finally:
        writer.close()
    assert disk_rows() == nrows + 5
    writer = tables.table(msname, readonly=False, ack=False)
    try:
        writer.addrows(1)
        assert table_lock_file.write_locked_here(writer)
        assert not table_lock_file.resync_unless_write_locked(writer)
        assert writer.nrows() == nrows + 6
    finally:
        writer.close()
    with tables.table(msname, ack=False) as reader:
        assert reader.nrows() == nrows + 6
        assert not table_lock_file.write_locked_here(reader)
        assert table_lock_file.resync_unless_write_locked(reader)


# --- the fingerprint -----------------------------------------------------------


def test_fingerprint_layout(ms_copy):
    msname = ms_copy("rich", "per_column")
    fingerprint = ms_fingerprint(msname, KEY_COLUMNS)
    assert json.loads(json.dumps(fingerprint)) == fingerprint
    assert fingerprint["v"] == 1
    main = fingerprint["main"]
    assert main["lock_ok"] and main["nrows"] == 1200
    with tables.table(msname, ack=False) as table:
        dms = table.getdminfo()
        assert main["ncolumns"] == len(table.colnames())
    assert len(main["dms"]) == len(dms)
    # one data manager per key column: 7 key data managers, each with its
    # counter and files
    assert len(main["key_dms"]) == 7
    for key_dm in main["key_dms"].values():
        assert isinstance(key_dm["counter"], int) and key_dm["files"]
    for name in ("FIELD", "STATE", "SOURCE"):
        assert set(fingerprint[name]) == {
            "nrows",
            "ncolumns",
            "lock_ok",
            "dms",
            "counters",
            "files",
        }
        assert fingerprint[name]["files"]
    shared = ms_fingerprint(ms_copy("rich"), KEY_COLUMNS)
    assert len(shared["main"]["key_dms"]) == 1  # one StandardStMan
    assert shared["main"] != main


def test_fingerprint_changes(ms_copy):
    """What changes the fingerprint and what does not."""
    msname = ms_copy("rich", "per_column")

    def fingerprint():
        return ms_fingerprint(msname, KEY_COLUMNS)

    def changed(before):
        after = fingerprint()
        return [key for key in sorted(after) if after[key] != before[key]]

    before = fingerprint()
    # a MAIN keyword, a HISTORY row, an open for update: no change
    with tables.table(msname, readonly=False, ack=False) as table:
        table.putkeyword("XRADIO_TEST", "value")
    with tables.table(os.path.join(msname, "HISTORY"), readonly=False, ack=False) as h:
        h.addrows(1)
    with tables.table(msname, readonly=False, ack=False):
        pass
    assert changed(before) == []
    # a data column (not a key column): no change
    with tables.table(msname, readonly=False, ack=False) as table:
        table.putcol("DATA", table.getcol("DATA") * 2)
    assert changed(before) == []
    # key columns
    with tables.table(msname, readonly=False, ack=False) as table:
        field = table.getcol("FIELD_ID")
        field[:10] = 1 - field[:10]
        table.putcol("FIELD_ID", field)
    assert changed(before) == ["main"]
    before = fingerprint()
    with tables.table(msname, readonly=False, ack=False) as table:
        table.putcol("ANTENNA2", table.getcol("ANTENNA2"))
    assert changed(before) == ["main"]
    # sub-tables (also columns that do not partition: whole sub-tables)
    before = fingerprint()
    with tables.table(os.path.join(msname, "FIELD"), readonly=False, ack=False) as t:
        t.putcell("NAME", 0, "renamed")
    assert changed(before) == ["FIELD"]
    before = fingerprint()
    with tables.table(os.path.join(msname, "STATE"), readonly=False, ack=False) as t:
        t.putcell("OBS_MODE", 0, "OBSERVE_TARGET#OFF_SOURCE")
    assert changed(before) == ["STATE"]
    # rows and columns
    before = fingerprint()
    with tables.table(msname, readonly=False, ack=False) as table:
        table.addcols(tables.maketabdesc([tables.makescacoldesc("XR_EXTRA", 0.0)]))
    assert changed(before) == ["main"]
    before = fingerprint()
    with tables.table(msname, readonly=False, ack=False) as table:
        table.copyrows(table, startrowin=0, nrow=3)
    assert changed(before) == ["main"]
    assert fingerprint()["main"]["nrows"] == 1203


def test_fingerprint_of_absent_sub_tables(ms_copy):
    msname = ms_copy("dense")
    os.rename(os.path.join(msname, "SOURCE"), os.path.join(msname, "SOURCE_MOVED"))
    fingerprint = ms_fingerprint(msname, KEY_COLUMNS)
    assert fingerprint["SOURCE"] is None and fingerprint["FIELD"] is not None
    with pytest.raises(FileNotFoundError):
        ms_fingerprint(os.path.join(msname, "NOT_THERE"), KEY_COLUMNS)


def test_history_nrows(ms_copy):
    msname = ms_copy("dense")
    with tables.table(os.path.join(msname, "HISTORY"), ack=False) as history:
        nrows = history.nrows()
    assert history_nrows(msname) == nrows
    with tables.table(os.path.join(msname, "HISTORY"), readonly=False, ack=False) as h:
        h.addrows(2)
    assert history_nrows(msname) == nrows + 2
    os.rename(os.path.join(msname, "HISTORY"), os.path.join(msname, "HISTORY_MOVED"))
    assert history_nrows(msname) is None


@pytest.mark.skipif(not os.path.isdir("/proc/self/fd"), reason="needs /proc")
def test_fingerprint_opens_no_table_for_update(ms_copy, monkeypatch):
    """The fingerprint reads: no table opened for update, every table
    closed."""
    msname = ms_copy("dense")
    opened = []
    table_class = table_lock_file.open_table_ro

    def spy(path):
        opened.append(path)
        return table_class(path)

    monkeypatch.setattr(table_lock_file, "open_table_ro", spy)
    ms_fingerprint(msname, KEY_COLUMNS)
    assert sorted(opened) == sorted(
        [msname] + [os.path.join(msname, n) for n in ("FIELD", "STATE", "SOURCE")]
    )
    root = os.path.realpath(msname)
    fds = []
    for fd in os.listdir("/proc/self/fd"):
        try:
            if os.readlink(f"/proc/self/fd/{fd}").startswith(root + os.sep):
                fds.append(fd)
        except OSError:
            pass
    assert fds == []
