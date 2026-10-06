"""
casacore's ``table.lock`` files, and the fingerprint of the tables of an MSv2
that its partitions are computed from (the staleness check of the partition
cache of the MSv2 xarray backend, ``partition_cache.py``).

After the lock request ids (its first 260 bytes), the ``table.lock`` file of a
casacore table holds the synchronization data of the table
(``casa/IO/LockFile.cc``, ``tables/Tables/TableSyncData.cc``): the number of
rows and columns, and a change counter of the table and of every data manager
(in the order of ``getdminfo``). Every process that writes the table updates
it when it flushes or releases a lock, in every lock mode. Reading it here
needs no table open, and gives

- the number of rows of a table on disk, also when this process holds the
  table open with an older number of rows (casacore shares one table object
  per path in a process, and a reader that takes no locks never re-reads
  it);
- the change counters of the data managers.

The stat of ``table.lock`` itself is never used: lock requests rewrite its
first bytes.
"""

import dataclasses
import os
import struct
from collections.abc import Mapping

import numpy as np

from xradio._utils._casacore.tables import casatools_serialized
from xradio.measurement_set._utils._msv2._tables.table_query import open_table_ro

LOCK_FILE = "table.lock"
# Version of the fingerprint layout (ms_fingerprint)
FINGERPRINT_VERSION = 1
# The lock request ids at the start of table.lock: a count and 32 (pid, host)
# pairs of int32 (LockFile.cc, SIZEREQID)
_REQUEST_ID_BYTES = (1 + 2 * 32) * 4
# AipsIO's magic value, at the start of an outermost object
_AIPSIO_MAGIC = 0xBEBEBEBE
# table.lock files are read up to this size (the sync data of a table with
# 10,000 data managers is about 40 kB)
_MAX_LOCK_FILE_BYTES = 1 << 20


@dataclasses.dataclass(frozen=True)
class TableLockInfo:
    """
    The synchronization data of a table, from its ``table.lock`` file
    (read_table_lock).

    Attributes
    ----------
    lock_ok : bool
        Whether the file holds complete synchronization data (the other
        attributes are None otherwise).
    version : int | None
        Version of the data (1: 32-bit number of rows, 2: 64-bit).
    nrrow : int | None
        Number of rows of the table.
    nrcolumn : int | None
        Number of columns.
    modify_counter, table_change_counter : int | None
        casacore's modify counter and table change counter.
    dm_change_counters : tuple[int, ...] | None
        The change counter of every data manager, in the order of
        ``getdminfo`` (``*1``, ``*2``, ...).
    error : str | None
        Why the data could not be read (lock_ok False).
    """

    lock_ok: bool
    version: int | None = None
    nrrow: int | None = None
    nrcolumn: int | None = None
    modify_counter: int | None = None
    table_change_counter: int | None = None
    dm_change_counters: tuple[int, ...] | None = None
    error: str | None = None


class _BigEndianReader:
    """Reads AipsIO's canonical (big-endian) values from a buffer."""

    def __init__(self, buffer: bytes):
        self.buffer = buffer
        self.pos = 0

    def _unpack(self, fmt: str) -> int:
        (value,) = struct.unpack_from(fmt, self.buffer, self.pos)
        self.pos += struct.calcsize(fmt)
        return value

    def uint32(self) -> int:
        return self._unpack(">I")

    def int32(self) -> int:
        return self._unpack(">i")

    def uint64(self) -> int:
        return self._unpack(">Q")

    def string(self) -> str:
        size = self.uint32()
        if self.pos + size > len(self.buffer):
            raise ValueError("string beyond the end of the data")
        value = self.buffer[self.pos : self.pos + size].decode("ascii")
        self.pos += size
        return value

    def object_start(self, object_type: str, outermost: bool) -> int:
        """Start of an AipsIO object: magic value (outermost object only),
        length, type and version; returns the version."""
        if outermost and self.uint32() != _AIPSIO_MAGIC:
            raise ValueError("no AipsIO magic value")
        self.uint32()  # length of the object
        found = self.string()
        if found != object_type:
            raise ValueError(f"AipsIO object {found!r}, expected {object_type!r}")
        return self.uint32()


def parse_sync_data(content: bytes) -> TableLockInfo:
    """
    The synchronization data in the content of a ``table.lock`` file.

    Parameters
    ----------
    content : bytes
        The file content.

    Returns
    -------
    TableLockInfo
        With lock_ok False (and the reason) if the content is too short,
        holds no synchronization data, or cannot be parsed.
    """
    if len(content) < _REQUEST_ID_BYTES + 4:
        return TableLockInfo(False, error=f"short lock file ({len(content)} bytes)")
    (size,) = struct.unpack_from(">I", content, _REQUEST_ID_BYTES)
    start = _REQUEST_ID_BYTES + 4
    if size == 0:
        return TableLockInfo(False, error="no synchronization data")
    if start + size > len(content):
        return TableLockInfo(False, error="truncated synchronization data")
    reader = _BigEndianReader(content[start : start + size])
    try:
        version = reader.object_start("sync", outermost=True)
        if version not in (1, 2):
            raise ValueError(f"synchronization data version {version}")
        nrrow = reader.uint32() if version == 1 else reader.uint64()
        nrcolumn = reader.int32()
        modify_counter = reader.uint32()
        if nrcolumn < 0:
            # (written for tables filled by an external filler: no counters)
            raise ValueError("no change counters")
        table_change_counter = reader.uint32()
        # Block<uInt> (a nested object: no magic value), count, values
        reader.object_start("Block", outermost=False)
        count = reader.uint32()
        if reader.pos + 4 * count > len(reader.buffer):
            raise ValueError("change counters beyond the end of the data")
        counters = tuple(reader.uint32() for _ in range(count))
    except (ValueError, struct.error, UnicodeDecodeError) as exc:
        return TableLockInfo(False, error=f"{type(exc).__name__}: {exc}")
    return TableLockInfo(
        True,
        version=version,
        nrrow=nrrow,
        nrcolumn=nrcolumn,
        modify_counter=modify_counter,
        table_change_counter=table_change_counter,
        dm_change_counters=counters,
    )


def read_table_lock(table_path: str) -> TableLockInfo | None:
    """
    The synchronization data of a table from its ``table.lock`` file.

    Parameters
    ----------
    table_path : str
        Path of the table (directory).

    Returns
    -------
    TableLockInfo | None
        None if the table has no lock file (or it cannot be read); lock_ok
        False if it holds no readable synchronization data.
    """
    try:
        with open(os.path.join(table_path, LOCK_FILE), "rb") as lock_file:
            content = lock_file.read(_MAX_LOCK_FILE_BYTES)
    except OSError:
        return None
    return parse_sync_data(content)


def write_locked_here(table) -> bool:
    """
    Whether this process holds the write lock of an opened table: casacore
    shares one table object, with its locks, per table in a process, so any
    handle (a read-only one too) tells. The process may then have changes
    of the table that are not flushed yet (casacore flushes a table before
    it releases its write lock). True if that cannot be told.
    """
    try:
        return bool(table.haslock(True))
    except Exception:
        return True


def resync_unless_write_locked(table) -> bool:
    """
    Re-synchronize an opened table with its files (``resync``: e.g. the rows
    that other processes added), unless this process holds its write lock:
    a re-synchronization drops the changes that the process has not flushed
    yet (e.g. rows that a user's writable handle added), so the process's
    own view is kept then.

    Returns
    -------
    bool
        Whether the table was re-synchronized.
    """
    if write_locked_here(table):
        return False
    table.resync()
    return True


def table_nrows(table_path: str, lock: TableLockInfo | None = None) -> int:
    """
    The number of rows of a table on disk: from its lock file, or else from
    the table, opened without locks and re-synchronized with its files (an
    open in a process that already has the table open shares its table
    object, which may have an older number of rows; not re-synchronized,
    and so the number of rows of this process, if the process holds its
    write lock: see resync_unless_write_locked).

    Parameters
    ----------
    table_path : str
        Path of the table.
    lock : TableLockInfo | None, optional
        read_table_lock of the table, by default read here.

    Returns
    -------
    int
        Number of rows.
    """
    if lock is None:
        lock = read_table_lock(table_path)
    if lock is not None and lock.lock_ok:
        return int(lock.nrrow)
    with casatools_serialized(), open_table_ro(table_path) as table:
        resync_unless_write_locked(table)
        return int(table.nrows())


def data_managers(table) -> list[list]:
    """
    The data managers of an opened table, in their order (the order of the
    change counters of its lock file): ``[TYPE, NAME, SEQNR, sorted column
    names]`` each (SEQNR None if not given).
    """
    dminfo = table.getdminfo()
    found = []
    for key in sorted(dminfo, key=lambda name: int(str(name).lstrip("*"))):
        info = dminfo[key]
        seqnr = info.get("SEQNR")
        columns = np.asarray(info.get("COLUMNS", []), dtype=object).ravel()
        found.append(
            [
                str(info["TYPE"]),
                str(info["NAME"]),
                None if seqnr is None else int(seqnr),
                sorted(str(column) for column in columns),
            ]
        )
    return found


def _file_stats(table_path: str, names: list[str]) -> dict[str, list[int]]:
    stats = {}
    for name in names:
        try:
            stat = os.stat(os.path.join(table_path, name))
        except FileNotFoundError:
            continue
        stats[name] = [int(stat.st_size), int(stat.st_mtime_ns)]
    return stats


def data_manager_files(
    table_path: str, seqnr: int | None, file_names: list[str] | None = None
) -> dict[str, list[int]]:
    """
    ``[size, mtime_ns]`` of the files of a data manager: ``table.f<SEQNR>``,
    ``table.f<SEQNR>i`` and ``table.f<SEQNR>_*`` (e.g. ``table.f3_TSM0``).

    Parameters
    ----------
    table_path : str
        Path of the table.
    seqnr : int | None
        SEQNR of the data manager (None: no files).
    file_names : list[str] | None, optional
        The names in the table directory, by default listed here.

    Returns
    -------
    dict[str, list[int]]
        By file name.
    """
    if seqnr is None:
        return {}
    if file_names is None:
        file_names = os.listdir(table_path)
    prefix = f"table.f{seqnr}"
    names = sorted(
        name
        for name in file_names
        if name in (prefix, prefix + "i") or name.startswith(prefix + "_")
    )
    return _file_stats(table_path, names)


# The types of tables whose rows live in other tables (table.dat)
_REFERENCING_TABLE_TYPES = ("RefTable", "ConcatTable")


def table_type(table_path: str) -> str | None:
    """
    casacore's type of a table, from the start of its ``table.dat`` (the
    AipsIO object "Table": version, number of rows, format, then the type):
    "PlainTable", "RefTable" (a reference table: rows of another table) or
    "ConcatTable" (rows of several tables); None if it cannot be read. Needs
    no table open (casatools has no ``partnames``).
    """
    try:
        with open(os.path.join(table_path, "table.dat"), "rb") as dat_file:
            content = dat_file.read(1024)
        reader = _BigEndianReader(content)
        version = reader.object_start("Table", outermost=True)
        if version > 2:
            reader.uint64()  # number of rows
        else:
            reader.uint32()
        reader.uint32()  # format (endianness)
        return reader.string()
    except (OSError, ValueError, struct.error, UnicodeDecodeError):
        return None


def table_fingerprint(
    table_path: str, key_columns: list[str] | tuple[str, ...] | None = None
) -> dict | None:
    """
    The fingerprint of one table (see ms_fingerprint).

    Parameters
    ----------
    table_path : str
        Path of the table.
    key_columns : list[str] | tuple[str, ...] | None, optional
        With key columns: the change counters and files of the data managers
        that hold them ("key_dms"). Without: those of every data manager
        ("counters", and "files": every ``table.f*`` file).

    Returns
    -------
    dict | None
        None if there is no table at ``table_path``.
    """
    return followed_table_fingerprint(table_path, key_columns)[0]


def followed_table_fingerprint(
    table_path: str,
    key_columns: list[str] | tuple[str, ...] | None = None,
    followed_columns: list[str] | tuple[str, ...] | None = None,
) -> tuple[dict | None, str | None]:
    """
    The fingerprint of one table (``table_fingerprint``) and why it cannot
    tell the writes of the table, if it cannot (``unfollowed``): the
    fingerprint describes the table's own lock file and data-manager files
    (only their size and mtime without a readable lock file), which do not
    change when

    - the rows live in other tables: a reference or concatenated table
      (``table_type``; such tables have no lock file and no data-manager
      files of their own);
    - a column is held by a data manager without files of its own (e.g.
      ``ForwardColumnEngine``, whose columns are read from another table, or
      a virtual column engine): only for ``followed_columns`` (default: the
      key columns).

    Parameters
    ----------
    table_path : str
        Path of the table.
    key_columns : list[str] | tuple[str, ...] | None, optional
        As for ``table_fingerprint``.
    followed_columns : list[str] | tuple[str, ...] | None, optional
        The columns whose data managers must have files of their own (default
        ``key_columns``; none if both are None).

    Returns
    -------
    tuple[dict | None, str | None]
        (fingerprint, None if it follows the writes of the table, else the
        reason as a predicate of the table: "is a reference or concatenated
        table" or "uses <data manager type>"). (None, None) if there is no
        table.
    """
    if not os.path.isfile(os.path.join(table_path, "table.dat")):
        return None, None
    lock = read_table_lock(table_path)
    lock_ok = lock is not None and lock.lock_ok
    with casatools_serialized(), open_table_ro(table_path) as table:
        if not lock_ok:
            # (the rows of this process if it holds the table's write lock)
            resync_unless_write_locked(table)
        nrows = int(lock.nrrow) if lock_ok else int(table.nrows())
        dms = data_managers(table)
    fingerprint = _make_fingerprint(
        table_path, key_columns, lock if lock_ok else None, nrows, dms
    )
    unfollowed = None
    if table_type(table_path) in _REFERENCING_TABLE_TYPES:
        unfollowed = "is a reference or concatenated table"
    else:
        followed = key_columns if followed_columns is None else followed_columns
        file_names = os.listdir(table_path)
        for dm_type, _, seqnr, columns in dms:
            if set(columns) & set(followed or ()) and not data_manager_files(
                table_path, seqnr, file_names
            ):
                unfollowed = f"uses {dm_type}"
                break
    return fingerprint, unfollowed


def _make_fingerprint(
    table_path: str,
    key_columns: list[str] | tuple[str, ...] | None,
    lock: TableLockInfo | None,
    nrows: int,
    dms: list[list],
) -> dict:
    """The fingerprint of table_fingerprint from the rows and data managers of
    a table, with its lock file data (None: not readable)."""
    lock_ok = lock is not None
    counters = list(lock.dm_change_counters) if lock_ok else None
    fingerprint = {
        "nrows": nrows,
        # (from the lock file: also right when this process holds the table
        # open with an older column set)
        "ncolumns": int(lock.nrcolumn) if lock_ok else None,
        "lock_ok": lock_ok,
        "dms": dms,
    }
    file_names = sorted(os.listdir(table_path))
    if key_columns is None:
        fingerprint["counters"] = counters
        fingerprint["files"] = _file_stats(
            table_path, [name for name in file_names if name.startswith("table.f")]
        )
        return fingerprint
    key_dms = {}
    for index, (_, _, seqnr, columns) in enumerate(dms):
        if not set(columns) & set(key_columns):
            continue
        key_dms[str(seqnr)] = {
            "counter": (
                counters[index]
                if counters is not None and index < len(counters)
                else None
            ),
            "files": data_manager_files(table_path, seqnr, file_names),
        }
    fingerprint["key_dms"] = key_dms
    return fingerprint


def ms_fingerprint(
    ms_path: str,
    main_key_columns: list[str] | tuple[str, ...],
    subtables: tuple[str, ...] = ("FIELD", "STATE", "SOURCE"),
) -> dict:
    """
    A fingerprint of the tables of an MS that its partitions are computed
    from: equal fingerprints mean that the inputs of the partitions did not
    change (no change of these tables was ever missed in 44 measured
    modifications; some changes that keep the partitions also change it,
    e.g. writes into a storage manager shared with a key column).

    - MAIN: the number of rows (lock file) and of columns, every data manager
      (type, name, SEQNR, columns: any change of the list), and the change
      counter (lock file) and files (size, mtime_ns) of every data manager
      that holds one of ``main_key_columns``;
    - every table of ``subtables`` (None if absent): its number of rows and
      columns, data managers, all their change counters and files.

    Not included: the table change counter of MAIN and its ``table.dat`` /
    ``table.info`` (a new MAIN keyword changes them), the stat of
    ``table.lock`` files, the HISTORY table.

    Parameters
    ----------
    ms_path : str
        Path of the MS.
    main_key_columns : list[str] | tuple[str, ...]
        The MAIN columns the partitions are computed from.
    subtables : tuple[str, ...], optional
        The sub-tables the partitions are computed from.

    Returns
    -------
    dict
        JSON-compatible (``json.loads(json.dumps(f)) == f``).

    Raises
    ------
    Exception
        Whatever opening and describing the tables raises (e.g. a MAIN table
        that does not exist).
    """
    return followed_ms_fingerprint(ms_path, main_key_columns, subtables)[0]


def followed_ms_fingerprint(
    ms_path: str,
    main_key_columns: list[str] | tuple[str, ...],
    subtables: tuple[str, ...] = ("FIELD", "STATE", "SOURCE"),
    subtable_columns: Mapping[str, tuple[str, ...]] | None = None,
) -> tuple[dict, str | None]:
    """
    ``ms_fingerprint`` and why it cannot tell the writes of the tables it
    describes, if it cannot (``followed_table_fingerprint`` of MAIN, for its
    key columns, and of every sub-table, for its ``subtable_columns``).

    Returns
    -------
    tuple[dict, str | None]
        (fingerprint, None, or the reason, e.g. "MAIN is a reference or
        concatenated table", "MAIN uses ForwardColumnEngine").
    """
    main, unfollowed = followed_table_fingerprint(ms_path, list(main_key_columns))
    if main is None:
        raise FileNotFoundError(f"No MAIN table in {ms_path}")
    reason = None if unfollowed is None else f"MAIN {unfollowed}"
    fingerprint = {"v": FINGERPRINT_VERSION, "main": main}
    for name in subtables:
        columns = (subtable_columns or {}).get(name, ())
        fingerprint[name], unfollowed = followed_table_fingerprint(
            os.path.join(ms_path, name), None, columns
        )
        if reason is None and unfollowed is not None:
            reason = f"{name} {unfollowed}"
    return fingerprint, reason


def history_nrows(ms_path: str) -> int | None:
    """
    The number of rows of the HISTORY table of an MS (table_nrows), None if
    it has none.
    """
    path = os.path.join(ms_path, "HISTORY")
    if not os.path.isfile(os.path.join(path, "table.dat")):
        return None
    return table_nrows(path)
