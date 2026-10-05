"""
The partitions of an MSv2 for the MSv2 xarray backend (engine
``xradio_msv2``): computed with ``create_partitions_with_main_rows`` on the
first open of an MS, stored inside the MS (the sub-table
``XRADIO_PARTITIONS``, linked from MAIN by the keyword ``XRADIO_PARTITIONS``,
and a HISTORY row per stored content), and reused while the MS is unchanged.
Validated results are also kept in a per-process memo.

Modes (``partition_cache``, default ``$XRADIO_MSV2_PARTITION_CACHE`` or
"auto"):

- "auto": a valid memo entry or stored row is used; otherwise the partitions
  are computed, memoised and stored in the MS when it can be written (else
  kept in memory, with a PartitionCacheWarning or an INFO log once per MS
  and reason);
- "read": as "auto", but nothing is ever written;
- "rebuild": the partitions are computed again, memoised and stored (a
  HISTORY row only if they changed);
- "off": the partitions are computed; the memo and the stored rows are
  neither used nor filled.

A stored row is used only when (the staleness layers)

- L1: the fingerprint of the MS (``table_lock_file.ms_fingerprint``: MAIN
  rows, columns and data managers, the change counters and files of the
  data managers of the key columns; FIELD, STATE and SOURCE) equals the one
  stored with it;
- L2: the HISTORY table has no row at or after the stored anchor (its number
  of rows when the partitions were computed) but rows of xradio's partition
  cache, and the HISTORY row of the stored content is still there;
- L3: the row is intact (its CHECKSUM) and consistent (versions; one
  description per run range; ascending, disjoint runs inside MAIN that cover
  every row unless the scheme has ANTENNA1; single-valued, distinct grouping
  keys).

A stale row is never parsed (L1 and L2 come first). Recomputed partitions
equal to a stale row's only refresh its fingerprint and anchor (no HISTORY
row: a harmless task, e.g. flagdata, costs one recomputation). The engine
also checks every partition it opens from a stored row or the memo against
its MAIN rows (backend_partition.verify_partition_rows), and computes the
partitions again if one does not describe its rows.

Writes (python-casacore only; ``store_partitions``): every lock is tried once
(``nattempts=1``, checked with ``haslock``: an open never waits for a lock),
in the order MAIN (alone), the sub-table, HISTORY; one writer per MS in a
process (a mutex: casacore's locks belong to the process). The sub-table is
created under a temporary name and renamed into place before MAIN links it,
so no failure leaves a keyword without its sub-table. Another thread of the
process that closes a MAIN handle releases the MAIN write lock (python-
casacore's close unlocks the table object the process shares): the MAIN
keyword writes take it again (``_write_main_keywords``).

A memo entry is valid while the fingerprint and the number of HISTORY rows
are those it was made with. The memo hands out copies, holds at most
``MEMO_MAX_ENTRIES`` results and ``MEMO_MAX_BYTES`` of row runs, computes a
key once when several threads open the same MS, and is emptied in a fork
child.
"""

import collections
import contextlib
import copy
import dataclasses
import hashlib
import json
import operator
import os
import shutil
import socket
import threading
import time
import traceback
import uuid
import warnings
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from xradio._utils._casacore.tables import casatools_serialized, uses_casatools
from xradio._utils.logging import xradio_logger
from xradio.measurement_set._utils._msv2._tables.table_lock_file import (
    history_nrows,
    ms_fingerprint,
    resync_unless_write_locked,
    write_locked_here,
)
from xradio.measurement_set._utils._msv2._tables.table_query import open_table_ro
from xradio.measurement_set._utils._msv2.backend_errors import PartitionCacheWarning
from xradio.measurement_set._utils._msv2.partition_queries import (
    MANDATORY_PARTITION_KEYS,
    PARTITION_ALGORITHM_VERSION,
    PARTITION_MAIN_KEY_COLUMNS,
    MainRowRuns,
    _stack_digests,
    canonical_scheme_key,
    create_partitions_with_main_rows,
    partition_axis_names,
    partition_selection_digest,
    validate_partition_scheme,
)

# The values of partition_cache, and the environment variable that sets the
# default
PARTITION_CACHE_MODES = ("auto", "read", "off", "rebuild")
PARTITION_CACHE_ENV = "XRADIO_MSV2_PARTITION_CACHE"
# Bounds of the per-process memo of partitions
MEMO_MAX_ENTRIES = 16
MEMO_MAX_BYTES = 256 * 2**20

# The sub-table of the stored partitions, and the MAIN keyword that links it
SUBTABLE_NAME = "XRADIO_PARTITIONS"
# Version of the layout of the sub-table (table keyword and every row)
FORMAT_VERSION = 1
# The HISTORY rows of stored partitions
HISTORY_APPLICATION = "xradio"
HISTORY_ORIGIN = "xradio.measurement_set.open_msv2"
# The inputs of the partitions (create_partitions_with_main_rows): the MAIN
# key columns (and ANTENNA2, for the ANTENNA1 rule) and sub-tables
FINGERPRINT_MAIN_COLUMNS = tuple(PARTITION_MAIN_KEY_COLUMNS) + ("ANTENNA2",)
FINGERPRINT_SUBTABLES = ("FIELD", "STATE", "SOURCE")
# Attempts to read a stored row (a row being rewritten reads torn: its
# CHECKSUM does not match, or the read raises)
READ_ATTEMPTS = 3
# Seconds from MJD 0 to 1970-01-01 (TIME columns hold MJD seconds)
MJD_UNIX_OFFSET = 3506716800.0
# At most this many rows are stored (the oldest TIME is evicted), and no row
# of more runs (64 MiB of run arrays)
MAX_STORED_ROWS = 8
MAX_STORED_RUNS = 2**22
# The temporary names of sub-tables being created
# (".XRADIO_PARTITIONS.tmp-<host>-<pid>-<random>"), removed when their
# process is gone or after TMP_MAX_AGE seconds
TMP_PREFIX = f".{SUBTABLE_NAME}.tmp-"
TMP_MAX_AGE = 3600.0
# The MAIN data managers of MSs whose cache is stored (opening MAIN for
# update with others, e.g. LofarStMan, DyscoStMan or AdiosStMan, is untested)
WRITABLE_DATA_MANAGERS = frozenset(
    {
        "StandardStMan",
        "IncrementalStMan",
        "TiledShapeStMan",
        "TiledColumnStMan",
        "TiledCellStMan",
        "TiledDataStMan",
        "StManAipsIO",
    }
)
# Why partitions are not stored while this process holds MAIN's write lock
# (why_not_writable), and why they are computed from the rows of this
# process, without the memo and the stored rows (compute_in_memory, see
# backend_open._check_main_is_current)
MAIN_WRITE_LOCKED = "MAIN write-locked by this process"
MAIN_NOT_FLUSHED = "MAIN rows of this process not flushed"
# The HISTORY columns of the row of a stored content
HISTORY_COLUMNS = (
    "TIME",
    "OBSERVATION_ID",
    "MESSAGE",
    "PRIORITY",
    "ORIGIN",
    "OBJECT_ID",
    "APPLICATION",
    "CLI_COMMAND",
    "APP_PARAMS",
)
# The readme of the table.info of the sub-table
SUBTABLE_README = (
    "Partitions of this MeasurementSet stored by xradio's xradio_msv2 xarray "
    "engine; remove them with xradio.measurement_set.remove_msv2_partition_cache"
)

# The columns of the sub-table, in their order (also that of the CHECKSUM),
# and how their values are normalised: "str", "int" (Int), "count" (Double
# holding an integer: no Int64 anywhere, casatools 6.7 can neither read Int64
# array cells nor getcol Int64 columns), "time" (Double) or "rows" (Double
# array of row numbers, exact below 2**53)
_COLUMNS = (
    ("CACHE_ID", "str", "32 hex characters, new for every stored content"),
    ("SCHEME_KEY", "str", "canonical JSON of the partition scheme (sorted keys)"),
    ("SCHEME", "str", "JSON of the partition scheme as requested"),
    ("FORMAT_VERSION", "int", "layout version of the row"),
    ("ALGORITHM_VERSION", "int", "version of the partitioning algorithm"),
    ("XRADIO_VERSION", "str", "xradio version that stored the row"),
    ("TIME", "time", "time the row was stored"),
    ("MAIN_NROWS", "count", "rows of the MAIN table"),
    ("N_RUNS", "count", "number of MAIN row runs"),
    ("N_ROWS_COVERED", "count", "MAIN rows in a partition"),
    ("N_PARTITIONS", "int", "number of partitions"),
    ("PARTITIONS", "str", "columnar JSON of the partition descriptions"),
    ("ROW_STARTS", "rows", "first MAIN row of every run"),
    ("ROW_LENGTHS", "rows", "number of MAIN rows of every run"),
    ("PARTITION_BOUNDS", "rows", "run range of every partition"),
    ("FINGERPRINT", "str", "JSON fingerprint of the MS the partitions are of"),
    ("HISTORY_ROW", "int", "the HISTORY row of the stored content"),
    ("HISTORY_NROWS_AT_BUILD", "int", "HISTORY rows with the fingerprint"),
    ("CHECKSUM", "str", "blake2b-128 of the other columns"),
)
SUBTABLE_COLUMNS = tuple(name for name, _, _ in _COLUMNS)
_COLUMN_KINDS = {name: kind for name, kind, _ in _COLUMNS}
# Largest integer a Double holds exactly
_MAX_EXACT_DOUBLE = 2**53


def resolve_partition_cache_mode(partition_cache: str | None) -> str:
    """
    The partition cache mode: ``partition_cache``, or the environment
    variable XRADIO_MSV2_PARTITION_CACHE if it is None, or "auto".

    Raises
    ------
    ValueError
        If the mode is not one of PARTITION_CACHE_MODES.
    """
    if partition_cache is None:
        mode = os.environ.get(PARTITION_CACHE_ENV, "auto")
        source = f" (from the environment variable {PARTITION_CACHE_ENV})"
    else:
        mode, source = partition_cache, ""
    if not isinstance(mode, str) or mode not in PARTITION_CACHE_MODES:
        raise ValueError(
            f"partition_cache must be one of {list(PARTITION_CACHE_MODES)}, got "
            f"{mode!r}{source}"
        )
    return mode


# --- the stored layout ------------------------------------------------------------


def subtable_description() -> dict:
    """
    The table description of the XRADIO_PARTITIONS sub-table: a casacore
    table description record (as python-casacore's ``maketabdesc`` makes
    it), one column per SUBTABLE_COLUMNS, all in the default StandardStMan.
    """
    value_types = {
        "str": "string",
        "int": "int",
        "count": "double",
        "time": "double",
        "rows": "double",
    }
    description = {}
    for name, kind, comment in _COLUMNS:
        column = {
            "valueType": value_types[kind],
            "dataManagerType": "",
            "dataManagerGroup": "",
            "option": 0,
            "maxlen": 0,
            "comment": comment,
            "keywords": {},
        }
        if kind == "time":
            column["keywords"] = {
                "QuantumUnits": ["s"],
                "MEASINFO": {"type": "epoch", "Ref": "UTC"},
            }
        if kind == "rows":
            column.update({"ndim": 1, "shape": [], "_c_order": True})
        description[name] = column
    return description


class InvalidRowError(ValueError):
    """A stored row of XRADIO_PARTITIONS is torn, corrupt or inconsistent."""


def _as_int(value: Any, name: str) -> int:
    """An integral value (an int, or an integral float of a Double column)."""
    if isinstance(value, bool | np.bool_):
        raise InvalidRowError(f"{name} is a boolean")
    if isinstance(value, int | np.integer):
        return int(value)
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise InvalidRowError(f"{name} is not a number") from None
    if not (np.isfinite(value) and value == np.floor(value)):
        raise InvalidRowError(f"{name} is not an integer: {value!r}")
    if abs(value) >= _MAX_EXACT_DOUBLE:
        raise InvalidRowError(f"{name} is out of range: {value!r}")
    return int(value)


def _as_rows(value: Any, name: str) -> np.ndarray:
    """A Double array of row numbers (non-negative integers) as int64."""
    try:
        values = np.asarray(value, dtype=np.float64).ravel()
    except (TypeError, ValueError):
        raise InvalidRowError(f"{name} is not an array of numbers") from None
    if not (
        np.all(np.isfinite(values))
        and np.all(values == np.floor(values))
        and np.all(values >= 0)
        and np.all(values < _MAX_EXACT_DOUBLE)
    ):
        raise InvalidRowError(f"{name} holds values that are not row numbers")
    return values.astype(np.int64)


def normalise_row(values: Mapping[str, Any]) -> dict[str, Any]:
    """
    The cells of a stored row as Python values: str, int, float (TIME) and
    int64 arrays (the run arrays).

    Raises
    ------
    InvalidRowError
        If a column is missing or holds a value of the wrong kind.
    """
    row = {}
    for name in SUBTABLE_COLUMNS:
        if name not in values:
            raise InvalidRowError(f"no {name}")
        value, kind = values[name], _COLUMN_KINDS[name]
        if kind == "str":
            if not isinstance(value, str | np.str_):
                raise InvalidRowError(f"{name} is not a string")
            row[name] = str(value)
        elif kind in ("int", "count"):
            row[name] = _as_int(value, name)
        elif kind == "time":
            row[name] = float(value)
        else:
            row[name] = _as_rows(value, name)
    return row


def row_checksum(values: Mapping[str, Any]) -> str:
    """
    The CHECKSUM of a row: blake2b-128 (hex) of the normalised values of the
    other columns, in their order: strings as UTF-8, integers as decimal
    integers (an Int and a Double of the same value give the same bytes),
    TIME as ``float.hex``, the run arrays as little-endian int64.

    Parameters
    ----------
    values : Mapping[str, Any]
        The values of the row (normalise_row or encode_row).
    """
    digest = hashlib.blake2b(digest_size=16)
    for name, kind, _ in _COLUMNS:
        if name == "CHECKSUM":
            continue
        value = values[name]
        if kind == "str":
            data = str(value).encode("utf-8")
        elif kind in ("int", "count"):
            data = str(_as_int(value, name)).encode()
        elif kind == "time":
            data = float(value).hex().encode()
        else:
            data = np.ascontiguousarray(_as_rows(value, name), dtype="<i8").tobytes()
        digest.update(f"{name}:{len(data)}:".encode())
        digest.update(data)
    return digest.hexdigest()


def encode_partitions(
    partitions: Sequence[Mapping[str, list]], keys: Sequence[str]
) -> str:
    """
    The partition descriptions as columnar JSON: ``{"keys": [...],
    "columns": {key: [the values of every partition]}}`` (smaller and faster
    to parse than a list of dicts).

    Raises
    ------
    ValueError
        If a description does not have exactly ``keys``, in that order.
    """
    for partition in partitions:
        if list(partition) != list(keys):
            raise ValueError(
                f"A partition description has the keys {list(partition)}, "
                f"expected {list(keys)}"
            )
    columns = {key: [list(partition[key]) for partition in partitions] for key in keys}
    return json.dumps({"keys": list(keys), "columns": columns}, separators=(",", ":"))


def decode_partitions(text: str) -> list[dict[str, list]]:
    """
    The partition descriptions of encode_partitions (new lists and dicts).

    Raises
    ------
    InvalidRowError
        If the JSON is not that of encode_partitions.
    """
    try:
        content = json.loads(text)
        keys, columns = content["keys"], content["columns"]
        if not isinstance(keys, list) or not isinstance(columns, dict):
            raise TypeError("no keys or columns")
        if sorted(columns) != sorted(keys) or len(set(keys)) != len(keys):
            raise ValueError("the columns are not the keys")
        lengths = {len(columns[key]) for key in keys}
        if len(lengths) > 1:
            raise ValueError("columns of different lengths")
        count = lengths.pop() if lengths else 0
        partitions = [{key: columns[key][i] for key in keys} for i in range(count)]
    except (ValueError, TypeError, KeyError) as exc:
        raise InvalidRowError(f"PARTITIONS is not partition JSON: {exc}") from None
    for partition in partitions:
        for key, values in partition.items():
            if not isinstance(values, list) or not values:
                raise InvalidRowError(f"PARTITIONS: {key} is not a list of values")
    return partitions


def _xradio_version() -> str:
    import importlib.metadata

    try:
        return importlib.metadata.version("xradio")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def encode_row(
    partitions: Sequence[Mapping[str, list]],
    runs: MainRowRuns,
    partition_scheme: Sequence[str],
    fingerprint: str,
    history_row: int,
    history_nrows_at_build: int,
    *,
    cache_id: str | None = None,
    build_time: float | None = None,
) -> dict[str, Any]:
    """
    The cell values of the stored row of a result, with its CHECKSUM.

    Parameters
    ----------
    partitions : Sequence[Mapping[str, list]]
        The partition descriptions.
    runs : MainRowRuns
        Their MAIN rows.
    partition_scheme : Sequence[str]
        The partition scheme.
    fingerprint : str
        JSON of the fingerprint the partitions were computed with
        (fingerprint_json).
    history_row : int
        The HISTORY row of the stored content.
    history_nrows_at_build : int
        The HISTORY rows when the fingerprint was taken (the anchor).
    cache_id : str | None, optional
        By default a new one (32 hex characters).
    build_time : float | None, optional
        MJD seconds, by default now.

    Returns
    -------
    dict[str, Any]
        By column of SUBTABLE_COLUMNS: str, int, float and float64 arrays.

    Raises
    ------
    ValueError
        If the descriptions cannot be encoded.
    """
    scheme = validate_partition_scheme(partition_scheme)
    if build_time is None:
        build_time = time.time() + MJD_UNIX_OFFSET
    values = {
        "CACHE_ID": cache_id or uuid.uuid4().hex,
        "SCHEME_KEY": canonical_scheme_key(scheme),
        "SCHEME": json.dumps(list(partition_scheme)),
        "FORMAT_VERSION": FORMAT_VERSION,
        "ALGORITHM_VERSION": PARTITION_ALGORITHM_VERSION,
        "XRADIO_VERSION": _xradio_version(),
        "TIME": float(build_time),
        "MAIN_NROWS": int(runs.main_nrows),
        "N_RUNS": int(runs.starts.size),
        "N_ROWS_COVERED": int(runs.lengths.sum()),
        "N_PARTITIONS": len(partitions),
        "PARTITIONS": encode_partitions(partitions, partition_axis_names(scheme)),
        "ROW_STARTS": runs.starts.astype(np.float64),
        "ROW_LENGTHS": runs.lengths.astype(np.float64),
        "PARTITION_BOUNDS": runs.bounds.astype(np.float64),
        "FINGERPRINT": fingerprint,
        "HISTORY_ROW": int(history_row),
        "HISTORY_NROWS_AT_BUILD": int(history_nrows_at_build),
    }
    values["CHECKSUM"] = row_checksum(values)
    return values


def check_row_checksum(row: Mapping[str, Any]) -> None:
    """Raise InvalidRowError if the CHECKSUM of a (normalised) row does not
    match its other columns."""
    if row_checksum(row) != row["CHECKSUM"]:
        raise InvalidRowError("CHECKSUM mismatch (a torn or modified row)")


def decode_row(
    row: Mapping[str, Any], partition_scheme: Sequence[str]
) -> tuple[list[dict], MainRowRuns]:
    """
    The partitions and runs of a stored row (normalise_row), checked for
    consistency (L3; check_row_checksum checks the CHECKSUM).

    Raises
    ------
    InvalidRowError
        If the row is not consistent.
    """
    scheme = validate_partition_scheme(partition_scheme)

    def check(condition: bool, what: str) -> None:
        if not condition:
            raise InvalidRowError(what)

    check(row["FORMAT_VERSION"] == FORMAT_VERSION, "unknown FORMAT_VERSION")
    check(
        row["ALGORITHM_VERSION"] == PARTITION_ALGORITHM_VERSION,
        "another ALGORITHM_VERSION",
    )
    check(row["SCHEME_KEY"] == canonical_scheme_key(scheme), "another SCHEME_KEY")
    partitions = decode_partitions(row["PARTITIONS"])
    check(
        not partitions or list(partitions[0]) == partition_axis_names(scheme),
        "PARTITIONS has other keys than the descriptions of the scheme",
    )
    starts, lengths = row["ROW_STARTS"], row["ROW_LENGTHS"]
    bounds = row["PARTITION_BOUNDS"]
    main_nrows, n_runs = row["MAIN_NROWS"], row["N_RUNS"]
    check(
        row["N_PARTITIONS"] == len(partitions) == bounds.size - 1,
        "N_PARTITIONS, PARTITIONS and PARTITION_BOUNDS disagree",
    )
    check(
        starts.size == lengths.size == n_runs,
        "ROW_STARTS, ROW_LENGTHS and N_RUNS disagree",
    )
    check(
        bounds[0] == 0 and bounds[-1] == n_runs and bool(np.all(np.diff(bounds) >= 0)),
        "PARTITION_BOUNDS are not run ranges",
    )
    check(bool(np.all(lengths > 0)), "empty runs")
    check(bool(np.all(starts + lengths <= main_nrows)), "runs beyond MAIN_NROWS")
    if n_runs > 1:
        # the runs of a partition ascend and do not touch
        same = np.ones(n_runs - 1, dtype=bool)
        inner = bounds[1:-1]
        same[inner[(inner >= 1) & (inner <= n_runs - 1)] - 1] = False
        check(
            bool(np.all(starts[1:][same] > (starts[:-1] + lengths[:-1])[same])),
            "the runs of a partition do not ascend",
        )
        # no row is in two runs
        order = np.argsort(starts, kind="stable")
        check(
            bool(np.all((starts + lengths)[order][:-1] <= starts[order][1:])),
            "overlapping runs",
        )
    check(int(lengths.sum()) == row["N_ROWS_COVERED"], "N_ROWS_COVERED disagrees")
    try:
        fingerprint_nrows = json.loads(row["FINGERPRINT"])["main"]["nrows"]
    except (ValueError, TypeError, KeyError):
        raise InvalidRowError("FINGERPRINT is not a fingerprint") from None
    check(main_nrows == fingerprint_nrows, "MAIN_NROWS is not that of FINGERPRINT")
    # one value of every grouping key per partition, distinct partitions
    grouping_keys = list(MANDATORY_PARTITION_KEYS) + scheme
    check(
        all(len(p[key]) == 1 for p in partitions for key in grouping_keys),
        "a grouping key of a partition has several values",
    )
    check(
        len({tuple(p[key][0] for key in grouping_keys) for p in partitions})
        == len(partitions),
        "two partitions have the same grouping keys",
    )
    if "ANTENNA1" not in scheme:
        check(row["N_ROWS_COVERED"] == main_nrows, "MAIN rows in no partition")
    runs = MainRowRuns(
        starts,
        lengths,
        bounds,
        _stack_digests([partition_selection_digest(p) for p in partitions]),
        main_nrows,
    )
    return partitions, runs


# --- the fingerprint and the HISTORY rule ------------------------------------------


def fingerprint_json(path: str) -> str:
    """The fingerprint of the inputs of the partitions of an MS
    (table_lock_file.ms_fingerprint) as JSON, with sorted keys."""
    fingerprint = ms_fingerprint(path, FINGERPRINT_MAIN_COLUMNS, FINGERPRINT_SUBTABLES)
    return json.dumps(fingerprint, sort_keys=True, separators=(",", ":"))


def same_fingerprint(stored: str, current: str) -> bool:
    """Whether a stored fingerprint (JSON) equals the current one."""
    if stored == current:
        return True
    try:
        return json.loads(stored) == json.loads(current)
    except ValueError:
        return False


def _fingerprint_nrows(fingerprint: str) -> int:
    return int(json.loads(fingerprint)["main"]["nrows"])


def _app_params(cell: Any) -> list[str]:
    return [str(value) for value in np.asarray(cell, dtype=object).ravel()]


def _is_content_history_row(table, index: int, cache_id: str) -> bool:
    """Whether a row of an opened HISTORY table is the row of a stored
    content (xradio's cache rows, with the content's cache_id)."""
    return (
        table.getcell("APPLICATION", index) == HISTORY_APPLICATION
        and table.getcell("ORIGIN", index) == HISTORY_ORIGIN
        and f"cache_id={cache_id}" in _app_params(table.getcell("APP_PARAMS", index))
    )


def history_rule(
    path: str, row: Mapping[str, Any], n_history: int | None
) -> str | None:
    """
    The HISTORY rule (L2) for a stored row: why HISTORY says that the MS may
    have changed since the row's partitions were computed, or None.

    Stale if HISTORY is missing or has fewer rows than the anchor
    (HISTORY_NROWS_AT_BUILD); if HISTORY_ROW is not the HISTORY row of the
    stored content (APPLICATION and ORIGIN of xradio's cache rows, the row's
    CACHE_ID in APP_PARAMS); or if a row at or after the anchor is not a row
    of xradio's partition cache. By row number (HISTORY TIME is not
    monotonic); only the APPLICATION and ORIGIN cells after the anchor are
    read.

    Parameters
    ----------
    path : str
        Path of the MS.
    row : Mapping[str, Any]
        The stored row (normalise_row).
    n_history : int | None
        The number of HISTORY rows, taken with the fingerprint
        (table_lock_file.history_nrows).
    """
    anchor, history_row = row["HISTORY_NROWS_AT_BUILD"], row["HISTORY_ROW"]
    if n_history is None:
        return "no HISTORY table"
    if n_history < anchor:
        return f"HISTORY has {n_history} rows, {anchor} when the partitions were stored"
    if not 0 <= history_row < n_history:
        return f"the HISTORY row {history_row} of the partitions is gone"
    with casatools_serialized(), open_table_ro(os.path.join(path, "HISTORY")) as table:
        if table.nrows() < n_history:
            # (a table object of this process that has not seen new rows; kept
            # if the process holds its write lock)
            resync_unless_write_locked(table)
        if table.nrows() < n_history:
            return "HISTORY changed while it was read"
        if not _is_content_history_row(table, history_row, row["CACHE_ID"]):
            return f"the HISTORY row {history_row} is not that of the partitions"
        if n_history > anchor:
            count = n_history - anchor
            applications = table.getcol("APPLICATION", anchor, count)
            origins = table.getcol("ORIGIN", anchor, count)
            for offset, (application, origin) in enumerate(
                zip(applications, origins, strict=True)
            ):
                if application != HISTORY_APPLICATION or origin != HISTORY_ORIGIN:
                    return (
                        f"the HISTORY row {anchor + offset} ({application}, {origin}) "
                        "is newer than the partitions"
                    )
    return None


# --- reading the stored row -----------------------------------------------------------


def subtable_path(path: str) -> str:
    """The path of the XRADIO_PARTITIONS sub-table of an MS."""
    return os.path.join(path, SUBTABLE_NAME)


def link_state(path: str) -> str:
    """
    How MAIN links the XRADIO_PARTITIONS sub-table: "linked" (keyword and
    table), "absent" (neither), "unlinked" (a table without the keyword),
    "dangling" (the keyword without the table) or "foreign" (a keyword
    XRADIO_PARTITIONS that is no link to the sub-table).
    """
    with casatools_serialized(), open_table_ro(path) as main_tb:
        keyword = (
            main_tb.getkeyword(SUBTABLE_NAME)
            if SUBTABLE_NAME in main_tb.keywordnames()
            else None
        )
    exists = os.path.isfile(os.path.join(subtable_path(path), "table.dat"))
    if keyword is None:
        return "unlinked" if exists else "absent"
    if not _is_subtable_link(keyword):
        return "foreign"
    return "linked" if exists else "dangling"


def _is_subtable_link(keyword: Any) -> bool:
    """Whether the value of the MAIN keyword XRADIO_PARTITIONS is a link to
    the sub-table (a table keyword: "Table: <path>/XRADIO_PARTITIONS")."""
    return (
        isinstance(keyword, str)
        and keyword.startswith("Table: ")
        and keyword.rstrip("/").endswith("/" + SUBTABLE_NAME)
    )


@dataclasses.dataclass
class StoredLookup:
    """
    What the XRADIO_PARTITIONS sub-table holds for a scheme
    (lookup_stored_row).

    Attributes
    ----------
    state : str
        "row" (``row`` holds its cells), or why there is none: "absent",
        "unlinked", "dangling", "foreign" (link_state), "not an xradio
        cache", "newer format", "no row", "duplicate rows", "unreadable" (the
        reads raised) or "torn" (CHECKSUM mismatch on every attempt).
    row : dict | None
        The normalised cells of the row (its CHECKSUM checked).
    index : int | None
        The row number in the sub-table.
    detail : str
        The error of the last attempt, if any.
    """

    state: str
    row: dict | None = None
    index: int | None = None
    detail: str = ""


def subtable_format(table) -> int | None:
    """The FORMAT_VERSION of an opened XRADIO_PARTITIONS table, None if it
    is not an xradio partition cache."""
    keywords = table.keywordnames()
    if "FORMAT_VERSION" not in keywords or "CREATOR" not in keywords:
        return None
    if table.getkeyword("CREATOR") != "xradio":
        return None
    try:
        return operator.index(table.getkeyword("FORMAT_VERSION"))
    except TypeError:
        return None


def find_rows(table, scheme_key: str) -> list[int]:
    """The rows of an opened XRADIO_PARTITIONS table for a scheme key and
    this ALGORITHM_VERSION."""
    if table.nrows() == 0:
        return []
    keys = table.getcol("SCHEME_KEY")
    versions = table.getcol("ALGORITHM_VERSION")
    return [
        index
        for index, (key, version) in enumerate(zip(keys, versions, strict=True))
        if key == scheme_key and int(version) == PARTITION_ALGORITHM_VERSION
    ]


def read_row_cells(table, index: int) -> dict[str, Any]:
    """The normalised cells of a row of an opened XRADIO_PARTITIONS table."""
    return normalise_row(
        {name: table.getcell(name, index) for name in SUBTABLE_COLUMNS}
    )


def lookup_stored_row(path: str, scheme_key: str) -> StoredLookup:
    """
    The stored row of a scheme. Reader protocol: MAIN and the sub-table are
    opened without locks and closed; a row whose CHECKSUM does not match or
    whose read raises (a row being written) is read again, at most
    READ_ATTEMPTS times.

    Parameters
    ----------
    path : str
        Path of the MS.
    scheme_key : str
        canonical_scheme_key of the scheme.

    Returns
    -------
    StoredLookup
        The row's cells (CHECKSUM checked; the other checks are the
        caller's), or why there is none.
    """
    state = link_state(path)
    if state != "linked":
        return StoredLookup(state)
    detail = ""
    for attempt in range(READ_ATTEMPTS):
        if attempt:
            time.sleep(0.01 * attempt)
        try:
            with casatools_serialized(), open_table_ro(subtable_path(path)) as table:
                version = subtable_format(table)
                if version is None:
                    return StoredLookup("not an xradio cache")
                if version > FORMAT_VERSION:
                    return StoredLookup("newer format")
                indices = find_rows(table, scheme_key)
                if not indices:
                    return StoredLookup("no row")
                if len(indices) > 1:
                    return StoredLookup("duplicate rows")
                row = read_row_cells(table, indices[0])
            check_row_checksum(row)
            return StoredLookup("row", row, indices[0])
        except InvalidRowError as exc:
            state, detail = "torn", str(exc)
        except Exception as exc:  # (torn reads of a row being written raise)
            state, detail = "unreadable", f"{type(exc).__name__}: {exc}"
    return StoredLookup(state, detail=detail)


def check_stored_row(
    path: str,
    lookup: StoredLookup,
    partition_scheme: list[str],
    fingerprint: str,
    n_history: int | None,
) -> tuple[tuple[list[dict], MainRowRuns] | None, str]:
    """
    The partitions of a stored row if it is valid for the MS now (L1 and
    L2, then L3: a stale row is never parsed), or why it is not.

    Returns
    -------
    tuple[tuple[list[dict], MainRowRuns] | None, str]
        ``((partitions, runs), "")``, or ``(None, reason)`` with the reason
        "<the lookup state>", "fingerprint", "history: <what>" or "corrupt:
        <what>".
    """
    if lookup.state != "row":
        return None, lookup.state
    row = lookup.row
    if not same_fingerprint(row["FINGERPRINT"], fingerprint):
        return None, "fingerprint"
    stale = history_rule(path, row, n_history)
    if stale is not None:
        return None, f"history: {stale}"
    try:
        return decode_row(row, partition_scheme), ""
    except InvalidRowError as exc:
        return None, f"corrupt: {exc}"


# --- the per-process memo --------------------------------------------------------------


def _copy_runs(runs: MainRowRuns) -> MainRowRuns:
    return MainRowRuns(
        runs.starts.copy(),
        runs.lengths.copy(),
        runs.bounds.copy(),
        runs.digests.copy(),
        runs.main_nrows,
    )


@dataclasses.dataclass
class PartitionsResult:
    """
    The partitions of an MS for one partition scheme.

    Attributes
    ----------
    partitions : list[dict]
        Partition descriptions (create_partitions).
    runs : MainRowRuns
        Their MAIN rows.
    source : str
        "fresh" (computed by this call), "memo" or "stored" (the row of
        XRADIO_PARTITIONS).
    status : str
        How they were obtained: "hit" (the stored row), "hit-memory" (the
        memo); computed and "stored" (a new row), "revalidated" (equal to the
        stored row, whose fingerprint and anchor were renewed) or "hit-race"
        (another process stored them meanwhile); or computed and kept in
        memory, "memory:<reason>": "mode-off", "mode-read", "no-fingerprint"
        (the fingerprint of the MS could not be computed: neither stored
        rows nor the memo are used), "changed-during-build" (the MS changed
        while they were computed: not memoised), "locked", "write failed",
        a reason of why_not_writable, or MAIN_NOT_FLUSHED (compute_in_memory:
        the view of this process, which holds MAIN's write lock with rows
        not flushed yet).
    fingerprint, history_nrows : str | None, int | None
        For partitions from the memo or a stored row: the fingerprint and
        HISTORY rows of the MS they were validated with (changed_since).
    """

    partitions: list[dict]
    runs: MainRowRuns
    source: str
    status: str
    fingerprint: str | None = None
    history_nrows: int | None = None

    @property
    def main_nrows(self) -> int:
        """The MAIN rows the partitions were computed for."""
        return self.runs.main_nrows


@dataclasses.dataclass
class _MemoEntry:
    fingerprint: str
    history_nrows: int | None
    partitions: list[dict]
    runs: MainRowRuns

    def result(self, source: str, status: str) -> PartitionsResult:
        return PartitionsResult(
            copy.deepcopy(self.partitions),
            _copy_runs(self.runs),
            source,
            status,
            self.fingerprint,
            self.history_nrows,
        )


class _PartitionsMemo:
    """LRU memo of validated partitions by (realpath, scheme key, algorithm
    version), with one build lock per key."""

    def __init__(self, max_entries: int, max_bytes: int):
        self.max_entries = int(max_entries)
        self.max_bytes = int(max_bytes)
        self.reset_after_fork()

    def reset_after_fork(self) -> None:
        """Empty, with new locks (registered with os.register_at_fork)."""
        self._lock = threading.Lock()
        self._entries: collections.OrderedDict[tuple, _MemoEntry] = (
            collections.OrderedDict()
        )
        self._build_locks: dict[tuple, threading.Lock] = {}
        self.stats: collections.Counter = collections.Counter()

    def build_lock(self, key: tuple) -> threading.Lock:
        """The lock held while the partitions of ``key`` are looked up or
        computed (threads opening one MS compute them once)."""
        with self._lock:
            return self._build_locks.setdefault(key, threading.Lock())

    def get(self, key: tuple) -> _MemoEntry | None:
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None:
                self._entries.move_to_end(key)
            return entry

    def put(self, key: tuple, entry: _MemoEntry) -> None:
        with self._lock:
            self._entries.pop(key, None)
            self._entries[key] = entry
            while len(self._entries) > 1 and (
                len(self._entries) > self.max_entries
                or sum(e.runs.nbytes for e in self._entries.values()) > self.max_bytes
            ):
                self._entries.popitem(last=False)
                self.stats["evictions"] += 1

    def discard_path(self, path: str) -> None:
        """Drop the entries of an MS."""
        realpath = os.path.realpath(path)
        with self._lock:
            for key in [key for key in self._entries if key[0] == realpath]:
                del self._entries[key]

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self.stats.clear()

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, key: tuple) -> bool:
        return key in self._entries


PARTITIONS_MEMO = _PartitionsMemo(MEMO_MAX_ENTRIES, MEMO_MAX_BYTES)

# The notices given (once per (realpath, reason) per process)
_NOTICES: set[tuple[str, str]] = set()
_NOTICES_LOCK = threading.Lock()


def _reset_after_fork() -> None:
    """Renew the memo and the locks of this module in a fork child."""
    global _NOTICES_LOCK
    PARTITIONS_MEMO.reset_after_fork()
    _NOTICES_LOCK = threading.Lock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_after_fork)


def clear_partition_memo() -> None:
    """Empty the per-process memo of partitions (for tests)."""
    PARTITIONS_MEMO.clear()


def memo_key(path: str, partition_scheme: list[str]) -> tuple:
    """Key of the partitions of an MS for a scheme in PARTITIONS_MEMO."""
    return (
        os.path.realpath(path),
        canonical_scheme_key(partition_scheme),
        PARTITION_ALGORITHM_VERSION,
    )


def _first_notice(path: str, reason: str) -> bool:
    """Whether this is the first notice of ``reason`` for an MS in this
    process."""
    key = (os.path.realpath(path), reason)
    with _NOTICES_LOCK:
        if key in _NOTICES:
            return False
        _NOTICES.add(key)
        return True


# --- writing ----------------------------------------------------------------------------


class CacheNotStored(Exception):
    """
    Why partitions are not stored in an MS (the status "memory:<reason>").

    Parameters
    ----------
    reason : str
        The reason (e.g. "locked").
    warn : bool, optional
        Whether to notify it with a PartitionCacheWarning (else an INFO log).
    """

    def __init__(self, reason: str, warn: bool = True):
        super().__init__(reason)
        self.reason = reason
        self.warn = warn


# The reasons whose notice says more than the reason
_REASON_DETAILS = {"locked": "locked: another process has the MS open"}
# Writers open tables for update without read locks (an open with read
# locking waits for any write lock), then lock them
_WRITE_LOCKOPTIONS = {"option": "usernoread"}

_WRITE_MUTEXES: dict[str, threading.RLock] = {}
_WRITE_MUTEXES_LOCK = threading.Lock()


def _reset_write_mutexes_after_fork() -> None:
    global _WRITE_MUTEXES_LOCK
    _WRITE_MUTEXES.clear()
    _WRITE_MUTEXES_LOCK = threading.Lock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_write_mutexes_after_fork)


def write_mutex(path: str) -> threading.RLock:
    """
    The mutex (reentrant) of the partition cache of an MS in this process.

    casacore's locks belong to a process, and a process shares one table
    object per table: two threads would both hold a write lock, and closing
    any handle of a table releases its locks (python-casacore's close
    unlocks, which flushes). So the cache's own reads (fingerprint, stored
    row, HISTORY) and writes of an MS hold this mutex. Other readers of MAIN
    in this process (partition builds, lazy reads of other trees) do not:
    one that closes its MAIN handle while the MAIN keyword is written
    releases the MAIN write lock. The keyword writes take the lock again in
    that case (``_write_main_keywords``: one attempt before the change and
    before the flush, and a retry when casacore reports the table unlocked),
    which leaves only a window of a few Python statements in which another
    process could take the lock and write MAIN's table.dat too.
    """
    realpath = os.path.realpath(path)
    with _WRITE_MUTEXES_LOCK:
        return _WRITE_MUTEXES.setdefault(realpath, threading.RLock())


def notify_not_stored(path: str, reason: str, warn: bool, detail: str = "") -> None:
    """
    Tell, once per MS and reason in a process, that the partitions of an MS
    are not stored: a PartitionCacheWarning, or an INFO log for reasons that
    are no user error (``warn`` False).
    """
    if not _first_notice(path, reason):
        return
    what = _REASON_DETAILS.get(reason, reason) + (f": {detail}" if detail else "")
    message = (
        f"The partition cache of {path} is not stored ({what}); the partitions are "
        "computed in memory. Pass partition_cache='read' or 'off' to silence this."
    )
    if warn:
        warnings.warn(message, PartitionCacheWarning, stacklevel=2)
    else:
        xradio_logger().info(message)


def _tables_module():
    """The casacore tables of xradio: python-casacore, or the casatools
    shim."""
    from xradio.measurement_set._utils._msv2._tables.table_query import tables

    return tables


def _writable_table(table_path: str) -> bool:
    """Whether a table directory and every file in it can be written."""
    if not os.access(table_path, os.W_OK | os.X_OK):
        return False
    for name in os.listdir(table_path):
        file_path = os.path.join(table_path, name)
        if os.path.isfile(file_path) and not os.access(file_path, os.W_OK):
            return False
    return True


def why_not_writable(
    path: str, runs: MainRowRuns | None = None
) -> tuple[str, bool] | None:
    """
    The first reason not to store partitions in an MS, and whether to warn
    about it (checked before any write: a failed store would leave partial
    state); None if they can be stored.

    Parameters
    ----------
    path : str
        Path of the MS.
    runs : MainRowRuns | None, optional
        The runs to store (too many are not stored).

    Returns
    -------
    tuple[str, bool] | None
        (reason, warn) or None.
    """
    if uses_casatools():
        return "casatools only", False  # (the shim cannot create tables)
    from casacore import tables

    if not os.access(path, os.W_OK | os.X_OK):
        return "MS directory not writable", True
    if not tables.tableiswritable(path):
        return "MAIN table not writable", True
    with open_table_ro(path) as main_tb:
        parts = [os.path.realpath(name) for name in main_tb.partnames()]
        dm_types = sorted({str(dm["TYPE"]) for dm in main_tb.getdminfo().values()})
        keyword = (
            main_tb.getkeyword(SUBTABLE_NAME)
            if SUBTABLE_NAME in main_tb.keywordnames()
            else None
        )
        writing = write_locked_here(main_tb)
    if parts != [os.path.realpath(path)]:
        # (before the lock file: such a table has none)
        return "MAIN is a reference or concatenated table", False
    if not os.access(os.path.join(path, "table.lock"), os.W_OK):
        return "MAIN lock file not writable", True
    history = os.path.join(path, "HISTORY")
    if not os.path.isfile(os.path.join(history, "table.dat")):
        return "HISTORY missing", True
    if not (tables.tableiswritable(history) and _writable_table(history)):
        return "HISTORY not writable", True
    with open_table_ro(history) as table:
        missing = [name for name in HISTORY_COLUMNS if name not in table.colnames()]
    if missing:
        return f"HISTORY has no {missing[0]} column", True
    subtable = subtable_path(path)
    if os.path.lexists(subtable):
        if not os.path.isfile(os.path.join(subtable, "table.dat")):
            return f"{SUBTABLE_NAME} is not an xradio partition cache", False
        if not (tables.tableiswritable(subtable) and _writable_table(subtable)):
            return f"{SUBTABLE_NAME} not writable", True
        with open_table_ro(subtable) as table:
            version = subtable_format(table)
        if version is None:
            return f"{SUBTABLE_NAME} is not an xradio partition cache", False
        if version > FORMAT_VERSION:
            return "cache written by a newer xradio", False
        if version < FORMAT_VERSION:  # (none yet)
            return "cache written by an older xradio", False
    if keyword is not None and not _is_subtable_link(keyword):
        return f"the MAIN keyword {SUBTABLE_NAME} is no link to the cache", False
    if writing:
        # (a writable handle of this process, e.g. the user's, may have
        # changes not flushed yet: the partitions computed from them would be
        # stored with the fingerprint of the files, and the MAIN keyword
        # write would flush them and release the handle's lock)
        return MAIN_WRITE_LOCKED, False
    others = [name for name in dm_types if name not in WRITABLE_DATA_MANAGERS]
    if others:
        return f"MAIN uses {others[0]}", False
    if runs is not None and runs.starts.size > MAX_STORED_RUNS:
        return "too many runs", False
    return None


@contextlib.contextmanager
def _locked_for_update(table_path: str):
    """
    A table opened for update (without read locks: the open never waits) and
    write-locked with one attempt: CacheNotStored("locked") if another
    process holds a lock (python-casacore's lock() does not raise when it
    fails: haslock tells). Unlocked (which flushes) and closed on exit.
    """
    tables = _tables_module()
    table = tables.table(
        table_path, readonly=False, lockoptions=_WRITE_LOCKOPTIONS, ack=False
    )
    try:
        table.lock(write=True, nattempts=1)
        if not table.haslock(write=True):
            raise CacheNotStored("locked")
        try:
            yield table
        finally:
            table.unlock()
    finally:
        table.close()


# Attempts of a MAIN keyword write whose write lock another thread of this
# process released (see _write_main_keywords)
KEYWORD_WRITE_ATTEMPTS = 3


def _hold_write_lock(table) -> None:
    """Take the write lock of a table opened for update again if this
    process lost it (one attempt): CacheNotStored("locked") if another
    process holds a lock."""
    if not table.haslock(write=True):
        table.lock(write=True, nattempts=1)
        if not table.haslock(write=True):
            raise CacheNotStored("locked")


def _write_main_keywords(main_tb, change) -> bool:
    """
    ``change(main_tb)`` (keyword changes of MAIN, returning whether it
    changed anything) and, if it did, a flush, holding the MAIN write lock
    that ``_locked_for_update`` took. Returns what ``change`` returned.

    python-casacore's close() unlocks (and flushes) the table object that a
    process shares per table, so another thread of this process that closes
    a handle of MAIN releases the lock: it is taken again (one attempt)
    before the change and before the flush, and ``change`` is retried
    (KEYWORD_WRITE_ATTEMPTS in all) when casacore reports that the table is
    not locked. ``change`` must be idempotent and check MAIN again (a lock
    taken again re-reads a table that another process changed).
    """
    for attempt in range(1, KEYWORD_WRITE_ATTEMPTS + 1):
        _hold_write_lock(main_tb)
        try:
            changed = bool(change(main_tb))
            if changed:
                # (lost meanwhile: the unlock of the closing handle flushed)
                _hold_write_lock(main_tb)
                main_tb.flush()
            return changed
        except RuntimeError as exc:
            if "should be locked" not in str(exc) or attempt == KEYWORD_WRITE_ATTEMPTS:
                raise
            xradio_logger().debug(
                f"The write lock of {main_tb.name()} was released by another "
                f"thread of this process ({exc}): taken again"
            )
    raise AssertionError("unreachable")  # pragma: no cover


def _process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return True
    return True


def remove_stale_tmp_tables(path: str) -> list[str]:
    """
    Remove the temporary sub-tables (TMP_PREFIX) that a writer left behind:
    those of a process of this host that is gone, and those older than
    TMP_MAX_AGE. Best effort. Returns the names removed.
    """
    try:
        names = os.listdir(path)
    except OSError:
        return []
    host, now, removed = socket.gethostname(), time.time(), []
    for name in names:
        if not name.startswith(TMP_PREFIX):
            continue
        full = os.path.join(path, name)
        try:
            owner, pid, _ = name[len(TMP_PREFIX) :].rsplit("-", 2)
            gone = owner == host and not _process_alive(int(pid))
        except ValueError:
            gone = False
        try:
            old = now - os.lstat(full).st_mtime > TMP_MAX_AGE
        except OSError:
            continue
        if gone or old:
            shutil.rmtree(full, ignore_errors=True)
            removed.append(name)
    return removed


def _create_subtable(path: str) -> str | None:
    """
    Create the empty sub-table under a temporary name and rename it into
    place (never at its final name: casacore would replace a table there).
    Returns its path, or None if another writer's is already there.
    """
    from casacore import tables

    final = subtable_path(path)
    tmp = os.path.join(
        path, f"{TMP_PREFIX}{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    )
    try:
        with tables.table(
            tmp, subtable_description(), nrow=0, readonly=False, ack=False
        ) as table:
            table.putkeyword("FORMAT_VERSION", FORMAT_VERSION)
            table.putkeyword("CREATOR", "xradio")
            table.putinfo(
                {"type": "XRADIO Partitions", "subType": "", "readme": SUBTABLE_README}
            )
        try:
            _rename_subtable(tmp, final)
        except OSError:
            if not os.path.isfile(os.path.join(final, "table.dat")):
                raise
            shutil.rmtree(tmp, ignore_errors=True)
            return None
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return final


def _rename_subtable(tmp: str, final: str) -> None:
    os.rename(tmp, final)


def _put_main_keyword(main_tb, path: str) -> bool:
    """Link the sub-table from MAIN, unless MAIN has the keyword (the caller
    flushes, see _write_main_keywords). Returns whether it was written."""
    if SUBTABLE_NAME in main_tb.keywordnames():
        return False
    # (python-casacore makes a table keyword of "Table: <path>", which casacore
    # stores relative to MAIN: copies and renames of the MS keep the link)
    main_tb.putkeyword(SUBTABLE_NAME, "Table: " + subtable_path(path))
    return True


def _ensure_linked_subtable(path: str) -> None:
    """
    Steps 1 and 2 of the write protocol: the sub-table exists and MAIN links
    it. Both are made under the MAIN write lock, the sub-table first: no
    failure leaves the keyword without the sub-table, and a failure that
    Python sees leaves no sub-table without the keyword.
    """
    state = link_state(path)
    if state == "linked":
        return
    if state == "foreign":
        raise CacheNotStored(
            f"the MAIN keyword {SUBTABLE_NAME} is no link to the cache", warn=False
        )
    created = None
    try:
        with _locked_for_update(path) as main_tb:
            if not os.path.isfile(os.path.join(subtable_path(path), "table.dat")):
                created = _create_subtable(path)
            # (the lock may have been released by another thread meanwhile)
            _write_main_keywords(main_tb, lambda tb: _put_main_keyword(tb, path))
        if link_state(path) != "linked":
            raise RuntimeError(f"MAIN does not link {SUBTABLE_NAME} after writing it")
    except BaseException:
        if created is not None and link_state(path) == "unlinked":
            shutil.rmtree(created, ignore_errors=True)
        raise


def _same_content(
    stored: tuple[list[dict], MainRowRuns], partitions: list[dict], runs: MainRowRuns
) -> bool:
    stored_partitions, stored_runs = stored
    return (
        stored_partitions == partitions
        and stored_runs.main_nrows == runs.main_nrows
        and np.array_equal(stored_runs.starts, runs.starts)
        and np.array_equal(stored_runs.lengths, runs.lengths)
        and np.array_equal(stored_runs.bounds, runs.bounds)
    )


def _find_content_history_row(path: str, row: Mapping[str, Any]) -> int | None:
    """
    The row number of the HISTORY row of a stored content: HISTORY_ROW, or,
    if HISTORY rows before it were removed (or the table rewritten), the row
    of xradio's cache with the content's cache_id; None if there is none.
    """
    index, cache_id = row["HISTORY_ROW"], row["CACHE_ID"]
    with open_table_ro(os.path.join(path, "HISTORY")) as table:
        n_rows = history_nrows(path) or 0
        if table.nrows() < n_rows:
            resync_unless_write_locked(table)
        n_rows = table.nrows()
        if 0 <= index < n_rows and _is_content_history_row(table, index, cache_id):
            return index
        if n_rows == 0:
            return None
        candidates = np.flatnonzero(
            (
                np.asarray(table.getcol("APPLICATION"), dtype=object)
                == HISTORY_APPLICATION
            )
            & (np.asarray(table.getcol("ORIGIN"), dtype=object) == HISTORY_ORIGIN)
        )
        for candidate in candidates[::-1].tolist():
            if f"cache_id={cache_id}" in _app_params(
                table.getcell("APP_PARAMS", candidate)
            ):
                return candidate
    return None


def _append_history_row(
    path: str,
    partition_scheme: list[str],
    partitions: list[dict],
    runs: MainRowRuns,
    cache_id: str,
    reason: str,
) -> int:
    """Append the HISTORY row of a stored content (CASA's conventions for
    tasks); returns its row number."""
    version = _xradio_version()
    scheme_key = canonical_scheme_key(partition_scheme)
    message = (
        f"xradio {version}: stored {len(partitions)} partitions ({runs.starts.size} "
        f"MAIN row runs) of partition_scheme {json.dumps(list(partition_scheme))} in "
        f"{SUBTABLE_NAME} ({reason})"
    )
    app_params = [
        f"subtable={SUBTABLE_NAME}",
        f"cache_id={cache_id}",
        f"scheme_key={scheme_key}",
        f"format_version={FORMAT_VERSION}",
        f"algorithm_version={PARTITION_ALGORITHM_VERSION}",
        f"xradio_version={version}",
        f"reason={reason}",
    ]
    with _locked_for_update(os.path.join(path, "HISTORY")) as table:
        row = table.nrows()
        table.addrows(1)
        table.putcell("TIME", row, time.time() + MJD_UNIX_OFFSET)
        table.putcell("OBSERVATION_ID", row, -1)
        table.putcell("MESSAGE", row, message)
        table.putcell("PRIORITY", row, "INFO")
        table.putcell("ORIGIN", row, HISTORY_ORIGIN)
        table.putcell("OBJECT_ID", row, 0)
        table.putcell("APPLICATION", row, HISTORY_APPLICATION)
        table.putcell("CLI_COMMAND", row, np.array([""]))
        table.putcell("APP_PARAMS", row, np.array(app_params))
        table.flush()
    return row


def _put_row_cells(table, index: int, values: Mapping[str, Any]) -> None:
    """Write the cells of a row, its CHECKSUM last (a reader of a half
    written row finds a CHECKSUM mismatch)."""
    for name in SUBTABLE_COLUMNS:
        if name != "CHECKSUM":
            table.putcell(name, index, values[name])
    table.putcell("CHECKSUM", index, values["CHECKSUM"])


def _remove_rows(table, keep: int, remove: Sequence[int]) -> None:
    """Remove the rows ``remove``, and the oldest rows (TIME) but ``keep``
    beyond MAX_STORED_ROWS."""
    remove = set(remove) - {keep}
    excess = table.nrows() - len(remove) - MAX_STORED_ROWS
    if excess > 0:
        times = table.getcol("TIME")
        oldest = sorted(
            (float(when), index)
            for index, when in enumerate(times)
            if index != keep and index not in remove
        )
        remove.update(index for _, index in oldest[:excess])
    if remove:
        table.removerows(sorted(remove))


def _write_row(
    path: str,
    partition_scheme: list[str],
    partitions: list[dict],
    runs: MainRowRuns,
    fingerprint: str,
    n_history: int,
    reason: str,
    rebuild: bool,
) -> tuple[str, int | None]:
    """Step 3 of the write protocol (see store_partitions)."""
    with _locked_for_update(subtable_path(path)) as table:
        version = subtable_format(table)
        if version != FORMAT_VERSION:
            raise CacheNotStored(f"{SUBTABLE_NAME} format {version}", warn=False)
        indices = find_rows(table, canonical_scheme_key(partition_scheme))
        stored, row = None, None
        if indices:
            try:
                row = read_row_cells(table, indices[0])
                check_row_checksum(row)
                stored = decode_row(row, partition_scheme)
            except Exception:  # (a torn or half written row: replaced)
                stored = None
        if stored is not None and _same_content(stored, partitions, runs):
            if (
                not rebuild
                and same_fingerprint(row["FINGERPRINT"], fingerprint)
                and history_rule(path, row, history_nrows(path)) is None
            ):
                return "hit-race", None  # (stored by another writer meanwhile)
            history_row = _find_content_history_row(path, row)
            if history_row is not None:
                # same partitions: a new fingerprint and anchor (and the row of
                # the content's HISTORY row, which moves when earlier HISTORY
                # rows are removed), no HISTORY row
                row = dict(
                    row,
                    FINGERPRINT=fingerprint,
                    HISTORY_NROWS_AT_BUILD=n_history,
                    HISTORY_ROW=history_row,
                )
                row["CHECKSUM"] = row_checksum(row)
                for name in (
                    "FINGERPRINT",
                    "HISTORY_NROWS_AT_BUILD",
                    "HISTORY_ROW",
                    "CHECKSUM",
                ):
                    table.putcell(name, indices[0], row[name])
                _remove_rows(table, indices[0], indices[1:])
                table.flush()
                return "revalidated", None
        cache_id = uuid.uuid4().hex
        history_row = _append_history_row(
            path, partition_scheme, partitions, runs, cache_id, reason
        )
        values = encode_row(
            partitions,
            runs,
            partition_scheme,
            fingerprint,
            history_row,
            n_history,
            cache_id=cache_id,
        )
        if indices:
            index = indices[0]
        else:
            index = table.nrows()
            table.addrows(1)
        _put_row_cells(table, index, values)
        _remove_rows(table, index, indices[1:])
        table.flush()
    return "stored", history_row


def store_partitions(
    path: str,
    partition_scheme: list[str],
    partitions: list[dict],
    runs: MainRowRuns,
    fingerprint: str,
    n_history: int,
    reason: str = "first",
    rebuild: bool = False,
) -> tuple[str, int | None]:
    """
    Store the partitions of a scheme in an MS (python-casacore; the caller
    checks why_not_writable first). The write protocol, in the MS's writer
    mutex:

    0. remove the temporary sub-tables of writers that are gone;
    1-2. under the MAIN write lock (one attempt): create the sub-table if
       missing (temporary name, then renamed into place), then the MAIN
       keyword if missing;
    3. under the sub-table's write lock (one attempt), read the scheme's row
       again: (a) the same partitions, fingerprint and a valid HISTORY anchor
       (another writer stored them meanwhile): "hit-race"; (b) the same
       partitions: a new fingerprint and anchor, no HISTORY row:
       "revalidated"; (c) otherwise a HISTORY row (under its write lock), then
       the row (CHECKSUM last), replacing the old one; the oldest rows beyond
       MAX_STORED_ROWS are removed: "stored".

    Parameters
    ----------
    path : str
        Path of the MS.
    partition_scheme : list[str]
        The partition scheme (validated).
    partitions : list[dict]
        Their descriptions.
    runs : MainRowRuns
        Their MAIN rows.
    fingerprint : str
        fingerprint_json of the MS, taken before they were computed.
    n_history : int
        The HISTORY rows taken with the fingerprint (the anchor).
    reason : str, optional
        Of the HISTORY row: "first", "stale:<what>" or "rebuild".
    rebuild : bool, optional
        Never "hit-race" (mode "rebuild").

    Returns
    -------
    tuple[str, int | None]
        The status, and the HISTORY row added ("stored") or None.

    Raises
    ------
    CacheNotStored
        A lock held by another process ("locked"), or a sub-table of another
        format.
    """
    with write_mutex(path):
        remove_stale_tmp_tables(path)
        _ensure_linked_subtable(path)
        return _write_row(
            path,
            partition_scheme,
            partitions,
            runs,
            fingerprint,
            n_history,
            reason,
            rebuild,
        )


def _remove_link_keyword(main_tb, path: str, only_dangling: bool) -> bool:
    """Remove MAIN's keyword XRADIO_PARTITIONS if it links the sub-table (and
    the sub-table is gone, with ``only_dangling``); the caller flushes.
    Returns whether it was removed."""
    if SUBTABLE_NAME not in main_tb.keywordnames():
        return False
    if not _is_subtable_link(main_tb.getkeyword(SUBTABLE_NAME)):
        return False
    if only_dangling and os.path.isfile(os.path.join(subtable_path(path), "table.dat")):
        return False
    main_tb.removekeyword(SUBTABLE_NAME)
    return True


def repair_dangling_keyword(path: str) -> bool:
    """
    Remove the MAIN keyword XRADIO_PARTITIONS if its sub-table is gone (a
    dangling keyword breaks CASA's mstransform), under the writer mutex and
    the MAIN write lock (one attempt). Returns whether it was removed.
    """
    with write_mutex(path):
        try:
            with _locked_for_update(path) as main_tb:
                return _write_main_keywords(
                    main_tb, lambda tb: _remove_link_keyword(tb, path, True)
                )
        except CacheNotStored:
            xradio_logger().debug(
                f"The dangling keyword {SUBTABLE_NAME} of {path} is not removed: "
                "MAIN is locked"
            )
    return False


def _check_own_subtable(subtable: str) -> None:
    """Raise ValueError if the table at ``subtable`` is not a partition cache
    of xradio (its keyword CREATOR), or not a readable table."""
    try:
        with open_table_ro(subtable) as table:
            version = subtable_format(table)
    except Exception as exc:
        raise ValueError(
            f"{subtable} is not a readable table ({type(exc).__name__}: {exc}): it is "
            "not removed"
        ) from exc
    if version is None:
        raise ValueError(
            f"{subtable} is not a partition cache of xradio (no keyword CREATOR "
            "'xradio'): it is not removed"
        )


def remove_partition_cache(path: str) -> bool:
    """
    Remove the stored partitions of an MS: the MAIN keyword first, then the
    sub-table (so that no keyword is left without its sub-table), and the
    temporary sub-tables of gone writers. The sub-table is removed only if
    it is xradio's (its keyword CREATOR), and while its write lock, which
    writers hold while they store a row, could be taken (one attempt); the
    MAIN keyword is removed under the MAIN write lock (one attempt).

    Parameters
    ----------
    path : str
        Path of the MS.

    Returns
    -------
    bool
        Whether anything was removed.

    Raises
    ------
    FileNotFoundError
        If there is no MAIN table at ``path``.
    PermissionError
        If the MS cannot be written.
    ValueError
        If ``<path>/XRADIO_PARTITIONS`` is not a partition cache of xradio
        (nothing is removed).
    RuntimeError
        If another process holds a lock on MAIN or on the sub-table (nothing
        is removed).
    """
    path = os.path.abspath(os.path.expanduser(os.fspath(path)))
    if not os.path.isfile(os.path.join(path, "table.dat")):
        raise FileNotFoundError(f"No MeasurementSet at {path}")
    with write_mutex(path), casatools_serialized():
        state = link_state(path)
        subtable = subtable_path(path)
        has_table = os.path.lexists(subtable)
        if state in ("absent", "foreign") and not has_table:
            return False
        if not (
            os.access(path, os.W_OK | os.X_OK)
            and os.access(os.path.join(path, "table.dat"), os.W_OK)
        ):
            raise PermissionError(
                f"{path} is not writable: its partition cache cannot be removed"
            )
        if has_table:
            _check_own_subtable(subtable)
        with contextlib.ExitStack() as stack:
            try:
                if has_table:
                    # (held until the keyword is gone: no writer stores a row)
                    stack.enter_context(_locked_for_update(subtable))
                if state in ("linked", "dangling"):
                    with _locked_for_update(path) as main_tb:
                        _write_main_keywords(
                            main_tb, lambda tb: _remove_link_keyword(tb, path, False)
                        )
            except CacheNotStored:
                raise RuntimeError(
                    f"Another process has a lock on the MAIN table of {path} or on "
                    f"its {SUBTABLE_NAME} sub-table: its partition cache was not "
                    "removed"
                ) from None
        if has_table:
            shutil.rmtree(subtable)
        remove_stale_tmp_tables(path)
    PARTITIONS_MEMO.discard_path(path)
    return True


# --- load ------------------------------------------------------------------------------


def _compute(path: str, partition_scheme: list[str]) -> tuple[list[dict], MainRowRuns]:
    with casatools_serialized():  # (a no-op with python-casacore)
        return create_partitions_with_main_rows(path, partition_scheme)


def _ms_state(path: str) -> tuple[str | None, int | None]:
    """(fingerprint JSON, HISTORY rows) of an MS, or (None, None) if the
    fingerprint cannot be computed (then neither stored rows nor the memo
    are used)."""
    try:
        return fingerprint_json(path), history_nrows(path)
    except Exception as exc:
        if _first_notice(path, "no-fingerprint"):
            xradio_logger().info(
                f"The fingerprint of {path} could not be computed "
                f"({type(exc).__name__}: {exc}): its partitions are computed in "
                "memory"
            )
        return None, None


def _history_reason(reason: str) -> str:
    """The reason of the HISTORY row of partitions stored because the stored
    row was not used for ``reason`` (check_stored_row)."""
    if reason in ("absent", "unlinked", "dangling", "no row"):
        return "first"
    if reason == "fingerprint":
        return "stale:fingerprint"
    if reason.startswith("history"):
        return "stale:history"
    return "stale:corrupt"


def _store(
    path: str,
    partition_scheme: list[str],
    entry: _MemoEntry,
    n_history: int,
    reason: str,
    rebuild: bool,
) -> str:
    """Store a computed result if the MS can be written (else notify why
    not); returns the status. In the MS's write mutex."""
    try:
        with write_mutex(path):
            not_writable = why_not_writable(path, entry.runs)
            if not_writable is not None:
                notify_not_stored(path, *not_writable)
                return f"memory:{not_writable[0]}"
            status, history_row = store_partitions(
                path,
                partition_scheme,
                entry.partitions,
                entry.runs,
                entry.fingerprint,
                n_history,
                reason,
                rebuild,
            )
    except CacheNotStored as exc:
        notify_not_stored(path, exc.reason, exc.warn)
        return f"memory:{exc.reason}"
    except Exception as exc:
        xradio_logger().debug(
            f"Storing the partitions of {path} failed:\n{traceback.format_exc()}"
        )
        notify_not_stored(path, "write failed", True, f"{type(exc).__name__}: {exc}")
        return "memory:write failed"
    if history_row == n_history and history_nrows(path) == n_history + 1:
        # only the row of the stored content was added: the memo entry stays
        # valid (as the stored row: the HISTORY rule holds)
        entry.history_nrows = n_history + 1
    return status


def changed_since(path: str, result: PartitionsResult) -> bool:
    """
    Whether an MS changed since partitions from the memo or a stored row
    were validated: its fingerprint or number of HISTORY rows is another
    now (a write made while the MS was opened), or cannot be computed. A
    result without them (computed) counts as changed.
    """
    if result.fingerprint is None:
        return True
    with write_mutex(path):
        fingerprint, n_history = _ms_state(path)
    return fingerprint != result.fingerprint or n_history != result.history_nrows


def compute_in_memory(
    path: str, partition_scheme: list[str], reason: str
) -> PartitionsResult:
    """
    The partitions of an MS computed from what this process sees of it,
    neither taken from the memo or the stored rows nor kept there (status
    "memory:<reason>", logged at INFO once per MS and reason): for a MAIN
    table whose write lock this process holds, with another number of rows
    than its files (rows not flushed yet), which the fingerprint of the
    files does not describe.
    """
    partition_scheme = validate_partition_scheme(partition_scheme)
    notify_not_stored(path, reason, False)
    partitions, runs = _compute(path, partition_scheme)
    return PartitionsResult(partitions, runs, "fresh", f"memory:{reason}")


def load_or_create_partitions(
    path: str, partition_scheme: list[str], mode: str
) -> PartitionsResult:
    """
    The partitions of an MS and their MAIN rows (copies, which the caller may
    modify).

    Parameters
    ----------
    path : str
        Absolute path of the MS.
    partition_scheme : list[str]
        The partition scheme (see validate_partition_scheme).
    mode : str
        One of PARTITION_CACHE_MODES (see the module docstring).

    Returns
    -------
    PartitionsResult
        The partitions, their rows and where they come from.
    """
    if mode not in PARTITION_CACHE_MODES:
        raise ValueError(f"Unknown partition cache mode {mode!r}")
    partition_scheme = validate_partition_scheme(partition_scheme)
    if mode == "off":
        partitions, runs = _compute(path, partition_scheme)
        return PartitionsResult(partitions, runs, "fresh", "memory:mode-off")
    key = memo_key(path, partition_scheme)
    with PARTITIONS_MEMO.build_lock(key):
        with write_mutex(path):
            found = _valid_partitions(path, partition_scheme, key, mode)
        fingerprint, n_history, result, reason = found
        if result is not None:
            return result
        if fingerprint is None:
            partitions, runs = _compute(path, partition_scheme)
            return PartitionsResult(partitions, runs, "fresh", "memory:no-fingerprint")
        PARTITIONS_MEMO.stats["computed"] += 1
        partitions, runs = _compute(path, partition_scheme)
        with write_mutex(path):
            after, _ = _ms_state(path)
        if after != fingerprint or runs.main_nrows != _fingerprint_nrows(fingerprint):
            return PartitionsResult(
                partitions, runs, "fresh", "memory:changed-during-build"
            )
        entry = _MemoEntry(fingerprint, n_history, partitions, runs)
        PARTITIONS_MEMO.put(key, entry)
        if mode == "read":
            return entry.result("fresh", "memory:mode-read")
        status = _store(
            path, partition_scheme, entry, n_history, reason, mode == "rebuild"
        )
        return entry.result("fresh", status)


def _valid_partitions(
    path: str, partition_scheme: list[str], key: tuple, mode: str
) -> tuple[str | None, int | None, PartitionsResult | None, str]:
    """
    The fingerprint and HISTORY rows of an MS, and its partitions from the
    memo or the stored row if they are valid (else the reason of the HISTORY
    row of a store); a dangling MAIN keyword is removed ("auto").

    Returns
    -------
    tuple[str | None, int | None, PartitionsResult | None, str]
        (fingerprint, HISTORY rows, result or None, reason).
    """
    fingerprint, n_history = _ms_state(path)
    if fingerprint is None or mode == "rebuild":
        return fingerprint, n_history, None, "rebuild"
    entry = PARTITIONS_MEMO.get(key)
    if (
        entry is not None
        and entry.fingerprint == fingerprint
        and entry.history_nrows == n_history
    ):
        PARTITIONS_MEMO.stats["hits"] += 1
        return fingerprint, n_history, entry.result("memo", "hit-memory"), ""
    lookup = lookup_stored_row(path, key[1])
    stored, why = check_stored_row(
        path, lookup, partition_scheme, fingerprint, n_history
    )
    if stored is not None:
        PARTITIONS_MEMO.stats["stored hits"] += 1
        entry = _MemoEntry(fingerprint, n_history, *stored)
        PARTITIONS_MEMO.put(key, entry)
        return fingerprint, n_history, entry.result("stored", "hit"), ""
    xradio_logger().debug(
        f"The stored partitions of {path} for partition_scheme "
        f"{partition_scheme} are not used: {why} {lookup.detail}".rstrip()
    )
    if lookup.state == "dangling" and mode == "auto" and why_not_writable(path) is None:
        repair_dangling_keyword(path)
    return fingerprint, n_history, None, _history_reason(why)
