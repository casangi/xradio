"""
The partitions of an MSv2 for the MSv2 xarray backend (engine
``xradio_msv2``): computed with ``create_partitions_with_main_rows``, stored
inside the MS (the sub-table ``XRADIO_PARTITIONS``, linked from MAIN by the
keyword ``XRADIO_PARTITIONS``, with a HISTORY row per stored content), and
reused while the MS is unchanged. Validated results are also kept in a
per-process memo.

Modes (``partition_cache``, default ``$XRADIO_MSV2_PARTITION_CACHE`` or
"auto"):

- "auto" and "read": a valid memo entry or stored row is used, otherwise the
  partitions are computed (and memoised);
- "rebuild": the partitions are computed again (and memoised);
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

A stale row is never parsed (L1 and L2 come first). A memo entry is valid
while the fingerprint and the number of HISTORY rows are those it was made
with. The memo hands out copies, holds at most ``MEMO_MAX_ENTRIES`` results
and ``MEMO_MAX_BYTES`` of row runs, computes a key once when several threads
open the same MS, and is emptied in a fork child.
"""

import collections
import copy
import dataclasses
import hashlib
import json
import operator
import os
import threading
import time
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from xradio._utils._casacore.tables import casatools_serialized
from xradio._utils.logging import xradio_logger
from xradio.measurement_set._utils._msv2._tables.table_lock_file import (
    history_nrows,
    ms_fingerprint,
)
from xradio.measurement_set._utils._msv2._tables.table_query import open_table_ro
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
    import uuid

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
            # (a table object of this process that has not seen new rows)
            table.resync()
        if table.nrows() < n_history:
            return "HISTORY changed while it was read"
        ours = (
            table.getcell("APPLICATION", history_row) == HISTORY_APPLICATION
            and table.getcell("ORIGIN", history_row) == HISTORY_ORIGIN
            and f"cache_id={row['CACHE_ID']}"
            in _app_params(table.getcell("APP_PARAMS", history_row))
        )
        if not ours:
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
    if not (
        isinstance(keyword, str)
        and keyword.startswith("Table: ")
        and keyword.rstrip("/").endswith("/" + SUBTABLE_NAME)
    ):
        return "foreign"
    return "linked" if exists else "dangling"


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
        How they were obtained: "hit" (stored row), "hit-memory" (memo), or
        "memory:<reason>" (computed, not stored): "computed", "mode-off",
        "mode-read", "no-fingerprint" (the fingerprint of the MS could not be
        computed: neither stored rows nor the memo are used),
        "changed-during-build" (the MS changed while they were computed:
        not memoised).
    """

    partitions: list[dict]
    runs: MainRowRuns
    source: str
    status: str

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
            copy.deepcopy(self.partitions), _copy_runs(self.runs), source, status
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
        fingerprint, n_history = _ms_state(path)
        if fingerprint is None:
            partitions, runs = _compute(path, partition_scheme)
            return PartitionsResult(partitions, runs, "fresh", "memory:no-fingerprint")
        if mode != "rebuild":
            entry = PARTITIONS_MEMO.get(key)
            if (
                entry is not None
                and entry.fingerprint == fingerprint
                and entry.history_nrows == n_history
            ):
                PARTITIONS_MEMO.stats["hits"] += 1
                return entry.result("memo", "hit-memory")
            lookup = lookup_stored_row(path, key[1])
            stored, reason = check_stored_row(
                path, lookup, partition_scheme, fingerprint, n_history
            )
            if stored is not None:
                PARTITIONS_MEMO.stats["stored hits"] += 1
                entry = _MemoEntry(fingerprint, n_history, *stored)
                PARTITIONS_MEMO.put(key, entry)
                return entry.result("stored", "hit")
            xradio_logger().debug(
                f"The stored partitions of {path} for partition_scheme "
                f"{partition_scheme} are not used: {reason} {lookup.detail}".rstrip()
            )
        PARTITIONS_MEMO.stats["computed"] += 1
        partitions, runs = _compute(path, partition_scheme)
        after, _ = _ms_state(path)
        if after != fingerprint or runs.main_nrows != _fingerprint_nrows(fingerprint):
            return PartitionsResult(
                partitions, runs, "fresh", "memory:changed-during-build"
            )
        entry = _MemoEntry(fingerprint, n_history, partitions, runs)
        PARTITIONS_MEMO.put(key, entry)
        status = "memory:mode-read" if mode == "read" else "memory:computed"
        return entry.result("fresh", status)
