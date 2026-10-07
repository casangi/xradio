"""
The lazy pointing_xds of the MSv2 xarray backend (engine ``xradio_msv2``).

The converter builds the pointing_xds of a partition from the POINTING rows
of the partition's antennas in its (projected) time range, pivoted to a
(time, antenna) grid (``read_pointing.py``). Its data variables
(POINTING_BEAM, POINTING_DISH_MEASURED, POINTING_OVER_THE_TOP) grow with
POINTING: a dense grid of up to about 1.8 GB per partition on VLASS. Here,
when an MSv2 is opened, only the grid is defined; the values are read when
the variables are indexed or computed:

- At open (``deferred_pointing_generic_xds``, the ``generic_loader`` of
  ``create_pointing_xds``): the POINTING index (TIME, ANTENNA_ID and row
  numbers sorted by TIME, 16 bytes per row, whatever the number of rows:
  ``read_pointing_index``) gives the partition's rows, its unique times
  (``time_pointing``) and antennas (``antenna_name``). The data variables
  are placeholders, described by a :class:`DeferredPointingVariable`, and
  the rest of the pointing_xds (dimension names, attributes, encoding) is
  built by the converter's own code. No data column is read but one cell
  each (its dtype); the cell shapes come from the column descriptions (a
  fixed shape) or from the shapes of the cells (``getcolshapestring``,
  read in bounded chunks: no values). The index is kept in a per-process
  memo while the table's fingerprint (``table_fingerprint``) is unchanged.
  The scan of the cell shapes (about 0.2 s per column for 550,000 rows) is
  stored in the MS (a row of its XRADIO_PARTITIONS sub-table,
  ``partition_cache.store_pointing_shapes``) by the ``partition_cache``
  modes that store partitions, with the fingerprint of the POINTING table
  it was made from: later opens (in any process) take it from there while
  the fingerprint is unchanged (``open_pointing_index``).
- On access (:class:`PointingColumnArray`, outer indexing): the rows of the
  selected times and antennas, checked against the index (their TIME and
  ANTENNA_ID), the first row (lowest row number) of every (time, antenna)
  cell, their values read in ascending row order, and the pivot with
  ``pivot_time_antenna`` (xarray's promoted fill value for missing cells).
  These are the selection, deduplication and pivot of the converter's
  cached path (``pointing_generic_xds``), restricted to the selection, so
  the values are those of the converter's pointing_xds. Reads are done in
  time sub-blocks of bounded size; every read opens and closes the table
  and, with casatools, holds the process-wide casatools lock (where
  fragmented rows are read in windows of consecutive rows: the shim has no
  row selection; a window never spans a cell of another shape).

Cells of more than one shape: an array column whose description does not
fix the shape of its cells has the shape of most of its cells; the rows
whose cells have another shape, or no value, are recorded in the index
with their shapes (``ColumnShapes``; ``odd_rows``: usually none). The
converter reads the rows of each partition on their own
(``load_generic_table``), so the pointing_xds of a partition whose POINTING
rows include such rows follows from the shapes of its cells
(``partition_layout``, no values): with 1,000 rows or more the converter
reads every column at once and leaves out a column whose cells vary there;
the others have one shape (maybe not that of most cells) and are read
lazily as above. With fewer rows it stacks the rows and pads a column whose
cells vary (or leaves it out, where a cell does not fit): the data
variables are then :class:`PointingBuildArray`, which build the
pointing_xds by the converter's code when read (one build per partition is
kept in a bounded per-process memo, so that its variables share it; fewer
than 1,000 rows). Either way nothing but the index is read at open, and an
error of the converter's build (e.g. DIRECTION cells of two polynomial
terms) is raised at open, as by the converter. Tables that cannot be
described from their column descriptions, cell shapes and index (no
DIRECTION column, unusual value types, a column without a cell of a usual
shape, TIME values that are not finite, ...), and the rare partitions that
``partition_layout`` does not describe (cells of zero size, or of several
shapes in a column of unusual values), have their pointing_xds built when
the MS is opened by the converter's code, as by the converter, with
:class:`PointingBuildArray` variables (the values are not kept). With
``pointing_interpolate=True`` (the interpolation needs the values) the
pointing_xds is built eagerly.

Staleness: an array records the number of POINTING rows and a token of its
partition's selection (times, antennas, row numbers). A read finds the
index in the memo (in this process, the index of the open) or reads it
again (other processes); a POINTING table with another number of rows, or
a selection with another token, raises :class:`MSv2ChangedError`. A read
whose rows no longer have the TIME and ANTENNA_ID of the index (rewritten in
place) reads the index again, as another process would: the outcome does
not depend on the memo; a read whose cells no longer have the shape of the
open raises MSv2ChangedError. (The rows are checked one by one only when
the table's fingerprint changed since the open, or a handle of the process
has it open for writing.) A pointing_xds built again on read must have the
coordinates (times, antennas) of the one opened, and the shapes and
dtypes of its variables, else MSv2ChangedError. Every read also checks that
the partition's MAIN rows, whose time range and antennas select its
POINTING rows, are those of the open (``PartitionIndex.verify_current``).
"""

import collections
import copy
import dataclasses
import hashlib
import json
import os
import threading
from collections.abc import Callable, Hashable, Mapping
from typing import Any

import numpy as np
import xarray as xr
from numpy.typing import DTypeLike

from xradio._utils._casacore.tables import casatools_serialized, uses_casatools
from xradio._utils.logging import xradio_logger
from xradio.measurement_set._utils._msv2 import partition_cache
from xradio.measurement_set._utils._msv2._tables.read import (
    add_units_measures,
    convert_casacore_time,
    extract_table_attributes,
    find_best_col_loader,
    find_loadable_cols,
    is_nested_ms,
    load_fixed_size_cols,
    projection_tolerance,
)
from xradio.measurement_set._utils._msv2._tables.read_pointing import (
    _SCALAR_VALUE_TYPES,
    POINTING_TABLE,
    PointingColumns,
    generic_dims,
    maybe_promote,
    pivot_time_antenna,
    select_pointing_rows,
)
from xradio.measurement_set._utils._msv2._tables.read_rows import (
    CASACORE_TO_NUMPY_DTYPE,
    FRAGMENTED_RUNS,
    column_dtype,
    has_in_place_reads,
    parse_shape_string,
    read_column_rows,
    read_row_range,
    rows_to_runs,
)
from xradio.measurement_set._utils._msv2._tables.subtable_cache import (
    active_subtable_cache,
)
from xradio.measurement_set._utils._msv2._tables.table_lock_file import (
    table_fingerprint,
    table_type,
)
from xradio.measurement_set._utils._msv2._tables.table_query import open_table_ro
from xradio.measurement_set._utils._msv2.backend_arrays import (
    MSv2BackendArray,
    PartitionIndex,
    _IndexMemo,
    bounding_slices,
    is_range,
    take_from_block,
)
from xradio.measurement_set._utils._msv2.backend_errors import (
    MSv2ChangedError,
    MSv2ReadError,
)
from xradio.measurement_set._utils._msv2.msv4_sub_xdss import create_pointing_xds
from xradio.measurement_set._utils._msv2.stream_write import deferred_placeholder

# Bound of the per-process memo of POINTING indices (16 bytes per POINTING
# row). The most recently used index is always kept.
POINTING_INDEX_MEMO_MAX_BYTES = 256 * 2**20
# Bound of the per-process memo of the rows of partitions (about 8 bytes per
# selected POINTING row). The most recently used selection is always kept.
POINTING_SELECTION_MEMO_MAX_BYTES = 128 * 2**20
# Largest time sub-block of a read: the values read, the grid (with its
# promoted dtype) and the per-row index arrays of the sub-block, in bytes.
POINTING_SUB_BLOCK_BYTES = 128 * 2**20
# Bytes of the per-row index arrays of a read (positions, antenna and time
# indices, keys, row numbers, orders), per row
_ROW_INDEX_BYTES = 64
# Tables without in-place reads (the casatools shim) are read with one getcol
# per run of consecutive rows. Rows in more runs than FRAGMENTED_RUNS (e.g.
# one antenna of a table ordered by time) are read there in windows of
# consecutive rows, gaps included: a window ends at a gap of more than
# _WINDOW_MAX_GAP rows, or at _WINDOW_BYTES of cells.
_WINDOW_MAX_GAP = 1024
_WINDOW_BYTES = 16 * 2**20
# Rows of a column whose cell shapes are read in one call when the index is
# read (getcolshapestring: no values; about 60 bytes per row while it lasts)
SHAPE_SCAN_ROWS = 2**18
# A scan of cell shapes that meets more calls failing on cells without a
# value than this gives up: the table is then built by the converter's code
SHAPE_SCAN_MAX_ERRORS = 64
# Version of the scan of the cell shapes of the POINTING array columns
# (_scan_cell_shapes, ColumnShapes): the ALGORITHM_VERSION of its row in the
# MS's XRADIO_PARTITIONS sub-table (partition_cache.POINTING_SHAPES_KEY)
POINTING_SHAPES_VERSION = 1
# Cell shapes holding more rows of other shapes than this (in all columns)
# are not stored in the MS (scanned on every open)
POINTING_SHAPES_MAX_STORED = 2**16
# Bound of the per-process memo of pointing_xds built again on read
# (PointingBuildArray: the variables of a partition share one build). A
# build larger than this is not kept.
POINTING_BUILD_MEMO_MAX_BYTES = 128 * 2**20
# Value dtypes whose round trip through xarray's promoted fill dtype (for
# grids with missing cells) keeps every value: a block pivoted with or without
# missing cells gives the values of the whole grid. (POINTING data columns
# have one of these types, see read_pointing_index.)
_EXACT_PROMOTION_KINDS = "bfc"
_EXACT_PROMOTION_INTS = (np.dtype(np.int8), np.dtype(np.int16), np.dtype(np.int32))


@dataclasses.dataclass(frozen=True)
class ColumnShapes:
    """
    The cell shapes of an array column whose description does not fix them,
    as its scan found them (``_scan_cell_shapes``: the shapes of all cells,
    not their values).

    Attributes
    ----------
    shape : tuple[int, ...] | None
        The shape of most cells (numpy order; ties: the shape seen first),
        None if no cell has a value.
    first : int
        The first row with ``shape`` (-1 if None).
    rows : np.ndarray
        The rows whose cells have another shape or no value (int64,
        ascending; usually none; none either if ``shape`` is None).
    codes : np.ndarray
        The shape of every row of ``rows``: its index in ``others`` (int32).
    others : tuple[tuple[int, ...] | None, ...]
        The other shapes (None: no value).
    """

    shape: tuple[int, ...] | None
    first: int
    rows: np.ndarray
    codes: np.ndarray
    others: tuple[tuple[int, ...] | None, ...]

    @property
    def nbytes(self) -> int:
        return self.rows.nbytes + self.codes.nbytes

    def shapes_among(self, rows: np.ndarray) -> set[tuple[int, ...] | None]:
        """The shapes of the cells of ``rows`` (distinct row numbers; None:
        a cell without a value)."""
        if self.shape is None:
            return {None} if np.asarray(rows).size else set()
        hit = np.isin(self.rows, rows) if self.rows.size else np.zeros(0, bool)
        n_odd = int(np.count_nonzero(hit))
        found = {self.others[code] for code in np.unique(self.codes[hit]).tolist()}
        if np.asarray(rows).size > n_odd:
            found.add(self.shape)
        return found

    def window_breaks(self, rows: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
        """
        For ascending ``rows`` whose cells have ``shape``: whether a row
        between ``rows[i]`` and ``rows[i + 1]`` has a cell of another shape
        (or no value), for every i (a read of the consecutive rows between
        them would fail).
        """
        if rows.size < 2:
            return np.zeros(0, dtype=bool)
        lo, hi = rows[:-1], rows[1:]
        if shape == self.shape:
            # the rows of other shapes are those of self.rows
            between = np.searchsorted(self.rows, hi, "left") - np.searchsorted(
                self.rows, lo, "right"
            )
            return between > 0
        same = [i for i, other in enumerate(self.others) if other == shape]
        rows_same = self.rows[np.isin(self.codes, same)]
        between = np.searchsorted(rows_same, hi, "left") - np.searchsorted(
            rows_same, lo, "right"
        )
        return between != (hi - lo - 1)

    def to_json(self) -> dict:
        return {
            "shape": None if self.shape is None else list(self.shape),
            "first": int(self.first),
            "rows": self.rows.tolist(),
            "codes": self.codes.tolist(),
            "others": [None if s is None else list(s) for s in self.others],
        }

    @classmethod
    def from_json(cls, content: Any, nrows: int) -> "ColumnShapes":
        """The shapes of ``to_json``, checked against the table's rows
        (ValueError, TypeError or KeyError if they do not fit)."""

        def shape_of(value: Any) -> tuple[int, ...]:
            if not isinstance(value, list) or not value:
                raise ValueError(f"not a cell shape: {value!r}")
            if not all(type(n) is int and n >= 0 for n in value):
                raise ValueError(f"not a cell shape: {value!r}")
            return tuple(value)

        shape = None if content["shape"] is None else shape_of(content["shape"])
        others = tuple(None if s is None else shape_of(s) for s in content["others"])
        if not all(
            isinstance(values, list) and all(type(n) is int for n in values)
            for values in (content["rows"], content["codes"])
        ):
            raise ValueError("rows or codes that are not lists of integers")
        rows = np.asarray(content["rows"], dtype=np.int64)
        codes = np.asarray(content["codes"], dtype=np.int32)
        first = content["first"]
        if type(first) is not int or rows.ndim != 1 or codes.shape != rows.shape:
            raise ValueError("rows, codes or first of another kind")
        if rows.size and (
            rows[0] < 0 or rows[-1] >= nrows or np.any(np.diff(rows) <= 0)
        ):
            raise ValueError("rows that are not ascending rows of the table")
        if codes.size and (codes.min() < 0 or codes.max() >= len(others)):
            raise ValueError("codes out of range")
        if shape is None:
            if first != -1 or rows.size or others:
                raise ValueError("a column without values has other cells")
        elif not 0 <= first < nrows or first in set(rows.tolist()):
            raise ValueError("first is no row of the shape")
        if shape in others:
            raise ValueError("the shape of most cells among the others")
        return cls(shape, first, rows, codes, others)


@dataclasses.dataclass(frozen=True)
class PointingIndex:
    """
    The POINTING index of a table: TIME, ANTENNA_ID and the row numbers,
    sorted by TIME (``PointingColumns`` without data columns), with the dtype
    and cell shape of every data column.

    Attributes
    ----------
    columns : PointingColumns
        TIME, ANTENNA_ID and the row numbers (``data`` None), sorted by TIME.
    dtypes : dict[str, np.dtype]
        dtype of the values of every data column (as read).
    cell_shapes : dict[str, tuple[int, ...]]
        Cell shape of every data column (numpy order).
    odd_rows : np.ndarray
        The rows (ascending table row numbers) with a cell of another shape
        than its column's (or without a value) in an array column: the
        pointing_xds of a partition that selects one is described by
        ``partition_layout``.
    token : str
        Token of the table's fingerprint when the index was read.
    shapes : dict[str, ColumnShapes]
        The cell shapes of every array column whose description does not
        fix them (the scan, or the scan stored in the MS).
    layout : tuple[tuple[str, bool, tuple[int, ...] | None, str], ...]
        Every column a partition's read of POINTING loads (the converter's
        ``load_generic_table``: not SOURCE_MODEL, no record columns), in
        load order: (name, is a coordinate, cell shape (None: in
        ``shapes``), value type).
    row_dtypes : dict[str, np.dtype]
        dtype of the values of a table row (``table.row``) of every array
        data column (only for tables with ``odd_rows``).
    shapes_source : str
        "scan" (the shapes were scanned) or "stored" (from the MS).
    """

    columns: PointingColumns
    dtypes: dict[str, np.dtype]
    cell_shapes: dict[str, tuple[int, ...]]
    odd_rows: np.ndarray
    token: str
    shapes: dict[str, ColumnShapes] = dataclasses.field(default_factory=dict)
    layout: tuple = ()
    row_dtypes: dict[str, np.dtype] = dataclasses.field(default_factory=dict)
    shapes_source: str = "scan"

    @property
    def nrows(self) -> int:
        """Rows of the POINTING table."""
        return int(self.columns.time.size)

    @property
    def nbytes(self) -> int:
        return (
            self.columns.nbytes
            + self.odd_rows.nbytes
            + sum(shapes.nbytes for shapes in self.shapes.values())
        )

    def cells_of(
        self, col: str, rows: np.ndarray
    ) -> set[tuple[int, ...] | None] | None:
        """The shapes of the cells of ``rows`` of an array column (None: no
        value), or None for a column whose description fixes them."""
        shapes = self.shapes.get(col)
        return None if shapes is None else shapes.shapes_among(rows)

    def selects_odd_rows(self, selection: "_Selection") -> bool:
        """Whether a partition's rows include rows of ``odd_rows``."""
        if self.odd_rows.size == 0:
            return False
        rows = self.columns.row[selection.sel]
        return bool(np.isin(rows, self.odd_rows).any())


class _IndexEntry:
    """A memo entry: the index, or None for a table read eagerly."""

    __slots__ = ("index", "nbytes")

    def __init__(self, index: PointingIndex | None):
        self.index = index
        self.nbytes = 0 if index is None else index.nbytes


class _Selection:
    """
    The POINTING rows of a partition, by time: the positions of the rows in
    the index (``sel``, in time order), the antenna index of every row
    (``acode``, into the partition's unique antennas) and, for every time
    index t, the first position of its rows (``time_starts``, n_times + 1).
    """

    __slots__ = ("sel", "acode", "time_starts", "nbytes")

    def __init__(self, sel: np.ndarray, acode: np.ndarray, time_starts: np.ndarray):
        # (sel is ascending: its last position is the largest)
        small = sel.size == 0 or int(sel[-1]) < 2**31
        self.sel = np.asarray(sel, dtype=np.int32 if small else np.int64)
        self.acode = np.asarray(acode, dtype=np.int32)
        self.time_starts = np.asarray(time_starts, dtype=np.int64)
        self.nbytes = self.sel.nbytes + self.acode.nbytes + self.time_starts.nbytes


POINTING_INDEX_MEMO = _IndexMemo(POINTING_INDEX_MEMO_MAX_BYTES)
POINTING_SELECTION_MEMO = _IndexMemo(POINTING_SELECTION_MEMO_MAX_BYTES)

if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=POINTING_INDEX_MEMO.reset_after_fork)
    os.register_at_fork(after_in_child=POINTING_SELECTION_MEMO.reset_after_fork)


def clear_pointing_memos() -> None:
    """Empty the per-process memos of POINTING indices, selections and
    builds (for tests)."""
    POINTING_INDEX_MEMO.clear()
    POINTING_SELECTION_MEMO.clear()
    POINTING_BUILD_MEMO.clear()


def _digest(*arrays: np.ndarray) -> str:
    digest = hashlib.blake2b(digest_size=16)
    for values in arrays:
        values = np.ascontiguousarray(values)
        digest.update(str(values.dtype).encode())
        digest.update(np.int64(values.size).tobytes())
        digest.update(values.tobytes())
    return digest.hexdigest()


def _fingerprint_token(fingerprint: dict) -> str:
    text = json.dumps(fingerprint, sort_keys=True, default=str)
    return hashlib.blake2b(text.encode(), digest_size=16).hexdigest()


def pointing_table_token(table_path: str) -> str | None:
    """
    Token of the fingerprint of a POINTING table (``table_fingerprint``:
    rows, columns, data managers, their change counters and files), None if
    there is no table.
    """
    fingerprint = table_fingerprint(table_path)
    if fingerprint is None:
        return None
    return _fingerprint_token(fingerprint)


def _fingerprint_json(fingerprint: dict) -> str:
    return json.dumps(fingerprint, sort_keys=True, separators=(",", ":"))


def _follows_writes(table_path: str, fingerprint: dict) -> bool:
    """
    Whether the fingerprint of a POINTING table tells every write of its
    cells: a plain table (not a reference or concatenated one, whose rows
    live in other tables) whose columns are all held by data managers with
    files of their own (``table.f<SEQNR>*``, in the fingerprint). Only then
    are its cell shapes stored in, and taken from, the MS.
    """
    if table_type(table_path) != "PlainTable":
        return False
    files = fingerprint.get("files") or {}
    for _, _, seqnr, columns in fingerprint.get("dms", []):
        if not columns:
            continue
        prefix = f"table.f{seqnr}"
        if seqnr is None or not any(
            name in (prefix, prefix + "i") or name.startswith(prefix + "_")
            for name in files
        ):
            return False
    return True


def encode_pointing_shapes(
    shapes: Mapping[str, ColumnShapes], nrows: int
) -> str | None:
    """
    The cell shapes of the columns of a POINTING table of ``nrows`` rows
    (``PointingIndex.shapes``) as the JSON stored in the MS, or None if they
    hold more than POINTING_SHAPES_MAX_STORED rows of other shapes (not
    stored).
    """
    if sum(col.rows.size for col in shapes.values()) > POINTING_SHAPES_MAX_STORED:
        return None
    return json.dumps(
        {
            "version": POINTING_SHAPES_VERSION,
            "nrows": int(nrows),
            "columns": {name: col.to_json() for name, col in sorted(shapes.items())},
        },
        separators=(",", ":"),
    )


def decode_pointing_shapes(text: str, nrows: int) -> dict[str, ColumnShapes]:
    """
    The cell shapes of ``encode_pointing_shapes``, checked against a table
    of ``nrows`` rows (ValueError if they are not cell shapes of such a
    table).
    """
    try:
        content = json.loads(text)
        if content["version"] != POINTING_SHAPES_VERSION or content["nrows"] != nrows:
            raise ValueError("another version or number of rows")
        columns = content["columns"]
        if not isinstance(columns, dict):
            raise TypeError("no columns")
        return {
            str(name): ColumnShapes.from_json(col, nrows)
            for name, col in columns.items()
        }
    except (TypeError, KeyError, AttributeError, OverflowError) as exc:
        raise ValueError(f"not POINTING cell shapes: {exc}") from None


def _stored_shapes(
    table_path: str, fingerprint: dict | None = None
) -> dict[str, ColumnShapes] | None:
    """
    The cell shapes of a POINTING table stored in its MS
    (``partition_cache.lookup_pointing_shapes``), if they were scanned from
    the table as it is now (the fingerprint ``fingerprint``, by default
    taken here): else None (also for a table whose fingerprint does not
    tell every write, ``_follows_writes``).
    """
    if fingerprint is None:
        fingerprint = table_fingerprint(table_path)
    if fingerprint is None or not _follows_writes(table_path, fingerprint):
        return None
    ms_path = os.path.dirname(os.path.abspath(table_path))
    found = partition_cache.lookup_pointing_shapes(ms_path, POINTING_SHAPES_VERSION)
    if found is None:
        return None
    payload, stored_fingerprint = found
    if not partition_cache.same_fingerprint(
        stored_fingerprint, _fingerprint_json(fingerprint)
    ):
        xradio_logger().debug(
            f"The POINTING cell shapes stored in {ms_path} are of another state of "
            "its POINTING table: scanned again"
        )
        return None
    try:
        shapes = decode_pointing_shapes(payload, int(fingerprint["nrows"]))
    except Exception as exc:  # (whatever the stored row holds)
        xradio_logger().debug(
            f"The POINTING cell shapes stored in {ms_path} are not used: {exc}"
        )
        return None
    POINTING_INDEX_MEMO.stats["stored shapes"] += 1
    return shapes


def _store_shapes(table_path: str, index: PointingIndex, fingerprint: dict) -> str:
    """Store the cell shapes of a scanned index in the MS of its POINTING
    table (``partition_cache.store_pointing_shapes``); returns the status
    ("stored", "hit-race", "memory:<reason>")."""
    ms_path = os.path.dirname(os.path.abspath(table_path))
    if not _follows_writes(table_path, fingerprint):
        status = "memory:POINTING writes not followed"
    else:
        payload = encode_pointing_shapes(index.shapes, index.nrows)
        if payload is None:
            status = "memory:too many cells of other shapes"
        else:
            status = partition_cache.store_pointing_shapes(
                ms_path,
                payload,
                _fingerprint_json(fingerprint),
                POINTING_SHAPES_VERSION,
            )
    POINTING_INDEX_MEMO.stats[f"shapes {status}"] += 1
    xradio_logger().debug(f"POINTING cell shapes of {ms_path}: {status}")
    return status


def pointing_unchanged(table: Any, table_path: str, token: str) -> bool:
    """
    Whether an opened POINTING table is known to be as when ``token``
    (``pointing_table_token``) was taken, so that a read need not check the
    TIME and ANTENNA_ID of its rows one by one: its lock file could be read
    and the token is unchanged (no write flushed since), and no handle of
    this process has it open for writing (``iswritable`` of a read-only
    handle).
    """
    try:
        if table.iswritable():
            return False
    except Exception:
        return False
    fingerprint = table_fingerprint(table_path)
    if fingerprint is None or not fingerprint["lock_ok"]:
        return False
    return _fingerprint_token(fingerprint) == token


def _exact_promotion(dtype: np.dtype) -> bool:
    return dtype.kind in _EXACT_PROMOTION_KINDS or dtype in _EXACT_PROMOTION_INTS


# Option bit of a casacore column description: the cells have the shape of
# the description (every cell is defined, with that shape)
_FIXED_SHAPE_OPTION = 4


def _not_lazy(table_path: str, reason: str) -> None:
    xradio_logger().debug(f"Not reading {table_path} lazily: {reason}")


def _fixed_cell_shape(table: Any, col: str) -> tuple[int, ...] | None:
    """The cell shape of an array column (numpy order) if its description
    fixes it (every cell then has it), else None. Reads no data."""
    desc = table.getcoldesc(col)
    shape = desc.get("shape")
    if (
        int(desc.get("option", 0)) & _FIXED_SHAPE_OPTION
        and shape is not None
        and len(shape)
    ):
        return tuple(int(n) for n in np.asarray(shape).ravel())
    return None


def _shape_strings(table: Any, col: str, start: int, nrow: int) -> list[str]:
    """
    The shape strings of ``nrow`` cells of a column from ``start``
    (``getcolshapestring``; raises at a cell without a value). With the
    casatools shim, casatools' own strings (casacore axis order), which the
    shim would parse one by one; parse_cell_shape reads both.
    """
    if uses_casatools():
        import casatools

        return casatools.table.getcolshapestring(table, col, start, nrow)
    return table.getcolshapestring(col, start, nrow)


def parse_cell_shape(shape_string: str) -> tuple[int, ...]:
    """A cell shape (numpy order) from a string of _shape_strings."""
    shape = parse_shape_string(shape_string)
    return shape[::-1] if uses_casatools() else shape


class _ShapeScan:
    """Codes of the cell shapes of a column's rows (-1: no value), by the
    shape strings of a scan (_scan_cell_shapes)."""

    def __init__(self, table: Any, col: str):
        self.table, self.col = table, col
        self.codes: dict[str, int] = {}
        self.errors = 0

    def fill(self, start: int, out: np.ndarray) -> None:
        """The codes of the rows ``start`` + range(out.size) into ``out``;
        a call that fails (a cell without a value) is split in two."""
        try:
            strings = _shape_strings(self.table, self.col, start, out.size)
        except Exception:
            self.errors += 1
            if self.errors > SHAPE_SCAN_MAX_ERRORS:
                raise
            if out.size == 1:
                out[0] = -1
                return
            half = out.size // 2
            self.fill(start, out[:half])
            self.fill(start + half, out[half:])
            return
        unique, inverse = np.unique(np.asarray(strings), return_inverse=True)
        codes = np.array(
            [self.codes.setdefault(str(text), len(self.codes)) for text in unique],
            dtype=np.int32,
        )
        out[:] = codes[np.asarray(inverse).ravel()]


def _scan_cell_shapes(table: Any, col: str, nrows: int) -> ColumnShapes:
    """
    The cell shapes of an array column (``ColumnShapes``): the shape of most
    cells (numpy order), the first row with it, and the rows whose cells
    have another shape or no value, with their shapes. Reads the shapes of
    the cells (``getcolshapestring``) in chunks of SHAPE_SCAN_ROWS rows, no
    values. Raises if more than SHAPE_SCAN_MAX_ERRORS calls fail (cells
    without a value: the caller describes no such table).
    """
    scan = _ShapeScan(table, col)
    codes = np.empty(nrows, dtype=np.int32)
    for start in range(0, nrows, SHAPE_SCAN_ROWS):
        scan.fill(start, codes[start : start + SHAPE_SCAN_ROWS])
    texts = {code: text for text, code in scan.codes.items()}
    defined = codes >= 0
    if not defined.any():
        empty = np.empty(0, dtype=np.int64)
        return ColumnShapes(None, -1, empty, empty.astype(np.int32), ())
    # (ties: the shape seen first)
    common = int(np.argmax(np.bincount(codes[defined], minlength=len(scan.codes))))
    is_common = codes == common
    first = int(np.argmax(is_common))
    rows = np.flatnonzero(~is_common).astype(np.int64)
    del is_common, defined
    odd_codes, inverse = np.unique(codes[rows], return_inverse=True)
    others = tuple(
        None if code < 0 else parse_cell_shape(texts[code]) for code in odd_codes
    )
    return ColumnShapes(
        parse_cell_shape(texts[common]),
        first,
        rows,
        np.asarray(inverse, dtype=np.int32).ravel(),
        others,
    )


def read_pointing_index(
    table_path: str,
    data_columns: tuple[str, ...],
    token: str = "",
    stored_shapes: Mapping[str, ColumnShapes] | None = None,
) -> PointingIndex | None:
    """
    The POINTING index of a table, for a lazy pointing_xds: TIME, ANTENNA_ID
    and the row numbers, sorted by TIME (16 bytes per row, whatever the
    number of rows), with the dtype (one cell read) and the cell shape of
    every data column, and the rows with cells of another shape. No other
    data is read.

    These are PR 1's ``PointingColumns`` (without data), with the checks of
    its sub-table cache, made without the data: the dimensions of the
    generic dataset (``generic_dims``, from the cell shapes of all
    columns), usual value types, cells of non-zero size, finite TIME. The
    cell shape of an array column is that of its description (a fixed
    shape) or of most of its cells (``_scan_cell_shapes``: the shapes of all
    cells are read, not their values); the rows whose cells have another
    shape or no value in any array column are ``odd_rows``.

    Parameters
    ----------
    table_path : str
        Path of the POINTING table.
    data_columns : tuple[str, ...]
        Columns the pointing_xds is built from (those not in the table are
        ignored).
    token : str, optional
        Token of the table's fingerprint (recorded in the index).
    stored_shapes : Mapping[str, ColumnShapes] | None, optional
        The cell shapes of the columns that need a scan, from a scan of the
        table as it is (stored in the MS: ``_stored_shapes``): used instead
        of scanning when they are those of exactly these columns.

    Returns
    -------
    PointingIndex | None
        The index, or None if the table cannot be described so (no table,
        no rows, ...: see the module docstring). A MemoryError is raised.
    """
    try:
        return _read_pointing_index(
            table_path, tuple(data_columns), token, stored_shapes
        )
    except MemoryError:
        raise
    except Exception as exc:
        return _not_lazy(table_path, f"{type(exc).__name__}: {exc}")


def _read_pointing_index(
    table_path: str,
    data_columns: tuple[str, ...],
    token: str,
    stored_shapes: Mapping[str, ColumnShapes] | None = None,
) -> PointingIndex | None:
    if maybe_promote is None:
        return _not_lazy(table_path, "xarray.core.dtypes.maybe_promote missing")
    if not os.path.isdir(table_path):
        return _not_lazy(table_path, "no table")
    table_attrs = extract_table_attributes(table_path)
    if is_nested_ms({"other": {"msv2": {"ctds_attrs": table_attrs}}}):
        return _not_lazy(table_path, "looks like a MeasurementSet main table")

    with open_table_ro(table_path) as tb_tool:
        nrows = tb_tool.nrows()
        if nrows == 0:
            return _not_lazy(table_path, "no rows")
        col_types = find_loadable_cols(tb_tool, [])
        # columns of load_generic_table's "select *, !~p/SOURCE_MODEL/"
        colnames = [col for col in tb_tool.colnames() if col != "SOURCE_MODEL"]
        if col_types.get("TIME") != "double" or col_types.get("ANTENNA_ID") != "int":
            return _not_lazy(table_path, "no double TIME / int ANTENNA_ID column")
        if "DIRECTION" not in col_types:
            return _not_lazy(table_path, "no DIRECTION column")
        fixed_shapes = {
            col: () if tb_tool.isscalarcol(col) else _fixed_cell_shape(tb_tool, col)
            for col in col_types
        }
        to_scan = sorted(col for col, shape in fixed_shapes.items() if shape is None)
        if stored_shapes is not None and sorted(stored_shapes) == to_scan:
            shapes, source = dict(stored_shapes), "stored"
        else:
            shapes, source = {}, "scan"

        columns, data_cells, layout = [], {}, []
        for col, col_type in col_types.items():
            is_coord = col.endswith("_ID") or col == "TIME"
            is_key = col in ("TIME", "ANTENNA_ID")
            is_data = col in data_columns and not is_coord
            first_row = 0
            if tb_tool.isscalarcol(col):
                if is_data and col_type not in _SCALAR_VALUE_TYPES:
                    return _not_lazy(table_path, f"{col} is a {col_type} scalar")
                cell_shape = ()
            elif is_key:
                return _not_lazy(table_path, f"{col} is not a scalar column")
            else:
                if is_data and col_type not in CASACORE_TO_NUMPY_DTYPE:
                    return _not_lazy(table_path, f"{col} is a {col_type} array")
                cell_shape = fixed_shapes[col]
                if cell_shape is None:
                    if col not in shapes:
                        shapes[col] = _scan_cell_shapes(tb_tool, col, nrows)
                    if shapes[col].shape is None:
                        return _not_lazy(table_path, f"{col} has no cell with a value")
                    cell_shape, first_row = shapes[col].shape, shapes[col].first
            if is_data:
                data_cells[col] = (cell_shape, first_row)
            columns.append((col, is_coord, cell_shape))
            if col != "SOURCE_MODEL":
                layout.append(
                    (col, is_coord, None if col in shapes else cell_shape, col_type)
                )

        all_rows = np.arange(nrows)
        time = read_column_rows(tb_tool, "TIME", all_rows)
        antenna_id = read_column_rows(tb_tool, "ANTENNA_ID", all_rows)
        # one cell of every data column: the dtype of the values as read
        first_cells = {
            col: read_column_rows(tb_tool, col, np.array([first_row], dtype=np.int64))
            for col, (_, first_row) in data_cells.items()
        }
        odd_parts = [col_shapes.rows for col_shapes in shapes.values()]
        odd_rows = (
            np.unique(np.concatenate(odd_parts)).astype(np.int64)
            if odd_parts
            else np.empty(0, dtype=np.int64)
        )
        # the dtype of the values of a table row (the converter's reads of a
        # partition with fewer than 1,000 POINTING rows stack table rows)
        row_dtypes = {}
        if odd_rows.size:
            for col, (cell_shape, first_row) in data_cells.items():
                if cell_shape != ():
                    value = tb_tool.row([col])[first_row][col]
                    row_dtypes[col] = np.asarray(value).dtype

    data_cols = tuple(data_cells)
    var_dims, sizes = generic_dims(columns, nrows)
    data_dims = {col: var_dims[col][1:] for col in data_cols}
    data_sizes = {}
    for col in data_cols:
        data_sizes.update(zip(data_dims[col], data_cells[col][0], strict=True))
    sizes.pop("row")
    if data_sizes != sizes or 0 in sizes.values():
        # the other variables have dimensions of their own (or cells are empty)
        return _not_lazy(table_path, f"dimensions {sizes} vs {data_sizes}")
    dtypes, cell_shapes = {}, {}
    for col in data_cols:
        cell = first_cells[col]
        if cell.shape[1:] != data_cells[col][0]:
            return _not_lazy(table_path, f"{col}: a cell of {cell.shape[1:]}")
        if not _exact_promotion(cell.dtype):
            return _not_lazy(table_path, f"{col} has values of {cell.dtype}")
        dtypes[col], cell_shapes[col] = cell.dtype, tuple(cell.shape[1:])

    # attributes as load_generic_table() sets them
    var_attrs = {}
    for group in (data_cols, ["TIME", "ANTENNA_ID"]):
        placeholders = {col: xr.DataArray(np.zeros(1)) for col in group}
        add_units_measures(placeholders, table_attrs)
        var_attrs.update({col: placeholders[col].attrs for col in group})

    if not np.isfinite(time).all():
        # the TaQL time range of the per-partition reads would compare with NaN
        return _not_lazy(table_path, "TIME values that are not finite")
    row_dtype = np.dtype(np.int32 if nrows < 2**31 else np.int64)
    if nrows > 1 and np.all(time[1:] >= time[:-1]):
        order = np.arange(nrows, dtype=row_dtype)  # (time-ordered: no sort)
    else:
        # the sort order doubles as the row numbers (in their smallest dtype)
        order = np.argsort(time, kind="stable").astype(row_dtype, copy=False)
        time = time[order]
        antenna_id = antenna_id[order]
    columns = PointingColumns(
        time=time,
        tolerance=projection_tolerance(time),
        antenna_id=antenna_id,
        row=order,
        data_columns=data_cols,
        data=None,
        data_dims=data_dims,
        var_attrs=var_attrs,
        table_attrs=table_attrs,
        bad_cols=list(np.setdiff1d(colnames, list(col_types))),
        table_path=table_path,
    )
    xradio_logger().debug(
        f"POINTING index of {table_path}: {nrows} rows, "
        f"{columns.nbytes / 2**20:.1f} MiB, {odd_rows.size} rows with cells of "
        f"another shape (cell shapes: {source})"
    )
    return PointingIndex(
        columns,
        dtypes,
        cell_shapes,
        odd_rows,
        token,
        shapes=shapes,
        layout=tuple(layout),
        row_dtypes=row_dtypes,
        shapes_source=source,
    )


def _index_key(
    table_path: str, data_columns: tuple[str, ...], token: str
) -> tuple[str, tuple[str, ...], str]:
    return (os.path.realpath(table_path), tuple(data_columns), str(token))


def _memo_index(
    table_path: str,
    data_columns: tuple[str, ...],
    token: str,
    stored_shapes: bool = False,
    fingerprint: dict | None = None,
) -> tuple[PointingIndex | None, bool]:
    """The index of the memo under (path, columns, token), read on a miss
    (once, when several threads miss it), with the cell shapes stored in
    the MS if ``stored_shapes`` and they are those of the table now
    (``fingerprint``, by default taken on a miss). Called holding the
    casatools lock (``casatools_serialized``), before the memo's build lock.
    Returns (index, whether it was read here)."""
    key = _index_key(table_path, data_columns, token)
    entry = POINTING_INDEX_MEMO.get(key)
    read = False
    if entry is None:
        with POINTING_INDEX_MEMO.build_lock(key):
            entry = POINTING_INDEX_MEMO.get(key, count=False)
            if entry is None:
                POINTING_INDEX_MEMO.stats["reads"] += 1
                shapes = (
                    _stored_shapes(table_path, fingerprint) if stored_shapes else None
                )
                entry = _IndexEntry(
                    read_pointing_index(table_path, data_columns, token, shapes)
                )
                POINTING_INDEX_MEMO.put(key, entry)
                read = True
    return entry.index, read


def open_pointing_index(
    table_path: str, data_columns: tuple[str, ...], cache_mode: str | None = None
) -> PointingIndex | None:
    """
    The index of a POINTING table when an MS is opened: from the memo while
    the table's fingerprint is unchanged, else read. The cell shapes of its
    array columns, which the read scans (``_scan_cell_shapes``), are stored
    in the MS (the XRADIO_PARTITIONS sub-table, ``partition_cache``) by the
    ``partition_cache`` modes that store partitions ("auto" and "rebuild"),
    and taken from it by those that read them ("auto", "read") while the
    table's fingerprint is the one they were scanned with: only the first
    open of an MS scans them (and every open in a process whose memo has
    lost the index, with "off", or with a read-only MS that has none
    stored). "rebuild" scans again (not from the memo).

    Parameters
    ----------
    table_path : str
        Path of the POINTING table.
    data_columns : tuple[str, ...]
        Columns the pointing_xds is built from.
    cache_mode : str | None, optional
        The ``partition_cache`` mode of the open (None: the stored cell
        shapes are neither used nor stored).

    Returns
    -------
    PointingIndex | None
        None if there is no table, or its pointing_xds is built eagerly.
    """
    data_columns = tuple(data_columns)
    with casatools_serialized():
        fingerprint = table_fingerprint(table_path)
        if fingerprint is None:
            return None
        token = _fingerprint_token(fingerprint)
        if cache_mode == "rebuild":
            POINTING_INDEX_MEMO.discard(_index_key(table_path, data_columns, token))
        index, read = _memo_index(
            table_path,
            data_columns,
            token,
            stored_shapes=cache_mode in ("auto", "read"),
            fingerprint=fingerprint,
        )
    if (
        read
        and index is not None
        and index.shapes
        and index.shapes_source == "scan"
        and cache_mode in ("auto", "rebuild")
    ):
        _store_shapes(table_path, index, fingerprint)
    return index


@dataclasses.dataclass(frozen=True)
class PartitionPointing:
    """
    The POINTING grid of a partition: its unique times (Unix seconds) and
    antennas, and the token of its rows.
    """

    utime: np.ndarray
    uant: np.ndarray
    token: str


def select_partition(
    index: PointingIndex,
    time_min_max: tuple[float, float],
    antenna_ids: np.ndarray,
) -> tuple[_Selection, PartitionPointing] | None:
    """
    The POINTING rows of a partition, as ``pointing_generic_xds`` selects
    them (``select_pointing_rows``), and its grid.

    Parameters
    ----------
    index : PointingIndex
        The POINTING index.
    time_min_max : tuple[float, float]
        Min/max time of the partition (casacore seconds).
    antenna_ids : np.ndarray
        Antennas of the partition.

    Returns
    -------
    tuple[_Selection, PartitionPointing] | None
        None if no POINTING row is selected.
    """
    columns = index.columns
    sel = select_pointing_rows(
        columns, (np.float64(time_min_max[0]), np.float64(time_min_max[1])), antenna_ids
    )
    if sel.size == 0:
        return None
    # TIME is sorted: the unique times are those that differ from the previous
    # one (np.unique of pointing_generic_xds gives the same, it sorts first)
    time = convert_casacore_time(columns.time[sel], False)
    new_time = np.empty(time.size, dtype=bool)
    new_time[0] = True
    np.not_equal(time[1:], time[:-1], out=new_time[1:])
    utime = time[new_time]
    time_starts = np.append(np.flatnonzero(new_time), sel.size)
    del time, new_time
    uant, acode = np.unique(columns.antenna_id[sel], return_inverse=True)
    token = _digest(
        utime,
        np.asarray(uant, dtype=np.int64),
        np.asarray(columns.row[sel], dtype=np.int64),
    )
    return _Selection(sel, acode, time_starts), PartitionPointing(utime, uant, token)


@dataclasses.dataclass(frozen=True)
class DeferredPointingVariable:
    """
    A data variable of a lazy pointing_xds: what reading any block of it
    needs (picklable, O(antennas)).

    Attributes
    ----------
    name : str
        Data variable name (POINTING_BEAM, ...).
    col : str
        POINTING column.
    table_path : str
        Absolute path of the POINTING table.
    data_columns : tuple[str, ...]
        The data columns of the index (part of its memo key).
    index_token : str
        Token of the table's fingerprint at open (part of the memo key).
    nrows : int
        Rows of the POINTING table at open.
    time_min_max : tuple[float, float]
        Min/max time of the partition (casacore seconds).
    antenna_ids : tuple[int, ...]
        Antennas of the partition.
    n_times, n_antennas : int
        Size of the partition's (time, antenna) grid.
    selection_token : str
        Token of the partition's rows (PartitionPointing.token).
    dtype : np.dtype
        dtype of the values.
    cell_shape : tuple[int, ...]
        Cell shape of the cells read: those of the partition's rows (numpy
        order).
    column_shape : tuple[int, ...] | None, optional
        Cell shape of most cells of the column (the index's), if not
        ``cell_shape`` (a partition whose rows all have cells of another
        shape).
    stored_shapes : bool, optional
        Whether an index read again (another process) may take the cell
        shapes stored in the MS (the ``partition_cache`` mode of the open
        reads them).
    """

    name: str
    col: str
    table_path: str
    data_columns: tuple[str, ...]
    index_token: str
    nrows: int
    time_min_max: tuple[float, float]
    antenna_ids: tuple[int, ...]
    n_times: int
    n_antennas: int
    selection_token: str
    dtype: np.dtype
    cell_shape: tuple[int, ...]
    column_shape: tuple[int, ...] | None = None
    stored_shapes: bool = False

    def changed_error(self, what: str) -> MSv2ChangedError:
        """The error raised when POINTING no longer matches the open."""
        return MSv2ChangedError(
            f"The POINTING table {self.table_path} changed since the MS was opened "
            f"({what}); open it again"
        )

    def index(self) -> PointingIndex:
        """The POINTING index (memo, or read again in another process)."""
        index, _ = _memo_index(
            self.table_path,
            self.data_columns,
            self.index_token,
            stored_shapes=self.stored_shapes,
        )
        if index is None:
            raise self.changed_error("its cells can no longer be read lazily")
        if index.nrows != self.nrows:
            raise self.changed_error(
                f"it has {index.nrows} rows, {self.nrows} when the MS was opened"
            )
        column_shape = (
            self.cell_shape if self.column_shape is None else self.column_shape
        )
        if (
            index.dtypes.get(self.col) != self.dtype
            or index.cell_shapes.get(self.col) != column_shape
        ):
            raise self.changed_error(f"the cells of {self.col} changed")
        return index

    def _selection_key(self) -> tuple:
        return _index_key(self.table_path, self.data_columns, self.index_token) + (
            self.time_min_max,
            self.antenna_ids,
        )

    def discard(self) -> None:
        """Remove the index and the partition's selection from the memos (read
        again from the table on the next read)."""
        POINTING_INDEX_MEMO.discard(
            _index_key(self.table_path, self.data_columns, self.index_token)
        )
        POINTING_SELECTION_MEMO.discard(self._selection_key())

    def selection(self, index: PointingIndex) -> _Selection:
        """The rows of the partition (memo, or selected again and checked
        against the token of the open)."""
        key = self._selection_key()
        selection = POINTING_SELECTION_MEMO.get(key)
        if selection is None:
            with POINTING_SELECTION_MEMO.build_lock(key):
                selection = POINTING_SELECTION_MEMO.get(key, count=False)
                if selection is None:
                    POINTING_SELECTION_MEMO.stats["selections"] += 1
                    selected = select_partition(
                        index, self.time_min_max, np.asarray(self.antenna_ids)
                    )
                    if selected is None or selected[1].token != self.selection_token:
                        raise self.changed_error(
                            "the times, antennas or rows of a partition differ from "
                            "those when the MS was opened"
                        )
                    selection = selected[0]
                    POINTING_SELECTION_MEMO.put(key, selection)
        return selection


# Value types of the array columns whose cells of several shapes (or without
# a value) among the rows of a partition are described from the shapes
# (partition_layout): the dtypes of their table rows and their pad values are
# those of every casacore (python-casacore or casatools)
_ODD_CELL_VALUE_TYPES = (
    "boolean",
    "int",
    "int64",
    "float",
    "double",
    "complex",
    "dcomplex",
)


@dataclasses.dataclass(frozen=True)
class PartitionLayout:
    """
    The generic pointing dataset of a partition whose POINTING rows include
    cells of another shape (or without a value), as the converter's read of
    the partition's rows makes it (``load_generic_table``), described from
    the shapes of their cells (``partition_layout``).

    Attributes
    ----------
    stacked : bool
        Whether the converter stacks the table rows (fewer than 1,000 rows,
        ``load_generic_cols``: a column whose cells vary is padded with the
        pad value of its dtype), else reads every column at once
        (``load_fixed_size_cols``: a column whose cells vary is left out).
    columns : tuple[tuple[str, bool, tuple[int, ...]], ...]
        The columns loaded (name, is a coordinate, cell shape), in load
        order.
    var_dims : dict[str, tuple[str, ...]]
        Dimension names of every column loaded (row dimension first).
    dtypes : dict[str, np.dtype]
        dtype of every data column loaded.
    """

    stacked: bool
    columns: tuple[tuple[str, bool, tuple[int, ...]], ...]
    var_dims: dict[str, tuple[str, ...]]
    dtypes: dict[str, np.dtype]


class _NoLayout(Exception):
    """A partition whose generic pointing dataset is not described from the
    cell shapes (built by the converter's code)."""


def _loaded_cell_shape(
    found: set[tuple[int, ...] | None], stacked: bool, value_type: str
) -> tuple[int, ...] | None:
    """
    The cell shape of a column in the converter's generic dataset of a
    partition whose rows have cells of the shapes ``found`` (None: a cell
    without a value), or None if the column is left out:

    - one shape (every cell with a value): that shape;
    - read column by column (``load_fixed_size_cols``): left out (its getcol
      raises);
    - rows stacked (``stack_tablerow_column``, ``handle_variable_col_issues``):
      padded to the largest shape (Python's tuple order), where a row
      without a value (or of zero length) is all padding; left out where a
      cell does not fit (other dimensions, larger along one).

    Raises _NoLayout for value types whose stacking may differ between
    casacore libraries, and for zero-size cells.
    """
    defined = sorted(shape for shape in found if shape is not None)
    if len(found) == 1 and defined:
        if 0 in defined[0]:
            raise _NoLayout("cells of zero size")
        return defined[0]
    if value_type not in _ODD_CELL_VALUE_TYPES:
        raise _NoLayout(f"{value_type} cells of several shapes")
    if not stacked:
        return None
    if not defined:
        raise _NoLayout("no cell with a value")
    largest = max(defined)
    if 0 in largest:
        raise _NoLayout("cells of zero size")
    for shape in defined:
        if shape[0] == 0:
            continue
        if len(shape) != len(largest) or any(
            n > m for n, m in zip(shape, largest, strict=True)
        ):
            return None
    return largest


def partition_layout(index: PointingIndex, selection: _Selection) -> PartitionLayout:
    """
    The generic pointing dataset of a partition whose POINTING rows include
    cells of another shape (``PartitionLayout``), from the cell shapes of
    the index (no values): the converter reads the partition's rows on
    their own (``load_generic_table``; with 1,000 rows or more column by
    column, else row by row, see ``find_best_col_loader``), so the cell
    shape of every column is that of the cells of these rows
    (``_loaded_cell_shape``), and the dimension names follow from them
    (``generic_dims``).

    Raises
    ------
    _NoLayout
        If it is not described so: cells of zero size, value types other than
        _ODD_CELL_VALUE_TYPES with cells of several shapes, coordinate columns
        with dimensions of their own.
    """
    n_rows = int(selection.sel.size)
    stacked = (
        find_best_col_loader(index.columns.table_path, n_rows)
        is not load_fixed_size_cols
    )
    rows = np.asarray(index.columns.row[selection.sel], dtype=np.int64)
    columns, dtypes = [], {}
    data_columns = index.columns.data_columns
    for col, is_coord, fixed_shape, value_type in index.layout:
        if fixed_shape is None:
            shape = _loaded_cell_shape(index.cells_of(col, rows), stacked, value_type)
            if shape is None:
                continue
        else:
            shape = fixed_shape
        columns.append((col, is_coord, shape))
        if col in data_columns:
            if stacked and shape != () and col in index.row_dtypes:
                dtypes[col] = index.row_dtypes[col]
            else:
                dtypes[col] = index.dtypes[col]
    var_dims, sizes = generic_dims(columns, n_rows)
    sizes.pop("row")
    var_sizes = {}
    for col, is_coord, shape in columns:
        if not is_coord:
            var_sizes.update(zip(var_dims[col][1:], shape, strict=True))
    if var_sizes != sizes:
        raise _NoLayout(f"dimensions {sizes} vs {var_sizes}")
    if 0 in sizes.values():
        raise _NoLayout("cells of zero size")
    return PartitionLayout(stacked, tuple(columns), var_dims, dtypes)


def deferred_pointing_generic_xds(
    in_file: str,
    time_min_max: tuple[np.float64, np.float64] | None,
    antenna_ids: np.ndarray,
    data_columns: Mapping[str, str],
    *,
    specs: dict[str, DeferredPointingVariable],
    context: dict | None = None,
    cache_mode: str | None = None,
) -> xr.Dataset | None:
    """
    The ``generic_loader`` of ``create_pointing_xds`` for the MSv2 backend:
    the generic pointing dataset of a partition (``pointing_generic_xds``)
    with placeholders (``stream_write.deferred_placeholder``) for its data
    variables, described in ``specs``. Its coordinates, dimensions and
    attributes are those of ``pointing_generic_xds``.

    A partition whose POINTING rows include cells of another shape (or
    without a value) has the generic dataset of the converter's own read of
    its rows (``partition_layout``, from the cell shapes: no values): with
    1,000 rows or more, the columns whose cells vary are left out and the
    others are read lazily (``specs``, of the shape of the partition's
    cells); with fewer, they are padded, and the pointing_xds is built by
    the converter's code when read (no ``specs``: ``rebuilt_pointing_xds``).

    Parameters
    ----------
    in_file : str
        Absolute path of the MS.
    time_min_max : tuple[np.float64, np.float64] | None
        Min/max time of the partition (casacore seconds).
    antenna_ids : np.ndarray
        Antennas of the partition.
    data_columns : Mapping[str, str]
        Data variable name by POINTING data column.
    specs : dict[str, DeferredPointingVariable]
        Filled with the description of every placeholder read lazily, by
        data variable name.
    context : dict | None, optional
        Filled with the partition's ``time_min_max`` and ``antenna_ids``
        (for a pointing_xds that the converter's code builds again when
        read: ``rebuilt_pointing_xds``).
    cache_mode : str | None, optional
        The ``partition_cache`` mode of the open: whether the cell shapes of
        POINTING are taken from, and stored in, the MS
        (``open_pointing_index``).

    Returns
    -------
    xr.Dataset | None
        The dataset (empty if no POINTING row is selected), or None if
        POINTING is read eagerly (by the converter's code): a table the
        index cannot describe, or a partition that ``partition_layout``
        cannot describe.
    """
    if context is not None:
        context["time_min_max"] = time_min_max
        context["antenna_ids"] = tuple(int(ant) for ant in antenna_ids)
    if time_min_max is None or len(antenna_ids) == 0:
        return None
    table_path = os.path.join(in_file, POINTING_TABLE)
    columns_key = tuple(data_columns)
    subtable_cache = active_subtable_cache()
    if subtable_cache is None:
        index = open_pointing_index(table_path, columns_key, cache_mode)
    else:
        # one fingerprint per open (the partitions of an open share the cache)
        index = subtable_cache.get_or_build(
            ("pointing_index", table_path, columns_key),
            lambda: open_pointing_index(table_path, columns_key, cache_mode),
        )
    if index is None:
        return None
    selected = select_partition(index, time_min_max, antenna_ids)
    if selected is None:
        return xr.Dataset()
    layout = None
    if index.selects_odd_rows(selected[0]):
        # Cells of another shape (or without a value) among the partition's
        # rows: the converter reads such a table per partition
        try:
            layout = partition_layout(index, selected[0])
        except _NoLayout as exc:
            xradio_logger().debug(
                f"The pointing_xds of a partition of {in_file} is built by the "
                f"converter's code: {exc}"
            )
            # (PR 1's sub-table cache would refuse it after reading its data
            # columns: told so at once)
            if subtable_cache is not None:
                subtable_cache.get_or_build(
                    ("pointing_columns", table_path, columns_key), lambda: None
                )
            return None
    grid = selected[1]
    del selected

    columns = index.columns
    var_attrs = columns.var_attrs
    shape = (grid.utime.size, grid.uant.size)
    time_min_max = (float(time_min_max[0]), float(time_min_max[1]))
    antenna_ids = tuple(int(ant) for ant in antenna_ids)
    if layout is None:
        loaded = [
            (col, index.cell_shapes[col], columns.data_dims[col])
            for col in columns.data_columns
        ]
        dtypes, bad_cols = index.dtypes, list(columns.bad_cols)
    else:
        # (every variable of the converter's generic dataset: its dimensions)
        loaded = [
            (col, cell_shape, layout.var_dims[col][1:])
            for col, is_coord, cell_shape in layout.columns
            if not is_coord
        ]
        dtypes = layout.dtypes
        # (the columns of the query that are not loaded)
        names = [col for col, _, _ in layout.columns]
        bad_cols = list(
            np.setdiff1d(
                [col for col, _, _, _ in index.layout] + list(columns.bad_cols), names
            )
        )
    value_types = {col: value_type for col, _, _, value_type in index.layout}
    data_vars = {}
    for col, cell_shape, dims in loaded:
        if col not in columns.data_columns:
            # (a placeholder that the converter's code leaves out)
            dtype = CASACORE_TO_NUMPY_DTYPE.get(value_types[col], np.dtype(object))
            data_vars[col] = xr.Variable(
                ("TIME", "ANTENNA_ID") + tuple(dims),
                deferred_placeholder(f"pointing-{col}", shape + cell_shape, dtype),
            )
            continue
        name = data_columns[col]
        if layout is None or not layout.stacked:
            specs[name] = DeferredPointingVariable(
                name=name,
                col=col,
                table_path=os.path.abspath(table_path),
                data_columns=columns_key,
                index_token=index.token,
                nrows=index.nrows,
                time_min_max=time_min_max,
                antenna_ids=antenna_ids,
                n_times=shape[0],
                n_antennas=shape[1],
                selection_token=grid.token,
                dtype=dtypes[col],
                cell_shape=tuple(cell_shape),
                column_shape=index.cell_shapes[col],
                stored_shapes=cache_mode in ("auto", "read", "rebuild"),
            )
        data_vars[col] = xr.Variable(
            ("TIME", "ANTENNA_ID") + tuple(dims),
            deferred_placeholder(f"pointing-{name}", shape + cell_shape, dtypes[col]),
            attrs=copy.deepcopy(var_attrs[col]),
        )
    coords = {
        "TIME": xr.Variable("TIME", grid.utime, attrs=copy.deepcopy(var_attrs["TIME"])),
        "ANTENNA_ID": xr.Variable(
            "ANTENNA_ID", grid.uant, attrs=copy.deepcopy(var_attrs["ANTENNA_ID"])
        ),
    }
    attrs = {
        "other": {
            "msv2": {
                "ctds_attrs": copy.deepcopy(columns.table_attrs),
                "bad_cols": bad_cols,
            }
        }
    }
    return xr.Dataset(data_vars, coords=coords, attrs=attrs)


class PointingColumnArray(MSv2BackendArray):
    """
    One data variable of a lazy pointing_xds (see the module docstring).

    Parameters
    ----------
    spec : DeferredPointingVariable
        The variable's description.
    shape : tuple[int, ...]
        Shape of the data variable: (n_times, n_antennas) + its cell shape,
        which is the column's cell shape without the dimensions of size 1
        that the converter selects (``n_polynomial``).
    dtype : DTypeLike
        dtype of the data variable.
    node : str
        Name of the MSv4 node (for messages).
    partition : PartitionIndex | None
        The MAIN rows of the partition, whose time range and antennas select
        its POINTING rows: a read first checks that the partition is the one
        of the open (``PartitionIndex.verify_current``).
    """

    def __init__(
        self,
        spec: DeferredPointingVariable,
        shape: tuple[int, ...],
        dtype: DTypeLike,
        node: str = "",
        partition: PartitionIndex | None = None,
    ):
        super().__init__(shape, dtype)
        if self.shape[:2] != (spec.n_times, spec.n_antennas):
            raise ValueError(
                f"{spec.name}: shape {self.shape} does not start with the "
                f"partition's (time, antenna) grid {(spec.n_times, spec.n_antennas)}"
            )
        cell = tuple(n for n in self.shape[2:] if n != 1)
        if cell != tuple(n for n in spec.cell_shape if n != 1):
            raise ValueError(
                f"{spec.name}: cell shape {self.shape[2:]} is not that of the "
                f"{spec.col} cells {spec.cell_shape} without dimensions of size 1"
            )
        if self.dtype != spec.dtype:
            raise ValueError(f"{spec.name}: dtype {self.dtype}, {spec.dtype} read")
        self.spec = spec
        self.node = str(node)
        self.partition = partition

    def _message(self, key: tuple[slice, ...], exc: BaseException) -> str:
        block = ", ".join(f"{k.start}:{k.stop}" for k in key)
        message = (
            f"Reading the POINTING column {self.spec.col} for the data variable "
            f"{self.spec.name} of the pointing_xds of {self.node or 'an MSv4'} "
            f"(block [{block}]) from {self.spec.table_path} failed: "
            f"{type(exc).__name__}: {exc}"
        )
        return message

    def _raw_indexing_method(self, key: tuple[slice, ...]) -> np.ndarray:
        return self._read_selection(
            tuple(np.arange(k.start, k.stop, dtype=np.int64) for k in key)
        )

    def _read_selection(self, selections: tuple[np.ndarray, ...]) -> np.ndarray:
        """
        The values of the selected times, antennas and cell elements, read in
        time sub-blocks of bounded size. If rows of the selection no longer
        have the TIME and ANTENNA_ID of the index (rewritten in place), the
        index and the partition's selection are read again (checked against
        the open: MSv2ChangedError) and the selection read again with them.
        """
        if self.partition is not None:
            self.partition.verify_current()
        out = np.empty(tuple(s.size for s in selections), dtype=self.dtype)
        key = bounding_slices(selections)
        for _attempt in (1, 2):
            moved = self._read_into(out, selections, key)
            if not moved:
                return out
            xradio_logger().debug(
                f"{self.spec.table_path}: {moved} since the POINTING index was "
                "read: read again"
            )
            self.spec.discard()
        raise self.spec.changed_error(f"{moved} while it was read")

    def _read_into(
        self,
        out: np.ndarray,
        selections: tuple[np.ndarray, ...],
        key: tuple[slice, ...],
    ) -> str:
        """Read the selection into ``out``; "" or why rows of the selection no
        longer have the keys of the index."""
        times, antennas, cell_selections = selections[0], selections[1], selections[2:]
        cell_elems = int(np.prod(self.spec.cell_shape, dtype=np.int64)) or 1
        cell_bytes = cell_elems * (2 * self.dtype.itemsize + 8) + _ROW_INDEX_BYTES
        step = max(1, POINTING_SUB_BLOCK_BYTES // (antennas.size * cell_bytes))
        with casatools_serialized():
            table = None
            try:
                index = self.spec.index()
                selection = self.spec.selection(index)
                # opened in the thread / process that reads
                with open_table_ro(self.spec.table_path) as table:
                    nrows = table.nrows()
                    if nrows != self.spec.nrows:
                        raise self.spec.changed_error(
                            f"it has {nrows} rows, {self.spec.nrows} when the MS was "
                            "opened"
                        )
                    # (not written since the open: rows not checked one by one)
                    check_keys = not pointing_unchanged(
                        table, self.spec.table_path, self.spec.index_token
                    )
                    for p0 in range(0, times.size, step):
                        sub_times = times[p0 : p0 + step]
                        grid, moved = self._read_grid(
                            table, index, selection, sub_times, antennas, check_keys
                        )
                        if moved:
                            return moved
                        grid = grid.reshape(
                            (sub_times.size, antennas.size) + self.shape[2:]
                        )
                        out[p0 : p0 + sub_times.size] = take_from_block(
                            grid,
                            (np.arange(sub_times.size), np.arange(antennas.size))
                            + tuple(cell_selections),
                            [0] * grid.ndim,
                        )
                        del grid
            except MSv2ChangedError:
                raise
            except Exception as exc:
                raise MSv2ReadError(self._message(key, exc)) from exc
            finally:
                table = None  # (casatools: released holding the lock)
        return ""

    def _read_grid(
        self,
        table: Any,
        index: PointingIndex,
        selection: _Selection,
        times: np.ndarray,
        antennas: np.ndarray,
        check_keys: bool = True,
    ) -> tuple[np.ndarray | None, str]:
        """
        The (len(times), len(antennas)) + cell grid of the times ``times`` and
        antennas ``antennas`` (sorted unique indices into the partition's
        grid): the first row (lowest row number) of every cell, pivoted as
        ``pointing_generic_xds`` pivots them; or (None, why) if rows no
        longer have the TIME and ANTENNA_ID of the index (checked with
        ``check_keys``).
        """
        starts = selection.time_starts
        if is_range(times):
            p0, p1 = int(starts[times[0]]), int(starts[times[-1] + 1])
            picked = np.arange(p0, p1, dtype=np.int64)
            counts = np.diff(starts[times[0] : times[-1] + 2])
        else:
            lo, counts = starts[times], starts[times + 1] - starts[times]
            picked = np.repeat(lo - (np.cumsum(counts) - counts), counts) + np.arange(
                int(counts.sum()), dtype=np.int64
            )
        positions = selection.sel[picked]
        acode = selection.acode[picked].astype(np.int64)
        tcode = np.repeat(np.arange(times.size, dtype=np.int64), counts)
        del picked, counts
        if is_range(antennas):
            a0, a1 = int(antennas[0]), int(antennas[-1]) + 1
            if a0 > 0 or a1 < self.spec.n_antennas:
                keep = (acode >= a0) & (acode < a1)
                positions, acode, tcode = positions[keep], acode[keep], tcode[keep]
                del keep
            acode = acode - a0
        else:
            antenna_position = np.full(self.spec.n_antennas, -1, dtype=np.int64)
            antenna_position[antennas] = np.arange(antennas.size)
            acode = antenna_position[acode]
            keep = acode >= 0
            positions, acode, tcode = positions[keep], acode[keep], tcode[keep]
            del keep
        n_antennas = antennas.size
        cell_key = tcode * n_antennas + acode
        del acode, tcode
        rows = np.asarray(index.columns.row[positions], dtype=np.int64)
        moved = self._rows_moved(table, index, positions, rows) if check_keys else ""
        if moved:
            return None, moved
        del positions
        # first row (lowest row number) of every (time, antenna) cell
        order = np.lexsort((rows, cell_key))
        sorted_keys = cell_key[order]
        is_first = np.ones(order.size, dtype=bool)
        np.not_equal(sorted_keys[1:], sorted_keys[:-1], out=is_first[1:])
        first = order[is_first]
        del order, sorted_keys, is_first
        first_rows = rows[first]
        tcode_first, acode_first = np.divmod(cell_key[first], n_antennas)
        del rows, cell_key, first
        if check_keys:
            self._check_cells(table, first_rows)
        values = self._read_rows(table, first_rows, index)
        return (
            pivot_time_antenna(
                values, tcode_first, acode_first, (times.size, n_antennas)
            ),
            "",
        )

    def _rows_moved(
        self,
        table: Any,
        index: PointingIndex,
        positions: np.ndarray,
        rows: np.ndarray,
    ) -> str:
        """Whether ``rows`` (at ``positions`` of the index) no longer have its
        TIME and ANTENNA_ID: the first difference, or ""."""
        if rows.size == 0:
            return ""
        order = np.argsort(rows)
        sorted_rows = rows[order]
        for col, expected in (
            ("TIME", index.columns.time),
            ("ANTENNA_ID", index.columns.antenna_id),
        ):
            values = self._read_sorted_rows(
                table, sorted_rows, col, (), column_dtype(table, col)
            )
            expected = expected[positions[order]]
            if not np.array_equal(values, expected):
                changed = int(sorted_rows[np.flatnonzero(values != expected)[0]])
                return f"the {col} of POINTING row {changed} changed"
        return ""

    def _check_cells(self, table: Any, rows: np.ndarray) -> None:
        """Raise MSv2ChangedError if a cell of ``rows`` no longer has the
        shape of the open, or no value (their shapes are read, not their
        values; for a table written since the open)."""
        cell_shape = self.spec.cell_shape
        if cell_shape == () or rows.size == 0:
            return
        starts, lengths = rows_to_runs(np.sort(rows))
        for start, length in zip(starts.tolist(), lengths.tolist(), strict=True):
            for offset in range(0, length, SHAPE_SCAN_ROWS):
                n = min(SHAPE_SCAN_ROWS, length - offset)
                try:
                    strings = _shape_strings(table, self.spec.col, start + offset, n)
                except Exception:
                    raise self.spec.changed_error(
                        f"a cell of {self.spec.col} has no value"
                    ) from None
                for text in set(strings):
                    if parse_cell_shape(text) != cell_shape:
                        raise self.spec.changed_error(
                            f"a cell of {self.spec.col} has the shape "
                            f"{parse_cell_shape(text)}, {cell_shape} when the MS was "
                            "opened"
                        )

    def _read_rows(
        self, table: Any, rows: np.ndarray, index: PointingIndex | None = None
    ) -> np.ndarray:
        """The cells of ``rows`` (in that order), read in ascending row order
        (with the cell shapes of ``index``, for windowed reads)."""
        cell_shape = self.spec.cell_shape
        if rows.size == 0:
            return np.empty((0,) + cell_shape, dtype=self.spec.dtype)
        order = np.argsort(rows)
        sorted_values = self._read_sorted_rows(
            table,
            rows[order],
            self.spec.col,
            cell_shape,
            self.spec.dtype,
            None if index is None else index.shapes.get(self.spec.col),
        )
        if (
            sorted_values.dtype != self.spec.dtype
            or sorted_values.shape[1:] != cell_shape
        ):
            raise self.spec.changed_error(
                f"{self.spec.col} has cells of {sorted_values.dtype} "
                f"{sorted_values.shape[1:]}, {self.spec.dtype} {cell_shape} when the "
                "MS was opened"
            )
        values = np.empty_like(sorted_values)
        values[order] = sorted_values
        return values

    def _read_sorted_rows(
        self,
        table: Any,
        rows: np.ndarray,
        col: str,
        cell_shape: tuple[int, ...],
        dtype: np.dtype,
        shapes: ColumnShapes | None = None,
    ) -> np.ndarray:
        """The cells of ascending ``rows`` of a column (read_column_rows, or in
        windows of consecutive rows for fragmented rows without in-place
        reads; a window never spans a row whose cell has another shape than
        ``cell_shape`` or no value, by the column's ``shapes``)."""
        if has_in_place_reads(table) or rows_to_runs(rows)[0].size <= FRAGMENTED_RUNS:
            return read_column_rows(table, col, rows)
        cell_bytes = max(1, int(np.prod(cell_shape, dtype=np.int64)) * dtype.itemsize)
        window_rows = max(1, _WINDOW_BYTES // cell_bytes)
        # a window ends where the next row is too far from its first row or
        # from the row before, or where rows of other cells lie in between
        gaps = np.diff(rows) > _WINDOW_MAX_GAP
        if shapes is not None and cell_shape != ():
            gaps |= shapes.window_breaks(rows, cell_shape)
        ends = np.flatnonzero(gaps) + 1
        out = np.empty((rows.size,) + cell_shape, dtype=dtype)
        for lo, hi in zip(
            np.r_[0, ends].tolist(), np.r_[ends, rows.size].tolist(), strict=True
        ):
            i = lo
            while i < hi:
                start = int(rows[i])
                j = int(np.searchsorted(rows[i:hi], start + window_rows)) + i
                buf = np.empty((int(rows[j - 1]) - start + 1,) + cell_shape, dtype)
                read_row_range(table, col, start, buf.shape[0], buf)
                out[i:j] = buf[rows[i:j] - start]
                del buf
                i = j
        return out


def lazy_pointing_xds(
    pointing_xds: xr.Dataset,
    specs: Mapping[str, DeferredPointingVariable],
    node: str = "",
    partition: PartitionIndex | None = None,
) -> xr.Dataset:
    """
    The pointing_xds of a partition built with ``deferred_pointing_generic_xds``
    with its placeholders replaced by lazily indexed arrays
    (:class:`PointingColumnArray`, checking ``partition`` when read), keeping
    their dimensions, attributes and encoding.
    """
    lazy = {}
    for name, spec in specs.items():
        if name not in pointing_xds.data_vars:
            continue
        var = pointing_xds.variables[name]
        array = PointingColumnArray(
            spec, var.shape, var.dtype, node=node, partition=partition
        )
        lazy_var = xr.Variable(
            var.dims,
            xr.core.indexing.LazilyIndexedArray(array),
            attrs=copy.deepcopy(var.attrs),
        )
        lazy_var.encoding = dict(var.encoding)
        lazy[name] = lazy_var
    # (variables replaced in place: the order of the data variables is kept)
    return pointing_xds.assign(lazy)


# --- pointing_xds that the converter's code builds (again) -----------------------


class _BuildMemo:
    """
    Per-process LRU memo of the pointing_xds that PointingBuildArray builds
    on read, by build and POINTING fingerprint, bounded by bytes (a build
    larger than the bound is not kept), with striped locks so that threads
    reading the variables of one partition build it once. Renewed (empty,
    with new locks) in a fork child.
    """

    def __init__(self, max_bytes: int):
        self.max_bytes = int(max_bytes)
        self.reset_after_fork()

    def reset_after_fork(self) -> None:
        self._lock = threading.Lock()
        self._build_locks = [threading.Lock() for _ in range(16)]
        self._entries: collections.OrderedDict[Hashable, tuple[xr.Dataset, int]] = (
            collections.OrderedDict()
        )
        self._nbytes = 0
        self.stats: collections.Counter = collections.Counter()

    def _get(self, key: Hashable) -> xr.Dataset | None:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            self._entries.move_to_end(key)
            self.stats["hits"] += 1
            return entry[0]

    def get_or_build(
        self, key: Hashable, build: Callable[[], xr.Dataset]
    ) -> xr.Dataset:
        """The dataset of ``key``, built with ``build()`` on a miss (once,
        when several threads miss it; never called holding the casatools
        lock, which ``build`` takes)."""
        xds = self._get(key)
        if xds is not None:
            return xds
        with self._build_locks[hash(key) % len(self._build_locks)]:
            xds = self._get(key)
            if xds is not None:
                return xds
            self.stats["builds"] += 1
            xds = build()
            nbytes = sum(int(var.nbytes) for var in xds.variables.values())
            if nbytes <= self.max_bytes:
                with self._lock:
                    self._entries[key] = (xds, nbytes)
                    self._nbytes += nbytes
                    while self._nbytes > self.max_bytes:
                        _, (_, evicted) = self._entries.popitem(last=False)
                        self._nbytes -= evicted
            return xds

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._nbytes = 0
            self.stats.clear()

    def __len__(self) -> int:
        return len(self._entries)


POINTING_BUILD_MEMO = _BuildMemo(POINTING_BUILD_MEMO_MAX_BYTES)

if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=POINTING_BUILD_MEMO.reset_after_fork)


@dataclasses.dataclass(frozen=True)
class PointingBuild:
    """
    What building the pointing_xds of a partition with the converter's code
    needs (``create_pointing_xds`` without interpolation), picklable.

    Attributes
    ----------
    in_file : str
        Absolute path of the MS.
    time_min_max : tuple[np.float64, np.float64] | None
        Min/max time of the partition (casacore seconds).
    antenna_ids, antenna_names : tuple
        The antennas of the partition's antenna_xds (ids and names).
    """

    in_file: str
    time_min_max: tuple[np.float64, np.float64] | None
    antenna_ids: tuple[int, ...]
    antenna_names: tuple[str, ...]

    def build(self) -> xr.Dataset:
        """The pointing_xds, as build_partition builds it (casatools: holding
        the casatools lock)."""
        ant_xds_name_ids = xr.DataArray(
            np.array(self.antenna_names),
            dims="antenna_name",
            coords={"antenna_id": ("antenna_name", np.array(self.antenna_ids))},
            name="antenna_name",
        ).set_xindex("antenna_id")
        with casatools_serialized():
            return create_pointing_xds(
                self.in_file, ant_xds_name_ids, self.time_min_max, None
            )


def pointing_coords_token(pointing_xds: xr.Dataset) -> str:
    """
    Token of the coordinates of a pointing_xds (every coordinate variable:
    its name, dimensions and values; the time and antenna grid of the
    partition's POINTING rows).
    """
    digest = hashlib.blake2b(digest_size=16)
    for name in sorted(str(name) for name in pointing_xds.coords):
        var = pointing_xds.coords[name].variable
        values = np.asarray(var.values)
        if values.dtype.kind == "O":
            values = values.astype(str)
        digest.update(json.dumps([name, list(var.dims)]).encode())
        digest.update(_digest(values).encode())
    return digest.hexdigest()


class PointingBuildArray(MSv2BackendArray):
    """
    A data variable of a pointing_xds that only the converter's code can
    build (POINTING tables that ``read_pointing_index`` cannot describe;
    partitions of fewer than 1,000 POINTING rows with cells of another
    shape, which the converter pads, and those that ``partition_layout``
    cannot describe): a read takes the partition's pointing_xds from
    POINTING_BUILD_MEMO (by the POINTING table's fingerprint), or builds it
    again (:class:`PointingBuild`), and returns the selection of the
    variable. The variables of a partition so share one build; the open
    keeps no values. A build whose coordinates (times, antennas) are not
    those of the open (``coords_token``: e.g. POINTING TIME rewritten in
    place), or whose variable has another shape or dtype, raises
    MSv2ChangedError: its values would not be those of the coordinates of
    the opened pointing_xds.

    Parameters
    ----------
    build : PointingBuild
        How to build the pointing_xds.
    name : str
        Data variable name.
    shape : tuple[int, ...]
        Its shape (when the MS was opened).
    dtype : DTypeLike
        Its dtype.
    node : str
        Name of the MSv4 node (for messages).
    partition : PartitionIndex | None
        The MAIN rows of the partition (see PointingColumnArray).
    coords_token : str
        ``pointing_coords_token`` of the pointing_xds built at open ("": not
        checked).
    """

    def __init__(
        self,
        build: PointingBuild,
        name: str,
        shape: tuple[int, ...],
        dtype: DTypeLike,
        node: str = "",
        partition: PartitionIndex | None = None,
        coords_token: str = "",
    ):
        super().__init__(shape, dtype)
        self.build = build
        self.name = str(name)
        self.node = str(node)
        self.partition = partition
        self.coords_token = str(coords_token)

    def _raw_indexing_method(self, key: tuple[slice, ...]) -> np.ndarray:
        if self.partition is not None:
            self.partition.verify_current()
        table = os.path.join(self.build.in_file, POINTING_TABLE)
        try:
            with casatools_serialized():
                token = pointing_table_token(table)
            xds = POINTING_BUILD_MEMO.get_or_build(
                (
                    os.path.realpath(self.build.in_file),
                    self.build.time_min_max,
                    self.build.antenna_ids,
                    self.build.antenna_names,
                    token,
                ),
                self.build.build,
            )
        except Exception as exc:
            block = ", ".join(f"{k.start}:{k.stop}" for k in key)
            raise MSv2ReadError(
                f"Building the pointing_xds of {self.node or 'an MSv4'} from {table} "
                f"to read {self.name} (block [{block}]) failed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        var = xds.variables.get(self.name)
        if var is None or var.shape != self.shape or var.dtype != self.dtype:
            found = "none" if var is None else f"{var.dtype} {var.shape}"
            raise MSv2ChangedError(
                f"The POINTING table {table} changed since the MS was opened (its "
                f"{self.name} of {self.node or 'an MSv4'} is {found}, "
                f"{self.dtype} {self.shape} when it was opened); open it again"
            )
        if self.coords_token and pointing_coords_token(xds) != self.coords_token:
            raise MSv2ChangedError(
                f"The POINTING table {table} changed since the MS was opened (the "
                f"times or antennas of the pointing_xds of {self.node or 'an MSv4'} "
                "differ from those when it was opened); open it again"
            )
        # (a copy: the build may be kept in the memo)
        return np.array(var.values[key], copy=True)


def rebuilt_pointing_xds(
    pointing_xds: xr.Dataset,
    build: PointingBuild,
    node: str = "",
    partition: PartitionIndex | None = None,
    coords_token: str | None = None,
) -> xr.Dataset:
    """
    A pointing_xds built at open by the converter's code, or with
    placeholders described by ``partition_layout``, with its data variables
    replaced by lazily indexed arrays that build it by the converter's code
    when read (:class:`PointingBuildArray`, checking ``partition`` and the
    coordinates of the build when read), keeping their dimensions,
    attributes and encoding.

    ``coords_token`` is the ``pointing_coords_token`` of the pointing_xds
    of the open with every coordinate of the converter's build (default:
    that of ``pointing_xds``; a pointing_xds taken from a tree without the
    coordinates it inherits passes the token of the dataset with them).
    """
    lazy = {}
    if coords_token is None:
        coords_token = pointing_coords_token(pointing_xds)
    for name, var in pointing_xds.data_vars.items():
        array = PointingBuildArray(
            build,
            name,
            var.shape,
            var.dtype,
            node=node,
            partition=partition,
            coords_token=coords_token,
        )
        lazy_var = xr.Variable(
            var.dims,
            xr.core.indexing.LazilyIndexedArray(array),
            attrs=copy.deepcopy(var.attrs),
        )
        lazy_var.encoding = dict(var.variable.encoding)
        lazy[name] = lazy_var
    return pointing_xds.assign(lazy)
