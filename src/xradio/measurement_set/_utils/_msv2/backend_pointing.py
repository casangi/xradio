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
  each (its dtype); cell shapes come from the column descriptions (a fixed
  shape) or the first row. The index is kept in a per-process memo while
  the table's fingerprint (``table_fingerprint``) is unchanged.
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
  row selection).

Cells of a data column whose shape the column description does not fix are
taken to have the shape of its first row (as the converter's sub-table cache
requires of a table it holds); a read of a cell of another shape raises
:class:`MSv2ReadError` (the converter reads such a table per partition, and
leaves out or pads the column there).

Tables that cannot be described from their column descriptions, first row
and index (no DIRECTION column, unusual value types, an undefined or empty
cell in the first row, TIME values that are not finite, ...): their
pointing_xds is built at open by the converter's code, as by the converter,
and its data variables are replaced by :class:`PointingBuildArray`, which
builds it again when they are read (the values are not kept). With
``pointing_interpolate=True`` (the interpolation needs the values) the
pointing_xds is built eagerly.

Staleness: an array records the number of POINTING rows and a token of its
partition's selection (times, antennas, row numbers). A read finds the
index in the memo (in this process, the index of the open) or reads it
again (other processes); a POINTING table with another number of rows, or
a selection with another token, raises :class:`MSv2ChangedError`. A read
whose rows no longer have the TIME and ANTENNA_ID of the index (rewritten in
place) reads the index again, as another process would: the outcome does
not depend on the memo.
"""

import copy
import dataclasses
import hashlib
import json
import os
from collections.abc import Mapping
from typing import Any

import numpy as np
import xarray as xr
from numpy.typing import DTypeLike

from xradio._utils._casacore.tables import casatools_serialized
from xradio._utils.logging import xradio_logger
from xradio.measurement_set._utils._msv2._tables.read import (
    add_units_measures,
    convert_casacore_time,
    extract_table_attributes,
    find_loadable_cols,
    is_nested_ms,
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
)
from xradio.measurement_set._utils._msv2._tables.table_query import open_table_ro
from xradio.measurement_set._utils._msv2.backend_arrays import (
    MSv2BackendArray,
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
# Value dtypes whose round trip through xarray's promoted fill dtype (for
# grids with missing cells) keeps every value: a block pivoted with or without
# missing cells gives the values of the whole grid. (POINTING data columns
# have one of these types, see read_pointing_index.)
_EXACT_PROMOTION_KINDS = "bfc"
_EXACT_PROMOTION_INTS = (np.dtype(np.int8), np.dtype(np.int16), np.dtype(np.int32))


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
    verified : dict[str, bool]
        Whether the column description vouches for the cell shape of every
        row of a data column (a fixed shape), else it is the first row's.
    token : str
        Token of the table's fingerprint when the index was read.
    """

    columns: PointingColumns
    dtypes: dict[str, np.dtype]
    cell_shapes: dict[str, tuple[int, ...]]
    verified: dict[str, bool]
    token: str

    @property
    def nrows(self) -> int:
        """Rows of the POINTING table."""
        return int(self.columns.time.size)

    @property
    def nbytes(self) -> int:
        return self.columns.nbytes


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
    """Empty the per-process memos of POINTING indices and selections (for
    tests)."""
    POINTING_INDEX_MEMO.clear()
    POINTING_SELECTION_MEMO.clear()


def _digest(*arrays: np.ndarray) -> str:
    digest = hashlib.blake2b(digest_size=16)
    for values in arrays:
        values = np.ascontiguousarray(values)
        digest.update(str(values.dtype).encode())
        digest.update(np.int64(values.size).tobytes())
        digest.update(values.tobytes())
    return digest.hexdigest()


def pointing_table_token(table_path: str) -> str | None:
    """
    Token of the fingerprint of a POINTING table (``table_fingerprint``:
    rows, columns, data managers, their change counters and files), None if
    there is no table.
    """
    fingerprint = table_fingerprint(table_path)
    if fingerprint is None:
        return None
    text = json.dumps(fingerprint, sort_keys=True, default=str)
    return hashlib.blake2b(text.encode(), digest_size=16).hexdigest()


def _exact_promotion(dtype: np.dtype) -> bool:
    return dtype.kind in _EXACT_PROMOTION_KINDS or dtype in _EXACT_PROMOTION_INTS


# Option bit of a casacore column description: the cells have the shape of
# the description (every cell is defined, with that shape)
_FIXED_SHAPE_OPTION = 4


def _not_lazy(table_path: str, reason: str) -> None:
    xradio_logger().debug(f"Not reading {table_path} lazily: {reason}")


def _declared_cell_shape(table: Any, col: str) -> tuple[tuple[int, ...], bool]:
    """
    The cell shape of an array column (numpy order), and whether the column
    description fixes it (then every cell has it); otherwise the shape of the
    first row (a RuntimeError if it is undefined). Reads no data.
    """
    desc = table.getcoldesc(col)
    shape = desc.get("shape")
    if (
        int(desc.get("option", 0)) & _FIXED_SHAPE_OPTION
        and shape is not None
        and len(shape)
    ):
        return tuple(int(n) for n in np.asarray(shape).ravel()), True
    return parse_shape_string(table.getcolshapestring(col, 0, 1)[0]), False


def read_pointing_index(
    table_path: str, data_columns: tuple[str, ...], token: str = ""
) -> PointingIndex | None:
    """
    The POINTING index of a table, for a lazy pointing_xds: TIME, ANTENNA_ID
    and the row numbers, sorted by TIME (16 bytes per row, whatever the
    number of rows), with the dtype (one cell read) and the cell shape (the
    column description, or the first row) of every data column. No other
    data is read.

    These are PR 1's ``PointingColumns`` (without data), with the checks of
    its sub-table cache that need no data: the dimensions of the generic
    dataset (``generic_dims``, from the cell shapes of all columns), usual
    value types, defined cells of non-zero size in the first row, finite
    TIME. Cells of another shape than the first row's (in columns whose
    description does not fix the shape) are found by the reads.

    Parameters
    ----------
    table_path : str
        Path of the POINTING table.
    data_columns : tuple[str, ...]
        Columns the pointing_xds is built from (those not in the table are
        ignored).
    token : str, optional
        Token of the table's fingerprint (recorded in the index).

    Returns
    -------
    PointingIndex | None
        The index, or None if the table cannot be described so (no table,
        no rows, ...: see the module docstring). A MemoryError is raised.
    """
    try:
        return _read_pointing_index(table_path, tuple(data_columns), token)
    except MemoryError:
        raise
    except Exception as exc:
        return _not_lazy(table_path, f"{type(exc).__name__}: {exc}")


def _read_pointing_index(
    table_path: str, data_columns: tuple[str, ...], token: str
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

        columns, data_cells = [], {}
        for col, col_type in col_types.items():
            is_coord = col.endswith("_ID") or col == "TIME"
            is_key = col in ("TIME", "ANTENNA_ID")
            is_data = col in data_columns and not is_coord
            fixed = True
            if tb_tool.isscalarcol(col):
                if is_data and col_type not in _SCALAR_VALUE_TYPES:
                    return _not_lazy(table_path, f"{col} is a {col_type} scalar")
                cell_shape = ()
            elif is_key:
                return _not_lazy(table_path, f"{col} is not a scalar column")
            else:
                if is_data and col_type not in CASACORE_TO_NUMPY_DTYPE:
                    return _not_lazy(table_path, f"{col} is a {col_type} array")
                # (raises for an undefined first cell)
                cell_shape, fixed = _declared_cell_shape(tb_tool, col)
            if is_data:
                data_cells[col] = (cell_shape, fixed)
            columns.append((col, is_coord, cell_shape))

        all_rows = np.arange(nrows)
        time = read_column_rows(tb_tool, "TIME", all_rows)
        antenna_id = read_column_rows(tb_tool, "ANTENNA_ID", all_rows)
        # one cell of every data column: the dtype of the values as read
        first_cells = {
            col: read_column_rows(tb_tool, col, np.zeros(1, dtype=np.int64))
            for col in data_cells
        }

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
    dtypes, cell_shapes, verified = {}, {}, {}
    for col in data_cols:
        cell = first_cells[col]
        if cell.shape[1:] != data_cells[col][0]:
            return _not_lazy(table_path, f"{col}: a first cell of {cell.shape[1:]}")
        if not _exact_promotion(cell.dtype):
            return _not_lazy(table_path, f"{col} has values of {cell.dtype}")
        dtypes[col], cell_shapes[col] = cell.dtype, tuple(cell.shape[1:])
        verified[col] = data_cells[col][1]

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
        f"{columns.nbytes / 2**20:.1f} MiB"
    )
    return PointingIndex(columns, dtypes, cell_shapes, verified, token)


def _index_key(
    table_path: str, data_columns: tuple[str, ...], token: str
) -> tuple[str, tuple[str, ...], str]:
    return (os.path.realpath(table_path), tuple(data_columns), str(token))


def _memo_index(
    table_path: str, data_columns: tuple[str, ...], token: str
) -> PointingIndex | None:
    """The index of the memo under (path, columns, token), read on a miss
    (once, when several threads miss it). Called holding the casatools
    lock (``casatools_serialized``), before the memo's build lock."""
    key = _index_key(table_path, data_columns, token)
    entry = POINTING_INDEX_MEMO.get(key)
    if entry is None:
        with POINTING_INDEX_MEMO.build_lock(key):
            entry = POINTING_INDEX_MEMO.get(key, count=False)
            if entry is None:
                POINTING_INDEX_MEMO.stats["reads"] += 1
                entry = _IndexEntry(
                    read_pointing_index(table_path, data_columns, token)
                )
                POINTING_INDEX_MEMO.put(key, entry)
    return entry.index


def open_pointing_index(
    table_path: str, data_columns: tuple[str, ...]
) -> PointingIndex | None:
    """
    The index of a POINTING table when an MS is opened: from the memo while
    the table's fingerprint is unchanged, else read.

    Returns
    -------
    PointingIndex | None
        None if there is no table, or its pointing_xds is built eagerly.
    """
    with casatools_serialized():
        token = pointing_table_token(table_path)
        if token is None:
            return None
        return _memo_index(table_path, tuple(data_columns), token)


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
        Cell shape of the column (numpy order).
    verified : bool
        Whether the column description fixes the cell shape (else it is the
        first row's: a read finds other shapes).
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
    verified: bool = True

    def changed_error(self, what: str) -> MSv2ChangedError:
        """The error raised when POINTING no longer matches the open."""
        return MSv2ChangedError(
            f"The POINTING table {self.table_path} changed since the MS was opened "
            f"({what}); open it again"
        )

    def index(self) -> PointingIndex:
        """The POINTING index (memo, or read again in another process)."""
        index = _memo_index(self.table_path, self.data_columns, self.index_token)
        if index is None:
            raise self.changed_error("its cells can no longer be read lazily")
        if index.nrows != self.nrows:
            raise self.changed_error(
                f"it has {index.nrows} rows, {self.nrows} when the MS was opened"
            )
        if (
            index.dtypes.get(self.col) != self.dtype
            or index.cell_shapes.get(self.col) != self.cell_shape
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


def deferred_pointing_generic_xds(
    in_file: str,
    time_min_max: tuple[np.float64, np.float64] | None,
    antenna_ids: np.ndarray,
    data_columns: Mapping[str, str],
    *,
    specs: dict[str, DeferredPointingVariable],
    context: dict | None = None,
) -> xr.Dataset | None:
    """
    The ``generic_loader`` of ``create_pointing_xds`` for the MSv2 backend:
    the generic pointing dataset of a partition (``pointing_generic_xds``)
    with placeholders (``stream_write.deferred_placeholder``) for its data
    variables, described in ``specs``. Its coordinates, dimensions and
    attributes are those of ``pointing_generic_xds``.

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
        Filled with the description of every placeholder, by data variable
        name.
    context : dict | None, optional
        Filled with the partition's ``time_min_max`` and ``antenna_ids``
        (for a pointing_xds that the converter's code builds at open:
        ``rebuilt_pointing_xds``).

    Returns
    -------
    xr.Dataset | None
        The dataset (empty if no POINTING row is selected), or None if
        POINTING is read eagerly.
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
        index = open_pointing_index(table_path, columns_key)
    else:
        # one fingerprint per open (the partitions of an open share the cache)
        index = subtable_cache.get_or_build(
            ("pointing_index", table_path, columns_key),
            lambda: open_pointing_index(table_path, columns_key),
        )
    if index is None:
        return None
    selected = select_partition(index, time_min_max, antenna_ids)
    if selected is None:
        return xr.Dataset()
    grid = selected[1]
    del selected

    columns = index.columns
    var_attrs = columns.var_attrs
    shape = (grid.utime.size, grid.uant.size)
    time_min_max = (float(time_min_max[0]), float(time_min_max[1]))
    antenna_ids = tuple(int(ant) for ant in antenna_ids)
    data_vars = {}
    for col in columns.data_columns:
        name = data_columns[col]
        spec = DeferredPointingVariable(
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
            dtype=index.dtypes[col],
            cell_shape=index.cell_shapes[col],
            verified=index.verified[col],
        )
        specs[name] = spec
        data_vars[col] = xr.Variable(
            ("TIME", "ANTENNA_ID") + columns.data_dims[col],
            deferred_placeholder(
                f"pointing-{name}", shape + spec.cell_shape, spec.dtype
            ),
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
                "bad_cols": list(columns.bad_cols),
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
    """

    def __init__(
        self,
        spec: DeferredPointingVariable,
        shape: tuple[int, ...],
        dtype: DTypeLike,
        node: str = "",
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

    def _message(self, key: tuple[slice, ...], exc: BaseException) -> str:
        block = ", ".join(f"{k.start}:{k.stop}" for k in key)
        message = (
            f"Reading the POINTING column {self.spec.col} for the data variable "
            f"{self.spec.name} of the pointing_xds of {self.node or 'an MSv4'} "
            f"(block [{block}]) from {self.spec.table_path} failed: "
            f"{type(exc).__name__}: {exc}"
        )
        if not self.spec.verified:
            message += (
                f". The cells of {self.spec.col} were taken to have the shape of its "
                f"first row, {self.spec.cell_shape}, when the MS was opened (only a "
                "read can check them): if they vary in shape, convert the MS with "
                "convert_msv2_to_processing_set (which reads such a table per "
                f"partition), or open it with drop_variables=[{self.spec.name!r}] "
                "or with_pointing=False"
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
                    for p0 in range(0, times.size, step):
                        sub_times = times[p0 : p0 + step]
                        grid, moved = self._read_grid(
                            table, index, selection, sub_times, antennas
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
    ) -> tuple[np.ndarray | None, str]:
        """
        The (len(times), len(antennas)) + cell grid of the times ``times`` and
        antennas ``antennas`` (sorted unique indices into the partition's
        grid): the first row (lowest row number) of every cell, pivoted as
        ``pointing_generic_xds`` pivots them; or (None, why) if rows no
        longer have the TIME and ANTENNA_ID of the index.
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
        moved = self._rows_moved(table, index, positions, rows)
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
        values = self._read_rows(table, first_rows)
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

    def _read_rows(self, table: Any, rows: np.ndarray) -> np.ndarray:
        """The cells of ``rows`` (in that order), read in ascending row order."""
        cell_shape = self.spec.cell_shape
        if rows.size == 0:
            return np.empty((0,) + cell_shape, dtype=self.spec.dtype)
        order = np.argsort(rows)
        sorted_values = self._read_sorted_rows(
            table, rows[order], self.spec.col, cell_shape, self.spec.dtype
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
    ) -> np.ndarray:
        """The cells of ascending ``rows`` of a column (read_column_rows, or in
        windows of consecutive rows for fragmented rows without in-place
        reads)."""
        if has_in_place_reads(table) or rows_to_runs(rows)[0].size <= FRAGMENTED_RUNS:
            return read_column_rows(table, col, rows)
        cell_bytes = max(1, int(np.prod(cell_shape, dtype=np.int64)) * dtype.itemsize)
        window_rows = max(1, _WINDOW_BYTES // cell_bytes)
        # a window ends where the next row is too far from its first row or
        # from the row before
        ends = np.flatnonzero(np.diff(rows) > _WINDOW_MAX_GAP) + 1
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
) -> xr.Dataset:
    """
    The pointing_xds of a partition built with ``deferred_pointing_generic_xds``
    with its placeholders replaced by lazily indexed arrays
    (:class:`PointingColumnArray`), keeping their dimensions, attributes and
    encoding.
    """
    lazy = {}
    for name, spec in specs.items():
        if name not in pointing_xds.data_vars:
            continue
        var = pointing_xds.variables[name]
        array = PointingColumnArray(spec, var.shape, var.dtype, node=node)
        lazy_var = xr.Variable(
            var.dims,
            xr.core.indexing.LazilyIndexedArray(array),
            attrs=copy.deepcopy(var.attrs),
        )
        lazy_var.encoding = dict(var.encoding)
        lazy[name] = lazy_var
    # (variables replaced in place: the order of the data variables is kept)
    return pointing_xds.assign(lazy)


# --- pointing_xds that the converter's code builds -------------------------------


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


class PointingBuildArray(MSv2BackendArray):
    """
    A data variable of a pointing_xds that only the converter's code can
    build (POINTING tables that ``read_pointing_index`` cannot describe):
    every read builds the partition's pointing_xds again
    (:class:`PointingBuild`) and returns the selection of the variable, so
    the values are not kept between reads.

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
    """

    def __init__(
        self,
        build: PointingBuild,
        name: str,
        shape: tuple[int, ...],
        dtype: DTypeLike,
        node: str = "",
    ):
        super().__init__(shape, dtype)
        self.build = build
        self.name = str(name)
        self.node = str(node)

    def _raw_indexing_method(self, key: tuple[slice, ...]) -> np.ndarray:
        table = os.path.join(self.build.in_file, POINTING_TABLE)
        try:
            xds = self.build.build()
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
        return np.asarray(var.values[key])


def rebuilt_pointing_xds(
    pointing_xds: xr.Dataset, build: PointingBuild, node: str = ""
) -> xr.Dataset:
    """
    A pointing_xds built at open by the converter's code with its data
    variables replaced by lazily indexed arrays that build it again when
    read (:class:`PointingBuildArray`), keeping their dimensions, attributes
    and encoding.
    """
    lazy = {}
    for name, var in pointing_xds.data_vars.items():
        array = PointingBuildArray(build, name, var.shape, var.dtype, node=node)
        lazy_var = xr.Variable(
            var.dims,
            xr.core.indexing.LazilyIndexedArray(array),
            attrs=copy.deepcopy(var.attrs),
        )
        lazy_var.encoding = dict(var.variable.encoding)
        lazy[name] = lazy_var
    return pointing_xds.assign(lazy)
