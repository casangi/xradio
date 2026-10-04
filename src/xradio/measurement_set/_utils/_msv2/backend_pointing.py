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
  numbers sorted by TIME, 16 bytes per row: PR 1's ``PointingColumns``
  without data columns) gives the partition's rows, its unique times
  (``time_pointing``) and antennas (``antenna_name``). The data variables
  are placeholders, described by a :class:`DeferredPointingVariable`, and
  the rest of the pointing_xds (dimension names, attributes, encoding) is
  built by the converter's own code. The data columns are only checked (as
  PR 1's cache checks them, in bounded reads, the values discarded): nothing
  of the data is kept, so memory at open grows with the index only. The
  index is kept in a per-process memo while the table's fingerprint
  (``table_fingerprint``) is unchanged.
- On access (:class:`PointingColumnArray`): the rows of a block of times
  and antennas, the first row (lowest row number) of every (time, antenna)
  cell, their values read in ascending row order, and the pivot with
  ``pivot_time_antenna`` (xarray's promoted fill value for missing cells).
  These are the selection, deduplication and pivot of the converter's
  cached path (``pointing_generic_xds``), restricted to the block, so the
  values are those of the converter's pointing_xds. Blocks are read in time
  sub-blocks of bounded size; every read opens and closes the table and,
  with casatools, holds the process-wide casatools lock (where fragmented
  rows are read in windows of consecutive rows: the shim has no row
  selection).

The lazy path is taken when the converter's sub-table cache would hold the
table (``read_pointing_columns``: one cell shape per column, usual value
types, no empty dimensions, finite TIME, ...), for which its cached and
uncached reads give identical pointing_xds. Otherwise, and with
``pointing_interpolate=True`` (the interpolation needs the values), the
pointing_xds is built eagerly, as by the converter.

Staleness: an array records the number of POINTING rows and a token of its
partition's selection (times, antennas, row numbers). A read finds the
index in the memo (in this process, the index of the open) or reads it
again (other processes); a POINTING table with another number of rows, or
a selection with another token, raises :class:`MSv2ChangedError`.
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
from xradio.measurement_set._utils._msv2._tables.read import convert_casacore_time
from xradio.measurement_set._utils._msv2._tables.read_pointing import (
    POINTING_TABLE,
    PointingColumns,
    pivot_time_antenna,
    read_pointing_columns,
    select_pointing_rows,
)
from xradio.measurement_set._utils._msv2._tables.read_rows import (
    FRAGMENTED_RUNS,
    has_in_place_reads,
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
)
from xradio.measurement_set._utils._msv2.backend_errors import (
    MSv2ChangedError,
    MSv2ReadError,
)
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
# have one of these types, see read_pointing_columns.)
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
        ``read_pointing_columns(..., keep_data=False)``.
    dtypes : dict[str, np.dtype]
        dtype of the values of every data column (as read).
    cell_shapes : dict[str, tuple[int, ...]]
        Cell shape of every data column (numpy order).
    token : str
        Token of the table's fingerprint when the index was read.
    """

    columns: PointingColumns
    dtypes: dict[str, np.dtype]
    cell_shapes: dict[str, tuple[int, ...]]
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


def _read_index(
    table_path: str, data_columns: tuple[str, ...], token: str
) -> PointingIndex | None:
    """The index of a POINTING table, or None if its pointing_xds is built
    eagerly (see the module docstring)."""
    columns = read_pointing_columns(table_path, data_columns, keep_data=False)
    if columns is None:
        return None
    dtypes, cell_shapes = {}, {}
    for col in columns.data_columns:
        # (the dtype and cell shape of the values: of one cell, read)
        one = columns.read_data(col, np.zeros(1, dtype=np.int64))
        dtypes[col], cell_shapes[col] = one.dtype, tuple(one.shape[1:])
        if not _exact_promotion(one.dtype):
            xradio_logger().debug(
                f"Not reading {table_path} lazily: {col} has values of {one.dtype}"
            )
            return None
    return PointingIndex(columns, dtypes, cell_shapes, token)


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
                entry = _IndexEntry(_read_index(table_path, data_columns, token))
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

    def selection(self, index: PointingIndex) -> _Selection:
        """The rows of the partition (memo, or selected again and checked
        against the token of the open)."""
        key = _index_key(self.table_path, self.data_columns, self.index_token) + (
            self.time_min_max,
            self.antenna_ids,
        )
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

    Returns
    -------
    xr.Dataset | None
        The dataset (empty if no POINTING row is selected), or None if
        POINTING is read eagerly.
    """
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
        return (
            f"Reading the POINTING column {self.spec.col} for the data variable "
            f"{self.spec.name} of the pointing_xds of {self.node or 'an MSv4'} "
            f"(block [{block}]) from {self.spec.table_path} failed: "
            f"{type(exc).__name__}: {exc}"
        )

    def _raw_indexing_method(self, key: tuple[slice, ...]) -> np.ndarray:
        time_key, antenna_key = key[0], key[1]
        cell_key = (slice(None), slice(None)) + tuple(key[2:])
        n_antennas = antenna_key.stop - antenna_key.start
        out = np.empty(tuple(k.stop - k.start for k in key), dtype=self.dtype)
        cell_elems = int(np.prod(self.spec.cell_shape, dtype=np.int64)) or 1
        cell_bytes = cell_elems * (2 * self.dtype.itemsize + 8) + _ROW_INDEX_BYTES
        step = max(1, POINTING_SUB_BLOCK_BYTES // (n_antennas * cell_bytes))
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
                    for t0 in range(time_key.start, time_key.stop, step):
                        t1 = min(time_key.stop, t0 + step)
                        grid = self._read_grid(
                            table,
                            index,
                            selection,
                            t0,
                            t1,
                            antenna_key.start,
                            antenna_key.stop,
                        )
                        grid = grid.reshape((t1 - t0, n_antennas) + self.shape[2:])
                        out[t0 - time_key.start : t1 - time_key.start] = grid[cell_key]
                        del grid
            except MSv2ChangedError:
                raise
            except Exception as exc:
                raise MSv2ReadError(self._message(key, exc)) from exc
            finally:
                table = None  # (casatools: released holding the lock)
        return out

    def _read_grid(
        self,
        table: Any,
        index: PointingIndex,
        selection: _Selection,
        t0: int,
        t1: int,
        a0: int,
        a1: int,
    ) -> np.ndarray:
        """
        The (t1 - t0, a1 - a0) + cell grid of the times [t0, t1) and
        antennas [a0, a1): the first row (lowest row number) of every cell,
        pivoted as ``pointing_generic_xds`` pivots them.
        """
        n_antennas = a1 - a0
        p0, p1 = int(selection.time_starts[t0]), int(selection.time_starts[t1])
        positions = selection.sel[p0:p1]
        acode = selection.acode[p0:p1]
        tcode = np.repeat(
            np.arange(t1 - t0, dtype=np.int64),
            np.diff(selection.time_starts[t0 : t1 + 1]),
        )
        if a0 > 0 or a1 < self.spec.n_antennas:
            keep = (acode >= a0) & (acode < a1)
            positions, acode, tcode = positions[keep], acode[keep], tcode[keep]
            del keep
        cell_key = tcode * n_antennas + (acode.astype(np.int64) - a0)
        del acode, tcode
        rows = index.columns.row[positions]
        del positions
        # first row (lowest row number) of every (time, antenna) cell
        order = np.lexsort((rows, cell_key))
        sorted_keys = cell_key[order]
        is_first = np.ones(order.size, dtype=bool)
        np.not_equal(sorted_keys[1:], sorted_keys[:-1], out=is_first[1:])
        first = order[is_first]
        del order, sorted_keys, is_first
        first_rows = np.asarray(rows[first], dtype=np.int64)
        tcode_first, acode_first = np.divmod(cell_key[first], n_antennas)
        del rows, cell_key, first
        values = self._read_rows(table, first_rows)
        return pivot_time_antenna(
            values, tcode_first, acode_first, (t1 - t0, n_antennas)
        )

    def _read_rows(self, table: Any, rows: np.ndarray) -> np.ndarray:
        """The cells of ``rows`` (in that order), read in ascending row order."""
        cell_shape = self.spec.cell_shape
        if rows.size == 0:
            return np.empty((0,) + cell_shape, dtype=self.spec.dtype)
        order = np.argsort(rows)
        sorted_values = self._read_sorted_rows(table, rows[order])
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

    def _read_sorted_rows(self, table: Any, rows: np.ndarray) -> np.ndarray:
        """The cells of ascending ``rows`` (read_column_rows, or in windows of
        consecutive rows for fragmented rows without in-place reads)."""
        col = self.spec.col
        if has_in_place_reads(table) or rows_to_runs(rows)[0].size <= FRAGMENTED_RUNS:
            return read_column_rows(table, col, rows)
        cell_shape, dtype = self.spec.cell_shape, self.spec.dtype
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
