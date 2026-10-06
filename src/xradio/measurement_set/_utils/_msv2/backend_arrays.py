"""
Lazily indexable arrays of the MSv2 xarray backend (engine ``xradio_msv2``).

The data variables of the main xds of an MSv4 that come from MAIN columns
(VISIBILITY*, SPECTRUM*, FLAG, WEIGHT, UVW, TIME_CENTROID,
EFFECTIVE_INTEGRATION_TIME) are not read when an MSv2 is opened:
:class:`MSv2MainColumnArray` reads the values of one data variable of one
partition when it is indexed, and :class:`OnesArray` stands for the WEIGHT=1
fallback (nothing to read). Both derive from :class:`MSv2BackendArray`, which
supports outer indexing: every key xarray passes is reduced to the sorted
unique indices to read along every dimension, and the rest of the key
(integers, reversals, unsorted or repeated indices) is applied to what was
read.

Values: only the rows of the selected (time, baseline) cells are read, in
time sub-blocks of whole cells (at most ``SUB_BLOCK_BYTES`` each, at least one
time), with the converter's own read primitive (``read_grid``: cells without
a row padded with ``get_pad_value``, FLAG=False and NaN; for duplicated
(time, baseline) rows the last row wins; bounded calls of ascending rows) and
the converter's transform (TIME_CENTROID epoch, WEIGHT repeated along
frequency); the selected cell elements are taken afterwards in numpy. So the
values are those the converter writes, for any key, and a read holds at most
its result plus one sub-block. Channel-sliced reads: of a column whose values
the converter writes as read (CHANNEL_SLICED_COLUMNS) and whose 2-D (chan,
pol) cells are stored in tiles of fewer channels than a cell
(``channel_tiling``, found when the MS is opened), python-casacore reads only
the channels of the selection's range rounded out to whole tiles, with a
bounded tile cache; the values are the same. Every read opens the MAIN table
by name and closes it (no handle is kept); with casatools it holds the
process-wide casatools lock (``casatools_serialized``) over the open, the
read, the close and the release of the table.

Rows: :class:`PartitionIndex` holds the MAIN rows of a partition as runs, the
number of MAIN rows and the (time, baseline) grid shape when the MS was
opened, and a token of the grid (unique times and baselines). The (time,
baseline) index of every row is kept in a bounded per-process memo: seeded
from the build when the MS is opened, rebuilt with the converter's
``calc_indx_for_row_split`` in another process (or after an eviction). A MAIN
table with another number of rows, or a rebuilt index with another token,
raises :class:`MSv2ChangedError`.

Changes after the open: while no write of MAIN's key columns (the data
managers of ROW_KEY_COLUMNS) or of FIELD, STATE and SOURCE was flushed since
the open (``keys_token``, from the lock files) and no handle of the process
has MAIN open for writing, a partition is as when the MS was opened. Else a
read first checks the whole partition against the MS as it is now
(``PartitionIndex.check``, once per state of the MS in a process): its rows
must be exactly the MAIN rows that are in it now (none moved to another
partition, none moved into it: from the grouping keys DATA_DESC_ID,
OBSERVATION_ID, OBS_MODE, EPHEMERIS_ID and those of the partition scheme, as
create_partitions derives them from MAIN, FIELD, SOURCE and STATE) and their
(time, baseline) grid that of the open; MSv2ChangedError if not. The rows the
read reads must then have the TIME, ANTENNA1 and ANTENNA2 of their cells in
the index it reads with; if not, the index is made again from the MS (as in
another process), so that the outcome of a read does not depend on what the
process kept in memory. So a read returns the current values of the rows of
its partition, or raises.

The arrays pickle to their path, runs, grouping keys and column description
(about 1 kB plus 16 bytes per run), never per-row arrays or table handles.
"""

import collections
import contextlib
import hashlib
import json
import os
import threading
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from numpy.typing import DTypeLike

from xradio._utils._casacore.tables import casatools_serialized
from xradio._utils.list_and_array import get_pad_value
from xradio._utils.logging import xradio_logger
from xradio.measurement_set._utils._msv2._tables.read import convert_casacore_time
from xradio.measurement_set._utils._msv2._tables.read_rows import (
    ColumnNotReadableError,
    ColumnStorage,
    MainTableRows,
    _scan_cell_shapes,
    has_in_place_reads,
    make_row_grid_plan,
    read_column_rows,
    read_grid,
    read_rows_to_grid,
    rows_to_runs,
    runs_to_rows,
)
from xradio.measurement_set._utils._msv2._tables.table_lock_file import (
    followed_ms_fingerprint,
)
from xradio.measurement_set._utils._msv2._tables.table_query import open_table_ro
from xradio.measurement_set._utils._msv2.backend_errors import (
    MSv2ChangedError,
    MSv2ReadError,
)
from xradio.measurement_set._utils._msv2.conversion import calc_indx_for_row_split
from xradio.measurement_set._utils._msv2.partition_cache import (
    FINGERPRINT_SUBTABLE_COLUMNS,
    FINGERPRINT_SUBTABLES,
)
from xradio.measurement_set._utils._msv2.partition_queries import (
    MANDATORY_PARTITION_KEYS,
    PARTITION_MAIN_KEY_COLUMNS,
    PartitionKeyMaps,
    _add_derived_columns,
    partition_key_maps,
)

# Largest time sub-block (bytes of whole cells, as read and as transformed)
# that one read of a lazy block holds, besides its result: a block of more
# times is read in several sub-blocks (at least one time each).
SUB_BLOCK_BYTES = 128 * 2**20
# The MAIN columns whose values the converter writes as it reads them
# (postprocess_main_column converts only TIME_CENTROID and WEIGHT): of their
# 2-D (chan, pol) cells, a range of channels can be read alone (channel-sliced
# reads, see MSv2MainColumnArray)
CHANNEL_SLICED_COLUMNS = frozenset(
    {"DATA", "CORRECTED_DATA", "MODEL_DATA", "FLOAT_DATA", "FLAG", "WEIGHT_SPECTRUM"}
)
# The tiled storage managers whose hypercubes have (pol, chan, row) tiles
CHANNEL_TILED_DM_TYPES = ("TiledShapeStMan", "TiledColumnStMan", "TiledDataStMan")
# Smallest maximum size (MiB) of the tile cache of a column that a
# channel-sliced read sets (casacore's setmaxcachesize)
CHANNEL_SLICE_MIN_CACHE_MIB = 16
# Bound of the per-process memo of partition indices (bytes of index arrays).
# The most recently used index is always kept.
INDEX_MEMO_MAX_BYTES = 256 * 2**20
# Number of locks that serialize the rebuilds of indices (by key hash), so
# that threads reading one partition rebuild its index once
INDEX_BUILD_LOCKS = 64
# The MAIN columns that place a row in the (time, baseline) grid
GRID_KEY_COLUMNS = ("TIME", "ANTENNA1", "ANTENNA2")
# The MAIN columns that place a row in a partition (with FIELD, SOURCE and
# STATE), but ANTENNA1, which the grid keys check
PARTITION_KEY_COLUMNS = tuple(
    name for name in PARTITION_MAIN_KEY_COLUMNS if name not in GRID_KEY_COLUMNS
)
# The MAIN columns whose writes since the open make reads check their rows
ROW_KEY_COLUMNS = GRID_KEY_COLUMNS + PARTITION_KEY_COLUMNS
# Rows of MAIN whose key columns one window of a partition check reads (about
# 20-28 bytes per row while it lasts)
CHECK_WINDOW_ROWS = 2**20
# Bound of the per-process memo of partition checks (entries)
CHECK_MEMO_MAX_ENTRIES = 4096
# The partitions (INDEX_MEMO keys) whose WEIGHT_SPECTRUM cells a read of
# WEIGHT found all readable in this process (MSv2MainColumnArray.
# _check_weight_spectrum_cells), and the bound of their number
WEIGHT_SPECTRUM_CHECKED: set = set()
WEIGHT_SPECTRUM_CHECKED_MAX_ENTRIES = 4096
# The MAIN key column of every partition key, and the PartitionKeyMaps lookup
# of the keys derived from it (create_partitions' _add_derived_columns)
_KEY_COLUMN = {
    "DATA_DESC_ID": "DATA_DESC_ID",
    "OBSERVATION_ID": "OBSERVATION_ID",
    "FIELD_ID": "FIELD_ID",
    "SCAN_NUMBER": "SCAN_NUMBER",
    "STATE_ID": "STATE_ID",
    "ANTENNA1": "ANTENNA1",
    "SOURCE_ID": "FIELD_ID",
    "EPHEMERIS_ID": "FIELD_ID",
    "OBS_MODE": "STATE_ID",
    "SUB_SCAN_NUMBER": "STATE_ID",
}
_KEY_LOOKUP = {
    "SOURCE_ID": "field_source",
    "EPHEMERIS_ID": "field_ephemeris",
    "OBS_MODE": "state_obs_mode",
    "SUB_SCAN_NUMBER": "state_sub_scan",
}


# --- the base class ------------------------------------------------------------
# Adapted from the ASDM backend (_asdm/asdm_backend_arrays.py), which supports
# basic indexing only (the bounding block of a selection is read); here outer
# indexing (TODO: unify with _asdm/asdm_backend_arrays.py once both backends
# are merged).


def normalize_outer_key(
    key: tuple, shape: tuple[int, ...]
) -> tuple[tuple[np.ndarray, ...], list, tuple[int, ...]]:
    """
    Split an outer key (one int, slice or 1-D integer array per dimension)
    into the sorted unique indices to read along every dimension and the
    numpy indexing that turns those into the selection.

    Parameters
    ----------
    key : tuple
        Outer indexing key (missing trailing dimensions are selected
        entirely). Negative indices count from the end.
    shape : tuple[int, ...]
        Shape of the indexed array.

    Returns
    -------
    tuple[tuple[np.ndarray, ...], list, tuple[int, ...]]
        - selections: per dimension, the indices to read (int64, ascending,
          unique, in range).
        - post: per dimension, what to apply to the read values along it:
          ``0`` (an integer key: the dimension is dropped), ``slice(None)``,
          ``slice(None, None, -1)`` (a negative-step slice) or an int64
          array of positions into ``selections`` (unsorted or repeated array
          keys).
        - result_shape: shape of the selection.

    Raises
    ------
    IndexError
        If there are too many indices or an index is out of bounds.
    TypeError
        If an index is not an int, a slice or a 1-D integer array.
    """
    if not isinstance(key, tuple):
        key = (key,)
    if len(key) > len(shape):
        raise IndexError(
            f"Too many indices ({len(key)}) for an array with {len(shape)} dimensions"
        )
    key = key + (slice(None),) * (len(shape) - len(key))

    selections, post, result_shape = [], [], []
    for dim_key, dim_len in zip(key, shape, strict=True):
        if isinstance(dim_key, int | np.integer) and not isinstance(
            dim_key, bool | np.bool_
        ):
            index = int(dim_key)
            if not -dim_len <= index < dim_len:
                raise IndexError(
                    f"Index {index} is out of bounds for a dimension of size {dim_len}"
                )
            selections.append(np.array([index % dim_len], dtype=np.int64))
            post.append(0)
        elif isinstance(dim_key, slice):
            selected = range(dim_len)[dim_key]
            result_shape.append(len(selected))
            if selected.step > 0:
                selections.append(
                    np.arange(selected.start, selected.stop, selected.step)
                )
                post.append(slice(None))
            else:
                selections.append(
                    np.arange(selected.start, selected.stop, selected.step)[::-1].copy()
                )
                post.append(slice(None, None, -1))
        elif isinstance(dim_key, np.ndarray | list | tuple):
            indices = np.asarray(dim_key)
            if indices.ndim != 1 or (indices.size and indices.dtype.kind not in "iu"):
                raise TypeError(
                    f"Unsupported outer index {dim_key!r}: only 1-D integer arrays"
                )
            indices = indices.astype(np.int64)
            if indices.size and (indices.min() < -dim_len or indices.max() >= dim_len):
                raise IndexError(
                    f"Indices {indices} are out of bounds for a dimension of size "
                    f"{dim_len}"
                )
            indices = np.where(indices < 0, indices + dim_len, indices)
            unique, positions = np.unique(indices, return_inverse=True)
            selections.append(unique)
            result_shape.append(indices.size)
            if unique.size == indices.size and np.all(np.diff(indices) > 0):
                post.append(slice(None))
            else:
                post.append(positions.reshape(-1).astype(np.int64))
        else:
            raise TypeError(
                f"Unsupported index {dim_key!r} of type {type(dim_key)}: only "
                "integers, slices and 1-D integer arrays are supported"
            )
    return tuple(selections), post, tuple(result_shape)


def apply_outer_post(values: np.ndarray, post: list) -> np.ndarray:
    """Apply the ``post`` of normalize_outer_key to the values read for its
    ``selections`` (outer semantics: every dimension on its own)."""
    for axis, dim_post in enumerate(post):
        if isinstance(dim_post, np.ndarray):
            values = np.take(values, dim_post, axis=axis)
    basic = tuple(
        slice(None) if isinstance(dim_post, np.ndarray) else dim_post
        for dim_post in post
    )
    if any(dim_post != slice(None) for dim_post in basic):
        values = values[basic]
    return values


def bounding_slices(selections: tuple[np.ndarray, ...]) -> tuple[slice, ...]:
    """One ``slice(first, last + 1, 1)`` per dimension (non-empty sorted
    selections)."""
    return tuple(
        slice(int(selected[0]), int(selected[-1]) + 1, 1) for selected in selections
    )


def is_range(selected: np.ndarray) -> bool:
    """Whether sorted unique indices are consecutive."""
    return (
        selected.size == 0 or int(selected[-1]) - int(selected[0]) + 1 == selected.size
    )


def take_from_block(
    block: np.ndarray, selections: tuple[np.ndarray, ...], offsets
) -> np.ndarray:
    """The ``selections`` (indices relative to ``offsets``) of a block read
    for them, along every dimension (outer); slices where they are
    consecutive (no copy)."""
    basic = []
    for axis, (selected, offset) in enumerate(zip(selections, offsets, strict=True)):
        if is_range(selected):
            basic.append(
                slice(int(selected[0]) - offset, int(selected[-1]) + 1 - offset)
            )
        else:
            block = np.take(block, selected - offset, axis=axis)
            basic.append(slice(None))
    return block[tuple(basic)]


def _replace_empty_slices(
    key: xr.core.indexing.ExplicitIndexer, shape: tuple[int, ...]
) -> xr.core.indexing.ExplicitIndexer:
    """
    Replace slices that select nothing by ``slice(0, 0)``. xarray's key
    decomposition fails (IndexError) on empty slices with a negative step.
    """
    empty = [
        isinstance(dim_key, slice) and len(range(dim_len)[dim_key]) == 0
        for dim_key, dim_len in zip(key.tuple, shape, strict=True)
    ]
    if not any(empty):
        return key
    return type(key)(
        tuple(
            slice(0, 0) if is_empty else dim_key
            for dim_key, is_empty in zip(key.tuple, empty, strict=True)
        )
    )


class MSv2BackendArray(xr.backends.BackendArray):
    """
    Base class of the lazily indexable MSv2 backend arrays.

    xarray indexes it with outer keys (``IndexingSupport.OUTER``: per
    dimension an int, a slice or an integer array; vectorized keys are
    decomposed by xarray into an outer key and a numpy key applied to the
    result). :meth:`__getitem__` reduces every key to the sorted unique
    indices to read along every dimension (:func:`normalize_outer_key`) and
    calls :meth:`_read_selection` with them; integer keys (squeezed),
    negative steps, unsorted and repeated indices are applied to its result,
    empty selections skip it. The result is cast to the declared dtype and
    its shape checked.

    Subclasses implement :meth:`_raw_indexing_method`, which receives a
    normalised block key: a tuple with one ``slice(start, stop, 1)`` per
    dimension, with Python ints and ``0 <= start < stop <= dim_len``, and
    returns an array with exactly ``stop - start`` elements along every
    dimension. By default :meth:`_read_selection` reads the bounding block of
    the selection with it; subclasses that can read only the selected
    indices (the MAIN and POINTING columns: only the selected times and
    baselines or antennas) override :meth:`_read_selection`.

    Parameters
    ----------
    shape : tuple[int, ...]
        Shape of the array.
    dtype : DTypeLike
        Declared dtype. The arrays returned by indexing always have this dtype.
    """

    def __init__(self, shape: tuple[int, ...], dtype: DTypeLike):
        shape = tuple(int(dim_len) for dim_len in shape)
        if any(dim_len < 0 for dim_len in shape):
            raise ValueError(f"Invalid (negative) array shape: {shape}")
        self._shape = shape
        self._dtype = np.dtype(dtype)

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    def __getitem__(self, key: xr.core.indexing.ExplicitIndexer) -> np.ndarray:
        """
        Makes the MSv2 backend arrays indexable (subscriptable with []), via the
        LazilyIndexedArray wrapper class.

        'key' is an explicit indexer from xarray. Vectorized indexers are
        decomposed by xarray into an outer indexer (handled by
        :meth:`_getitem_outer`) and a NumPy indexer applied to its result.
        """
        return xr.core.indexing.explicit_indexing_adapter(
            _replace_empty_slices(key, self.shape),
            self.shape,
            xr.core.indexing.IndexingSupport.OUTER,
            self._getitem_outer,
        )

    def _getitem_outer(self, key: tuple) -> np.ndarray:
        """Index with an outer key, see the class docstring."""
        selections, post, result_shape = normalize_outer_key(key, self.shape)
        if any(selected.size == 0 for selected in selections):
            return np.empty(result_shape, dtype=self.dtype)
        expected = tuple(selected.size for selected in selections)
        try:
            values = np.asarray(self._read_selection(selections))
        except Exception as exc:
            message = str(exc).strip()
            summary = message.splitlines()[0][:300] if message else ""
            what = ", ".join(_describe_selection(selected) for selected in selections)
            xradio_logger().warning(
                f"Exception while loading {type(self).__name__} [{what}] (for "
                f"{key=}): {type(exc).__name__}: {summary}"
            )
            raise
        if values.shape != expected:
            raise RuntimeError(
                f"{type(self).__name__}: the loader returned an array of shape "
                f"{values.shape} for a selection of shape {expected}"
            )
        result = apply_outer_post(values, post)
        trivial = all(
            isinstance(dim_post, slice) and dim_post == slice(None) for dim_post in post
        )
        result = self._cast(result, copy=not trivial)
        if result.shape != result_shape:
            raise RuntimeError(
                f"{type(self).__name__}: indexing with {key=} produced shape "
                f"{result.shape}, expected {result_shape}"
            )
        return result

    def _cast(self, values: np.ndarray, copy: bool) -> np.ndarray:
        """
        Cast to the declared dtype. Returns a writeable array that does not keep
        a larger loaded block alive (copy when values is a view into it).
        """
        if np.iscomplexobj(values) and self.dtype.kind != "c":
            raise TypeError(
                f"{type(self).__name__}: the loader returned complex values "
                f"({values.dtype}) for an array declared as {self.dtype}"
            )
        if copy or values.dtype != self.dtype or not values.flags.writeable:
            return np.array(values, dtype=self.dtype)
        return values

    def _read_selection(self, selections: tuple[np.ndarray, ...]) -> np.ndarray:
        """
        The values at the outer product of ``selections`` (per dimension,
        sorted unique indices, none empty), of shape ``tuple(len(s) for s in
        selections)``. By default the bounding block, read with
        :meth:`_raw_indexing_method`, then selected.
        """
        block_key = bounding_slices(selections)
        block = np.asarray(self._raw_indexing_method(block_key))
        block_shape = tuple(dim_key.stop - dim_key.start for dim_key in block_key)
        if block.shape != block_shape:
            raise RuntimeError(
                f"{type(self).__name__}: the loader returned an array of shape "
                f"{block.shape} for the block {block_key}, expected {block_shape}"
            )
        return take_from_block(
            block, selections, [dim_key.start for dim_key in block_key]
        )

    def _raw_indexing_method(self, key: tuple[slice, ...]) -> np.ndarray:
        """Load the block given by a normalised key (one step-1 slice per dim)."""
        raise NotImplementedError


def _describe_selection(selected: np.ndarray) -> str:
    """A short description of the indices read along a dimension."""
    if is_range(selected):
        return f"{int(selected[0])}:{int(selected[-1]) + 1}"
    return f"{selected.size} of {int(selected[0])}..{int(selected[-1])}"


# --- the (time, baseline) index of the rows of a partition -------------------


def index_token(
    utime: np.ndarray,
    baseline_ant1: np.ndarray,
    baseline_ant2: np.ndarray,
    shape: tuple[int, int],
) -> str:
    """
    Token of the (time, baseline) grid of a partition: blake2b-128 of its
    shape, unique times and baselines (``calc_indx_for_row_split``). An index
    rebuilt from the MS must give the token it had when the MS was opened.
    """
    digest = hashlib.blake2b(digest_size=16)
    digest.update(np.asarray(shape, dtype=np.int64).tobytes())
    for values, dtype in (
        (utime, np.float64),
        (baseline_ant1, np.int64),
        (baseline_ant2, np.int64),
    ):
        values = np.ascontiguousarray(values, dtype=dtype)
        digest.update(np.int64(values.size).tobytes())
        digest.update(values.tobytes())
    return digest.hexdigest()


def _runs_digest(starts: np.ndarray, lengths: np.ndarray) -> str:
    digest = hashlib.blake2b(digest_size=16)
    digest.update(np.ascontiguousarray(starts, dtype=np.int64).tobytes())
    digest.update(np.ascontiguousarray(lengths, dtype=np.int64).tobytes())
    return digest.hexdigest()


def partition_grouping(
    partition_info: dict, partition_scheme: tuple[str, ...] | list[str] = ()
) -> tuple[tuple[str, Any], ...]:
    """
    The grouping keys of a partition and their values, which every row of
    the partition has (ANTENNA1: and its ANTENNA2, autocorrelations only):
    MANDATORY_PARTITION_KEYS and the keys of the scheme that its description
    has with one value (None, a key the MS does not have, is left out).
    """
    grouping = []
    for key in dict.fromkeys((*MANDATORY_PARTITION_KEYS, *partition_scheme)):
        values = partition_info.get(key)
        if values is None or len(values) != 1:
            continue
        value = values[0]
        if value is None:
            continue
        grouping.append((key, value.item() if isinstance(value, np.generic) else value))
    return tuple(grouping)


def keys_token(ms_path: str) -> str | None:
    """
    A token of what the partitions of an MS and their (time, baseline) grids
    are made from (``followed_ms_fingerprint``): the storage of MAIN's
    ROW_KEY_COLUMNS (its rows, columns and data managers, and the change
    counters (lock file) and files of the data managers that hold them) and
    the FIELD, STATE and SOURCE tables (rows, columns, all change counters and
    files). A token equal to an earlier one means that no write of them was
    flushed since. None if that cannot be told: a lock file of these tables
    cannot be read, or their files do not follow their rows (a reference or
    concatenated MAIN, key columns in a data manager without files of its
    own, e.g. forwarded to another MS).
    """
    fingerprint, unfollowed = followed_ms_fingerprint(
        ms_path,
        ROW_KEY_COLUMNS,
        FINGERPRINT_SUBTABLES,
        FINGERPRINT_SUBTABLE_COLUMNS,
    )
    tables_found = [fingerprint["main"]] + [
        fingerprint[name] for name in FINGERPRINT_SUBTABLES if fingerprint[name]
    ]
    if unfollowed is not None or not all(found["lock_ok"] for found in tables_found):
        return None
    text = json.dumps(fingerprint, sort_keys=True, default=str)
    return hashlib.blake2b(text.encode(), digest_size=16).hexdigest()


def current_keys_token(table: Any, ms_path: str) -> str | None:
    """
    ``keys_token`` of an MS whose MAIN table is opened as ``table``; None
    (unknown) while a handle of this process has MAIN open for writing: its
    writes are seen by this process before they are flushed (``iswritable``
    of a read-only handle tells).
    """
    try:
        if table.iswritable():
            return None
    except Exception:
        return None
    return keys_token(ms_path)


class _CheckMemo:
    """
    Per-process LRU memo of the outcome of whole-partition checks
    (``PartitionIndex.check``): "" or why the partition changed, keyed by the
    partition (its index key, grouping keys and scheme) and the keys_token of
    the MS when it was checked; at most CHECK_MEMO_MAX_ENTRIES. Striped locks
    let threads reading one partition check it once. Renewed (empty, with new
    locks) in a fork child.
    """

    def __init__(self, max_entries: int):
        self.max_entries = int(max_entries)
        self.reset_after_fork()

    def reset_after_fork(self) -> None:
        self._lock = threading.Lock()
        self._check_locks = [threading.Lock() for _ in range(16)]
        self._entries: collections.OrderedDict[tuple, str] = collections.OrderedDict()
        self.stats: collections.Counter = collections.Counter()

    def check_lock(self, key: tuple) -> threading.Lock:
        return self._check_locks[hash(key) % len(self._check_locks)]

    def get(self, key: tuple) -> str | None:
        with self._lock:
            found = self._entries.get(key)
            if found is not None:
                self._entries.move_to_end(key)
            return found

    def put(self, key: tuple, outcome: str) -> None:
        with self._lock:
            self._entries[key] = outcome
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self.stats.clear()

    def __len__(self) -> int:
        return len(self._entries)


CHECK_MEMO = _CheckMemo(CHECK_MEMO_MAX_ENTRIES)


def _max_cache_mib(table: Any, col: str) -> int | None:
    """The maximum tile cache size (MiB, 0: none) of the data manager of a
    column as it is now in the process (its MaxCacheSize property), or None
    if it cannot be told."""
    try:
        return int(table.getdmprop(col)["MaxCacheSize"])
    except Exception:
        return None


class _TileCacheBounds:
    """
    The bounds (MiB) of the tile caches of the MAIN columns that
    channel-sliced reads of this process set while they read
    (``MSv2MainColumnArray._read_with``, see ``_channel_cache_mib``).
    python-casacore shares one table object per table in a process, so a
    bound (``setmaxcachesize``) applies to every handle of MAIN open in the
    process while it is set, and stays set as long as one of them is open.
    So the first of the reads of a column under way records the maximum
    the column had before (``_max_cache_mib``; no bound is set if it cannot
    be told), the largest bound of the reads under way is set, and the last
    of them to end sets the maximum before again. Renewed (no read under
    way) in a fork child.
    """

    def __init__(self):
        self.reset_after_fork()

    def reset_after_fork(self) -> None:
        self._lock = threading.Lock()
        # (MAIN path, column) -> (bounds of the reads under way, maximum before)
        self._active: dict[tuple[str, str], tuple[list[int], int]] = {}

    def __len__(self) -> int:
        return len(self._active)

    @contextlib.contextmanager
    def bounded(self, table: Any, ms_path: str, col: str, mib: int):
        """Bound the tile cache of ``col`` to ``mib`` (MiB, or the largest
        bound of the other reads of the column under way) while the block
        runs; the table must stay open until it ends."""
        key = (os.path.realpath(ms_path), str(col))
        mib = int(mib)
        with self._lock:
            entry = self._active.get(key)
            if entry is None:
                before = _max_cache_mib(table, col)
                if before is None:
                    entry = None
                else:
                    entry = ([], before)
            if entry is not None:
                bounds = entry[0] + [mib]
                table.setmaxcachesize(col, max(bounds))
                entry[0].append(mib)
                self._active[key] = entry
        try:
            yield
        finally:
            if entry is not None:
                with self._lock:
                    bounds, before = entry
                    bounds.remove(mib)
                    if not bounds:
                        del self._active[key]
                    try:
                        table.setmaxcachesize(col, max(bounds) if bounds else before)
                    except Exception as exc:  # (the values are read)
                        xradio_logger().debug(
                            f"The tile cache bound of {col} of {ms_path} was not "
                            f"reset: {exc}"
                        )


TILE_CACHE_BOUNDS = _TileCacheBounds()


def _runs_mask(starts: np.ndarray, ends: np.ndarray, r0: int, r1: int) -> np.ndarray:
    """Which rows of [r0, r1) are in the runs [starts, ends) (ascending,
    disjoint)."""
    i0 = int(np.searchsorted(ends, r0, side="right"))
    i1 = int(np.searchsorted(starts, r1, side="left"))
    delta = np.zeros(r1 - r0 + 1, dtype=np.int64)
    np.add.at(delta, np.clip(starts[i0:i1], r0, r1) - r0, 1)
    np.add.at(delta, np.clip(ends[i0:i1], r0, r1) - r0, -1)
    return np.cumsum(delta[:-1]) > 0


def available_partition_keys(maps: PartitionKeyMaps) -> set[str]:
    """The partition keys an MS has (create_partitions groups by those of
    the scheme), with these FIELD / SOURCE / STATE lookups."""
    frame = pd.DataFrame(
        {name: np.zeros(0, dtype=np.int32) for name in PARTITION_MAIN_KEY_COLUMNS}
    )
    _add_derived_columns(frame, maps)
    return set(frame.columns)


def _key_matches(
    column: np.ndarray, key: str, value: Any, maps: PartitionKeyMaps
) -> np.ndarray | str:
    """
    Which rows have the partition key ``key`` equal to ``value``, from the
    values of its MAIN key column (``_KEY_COLUMN``); a derived key is looked
    up in FIELD / SOURCE / STATE as create_partitions looks it up (numpy
    indexing: a negative STATE_ID counts from the last STATE row). Or why
    that cannot be told: a value beyond the rows of the looked up table
    (create_partitions fails on it).
    """
    if key not in _KEY_LOOKUP:
        return column == value
    lookup = np.asarray(getattr(maps, _KEY_LOOKUP[key]))
    n = lookup.shape[0]
    outside = (column < -n) | (column >= n)
    if outside.any():
        return (
            f"cannot be derived: {_KEY_COLUMN[key]} {int(column[np.argmax(outside)])} "
            f"is beyond the {n} rows of its table"
        )
    hits = np.flatnonzero(lookup == value)
    return np.isin(column, np.concatenate([hits, hits - n]))


class _IndexEntry:
    """
    The (time, baseline) index of the rows of a partition, by time: rows
    (int64, ascending), their time and baseline indices (int32), the order of
    the rows by time (int64, None if the rows are time-ordered) and, for every
    time index t, the first position of that order with a time index >= t.
    Also the grid's unique times (Unix seconds) and the antennas of its
    baselines, against which a read checks the rows it reads (None: not
    checked, for indices made without them in tests).
    """

    __slots__ = (
        "rows",
        "tidxs",
        "bidxs",
        "order",
        "time_bounds",
        "utime",
        "baseline_ant1",
        "baseline_ant2",
        "nbytes",
    )

    def __init__(
        self,
        rows: np.ndarray,
        tidxs: np.ndarray,
        bidxs: np.ndarray,
        n_times: int,
        utime: np.ndarray | None = None,
        baseline_ant1: np.ndarray | None = None,
        baseline_ant2: np.ndarray | None = None,
    ):
        self.rows = np.array(rows, dtype=np.int64)
        self.tidxs = np.array(tidxs, dtype=np.int32)
        self.bidxs = np.array(bidxs, dtype=np.int32)
        if not self.rows.shape == self.tidxs.shape == self.bidxs.shape:
            raise ValueError(
                f"Got {self.tidxs.size} time and {self.bidxs.size} baseline indices "
                f"for {self.rows.size} rows"
            )
        if utime is None:
            self.utime = self.baseline_ant1 = self.baseline_ant2 = None
        else:
            self.utime = np.array(utime, dtype=np.float64)
            self.baseline_ant1 = np.array(baseline_ant1, dtype=np.int32)
            self.baseline_ant2 = np.array(baseline_ant2, dtype=np.int32)
        if self.tidxs.size < 2 or bool(np.all(np.diff(self.tidxs) >= 0)):
            self.order = None
            sorted_tidxs = self.tidxs
        else:
            self.order = np.argsort(self.tidxs, kind="stable").astype(np.int64)
            sorted_tidxs = self.tidxs[self.order]
        self.time_bounds = np.searchsorted(
            sorted_tidxs, np.arange(int(n_times) + 1)
        ).astype(np.int64)
        self.nbytes = sum(
            values.nbytes
            for values in (
                self.rows,
                self.tidxs,
                self.bidxs,
                self.time_bounds,
                self.order,
                self.utime,
                self.baseline_ant1,
                self.baseline_ant2,
            )
            if values is not None
        )


class _IndexMemo:
    """
    Per-process LRU memo of partition indices, bounded by bytes (the most
    recently used entry is always kept), keyed by ``PartitionIndex.memo_key``
    (no per-open part: re-opening an MS gives the same keys), with striped
    locks for the rebuilds (``build_lock``). Renewed (empty, with new locks)
    in a fork child.
    """

    def __init__(self, max_bytes: int):
        self.max_bytes = int(max_bytes)
        self.reset_after_fork()

    def get(self, key: tuple, count: bool = True) -> _IndexEntry | None:
        with self._lock:
            entry = self._entries.get(key)
            if count:
                self.stats["misses" if entry is None else "hits"] += 1
            if entry is not None:
                self._entries.move_to_end(key)
            return entry

    def build_lock(self, key: tuple) -> threading.Lock:
        """The lock held while the index of ``key`` is rebuilt (one of
        INDEX_BUILD_LOCKS, by hash). It is never taken holding the casatools
        lock (the rebuild takes that one)."""
        return self._build_locks[hash(key) % len(self._build_locks)]

    def put(self, key: tuple, entry: _IndexEntry) -> None:
        with self._lock:
            old = self._entries.pop(key, None)
            if old is not None:
                self._nbytes -= old.nbytes
            self._entries[key] = entry
            self._nbytes += entry.nbytes
            while self._nbytes > self.max_bytes and len(self._entries) > 1:
                _, evicted = self._entries.popitem(last=False)
                self._nbytes -= evicted.nbytes
                self.stats["evictions"] += 1

    def discard(self, key: tuple, entry: Any = None) -> None:
        """Remove the entry of ``key`` (only if it is ``entry``, when given:
        another thread may have replaced it meanwhile)."""
        with self._lock:
            found = self._entries.get(key)
            if found is not None and (entry is None or found is entry):
                del self._entries[key]
                self._nbytes -= found.nbytes
                self.stats["discards"] += 1

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._nbytes = 0
            self.stats.clear()

    @property
    def nbytes(self) -> int:
        return self._nbytes

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, key: tuple) -> bool:
        return key in self._entries

    def reset_after_fork(self) -> None:
        """Empty, with new locks (registered with os.register_at_fork)."""
        self._lock = threading.Lock()
        self._build_locks = [threading.Lock() for _ in range(INDEX_BUILD_LOCKS)]
        self._entries: collections.OrderedDict[tuple, _IndexEntry] = (
            collections.OrderedDict()
        )
        self._nbytes = 0
        self.stats: collections.Counter = collections.Counter()


INDEX_MEMO = _IndexMemo(INDEX_MEMO_MAX_BYTES)

if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=INDEX_MEMO.reset_after_fork)
    os.register_at_fork(after_in_child=CHECK_MEMO.reset_after_fork)
    os.register_at_fork(after_in_child=TILE_CACHE_BOUNDS.reset_after_fork)


def clear_index_memo() -> None:
    """Empty the per-process memos of partition indices and checks (for
    tests: as in another process)."""
    INDEX_MEMO.clear()
    CHECK_MEMO.clear()
    WEIGHT_SPECTRUM_CHECKED.clear()


class PartitionIndex:
    """
    The MAIN rows of a partition and its (time, baseline) grid, as when the MS
    was opened: the rows as runs, the number of MAIN rows, the grid shape and
    its token (``index_token``), and the keys that select its rows. The (time,
    baseline) index of every row is taken from the per-process memo
    (``INDEX_MEMO``), or rebuilt from the MS (``calc_indx_for_row_split``, the
    converter's own) and checked against the shape and token. Pickles to
    these fields only.

    Parameters
    ----------
    ms_path : str
        Absolute path of the MS.
    starts, lengths : np.ndarray
        Runs of the partition's MAIN rows (ascending).
    main_nrows : int
        Rows of the MAIN table when the MS was opened.
    shape : tuple[int, int]
        (n_times, n_baselines) of the partition's grid.
    token : str
        ``index_token`` of the grid.
    keys_token : str | None, optional
        ``keys_token`` of the MS (MAIN's ROW_KEY_COLUMNS: the grid's TIME,
        ANTENNA1, ANTENNA2 and the partition key columns; FIELD, STATE and
        SOURCE) taken before the partitions were computed: while it is
        unchanged, the partition is as when the MS was opened. None: always
        checked (``check``).
    grouping : tuple[tuple[str, Any], ...], optional
        ``partition_grouping`` of the partition: the keys that select its
        rows (``check``: the MAIN rows that have them must be its rows).
    scheme : tuple[str, ...], optional
        The partition scheme (with MANDATORY_PARTITION_KEYS, the keys the MS
        must still have exactly those of ``grouping`` of).
    """

    __slots__ = (
        "ms_path",
        "starts",
        "lengths",
        "main_nrows",
        "shape",
        "token",
        "keys_token",
        "grouping",
        "scheme",
    )

    def __init__(
        self,
        ms_path: str,
        starts: np.ndarray,
        lengths: np.ndarray,
        main_nrows: int,
        shape: tuple[int, int],
        token: str,
        keys_token: str | None = None,
        grouping: tuple[tuple[str, Any], ...] = (),
        scheme: tuple[str, ...] | list[str] = (),
    ):
        self.ms_path = str(ms_path)
        self.starts = np.asarray(starts, dtype=np.int64)
        self.lengths = np.asarray(lengths, dtype=np.int64)
        self.main_nrows = int(main_nrows)
        self.shape = tuple(int(n) for n in shape)
        self.token = str(token)
        self.keys_token = None if keys_token is None else str(keys_token)
        self.grouping = tuple((str(key), value) for key, value in grouping)
        self.scheme = tuple(str(key) for key in scheme)
        if len(self.shape) != 2:
            raise ValueError(f"A (time, baseline) shape is expected, got {shape}")

    @classmethod
    def seed(
        cls,
        ms_path: str,
        built: Any,
        keys_token: str | None = None,
        grouping: tuple[tuple[str, Any], ...] = (),
        scheme: tuple[str, ...] | list[str] = (),
    ) -> "PartitionIndex":
        """
        The index of a partition built by ``conversion.build_partition`` (a
        ``BuiltPartition``, inside its context), with the build's (time,
        baseline) indices put in the memo: in this process the reads use
        exactly the build's index. ``keys_token``: the ``keys_token`` of the
        MS taken before the partitions were computed; ``grouping``: the
        partition's ``partition_grouping``; ``scheme``: the partition scheme.
        """
        rows = built.main_rows.rows
        starts, lengths = rows_to_runs(rows)
        index = cls(
            ms_path,
            starts,
            lengths,
            built.main_rows.table.nrows(),
            built.time_baseline_shape,
            index_token(
                built.utime,
                built.baseline_ant1,
                built.baseline_ant2,
                built.time_baseline_shape,
            ),
            keys_token,
            grouping,
            scheme,
        )
        INDEX_MEMO.put(
            index.memo_key(),
            _IndexEntry(
                rows,
                built.tidxs,
                built.bidxs,
                index.shape[0],
                built.utime,
                built.baseline_ant1,
                built.baseline_ant2,
            ),
        )
        return index

    def __getstate__(self) -> dict:
        return {name: getattr(self, name) for name in self.__slots__}

    def __setstate__(self, state: dict) -> None:
        for name, value in state.items():
            setattr(self, name, value)

    @property
    def nrows(self) -> int:
        """Number of rows of the partition."""
        return int(self.lengths.sum())

    def memo_key(self) -> tuple:
        """Key of the index in INDEX_MEMO."""
        return (
            os.path.realpath(self.ms_path),
            self.main_nrows,
            _runs_digest(self.starts, self.lengths),
            self.token,
        )

    def entry(self) -> _IndexEntry:
        """The index of the rows (from the memo, or rebuilt from the MS: once,
        when several threads miss it)."""
        key = self.memo_key()
        entry = INDEX_MEMO.get(key)
        if entry is None:
            with INDEX_MEMO.build_lock(key):
                # (another thread may have rebuilt it meanwhile)
                entry = INDEX_MEMO.get(key, count=False)
                if entry is None:
                    entry = self._rebuild()
                    INDEX_MEMO.put(key, entry)
        return entry

    def changed_error(self, what: str) -> MSv2ChangedError:
        """The error raised when the MS no longer matches this index."""
        return MSv2ChangedError(
            f"{self.ms_path} changed since it was opened ({what}); open it again"
        )

    def verify_current(self) -> None:
        """
        Raise MSv2ChangedError if the partition is no longer the one of the
        open: MAIN has another number of rows, or ``check`` finds that its rows
        or grid changed. For the lazy arrays of a partition that read no MAIN
        rows (OnesArray) or other tables (the pointing_xds, whose rows are
        selected by the time range and antennas of the partition's rows).
        """
        with casatools_serialized():
            table = None
            try:
                with open_table_ro(self.ms_path) as table:
                    main_nrows = table.nrows()
                    if main_nrows != self.main_nrows:
                        raise self.changed_error(
                            f"the MAIN table has {main_nrows} rows, "
                            f"{self.main_nrows} when it was opened"
                        )
                    self.check(table)
            finally:
                table = None  # (casatools: released holding the lock)

    def _rebuild(self) -> _IndexEntry:
        """Rebuild the index from the MS with the converter's
        calc_indx_for_row_split, checking the number of MAIN rows, the grid
        shape and its token."""
        with casatools_serialized():
            with open_table_ro(self.ms_path) as main_tb:
                main_nrows = main_tb.nrows()
                if main_nrows != self.main_nrows:
                    raise self.changed_error(
                        f"the MAIN table has {main_nrows} rows, "
                        f"{self.main_nrows} when it was opened"
                    )
                entry = self._index_from(main_tb)
            del main_tb  # (casatools: destroyed holding the lock)
        if isinstance(entry, str):
            raise self.changed_error(entry)
        return entry

    def _index_from(self, main_tb: Any) -> "_IndexEntry | str":
        """The index of the rows made from an opened MAIN table
        (calc_indx_for_row_split), or why its grid is not that of the open
        (its shape or token)."""
        INDEX_MEMO.stats["rebuilds"] += 1
        rows = runs_to_rows(self.starts, self.lengths)
        main_rows = MainTableRows(main_tb, rows)
        try:
            tidxs, bidxs, ant1, ant2, utime = calc_indx_for_row_split(main_rows)
        finally:
            main_rows.close()
        del main_rows
        shape = (len(utime), len(ant1))
        if shape != self.shape:
            return (
                f"the rows of a partition have a (time, baseline) grid of {shape}, "
                f"{self.shape} when it was opened"
            )
        if index_token(utime, ant1, ant2, shape) != self.token:
            return (
                "the times or baselines of the rows of a partition differ from "
                "those when it was opened"
            )
        return _IndexEntry(rows, tidxs, bidxs, self.shape[0], utime, ant1, ant2)

    def rows_moved(
        self, table: Any, rows: np.ndarray, positions: np.ndarray, entry: _IndexEntry
    ) -> str:
        """
        Whether ``rows`` (ascending, at ``positions`` of the index) no longer
        have the TIME, ANTENNA1 and ANTENNA2 of their (time, baseline) cell
        in ``entry``: the description of the first difference, or "". A read
        places the values of a row with the index, so an index whose rows
        moved (keys rewritten in place since it was made) must be made
        again from the MS (see MSv2MainColumnArray._read_selection).
        """
        if entry.utime is None:
            return ""
        for col, expected in (
            ("TIME", entry.utime[entry.tidxs[positions]]),
            ("ANTENNA1", entry.baseline_ant1[entry.bidxs[positions]]),
            ("ANTENNA2", entry.baseline_ant2[entry.bidxs[positions]]),
        ):
            values = read_column_rows(table, col, rows)
            if col == "TIME":
                values = convert_casacore_time(values, False)
            if not np.array_equal(values, expected):
                changed = int(rows[np.flatnonzero(values != expected)[0]])
                return f"the {col} of MAIN row {changed} changed"
        return ""

    def check(self, table: Any) -> bool:
        """
        Whether the partition is known to be as when the MS was opened, so
        that a read need not check the keys of its rows one by one: no write
        of MAIN's key columns or of FIELD, STATE and SOURCE was flushed since
        (``keys_token`` unchanged) and no handle of this process has MAIN open
        for writing.

        Otherwise the whole partition is checked against the MS as it is now,
        once per state of the MS in a process (memoised by its current
        ``keys_token``; on every read while it cannot be told): its rows must
        be exactly the MAIN rows that are in it now (``membership_change``)
        and their (time, baseline) grid that of the open (its index is made
        again from the MS and put in the memo). MSv2ChangedError if not; then
        False: the read checks the TIME, ANTENNA1 and ANTENNA2 of its rows
        against the index it reads with (``rows_moved``).

        Parameters
        ----------
        table : Any
            The MAIN table, opened by the read (with the number of rows of
            the open).

        Returns
        -------
        bool
            True if the partition is as when the MS was opened.

        Raises
        ------
        MSv2ChangedError
            If the rows or the grid of the partition changed.
        """
        current = current_keys_token(table, self.ms_path)
        if current is not None and current == self.keys_token:
            return True
        key = None
        if current is not None:
            key = (self.memo_key(), self.grouping, self.scheme, current)
        if key is None:
            outcome = self._check_now(table)
        else:
            with CHECK_MEMO.check_lock(key):
                outcome = CHECK_MEMO.get(key)
                if outcome is None:
                    outcome = self._check_now(table)
                    CHECK_MEMO.put(key, outcome)
        if outcome:
            raise self.changed_error(outcome)
        return False

    def _check_now(self, table: Any) -> str:
        """The check of ``check``: "" (the index made again is put in the
        memo), or why the partition changed."""
        CHECK_MEMO.stats["checks"] += 1
        changed = self.membership_change(table)
        if changed:
            return changed
        entry = self._index_from(table)
        if isinstance(entry, str):
            return entry
        INDEX_MEMO.put(self.memo_key(), entry)
        return ""

    def membership_change(self, table: Any) -> str:
        """
        Whether the rows of the partition (its runs) are no longer exactly
        the MAIN rows that create_partitions_with_main_rows would put in it
        now: the rows that have its grouping keys (``grouping``, from the MAIN
        key columns and FIELD, SOURCE and STATE as they are now; an ANTENNA1
        partition: autocorrelations of its antenna), in an MS that has the
        same partition keys (MANDATORY_PARTITION_KEYS and the scheme's) as
        when it was opened. The description of the first difference, or "".
        Reads the key columns of every MAIN row (in windows of
        CHECK_WINDOW_ROWS rows) and FIELD, SOURCE and STATE.

        Parameters
        ----------
        table : Any
            The opened MAIN table.
        """
        maps = partition_key_maps(self.ms_path)
        available = available_partition_keys(maps)
        grouping = dict(self.grouping)
        for key in dict.fromkeys((*MANDATORY_PARTITION_KEYS, *self.scheme)):
            if (key in available) != (key in grouping):
                now = "now has" if key in available else "no longer has"
                return (
                    f"the MS {now} the partition key {key}: its partitions are made "
                    "with it"
                )
        columns = sorted({_KEY_COLUMN[key] for key in grouping})
        if "ANTENNA1" in grouping:
            columns.append("ANTENNA2")
        ends = self.starts + self.lengths
        nrows = int(table.nrows())
        for r0 in range(0, nrows, CHECK_WINDOW_ROWS):
            r1 = min(nrows, r0 + CHECK_WINDOW_ROWS)
            window = np.arange(r0, r1, dtype=np.int64)
            values = {col: read_column_rows(table, col, window) for col in columns}
            del window
            matches = {}
            for key, value in grouping.items():
                found = _key_matches(values[_KEY_COLUMN[key]], key, value, maps)
                if isinstance(found, str):
                    return f"the {key} of MAIN rows {r0}..{r1 - 1} {found}"
                matches[key] = found
            if "ANTENNA1" in grouping:
                matches["ANTENNA2"] = values["ANTENNA2"] == grouping["ANTENNA1"]
            member = np.logical_and.reduce(list(matches.values()))
            expected = _runs_mask(self.starts, ends, r0, r1)
            differ = np.flatnonzero(member != expected)
            if differ.size == 0:
                continue
            first = int(differ[0])
            row = r0 + first
            if member[first]:
                keys = ", ".join(f"{key} {value!r}" for key, value in grouping.items())
                return (
                    f"MAIN row {row} has the keys of the partition now ({keys}): it "
                    "was in another partition when the MS was opened"
                )
            key = next(key for key, match in matches.items() if not match[first])
            column = values[_KEY_COLUMN.get(key, key)]
            now = column[first]
            if key in _KEY_LOOKUP:
                now = getattr(maps, _KEY_LOOKUP[key])[now]
            now = now.item() if isinstance(now, np.generic) else now
            value = grouping.get(key, grouping.get("ANTENNA1"))
            return (
                f"MAIN row {row} has the {key} {now!r}, {value!r} when it was "
                "opened: it is in another partition now"
            )
        return ""

    def discard(self, entry: _IndexEntry) -> None:
        """Remove ``entry`` from the memo (rebuilt on the next read)."""
        INDEX_MEMO.discard(self.memo_key(), entry)

    def select(
        self,
        t0: int,
        t1: int,
        b0: int,
        b1: int,
        entry: _IndexEntry | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        The rows of the times [t0, t1) and baselines [b0, b1) of the grid, in
        ascending order, and the flat cell of every row in the (t1 - t0,
        b1 - b0) block grid.

        Parameters
        ----------
        t0, t1, b0, b1 : int
            Time and baseline ranges.
        entry : _IndexEntry | None, optional
            The index (``entry()``), by default looked up.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Rows (int64, ascending) and block cells (int64).
        """
        rows, cells, _ = self.select_outer(
            np.arange(t0, t1, dtype=np.int64),
            np.arange(b0, b1, dtype=np.int64),
            entry,
        )
        return rows, cells

    def select_outer(
        self,
        times: np.ndarray,
        baselines: np.ndarray,
        entry: _IndexEntry | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        The rows of the times ``times`` and baselines ``baselines`` of the
        grid (sorted unique indices), in ascending order, and the flat cell
        of every row in the (len(times), len(baselines)) grid of the
        selection; also the positions of the rows in the index.

        Parameters
        ----------
        times, baselines : np.ndarray
            Time and baseline indices (ascending, unique).
        entry : _IndexEntry | None, optional
            The index (``entry()``), by default looked up.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            Rows (int64, ascending), selection cells (int64) and positions
            of the rows in the index (int64).
        """
        if entry is None:
            entry = self.entry()
        times = np.asarray(times, dtype=np.int64)
        baselines = np.asarray(baselines, dtype=np.int64)
        if times.size == 0 or baselines.size == 0:
            empty = np.empty(0, dtype=np.int64)
            return empty, empty.copy(), empty.copy()
        if is_range(times):
            t0, t1 = int(times[0]), int(times[-1]) + 1
            lo, hi = int(entry.time_bounds[t0]), int(entry.time_bounds[t1])
            pos = np.arange(lo, hi, dtype=np.int64)
            time_position = None
        else:
            lo, hi = entry.time_bounds[times], entry.time_bounds[times + 1]
            counts = hi - lo
            # the positions of the time ranges, concatenated
            pos = np.repeat(lo - (np.cumsum(counts) - counts), counts) + np.arange(
                int(counts.sum()), dtype=np.int64
            )
            time_position = np.full(self.shape[0], -1, dtype=np.int64)
            time_position[times] = np.arange(times.size)
        if entry.order is not None:
            pos = np.sort(entry.order[pos])
        # (else ascending: the times are, and so are their ranges)
        bidxs = entry.bidxs[pos].astype(np.int64)
        if is_range(baselines):
            b0, b1 = int(baselines[0]), int(baselines[-1]) + 1
            if b0 > 0 or b1 < self.shape[1]:
                keep = (bidxs >= b0) & (bidxs < b1)
                pos, bidxs = pos[keep], bidxs[keep]
            baseline_cell = bidxs - b0
        else:
            baseline_position = np.full(self.shape[1], -1, dtype=np.int64)
            baseline_position[baselines] = np.arange(baselines.size)
            baseline_cell = baseline_position[bidxs]
            keep = baseline_cell >= 0
            pos, baseline_cell = pos[keep], baseline_cell[keep]
        tidxs = entry.tidxs[pos].astype(np.int64)
        time_cell = (
            tidxs - int(times[0]) if time_position is None else time_position[tidxs]
        )
        cells = time_cell * baselines.size + baseline_cell
        return entry.rows[pos], cells, pos


# --- the lazy data variables --------------------------------------------------


def _tile_bytes(cube: dict, tile: list[int], dtype: np.dtype) -> int:
    """Bytes of one tile of a hypercube: its BucketSize, or the tile's
    elements (bits for booleans, which the tiled storage managers store as
    bits)."""
    bucket = int(cube.get("BucketSize", 0) or 0)
    if bucket > 0:
        return bucket
    elements = int(np.prod(tile, dtype=np.int64))
    if dtype.kind == "b":
        return -(-elements // 8)
    return elements * dtype.itemsize


def channel_tiling(
    storage: ColumnStorage | None, cell_shape: tuple[int, ...], dtype: DTypeLike
) -> tuple[int, int]:
    """
    How the channels of the cells of a column are tiled, from the hypercubes
    of its tiled storage manager (``column_storage``): the hypercubes of the
    cells of shape ``cell_shape`` (2-D, numpy order (chan, pol)) all have
    tiles of the same number of channels T, fewer than the cell's channels.

    Parameters
    ----------
    storage : ColumnStorage | None
        How the column is stored (None: not known).
    cell_shape : tuple[int, ...]
        Cell shape of the partition (numpy order).
    dtype : DTypeLike
        dtype of the column's values (for the tile size if a hypercube does
        not give its BucketSize).

    Returns
    -------
    tuple[int, int]
        (T, band_bytes): the channels of a tile, and the bytes of the tiles
        that hold one band of T channels of all polarizations for the rows of
        one tile (the largest among the hypercubes). (0, 0) if the channels
        are not tiled that way: not a plain table, another storage manager,
        no hypercube of that cell shape, hypercubes of other channel tiles,
        tiles of all the channels, or the storage could not be described.
    """
    if (
        storage is None
        or storage.error
        or not storage.plain
        or storage.dm_type not in CHANNEL_TILED_DM_TYPES
        or len(cell_shape) != 2
    ):
        return 0, 0
    n_chan, n_pol = (int(n) for n in cell_shape)
    dtype = np.dtype(dtype)
    channels, band_bytes = set(), 0
    try:
        for cube in storage.hypercubes:
            cube_shape = [int(n) for n in np.asarray(cube.get("CubeShape", []))]
            cube_cell = cube.get("CellShape")
            cube_cell = cube_shape[:-1] if cube_cell is None else cube_cell
            if [int(n) for n in np.asarray(cube_cell)] != [n_pol, n_chan]:
                continue  # (cells of another shape: Fortran order)
            tile = [int(n) for n in np.asarray(cube.get("TileShape", []))]
            if len(tile) != 3 or min(tile) < 1:
                return 0, 0
            channels.add(tile[1])
            pol_tiles = -(-n_pol // tile[0])
            band_bytes = max(band_bytes, pol_tiles * _tile_bytes(cube, tile, dtype))
    except (TypeError, ValueError):
        return 0, 0
    if len(channels) != 1:
        return 0, 0
    tile_channels = channels.pop()
    if tile_channels >= n_chan:
        return 0, 0
    return tile_channels, band_bytes


def one_cell_shape(storage: ColumnStorage | None, cell_shape: tuple[int, ...]) -> bool:
    """
    Whether every cell of a column stored in hypercubes (``column_storage``)
    has the shape ``cell_shape`` (numpy order): every hypercube of its tiled
    storage manager has cells of that shape. False if that cannot be told.
    A channel-sliced read bridges gaps of rows of other partitions only
    then (``read_rows_to_grid(bridge=)``): casacore reads a range of
    channels beyond a smaller cell without an error when the run starts
    with a larger one.
    """
    if storage is None or storage.error or not storage.plain:
        return False
    expected = [int(n) for n in cell_shape][::-1]  # (Fortran order)
    try:
        cubes = list(storage.hypercubes)
        for cube in cubes:
            cube_cell = cube.get("CellShape")
            if cube_cell is None:
                cube_cell = np.asarray(cube.get("CubeShape", []))[:-1]
            if [int(n) for n in np.asarray(cube_cell)] != expected:
                return False
    except (TypeError, ValueError):
        return False
    return bool(cubes)


class MSv2MainColumnArray(MSv2BackendArray):
    """
    One data variable of the main xds of an MSv4, read from a MAIN column of
    its partition, in the channel order of the MSv2 (a decreasing frequency
    axis is reversed by the caller, lazily).

    Channel-sliced reads: the cells of a column of CHANNEL_SLICED_COLUMNS
    (values as read: VISIBILITY*, SPECTRUM*, FLAG, WEIGHT from WEIGHT_SPECTRUM)
    whose 2-D (chan, pol) cells are stored in tiles of ``tile_channels``
    channels (fewer than the cell's, ``channel_tiling``) are read, with
    python-casacore, for the range of channels of a selection that does not
    hold them all, rounded out to whole tiles (the tiles of a channel are read
    whole); with casatools (no in-place reads), and for other columns, whole
    cells. The selected channels and polarizations are taken in numpy
    afterwards, as from whole cells, so the values are the same.

    Parameters
    ----------
    index : PartitionIndex
        The partition's rows and grid.
    col : str
        MAIN column.
    name : str
        Data variable name (for messages).
    shape : tuple[int, ...]
        Shape of the data variable: (n_times, n_baselines) + the converted
        cell shape.
    dtype : DTypeLike
        dtype of the data variable.
    grid_dtype : DTypeLike
        dtype of the grid the column is read into (``DeferredVariable``).
    cell_shape : tuple[int, ...]
        Cell shape of the column (numpy order).
    transform : Callable | None
        The converter's conversion of the read grid (``DeferredVariable``), a
        picklable function.
    verified : bool
        Whether the storage manager vouched for every cell of the partition
        when the MS was opened (False: only a read can tell; a read of
        WEIGHT_SPECTRUM first checks the shapes of the cells of the whole
        partition, ``_check_weight_spectrum_cells``).
    node : str
        Name of the MSv4 node (for messages).
    tile_channels : int, optional
        Channels of the column's tiles (``channel_tiling``, when the MS was
        opened), 0 (default) for whole-cell reads. Ignored (0) for a column
        that is not in CHANNEL_SLICED_COLUMNS, cells that are not 2-D or a
        converted cell shape that is not the cell shape.
    tile_band_bytes : int, optional
        Bytes of the tiles of one band of ``tile_channels`` channels
        (``channel_tiling``), for the bound of the tile cache of a
        channel-sliced read.
    tile_bridge : bool, optional
        Whether a channel-sliced read may bridge gaps of rows of other
        partitions (``one_cell_shape`` of the column when the MS was
        opened), by default False: the partition's rows only.
    """

    def __init__(
        self,
        index: PartitionIndex,
        col: str,
        name: str,
        shape: tuple[int, ...],
        dtype: DTypeLike,
        grid_dtype: DTypeLike,
        cell_shape: tuple[int, ...],
        transform: Callable[[np.ndarray], np.ndarray] | None = None,
        verified: bool = True,
        node: str = "",
        tile_channels: int = 0,
        tile_band_bytes: int = 0,
        tile_bridge: bool = False,
    ):
        super().__init__(shape, dtype)
        if self.shape[:2] != index.shape:
            raise ValueError(
                f"{name}: shape {self.shape} does not start with the partition's "
                f"(time, baseline) grid {index.shape}"
            )
        self.index = index
        self.col = str(col)
        self.name = str(name)
        self.grid_dtype = np.dtype(grid_dtype)
        self.cell_shape = tuple(int(n) for n in cell_shape)
        self.transform = transform
        self.verified = bool(verified)
        self.node = str(node)
        sliced = (
            self.col in CHANNEL_SLICED_COLUMNS
            and len(self.cell_shape) == 2
            and self.shape[2:] == self.cell_shape
            and 0 < int(tile_channels) < self.cell_shape[0]
        )
        self.tile_channels = int(tile_channels) if sliced else 0
        self.tile_band_bytes = max(0, int(tile_band_bytes)) if sliced else 0
        self.tile_bridge = bool(tile_bridge) and sliced

    @classmethod
    def from_spec(
        cls,
        index: PartitionIndex,
        spec: Any,
        shape: tuple[int, ...],
        dtype: DTypeLike,
        node: str = "",
        storage: ColumnStorage | None = None,
    ) -> "MSv2MainColumnArray":
        """The array of a ``stream_write.DeferredVariable`` read from a column,
        whose channel tiling (``channel_tiling``) is found from ``storage``
        (``column_storage`` of the column when the MS is opened; None: whole
        cells are read)."""
        tile_channels, tile_band_bytes, tile_bridge = 0, 0, False
        if spec.col in CHANNEL_SLICED_COLUMNS:
            tile_channels, tile_band_bytes = channel_tiling(
                storage, spec.cell_shape, spec.grid_dtype
            )
            tile_bridge = one_cell_shape(storage, spec.cell_shape)
        return cls(
            index,
            spec.col,
            spec.name,
            shape,
            dtype,
            spec.grid_dtype,
            spec.cell_shape,
            spec.transform,
            spec.verified,
            node,
            tile_channels,
            tile_band_bytes,
            tile_bridge,
        )

    def _cell_bytes(self, channels: int | None = None) -> int:
        """Bytes of one (time, baseline) cell of a sub-block: the read cell and
        the converted one (of ``channels`` channels for a channel-sliced
        read, None: whole cells)."""
        read = int(np.prod(self.cell_shape, dtype=np.int64)) * self.grid_dtype.itemsize
        converted = int(np.prod(self.shape[2:], dtype=np.int64)) * self.dtype.itemsize
        if channels is not None:
            read = read // self.cell_shape[0] * channels
            converted = converted // self.shape[2] * channels
        return max(1, read + converted)

    def _channel_range(self, channels: np.ndarray) -> tuple[int, int] | None:
        """
        The channels [c0, c1) that a channel-sliced read of the selected
        ``channels`` (sorted unique) reads: their range rounded out to whole
        tiles of ``tile_channels`` channels, at most the cell's channels.
        None (whole cells) if the column's channels are not tiled that way or
        the range holds every channel.
        """
        width = self.tile_channels
        if not width or channels.size == 0:
            return None
        n_chan = self.cell_shape[0]
        c0 = int(channels[0]) // width * width
        c1 = min(n_chan, -(-(int(channels[-1]) + 1) // width) * width)
        if c0 == 0 and c1 == n_chan:
            return None
        return c0, c1

    def _channel_cache_mib(self, c0: int, c1: int) -> int:
        """
        The bound (MiB) of the column's tile cache for a read of the channels
        [c0, c1): twice the tiles of its bands of channels for the rows of a
        tile (a row of tiles, and the next one), at least
        CHANNEL_SLICE_MIN_CACHE_MIB.

        It caps memory, not the bytes read. casacore sizes the cache of every
        access itself (TSMDataColumn::accessSlicedCells): one tile for a range
        of whole tiles, but for a range that ends inside a tile (the last,
        partial band of cells whose channels are not a multiple of
        ``tile_channels``) every tile of the band in the hypercube, up to a
        quarter of the host's memory (not the process's limits), kept until
        the table is closed (the last channel of runs of rows spread over a
        column of 200,000 rows of 100 channels, in tiles of 8 channels and 64
        rows: the 25 MiB of tiles read were kept; nothing with the bound). The
        bytes read are the same without a bound and with bounds of 16 MiB to
        1 GiB:
        the tiles that the rows of two runs share are read again either way,
        as with whole cells. The bound is set while the read runs
        (TILE_CACHE_BOUNDS), on the column's data manager: for every handle of
        MAIN open in the process at that time.
        """
        bands = -(-(c1 - c0) // self.tile_channels)
        needed = -(-2 * bands * self.tile_band_bytes // 2**20)
        return max(CHANNEL_SLICE_MIN_CACHE_MIB, needed)

    def _read_channels(
        self,
        table: Any,
        plan: Any,
        grid_shape: tuple[int, int],
        chan_range: tuple[int, int],
    ) -> np.ndarray:
        """The cells of the channels ``chan_range`` of the rows of ``plan`` on
        the (time, baseline) grid of ``grid_shape``, as ``read_grid`` (padded
        with get_pad_value; for duplicated cells the last row wins) but for
        those channels only (the transform is the identity: the column is in
        CHANNEL_SLICED_COLUMNS). Gaps of rows of other partitions are bridged
        only in a column of one cell shape (``tile_bridge``)."""
        c0, c1 = chan_range
        shape = tuple(grid_shape) + (c1 - c0, self.cell_shape[1])
        if plan.grid_is_full:
            grid = np.empty(shape, dtype=self.grid_dtype)
        else:
            grid = np.full(shape, get_pad_value(self.grid_dtype), dtype=self.grid_dtype)
        if plan.rows.size:
            read_rows_to_grid(
                table,
                self.col,
                plan,
                grid,
                chan=slice(c0, c1),
                bridge=self.tile_bridge,
            )
        return grid

    def _message(self, key: tuple[slice, ...], exc: BaseException) -> str:
        block = ", ".join(f"{k.start}:{k.stop}" for k in key)
        message = (
            f"Reading the MSv2 column {self.col} for the data variable {self.name} "
            f"of {self.node or 'an MSv4'} (block [{block}]) from {self.index.ms_path} "
            f"failed: {type(exc).__name__}: {exc}"
        )
        if not self.verified:
            if self.col == "WEIGHT_SPECTRUM":
                converter = (
                    "convert_msv2_to_processing_set reads WEIGHT from the WEIGHT "
                    "column when WEIGHT_SPECTRUM cannot be read"
                )
            else:
                converter = (
                    f"convert_msv2_to_processing_set leaves {self.col} out (with the "
                    "data group and field_and_source_xds it makes) when it cannot "
                    "be read"
                )
            message += (
                f". The cells of {self.col} could not be checked when the MS was "
                f"opened (only a read can tell): {converter}. Open the MS with "
                f"skip_columns=[{self.col!r}] for the same processing set, or with "
                f"drop_variables=[{self.name!r}] to leave out only {self.name}"
            )
        return message

    def _check_weight_spectrum_cells(self, table: Any) -> None:
        """
        WEIGHT read from a WEIGHT_SPECTRUM column whose cells could not be
        checked when the MS was opened: raise ColumnNotReadableError if a
        cell of the partition, selected or not, has no value or another
        shape. The converter reads the whole partition, and where a cell of
        WEIGHT_SPECTRUM cannot be read it takes WEIGHT from the WEIGHT
        column, other values: a selection that avoids such cells must not
        return those of WEIGHT_SPECTRUM. Only the shapes of the cells are
        read (``_scan_cell_shapes``; of a StandardStMan column, that reads
        its file), once per partition in a process while they can all be
        read (WEIGHT_SPECTRUM_CHECKED); a partition with a cell that cannot
        be read is checked again on every read, which raises.
        """
        key = self.index.memo_key()
        if key in WEIGHT_SPECTRUM_CHECKED:
            return
        rows = runs_to_rows(self.index.starts, self.index.lengths)
        expected = str(list(self.cell_shape))
        try:
            first = table.getcolshapestring(self.col, int(rows[0]), 1)[0]
        except RuntimeError as exc:
            raise ColumnNotReadableError(
                f"Column {self.col}: the first cell of the partition is undefined"
            ) from exc
        if first != expected:
            raise ColumnNotReadableError(
                f"Column {self.col}: the first cell of the partition has the shape "
                f"{first}, {expected} when the MS was opened"
            )
        _scan_cell_shapes(table, self.col, rows, expected)
        if len(WEIGHT_SPECTRUM_CHECKED) >= WEIGHT_SPECTRUM_CHECKED_MAX_ENTRIES:
            WEIGHT_SPECTRUM_CHECKED.clear()
        WEIGHT_SPECTRUM_CHECKED.add(key)

    def _raw_indexing_method(self, key: tuple[slice, ...]) -> np.ndarray:
        return self._read_selection(
            tuple(np.arange(k.start, k.stop, dtype=np.int64) for k in key)
        )

    def _read_selection(self, selections: tuple[np.ndarray, ...]) -> np.ndarray:
        """
        The values of the selected times, baselines and cell elements: only
        the rows of the selected (time, baseline) cells are read, in time
        sub-blocks of at most SUB_BLOCK_BYTES of whole cells (the transform
        applies to whole cells), or of the cells' channels that a
        channel-sliced read reads (``_read_with``).

        When MAIN's key columns or FIELD, STATE and SOURCE may have been
        written since the open (``PartitionIndex.check``), the whole
        partition is first checked against the MS as it is now (once per
        state of the MS): MSv2ChangedError if its rows are no longer exactly
        those of the open (rows moved to or from another partition) or its
        (time, baseline) grid changed. Every sub-block then checks that its
        rows still have the TIME, ANTENNA1 and ANTENNA2 of their cells in the
        index the read uses; if not (keys rewritten in place after the open,
        within the grid), the index is made again from the MS, as in another
        process (``calc_indx_for_row_split``, checked against the grid of the
        open), and the selection read again with it: the outcome does not
        depend on whether the index was in the memo.
        """
        key = bounding_slices(selections)
        for _attempt in (1, 2):
            try:
                entry = self.index.entry()
            except MSv2ChangedError:
                raise
            except Exception as exc:
                raise MSv2ReadError(self._message(key, exc)) from exc
            values, moved = self._read_with(selections, entry, key)
            if not moved:
                return values
            del values
            xradio_logger().debug(
                f"{self.index.ms_path}: {moved} since the index of a partition was "
                "made: made again"
            )
            self.index.discard(entry)
        raise self.index.changed_error(f"{moved} while it was read")

    def _read_with(
        self,
        selections: tuple[np.ndarray, ...],
        entry: _IndexEntry,
        key: tuple[slice, ...],
    ) -> tuple[np.ndarray | None, str]:
        """The values of a selection read with the index ``entry``, or
        (None, why) if rows of the selection no longer have its keys. Of a
        column whose channels are tiled (``tile_channels``), with
        python-casacore, only the channels of the selection's range rounded
        out to whole tiles are read (``_channel_range``)."""
        times, baselines, cell_selections = selections[0], selections[1], selections[2:]
        n_baselines = baselines.size
        out = np.empty(tuple(s.size for s in selections), dtype=self.dtype)
        chan_range = (
            self._channel_range(cell_selections[0]) if self.tile_channels else None
        )
        check_keys = True
        with casatools_serialized():
            table = None
            try:
                with contextlib.ExitStack() as stack:
                    if out.size:
                        # opened in the thread / process that reads
                        table = stack.enter_context(open_table_ro(self.index.ms_path))
                        main_nrows = table.nrows()
                        if main_nrows != self.index.main_nrows:
                            raise self.index.changed_error(
                                f"the MAIN table has {main_nrows} rows, "
                                f"{self.index.main_nrows} when it was opened"
                            )
                        # (the partition as when the MS was opened: rows not
                        # checked one by one)
                        check_keys = not self.index.check(table)
                        if self.col == "WEIGHT_SPECTRUM" and not self.verified:
                            self._check_weight_spectrum_cells(table)
                        if chan_range is not None:
                            if has_in_place_reads(table):
                                # (set again when the read ends, before the
                                # table is closed)
                                stack.enter_context(
                                    TILE_CACHE_BOUNDS.bounded(
                                        table,
                                        self.index.ms_path,
                                        self.col,
                                        self._channel_cache_mib(*chan_range),
                                    )
                                )
                            else:  # (casatools: whole cells)
                                chan_range = None
                    if chan_range is None:
                        cell_bytes = self._cell_bytes()
                        offsets = [0] * len(selections)
                    else:
                        cell_bytes = self._cell_bytes(chan_range[1] - chan_range[0])
                        offsets = [0, 0, chan_range[0], 0]
                    step = max(1, SUB_BLOCK_BYTES // (n_baselines * cell_bytes))
                    for p0 in range(0, times.size, step):
                        sub_times = times[p0 : p0 + step]
                        rows, cells, positions = self.index.select_outer(
                            sub_times, baselines, entry
                        )
                        if rows.size and check_keys:
                            moved = self.index.rows_moved(table, rows, positions, entry)
                            if moved:
                                return None, moved
                        del positions
                        plan = make_row_grid_plan(
                            rows, cells, sub_times.size * n_baselines
                        )
                        del rows, cells
                        if chan_range is None:
                            grid = read_grid(
                                table,
                                self.col,
                                plan,
                                (sub_times.size, n_baselines) + self.cell_shape,
                                self.grid_dtype,
                                transform=self.transform,
                            )
                        else:
                            grid = self._read_channels(
                                table, plan, (sub_times.size, n_baselines), chan_range
                            )
                        del plan
                        out[p0 : p0 + sub_times.size] = take_from_block(
                            grid,
                            (
                                np.arange(sub_times.size),
                                np.arange(n_baselines),
                            )
                            + tuple(cell_selections),
                            offsets,
                        )
                        del grid
            except MSv2ChangedError:
                raise
            except Exception as exc:
                raise MSv2ReadError(self._message(key, exc)) from exc
            finally:
                table = None  # (casatools: released holding the lock)
        return out, ""


class OnesArray(MSv2BackendArray):
    """
    The WEIGHT=1 fallback of a partition without readable weights: ones of
    the shape of VISIBILITY / SPECTRUM (float64), made per block (no value is
    read, and no variable-sized array is held). With the partition's
    ``index``, a read first checks that the partition is the one of the open
    (``PartitionIndex.verify_current``).
    """

    def __init__(
        self,
        shape: tuple[int, ...],
        dtype: DTypeLike = np.float64,
        index: PartitionIndex | None = None,
    ):
        super().__init__(shape, dtype)
        self.index = index

    def _raw_indexing_method(self, key: tuple[slice, ...]) -> np.ndarray:
        if self.index is not None:
            self.index.verify_current()
        return np.ones(tuple(k.stop - k.start for k in key), dtype=self.dtype)

    def _read_selection(self, selections: tuple[np.ndarray, ...]) -> np.ndarray:
        if self.index is not None:
            self.index.verify_current()
        return np.ones(tuple(s.size for s in selections), dtype=self.dtype)
