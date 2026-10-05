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
its result plus one sub-block. Every read opens the MAIN table by name and
closes it (no handle is kept); with casatools it holds the process-wide
casatools lock (``casatools_serialized``) over the open, the read, the close
and the release of the table.

Rows: :class:`PartitionIndex` holds the MAIN rows of a partition as runs, the
number of MAIN rows and the (time, baseline) grid shape when the MS was
opened, and a token of the grid (unique times and baselines). The (time,
baseline) index of every row is kept in a bounded per-process memo: seeded
from the build when the MS is opened, rebuilt with the converter's
``calc_indx_for_row_split`` in another process (or after an eviction). A MAIN
table with another number of rows, or a rebuilt index with another token,
raises :class:`MSv2ChangedError`. Every read also checks that the rows it
reads still have the TIME, ANTENNA1 and ANTENNA2 of their cells in the index;
if not, the index is rebuilt (and checked) as in another process, so that the
outcome of a read does not depend on whether the index was in the memo. The arrays pickle to their path, runs and column
description (about 1 kB plus 16 bytes per run), never per-row arrays or table
handles.
"""

import collections
import contextlib
import hashlib
import os
import threading
from collections.abc import Callable
from typing import Any

import numpy as np
import xarray as xr
from numpy.typing import DTypeLike

from xradio._utils._casacore.tables import casatools_serialized
from xradio._utils.logging import xradio_logger
from xradio.measurement_set._utils._msv2._tables.read import convert_casacore_time
from xradio.measurement_set._utils._msv2._tables.read_rows import (
    MainTableRows,
    make_row_grid_plan,
    read_column_rows,
    read_grid,
    rows_to_runs,
    runs_to_rows,
)
from xradio.measurement_set._utils._msv2._tables.table_query import open_table_ro
from xradio.measurement_set._utils._msv2.backend_errors import (
    MSv2ChangedError,
    MSv2ReadError,
)
from xradio.measurement_set._utils._msv2.conversion import calc_indx_for_row_split

# Largest time sub-block (bytes of whole cells, as read and as transformed)
# that one read of a lazy block holds, besides its result: a block of more
# times is read in several sub-blocks (at least one time each).
SUB_BLOCK_BYTES = 128 * 2**20
# Bound of the per-process memo of partition indices (bytes of index arrays).
# The most recently used index is always kept.
INDEX_MEMO_MAX_BYTES = 256 * 2**20
# Number of locks that serialize the rebuilds of indices (by key hash), so
# that threads reading one partition rebuild its index once
INDEX_BUILD_LOCKS = 64


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


def clear_index_memo() -> None:
    """Empty the per-process memo of partition indices (for tests)."""
    INDEX_MEMO.clear()


class PartitionIndex:
    """
    The MAIN rows of a partition and its (time, baseline) grid, as when the MS
    was opened: the rows as runs, the number of MAIN rows, the grid shape and
    its token (``index_token``). The (time, baseline) index of every row is
    taken from the per-process memo (``INDEX_MEMO``), or rebuilt from the MS
    (``calc_indx_for_row_split``, the converter's own) and checked against the
    shape and token. Pickles to these fields only.

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
    """

    __slots__ = ("ms_path", "starts", "lengths", "main_nrows", "shape", "token")

    def __init__(
        self,
        ms_path: str,
        starts: np.ndarray,
        lengths: np.ndarray,
        main_nrows: int,
        shape: tuple[int, int],
        token: str,
    ):
        self.ms_path = str(ms_path)
        self.starts = np.asarray(starts, dtype=np.int64)
        self.lengths = np.asarray(lengths, dtype=np.int64)
        self.main_nrows = int(main_nrows)
        self.shape = tuple(int(n) for n in shape)
        self.token = str(token)
        if len(self.shape) != 2:
            raise ValueError(f"A (time, baseline) shape is expected, got {shape}")

    @classmethod
    def seed(cls, ms_path: str, built: Any) -> "PartitionIndex":
        """
        The index of a partition built by ``conversion.build_partition`` (a
        ``BuiltPartition``, inside its context), with the build's (time,
        baseline) indices put in the memo: in this process the reads use
        exactly the build's index.
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

    def _rebuild(self) -> _IndexEntry:
        """Rebuild the index from the MS with the converter's
        calc_indx_for_row_split, checking the number of MAIN rows, the grid
        shape and its token."""
        INDEX_MEMO.stats["rebuilds"] += 1
        rows = runs_to_rows(self.starts, self.lengths)
        with casatools_serialized():
            with open_table_ro(self.ms_path) as main_tb:
                main_nrows = main_tb.nrows()
                if main_nrows != self.main_nrows:
                    raise self.changed_error(
                        f"the MAIN table has {main_nrows} rows, "
                        f"{self.main_nrows} when it was opened"
                    )
                main_rows = MainTableRows(main_tb, rows)
                try:
                    tidxs, bidxs, ant1, ant2, utime = calc_indx_for_row_split(main_rows)
                finally:
                    main_rows.close()
                del main_rows
            del main_tb  # (casatools: destroyed holding the lock)
        shape = (len(utime), len(ant1))
        if shape != self.shape:
            raise self.changed_error(
                f"the rows of a partition have a (time, baseline) grid of {shape}, "
                f"{self.shape} when it was opened"
            )
        if index_token(utime, ant1, ant2, shape) != self.token:
            raise self.changed_error(
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


class MSv2MainColumnArray(MSv2BackendArray):
    """
    One data variable of the main xds of an MSv4, read from a MAIN column of
    its partition, in the channel order of the MSv2 (a decreasing frequency
    axis is reversed by the caller, lazily).

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
        when the MS was opened (False: only a read can tell).
    node : str
        Name of the MSv4 node (for messages).
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

    @classmethod
    def from_spec(
        cls,
        index: PartitionIndex,
        spec: Any,
        shape: tuple[int, ...],
        dtype: DTypeLike,
        node: str = "",
    ) -> "MSv2MainColumnArray":
        """The array of a ``stream_write.DeferredVariable`` read from a column."""
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
        )

    def _cell_bytes(self) -> int:
        """Bytes of one (time, baseline) cell of a sub-block: the read cell and
        the converted one."""
        read = int(np.prod(self.cell_shape, dtype=np.int64)) * self.grid_dtype.itemsize
        converted = int(np.prod(self.shape[2:], dtype=np.int64)) * self.dtype.itemsize
        return max(1, read + converted)

    def _message(self, key: tuple[slice, ...], exc: BaseException) -> str:
        block = ", ".join(f"{k.start}:{k.stop}" for k in key)
        message = (
            f"Reading the MSv2 column {self.col} for the data variable {self.name} "
            f"of {self.node or 'an MSv4'} (block [{block}]) from {self.index.ms_path} "
            f"failed: {type(exc).__name__}: {exc}"
        )
        if not self.verified:
            message += (
                f". The cells of {self.col} could not be checked when the MS was "
                "opened (convert_msv2_to_processing_set leaves such a column out); "
                f"open the MS with drop_variables=[{self.name!r}]"
            )
        return message

    def _raw_indexing_method(self, key: tuple[slice, ...]) -> np.ndarray:
        return self._read_selection(
            tuple(np.arange(k.start, k.stop, dtype=np.int64) for k in key)
        )

    def _read_selection(self, selections: tuple[np.ndarray, ...]) -> np.ndarray:
        """
        The values of the selected times, baselines and cell elements: only
        the rows of the selected (time, baseline) cells are read, in time
        sub-blocks of at most SUB_BLOCK_BYTES of whole cells (the transform
        applies to whole cells).

        Every sub-block first checks that its rows still have the TIME,
        ANTENNA1 and ANTENNA2 of their cells in the index. If not (keys
        rewritten in place after the open), the index is made again from the
        MS, as in another process (``calc_indx_for_row_split``, checked
        against the grid of the open: MSv2ChangedError if it differs), and
        the selection read again with it: the outcome does not depend on
        whether the index was in the memo.
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
        (None, why) if rows of the selection no longer have its keys."""
        times, baselines, cell_selections = selections[0], selections[1], selections[2:]
        n_baselines = baselines.size
        out = np.empty(tuple(s.size for s in selections), dtype=self.dtype)
        step = max(1, SUB_BLOCK_BYTES // (n_baselines * self._cell_bytes()))
        with casatools_serialized():
            table = None
            try:
                with contextlib.ExitStack() as stack:
                    for p0 in range(0, times.size, step):
                        sub_times = times[p0 : p0 + step]
                        rows, cells, positions = self.index.select_outer(
                            sub_times, baselines, entry
                        )
                        if rows.size and table is None:
                            # opened in the thread / process that reads
                            table = stack.enter_context(
                                open_table_ro(self.index.ms_path)
                            )
                            main_nrows = table.nrows()
                            if main_nrows != self.index.main_nrows:
                                raise self.index.changed_error(
                                    f"the MAIN table has {main_nrows} rows, "
                                    f"{self.index.main_nrows} when it was opened"
                                )
                        if rows.size:
                            moved = self.index.rows_moved(table, rows, positions, entry)
                            if moved:
                                return None, moved
                        del positions
                        plan = make_row_grid_plan(
                            rows, cells, sub_times.size * n_baselines
                        )
                        del rows, cells
                        grid = read_grid(
                            table,
                            self.col,
                            plan,
                            (sub_times.size, n_baselines) + self.cell_shape,
                            self.grid_dtype,
                            transform=self.transform,
                        )
                        del plan
                        out[p0 : p0 + sub_times.size] = take_from_block(
                            grid,
                            (
                                np.arange(sub_times.size),
                                np.arange(n_baselines),
                            )
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
        return out, ""


class OnesArray(MSv2BackendArray):
    """
    The WEIGHT=1 fallback of a partition without readable weights: ones of
    the shape of VISIBILITY / SPECTRUM (float64), made per block (nothing is
    read, and no variable-sized array is held).
    """

    def __init__(self, shape: tuple[int, ...], dtype: DTypeLike = np.float64):
        super().__init__(shape, dtype)

    def _raw_indexing_method(self, key: tuple[slice, ...]) -> np.ndarray:
        return np.ones(tuple(k.stop - k.start for k in key), dtype=self.dtype)

    def _read_selection(self, selections: tuple[np.ndarray, ...]) -> np.ndarray:
        return np.ones(tuple(s.size for s in selections), dtype=self.dtype)
