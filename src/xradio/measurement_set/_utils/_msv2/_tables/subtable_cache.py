"""
A per-conversion cache for MSv2 sub-table reads.

Without it, every MSv4 partition re-reads the sub-tables it needs: POINTING is
re-read, re-sorted and re-pivoted for every partition, and SYSCAL, WEATHER,
ANTENNA, FEED, SPECTRAL_WINDOW, etc. are loaded again with the same TaQL
selection. With a cache active (see ``activate_subtable_cache``):

- POINTING is read once per conversion (only the columns the pointing_xds
  needs) and every partition gets its time/antenna subset by numpy selection
  (``read_pointing.py``).
- The sorted time column used to project a partition's time range onto
  POINTING / PHASE_CAL / EPHEM* tables is read and sorted once per table.
- ``load_generic_table`` results of sub-tables that partitions load with the
  same arguments are memoized (bounded LRU; every caller gets its own deep copy).
- ``load_generic_table`` reads the columns of the other sub-tables with
  vectorized, bounded reads instead of one ``row()`` dict per table row.

The output is identical to the uncached reads. The MS must not change during a
conversion.

Backends: the cache is used with python-casacore only
(``subtable_cache_supported``). Its loaders reproduce the values and dtypes of
python-casacore's ``tables.row()`` / ``getcol()`` with in-place column reads;
with the casatools shim every partition reads the sub-tables itself, as
without cache.

Whole-table values (POINTING, sorted columns) only pay off when several
partitions use them: a cache that only one partition of this process is known
to use (``SubtableCache(n_partitions=1)``, the cache of a direct
``convert_and_write_partition`` call, a copy unpickled in a worker process)
builds them at the second request of the same value only; the first request
reads per partition, as without cache. Their total size is bounded
(``VALUES_MAX_TOTAL_BYTES``).

Sharing: the cache never holds an open casacore table (tables are opened, read
and closed inside the builders), so it is safe to share between the threads of
``parallel_mode="partition"`` and across ``fork()``. A pickled cache carries
only its token: the copies unpickled in one (worker) process share one state,
built in that process. A worker keeps the state while copies of the cache are
alive in it, and for ``PROCESS_STATE_IDLE_SECONDS`` after the last one is
released (so that consecutive tasks of a conversion share it), at most
``MAX_IDLE_PROCESS_STATES`` idle states.
"""

import collections
import contextlib
import contextvars
import threading
import time
import uuid
import weakref
from collections.abc import Callable, Generator, Hashable
from typing import Any

import xarray as xr

from xradio.measurement_set._utils._msv2._tables.read_rows import (
    backend_has_in_place_reads,
)

# Sub-tables whose load_generic_table results are memoized: partitions typically
# load them with identical arguments (same antennas, spectral window, ...).
# POINTING (own cache), FIELD, PHASE_CAL and EPHEM* (selections that depend on
# each partition's fields or time range) are not memoized.
MEMOIZED_TABLES = frozenset(
    {
        "ANTENNA",
        "DATA_DESCRIPTION",
        "DOPPLER",
        "FEED",
        "GAIN_CURVE",
        "OBSERVATION",
        "PHASED_ARRAY",
        "POLARIZATION",
        "PROCESSOR",
        "SOURCE",
        "SPECTRAL_WINDOW",
        "STATE",
        "SYSCAL",
        "WEATHER",
    }
)
# Bounds of the memo: number of datasets, their total size and the size of one
# dataset (larger results are not kept). Least recently used datasets go first.
MEMO_MAX_ENTRIES = 64
MEMO_MAX_TOTAL_BYTES = 128 * 1024 * 1024
MEMO_MAX_DATASET_BYTES = 16 * 1024 * 1024
# Total size of the values built with get_or_build (POINTING columns, sorted
# columns). A value that does not fit is used by the caller that built it and
# then marked as not cached (the next callers read per partition).
VALUES_MAX_TOTAL_BYTES = 768 * 1024 * 1024
# Unpickled caches (worker processes): idle states (no copy alive) are kept this
# long, at most this many of them.
PROCESS_STATE_IDLE_SECONDS = 30.0
MAX_IDLE_PROCESS_STATES = 2

_ACTIVE_SUBTABLE_CACHE: contextvars.ContextVar["SubtableCache | None"] = (
    contextvars.ContextVar("xradio_msv2_subtable_cache", default=None)
)


def is_memoized_table(table_name: str) -> bool:
    """Whether load_generic_table results of this sub-table are memoized."""
    return table_name in MEMOIZED_TABLES or table_name.startswith("ASDM_")


# Marks a get_or_build value that was built but not kept (over budget)
_NOT_KEPT = object()


def _value_nbytes(value: Any) -> int:
    """Memory held by a get_or_build value (arrays, tuples of them, objects
    with an nbytes attribute)."""
    if value is None or value is _NOT_KEPT:
        return 0
    nbytes = getattr(value, "nbytes", None)
    if isinstance(nbytes, int):
        return nbytes
    if isinstance(value, tuple | list):
        return sum(_value_nbytes(item) for item in value)
    return 0


class _SubtableCacheState:
    """The data of a SubtableCache (shared by its unpickled copies)."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.build_locks: dict[Hashable, threading.Lock] = {}
        self.values: dict[Hashable, Any] = {}
        self.values_bytes = 0
        self.requests: collections.Counter = collections.Counter()
        self.memo: collections.OrderedDict[Hashable, xr.Dataset] = (
            collections.OrderedDict()
        )
        self.memo_bytes = 0
        self.stats: collections.Counter = collections.Counter()
        # unpickled copies of the cache alive in this process
        self.n_copies = 0


class _ProcessStates:
    """
    States of the caches unpickled in this process, by token: a state lives
    while copies of its cache are alive, then idles for
    PROCESS_STATE_IDLE_SECONDS (at most MAX_IDLE_PROCESS_STATES idle states).
    """

    def __init__(self) -> None:
        # reentrant: release() runs from a weakref finalizer, which garbage
        # collection can trigger while this thread holds the lock
        self.lock = threading.RLock()
        self.states: weakref.WeakValueDictionary[str, _SubtableCacheState] = (
            weakref.WeakValueDictionary()
        )
        self.idle: collections.OrderedDict[str, tuple[_SubtableCacheState, float]] = (
            collections.OrderedDict()
        )
        self.timer: threading.Timer | None = None

    def acquire(self, token: str) -> _SubtableCacheState:
        """The state of a newly unpickled copy (one more copy alive)."""
        with self.lock:
            state = self.states.get(token)
            if state is None:
                state = _SubtableCacheState()
                self.states[token] = state
            self.idle.pop(token, None)
            state.n_copies += 1
            return state

    def release(self, token: str, state: _SubtableCacheState) -> None:
        """A copy was garbage collected: keep its state idle for a while."""
        with self.lock:
            state.n_copies -= 1
            if state.n_copies > 0 or self.states.get(token) is not state:
                return
            self.idle[token] = (state, time.monotonic())
            self.idle.move_to_end(token)
            while len(self.idle) > MAX_IDLE_PROCESS_STATES:
                self.idle.popitem(last=False)
            self._schedule_expiry()

    def forget(self, token: str) -> None:
        """Drop a state (SubtableCache.clear)."""
        with self.lock:
            self.idle.pop(token, None)
            self.states.pop(token, None)

    def _schedule_expiry(self) -> None:
        if self.timer is not None and self.timer.is_alive():
            return
        self.timer = threading.Timer(PROCESS_STATE_IDLE_SECONDS, self._expire)
        self.timer.daemon = True
        self.timer.start()

    def _expire(self) -> None:
        with self.lock:
            self.timer = None
            now = time.monotonic()
            for token, (_, idle_since) in list(self.idle.items()):
                if now - idle_since >= PROCESS_STATE_IDLE_SECONDS:
                    del self.idle[token]
            if self.idle:
                self._schedule_expiry()


_PROCESS_STATES = _ProcessStates()


class SubtableCache:
    """
    Sub-table data shared by the partitions of one conversion.

    Two kinds of entries:

    - values built once per key and shared read-only (``get_or_build``), e.g.
      the sorted POINTING columns. Concurrent callers of the same key wait for
      the first builder.
    - memoized datasets (``memo_dataset``), kept in a bounded LRU; every caller
      gets a deep copy, so callers may modify what they get.

    Parameters
    ----------
    n_partitions : int | None, optional
        Number of partitions that will use the cache in this process. With 2 or
        more, whole-table values are built at their first request. Otherwise
        (1, or None: unknown) only at their second request: the first one
        reads per partition (see ``get_or_build``).
    """

    def __init__(self, n_partitions: int | None = None) -> None:
        self._token = uuid.uuid4().hex
        self._state = _SubtableCacheState()
        self._build_at_request = 1 if (n_partitions or 0) >= 2 else 2

    @property
    def stats(self) -> collections.Counter:
        """Counters of cache hits, builds, fallbacks, ... (read only: use
        ``count`` to increment them)."""
        return self._state.stats

    def count(self, name: str, n: int = 1) -> None:
        """Increment the counter ``name`` of ``stats`` (thread safe)."""
        state = self._state
        with state.lock:
            state.stats[name] += n

    def get_or_build(
        self, key: Hashable, builder: Callable[[], Any], amortized: bool = False
    ) -> Any:
        """
        The value stored under ``key``, built with ``builder()`` on first use.

        Parameters
        ----------
        key : Hashable
            Cache key (should include the table path).
        builder : Callable[[], Any]
            Builds the value. Exceptions propagate and nothing is stored.
        amortized : bool, optional
            The value is a whole-table read that only pays off when several
            partitions use it. If only one partition of this process is known
            to use the cache, it is built at the second request of ``key``
            only; before that, None is returned and the caller reads per
            partition (as with a value that is not cacheable).

        Returns
        -------
        Any
            The shared value (not copied: callers must not modify it), or None
            (deferred, see ``amortized``, or built but too large to keep).
        """
        state = self._state
        with state.lock:
            state.requests[key] += 1
            if key in state.values:
                state.stats["value_hits"] += 1
                value = state.values[key]
                return None if value is _NOT_KEPT else value
            if amortized and state.requests[key] < self._build_at_request:
                state.stats["value_deferred"] += 1
                return None
            build_lock = state.build_locks.setdefault(key, threading.Lock())
        with build_lock:
            with state.lock:
                if key in state.values:
                    state.stats["value_hits"] += 1
                    value = state.values[key]
                    return None if value is _NOT_KEPT else value
            value = builder()
            nbytes = _value_nbytes(value)
            with state.lock:
                if state.values_bytes + nbytes <= VALUES_MAX_TOTAL_BYTES:
                    state.values[key] = value
                    state.values_bytes += nbytes
                else:
                    # used by this caller only; the next ones read per partition
                    state.values[key] = _NOT_KEPT
                    state.stats["value_not_kept"] += 1
                state.build_locks.pop(key, None)
                state.stats["value_builds"] += 1
        return value

    def memo_dataset(
        self, key: Hashable, loader: Callable[[], xr.Dataset]
    ) -> xr.Dataset:
        """
        A dataset loaded with ``loader()``, memoized under ``key``.

        Parameters
        ----------
        key : Hashable
            Memo key: everything the loaded dataset depends on.
        loader : Callable[[], xr.Dataset]
            Loads the dataset. Exceptions propagate and nothing is stored.

        Returns
        -------
        xr.Dataset
            A dataset owned by the caller (a deep copy of the memoized one).
        """
        state = self._state
        with state.lock:
            cached = state.memo.get(key)
            if cached is not None:
                state.memo.move_to_end(key)
                state.stats["memo_hits"] += 1
        if cached is not None:
            return cached.copy(deep=True)

        xds = loader()
        nbytes = xds.nbytes
        with state.lock:
            state.stats["memo_misses"] += 1
        if nbytes <= MEMO_MAX_DATASET_BYTES:
            kept = xds.copy(deep=True)
            with state.lock:
                if key not in state.memo:
                    state.memo[key] = kept
                    state.memo_bytes += nbytes
                while state.memo and (
                    len(state.memo) > MEMO_MAX_ENTRIES
                    or state.memo_bytes > MEMO_MAX_TOTAL_BYTES
                ):
                    _, dropped = state.memo.popitem(last=False)
                    state.memo_bytes -= dropped.nbytes
        return xds

    def clear(self) -> None:
        """Drop every cached value and memoized dataset."""
        state = self._state
        with state.lock:
            state.values.clear()
            state.values_bytes = 0
            state.requests.clear()
            state.memo.clear()
            state.memo_bytes = 0
        _PROCESS_STATES.forget(self._token)

    def __dask_tokenize__(self) -> tuple[str, str]:
        # dask.delayed tokenizes its arguments: identify the cache, not its
        # (changing) contents
        return (type(self).__name__, self._token)

    def __getstate__(self) -> dict:
        # Only the token: the copies unpickled in one process share one state,
        # built there (never data read by another process).
        return {"token": self._token}

    def __setstate__(self, pickled: dict) -> None:
        self._token = pickled["token"]
        self._state = _PROCESS_STATES.acquire(self._token)
        # how many partitions this process converts is not known
        self._build_at_request = 2
        weakref.finalize(self, _PROCESS_STATES.release, self._token, self._state)


def active_subtable_cache() -> SubtableCache | None:
    """The sub-table cache active in this context, or None (uncached reads)."""
    return _ACTIVE_SUBTABLE_CACHE.get()


def subtable_cache_supported() -> bool:
    """
    Whether sub-table reads are cached with the casacore bindings in use:
    with python-casacore, not with the casatools shim (no in-place column
    reads, see the module docstring).
    """
    return backend_has_in_place_reads()


def resolve_subtable_cache(
    subtable_cache: SubtableCache | None,
) -> SubtableCache | None:
    """
    The sub-table cache a partition conversion should use.

    Parameters
    ----------
    subtable_cache : SubtableCache | None
        Cache shared by the partitions of a conversion, if any.

    Returns
    -------
    SubtableCache | None
        None if the cache is not supported (``subtable_cache_supported``).
        Otherwise ``subtable_cache``, or a new cache for this partition only if
        none was given (it memoizes and vectorizes the sub-table loads, but
        builds no whole-table value for a single use).
    """
    if not subtable_cache_supported():
        return None
    if subtable_cache is not None:
        return subtable_cache
    return SubtableCache(n_partitions=1)


@contextlib.contextmanager
def activate_subtable_cache(
    subtable_cache: SubtableCache | None,
) -> Generator[SubtableCache | None, None, None]:
    """
    Make ``subtable_cache`` the active sub-table cache of this context (thread)
    for the duration of the ``with`` block. None deactivates caching.
    """
    token = _ACTIVE_SUBTABLE_CACHE.set(subtable_cache)
    try:
        yield subtable_cache
    finally:
        _ACTIVE_SUBTABLE_CACHE.reset(token)
