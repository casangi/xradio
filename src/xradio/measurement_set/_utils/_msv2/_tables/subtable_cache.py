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
  vectorized ``getcol`` calls instead of one ``row()`` dict per table row.

The output is identical to the uncached reads. The MS must not change during a
conversion.

Sharing: the cache never holds an open casacore table (tables are opened, read
and closed inside the builders), so it is safe to share between the threads of
``parallel_mode="partition"`` and across ``fork()``. A pickled cache carries
only its token: the copies unpickled in one (worker) process share one state,
built in that process on first use. A worker process keeps the state of the
most recent conversion only (``MAX_PROCESS_STATES``).
"""

import collections
import contextlib
import contextvars
import os
import threading
import uuid
from collections.abc import Callable, Generator, Hashable
from typing import Any

import xarray as xr

# TEMPORARY, EXPLORATION ONLY (remove before merging): "1" (default) enables the
# sub-table cache for the A/B benchmarks, "0" selects the previous per-partition
# sub-table reads everywhere.
SUBTABLE_CACHE_ENV_VAR = "XRADIO_MSV2_SUBTABLE_CACHE"
SUBTABLE_CACHE_MODES = ("0", "1")

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
# Unpickled caches (worker processes): states kept per process
MAX_PROCESS_STATES = 1

_ACTIVE_SUBTABLE_CACHE: contextvars.ContextVar["SubtableCache | None"] = (
    contextvars.ContextVar("xradio_msv2_subtable_cache", default=None)
)


def get_subtable_cache_mode() -> bool:
    """
    TEMPORARY, EXPLORATION ONLY: whether the sub-table cache is enabled, from the
    environment variable XRADIO_MSV2_SUBTABLE_CACHE ("1", the default, or "0").

    Returns
    -------
    bool
        True if sub-table reads are cached.
    """
    value = os.environ.get(SUBTABLE_CACHE_ENV_VAR, "").strip() or "1"
    if value not in SUBTABLE_CACHE_MODES:
        raise ValueError(
            f"{SUBTABLE_CACHE_ENV_VAR}={value!r} is not one of {SUBTABLE_CACHE_MODES}"
        )
    return value == "1"


def is_memoized_table(table_name: str) -> bool:
    """Whether load_generic_table results of this sub-table are memoized."""
    return table_name in MEMOIZED_TABLES or table_name.startswith("ASDM_")


class _SubtableCacheState:
    """The data of a SubtableCache (shared by its unpickled copies)."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.build_locks: dict[Hashable, threading.Lock] = {}
        self.values: dict[Hashable, Any] = {}
        self.memo: collections.OrderedDict[Hashable, xr.Dataset] = (
            collections.OrderedDict()
        )
        self.memo_bytes = 0
        self.stats: collections.Counter = collections.Counter()


_PROCESS_STATES: collections.OrderedDict[str, _SubtableCacheState] = (
    collections.OrderedDict()
)
_PROCESS_STATES_LOCK = threading.Lock()


class SubtableCache:
    """
    Sub-table data shared by the partitions of one conversion.

    Two kinds of entries:

    - values built once per key and shared read-only (``get_or_build``), e.g.
      the sorted POINTING columns. Concurrent callers of the same key wait for
      the first builder.
    - memoized datasets (``memo_dataset``), kept in a bounded LRU; every caller
      gets a deep copy, so callers may modify what they get.
    """

    def __init__(self) -> None:
        self._token = uuid.uuid4().hex
        self._state = _SubtableCacheState()

    @property
    def stats(self) -> collections.Counter:
        """Counters of cache hits, builds, fallbacks, ..."""
        return self._state.stats

    def get_or_build(self, key: Hashable, builder: Callable[[], Any]) -> Any:
        """
        The value stored under ``key``, built with ``builder()`` on first use.

        Parameters
        ----------
        key : Hashable
            Cache key (should include the table path).
        builder : Callable[[], Any]
            Builds the value. Exceptions propagate and nothing is stored.

        Returns
        -------
        Any
            The shared value (not copied: callers must not modify it).
        """
        state = self._state
        with state.lock:
            if key in state.values:
                state.stats["value_hits"] += 1
                return state.values[key]
            build_lock = state.build_locks.setdefault(key, threading.Lock())
        with build_lock:
            with state.lock:
                if key in state.values:
                    state.stats["value_hits"] += 1
                    return state.values[key]
            value = builder()
            with state.lock:
                state.values[key] = value
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
            state.memo.clear()
            state.memo_bytes = 0
        with _PROCESS_STATES_LOCK:
            _PROCESS_STATES.pop(self._token, None)

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
        with _PROCESS_STATES_LOCK:
            state = _PROCESS_STATES.get(self._token)
            if state is None:
                state = _SubtableCacheState()
                _PROCESS_STATES[self._token] = state
                while len(_PROCESS_STATES) > MAX_PROCESS_STATES:
                    _PROCESS_STATES.popitem(last=False)
            else:
                _PROCESS_STATES.move_to_end(self._token)
        self._state = state


def active_subtable_cache() -> SubtableCache | None:
    """The sub-table cache active in this context, or None (uncached reads)."""
    return _ACTIVE_SUBTABLE_CACHE.get()


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
        None when XRADIO_MSV2_SUBTABLE_CACHE=0. Otherwise ``subtable_cache``, or
        a new cache for this partition only if none was given.
    """
    if not get_subtable_cache_mode():
        return None
    return subtable_cache if subtable_cache is not None else SubtableCache()


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
