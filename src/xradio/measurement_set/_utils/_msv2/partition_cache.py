"""
The partitions of an MSv2 for the MSv2 xarray backend (engine
``xradio_msv2``): computed with ``create_partitions_with_main_rows`` and kept
in a per-process memo, valid while the MS is unchanged.

Modes (``partition_cache``, default ``$XRADIO_MSV2_PARTITION_CACHE`` or
"auto"):

- "auto" and "read": a valid memo entry is used, otherwise the partitions are
  computed and memoised;
- "rebuild": the partitions are computed again (and memoised);
- "off": the partitions are computed, and the memo is neither used nor
  filled.

The memo is valid while ``ms_state`` (sizes and modification times of the
files of the MAIN, FIELD, SOURCE, STATE and HISTORY tables) is unchanged.
It hands out copies, holds at most ``MEMO_MAX_ENTRIES`` results and
``MEMO_MAX_BYTES`` of row runs, computes a key once when several threads
open the same MS, and is emptied in a fork child.
"""

import collections
import copy
import dataclasses
import os
import threading

from xradio._utils._casacore.tables import casatools_serialized
from xradio.measurement_set._utils._msv2.partition_queries import (
    PARTITION_ALGORITHM_VERSION,
    MainRowRuns,
    canonical_scheme_key,
    create_partitions_with_main_rows,
    validate_partition_scheme,
)

# The values of partition_cache, and the environment variable that sets the
# default
PARTITION_CACHE_MODES = ("auto", "read", "off", "rebuild")
PARTITION_CACHE_ENV = "XRADIO_MSV2_PARTITION_CACHE"
# Bounds of the per-process memo of partitions
MEMO_MAX_ENTRIES = 16
MEMO_MAX_BYTES = 256 * 2**20
# The tables whose files the memo's validity follows (MAIN is "")
STATE_TABLES = ("", "FIELD", "SOURCE", "STATE", "HISTORY")


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


def ms_state(path: str) -> tuple:
    """
    The state of an MS the partitions depend on: size and modification time
    (ns) of every ``table.*`` file (but ``table.lock``, which lock requests
    rewrite) of the MAIN, FIELD, SOURCE, STATE and HISTORY tables, and which
    of these tables exist. Opens no table.
    """
    state = []
    for name in STATE_TABLES:
        directory = os.path.join(path, name)
        try:
            files = sorted(os.listdir(directory))
        except FileNotFoundError:
            state.append((name, None))
            continue
        for file_name in files:
            if not file_name.startswith("table.") or file_name == "table.lock":
                continue
            stat = os.stat(os.path.join(directory, file_name))
            state.append((name, file_name, stat.st_size, stat.st_mtime_ns))
    return tuple(state)


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
        "fresh" (computed by this call) or "memo".
    status : str
        How they were obtained: "hit-memory" (memo), "memory:computed",
        "memory:mode-off" or "memory:changed-during-build" (the MS changed
        while they were computed: not memoised).
    """

    partitions: list[dict]
    runs: MainRowRuns
    source: str
    status: str


def _copy_runs(runs: MainRowRuns) -> MainRowRuns:
    return MainRowRuns(
        runs.starts.copy(),
        runs.lengths.copy(),
        runs.bounds.copy(),
        runs.digests.copy(),
        runs.main_nrows,
    )


@dataclasses.dataclass
class _MemoEntry:
    state: tuple
    partitions: list[dict]
    runs: MainRowRuns

    def result(self, source: str, status: str) -> PartitionsResult:
        return PartitionsResult(
            copy.deepcopy(self.partitions), _copy_runs(self.runs), source, status
        )


class _PartitionsMemo:
    """LRU memo of partitions by (realpath, scheme key, algorithm version),
    with one build lock per key."""

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

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self.stats.clear()

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, key: tuple) -> bool:
        return key in self._entries


PARTITIONS_MEMO = _PartitionsMemo(MEMO_MAX_ENTRIES, MEMO_MAX_BYTES)

if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=PARTITIONS_MEMO.reset_after_fork)


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


def _compute(path: str, partition_scheme: list[str]) -> tuple[list[dict], MainRowRuns]:
    with casatools_serialized():  # (a no-op with python-casacore)
        return create_partitions_with_main_rows(path, partition_scheme)


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
        state = ms_state(path)
        if mode != "rebuild":
            entry = PARTITIONS_MEMO.get(key)
            if entry is not None and entry.state == state:
                PARTITIONS_MEMO.stats["hits"] += 1
                return entry.result("memo", "hit-memory")
        PARTITIONS_MEMO.stats["computed"] += 1
        partitions, runs = _compute(path, partition_scheme)
        if ms_state(path) != state:
            return PartitionsResult(
                partitions, runs, "fresh", "memory:changed-during-build"
            )
        entry = _MemoEntry(state, partitions, runs)
        PARTITIONS_MEMO.put(key, entry)
        return entry.result("fresh", "memory:computed")
