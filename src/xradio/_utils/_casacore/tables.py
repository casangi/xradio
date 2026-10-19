import atexit
import contextlib
import os
import threading
from collections.abc import Generator
from contextlib import contextmanager

from xradio._utils._casacore.backend import tables

# common casacore table handling code


class ProcessLock:
    """
    A reentrant lock for a whole process, renewed in a fork child.

    It wraps a ``threading.RLock`` (``with lock:``, ``acquire``,
    ``release``), so that module-level aliases of one ProcessLock all follow
    the renewal: a child forked while another thread of the parent held the
    lock gets an unlocked one (the holder does not exist in the child), and a
    thread that holds it across the fork releases the one it acquired.

    Parameters
    ----------
    name : str
        Name (for repr).
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._lock = threading.RLock()
        # the locks this thread acquired and has not released (innermost last)
        self._held = threading.local()

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        """Acquire the lock (``threading.RLock.acquire``)."""
        lock = self._lock
        acquired = lock.acquire(blocking, timeout)
        if acquired:
            self._held.locks = getattr(self._held, "locks", []) + [lock]
        return acquired

    def release(self) -> None:
        """Release the lock last acquired by this thread."""
        locks = getattr(self._held, "locks", [])
        if not locks:
            raise RuntimeError(f"cannot release un-acquired lock {self!r}")
        self._held.locks = locks[:-1]
        locks[-1].release()

    def __enter__(self) -> "ProcessLock":
        self.acquire()
        return self

    def __exit__(self, *exc_info) -> None:
        self.release()

    def reset_after_fork(self) -> None:
        """Renew the lock (registered with os.register_at_fork for children)."""
        self._lock = threading.RLock()

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.name} {self._lock!r}>"


# casatools tables (the casatools shim) must
# not be used from several threads at once: concurrent reads of dask's threaded
# scheduler crashed (segmentation fault), hung, failed with FiledesIO errors or
# returned wrong values (casatools 6.7). Every lazy read of a casatools table,
# by the image or the MSv2 readers, holds this one lock over the open, the read,
# the close and the release of the table object. Other backends' reads take no
# lock (see casatools_serialized).
CASATOOLS_LOCK = ProcessLock("casatools")

if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=CASATOOLS_LOCK.reset_after_fork)


def uses_casatools() -> bool:
    """Whether xradio's casacore tables are casatools ones (the casatools shim)."""
    return tables.__name__.endswith("casacore_from_casatools")


def casatools_serialized() -> contextlib.AbstractContextManager:
    """
    The context in which to open, read, close and release a table from code
    that may run in several threads at once: CASATOOLS_LOCK with casatools,
    nothing (a null context) with python-casacore.
    """
    return CASATOOLS_LOCK if uses_casatools() else contextlib.nullcontext()


def extract_table_attributes(infile: str) -> dict[str, dict]:
    """
    return a dictionary of table attributes created from MS keywords and column descriptions
    """
    with open_table_ro(infile) as tb_tool:
        kwd = tb_tool.getkeywords()
        attrs = {kk: kwd[kk] for kk in kwd if kk not in os.listdir(infile)}
        cols = tb_tool.colnames()
        column_descriptions = {}
        for col in cols:
            column_descriptions[col] = tb_tool.getcoldesc(col)
        attrs["column_descriptions"] = column_descriptions
        attrs["info"] = tb_tool.info()
    return attrs


def _get_default_ninstances() -> int:
    env_val = os.environ.get("XRADIO_ARCAE_NINSTANCES")
    if env_val is not None:
        try:
            return max(1, int(env_val))
        except ValueError:
            pass
    return 1


def _make_table_ro(infile: str, ninstances: int | None = None) -> tables.table:
    if ninstances is None or ninstances <= 0:
        ninstances = _get_default_ninstances()
    kwargs = {
        "readonly": True,
        "lockoptions": {"option": "usernoread"},
        "ack": False,
    }
    if ninstances > 1:
        try:
            return tables.table(infile, ninstances=ninstances, **kwargs)
        except TypeError:
            # Backend does not support ninstances (e.g. python-casacore)
            pass
    return tables.table(infile, **kwargs)


@contextmanager
def open_table_ro(
    infile: str, ninstances: int | None = None
) -> Generator[tables.table, None, None]:
    table = _make_table_ro(infile, ninstances=ninstances)
    try:
        yield table
    finally:
        table.close()


@contextmanager
def open_table_rw(outfile: str) -> Generator[tables.table, None, None]:
    table = tables.table(
        outfile, readonly=False, lockoptions={"option": "permanentwait"}, ack=False
    )
    try:
        yield table
    finally:
        table.close()


_TABLE_CACHE: dict[tuple[str, int], tables.table] = {}
_TABLE_CACHE_LOCK = threading.Lock()


def get_table_ro(infile: str, ninstances: int | None = None) -> tables.table:
    """Retrieve a cached thread-safe read-only table proxy or open a new one.

    When ninstances > 1, caching the table allows multiple worker threads
    in the same process to share the underlying arcae IsolatedTableProxy pool
    without reopening the table on every chunk.
    """
    if ninstances is None or ninstances <= 0:
        ninstances = _get_default_ninstances()
    path = os.path.abspath(os.path.expanduser(infile))
    key = (path, ninstances)
    with _TABLE_CACHE_LOCK:
        if key not in _TABLE_CACHE:
            _TABLE_CACHE[key] = _make_table_ro(path, ninstances=ninstances)
        return _TABLE_CACHE[key]


def clear_table_cache() -> None:
    """Close and clear all cached table handles."""
    with _TABLE_CACHE_LOCK:
        for tb in _TABLE_CACHE.values():
            try:
                if hasattr(tb, "close"):
                    tb.close()
            except Exception:
                pass
        _TABLE_CACHE.clear()


atexit.register(clear_table_cache)


def _reset_table_cache_after_fork() -> None:
    """Empty the table cache and renew its lock in a fork child."""
    global _TABLE_CACHE_LOCK
    _TABLE_CACHE.clear()
    _TABLE_CACHE_LOCK = threading.Lock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_table_cache_after_fork)
