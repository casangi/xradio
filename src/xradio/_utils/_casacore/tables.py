import atexit
import os
import threading
from collections.abc import Generator
from contextlib import contextmanager

from xradio._utils._casacore.backend import tables

# common casacore table handling code


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
