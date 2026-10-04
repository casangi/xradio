"""
The one process-wide lock of casatools reads (xradio._utils._casacore.tables):
shared by the image and MSv2 readers, reentrant, taken only with casatools,
and free again in a fork child. The bindings in use are set by patching
``uses_casatools`` (the tests run with python-casacore or casatools).
"""

import contextlib
import os
import threading
import types

import numpy as np
import pytest

from xradio._utils._casacore import tables as table_utils
from xradio._utils._casacore.tables import CASATOOLS_LOCK, ProcessLock

WAIT = 0.2  # seconds a blocked thread is given to show it is blocked


def test_one_lock_for_image_and_msv2_reads():
    from xradio.image._util._casacore import xds_from_casacore
    from xradio.measurement_set._utils._msv2._tables import read

    assert read._CASATOOLS_READ_LOCK is CASATOOLS_LOCK
    assert xds_from_casacore._CASATOOLS_READ_LOCK is CASATOOLS_LOCK


def _acquired_by_other_thread(lock) -> bool:
    """Whether another thread can acquire the lock now (it releases it)."""
    result = []

    def other():
        got = lock.acquire(timeout=WAIT)
        result.append(got)
        if got:
            lock.release()

    thread = threading.Thread(target=other)
    thread.start()
    thread.join()
    return result[0]


def test_process_lock_is_reentrant_and_exclusive():
    lock = ProcessLock("test")
    with lock:
        with lock:  # reentrant
            assert not _acquired_by_other_thread(lock)
        assert not _acquired_by_other_thread(lock)
    assert _acquired_by_other_thread(lock)
    assert lock.acquire(blocking=False)
    lock.release()
    with pytest.raises(RuntimeError, match="un-acquired"):
        lock.release()
    assert "test" in repr(lock)


def test_casatools_serialized(monkeypatch):
    from xradio.measurement_set._utils._msv2._tables import read_rows

    # casatools tables are the ones without python-casacore's in-place reads
    assert table_utils.uses_casatools() == (not read_rows.backend_has_in_place_reads())
    monkeypatch.setattr(table_utils, "uses_casatools", lambda: False)
    with table_utils.casatools_serialized() as ctx:
        assert ctx is None  # a null context: the lock is free
        assert _acquired_by_other_thread(CASATOOLS_LOCK)
    monkeypatch.setattr(table_utils, "uses_casatools", lambda: True)
    with table_utils.casatools_serialized() as ctx:
        assert ctx is CASATOOLS_LOCK
        assert not _acquired_by_other_thread(CASATOOLS_LOCK)
    assert _acquired_by_other_thread(CASATOOLS_LOCK)


def _blocked_while_held(read) -> bool:
    """Run read() in another thread while this one holds CASATOOLS_LOCK:
    whether it waited for the lock (it completes after the release)."""
    done = threading.Event()

    def run():
        read()
        done.set()

    with CASATOOLS_LOCK:
        thread = threading.Thread(target=run)
        thread.start()
        blocked = not done.wait(WAIT)
    thread.join(10)
    assert done.is_set()
    return blocked


@pytest.mark.parametrize("casatools", [False, True])
def test_image_and_msv2_reads_take_the_lock(monkeypatch, casatools):
    """The lazy image chunk reads and the lazy MAIN time-chunk reads hold the
    one lock with casatools, and run unlocked with python-casacore."""
    from xradio.image._util._casacore import xds_from_casacore
    from xradio.measurement_set._utils._msv2._tables import read

    monkeypatch.setattr(table_utils, "uses_casatools", lambda: casatools)
    chunk = np.zeros((2, 3))
    monkeypatch.setattr(
        xds_from_casacore, "_read_image_chunk_unlocked", lambda *args: chunk
    )

    def image_read():
        assert xds_from_casacore._read_image_chunk("x.im", (2, 3), (0, 0)) is chunk

    assert _blocked_while_held(image_read) == casatools

    @contextlib.contextmanager
    def no_table(in_file):
        yield None

    monkeypatch.setattr(read, "open_table_ro", no_table)
    monkeypatch.setattr(read, "read_time_chunk", lambda *args: chunk)
    chunk_rows = types.SimpleNamespace(chunk_n_rows=lambda k: 1)

    def msv2_read():
        assert (
            read._load_rows_time_chunk(
                "x.ms", "DATA", chunk_rows, 0, (2, 3), np.float64, 100
            )
            is chunk
        )

    assert _blocked_while_held(msv2_read) == casatools


def _in_fork_child(child) -> int:
    """Run child() in a fork child; its exit status (0: child() was True)."""
    pid = os.fork()
    if pid == 0:  # the child: never return into pytest
        status = 1
        try:
            status = 0 if child() else 1
        finally:
            os._exit(status)
    _, status = os.waitpid(pid, 0)
    return os.waitstatus_to_exitcode(status)


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs os.fork")
@pytest.mark.filterwarnings("ignore:This process .* is multi-threaded")
def test_fork_child_gets_free_locks():
    """
    A child forked while another thread holds the casatools lock and the
    sub-table cache's process-state lock can take both; a thread that holds
    the casatools lock across the fork releases it in the child.
    """
    from xradio.measurement_set._utils._msv2._tables import subtable_cache

    states = subtable_cache._PROCESS_STATES
    holding, finish = threading.Event(), threading.Event()

    def hold():
        with CASATOOLS_LOCK, states.lock:
            holding.set()
            finish.wait(30)

    holder = threading.Thread(target=hold)
    holder.start()
    try:
        assert holding.wait(10)

        def child():
            if not CASATOOLS_LOCK.acquire(timeout=5):
                return False
            CASATOOLS_LOCK.release()
            if not states.lock.acquire(timeout=5):
                return False
            states.lock.release()
            return states.timer is None and not states.idle

        assert _in_fork_child(child) == 0
    finally:
        finish.set()
        holder.join(10)

    with CASATOOLS_LOCK:

        def child_holding():
            CASATOOLS_LOCK.release()  # the one acquired before the fork
            return _acquired_by_other_thread(CASATOOLS_LOCK)

        assert _in_fork_child(child_holding) == 0
        assert not _acquired_by_other_thread(CASATOOLS_LOCK)  # parent unchanged
