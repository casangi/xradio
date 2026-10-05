"""
The lazy variables of the ``xradio_msv2`` engine read in other processes and
threads: a fork child (also one forked while other threads hold every lock of
the backend), dask's processes scheduler (spawned workers, which unpickle the
arrays and rebuild their indices from the MS), a distributed LocalCluster
with worker processes, and 8 dask threads that rebuild the indices at once.
All give the values of a synchronous read in this process (which equal the
converter's: test_open_msv2_equivalence.py).
"""

import contextlib
import hashlib
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import dask
import numpy as np
import pytest
import xarray as xr

from _xradio_xarray_backends import MSv2BackendEntrypoint
from xradio._utils._casacore.tables import CASATOOLS_LOCK
from xradio.measurement_set._utils._msv2 import (
    backend_arrays,
    backend_pointing,
    partition_cache,
)
from xradio.measurement_set._utils._msv2._tables import subtable_cache
from xradio.testing.measurement_set.equivalence import main_data_variables

ENGINE = MSv2BackendEntrypoint
# Several time chunks per main data variable (30 times): more tasks than
# workers. The session MSs are read only: partitions are never stored (the
# module fixture is set up before the autouse fixture that turns the cache
# off).
OPTIONS = {"main_chunksize": {"time": 7}, "partition_cache": "off"}


def lazy_variables(tree: xr.DataTree) -> dict[str, xr.DataArray]:
    """The variables read from the MS on access: the main data variables and
    the pointing_xds data variables of every MSv4, by path."""
    found = {}
    for name, node in sorted(tree.children.items()):
        for var in main_data_variables(node):
            found[f"{name}/{var}"] = node[var]
        if "pointing_xds" in node.children:
            for var, data in node["pointing_xds"].data_vars.items():
                found[f"{name}/pointing_xds/{var}"] = data
    return found


def digests(values: dict) -> dict[str, str]:
    """dtype, shape and sha256 of the bytes of every array."""
    return {
        name: f"{arr.dtype}|{arr.shape}|"
        + hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()
        for name, arr in ((name, np.asarray(v)) for name, v in values.items())
    }


def compute(arrays: dict[str, xr.DataArray], **kwargs) -> dict:
    """The values of the (dask-backed) arrays, computed together."""
    names = list(arrays)
    results = dask.compute(*(arrays[name].data for name in names), **kwargs)
    return dict(zip(names, results, strict=True))


def in_new_threads(n: int) -> dict:
    """dask.compute options: the threaded scheduler with a new pool of
    ``n`` threads. (A fork child cannot use dask's default pool, or any pool
    the parent used: their threads do not exist in the child, so their
    tasks would never run.)"""
    return {"scheduler": "threads", "pool": ThreadPoolExecutor(n)}


def clear_memos() -> None:
    """Forget the indices of this process (reads rebuild them from the MS)."""
    backend_arrays.clear_index_memo()
    backend_pointing.clear_pointing_memos()


@pytest.fixture(scope="module")
def opened(backend_ms):
    """The MS ("rich": 3 data groups, POINTING), its tree (chunks={}) and
    the digests of every lazy variable read synchronously."""
    msname = backend_ms("rich")
    tree = xr.open_datatree(msname, engine=ENGINE, chunks={}, **OPTIONS)
    arrays = lazy_variables(tree)
    assert any("pointing_xds" in name for name in arrays)
    expected = digests(compute(arrays, scheduler="synchronous"))
    return msname, tree, expected


def wait_child(pid: int, seconds: float = 120.0) -> int | None:
    """The exit code of a forked child; a child still running after
    ``seconds`` (e.g. deadlocked) is killed: None."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        done, status = os.waitpid(pid, os.WNOHANG)
        if done:
            return os.waitstatus_to_exitcode(status)
        time.sleep(0.05)
    os.kill(pid, 9)
    os.waitpid(pid, 0)
    return None


def backend_locks(msname: str) -> list:
    """Every lock of the backend that a thread may hold while another
    thread forks."""
    key = partition_cache.memo_key(os.path.abspath(msname), [])
    return [
        CASATOOLS_LOCK,
        backend_arrays.INDEX_MEMO._lock,
        *backend_arrays.INDEX_MEMO._build_locks,
        partition_cache.PARTITIONS_MEMO._lock,
        partition_cache.PARTITIONS_MEMO.build_lock(key),
        partition_cache._NOTICES_LOCK,
        partition_cache._WRITE_MUTEXES_LOCK,
        partition_cache.write_mutex(msname),
        backend_pointing.POINTING_INDEX_MEMO._lock,
        *backend_pointing.POINTING_INDEX_MEMO._build_locks,
        backend_pointing.POINTING_SELECTION_MEMO._lock,
        *backend_pointing.POINTING_SELECTION_MEMO._build_locks,
        subtable_cache._PROCESS_STATES.lock,
    ]


@contextlib.contextmanager
def held_by_another_thread(locks: list):
    """While active, another thread holds ``locks``."""
    holding, finish = threading.Event(), threading.Event()

    def hold():
        with contextlib.ExitStack() as stack:
            for lock in locks:
                stack.enter_context(lock)
            holding.set()
            finish.wait(120)

    holder = threading.Thread(target=hold)
    holder.start()
    try:
        assert holding.wait(30)
        yield
    finally:
        finish.set()
        holder.join(30)


def fork_and_check(child) -> int | None:
    """Run ``child()`` (True: success) in a fork child; its exit code."""
    pid = os.fork()
    if pid == 0:  # the child: never return into pytest
        code = 1
        try:
            code = 0 if child() else 2
        finally:
            os._exit(code)
    return wait_child(pid)


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs os.fork")
@pytest.mark.filterwarnings("ignore:This process .* is multi-threaded")
def test_fork_child_reads_the_values(opened):
    """A fork child of a process that opened the MS and read its variables
    reads the same values from the parent's tree (its memos are empty: the
    indices are rebuilt from the MS) and from a new open."""
    msname, tree, expected = opened
    arrays = lazy_variables(tree)
    compute(arrays, scheduler="threads")  # (the parent's memos are filled)

    def child():
        if len(backend_arrays.INDEX_MEMO) or len(partition_cache.PARTITIONS_MEMO):
            return False
        same = digests(compute(arrays, scheduler="synchronous")) == expected
        reopened = xr.open_datatree(msname, engine=ENGINE, chunks={}, **OPTIONS)
        new = lazy_variables(reopened)
        return same and digests(compute(new, **in_new_threads(4))) == expected

    assert fork_and_check(child) == 0


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs os.fork")
@pytest.mark.filterwarnings("ignore:This process .* is multi-threaded")
def test_fork_while_other_threads_hold_the_locks(opened, ms_copy):
    """
    A child forked while another thread holds every lock of the backend (the
    casatools lock, the index, partition and pointing memos and their build
    locks, the cache's write mutexes and notices, the sub-table cache state)
    opens a copy of the MS (storing its partitions) and reads every lazy
    variable, instead of deadlocking.
    """
    _, _, expected = opened
    msname = ms_copy("rich", name="rich.ms")
    clear_memos()

    def child():
        tree = xr.open_datatree(
            msname, engine=ENGINE, chunks={}, **(OPTIONS | {"partition_cache": "auto"})
        )
        values = compute(lazy_variables(tree), **in_new_threads(4))
        return digests(values) == expected and os.path.isdir(
            os.path.join(msname, partition_cache.SUBTABLE_NAME)
        )

    with held_by_another_thread(backend_locks(msname)):
        assert fork_and_check(child) == 0


@pytest.mark.filterwarnings("ignore:This process .* is multi-threaded")
def test_dask_processes_scheduler(opened):
    """dask's processes scheduler (spawned workers: the arrays are pickled,
    and every worker rebuilds the indices it needs) gives the values."""
    _, tree, expected = opened
    values = compute(lazy_variables(tree), scheduler="processes", num_workers=2)
    assert digests(values) == expected


def test_distributed_local_cluster(opened):
    """A distributed LocalCluster with worker processes gives the values."""
    distributed = pytest.importorskip("distributed")
    _, tree, expected = opened
    arrays = lazy_variables(tree)
    with (
        distributed.LocalCluster(
            n_workers=2,
            threads_per_worker=2,
            processes=True,
            dashboard_address=None,
        ) as cluster,
        distributed.Client(cluster) as client,
    ):
        names = list(arrays)
        futures = client.compute([arrays[name].data for name in names])
        values = dict(zip(names, client.gather(futures), strict=True))
    assert digests(values) == expected


def test_eight_threads_rebuild_and_read(opened):
    """8 dask threads, starting with empty memos (so that they rebuild the
    indices at once), give the values; every index is built once."""
    _, tree, expected = opened
    arrays = lazy_variables(tree)
    clear_memos()
    values = compute(arrays, scheduler="threads", num_workers=8)
    assert digests(values) == expected
    n_main = sum(1 for name in arrays if "pointing_xds" not in name)
    assert 0 < backend_arrays.INDEX_MEMO.stats["rebuilds"] <= len(tree.children)
    assert n_main > len(tree.children)
