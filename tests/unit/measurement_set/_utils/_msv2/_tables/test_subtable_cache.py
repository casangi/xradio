import pickle
import threading

import numpy as np
import pytest
import xarray as xr

from xradio.measurement_set._utils._msv2._tables import subtable_cache as sc


def test_resolve_subtable_cache():
    assert sc.subtable_cache_supported()  # python-casacore
    shared = sc.SubtableCache()
    assert sc.resolve_subtable_cache(shared) is shared
    own = sc.resolve_subtable_cache(None)
    assert isinstance(own, sc.SubtableCache) and own is not shared


def test_subtable_cache_not_used_with_casatools_like_tables(casatools_like_tables):
    """The casatools shim (no in-place reads) reads the sub-tables per
    partition: no sub-table cache."""
    assert not sc.subtable_cache_supported()
    assert sc.resolve_subtable_cache(sc.SubtableCache()) is None
    assert sc.resolve_subtable_cache(None) is None


def test_activate_subtable_cache_nests_and_resets():
    outer, inner = sc.SubtableCache(), sc.SubtableCache()
    assert sc.active_subtable_cache() is None
    with sc.activate_subtable_cache(outer):
        assert sc.active_subtable_cache() is outer
        with sc.activate_subtable_cache(inner):
            assert sc.active_subtable_cache() is inner
        with sc.activate_subtable_cache(None):
            assert sc.active_subtable_cache() is None
        assert sc.active_subtable_cache() is outer
    assert sc.active_subtable_cache() is None


def test_activate_subtable_cache_is_per_thread():
    cache = sc.SubtableCache()
    seen = []
    with sc.activate_subtable_cache(cache):
        thread = threading.Thread(
            target=lambda: seen.append(sc.active_subtable_cache())
        )
        thread.start()
        thread.join()
    assert seen == [None]


@pytest.mark.parametrize(
    "name, expected",
    [
        ("SYSCAL", True),
        ("WEATHER", True),
        ("ANTENNA", True),
        ("ASDM_STATION", True),
        ("POINTING", False),
        ("FIELD", False),
        ("PHASE_CAL", False),
        ("FIELD/EPHEM0_Venus.tab", False),
    ],
)
def test_is_memoized_table(name, expected):
    assert sc.is_memoized_table(name) is expected


def test_get_or_build_builds_once():
    cache = sc.SubtableCache()
    calls = []

    def builder():
        calls.append(1)
        return np.arange(3)

    first = cache.get_or_build(("k", 1), builder)
    second = cache.get_or_build(("k", 1), builder)
    assert first is second
    assert len(calls) == 1
    assert cache.get_or_build(("k", 2), lambda: None) is None
    assert cache.get_or_build(("k", 2), builder) is None  # None is a value too
    assert len(calls) == 1
    assert cache.stats["value_builds"] == 2
    assert cache.stats["value_hits"] == 2


def test_get_or_build_exception_not_stored():
    cache = sc.SubtableCache()

    def failing():
        raise RuntimeError("cannot read")

    with pytest.raises(RuntimeError, match="cannot read"):
        cache.get_or_build("k", failing)
    assert cache.get_or_build("k", lambda: 5) == 5


def test_get_or_build_concurrent_callers_wait_for_one_build():
    cache = sc.SubtableCache()
    started = threading.Event()
    release = threading.Event()
    calls = []

    def slow_builder():
        calls.append(1)
        started.set()
        release.wait(5)
        return "value"

    results = []
    threads = [
        threading.Thread(
            target=lambda: results.append(cache.get_or_build("k", slow_builder))
        )
        for _ in range(4)
    ]
    threads[0].start()
    started.wait(5)
    for thread in threads[1:]:
        thread.start()
    release.set()
    for thread in threads:
        thread.join()
    assert results == ["value"] * 4
    assert len(calls) == 1


def _dataset(nbytes_values: int = 8) -> xr.Dataset:
    return xr.Dataset(
        {"A": ("x", np.arange(nbytes_values // 8, dtype=np.float64), {"u": ["m"]})},
        attrs={"other": {"k": [1]}},
    )


def test_memo_dataset_returns_independent_copies():
    cache = sc.SubtableCache()
    calls = []

    def loader():
        calls.append(1)
        return _dataset(80)

    first = cache.memo_dataset("k", loader)
    first["A"].values[:] = -1
    first.attrs["other"]["k"].append(2)
    first["A"].attrs["u"].append("s")
    second = cache.memo_dataset("k", loader)
    third = cache.memo_dataset("k", loader)
    assert len(calls) == 1
    xr.testing.assert_identical(second, _dataset(80))
    assert second.attrs == {"other": {"k": [1]}}
    assert second["A"].attrs == {"u": ["m"]}
    assert second["A"].values is not third["A"].values
    assert cache.stats["memo_misses"] == 1 and cache.stats["memo_hits"] == 2


def test_memo_dataset_bounds(monkeypatch):
    monkeypatch.setattr(sc, "MEMO_MAX_ENTRIES", 3)
    monkeypatch.setattr(sc, "MEMO_MAX_TOTAL_BYTES", 400)
    monkeypatch.setattr(sc, "MEMO_MAX_DATASET_BYTES", 200)
    cache = sc.SubtableCache()
    calls = []

    def loader(nbytes):
        def load():
            calls.append(nbytes)
            return _dataset(nbytes)

        return load

    cache.memo_dataset("big", loader(800))  # larger than one dataset: not kept
    cache.memo_dataset("big", loader(800))
    assert calls == [800, 800]
    for key in ("a", "b", "c"):
        cache.memo_dataset(key, loader(160))  # 3 x 160 > 400: "a" is dropped
    calls.clear()
    cache.memo_dataset("c", loader(160))
    cache.memo_dataset("b", loader(160))
    assert calls == []
    cache.memo_dataset("a", loader(160))
    assert calls == [160]
    for key in ("d", "e", "f", "g"):
        cache.memo_dataset(key, loader(8))  # at most 3 entries
    calls.clear()
    cache.memo_dataset("g", loader(8))
    cache.memo_dataset("d", loader(8))
    assert calls == [8]


def test_clear():
    cache = sc.SubtableCache()
    cache.get_or_build("k", lambda: 1)
    cache.memo_dataset("m", _dataset)
    cache.clear()
    assert cache.get_or_build("k", lambda: 2) == 2
    calls = []
    cache.memo_dataset("m", lambda: calls.append(1) or _dataset())
    assert calls == [1]


class _ManualTimer:
    """threading.Timer stand-in that runs only when the test fires it."""

    def __init__(self, interval, function):
        self.interval, self.function = interval, function
        self.daemon = False
        self.started = self.fired = False

    def start(self):
        self.started = True

    def is_alive(self):
        return self.started and not self.fired

    def fire(self):
        self.fired = True
        self.function()


@pytest.fixture
def process_states(monkeypatch):
    """A fresh registry of unpickled cache states (as in a new worker), whose
    expiry timers run only when a test fires them (no timer thread)."""
    states = sc._ProcessStates()
    states.timer_factory = _ManualTimer
    monkeypatch.setattr(sc, "_PROCESS_STATES", states)
    return states


def test_pickled_copies_share_one_state_per_process(process_states):
    import gc

    cache = sc.SubtableCache(n_partitions=4)
    cache.get_or_build("k", lambda: "parent data")
    data = pickle.dumps(cache)
    assert b"parent data" not in data
    copy_a, copy_b = pickle.loads(data), pickle.loads(data)
    # a worker does not know how many partitions it converts: deferred build
    assert copy_a.get_or_build("k", lambda: "x", amortized=True) is None
    assert copy_b.get_or_build("k", lambda: "worker data", amortized=True) == (
        "worker data"
    )
    assert copy_a.get_or_build("k", lambda: "other") == "worker data"
    # another conversion's copies do not replace this state while it is alive
    other = pickle.loads(pickle.dumps(sc.SubtableCache()))
    other.get_or_build("k", lambda: "other conversion")
    assert pickle.loads(data).get_or_build("k", lambda: "rebuilt") == "worker data"
    # all copies released: the state idles, a later task of the conversion
    # (a new copy) gets it back
    state_ref = sc.weakref.ref(copy_a._state)
    del copy_a, copy_b
    gc.collect()
    assert state_ref() is not None and len(process_states.idle) >= 1
    assert pickle.loads(data).get_or_build("k", lambda: "rebuilt") == "worker data"


def test_idle_process_states_do_not_evict_each_other(process_states):
    """Two conversions whose tasks alternate in one worker keep their states."""
    import gc

    data = {name: pickle.dumps(sc.SubtableCache()) for name in ("a", "b")}
    for _ in range(3):
        for name, pickled in data.items():
            copy = pickle.loads(pickled)
            assert copy.get_or_build("k", lambda name=name: name) == name
            del copy
            gc.collect()
    assert len(process_states.idle) == 2


def test_idle_process_states_expire(process_states):
    """
    A worker does not keep a finished conversion's data: an idle state expires
    PROCESS_STATE_IDLE_SECONDS after its last copy was released (the clock and
    the timer are driven by the test, no wall-clock window).
    """
    import gc

    now = [1000.0]
    timers = []

    def timer_factory(interval, function):
        timers.append(_ManualTimer(interval, function))
        return timers[-1]

    process_states.clock = lambda: now[0]
    process_states.timer_factory = timer_factory
    idle_seconds = sc.PROCESS_STATE_IDLE_SECONDS
    data = pickle.dumps(sc.SubtableCache())
    copy = pickle.loads(data)
    copy.get_or_build("k", lambda: np.zeros(10))
    state_ref = sc.weakref.ref(copy._state)
    del copy
    gc.collect()
    assert state_ref() is not None  # idle, kept for the next task
    assert [(t.interval, t.started) for t in timers] == [(idle_seconds, True)]
    # the timer fires before the state was idle long enough: kept, rescheduled
    now[0] += idle_seconds / 2
    timers[0].fire()
    gc.collect()
    assert state_ref() is not None and list(process_states.idle)
    assert len(timers) == 2 and timers[1].started
    # idle long enough: dropped, no timer left
    now[0] += idle_seconds
    timers[1].fire()
    gc.collect()
    assert state_ref() is None
    assert not process_states.idle and process_states.timer is None
    assert len(timers) == 2
    # a new copy builds again
    assert pickle.loads(data).get_or_build("k", lambda: "new") == "new"


def test_max_idle_process_states(process_states):
    import gc

    for _ in range(sc.MAX_IDLE_PROCESS_STATES + 3):
        copy = pickle.loads(pickle.dumps(sc.SubtableCache()))
        copy.get_or_build("k", lambda: 1)
        del copy
        gc.collect()
    assert len(process_states.idle) == sc.MAX_IDLE_PROCESS_STATES


@pytest.mark.parametrize(
    "n_partitions, built_at", [(None, 2), (1, 2), (2, 1), (100, 1)]
)
def test_get_or_build_amortized(n_partitions, built_at):
    """Whole-table values are built at their first request only if two or more
    partitions share the cache; otherwise at the second one (the first reads
    per partition)."""
    cache = sc.SubtableCache(n_partitions=n_partitions)
    calls = []

    def builder():
        calls.append(1)
        return "value"

    results = [cache.get_or_build("k", builder, amortized=True) for _ in range(3)]
    assert results == [None] * (built_at - 1) + ["value"] * (4 - built_at)
    assert len(calls) == 1
    assert cache.stats["value_deferred"] == built_at - 1
    # values that are not amortized are always built at the first request
    assert cache.get_or_build("other", lambda: 5) == 5


def test_resolve_subtable_cache_for_one_partition_defers():
    own = sc.resolve_subtable_cache(None)
    assert own.get_or_build("k", lambda: 1, amortized=True) is None
    assert own.get_or_build("k", lambda: 1, amortized=True) == 1


def test_get_or_build_values_budget(monkeypatch):
    monkeypatch.setattr(sc, "VALUES_MAX_TOTAL_BYTES", 1000)
    cache = sc.SubtableCache(n_partitions=2)
    small = cache.get_or_build("small", lambda: (np.zeros(50), np.float64(1)))
    assert small[0].nbytes == 400
    calls = []

    def big():
        calls.append(1)
        return np.zeros(100)  # 800 B: 400 + 800 > 1000

    assert cache.get_or_build("big", big).size == 100  # used by its builder
    assert cache.get_or_build("big", big) is None  # then read per partition
    assert len(calls) == 1
    assert cache.stats["value_not_kept"] == 1
    assert cache.get_or_build("small", lambda: None) is small
    cache.clear()
    assert cache.get_or_build("big", big).size == 100  # budget freed


def test_count_is_thread_safe():
    cache = sc.SubtableCache()

    def work():
        for _ in range(2000):
            cache.count("n")

    threads = [threading.Thread(target=work) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert cache.stats["n"] == 16000


def test_dask_tokenize_identifies_the_instance():
    dask = pytest.importorskip("dask")
    cache = sc.SubtableCache()
    token = dask.base.tokenize(cache)
    cache.get_or_build("k", lambda: 1)
    assert dask.base.tokenize(cache) == token
    assert dask.base.tokenize(sc.SubtableCache()) != token
