import pickle
import threading

import numpy as np
import pytest
import xarray as xr

from xradio.measurement_set._utils._msv2._tables import subtable_cache as sc


def test_get_subtable_cache_mode(monkeypatch):
    monkeypatch.delenv(sc.SUBTABLE_CACHE_ENV_VAR, raising=False)
    assert sc.get_subtable_cache_mode() is True
    monkeypatch.setenv(sc.SUBTABLE_CACHE_ENV_VAR, "0")
    assert sc.get_subtable_cache_mode() is False
    monkeypatch.setenv(sc.SUBTABLE_CACHE_ENV_VAR, " 1 ")
    assert sc.get_subtable_cache_mode() is True
    monkeypatch.setenv(sc.SUBTABLE_CACHE_ENV_VAR, "yes")
    with pytest.raises(ValueError, match="XRADIO_MSV2_SUBTABLE_CACHE"):
        sc.get_subtable_cache_mode()


def test_resolve_subtable_cache(monkeypatch):
    shared = sc.SubtableCache()
    monkeypatch.setenv(sc.SUBTABLE_CACHE_ENV_VAR, "1")
    assert sc.resolve_subtable_cache(shared) is shared
    own = sc.resolve_subtable_cache(None)
    assert isinstance(own, sc.SubtableCache) and own is not shared
    monkeypatch.setenv(sc.SUBTABLE_CACHE_ENV_VAR, "0")
    assert sc.resolve_subtable_cache(shared) is None
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


def test_pickled_copies_share_one_state_per_process(monkeypatch):
    monkeypatch.setattr(sc, "_PROCESS_STATES", sc.collections.OrderedDict())
    cache = sc.SubtableCache()
    cache.get_or_build("k", lambda: "parent data")
    data = pickle.dumps(cache)
    assert b"parent data" not in data
    copy_a, copy_b = pickle.loads(data), pickle.loads(data)
    assert copy_a.get_or_build("k", lambda: "worker data") == "worker data"
    assert copy_b.get_or_build("k", lambda: "other") == "worker data"
    # another conversion replaces the state kept in this process
    other = pickle.loads(pickle.dumps(sc.SubtableCache()))
    other.get_or_build("k", lambda: "other conversion")
    assert pickle.loads(data).get_or_build("k", lambda: "rebuilt") == "rebuilt"


def test_dask_tokenize_identifies_the_instance():
    dask = pytest.importorskip("dask")
    cache = sc.SubtableCache()
    token = dask.base.tokenize(cache)
    cache.get_or_build("k", lambda: 1)
    assert dask.base.tokenize(cache) == token
    assert dask.base.tokenize(sc.SubtableCache()) != token
