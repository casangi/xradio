"""Tests of the lazy arrays of the MSv2 xarray backend (backend_arrays.py)."""

import contextlib
import os
import pickle
import shutil
import time

import cloudpickle
import dask
import numpy as np
import pytest
import xarray as xr

from xradio.measurement_set._utils._msv2 import backend_arrays, conversion, stream_write
from xradio.measurement_set._utils._msv2.backend_arrays import (
    INDEX_MEMO,
    MSv2BackendArray,
    MSv2MainColumnArray,
    OnesArray,
    PartitionIndex,
    _IndexEntry,
    _IndexMemo,
    index_token,
    normalize_basic_key,
)
from xradio.measurement_set._utils._msv2.backend_errors import (
    MSv2ChangedError,
    MSv2ReadError,
)
from xradio.measurement_set._utils._msv2.partition_queries import (
    create_partitions_with_main_rows,
)

# --- the block base class (cases copied from the ASDM backend's tests) -------

DIMS = ("time", "baseline_id", "frequency", "polarization")
SHAPE = (7, 6, 8, 2)


def reference_values(shape=SHAPE, dtype=np.float64) -> np.ndarray:
    """Values that encode their position (every element distinct)."""
    values = np.arange(np.prod(shape), dtype=np.float64).reshape(shape)
    if np.dtype(dtype).kind == "c":
        values = values + 1j * (values + 0.5)
    return values.astype(dtype)


def check_normalised_key(key, shape):
    """One slice(start, stop, 1) per dim, Python ints, 0 <= start < stop <= len."""
    assert isinstance(key, tuple)
    assert len(key) == len(shape)
    for dim_key, dim_len in zip(key, shape, strict=True):
        assert isinstance(dim_key, slice)
        assert type(dim_key.start) is int and type(dim_key.stop) is int
        assert dim_key.step == 1
        assert 0 <= dim_key.start < dim_key.stop <= dim_len


class FakeArray(MSv2BackendArray):
    """Backend array whose loader returns blocks of a reference array, recording
    the keys it receives."""

    def __init__(self, values: np.ndarray, dtype=None, loader_dtype=None):
        super().__init__(values.shape, values.dtype if dtype is None else dtype)
        self.values = values if loader_dtype is None else values.astype(loader_dtype)
        self.keys = []

    def _raw_indexing_method(self, key):
        check_normalised_key(key, self.shape)
        self.keys.append(key)
        return self.values[key].copy()


def lazy_variable(backend_array, dims=DIMS) -> xr.Variable:
    return xr.Variable(dims, xr.core.indexing.LazilyIndexedArray(backend_array))


LAZY_KEYS = {
    "int_time": {"time": 4},
    "int_negative": {"baseline_id": -1},
    "time_step2": {"time": slice(None, None, 2)},
    "time_step3_from1": {"time": slice(1, None, 3)},
    "time_reversed": {"time": slice(None, None, -1)},
    "time_neg_step": {"time": slice(6, 1, -3)},
    "neg_step_to_start": {"frequency": slice(7, None, -3)},
    "empty": {"time": slice(4, 4)},
    "empty_reversed": {"frequency": slice(2, 5, -1)},
    "list": {"time": [6, 0, 3]},
    "bl_step": {"baseline_id": slice(1, None, 3)},
    "freq_int_first": {"frequency": 0},
    "freq_int_last": {"frequency": 7},
    "pol_int": {"polarization": 1},
    "ints_all": {"time": 3, "baseline_id": 2, "frequency": 5, "polarization": 0},
    "mixed": {
        "time": slice(1, 7, 2),
        "baseline_id": slice(3, 6),
        "frequency": slice(2, 7, 2),
        "polarization": -1,
    },
}


def numpy_key(dims, isel):
    return tuple(isel.get(dim, slice(None)) for dim in dims)


@pytest.mark.parametrize(
    "key, shape, expected",
    [
        (
            (slice(None), 3, slice(2, 5), -1),
            SHAPE,
            (
                (slice(0, 7, 1), slice(3, 4, 1), slice(2, 5, 1), slice(1, 2, 1)),
                (slice(None), 0, slice(None), 0),
                (7, 3),
            ),
        ),
        (
            (slice(None, None, 2), slice(6, 1, -3)),
            (7, 8),
            (
                (slice(0, 7, 1), slice(3, 7, 1)),
                (slice(0, None, 2), slice(3, None, -3)),
                (4, 2),
            ),
        ),
        (
            (slice(3, 3),),
            (7, 8),
            ((slice(0, 0, 1), slice(0, 8, 1)), (slice(None), slice(None)), (0, 8)),
        ),
        (
            (np.int64(2), slice(None, None, -1)),
            (7, 8),
            ((slice(2, 3, 1), slice(0, 8, 1)), (0, slice(7, None, -1)), (8,)),
        ),
    ],
)
def test_normalize_basic_key(key, shape, expected):
    block_key, residual_key, result_shape = normalize_basic_key(key, shape)
    assert (block_key, residual_key, result_shape) == expected
    reference = reference_values(shape)
    if 0 not in result_shape:
        np.testing.assert_array_equal(
            reference[block_key][residual_key], reference[key]
        )
        for dim_key in block_key:
            assert type(dim_key.start) is int and type(dim_key.stop) is int


@pytest.mark.parametrize(
    "key, error",
    [
        ((7,), IndexError),
        ((-8,), IndexError),
        ((0, 0, 0), IndexError),
        (([0, 1],), TypeError),
        ((None,), TypeError),
    ],
)
def test_normalize_basic_key_errors(key, error):
    with pytest.raises(error):
        normalize_basic_key(key, (7, 8))


def test_MSv2BackendArray_base():
    backend_array = MSv2BackendArray((2, 3), np.float32)
    assert backend_array.shape == (2, 3)
    assert backend_array.dtype == np.dtype("float32")
    with pytest.raises(NotImplementedError):
        backend_array._raw_indexing_method((slice(0, 1, 1), slice(0, 2, 1)))
    with pytest.raises(NotImplementedError):
        lazy_variable(backend_array, ("a", "b")).values  # noqa: B018
    with pytest.raises(TypeError):
        MSv2BackendArray(None, np.float32)
    with pytest.raises(ValueError, match="negative"):
        MSv2BackendArray((2, -1), np.float32)


@pytest.mark.parametrize("key_name", list(LAZY_KEYS))
@pytest.mark.parametrize("dtype", [np.float64, np.complex64, bool])
def test_lazy_indexing_matches_numpy(key_name, dtype):
    """Every xarray selection (ints, steps, negative steps, empty, lists, mixed)
    gives the values, shape and dtype of numpy indexing of the full array, and
    the loader only receives normalised step-1 slices."""
    reference = reference_values(dtype=dtype)
    fake = FakeArray(reference)
    isel = LAZY_KEYS[key_name]
    expected = reference[numpy_key(DIMS, isel)]
    result = lazy_variable(fake).isel(isel)
    assert result.shape == expected.shape
    values = result.values
    assert values.dtype == np.dtype(dtype)
    np.testing.assert_array_equal(values, expected)
    if 0 in expected.shape:
        assert fake.keys == []
    else:
        assert len(fake.keys) == 1


def test_lazy_indexing_loads_only_the_bounding_block():
    fake = FakeArray(reference_values())
    values = (
        lazy_variable(fake)
        .isel(time=5, baseline_id=slice(4, 0, -2), frequency=[6, 3])
        .values
    )
    np.testing.assert_array_equal(values, fake.values[5, 4:0:-2][:, [6, 3]])
    assert fake.keys == [
        (slice(5, 6, 1), slice(2, 5, 1), slice(3, 7, 1), slice(0, 2, 1))
    ]


@pytest.mark.parametrize(
    "chunks",
    [
        {"time": 3},
        {"baseline_id": 4},
        {"frequency": 3},
        {"polarization": 1},
        {"time": 2, "baseline_id": 5, "frequency": 3},
    ],
)
def test_dask_chunks_match_numpy(chunks):
    reference = reference_values(dtype=np.complex64)
    fake = FakeArray(reference)
    chunked = lazy_variable(fake).chunk(chunks)
    np.testing.assert_array_equal(chunked.values, reference)
    np.testing.assert_array_equal(
        chunked[2, 1:5, ::-3, 0].values, reference[2, 1:5, ::-3, 0]
    )


def test_cast_to_declared_dtype_and_writeable():
    """The result always has the declared dtype and is a writeable array,
    even when the loader returns a read-only view."""
    reference = reference_values(dtype=np.float32)
    fake = FakeArray(reference, dtype=np.complex64)
    values = lazy_variable(fake).isel(time=slice(0, 3)).values
    assert values.dtype == np.complex64
    np.testing.assert_array_equal(values.real, reference[0:3])
    np.testing.assert_array_equal(values.imag, 0.0)

    class ReadOnlyFake(FakeArray):
        def _raw_indexing_method(self, key):
            block = super()._raw_indexing_method(key)
            return np.broadcast_to(block, block.shape)

    values = lazy_variable(ReadOnlyFake(reference)).values
    assert values.flags.writeable
    values[0, 0, 0, 0] = -1.0


def test_complex_values_for_real_dtype_are_rejected():
    fake = FakeArray(reference_values(dtype=np.complex64), dtype=np.float32)
    with pytest.raises(TypeError, match="complex"):
        lazy_variable(fake).values  # noqa: B018


def test_loader_with_wrong_shape_is_an_error():
    class WrongShape(FakeArray):
        def _raw_indexing_method(self, key):
            return super()._raw_indexing_method(key)[:, :1]

    with pytest.raises(RuntimeError, match="expected"):
        lazy_variable(WrongShape(reference_values())).isel(time=0).values  # noqa: B018


def test_loader_exceptions_propagate():
    class Failing(FakeArray):
        def _raw_indexing_method(self, key):
            raise NotImplementedError("not supported")

    with pytest.raises(NotImplementedError, match="not supported"):
        lazy_variable(Failing(reference_values())).isel(time=0).values  # noqa: B018


# --- PartitionIndex: rows of a block, memo -----------------------------------


def _synthetic_index(tidxs, bidxs, shape, rows=None, path="/no/such.ms"):
    """A PartitionIndex whose memo entry holds the given (time, baseline)
    indices (rows 10, 11, ... by default)."""
    tidxs, bidxs = np.asarray(tidxs), np.asarray(bidxs)
    if rows is None:
        rows = 10 + np.arange(tidxs.size)
    starts, lengths = backend_arrays.rows_to_runs(np.asarray(rows))
    index = PartitionIndex(path, starts, lengths, 10**6, shape, "token")
    INDEX_MEMO.put(index.memo_key(), _IndexEntry(rows, tidxs, bidxs, shape[0]))
    return index, np.asarray(rows)


def _brute_force_select(rows, tidxs, bidxs, t0, t1, b0, b1):
    keep = (tidxs >= t0) & (tidxs < t1) & (bidxs >= b0) & (bidxs < b1)
    return rows[keep], (tidxs[keep] - t0) * (b1 - b0) + (bidxs[keep] - b0)


SELECT_LAYOUTS = {
    "time_ordered": lambda nt, nb: (
        np.repeat(np.arange(nt), nb),
        np.tile(np.arange(nb), nt),
    ),
    "reversed": lambda nt, nb: (
        np.repeat(np.arange(nt)[::-1], nb),
        np.tile(np.arange(nb), nt),
    ),
    "baseline_major": lambda nt, nb: (
        np.tile(np.arange(nt), nb),
        np.repeat(np.arange(nb), nt),
    ),
    "interleaved": lambda nt, nb: (
        np.repeat(np.arange(nt), nb).reshape(nt // 2, 2, nb).transpose(1, 0, 2).ravel(),
        np.tile(np.arange(nb), nt),
    ),
    "sparse": lambda nt, nb: (
        np.repeat(np.arange(nt), nb)[::3],
        np.tile(np.arange(nb), nt)[::3],
    ),
    "empty": lambda nt, nb: (np.empty(0, np.int64), np.empty(0, np.int64)),
}


@pytest.mark.parametrize("layout", list(SELECT_LAYOUTS))
@pytest.mark.parametrize(
    "window", [(0, 8, 0, 5), (2, 5, 0, 5), (0, 8, 1, 3), (3, 4, 4, 5), (7, 8, 0, 1)]
)
def test_partition_index_select_matches_brute_force(layout, window):
    nt, nb = 8, 5
    tidxs, bidxs = SELECT_LAYOUTS[layout](nt, nb)
    index, rows = _synthetic_index(tidxs, bidxs, (nt, nb))
    try:
        got_rows, got_cells = index.select(*window)
        exp_rows, exp_cells = _brute_force_select(rows, tidxs, bidxs, *window)
        assert np.all(np.diff(got_rows) > 0)  # one ascending pass
        np.testing.assert_array_equal(got_rows, exp_rows)
        np.testing.assert_array_equal(got_cells, exp_cells)
    finally:
        backend_arrays.clear_index_memo()


def test_index_memo_lru_by_bytes():
    memo = _IndexMemo(max_bytes=1)
    entries = [
        _IndexEntry(np.arange(10), np.arange(10), np.zeros(10), 10) for _ in "ab"
    ]
    memo.put("a", entries[0])
    assert len(memo) == 1 and "a" in memo  # the most recent entry is always kept
    memo.put("b", entries[1])
    assert len(memo) == 1 and "b" in memo and "a" not in memo
    assert memo.stats["evictions"] == 1
    memo = _IndexMemo(max_bytes=entries[0].nbytes * 2)
    memo.put("a", entries[0])
    memo.put("b", entries[1])
    assert memo.get("a") is entries[0]  # now the most recent
    memo.put("c", _IndexEntry(np.arange(10), np.arange(10), np.zeros(10), 10))
    assert "a" in memo and "b" not in memo and "c" in memo
    assert memo.nbytes == 2 * entries[0].nbytes
    memo.clear()
    assert len(memo) == 0 and memo.nbytes == 0


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs os.fork")
@pytest.mark.filterwarnings("ignore:This process .* is multi-threaded")
def test_index_memo_is_reset_in_a_fork_child():
    """A fork child starts with an empty memo and a lock it can take, also
    when another thread of the parent held the memo lock at the fork."""
    index, _ = _synthetic_index([0, 1], [0, 0], (2, 1))
    try:
        assert len(INDEX_MEMO) >= 1
        with INDEX_MEMO._lock:
            pid = os.fork()
            if pid == 0:  # child
                ok = len(INDEX_MEMO) == 0 and INDEX_MEMO._lock.acquire(timeout=5)
                os._exit(0 if ok else 1)
        deadline = time.monotonic() + 60
        while not (done := os.waitpid(pid, os.WNOHANG))[0]:
            if time.monotonic() > deadline:  # (deadlocked: killed)
                os.kill(pid, 9)
                done = os.waitpid(pid, 0)
                break
            time.sleep(0.05)
        assert os.waitstatus_to_exitcode(done[1]) == 0
        assert index.memo_key() in INDEX_MEMO  # the parent keeps its memo
    finally:
        backend_arrays.clear_index_memo()


# --- MSv2MainColumnArray on generated MSs -------------------------------------


@contextlib.contextmanager
def built_partition(msname, idx=0, scheme=(), **kw):
    """build_partition (MAIN columns deferred) of partition ``idx``."""
    partitions, runs = create_partitions_with_main_rows(msname, list(scheme))
    with conversion.build_partition(
        msname,
        partitions[idx],
        main_row_runs=runs[idx],
        defer_main_columns=True,
        **kw,
    ) as built:
        yield built


def lazy_and_reference(msname, idx=0, scheme=(), **kw):
    """
    The lazy data variables of a partition (MSv2MainColumnArray / OnesArray,
    reversed along frequency as the converter does) and the values the
    converter reads for them (read_deferred_variables), by name.
    """
    with built_partition(msname, idx, scheme, **kw) as built:
        index = PartitionIndex.seed(os.path.abspath(msname), built)
        xds = built.ms_xdt.to_dataset(inherit=False)
        lazy = {}
        for name, spec in built.deferred.items():
            if name not in xds.data_vars:
                continue
            var = xds.variables[name]
            if spec.col is None:
                array = OnesArray(var.shape)
            else:
                array = MSv2MainColumnArray.from_spec(
                    index, spec, var.shape, var.dtype, node="node"
                )
            variable = lazy_variable(array, var.dims)
            if (
                built.reverse_frequency
                and "frequency" in var.dims
                and not spec.frequency_constant
            ):
                variable = variable.isel(frequency=slice(None, None, -1))
            lazy[name] = variable
        reference = xds.copy()
        stream_write.read_deferred_variables(
            reference,
            built.deferred,
            built.main_rows,
            built.tidxs,
            built.bidxs,
            built.reverse_frequency,
        )
    return lazy, {name: reference[name].variable for name in lazy}, index


def assert_same_values(got: np.ndarray, expected: np.ndarray, where=""):
    assert got.dtype == expected.dtype, where
    assert got.shape == expected.shape, where
    assert got.tobytes() == expected.tobytes(), where


def _selections(variable):
    nt = variable.sizes["time"]
    bdim = variable.dims[1]
    nb = variable.sizes[bdim]
    sels = [
        {},
        {"time": slice(1, None, 3)},
        {"time": nt // 2},
        {"time": slice(nt - 1, None, -2)},
        {bdim: [nb - 1, 0, nb // 2]},
        {"time": slice(2, 7), bdim: slice(1, None, 2)},
    ]
    if "frequency" in variable.dims:
        sels += [{"frequency": slice(1, None)}, {"frequency": 3, "time": [4, 1]}]
    return sels


# (partition index: DDIs 2 and 3 have the decreasing frequencies of SPW 1)
VALUE_CASES = [
    ("dense", 0),
    ("dense", 2),
    ("sparse_dup", 3),
    ("baseline_major", 1),
    ("time_descending", 2),
    ("shuffled", 0),
    ("wsp_partial", 3),  # WEIGHT from the WEIGHT column (tiled along frequency)
    ("no_weight", 0),  # WEIGHT=1
    ("rich", 5),  # WEIGHT_SPECTRUM, MODEL_DATA (StandardStMan), TSM data
]


@pytest.mark.parametrize("variant, idx", VALUE_CASES)
def test_values_equal_the_converter_reads(backend_ms, variant, idx):
    """
    Every lazy data variable (whole, and for steps, reversed steps, scalars,
    baseline lists, frequency slices on reversed SPWs) equals the values the
    converter reads into its MSv4, bit for bit, also through dask chunks.
    """
    msname = backend_ms(variant)
    lazy, reference, _ = lazy_and_reference(msname, idx)
    expected_names = {"VISIBILITY", "FLAG", "WEIGHT", "UVW", "TIME_CENTROID"}
    assert expected_names <= set(lazy)
    for name, variable in lazy.items():
        ref = reference[name]
        for sel in _selections(variable):
            assert_same_values(
                variable.isel(sel).values, ref.isel(sel).values, f"{name} {sel}"
            )
        chunked = variable.chunk({"time": 4})
        with dask.config.set(scheduler="threads"):
            assert_same_values(chunked.values, ref.values, f"{name} chunked")
    if variant == "sparse_dup":  # padded cells: NaN and FLAG=False
        flag, vis = reference["FLAG"].values, reference["VISIBILITY"].values
        assert np.isnan(vis).any() and not flag[np.isnan(vis)].any()
    if variant == "no_weight":
        assert isinstance(lazy["WEIGHT"]._data.array, OnesArray) and np.all(
            lazy["WEIGHT"].values == 1
        )
    if variant == "wsp_partial":
        assert lazy["WEIGHT"]._data.array.col == "WEIGHT"


def test_reads_hold_bounded_time_sub_blocks(backend_ms, monkeypatch):
    """A block is read in time sub-blocks of whole cells of at most
    SUB_BLOCK_BYTES (at least one time), with the same values."""
    lazy, reference, _ = lazy_and_reference(backend_ms("shuffled"), 0)
    monkeypatch.setattr(backend_arrays, "SUB_BLOCK_BYTES", 3 * 10 * 16 * 2 * 8 * 2)
    shapes = []
    read_grid = backend_arrays.read_grid

    def spy(table, col, plan, shape, *args, **kwargs):
        shapes.append(shape)
        return read_grid(table, col, plan, shape, *args, **kwargs)

    monkeypatch.setattr(backend_arrays, "read_grid", spy)
    for name in ("VISIBILITY", "WEIGHT", "TIME_CENTROID"):
        shapes.clear()
        assert_same_values(lazy[name].values, reference[name].values, name)
        array = lazy[name]._data.array
        bound = max(1, backend_arrays.SUB_BLOCK_BYTES // (10 * array._cell_bytes()))
        assert len(shapes) == -(-30 // bound) and max(s[0] for s in shapes) <= bound
    shapes.clear()
    lazy["VISIBILITY"].isel(time=5, frequency=2).values  # noqa: B018
    assert [s[0] for s in shapes] == [1]


def test_seeded_index_is_used_and_rebuilt_equal(backend_ms):
    """The build seeds the memo (no rebuild when reading in the same
    process); with an empty memo the index is rebuilt from the MS with the
    same values, once."""
    backend_arrays.clear_index_memo()
    lazy, reference, index = lazy_and_reference(backend_ms("sparse_dup"), 1)
    seeded = INDEX_MEMO.get(index.memo_key())
    assert_same_values(lazy["VISIBILITY"].values, reference["VISIBILITY"].values)
    assert INDEX_MEMO.stats["rebuilds"] == 0
    backend_arrays.clear_index_memo()
    for name in ("VISIBILITY", "FLAG"):
        assert_same_values(lazy[name].values, reference[name].values, name)
    assert INDEX_MEMO.stats["rebuilds"] == 1
    rebuilt = INDEX_MEMO.get(index.memo_key())
    for field in ("rows", "tidxs", "bidxs", "time_bounds"):
        np.testing.assert_array_equal(getattr(rebuilt, field), getattr(seeded, field))
    # the key has no per-open part: building the partition again gives it
    _, _, again = lazy_and_reference(backend_ms("sparse_dup"), 1)
    assert again.memo_key() == index.memo_key() and again.token == index.token


def test_index_is_rebuilt_once_by_concurrent_reads(backend_ms):
    """Threads reading blocks of one partition with an empty memo (as dask
    threads in another process) rebuild its index once."""
    import threading

    lazy, reference, _ = lazy_and_reference(backend_ms("shuffled"), 1)
    backend_arrays.clear_index_memo()
    results, errors = {}, []
    barrier = threading.Barrier(8)

    def read(t):
        try:
            barrier.wait()
            results[t] = lazy["VISIBILITY"].isel(time=t).values
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=read, args=(t,)) for t in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    assert INDEX_MEMO.stats["rebuilds"] == 1
    for t, values in results.items():
        assert_same_values(values, reference["VISIBILITY"].isel(time=t).values)


@pytest.mark.parametrize("pickler", [pickle, cloudpickle])
def test_arrays_pickle_small_and_read_in_another_process_state(backend_ms, pickler):
    """Pickled arrays carry no per-row array or table: unpickled with an
    empty memo (as in another process) they rebuild the index and read the
    same values."""
    lazy, reference, index = lazy_and_reference(backend_ms("baseline_major"), 0)
    for name in ("VISIBILITY", "WEIGHT", "TIME_CENTROID", "UVW"):
        array = lazy[name]._data.array
        payload = pickler.dumps(array)
        assert len(payload) < 4096, (name, len(payload))
        backend_arrays.clear_index_memo()
        copy = pickle.loads(payload)
        variable = lazy_variable(copy, lazy[name].dims)
        assert_same_values(variable.values, lazy[name].values, name)
    ones = OnesArray((3, 2, 4, 2))
    assert pickle.loads(pickle.dumps(ones)).shape == (3, 2, 4, 2)
    assert pickle.loads(pickle.dumps(index)).memo_key() == index.memo_key()


def test_pickled_size_of_a_large_time_ordered_partition():
    """A time-ordered partition of 10,000 rows (one run) pickles to < 4 kB."""
    index = PartitionIndex("/x.ms", [0], [10000], 20000, (1000, 10), "t" * 32)
    transform = conversion.functools.partial(
        conversion.postprocess_main_column,
        "WEIGHT",
        parallel_mode="none",
        main_sizes={"frequency": 64},
        main_chunksize=None,
    )
    array = MSv2MainColumnArray(
        index,
        "WEIGHT",
        "WEIGHT",
        (1000, 10, 64, 4),
        np.float32,
        np.float32,
        (4,),
        transform,
        True,
        "x_0",
    )
    assert len(pickle.dumps(array)) < 4096
    assert len(cloudpickle.dumps(array)) < 4096


def _copy_ms(backend_ms, variant, tmp_path):
    copy = str(tmp_path / f"{variant}_copy.ms")
    shutil.copytree(backend_ms(variant), copy)
    return copy


def test_rows_added_after_the_open_raise_changed(backend_ms, tmp_path):
    from casacore import tables

    msname = _copy_ms(backend_ms, "dense", tmp_path)
    lazy, reference, _ = lazy_and_reference(msname, 0)
    with tables.table(msname, readonly=False, ack=False) as main_tb:
        main_tb.addrows(3)
    with pytest.raises(MSv2ChangedError, match="rows"):
        lazy["VISIBILITY"].values  # noqa: B018
    backend_arrays.clear_index_memo()  # rebuild: the same check
    with pytest.raises(MSv2ChangedError, match="rows"):
        lazy["FLAG"].values  # noqa: B018


def test_rebuilt_index_with_other_baselines_raises_changed(backend_ms, tmp_path):
    from casacore import tables

    msname = _copy_ms(backend_ms, "dense", tmp_path)
    lazy, _, _ = lazy_and_reference(msname, 0)
    with tables.table(msname, readonly=False, ack=False) as main_tb:
        ant2 = main_tb.getcol("ANTENNA2")
        ant2[:300] = 4 - (ant2[:300] % 2)  # other baselines in partition 0
        main_tb.putcol("ANTENNA2", ant2)
    backend_arrays.clear_index_memo()
    with pytest.raises(MSv2ChangedError, match="changed since it was opened"):
        lazy["VISIBILITY"].values  # noqa: B018


def test_rebuilt_index_with_other_times_raises_changed(backend_ms, tmp_path):
    """The same grid shape, other times: the token of the grid tells."""
    from casacore import tables

    msname = _copy_ms(backend_ms, "dense", tmp_path)
    lazy, _, _ = lazy_and_reference(msname, 0)
    with tables.table(msname, readonly=False, ack=False) as main_tb:
        times = main_tb.getcol("TIME")
        times[:300] += 0.5  # (partition 0: its times shifted)
        main_tb.putcol("TIME", times)
    backend_arrays.clear_index_memo()
    with pytest.raises(MSv2ChangedError, match="times or baselines"):
        lazy["VISIBILITY"].values  # noqa: B018


def test_read_errors_name_the_block_and_the_remedy(backend_ms, monkeypatch):
    lazy, _, _ = lazy_and_reference(backend_ms("dense"), 0)

    def failing(*args, **kwargs):
        raise RuntimeError("simulated casacore failure")

    monkeypatch.setattr(backend_arrays, "read_grid", failing)
    array = lazy["VISIBILITY"]._data.array
    with pytest.raises(MSv2ReadError, match="simulated casacore failure") as raised:
        lazy["VISIBILITY"].isel(time=slice(2, 5)).values  # noqa: B018
    message = str(raised.value)
    assert "column DATA" in message and "VISIBILITY" in message and "node" in message
    assert "2:5" in message and "drop_variables" not in message
    array.verified = False
    with pytest.raises(MSv2ReadError, match=r"drop_variables=\['VISIBILITY'\]"):
        lazy["VISIBILITY"].values  # noqa: B018


class GetcolOnlyTable:
    """A python-casacore table with the read API of the casatools shim (no
    getcolnp, getcolslicenp or selectrows)."""

    def __init__(self, table):
        self._table = table

    def __getattr__(self, name):
        if name in ("getcolnp", "getcolslicenp", "selectrows"):
            raise AttributeError(name)
        return getattr(self._table, name)


def test_reads_without_in_place_reads_and_with_the_casatools_lock(
    backend_ms, monkeypatch
):
    """With the read API of the casatools shim (getcol only) the values are
    the same, and every read and index rebuild holds casatools_serialized."""
    lazy, reference, _ = lazy_and_reference(backend_ms("sparse_dup"), 0)
    open_table_ro = backend_arrays.open_table_ro
    entered = []

    @contextlib.contextmanager
    def getcol_only(path):
        with open_table_ro(path) as table:
            yield GetcolOnlyTable(table)

    @contextlib.contextmanager
    def serialized():
        entered.append(True)
        yield

    monkeypatch.setattr(backend_arrays, "open_table_ro", getcol_only)
    monkeypatch.setattr(backend_arrays, "casatools_serialized", serialized)
    backend_arrays.clear_index_memo()
    for name in ("VISIBILITY", "FLAG", "WEIGHT", "UVW"):
        assert_same_values(lazy[name].values, reference[name].values, name)
    assert len(entered) == 4 + 1  # four reads and one rebuild


def test_index_token():
    utime, ant1, ant2 = np.arange(3.0), np.array([0, 0]), np.array([1, 2])
    token = index_token(utime, ant1, ant2, (3, 2))
    assert token == index_token(utime, ant1.astype(np.int32), ant2, (3, 2))
    assert token != index_token(utime + 1, ant1, ant2, (3, 2))
    assert token != index_token(utime, ant1, ant2[::-1], (3, 2))
    assert len(token) == 32
