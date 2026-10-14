"""Tests of the lazy arrays of the MSv2 xarray backend (backend_arrays.py)."""

import contextlib
import copy
import os
import pickle
import re
import shutil
import time

import cloudpickle
import dask
import numpy as np
import pytest
import xarray as xr

from xradio.measurement_set._utils._msv2 import backend_arrays, conversion, stream_write
from xradio.measurement_set._utils._msv2._tables.read_rows import ColumnStorage
from xradio.measurement_set._utils._msv2.backend_arrays import (
    INDEX_MEMO,
    MSv2BackendArray,
    MSv2MainColumnArray,
    OnesArray,
    PartitionIndex,
    _IndexEntry,
    _IndexMemo,
    apply_outer_post,
    index_token,
    normalize_outer_key,
)
from xradio.measurement_set._utils._msv2.backend_errors import (
    MSv2ChangedError,
    MSv2ReadError,
)
from xradio.measurement_set._utils._msv2.partition_queries import (
    create_partitions_with_main_rows,
)

# --- the base class (cases adapted from the ASDM backend's tests) ------------

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
    "list_repeated": {"time": [3, 3, 0]},
    "lists_outer": {"time": [5, 1], "frequency": [2, 7, 2], "polarization": [1]},
}


def outer_numpy(values, dims, isel):
    """numpy indexing of ``values`` with ``isel`` (lists as xarray's outer
    indexing: every dimension on its own)."""
    for axis in reversed(range(len(dims))):
        key = isel.get(dims[axis], slice(None))
        values = np.take(values, np.arange(values.shape[axis])[key], axis=axis)
    return values


@pytest.mark.parametrize(
    "key, shape, expected_selections, expected_post, result_shape",
    [
        (
            (slice(None), 3, slice(2, 5), -1),
            SHAPE,
            ([0, 1, 2, 3, 4, 5, 6], [3], [2, 3, 4], [1]),
            [slice(None), 0, slice(None), 0],
            (7, 3),
        ),
        (
            (slice(None, None, 2), slice(6, 1, -3)),
            (7, 8),
            ([0, 2, 4, 6], [3, 6]),
            [slice(None), slice(None, None, -1)],
            (4, 2),
        ),
        (
            (slice(3, 3),),
            (7, 8),
            ([], [0, 1, 2, 3, 4, 5, 6, 7]),
            [slice(None), slice(None)],
            (0, 8),
        ),
        (
            (np.int64(2), slice(None, None, -1)),
            (7, 8),
            ([2], [0, 1, 2, 3, 4, 5, 6, 7]),
            [0, slice(None, None, -1)],
            (8,),
        ),
        (
            (np.array([6, 0, 3, 0]), [-1, 2]),
            (7, 8),
            ([0, 3, 6], [2, 7]),
            [[2, 0, 1, 0], [1, 0]],
            (4, 2),
        ),
        (
            (np.array([1, 4, 5]),),
            (7, 8),
            ([1, 4, 5], [0, 1, 2, 3, 4, 5, 6, 7]),
            [slice(None), slice(None)],
            (3, 8),
        ),
    ],
)
def test_normalize_outer_key(
    key, shape, expected_selections, expected_post, result_shape
):
    selections, post, got_shape = normalize_outer_key(key, shape)
    assert [s.tolist() for s in selections] == list(expected_selections)
    assert all(s.dtype == np.int64 for s in selections)
    assert [p.tolist() if isinstance(p, np.ndarray) else p for p in post] == (
        expected_post
    )
    assert got_shape == result_shape
    reference = reference_values(shape)
    if 0 not in result_shape:
        read = reference[np.ix_(*selections)]
        expected = reference
        full_key = tuple(key) + (slice(None),) * (len(shape) - len(key))
        for axis in reversed(range(len(shape))):  # outer, from the last axis
            expected = np.take(expected, np.arange(shape[axis])[full_key[axis]], axis)
        np.testing.assert_array_equal(apply_outer_post(read, post), expected)


@pytest.mark.parametrize(
    "key, error",
    [
        ((7,), IndexError),
        ((-8,), IndexError),
        ((0, 0, 0), IndexError),
        ((np.array([0, 7]),), IndexError),
        ((np.array([0.5]),), TypeError),
        ((np.array([[0, 1]]),), TypeError),
        ((None,), TypeError),
    ],
)
def test_normalize_outer_key_errors(key, error):
    with pytest.raises(error):
        normalize_outer_key(key, (7, 8))


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
    expected = outer_numpy(reference, DIMS, isel)
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


@pytest.mark.parametrize("layout", list(SELECT_LAYOUTS))
@pytest.mark.parametrize(
    "times, baselines",
    [([0, 3, 7], [0, 1, 2, 3, 4]), ([1, 2, 6], [4, 0]), ([5], [1, 3]), ([2, 4], [2])],
)
def test_partition_index_select_outer_matches_brute_force(layout, times, baselines):
    nt, nb = 8, 5
    tidxs, bidxs = SELECT_LAYOUTS[layout](nt, nb)
    index, rows = _synthetic_index(tidxs, bidxs, (nt, nb))
    times, baselines = np.asarray(times), np.sort(np.asarray(baselines))
    try:
        got_rows, got_cells, positions = index.select_outer(times, baselines)
        keep = np.isin(tidxs, times) & np.isin(bidxs, baselines)
        exp_cells = np.searchsorted(times, tidxs[keep]) * baselines.size + (
            np.searchsorted(baselines, bidxs[keep])
        )
        assert np.all(np.diff(got_rows) > 0)  # one ascending pass
        np.testing.assert_array_equal(got_rows, rows[keep])
        np.testing.assert_array_equal(got_cells, exp_cells)
        np.testing.assert_array_equal(rows[positions], got_rows)
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
    converter reads for them (read_deferred_variables), by name. The arrays
    know the channel tiling of their columns, as when the MS is opened.
    """
    keys = backend_arrays.keys_token(os.path.abspath(msname))
    partitions, _ = create_partitions_with_main_rows(msname, list(scheme))
    grouping = backend_arrays.partition_grouping(partitions[idx], tuple(scheme))
    with built_partition(msname, idx, scheme, **kw) as built:
        index = PartitionIndex.seed(
            os.path.abspath(msname), built, keys, grouping, tuple(scheme)
        )
        xds = built.ms_xdt.to_dataset(inherit=False)
        lazy = {}
        for name, spec in built.deferred.items():
            if name not in xds.data_vars:
                continue
            var = xds.variables[name]
            if spec.col is None:
                array = OnesArray(var.shape, index=index)
            else:
                array = MSv2MainColumnArray.from_spec(
                    index,
                    spec,
                    var.shape,
                    var.dtype,
                    node="node",
                    storage=built.main_rows.column_storage(spec.col),
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


def test_selections_read_only_their_times_and_baselines(backend_ms, monkeypatch):
    """Lists and steps along time and baseline (array_backend "xarray") read
    only the rows of the selected cells: a read holds its result plus one
    time sub-block, never the bounding block."""
    lazy, reference, _ = lazy_and_reference(backend_ms("shuffled"), 0)
    reads = []
    read_grid = backend_arrays.read_grid

    def spy(table, col, plan, shape, *args, **kwargs):
        reads.append((shape[:2], plan.rows.size))
        return read_grid(table, col, plan, shape, *args, **kwargs)

    monkeypatch.setattr(backend_arrays, "read_grid", spy)
    cases = [
        ({"time": [0, -1]}, [((2, 10), 20)]),
        ({"time": slice(None, None, 10)}, [((3, 10), 30)]),
        ({"time": [29, 0, 29], "baseline_id": [7, 2]}, [((2, 2), 4)]),
        ({"time": slice(28, 1, -13), "baseline_id": slice(1, None, 4)}, [((3, 3), 9)]),
        ({"time": 4, "baseline_id": [9]}, [((1, 1), 1)]),
    ]
    for isel, expected in cases:
        reads.clear()
        for name in ("VISIBILITY", "FLAG", "UVW"):
            assert_same_values(
                lazy[name].isel(isel).values,
                reference[name].isel(isel).values,
                f"{name} {isel}",
            )
        assert reads == expected * 3, isel
    # a list of times in several sub-blocks: each sub-block only the selected
    monkeypatch.setattr(backend_arrays, "SUB_BLOCK_BYTES", 1)
    reads.clear()
    isel = {"time": [3, 9, 20, 21]}
    assert_same_values(
        lazy["VISIBILITY"].isel(isel).values, reference["VISIBILITY"].isel(isel).values
    )
    assert reads == [((1, 10), 10)] * 4


# --- channel-sliced reads (columns stored in tiles of fewer channels) ---------


def _cube(cell, tile, bucket=None, with_cell=True):
    """A hypercube description (Fortran order) as getdminfo gives it."""
    cube = {"CubeShape": list(cell) + [100], "TileShape": list(tile)}
    if with_cell:
        cube["CellShape"] = list(cell)
    if bucket is not None:
        cube["BucketSize"] = bucket
    return cube


@pytest.mark.parametrize(
    "storage, cell_shape, dtype, expected",
    [
        # (T, bytes of the tiles of a band of T channels: pol tiles x tile)
        (ColumnStorage(True, "TiledShapeStMan", 0, (_cube([4, 64], [4, 7, 146], 32704),)), (64, 4), np.complex64, (7, 32704)),
        (ColumnStorage(True, "TiledColumnStMan", 0, (_cube([4, 64], [2, 8, 16]),)), (64, 4), np.complex64, (8, 2 * 2 * 8 * 16 * 8)),
        (ColumnStorage(True, "TiledShapeStMan", 0, (_cube([4, 64], [4, 8, 16]),)), (64, 4), np.bool_, (8, 4 * 8 * 16 // 8)),
        (ColumnStorage(True, "TiledDataStMan", 0, (_cube([4, 64], [4, 8, 16], with_cell=False),)), (64, 4), np.float32, (8, 4 * 8 * 16 * 4)),
        # hypercubes of other cell shapes are not looked at
        (ColumnStorage(True, "TiledShapeStMan", 0, (_cube([4, 64], [4, 8, 16], 10), _cube([4, 32], [4, 32, 16], 20))), (64, 4), np.complex64, (8, 10)),
        # several hypercubes of the cell shape: one channel tiling, the largest tiles
        (ColumnStorage(True, "TiledShapeStMan", 0, (_cube([4, 64], [4, 8, 16], 10), _cube([4, 64], [4, 8, 32], 20))), (64, 4), np.complex64, (8, 20)),
        # not one channel tiling, or tiles of all the channels: whole cells
        (ColumnStorage(True, "TiledShapeStMan", 0, (_cube([4, 64], [4, 8, 16], 10), _cube([4, 64], [4, 16, 16], 20))), (64, 4), np.complex64, (0, 0)),
        (ColumnStorage(True, "TiledShapeStMan", 0, (_cube([4, 64], [4, 64, 16]),)), (64, 4), np.complex64, (0, 0)),
        (ColumnStorage(True, "TiledShapeStMan", 0, (_cube([4, 64], [4, 128, 16]),)), (64, 4), np.complex64, (0, 0)),
        (ColumnStorage(True, "TiledShapeStMan", 0, (_cube([4, 32], [4, 8, 16]),)), (64, 4), np.complex64, (0, 0)),
        (ColumnStorage(True, "TiledShapeStMan", 0, ()), (64, 4), np.complex64, (0, 0)),
        (ColumnStorage(True, "TiledShapeStMan", 0, (_cube([4, 64], [4, 8]),)), (64, 4), np.complex64, (0, 0)),
        # other storage, a reference table, an error, unknown, other cells
        (ColumnStorage(True, "StandardStMan", 0, ()), (64, 4), np.complex64, (0, 0)),
        (ColumnStorage(True, "TiledCellStMan", 0, (_cube([4, 64], [4, 8, 16]),)), (64, 4), np.complex64, (0, 0)),
        (ColumnStorage(False, "TiledShapeStMan", 0, (_cube([4, 64], [4, 8, 16]),)), (64, 4), np.complex64, (0, 0)),
        (ColumnStorage(False, error="RuntimeError: no"), (64, 4), np.complex64, (0, 0)),
        (None, (64, 4), np.complex64, (0, 0)),
        (ColumnStorage(True, "TiledColumnStMan", 0, (_cube([3], [3, 16]),)), (3,), np.float64, (0, 0)),
    ],
)  # fmt: skip
def test_channel_tiling(storage, cell_shape, dtype, expected):
    assert backend_arrays.channel_tiling(storage, cell_shape, dtype) == expected


def test_channel_tiling_of_generated_columns(backend_ms):
    """The channel tiling of the columns of the "narrow_tiles" MS (two
    hypercubes per column, one per cell shape), as the arrays get it when
    the MS is opened; whole cells for MODEL_DATA (tiles of all channels) and
    for the columns of the other MSs."""
    from casacore import tables

    from xradio.measurement_set._utils._msv2._tables.read_rows import column_storage

    expected = {
        "DATA": (3, 336),  # 2 x 3 x 7 x 8 bytes
        "CORRECTED_DATA": (5, 880),
        "FLAG": (4, 13),  # booleans: 2 x 4 x 13 bits
        "WEIGHT_SPECTRUM": (4, 2 * 112),  # tiles of 1 polarization
        "MODEL_DATA": (0, 0),
    }
    with tables.table(backend_ms("narrow_tiles"), ack=False) as main_tb:
        for col, tiling in expected.items():
            storage = column_storage(main_tb, col)
            assert len(storage.hypercubes) == 2, col
            assert storage.dm_name == f"TSM_{col}"
            dtype = main_tb.getcell(col, 0).dtype
            for cell_shape in ((16, 2), (24, 2)):
                found = backend_arrays.channel_tiling(storage, cell_shape, dtype)
                assert found == tiling, (col, cell_shape)
    for variant, col in (("dense", "DATA"), ("rich", "DATA"), ("rich", "FLAG")):
        with tables.table(backend_ms(variant), ack=False) as main_tb:
            storage = column_storage(main_tb, col)
            assert backend_arrays.channel_tiling(storage, (16, 2), np.complex64) == (
                0,
                0,
            )


def test_channel_sliced_columns_are_read_as_written():
    """The converter writes the values of CHANNEL_SLICED_COLUMNS as it reads
    them (postprocess_main_column is the identity for them), the condition
    for reading a range of their channels alone."""
    values = np.zeros((2, 3, 4, 2), dtype=np.complex64)
    for col in backend_arrays.CHANNEL_SLICED_COLUMNS:
        assert conversion.postprocess_main_column(col, values) is values
    for col in ("WEIGHT", "TIME_CENTROID"):
        assert col not in backend_arrays.CHANNEL_SLICED_COLUMNS


@pytest.mark.parametrize(
    "col, cell_shape, shape, tile_channels, expected",
    [
        ("DATA", (16, 2), (2, 3, 16, 2), 3, 3),
        ("FLAG", (16, 2), (2, 3, 16, 2), 15, 15),
        ("DATA", (16, 2), (2, 3, 16, 2), 16, 0),  # tiles of all channels
        ("WEIGHT", (2,), (2, 3, 16, 2), 3, 0),  # repeated along frequency
        ("UVW", (3,), (2, 3, 3), 1, 0),
        ("TIME_CENTROID", (), (2, 3), 1, 0),
        ("SIGMA_SPECTRUM", (16, 2), (2, 3, 16, 2), 3, 0),  # not a data variable
    ],
)
def test_channel_sliced_arrays(col, cell_shape, shape, tile_channels, expected):
    """Only arrays of CHANNEL_SLICED_COLUMNS with 2-D cells read slices of
    channels; the tiling and the data manager are kept in the pickled
    state."""
    index = PartitionIndex("/x.ms", [0], [6], 6, shape[:2], "t" * 32)
    array = MSv2MainColumnArray(
        index,
        col,
        col,
        shape,
        np.float32,
        np.float32,
        cell_shape,
        tile_channels=tile_channels,
        tile_band_bytes=100,
        tile_dm="TiledData",
    )
    assert array.tile_channels == expected
    assert array.tile_band_bytes == (100 if expected else 0)
    assert array.tile_dm == ("TiledData" if expected else "")
    restored = pickle.loads(pickle.dumps(array))
    assert (restored.tile_channels, restored.tile_band_bytes, restored.tile_dm) == (
        array.tile_channels,
        array.tile_band_bytes,
        array.tile_dm,
    )


def _channel_selections(n_chan):
    """Channel selections that start and end inside tiles, span several
    tiles, hold one channel (the first, a middle one, the last, in the last
    partial tile), lists, steps and reversed steps, and every channel."""
    return [
        {"frequency": slice(0, 1)},
        {"frequency": slice(4, 5)},
        {"frequency": slice(n_chan - 1, n_chan)},
        {"frequency": slice(2, 7)},
        {"frequency": slice(5, n_chan - 1)},
        {"frequency": slice(1, n_chan - 1)},
        {"frequency": slice(3, 13, 4)},
        {"frequency": slice(n_chan - 2, 1, -5)},
        {"frequency": [n_chan - 3, 4, 6, 4]},
        {"frequency": 7, "polarization": 1},
        {"frequency": slice(6, 9), "time": [5, 0, 31], "baseline_id": slice(2, 9, 3)},
        {"frequency": slice(None, None, -1)},
        {},
    ]


# (partitions of "narrow_tiles": 0 and 1 SPW 0, 16 channels; 2 and 3 SPW 1,
# 24 decreasing channels)
@pytest.mark.parametrize("idx", [0, 3])
def test_channel_sliced_reads_equal_the_converter_reads(backend_ms, monkeypatch, idx):
    """
    Channel-sliced reads (tiles of 3, 5 and 4 channels, of one or two
    polarizations, two hypercubes per column, a reversed SPW, padded and
    duplicated cells) give the values the converter reads, bit for bit, for
    channel ranges that straddle tiles, single channels, lists and steps,
    also through dask chunks; MODEL_DATA (tiles of all channels) and a
    selection of every channel are read in whole cells. The read channels
    are the selection's range rounded out to whole tiles.
    """
    lazy, reference, _ = lazy_and_reference(backend_ms("narrow_tiles"), idx)
    sliced = []
    read_rows_to_grid = backend_arrays.read_rows_to_grid

    def spy(table, col, plan, grid, *args, **kwargs):
        sliced.append((col, kwargs.get("chan")))
        return read_rows_to_grid(table, col, plan, grid, *args, **kwargs)

    monkeypatch.setattr(backend_arrays, "read_rows_to_grid", spy)
    tiles = {
        "VISIBILITY": 3,
        "VISIBILITY_CORRECTED": 5,
        "FLAG": 4,
        "WEIGHT": 4,
        "VISIBILITY_MODEL": 0,
    }
    vis = reference["VISIBILITY"].values
    assert np.isnan(vis).any()  # padded cells
    for name, width in tiles.items():
        variable = lazy[name]
        array = variable._data.array
        assert array.tile_channels == width, name
        n_chan = variable.sizes["frequency"]
        for sel in _channel_selections(n_chan):
            sliced.clear()
            got = variable.isel(sel).values
            assert_same_values(got, reference[name].isel(sel).values, f"{name} {sel}")
            channels = np.unique(
                np.arange(n_chan)[sel.get("frequency", slice(None))]
            ).reshape(-1)
            if idx >= 2:  # (reversed: the MS's channel order)
                channels = np.sort(n_chan - 1 - channels)
            c0 = channels[0] // width * width if width else 0
            c1 = min(n_chan, -(-(channels[-1] + 1) // width) * width) if width else 0
            if width and (c0, c1) != (0, n_chan):
                assert sliced and all(chan == slice(c0, c1) for _, chan in sliced), (
                    name,
                    sel,
                    sliced,
                )
            else:
                assert not sliced, (name, sel, sliced)
        chunked = variable.isel(frequency=slice(2, 9)).chunk({"time": 7})
        with dask.config.set(scheduler="threads"):
            assert_same_values(
                chunked.values,
                reference[name].isel(frequency=slice(2, 9)).values,
                f"{name} chunked",
            )


def test_channel_sliced_reads_bound_the_tile_cache(backend_ms, monkeypatch):
    """A channel-sliced read bounds the column's tile cache
    (setmaxcachesize, MiB) to twice the tiles of its bands for the rows of a
    tile, at least CHANNEL_SLICE_MIN_CACHE_MIB, and sets the maximum before
    (getdmprop) again when it ends; its time sub-blocks are sized on the
    read channels; with the casatools read API (no in-place reads) whole
    cells are read."""
    lazy, reference, _ = lazy_and_reference(backend_ms("narrow_tiles"), 0)
    open_table_ro = backend_arrays.open_table_ro
    calls = []

    class Recording(GetcolOnlyTable):
        hidden = ()

        def __getattr__(self, name):
            if name in self.hidden:
                raise AttributeError(name)
            if name == "setmaxcachesize":
                return lambda col, size: calls.append((col, size))
            return getattr(self._table, name)

    @contextlib.contextmanager
    def recording(path):
        with open_table_ro(path) as table:
            yield Recording(table)

    monkeypatch.setattr(backend_arrays, "open_table_ro", recording)
    variable, array = lazy["VISIBILITY"], lazy["VISIBILITY"]._data.array
    sel = {"frequency": slice(2, 7)}  # tiles [0, 3) and [3, 6) and [6, 9)
    assert_same_values(
        variable.isel(sel).values, reference["VISIBILITY"].isel(sel).values
    )
    # (the maximum before: none, 0)
    assert calls == [("DATA", backend_arrays.CHANNEL_SLICE_MIN_CACHE_MIB), ("DATA", 0)]
    array.tile_band_bytes = 7 * 2**20  # 3 bands: 2 x 3 x 7 MiB
    calls.clear()
    variable.isel(sel).values  # noqa: B018
    assert calls == [("DATA", 42), ("DATA", 0)]
    assert array._channel_cache_mib(0, 3) == 16 and array._channel_cache_mib(0, 9) == 42
    assert len(backend_arrays.TILE_CACHE_BOUNDS) == 0
    # sub-blocks of the read channels (9 of 16): 3 times of all baselines each,
    # where whole cells would give 1
    assert (
        array._cell_bytes(9) == 9 * 2 * 8 * 2 and array._cell_bytes() == 16 * 2 * 8 * 2
    )
    n_times, n_baselines = variable.sizes["time"], variable.sizes["baseline_id"]
    monkeypatch.setattr(
        backend_arrays, "SUB_BLOCK_BYTES", 3 * n_baselines * array._cell_bytes(9)
    )
    steps = []
    select_outer = backend_arrays.PartitionIndex.select_outer

    def spy_select(self, times, *args, **kwargs):
        steps.append(times.size)
        return select_outer(self, times, *args, **kwargs)

    monkeypatch.setattr(backend_arrays.PartitionIndex, "select_outer", spy_select)
    assert_same_values(
        variable.isel(sel).values, reference["VISIBILITY"].isel(sel).values
    )
    assert steps == [3] * (n_times // 3) + [n_times % 3] * bool(n_times % 3)
    monkeypatch.setattr(backend_arrays.PartitionIndex, "select_outer", select_outer)
    monkeypatch.setattr(backend_arrays, "SUB_BLOCK_BYTES", 128 * 2**20)
    # without in-place reads (the casatools shim): whole cells
    Recording.hidden = ("getcolnp", "getcolslicenp", "selectrows")
    calls.clear()
    sliced = []
    read_rows_to_grid = backend_arrays.read_rows_to_grid

    def spy(table, col, plan, grid, *args, **kwargs):
        sliced.append(kwargs.get("chan"))
        return read_rows_to_grid(table, col, plan, grid, *args, **kwargs)

    monkeypatch.setattr(backend_arrays, "read_rows_to_grid", spy)
    for name in ("VISIBILITY", "FLAG", "WEIGHT"):
        assert_same_values(
            lazy[name].isel(sel).values, reference[name].isel(sel).values, name
        )
    assert calls == [] and sliced == []


def test_channel_sliced_reads_set_the_tile_cache_bound_back(backend_ms, monkeypatch):
    """
    A channel-sliced read bounds the column's tile cache only while it
    reads. python-casacore shares one table object per table in a process:
    a handle of MAIN held open (here with a maximum of its own, 3 MiB) sees
    the bound during the read and its own maximum again after it. Reads of
    a column under way at once (nested here) keep the largest of their
    bounds until the last one ends, which sets the maximum before again.
    """
    from xradio.measurement_set._utils._msv2._tables.table_query import (
        open_table_ro,
    )

    msname = backend_ms("narrow_tiles")
    lazy, reference, _ = lazy_and_reference(msname, 0)
    seen = []
    read_rows_to_grid = backend_arrays.read_rows_to_grid

    def spy(table, col, plan, grid, *args, **kwargs):
        seen.append(table.getdmprop(col)["MaxCacheSize"])
        return read_rows_to_grid(table, col, plan, grid, *args, **kwargs)

    monkeypatch.setattr(backend_arrays, "read_rows_to_grid", spy)
    bounds = backend_arrays.TILE_CACHE_BOUNDS
    sel = {"frequency": slice(2, 7)}
    with open_table_ro(msname) as handle:
        handle.setmaxcachesize("DATA", 3)
        assert_same_values(
            lazy["VISIBILITY"].isel(sel).values,
            reference["VISIBILITY"].isel(sel).values,
        )
        assert seen and set(seen) == {backend_arrays.CHANNEL_SLICE_MIN_CACHE_MIB}
        assert handle.getdmprop("DATA")["MaxCacheSize"] == 3
        assert len(bounds) == 0
        with bounds.bounded(handle, msname, "DATA", 16):
            with bounds.bounded(handle, msname, "DATA", 42):
                assert handle.getdmprop("DATA")["MaxCacheSize"] == 42
            assert handle.getdmprop("DATA")["MaxCacheSize"] == 16
            with pytest.raises(ValueError):
                with bounds.bounded(handle, msname, "DATA", 8):
                    assert handle.getdmprop("DATA")["MaxCacheSize"] == 16
                    raise ValueError("a read that fails")
            assert handle.getdmprop("DATA")["MaxCacheSize"] == 16
        assert handle.getdmprop("DATA")["MaxCacheSize"] == 3
        assert len(bounds) == 0
        handle.setmaxcachesize("DATA", 0)


# The data manager of the columns of shared_dm_narrow_tiles, its columns and
# its tiles (Fortran order: pol, chan, rows)
SHARED_DM = "TiledShared"
SHARED_DM_COLUMNS = ("DATA", "CORRECTED_DATA", "FLAG")
SHARED_DM_TILE_SHAPE = (2, 3, 7)


@pytest.fixture(scope="module")
def shared_dm_narrow_tiles(backend_ms, tmp_path_factory):
    """A deep copy of the "dense" MS whose DATA, CORRECTED_DATA and FLAG are
    in one TiledShapeStMan (SHARED_DM, as DATA and FLAG of some MSs of the
    test corpus), in tiles of 3 of the 16 channels: one tile cache for the
    three columns."""
    from casacore import tables

    base = tmp_path_factory.mktemp("shared_dm")
    msname = str(base / "shared_dm.ms")
    with tables.table(backend_ms("dense"), ack=False) as main_tb:
        dminfo = {}
        for info in main_tb.getdminfo().values():
            columns = [c for c in info["COLUMNS"] if c not in SHARED_DM_COLUMNS]
            if columns:
                dminfo[f"*{len(dminfo) + 1}"] = dict(info, COLUMNS=columns)
        dminfo[f"*{len(dminfo) + 1}"] = {
            "TYPE": "TiledShapeStMan",
            "NAME": SHARED_DM,
            "SPEC": {"DEFAULTTILESHAPE": np.array(SHARED_DM_TILE_SHAPE, np.int32)},
            "COLUMNS": list(SHARED_DM_COLUMNS),
        }
        main_tb.copy(msname, deep=True, valuecopy=True, dminfo=dminfo).close()
    yield msname
    shutil.rmtree(base, ignore_errors=True)


def test_channel_sliced_reads_bound_the_tile_cache_of_the_data_manager(
    shared_dm_narrow_tiles, monkeypatch
):
    """
    casacore keeps one tile cache per tiled data manager, which
    setmaxcachesize of any of its columns bounds. Reads of two columns of one
    data manager under way at once (DATA and FLAG, here interleaved: the
    first ends while the second still reads) keep the largest of their
    bounds until the last one ends, whatever its column, which sets the
    maximum before (here 3 MiB, of a handle held open) again: the bound is
    never lifted while one of them reads, and none is left after them. Also
    with dask threads reading VISIBILITY and FLAG one channel at a time.
    """
    from casacore import tables

    from xradio.measurement_set._utils._msv2._tables.table_query import (
        open_table_ro,
    )

    msname = shared_dm_narrow_tiles
    with tables.table(msname, ack=False) as main_tb:
        for col in SHARED_DM_COLUMNS:
            assert main_tb.getdminfo(col)["NAME"] == SHARED_DM
    lazy, reference, _ = lazy_and_reference(msname, 0)
    for name in ("VISIBILITY", "VISIBILITY_CORRECTED", "FLAG"):
        array = lazy[name]._data.array
        assert (array.tile_channels, array.tile_dm) == (3, SHARED_DM), name
        sel = {"frequency": slice(4, 9)}
        assert_same_values(
            lazy[name].isel(sel).values, reference[name].isel(sel).values, name
        )

    bounds = backend_arrays.TILE_CACHE_BOUNDS
    with open_table_ro(msname) as handle:
        handle.setmaxcachesize("DATA", 3)
        assert handle.getdmprop("FLAG")["MaxCacheSize"] == 3  # (one cache)
        data = bounds.bounded(handle, msname, "DATA", 16, SHARED_DM)
        flag = bounds.bounded(handle, msname, "FLAG", 42, SHARED_DM)
        data.__enter__()
        assert handle.getdmprop("FLAG")["MaxCacheSize"] == 16
        flag.__enter__()
        assert handle.getdmprop("DATA")["MaxCacheSize"] == 42
        data.__exit__(None, None, None)  # (FLAG still reads)
        assert handle.getdmprop("FLAG")["MaxCacheSize"] == 42
        assert len(bounds) == 1
        flag.__exit__(None, None, None)
        assert handle.getdmprop("DATA")["MaxCacheSize"] == 3
        assert len(bounds) == 0

        # dask threads: one-channel reads of both variables at once
        seen = []
        read_rows_to_grid = backend_arrays.read_rows_to_grid

        def spy(table, col, plan, grid, *args, **kwargs):
            seen.append(table.getdmprop(col)["MaxCacheSize"])
            return read_rows_to_grid(table, col, plan, grid, *args, **kwargs)

        monkeypatch.setattr(backend_arrays, "read_rows_to_grid", spy)
        n_chan = lazy["FLAG"].sizes["frequency"]
        names = ("VISIBILITY", "FLAG")
        selections = [
            lazy[name].isel(frequency=slice(k, k + 1)).chunk({"time": 10})
            for k in range(n_chan)
            for name in names
        ]
        values = dask.compute(*selections, scheduler="threads", num_workers=4)
        for k in range(n_chan):
            for j, name in enumerate(names):
                assert_same_values(
                    np.asarray(values[2 * k + j]),
                    reference[name].isel(frequency=slice(k, k + 1)).values,
                    (name, k),
                )
        assert len(seen) >= 2 * n_chan
        assert min(seen) >= backend_arrays.CHANNEL_SLICE_MIN_CACHE_MIB
        assert handle.getdmprop("FLAG")["MaxCacheSize"] == 3
        assert len(bounds) == 0
        handle.setmaxcachesize("DATA", 0)


def _interleaved_order(first: list, second: list) -> np.ndarray:
    """
    An order of two lists of rows (each kept in its order) in which runs of
    8 rows of one are separated by single rows of the other: 8 rows of
    ``first`` and 1 of ``second`` for the first half of ``first`` (the gaps
    of the runs of its first rows are bridgeable), then 8 of ``second`` and
    1 of ``first``, then the rows left.
    """
    first, second = list(first), list(second)
    half = len(first) // 2
    order = []
    while len(order) < half + half // 8 and first:
        order += first[:8] + second[:1]
        del first[:8], second[:1]
    while second and first:
        order += second[:8] + first[:1]
        del second[:8], first[:1]
    return np.array(order + first + second, dtype=np.int64)


@pytest.fixture(scope="module")
def interleaved_narrow_tiles(backend_ms, tmp_path_factory):
    """
    Deep copies of the "narrow_tiles" MS (the tiling of the copies is that
    of the original) with their rows in an _interleaved_order:

    - "two_shapes": the rows of SPW 0 (16 channels) and SPW 1 (24): the
      rows of a partition interleave with rows of another cell shape;
    - "one_shape": the rows of SPW 0 only (DDIs 0 and 1, other partitions
      of cells of one shape), interleaved.
    """
    from casacore import tables

    msname = backend_ms("narrow_tiles")
    base = tmp_path_factory.mktemp("interleaved")
    with (
        tables.table(os.path.join(msname, "DATA_DESCRIPTION"), ack=False) as dd_tb,
        tables.table(msname, ack=False) as main_tb,
    ):
        ddi = main_tb.getcol("DATA_DESC_ID")
        spw = dd_tb.getcol("SPECTRAL_WINDOW_ID")[ddi]
        orders = {
            "two_shapes": _interleaved_order(
                np.flatnonzero(spw == 0), np.flatnonzero(spw == 1)
            ),
            "one_shape": _interleaved_order(
                np.flatnonzero(ddi == 0), np.flatnonzero(ddi == 1)
            ),
        }
        paths = {}
        for kind, order in orders.items():
            paths[kind] = str(base / f"{kind}.ms")
            ordered = main_tb.selectrows(order)
            ordered.copy(paths[kind], deep=True, valuecopy=True).close()
            ordered.close()
    yield paths
    shutil.rmtree(base, ignore_errors=True)


@pytest.mark.parametrize(
    "kind, idx",
    [("two_shapes", 0), ("two_shapes", 1), ("two_shapes", 3), ("one_shape", 0)],
)
def test_channel_sliced_reads_of_interleaved_rows(
    interleaved_narrow_tiles, monkeypatch, kind, idx
):
    """
    Channel-sliced reads of partitions whose rows interleave with rows of
    other partitions, in one time sub-block and in several (SUB_BLOCK_BYTES
    lowered: one time each, and a few), give the values the converter
    reads, bit for bit. In a column of one cell shape, read_rows_to_grid
    bridges the gaps of single rows of other partitions; in a column with
    cells of another shape (where casacore reads a range of channels beyond
    a smaller cell without an error), it reads the partition's rows only
    (runs, or selectrows).
    """
    msname = interleaved_narrow_tiles[kind]
    lazy, reference, _ = lazy_and_reference(msname, idx)
    stats = {}
    read_rows_to_grid = backend_arrays.read_rows_to_grid

    def spy(table, col, plan, grid, *args, **kwargs):
        return read_rows_to_grid(table, col, plan, grid, *args, stats=stats, **kwargs)

    monkeypatch.setattr(backend_arrays, "read_rows_to_grid", spy)
    steps = []
    select_outer = backend_arrays.PartitionIndex.select_outer

    def spy_select(self, times, *args, **kwargs):
        steps.append(times.size)
        return select_outer(self, times, *args, **kwargs)

    monkeypatch.setattr(backend_arrays.PartitionIndex, "select_outer", spy_select)
    names = ("VISIBILITY", "VISIBILITY_CORRECTED", "FLAG", "WEIGHT")
    for name in names:
        array = lazy[name]._data.array
        assert array.tile_channels and array.tile_bridge == (kind == "one_shape")
    n_times = lazy["VISIBILITY"].sizes["time"]
    for sub_block_bytes in (128 * 2**20, 1, 5000):
        monkeypatch.setattr(backend_arrays, "SUB_BLOCK_BYTES", sub_block_bytes)
        steps.clear()
        for name in names:
            variable = lazy[name]
            n_chan = variable.sizes["frequency"]
            for sel in _channel_selections(n_chan):
                assert_same_values(
                    variable.isel(sel).values,
                    reference[name].isel(sel).values,
                    f"{name} {sel} {sub_block_bytes}",
                )
        if sub_block_bytes == 1:
            assert set(steps) == {1}
        elif sub_block_bytes == 5000:
            assert 1 < max(steps) < n_times
    assert stats["calls"] and stats["selectrows_calls"]
    if kind == "one_shape":
        assert stats["gap_rows"]
    else:
        assert not stats.get("gap_rows") and not stats.get("bridge_fallbacks")


@pytest.mark.skipif(
    not os.path.exists("/proc/self/io"), reason="needs /proc/self/io (Linux)"
)
def test_channel_sliced_reads_read_fewer_bytes(backend_ms):
    """
    A one-channel read of a column stored in tiles of 3 channels reads the
    tiles of one band of channels: the bytes the process reads
    (/proc/self/io rchar) are fewer than with whole cells by the tiles of
    the other bands of the partition's rows (SPW 1: 24 channels, 8 bands of
    tiles of 2 x 3 channels x 7 rows, 336 bytes).
    """
    from casacore import tables

    msname = backend_ms("narrow_tiles")
    lazy, reference, index = lazy_and_reference(msname, 2)
    variable = lazy["VISIBILITY"]
    array = variable._data.array
    whole = copy.copy(array)
    whole.tile_channels = 0
    whole_variable = lazy_variable(whole, variable.dims).isel(
        frequency=slice(None, None, -1)
    )

    def rchar():
        with open("/proc/self/io") as io:
            return next(int(line.split()[1]) for line in io if line.startswith("rchar"))

    def bytes_read(var):
        before = rchar()
        values = var.isel(frequency=slice(10, 11)).values
        return rchar() - before, values

    with tables.table(msname, ack=False) as main_tb:
        ddi = main_tb.getcol("DATA_DESC_ID")
    # the positions of the partition's rows in the 24-channel hypercube (the
    # rows of DDIs 2 and 3, in row order), and the row tiles that hold them
    rows = np.concatenate(
        [np.arange(a, a + n) for a, n in zip(index.starts, index.lengths, strict=True)]
    )
    assert set(ddi[rows]) == {2}
    cube_rows = np.searchsorted(np.flatnonzero(ddi >= 2), rows)
    row_tiles = np.unique(cube_rows // 7).size
    expected_saving = row_tiles * (8 - 1) * 336
    bytes_read(variable)  # (the first read of the process may read more)
    sliced_bytes, values = bytes_read(variable)
    whole_bytes, whole_values = bytes_read(whole_variable)
    assert_same_values(values, whole_values)
    assert_same_values(
        values, reference["VISIBILITY"].isel(frequency=slice(10, 11)).values
    )
    assert whole_bytes - sliced_bytes >= 0.9 * expected_saving, (
        sliced_bytes,
        whole_bytes,
        expected_saving,
    )


def test_open_msv2_arrays_read_channel_slices(backend_ms):
    """The arrays of an MS opened with the engine know the channel tiling
    of their columns (found when it is opened) and give the values of whole
    cells for channel selections."""
    from xradio.measurement_set import open_msv2

    tree = open_msv2(backend_ms("narrow_tiles"), array_backend="xarray")
    seen = set()
    for node in tree.children.values():
        ds = node.to_dataset(inherit=False)
        for name, width in (("VISIBILITY", 3), ("FLAG", 4), ("WEIGHT", 4)):
            data = ds[name].variable._data
            while not isinstance(data, MSv2BackendArray):
                data = data.array
            assert data.tile_channels == width, (node.name, name)
            whole = copy.copy(data)
            whole.tile_channels = 0
            n_chan = ds.sizes["frequency"]
            for sel in ({"frequency": slice(4, 5)}, {"frequency": [n_chan - 1, 2]}):
                key = xr.core.indexing.OuterIndexer(
                    (slice(None), slice(None))
                    + (np.arange(n_chan)[sel["frequency"]],)
                    + (slice(None),)
                )
                assert_same_values(data[key], whole[key], f"{node.name} {name} {sel}")
            seen.add(n_chan)
    assert seen == {16, 24}


# The layouts of the ms_copy fixture (conftest.MS_COPY_LAYOUTS): MAIN's key
# columns in one StandardStMan shared with other columns, or in a data manager
# each
LAYOUTS = ("shared", "per_column")


def _update_main(msname, column, change):
    from casacore import tables

    with tables.table(msname, readonly=False, ack=False) as main_tb:
        values = main_tb.getcol(column)
        main_tb.putcol(column, change(values.copy()))
    return values


def _row(index, value):
    def change(values):
        values[index] = value
        return values

    return change


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("memo", ["index of the open", "rebuilt"])
@pytest.mark.parametrize(
    "column, change, outcome",
    [
        ("TIME", lambda v: v + 0.5, "times or baselines"),
        ("ANTENNA2", _row(4, 0), r"grid of \(30, 11\)"),  # a new baseline (1, 0)
        ("ANTENNA1", _row(4, 0), None),  # row 4 to baseline (0, 2), still a grid
        ("ANTENNA2", _row(slice(0, 2), np.array([2, 1])), None),  # swapped
        ("DATA", lambda v: v * 2, None),
    ],
)
def test_keys_rewritten_after_the_open(ms_copy, layout, memo, column, change, outcome):
    """TIME, ANTENNA1 or ANTENNA2 rewritten in place after the open (the same
    number of rows): with the index of the open in the memo or rebuilt (as in
    another process), a read gives the same outcome: MSv2ChangedError if the
    partition's (time, baseline) grid changed, else the values placed by the
    current keys (those of an open of the changed MS). With the key columns
    in one data manager shared with other columns, or in a data manager each
    (as CASA split outputs: a write of one key column changes only its own
    data manager)."""
    msname = ms_copy("dense", layout)
    lazy, _, _ = lazy_and_reference(msname, 0)
    _update_main(msname, column, change)
    if memo == "rebuilt":
        backend_arrays.clear_index_memo()
    rebuilds = INDEX_MEMO.stats["rebuilds"]
    if outcome is not None:
        for name in ("VISIBILITY", "FLAG"):
            with pytest.raises(MSv2ChangedError, match=outcome):
                lazy[name].values  # noqa: B018
        return
    values = {name: lazy[name].values for name in ("VISIBILITY", "FLAG", "UVW")}
    rebuilds = INDEX_MEMO.stats["rebuilds"] - rebuilds
    _, reference, _ = lazy_and_reference(msname, 0)  # (an open of the changed MS)
    for name, got in values.items():
        assert_same_values(got, reference[name].values, f"{name} {column}")
    if column == "DATA":
        assert rebuilds == (1 if memo == "rebuilt" else 0)
    else:
        # the check of the partition (its keys were written) made the index
        # again, once; with an empty memo the read had made it first
        assert rebuilds == (2 if memo == "rebuilt" else 1)


@pytest.mark.parametrize("memo", ["index of the open", "rebuilt"])
def test_grid_changed_outside_the_selection(ms_copy, memo):
    """The TIME of one row rewritten to a time its partition did not have
    (its grid has one more time now): a read of other times, which reads
    none of its rows, raises MSv2ChangedError too (the partition is checked
    whole), with the index of the open in the memo or rebuilt."""
    msname = ms_copy("dense")
    lazy, _, _ = lazy_and_reference(msname, 0)

    def change(values):
        values[0] += 1000.5
        return values

    _update_main(msname, "TIME", change)
    if memo == "rebuilt":
        backend_arrays.clear_index_memo()
    for name in ("VISIBILITY", "UVW"):
        with pytest.raises(MSv2ChangedError, match=r"grid of \(31, 10\)"):
            lazy[name].isel(time=slice(5, 8)).values  # noqa: B018


@pytest.mark.parametrize("layout", LAYOUTS)
def test_rows_are_checked_only_after_writes(ms_copy, layout, monkeypatch):
    """The keys of the rows read are checked one by one only when the data
    managers of MAIN's key columns (ROW_KEY_COLUMNS) were written since the
    open (keys_token, from the lock file), or while a handle of this process
    has MAIN open for writing (its writes are seen before they are flushed).
    With a data manager per key column, a write of another column (FLAG_ROW)
    is no write of them, and a write of any one key column is."""
    from casacore import tables

    msname = ms_copy("dense", layout)
    lazy, reference, index = lazy_and_reference(msname, 0)
    assert index.keys_token is not None
    checks = []
    rows_moved = PartitionIndex.rows_moved

    def spy(self, *args):
        checks.append(args[1].size)
        return rows_moved(self, *args)

    monkeypatch.setattr(PartitionIndex, "rows_moved", spy)
    assert_same_values(lazy["FLAG"].values, reference["FLAG"].values)
    assert checks == []
    writer = tables.table(msname, readonly=False, ack=False)
    try:
        assert_same_values(lazy["FLAG"].values, reference["FLAG"].values)
        assert checks == [300]
    finally:
        writer.close()
    checks.clear()
    # (shared: the SSM of the key columns; per_column: a data manager of its own)
    _update_main(msname, "FLAG_ROW", lambda v: v)
    assert_same_values(lazy["FLAG"].values, reference["FLAG"].values)
    assert checks == ([300] if layout == "shared" else [])
    # (a grid key rewritten, a key that does not group rows changed)
    for column, change in (("ANTENNA2", lambda v: v), ("SCAN_NUMBER", _row(0, 2))):
        lazy, reference, index = lazy_and_reference(msname, 0)
        checks.clear()
        _update_main(msname, column, change)
        assert_same_values(lazy["FLAG"].values, reference["FLAG"].values)
        assert checks == [300], column
    # without a token (lock file not readable): always checked
    checks.clear()
    lazy, reference, index = lazy_and_reference(msname, 0)
    index.keys_token = None
    assert_same_values(lazy["UVW"].values, reference["UVW"].values)
    assert checks == [300]


def _partition_of(msname, scheme, **keys):
    """The index of the partition whose description has these values."""
    partitions, _ = create_partitions_with_main_rows(msname, list(scheme))
    (idx,) = (
        idx
        for idx, info in enumerate(partitions)
        if all(info[key] == [value] for key, value in keys.items())
    )
    return idx


def _update_subtable(subtable, column, row, value):
    def change(msname):
        from casacore import tables

        with tables.table(
            os.path.join(msname, subtable), readonly=False, ack=False
        ) as table:
            table.putcell(column, row, value)

    return change


def _update_rows(column, value):
    def change(msname):
        _update_main(msname, column, _row(slice(0, 10), value))

    return change


def _remove_field_ephemeris_id(msname):
    from casacore import tables

    with tables.table(os.path.join(msname, "FIELD"), readonly=False, ack=False) as t:
        t.removecols(["EPHEMERIS_ID"])


CAL, TARGET = "CALIBRATE_PHASE#ON_SOURCE", "OBSERVE_TARGET#ON_SOURCE"
# (variant, scheme, the partition that loses rows, the one that gains them,
# the change, what the first one's reads say, a partition it does not touch):
# rows 0-9 are time 0 of DDI 0; in "rich" they are field 0 (source 0), scan
# 1, state 0 (CAL; state 1 is TARGET)
MOVED_CASES = {
    "DATA_DESC_ID": (
        "dense",
        [],
        {"DATA_DESC_ID": 0},
        {"DATA_DESC_ID": 2},
        _update_rows("DATA_DESC_ID", 2),
        "has the DATA_DESC_ID 2, 0 when it was opened: it is in another partition",
        {"DATA_DESC_ID": 1},
    ),
    "FIELD_ID, scheme FIELD_ID": (
        "rich",
        ["FIELD_ID"],
        {"DATA_DESC_ID": 0, "FIELD_ID": 0, "OBS_MODE": CAL},
        {"DATA_DESC_ID": 0, "FIELD_ID": 1, "OBS_MODE": CAL},
        _update_rows("FIELD_ID", 1),
        "has the FIELD_ID 1, 0 when it was opened: it is in another partition",
        {"DATA_DESC_ID": 3, "FIELD_ID": 0, "OBS_MODE": CAL},
    ),
    "STATE_ID to another OBS_MODE": (
        "rich",
        [],
        {"DATA_DESC_ID": 0, "OBS_MODE": CAL},
        {"DATA_DESC_ID": 0, "OBS_MODE": TARGET},
        _update_rows("STATE_ID", 1),
        f"has the OBS_MODE '{TARGET}', '{CAL}' when it was opened",
        {"DATA_DESC_ID": 3, "OBS_MODE": TARGET},
    ),
    # (FIELD, STATE and SOURCE changed after the open: the rows of the
    # partition are those with its keys as create_partitions derives them)
    "STATE OBS_MODE": (
        "rich",
        [],
        {"DATA_DESC_ID": 0, "OBS_MODE": CAL},
        {"DATA_DESC_ID": 0, "OBS_MODE": TARGET},
        _update_subtable("STATE", "OBS_MODE", 0, TARGET),
        f"has the OBS_MODE '{TARGET}', '{CAL}' when it was opened",
        None,
    ),
    "FIELD SOURCE_ID, scheme SOURCE_ID": (
        "rich",
        ["SOURCE_ID"],
        {"DATA_DESC_ID": 0, "SOURCE_ID": 0, "OBS_MODE": CAL},
        {"DATA_DESC_ID": 0, "SOURCE_ID": 1, "OBS_MODE": CAL},
        _update_subtable("FIELD", "SOURCE_ID", 0, 1),
        "has the SOURCE_ID 1, 0 when it was opened",
        None,
    ),
    "FIELD EPHEMERIS_ID removed": (
        "rich",
        [],
        {"DATA_DESC_ID": 0, "OBS_MODE": CAL},
        None,
        _remove_field_ephemeris_id,
        "the MS no longer has the partition key EPHEMERIS_ID",
        None,
    ),
}


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("case", list(MOVED_CASES))
def test_rows_moved_to_another_partition(ms_copy, layout, case):
    """
    Rows that belong to another partition after the open (a key column of
    MAIN rewritten in place: DATA_DESC_ID; FIELD_ID with the scheme FIELD_ID;
    STATE_ID to a state of another OBS_MODE; or FIELD / STATE changed: an
    OBS_MODE, a SOURCE_ID; or a partition key the MS no longer has): every
    read of the partition that lost them and of the one that gained them
    raises MSv2ChangedError, also a selection without the moved rows (the
    partition is not the one of the open). A partition the change does not
    touch reads as before. The outcome is the same with the memos of the
    open or empty (as in another process).
    """
    variant, scheme, lost_keys, gained_keys, change, message, other_keys = MOVED_CASES[
        case
    ]
    msname = ms_copy(variant, layout)

    def partition(keys):
        if keys is None:
            return None
        return lazy_and_reference(msname, _partition_of(msname, scheme, **keys), scheme)

    lost, gained, other = (partition(k) for k in (lost_keys, gained_keys, other_keys))
    change(msname)
    for memo in ("of the open", "empty"):
        if memo == "empty":
            backend_arrays.clear_index_memo()
        for name in ("VISIBILITY", "FLAG"):
            with pytest.raises(MSv2ChangedError, match=re.escape(message)):
                lost[0][name].values  # noqa: B018
            with pytest.raises(MSv2ChangedError, match=re.escape(message)):
                lost[0][name].isel(time=slice(1, None)).values  # noqa: B018
            if gained is not None:
                with pytest.raises(
                    MSv2ChangedError, match="has the keys of the partition now"
                ):
                    gained[0][name].isel(time=slice(1, 4)).values  # noqa: B018
        if other is not None:
            assert_same_values(
                other[0]["VISIBILITY"].values, other[1]["VISIBILITY"].values
            )


def test_weight_ones_check_the_partition(ms_copy):
    """The WEIGHT=1 fallback reads no MAIN value, but a read of it checks the
    partition like the other variables: MSv2ChangedError after rows moved to
    another partition, also without the memos of the open."""
    msname = ms_copy("no_weight")
    lazy, reference, _ = lazy_and_reference(msname, 0)
    assert isinstance(lazy["WEIGHT"]._data.array, OnesArray)
    assert_same_values(lazy["WEIGHT"].values, reference["WEIGHT"].values)
    _update_main(msname, "DATA_DESC_ID", _row(slice(0, 10), 2))
    for _ in range(2):
        with pytest.raises(MSv2ChangedError, match="another partition now"):
            lazy["WEIGHT"].isel(time=slice(3, 5)).values  # noqa: B018
        backend_arrays.clear_index_memo()
    assert pickle.loads(pickle.dumps(lazy["WEIGHT"]._data.array)).index is not None


def test_partition_is_checked_once_per_state(ms_copy, monkeypatch):
    """After a write of the data manager of MAIN's key columns (here FLAG_ROW,
    which shares it), the whole partition is checked once per state of the
    MS (CHECK_MEMO, by keys_token), not on every read; a further write checks
    it again; while a handle of this process has MAIN open for writing (its
    writes are not flushed), on every read."""
    from casacore import tables

    msname = ms_copy("rich")
    lazy, reference, _ = lazy_and_reference(msname, 0)
    stats = backend_arrays.CHECK_MEMO.stats
    checks = stats["checks"]
    for _ in range(3):
        assert_same_values(lazy["FLAG"].values, reference["FLAG"].values)
    assert stats["checks"] == checks  # (unchanged since the open)
    for expected in (1, 2):
        _update_main(msname, "FLAG_ROW", lambda v: v)
        for _ in range(3):
            assert_same_values(lazy["FLAG"].values, reference["FLAG"].values)
            assert_same_values(
                lazy["VISIBILITY"].isel(time=2).values,
                reference["VISIBILITY"].isel(time=2).values,
            )
        assert stats["checks"] == checks + expected
    writer = tables.table(msname, readonly=False, ack=False)
    try:
        for count in (1, 2):
            assert_same_values(lazy["FLAG"].values, reference["FLAG"].values)
            assert stats["checks"] == checks + 2 + count
    finally:
        writer.close()


@pytest.mark.parametrize("layout", LAYOUTS)
def test_rows_changed_within_their_partition(ms_copy, layout):
    """A key that does not group the rows (SCAN_NUMBER under the scheme [])
    rewritten in place: the rows stay in their partition, which a read
    reads as an open of the changed MS does."""
    msname = ms_copy("rich", layout)
    idx = _partition_of(
        msname, [], DATA_DESC_ID=0, OBS_MODE="CALIBRATE_PHASE#ON_SOURCE"
    )
    lazy, _, _ = lazy_and_reference(msname, idx)
    _update_main(msname, "SCAN_NUMBER", _row(slice(0, 10), 2))
    values = {name: lazy[name].values for name in ("VISIBILITY", "FLAG")}
    _, reference, _ = lazy_and_reference(msname, idx)
    for name, got in values.items():
        assert_same_values(got, reference[name].values, name)


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
