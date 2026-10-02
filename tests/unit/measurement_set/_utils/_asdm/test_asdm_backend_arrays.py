import importlib.util
import sys
import tracemalloc
from pathlib import Path

import numpy as np
import pyasdm
import pytest
import xarray as xr
from astropy.coordinates import SkyCoord
from astropy.time import Time

from xradio.measurement_set._utils._asdm import asdm_backend_arrays
from xradio.measurement_set._utils._asdm._utils.calculate_uvw import calculate_uvw
from xradio.measurement_set._utils._asdm.asdm_backend_arrays import (
    ASDMBackendArray,
    FlagArray,
    PerTimeArray,
    SpectrumArray,
    UVWArray,
    VisibilityArray,
    WeightArray,
    normalize_basic_key,
)

DIMS = ("time", "baseline_id", "frequency", "polarization")
SHAPE = (7, 6, 8, 2)


def reference_values(shape=SHAPE, dtype=np.float64) -> np.ndarray:
    """Values that encode their position (every element distinct)."""
    values = np.arange(np.prod(shape), dtype=np.float64).reshape(shape)
    if np.dtype(dtype).kind == "c":
        values = values + 1j * (values + 0.5)
    return values.astype(dtype)


def check_normalised_key(key, shape):
    """K4: one slice(start, stop, 1) per dim, Python ints, 0 <= start < stop <= len."""
    assert isinstance(key, tuple)
    assert len(key) == len(shape)
    for dim_key, dim_len in zip(key, shape, strict=True):
        assert isinstance(dim_key, slice)
        assert type(dim_key.start) is int and type(dim_key.stop) is int
        assert dim_key.step == 1
        assert 0 <= dim_key.start < dim_key.stop <= dim_len


class FakeArray(ASDMBackendArray):
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


def test_ASDMBackendArray_base():
    backend_array = ASDMBackendArray((2, 3), np.float32)
    assert backend_array.shape == (2, 3)
    assert backend_array.dtype == np.dtype("float32")
    with pytest.raises(NotImplementedError):
        backend_array._raw_indexing_method((slice(0, 1, 1), slice(0, 2, 1)))
    with pytest.raises(NotImplementedError):
        lazy_variable(backend_array, ("a", "b")).values  # noqa: B018
    with pytest.raises(TypeError):
        ASDMBackendArray(None, np.float32)
    with pytest.raises(ValueError, match="negative"):
        ASDMBackendArray((2, -1), np.float32)


@pytest.mark.parametrize("key_name", list(LAZY_KEYS))
@pytest.mark.parametrize("dtype", [np.float64, np.complex64, bool])
def test_lazy_indexing_matches_numpy(key_name, dtype):
    """Every xarray selection (ints, steps, negative steps, empty, lists, mixed)
    gives the values, shape and dtype of numpy indexing of the full array, and
    the loader only receives normalised step-1 slices (F05, K4)."""
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
    """The result always has the declared dtype (loader float32 -> declared
    complex64 with imag 0) and is a writeable array, even when the loader
    returns a read-only view."""
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
            raise NotImplementedError("numBin > 1 is not supported")

    with pytest.raises(NotImplementedError, match="numBin"):
        lazy_variable(Failing(reference_values())).isel(time=0).values  # noqa: B018


@pytest.fixture
def fake_bdf_loaders(monkeypatch):
    """Replace the BDF loaders used by the backend arrays by fakes returning
    blocks of known arrays, recording their arguments."""
    calls = []
    arrays = {}

    def fake_load_visibilities(bdf_paths, spw_id, time_indices_by_bdf, key):
        calls.append(("vis", bdf_paths, spw_id, time_indices_by_bdf, key))
        return arrays["vis"][key].copy()

    def fake_load_flags(bdf_paths, spw_id, time_indices_by_bdf, key):
        calls.append(("flag", bdf_paths, spw_id, time_indices_by_bdf, key))
        return arrays["flag"][key]

    monkeypatch.setattr(
        asdm_backend_arrays,
        "load_visibilities_from_partition_bdfs",
        fake_load_visibilities,
    )
    monkeypatch.setattr(
        asdm_backend_arrays, "load_flags_from_partition_bdfs", fake_load_flags
    )
    return calls, arrays


BDF_PATHS = ["/abs/ASDMBinary/bdf1", "/abs/ASDMBinary/bdf2"]
TIME_INDICES = {"bdf_names": BDF_PATHS, "bdf_start": [0, 4, 7]}


@pytest.mark.parametrize("loader_dtype", [np.float32, np.complex64, np.complex128])
def test_VisibilityArray_values_and_dtype(fake_bdf_loaders, loader_dtype):
    """VISIBILITY is complex64 (K3), with the loaded values, for real (autos /
    single dish) and complex loader outputs (F66), and the loader gets the BDF
    description and a normalised key."""
    calls, arrays = fake_bdf_loaders
    arrays["vis"] = reference_values(dtype=loader_dtype)
    vis = VisibilityArray(SHAPE, BDF_PATHS, 1, TIME_INDICES)
    assert vis.shape == SHAPE
    assert vis.dtype == np.complex64
    variable = lazy_variable(vis)
    assert variable.dtype == np.complex64
    selected = variable.isel(time=slice(5, 1, -2), polarization=0).values
    assert selected.dtype == np.complex64
    np.testing.assert_array_equal(
        selected, arrays["vis"][5:1:-2, :, :, 0].astype(np.complex64)
    )
    assert calls == [
        (
            "vis",
            BDF_PATHS,
            1,
            TIME_INDICES,
            (slice(3, 6, 1), slice(0, 6, 1), slice(0, 8, 1), slice(0, 1, 1)),
        )
    ]


def test_VisibilityArray_dtype_option(fake_bdf_loaders):
    calls, arrays = fake_bdf_loaders
    arrays["vis"] = reference_values(dtype=np.complex128)
    vis = VisibilityArray(SHAPE, BDF_PATHS, 0, TIME_INDICES, dtype=np.complex128)
    assert vis.dtype == np.complex128
    np.testing.assert_array_equal(lazy_variable(vis).values, arrays["vis"])
    with pytest.raises(ValueError, match="complex"):
        VisibilityArray(SHAPE, BDF_PATHS, 0, TIME_INDICES, dtype=np.float32)


def test_SpectrumArray_values_and_dtype(fake_bdf_loaders):
    """SPECTRUM (single dish, K12) is the real part of the loaded auto-correlation
    data, float32, with dims (time, antenna_name, frequency, polarization)."""
    calls, arrays = fake_bdf_loaders
    shape = (5, 3, 4, 2)
    arrays["vis"] = reference_values(shape, dtype=np.complex64)
    spectrum = SpectrumArray(shape, BDF_PATHS, 2, TIME_INDICES)
    assert spectrum.dtype == np.float32
    variable = lazy_variable(
        spectrum, ("time", "antenna_name", "frequency", "polarization")
    )
    values = variable.values
    assert values.dtype == np.float32
    np.testing.assert_array_equal(values, arrays["vis"].real)
    np.testing.assert_array_equal(
        variable.isel(antenna_name=1, frequency=slice(1, None)).values,
        arrays["vis"].real[:, 1, 1:],
    )
    arrays["vis"] = reference_values(shape, dtype=np.float32)
    np.testing.assert_array_equal(variable.values, arrays["vis"])
    with pytest.raises(ValueError, match="floating"):
        SpectrumArray(shape, BDF_PATHS, 2, TIME_INDICES, dtype=np.complex64)


@pytest.mark.parametrize(
    "flag_words",
    [
        lambda shape: (np.arange(np.prod(shape)).reshape(shape) % 5).astype(np.uint32),
        lambda shape: np.arange(np.prod(shape)).reshape(shape) % 3 == 0,
        lambda shape: np.broadcast_to(
            (
                np.arange(np.prod(shape[:2]) * shape[3]).reshape(
                    shape[:2] + (1,) + shape[3:]
                )
                % 4
            )
            == 1,
            shape,
        ),
    ],
    ids=["uint_words", "bool", "broadcast_bool"],
)
def test_FlagArray_values_and_dtype(fake_bdf_loaders, flag_words):
    """FLAG is boolean: flagged = any bit of the flag word set (K2)."""
    calls, arrays = fake_bdf_loaders
    arrays["flag"] = flag_words(SHAPE)
    flag = FlagArray(SHAPE, BDF_PATHS, 1, TIME_INDICES)
    assert flag.dtype == np.dtype("bool")
    variable = lazy_variable(flag)
    values = variable.values
    assert values.dtype == np.dtype("bool")
    assert values.flags.writeable
    np.testing.assert_array_equal(values, arrays["flag"] != 0)
    np.testing.assert_array_equal(
        variable.isel(frequency=3, baseline_id=slice(None, None, 2)).values,
        arrays["flag"][:, ::2, 3] != 0,
    )
    assert calls[-1][:4] == ("flag", BDF_PATHS, 1, TIME_INDICES)


@pytest.mark.parametrize("key_name", list(LAZY_KEYS))
def test_WeightArray_values(key_name):
    weight = WeightArray(SHAPE)
    assert weight.shape == SHAPE
    assert weight.dtype == np.dtype("float64")
    isel = LAZY_KEYS[key_name]
    expected = np.ones(SHAPE)[numpy_key(DIMS, isel)]
    values = lazy_variable(weight).isel(isel).values
    assert values.dtype == np.float64
    assert values.shape == expected.shape
    np.testing.assert_array_equal(values, expected)


def test_WeightArray_materialises_only_the_selection(peak_memory):
    """Reading one element of WEIGHT does not allocate the whole partition (F06)."""
    shape = (100_000, 100_000, 10_000, 4)  # 3.2 PB of float64 if allocated
    weight = WeightArray(shape)
    block = weight._raw_indexing_method((slice(0, 1000, 1),) * 4)
    assert block.shape == (1000,) * 4 and block.nbytes > 1e12  # view, no memory
    variable = lazy_variable(weight)
    (selected, strided), peak = peak_memory(
        lambda: (
            variable[5, 10:12, 100:103, :].values,
            variable[::50_000, 3, ::5_000, 0].values,
        )
    )
    assert peak < 10 * 1024**2
    np.testing.assert_array_equal(selected, np.ones((2, 3, 4)))
    np.testing.assert_array_equal(strided, np.ones((2, 2)))
    assert selected.flags.writeable


PER_TIME_DIMS = ("time", "baseline_id")
PER_TIME_KEYS = {
    name: isel for name, isel in LAZY_KEYS.items() if set(isel) <= set(PER_TIME_DIMS)
}
PER_TIME_VALUES = np.array([10.0, 11.5, 12.0, 13.25, 14.0, 15.0, 16.5])


@pytest.mark.parametrize("key_name", list(PER_TIME_KEYS))
def test_PerTimeArray_values(key_name):
    """TIME_CENTROID / EFFECTIVE_INTEGRATION_TIME data: [t, b] == values[t]
    for every selection, as writeable float64 arrays that do not alter the
    stored values when written to."""
    array = PerTimeArray(PER_TIME_VALUES, SHAPE[1])
    assert array.shape == SHAPE[:2]
    assert array.dtype == np.dtype("float64")
    isel = PER_TIME_KEYS[key_name]
    full = np.repeat(PER_TIME_VALUES[:, None], SHAPE[1], axis=1)
    expected = full[numpy_key(PER_TIME_DIMS, isel)]
    values = lazy_variable(array, PER_TIME_DIMS).isel(isel).values
    assert values.dtype == np.float64
    assert values.shape == expected.shape
    np.testing.assert_array_equal(values, expected)
    assert values.flags.writeable
    values[...] = -1.0
    np.testing.assert_array_equal(lazy_variable(array, PER_TIME_DIMS).values, full)


def test_PerTimeArray_keeps_own_values():
    values = PER_TIME_VALUES.copy()
    array = PerTimeArray(values, 2)
    values[:] = 0.0
    np.testing.assert_array_equal(
        lazy_variable(array, PER_TIME_DIMS).values[:, 1], PER_TIME_VALUES
    )
    with pytest.raises(ValueError, match="one value per integration"):
        PerTimeArray(np.ones((7, 2)), 2)


def test_PerTimeArray_materialises_only_the_selection(peak_memory):
    """Only the per-integration values are stored, and reading some elements
    allocates only those, not a (time, baseline) array."""
    num_time, num_rows = 100_000, 10_000_000  # 8 TB of float64 if allocated
    array = PerTimeArray(np.arange(num_time, dtype=np.float64), num_rows)
    variable = lazy_variable(array, PER_TIME_DIMS)
    (row, block), peak = peak_memory(
        lambda: (variable[5, :10].values, variable[10:12, 3:6].values)
    )
    assert peak < 1024**2
    np.testing.assert_array_equal(row, np.full(10, 5.0))
    np.testing.assert_array_equal(block, [[10.0] * 3, [11.0] * 3])


@pytest.mark.parametrize("already_tracing", [False, True])
def test_peak_memory_helper(peak_memory, already_tracing):
    """The helper of the memory regression tests measures the allocations of
    the function only, also when tracemalloc is already tracing
    (PYTHONTRACEMALLOC, -X tracemalloc), and leaves the tracing state as it
    was."""
    started_here = already_tracing and not tracemalloc.is_tracing()
    if started_here:
        tracemalloc.start()
    try:
        ballast = np.ones(4 * 2**20)  # 32 MiB, traced when already tracing
        tracing = tracemalloc.is_tracing()
        result, peak = peak_memory(lambda: float(np.ones(2**20).sum()))  # 8 MiB
        assert tracemalloc.is_tracing() == tracing
        assert result == 2**20
        assert 8 * 2**20 <= peak < 9 * 2**20
        del ballast
    finally:
        if started_here:
            tracemalloc.stop()


def test_VisibilityArray_missing_bdf():
    """Errors from the BDF reader propagate (no silent zeros), as a RuntimeError
    naming the BDF that is also a pyasdm BDFReaderException with pyasdm's
    message."""
    vis = VisibilityArray((2, 3, 6, 2), ["/foo/bdf1", "/foo/bdf2"], 1, TIME_INDICES)
    with pytest.raises(RuntimeError, match="Cannot open the BDF .*bdf1") as exc:
        lazy_variable(vis).isel(time=0).values  # noqa: B018
    assert isinstance(exc.value, pyasdm.exceptions.BDFReaderException)
    assert "Error while opening" in str(exc.value)


def _load_synthetic_bdf():
    """synthetic_bdf.py (same module object as in the _bdf tests)."""
    name = "xradio_tests_asdm_bdf_synthetic_bdf"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            name, Path(__file__).parent / "_utils" / "_bdf" / "synthetic_bdf.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def test_VisibilityArray_one_integration_peak_memory(
    tmp_path, monkeypatch, peak_memory
):
    """Loading one integration through VisibilityArray (what xarray and dask
    call) allocates the result about once, with no per-component, concatenated
    or per-subset copies of it (F20; the peak was about 3x the result). Reads
    are limited to 64 rows, as for real data where a read block is small
    relative to the result."""
    from xradio.measurement_set._utils._asdm._utils._bdf import (
        pyasdm_get_ndarray_load_function as callback_module,
    )
    from xradio.measurement_set._utils._asdm._utils._bdf.load_time import (
        load_times_from_bdfs,
    )

    sbdf = _load_synthetic_bdf()
    # 64 antennas, 2 full-pol SPWs, 3 integrations: about 1 MB of complex64
    # visibilities per integration for SPW 0
    bdef = sbdf.BDFDef([[sbdf._full(16)], [sbdf._full(8)]], num_antenna=64)
    path = sbdf.write_bdf(bdef, str(tmp_path / "bdf.bin"))
    monkeypatch.setattr(callback_module, "MAX_VALUES_PER_READ", 64 * (16 + 8) * 4 * 2)

    time_centers, _, _, _, time_indices = load_times_from_bdfs([path])
    expected = sbdf.expected_visibility(bdef, (0, 0), [1]).astype(np.complex64)
    shape = (len(time_centers),) + expected.shape[1:]
    vis_array = VisibilityArray(shape, [path], 0, time_indices)
    key = (slice(1, 2, 1),) + tuple(slice(0, size, 1) for size in shape[1:])

    vis_array._raw_indexing_method(key)  # warm-up (imports, caches)
    vis, peak = peak_memory(lambda: vis_array._raw_indexing_method(key))
    np.testing.assert_array_equal(vis, expected)
    assert vis.nbytes > 1e6
    assert peak / vis.nbytes < 1.5


@pytest.fixture
def uvw_inputs():
    names = ["A", "B", "C", "D"]
    center = np.array([2225142.180268967, -5440307.370348562, -2481029.851873547])
    positions = center + np.array(
        [
            [0.0, 0.0, 0.0],
            [312.25, 141.5, -96.75],
            [-205.5, 498.0, 251.25],
            [790, -310.5, 402],
        ]
    )
    pairs = [("B", "A"), ("C", "A"), ("C", "B"), ("D", "A"), ("D", "B"), ("D", "C")]
    pairs += [(name, name) for name in names]
    unix_times = Time("2024-03-01T05:59:23", scale="utc").unix + np.arange(7) * 6.048
    time = xr.DataArray(
        unix_times,
        dims="time",
        attrs={"type": "time", "units": "s", "format": "unix", "scale": "utc"},
    )
    antenna_position = xr.DataArray(
        positions,
        dims=["antenna_name", "cartesian_pos_label"],
        coords={"antenna_name": names, "cartesian_pos_label": ["x", "y", "z"]},
        attrs={"units": "m"},
    )
    antenna1 = xr.DataArray([pair[0] for pair in pairs], dims="baseline_id")
    antenna2 = xr.DataArray([pair[1] for pair in pairs], dims="baseline_id")
    phase_center = xr.DataArray(
        np.stack([np.linspace(1.0, 1.3, 7), np.linspace(-0.5, -0.2, 7)], axis=-1),
        dims=["time", "sky_dir_label"],
        coords={"sky_dir_label": ["ra", "dec"]},
        attrs={"type": "sky_coord", "units": "rad", "frame": "icrs"},
    )
    shape = (len(unix_times), len(pairs), 3)
    return shape, (time, antenna1, antenna2, antenna_position, phase_center)


UVW_DIMS = ("time", "baseline_id", "uvw_label")
UVW_KEYS = [
    {"time": 0},
    {"uvw_label": 2},
    {"uvw_label": slice(0, 2)},
    {"time": 6, "uvw_label": 1},
    {"time": slice(None, None, 3)},
    {"time": slice(None, None, -1)},
    {"baseline_id": slice(None, None, -2)},
    {"baseline_id": 7},
    {"time": [5, 0, 3], "baseline_id": slice(1, 5)},
    {"time": slice(2, 2)},
]


@pytest.mark.parametrize("isel", UVW_KEYS, ids=[str(key) for key in UVW_KEYS])
def test_UVWArray_lazy_indexing(uvw_inputs, isel):
    """Lazy UVW selections (int time / label, steps, chunks) equal numpy indexing
    of the full UVW computed directly (F22, K10)."""
    shape, args = uvw_inputs
    full = calculate_uvw(None, *args)
    assert full.shape == shape
    uvw = UVWArray(shape, *args)
    assert uvw.dtype == np.float64
    expected = full[numpy_key(UVW_DIMS, isel)]
    values = lazy_variable(uvw, UVW_DIMS).isel(isel).values
    assert values.shape == expected.shape
    np.testing.assert_allclose(values, expected, rtol=0, atol=1e-9)


@pytest.mark.parametrize(
    "chunks", [{"time": 3}, {"uvw_label": 1}, {"time": 4, "baseline_id": 3}]
)
def test_UVWArray_dask_chunks(uvw_inputs, chunks):
    shape, args = uvw_inputs
    full = calculate_uvw(None, *args)
    chunked = lazy_variable(UVWArray(shape, *args), UVW_DIMS).chunk(chunks)
    np.testing.assert_allclose(chunked.values, full, rtol=0, atol=1e-9)


def test_UVWArray_bad_shape():
    with pytest.raises(ValueError, match="uvw_label"):
        UVWArray((3, 2), None, None, None, None, None)


@pytest.mark.parametrize("frame", ["hadec", "altaz", "itrs"])
def test_UVWArray_unsupported_frame_fails_when_created(uvw_inputs, frame, monkeypatch):
    """A phase center frame the UVW calculation does not support (Earth-fixed /
    topocentric frames that field directions can have) fails when the array is
    created, that is when the partition is opened (where open_asdm skips it,
    K13), not later in a compute of the processing set. No UVW is calculated."""
    shape, (time, antenna1, antenna2, antenna_position, phase_center) = uvw_inputs
    phase_center = phase_center.copy()
    phase_center.attrs["frame"] = frame

    def fail(*args, **kwargs):
        raise AssertionError("UVW must not be calculated")

    monkeypatch.setattr(asdm_backend_arrays, "calculate_uvw", fail)
    with pytest.raises(NotImplementedError, match=f"frame '{frame}' is not supported"):
        UVWArray(shape, time, antenna1, antenna2, antenna_position, phase_center)


def test_UVWArray_supergalactic_phase_center(uvw_inputs):
    """A supergalactic phase center (ASDM SUPERGAL) is supported: the UVW equal
    those of the same directions converted to ICRS."""
    shape, (time, antenna1, antenna2, antenna_position, phase_center) = uvw_inputs
    supergalactic = phase_center.copy()
    supergalactic.attrs["frame"] = "supergalactic"
    values = supergalactic.values
    icrs = SkyCoord(values[:, 0], values[:, 1], unit="rad", frame="supergalactic").icrs
    icrs_center = phase_center.copy(data=np.stack([icrs.ra.rad, icrs.dec.rad], axis=-1))
    uvw = UVWArray(shape, time, antenna1, antenna2, antenna_position, supergalactic)
    values = lazy_variable(uvw, UVW_DIMS).values
    expected = calculate_uvw(
        None, time, antenna1, antenna2, antenna_position, icrs_center
    )
    np.testing.assert_allclose(values, expected, rtol=0, atol=1e-6)
    # not the UVW of the same numbers labelled ICRS
    assert np.abs(values - calculate_uvw(None, *uvw_inputs[1])).max() > 1.0


def test_UVWArray_checks_inputs_when_created(uvw_inputs):
    """Inputs that cannot give the UVW of the declared shape fail when the array
    is created (not when it is computed)."""
    shape, args = uvw_inputs
    time, antenna1, antenna2, antenna_position, phase_center = args

    with pytest.raises(ValueError, match="does not match the inputs"):
        UVWArray((shape[0] + 1, shape[1], 3), *args)
    with pytest.raises(ValueError, match="does not match the inputs"):
        UVWArray((shape[0], shape[1] - 1, 3), *args)
    with pytest.raises(ValueError, match="does not match the inputs"):
        UVWArray((shape[0], shape[1], 2), *args)

    no_scale = time.copy()
    del no_scale.attrs["scale"]
    with pytest.raises(ValueError, match="scale"):
        UVWArray(shape, no_scale, antenna1, antenna2, antenna_position, phase_center)

    unknown = antenna1.copy(data=["X"] * antenna1.size)
    with pytest.raises(ValueError, match="'X'"):
        UVWArray(shape, time, unknown, antenna2, antenna_position, phase_center)

    with pytest.raises(ValueError, match="times"):
        UVWArray(shape, time, antenna1, antenna2, antenna_position, phase_center[:-1])
