from unittest import mock

import numpy as np
import pyasdm
import pytest

import xradio.measurement_set._utils._asdm._utils._bdf.robust_load_data_flags as rldf

STOKES = pyasdm.enumerations.StokesParameter


def make_spw(nchan, cross=("XX", "YY"), sd=("XX", "YY"), num_bin=1, sw="1"):
    return {
        "crossPolProducts": [getattr(STOKES, pol) for pol in cross],
        "sdPolProducts": [getattr(STOKES, pol) for pol in sd],
        "scaleFactor": 1.0,
        "numSpectralPoint": nchan,
        "numBin": num_bin,
        "sideband": None,
        "sw": sw,
    }


#: CROSS_AND_AUTO, 3 antennas: BB_1 with SPWs of 8 and 1 channels, BB_2 with 4
basebands_simple = [
    {"name": "BB_1", "spectralWindows": [make_spw(8), make_spw(1, sw="2")]},
    {"name": "BB_2", "spectralWindows": [make_spw(4)]},
]


def configure_header_mock(
    header,
    basebands=basebands_simple,
    correlation_mode=pyasdm.enumerations.CorrelationMode.CROSS_AND_AUTO,
    num_antenna=3,
    present=("flags", "crossData", "autoData"),
    dimensionality=1,
    num_time=0,
):
    """A BDF header mock with consistent sizes for the components present."""
    from xradio.measurement_set._utils._asdm._utils._bdf.bdf_description_checks import (
        expected_data_sizes,
    )

    header.getDimensionality.return_value = dimensionality
    header.getNumTime.return_value = num_time
    header.getProcessorType.return_value = pyasdm.enumerations.ProcessorType.CORRELATOR
    header.getBinaryTypes.return_value = [
        "flags",
        "actualTimes",
        "actualDurations",
        "zeroLags",
        "crossData",
        "autoData",
    ]
    header.getCorrelationMode.return_value = correlation_mode
    header.getAPClist.return_value = [
        pyasdm.enumerations.AtmPhaseCorrection.AP_UNCORRECTED
    ]
    header.getNumAntenna.return_value = num_antenna
    header.getBasebandsList.return_value = basebands
    header.hasBinary.side_effect = lambda name: name in present
    sizes = expected_data_sizes(
        {
            "dimensionality": dimensionality,
            "num_time": num_time,
            "num_antenna": num_antenna,
            "apc": header.getAPClist.return_value,
            "basebands": basebands,
            "processor_type": header.getProcessorType.return_value,
        }
    )
    header.getSize.side_effect = lambda name: sizes.get(name, [1])[0]
    header.getAxesNames.return_value = []
    return header


# ---------------------------------------------------------------------------
# Partition level, with fake per-BDF loaders
# ---------------------------------------------------------------------------

#: integrations per BDF (including an empty BDF)
BDF_LENGTHS = [3, 2, 0, 4]
FULL_SHAPE = (sum(BDF_LENGTHS), 6, 8, 2)


@pytest.fixture
def fake_partition(monkeypatch):
    """
    A partition of 4 BDFs whose per-BDF loaders return blocks of known full
    arrays. Records the per-BDF calls.
    """
    bdf_names = [f"bdf_{idx}" for idx in range(len(BDF_LENGTHS))]
    bdf_start = np.concatenate([[0], np.cumsum(BDF_LENGTHS)]).tolist()
    size = int(np.prod(FULL_SHAPE))
    full_vis = (np.arange(size) + 1j * np.arange(size)[::-1]).reshape(FULL_SHAPE)
    full_flags = (np.arange(size) % 3 == 0).reshape(FULL_SHAPE)
    calls = []

    def make_fake(full):
        def fake_load(bdf_path, spw_id, array_slice):
            calls.append((bdf_path, spw_id, array_slice))
            assert len(array_slice) == 4
            for key in array_slice:
                assert isinstance(key, slice) and key.step == 1
                assert 0 <= key.start < key.stop
            offset = bdf_start[bdf_names.index(bdf_path)]
            time_key = array_slice[0]
            assert time_key.stop <= BDF_LENGTHS[bdf_names.index(bdf_path)]
            return full[
                offset + time_key.start : offset + time_key.stop, *array_slice[1:]
            ]

        return fake_load

    monkeypatch.setattr(rldf, "load_visibilities_from_bdf", make_fake(full_vis))
    monkeypatch.setattr(rldf, "load_flags_from_bdf", make_fake(full_flags))
    monkeypatch.setattr(rldf, "_partition_data_dims", lambda path, spw: FULL_SHAPE[1:])

    return {
        "bdf_names": bdf_names,
        "time_indices_by_bdf": {"bdf_names": bdf_names, "bdf_start": bdf_start},
        "full_vis": full_vis,
        "full_flags": full_flags,
        "calls": calls,
    }


PARTITION_KEYS = [
    (slice(0, 9), slice(0, 6), slice(0, 8), slice(0, 2)),
    (slice(None), slice(None), slice(None), slice(None)),
    (None, None, None, None),
    (),
    (slice(1, 2),),
    # across BDF boundaries (and the empty BDF)
    (slice(2, 6), slice(1, 4), slice(0, 3), slice(1, 2)),
    (slice(4, 9), slice(0, 6), slice(2, 8), slice(0, 2)),
    # frequency selections starting at 0, with steps (F21)
    (slice(0, 9), slice(0, 6), slice(0, 4), slice(0, 2)),
    (slice(None), slice(None), slice(0, 8, 2), slice(None)),
    (slice(None), slice(None), slice(1, None, 3), slice(None)),
    # int keys drop the dimension (F05, F16)
    (0, slice(None), slice(None), slice(None)),
    (8, 3, slice(None), 0),
    (slice(None), slice(None), 7, 1),
    (slice(None), 5, slice(None), slice(None)),
    (-1, -1, -1, -1),
    # steps and negative indices along any dimension
    (slice(None, None, 2), slice(None), slice(None), slice(None)),
    (slice(None, None, -1), slice(None), slice(None), slice(None)),
    (slice(-4, None), slice(None, -2), slice(None), slice(None)),
    (slice(1, 8, 3), slice(0, 6, 5), slice(None, None, -2), 1),
    # empty selections, at BDF boundaries and at the end (F56)
    (slice(0, 0), slice(None), slice(None), slice(None)),
    (slice(3, 3), slice(None), slice(None), slice(None)),
    (slice(5, 5), slice(0, 6), slice(0, 8), slice(0, 2)),
    (slice(9, 9), slice(0, 6), slice(0, 8), slice(0, 2)),
    (slice(9, None), slice(None), slice(None), slice(None)),
    (slice(2, 4), slice(3, 3), slice(None), slice(None)),
    (slice(None), slice(None), slice(4, 2), 0),
]


@pytest.mark.parametrize("key", PARTITION_KEYS)
@pytest.mark.parametrize("var", ["vis", "flags"])
def test_load_from_partition_bdfs_matches_numpy_indexing(fake_partition, key, var):
    """Any numpy basic index gives the same values as indexing the full array, and
    the per-BDF loaders only get BDF-local contiguous blocks."""
    if var == "vis":
        load_function = rldf.load_visibilities_from_partition_bdfs
        full = fake_partition["full_vis"]
        expected_dtype = np.complex64
    else:
        load_function = rldf.load_flags_from_partition_bdfs
        full = fake_partition["full_flags"]
        expected_dtype = np.bool_

    result = load_function(
        fake_partition["bdf_names"], 0, fake_partition["time_indices_by_bdf"], key
    )

    numpy_key = tuple(slice(None) if dim_key is None else dim_key for dim_key in key)
    expected = full[numpy_key]
    assert result.dtype == expected_dtype
    assert result.shape == expected.shape
    np.testing.assert_array_equal(result, expected.astype(expected_dtype))

    calls = fake_partition["calls"]
    assert "bdf_2" not in [call[0] for call in calls]
    if expected.size == 0:
        assert calls == []


def test_load_from_partition_bdfs_per_bdf_keys(fake_partition):
    """The time selection is split into BDF-local selections, the other dimensions
    are passed unchanged (K5)."""
    key = (slice(2, 8, 1), slice(1, 4, 1), slice(0, 3, 1), slice(1, 2, 1))
    rldf.load_visibilities_from_partition_bdfs(
        fake_partition["bdf_names"], 3, fake_partition["time_indices_by_bdf"], key
    )
    assert fake_partition["calls"] == [
        ("bdf_0", 3, (slice(2, 3, 1), *key[1:])),
        ("bdf_1", 3, (slice(0, 2, 1), *key[1:])),
        ("bdf_3", 3, (slice(0, 3, 1), *key[1:])),
    ]


def test_load_from_partition_bdfs_debug_log(fake_partition, monkeypatch):
    """One DEBUG (not INFO) record per call, counting the BDFs read (G101)."""
    mock_logger = mock.MagicMock()
    monkeypatch.setattr(rldf, "xradio_logger", lambda: mock_logger)
    rldf.load_flags_from_partition_bdfs(
        fake_partition["bdf_names"],
        0,
        fake_partition["time_indices_by_bdf"],
        (slice(3, 5), slice(0, 6), slice(0, 8), slice(0, 2)),
    )
    mock_logger.info.assert_not_called()
    mock_logger.debug.assert_called_once()
    assert "from 1 BDFs (of 4 in the partition)" in mock_logger.debug.call_args[0][0]


def test_load_from_partition_bdfs_wrong_block_shape(fake_partition, monkeypatch):
    monkeypatch.setattr(
        rldf,
        "load_visibilities_from_bdf",
        lambda bdf_path, spw_id, array_slice: np.zeros((1, 6, 8, 2)),
    )
    with pytest.raises(RuntimeError, match=r"shape \(1, 6, 8, 2\) from BDF bdf_0"):
        rldf.load_visibilities_from_partition_bdfs(
            fake_partition["bdf_names"],
            0,
            fake_partition["time_indices_by_bdf"],
            (slice(0, 3), slice(0, 6), slice(0, 8), slice(0, 2)),
        )


@pytest.mark.parametrize(
    "key, expected_error",
    [
        ((slice(0, 10), slice(0, 6), slice(0, 8), slice(0, 2)), IndexError),
        ((9, slice(None), slice(None), slice(None)), IndexError),
        ((slice(None),) * 5, ValueError),
        (slice(None), TypeError),
    ],
)
def test_load_from_partition_bdfs_errors(fake_partition, key, expected_error):
    with pytest.raises(expected_error):
        rldf.load_visibilities_from_partition_bdfs(
            fake_partition["bdf_names"], 0, fake_partition["time_indices_by_bdf"], key
        )


@pytest.mark.parametrize(
    "time_indices_by_bdf",
    [{"bdf_names": [], "bdf_start": []}, {"bdf_names": ["a"], "bdf_start": [0]}],
)
def test_load_from_partition_bdfs_no_bdfs(time_indices_by_bdf):
    with pytest.raises(ValueError, match="at least one BDF"):
        rldf.load_visibilities_from_partition_bdfs([], 0, time_indices_by_bdf)
    with pytest.raises(ValueError, match="at least one BDF"):
        rldf.load_flags_from_partition_bdfs([], 0, time_indices_by_bdf)


def test_load_flags_from_partition_bdfs_inexistent():
    """The int key needs the dimensions from the first BDF header
    (_partition_data_dims), whose open error names the BDF."""
    bdf_paths = ["/inexistent_path_to_flags/foo/"]
    times_by_bdf = {"bdf_names": bdf_paths, "bdf_start": [0, 4]}
    with pytest.raises(
        RuntimeError, match="Cannot open the BDF /inexistent_path_to_flags/foo/ "
    ) as exc_info:
        rldf.load_flags_from_partition_bdfs(
            bdf_paths, 0, times_by_bdf, (slice(0, 4), slice(0, 6), slice(0, 1), 0)
        )
    assert isinstance(exc_info.value, pyasdm.exceptions.BDFReaderException)


@pytest.mark.parametrize(
    "contents",
    [b"", b"MIME-Version: 1.0\nContent-Type: garbage\n"],
    ids=["empty", "corrupt_header"],
)
@pytest.mark.parametrize(
    "key",
    [
        (slice(0, 1, 1),) * 4,  # block keys => BDF-level loader directly
        (0, slice(None), slice(None), slice(None)),  # header read for the dims
    ],
)
@pytest.mark.parametrize(
    "load_function",
    [rldf.load_visibilities_from_partition_bdfs, rldf.load_flags_from_partition_bdfs],
)
def test_load_from_partition_bdfs_broken_bdf(tmp_path, contents, key, load_function):
    """An empty / header-corrupt BDF raises a RuntimeError naming that BDF."""
    bdf_path = str(tmp_path / "uid___A002_X1_broken")
    with open(bdf_path, "wb") as bdf_file:
        bdf_file.write(contents)
    times_by_bdf = {"bdf_names": [bdf_path], "bdf_start": [0, 2]}
    with pytest.raises(RuntimeError, match=f"Cannot open the BDF {bdf_path} "):
        load_function([bdf_path], 0, times_by_bdf, key)


@pytest.mark.parametrize(
    "load_function", [rldf.load_visibilities_from_bdf, rldf.load_flags_from_bdf]
)
def test_load_from_bdf_open_error(load_function):
    """BDF level: header parse errors name the BDF and close the reader."""
    with mock.patch("pyasdm.bdf.BDFReader") as mock_bdf_reader:
        reader = mock_bdf_reader.return_value
        reader.open.side_effect = pyasdm.exceptions.BDFReaderException(
            "could not detect a well formed MIME header field in b''"
        )
        with pytest.raises(RuntimeError, match="BDF /data/bdf_X7 .*MIME header"):
            load_function("/data/bdf_X7", 0, BDF_KEY)
    reader.close.assert_called_once()
    reader.getHeader.assert_not_called()


# ---------------------------------------------------------------------------
# Partition level: no second full-size copy of the result (F20)
# ---------------------------------------------------------------------------


def one_bdf_partition(monkeypatch, loaded, var):
    """Partition of 2 BDFs of 3 integrations whose BDF-level loader returns
    `loaded` (a callable of the BDF key, or an array)."""
    names = ["bdf_a", "bdf_b"]
    calls = []

    def fake_load(bdf_path, spw_id, array_slice):
        calls.append((bdf_path, array_slice))
        return loaded(array_slice) if callable(loaded) else loaded

    attr = "load_visibilities_from_bdf" if var == "vis" else "load_flags_from_bdf"
    monkeypatch.setattr(rldf, attr, fake_load)
    function = (
        rldf.load_visibilities_from_partition_bdfs
        if var == "vis"
        else rldf.load_flags_from_partition_bdfs
    )
    return (
        lambda key: function(
            names, 0, {"bdf_names": names, "bdf_start": [0, 3, 6]}, key
        ),
        calls,
    )


@pytest.mark.parametrize(
    "var, dtype", [("vis", np.complex64), ("flags", np.bool_)], ids=["vis", "flags"]
)
def test_load_from_partition_bdfs_one_bdf_no_copy(monkeypatch, var, dtype):
    """A selection in one BDF returns the array loaded from the BDF itself."""
    loaded = np.ones((1, 6, 8, 2), dtype=dtype)
    load, calls = one_bdf_partition(monkeypatch, loaded, var)
    key = (slice(4, 5, 1), slice(0, 6, 1), slice(0, 8, 1), slice(0, 2, 1))
    result = load(key)
    assert result is loaded
    assert calls == [("bdf_b", (slice(1, 2, 1), *key[1:]))]


@pytest.mark.parametrize(
    "loaded",
    [
        # broadcast BDF flags (read-only view)
        np.broadcast_to(np.array([True, False])[None, None, None, :], (1, 6, 8, 2)),
        # view of a larger array
        np.zeros((2, 6, 8, 2), dtype=np.complex64)[:1],
        # non-contiguous
        np.zeros((1, 6, 2, 8), dtype=np.complex64).transpose(0, 1, 3, 2),
        # other dtype (cast)
        np.full((1, 6, 8, 2), 2 + 1j, dtype=np.complex128),
    ],
    ids=["broadcast", "view", "non_contiguous", "complex128"],
)
def test_load_from_partition_bdfs_one_bdf_copy_when_needed(monkeypatch, loaded):
    """Read-only views, views of other arrays, non-contiguous arrays or other
    dtypes are copied into a writable, C-contiguous result of the right dtype."""
    var = "flags" if loaded.dtype == np.bool_ else "vis"
    load, _ = one_bdf_partition(monkeypatch, loaded, var)
    result = load((slice(0, 1, 1), slice(0, 6, 1), slice(0, 8, 1), slice(0, 2, 1)))
    assert result.dtype == (np.bool_ if var == "flags" else np.complex64)
    assert result.flags.writeable and result.flags.c_contiguous
    assert result.flags.owndata
    assert not np.shares_memory(result, loaded)
    np.testing.assert_array_equal(result, loaded.astype(result.dtype))


def test_load_from_partition_bdfs_peak_memory(monkeypatch, peak_memory):
    """Loading one integration (one BDF) allocates the result only once: the peak
    memory stays well below two copies of the result (F20)."""
    shape = (1, 1000, 512, 4)  # 16 MiB of complex64
    nbytes = int(np.prod(shape)) * 8

    load, _ = one_bdf_partition(
        monkeypatch, lambda key: np.ones(shape, dtype=np.complex64), "vis"
    )
    key = (slice(2, 3, 1), slice(0, 1000, 1), slice(0, 512, 1), slice(0, 4, 1))
    result, peak = peak_memory(lambda: load(key))
    assert result.shape == shape
    assert peak < 1.25 * nbytes


def test_load_from_partition_bdfs_across_bdfs_values(monkeypatch):
    """Across BDFs the BDF blocks are assembled in time order."""
    load, calls = one_bdf_partition(
        monkeypatch,
        lambda key: np.full(
            (key[0].stop - key[0].start, 2, 3, 1), key[0].start, dtype=np.complex64
        ),
        "vis",
    )
    result = load((slice(1, 5, 1), slice(0, 2, 1), slice(0, 3, 1), slice(0, 1, 1)))
    assert [call[0] for call in calls] == ["bdf_a", "bdf_b"]
    np.testing.assert_array_equal(result[:, 0, 0, 0], [1, 1, 0, 0])
    assert result.flags.writeable and result.flags.owndata


# ---------------------------------------------------------------------------
# Real (synthetic) BDFs, values against the truth of the writer
# ---------------------------------------------------------------------------


def partition_of_spw(truth, spw_id):
    """BDF paths, time indices and integrations of the (single) partition of an
    SPW of a synthetic ASDM."""
    from xradio.measurement_set._utils._asdm._utils._bdf.load_time import (
        load_times_from_partition_bdfs,
    )

    refs = truth.integrations(spw_id)
    bdf_paths = list(dict.fromkeys(ref.bdf.path for ref in refs))
    *_, time_indices_by_bdf = load_times_from_partition_bdfs(bdf_paths)
    return bdf_paths, time_indices_by_bdf, refs


# synth_interferometric: 4 antennas => 6 cross baselines + 4 autos; SPW 0 has 8
# channels; 10 integrations in BDFs of 3, 2, 2, 3 integrations
REAL_KEYS = [
    (slice(None), slice(None), slice(None), slice(None)),
    (slice(0, 10), slice(0, 10), slice(0, 8), slice(0, 2)),
    # frequency selections starting at channel 0, and with a step (F21)
    (slice(0, 10), slice(0, 10), slice(0, 4), slice(0, 2)),
    (slice(None), slice(None), slice(0, 8, 2), slice(None)),
    (slice(None), slice(None), slice(3, 6), slice(None)),
    # baseline selections ending at the cross/auto boundary (F10)
    (slice(0, 10), slice(0, 6), slice(0, 8), slice(0, 2)),
    (slice(0, 10), slice(4, 6), slice(0, 8), slice(0, 2)),
    (slice(0, 10), slice(6, 10), slice(0, 8), slice(0, 2)),
    (slice(0, 10), slice(5, 7), slice(0, 8), slice(0, 2)),
    # time selections across BDF boundaries, empty ones (F56)
    (slice(2, 6), slice(None), slice(None), slice(None)),
    (slice(3, 3), slice(None), slice(None), slice(None)),
    (slice(10, 10), slice(0, 10), slice(0, 8), slice(0, 2)),
    # ints
    (4, 7, slice(None), 1),
    (slice(None), slice(None), 0, 0),
]


@pytest.mark.parametrize("key", REAL_KEYS)
def test_load_partition_values_real_bdfs(
    synth_interferometric, synthetic_asdm_module, key
):
    synth = synthetic_asdm_module
    truth = synth_interferometric
    spw = truth.spws[0]
    bdf_paths, time_indices_by_bdf, refs = partition_of_spw(truth, spw.spw_id)

    vis = rldf.load_visibilities_from_partition_bdfs(
        bdf_paths, spw.bdf_index, time_indices_by_bdf, key
    )
    expected_vis = synth.expected_visibility(truth, spw.spw_id, refs)[key]
    assert vis.dtype == np.complex64
    assert vis.shape == expected_vis.shape
    np.testing.assert_allclose(vis, expected_vis, rtol=1e-6, atol=1e-6)

    flags = rldf.load_flags_from_partition_bdfs(
        bdf_paths, spw.bdf_index, time_indices_by_bdf, key
    )
    expected_flags = synth.expected_flags(truth, spw.spw_id, refs)[key]
    assert flags.dtype == np.bool_
    assert flags.shape == expected_flags.shape
    np.testing.assert_array_equal(flags, expected_flags)


@pytest.mark.parametrize("spw_idx", [0, 1, 2])
def test_load_partition_values_real_bdfs_all_spws(
    synth_interferometric, synthetic_asdm_module, spw_idx
):
    synth = synthetic_asdm_module
    truth = synth_interferometric
    spw = truth.spws[spw_idx]
    bdf_paths, time_indices_by_bdf, refs = partition_of_spw(truth, spw.spw_id)
    vis = rldf.load_visibilities_from_partition_bdfs(
        bdf_paths, spw.bdf_index, time_indices_by_bdf
    )
    np.testing.assert_allclose(
        vis, synth.expected_visibility(truth, spw.spw_id, refs), rtol=1e-6, atol=1e-6
    )
    flags = rldf.load_flags_from_partition_bdfs(
        bdf_paths, spw.bdf_index, time_indices_by_bdf
    )
    np.testing.assert_array_equal(flags, synth.expected_flags(truth, spw.spw_id, refs))


def test_load_partition_values_packed_bdfs(synth_interleaved, synthetic_asdm_module):
    """Packed WVR BDFs (numTime=4): every integration is loaded (F19, K5)."""
    synth = synthetic_asdm_module
    truth = synth_interleaved
    spw = truth.spws[0]
    bdf_paths, time_indices_by_bdf, refs = partition_of_spw(truth, spw.spw_id)
    assert time_indices_by_bdf["bdf_start"][-1] == len(refs) == 4 * len(bdf_paths)

    for key in [(slice(None),) * 4, (slice(2, 7), slice(1, 3), slice(0, 2), 0)]:
        vis = rldf.load_visibilities_from_partition_bdfs(
            bdf_paths, spw.bdf_index, time_indices_by_bdf, key
        )
        expected = synth.expected_visibility(truth, spw.spw_id, refs)[key]
        np.testing.assert_allclose(vis, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    "apc", [("AP_UNCORRECTED", "AP_CORRECTED"), ("AP_CORRECTED", "AP_UNCORRECTED")]
)
def test_load_partition_values_two_apc(make_synthetic_asdm, synthetic_asdm_module, apc):
    """With two APC values, the AP_UNCORRECTED data are loaded, whatever their
    position (K6, F11).

    The synthetic writer encodes the AP_UNCORRECTED block exactly as the data of
    a single-APC (AP_UNCORRECTED) ASDM, and adds 1000 to the real part and -1000
    to the imaginary part of the AP_CORRECTED block. The expected values are
    therefore those of the same ASDM written with only AP_UNCORRECTED.
    """
    synth = synthetic_asdm_module
    truth = make_synthetic_asdm(synth.small_dtype_spec("INT32_TYPE", 2.0, apc=apc))
    reference = make_synthetic_asdm(
        synth.small_dtype_spec("INT32_TYPE", 2.0, apc=("AP_UNCORRECTED",))
    )
    spw = truth.spws[0]
    bdf_paths, time_indices_by_bdf, refs = partition_of_spw(truth, spw.spw_id)
    vis = rldf.load_visibilities_from_partition_bdfs(
        bdf_paths, spw.bdf_index, time_indices_by_bdf
    )
    expected = synth.expected_visibility(
        reference, spw.spw_id, reference.integrations(spw.spw_id)
    )
    assert len(refs) == expected.shape[0]
    np.testing.assert_allclose(vis, expected, rtol=1e-6, atol=1e-6)
    # cross baselines: nothing from the AP_CORRECTED block (|re|, |im| >= 500)
    nbl = len(truth.cross_baselines)
    assert np.all(np.abs(vis[:, :nbl].real) < 500)
    assert np.all(np.abs(vis[:, :nbl].imag) < 500)


def test_load_num_bin_not_supported(make_synthetic_asdm, synthetic_asdm_module):
    """numBin > 1 raises NotImplementedError, never silently wrong data (K6, F11)."""
    synth = synthetic_asdm_module
    truth = make_synthetic_asdm(synth.small_dtype_spec("FLOAT32_TYPE", 1.0, num_bin=2))
    spw = truth.spws[0]
    bdf_paths, time_indices_by_bdf, _refs = partition_of_spw(truth, spw.spw_id)
    with pytest.raises(NotImplementedError, match="numBin"):
        rldf.load_visibilities_from_partition_bdfs(
            bdf_paths, spw.bdf_index, time_indices_by_bdf
        )
    with pytest.raises(NotImplementedError, match="numBin"):
        rldf.load_flags_from_partition_bdfs(
            bdf_paths, spw.bdf_index, time_indices_by_bdf
        )


def test_load_flags_without_flags_component(make_synthetic_asdm, synthetic_asdm_module):
    """A BDF without flags binary component gives all-False flags (of the selected
    shape)."""
    synth = synthetic_asdm_module
    spec = synth.small_dtype_spec("FLOAT32_TYPE", 1.0, name="uid___A002_X88_X1")
    spec.configs[0].with_flags = False
    truth = make_synthetic_asdm(spec)
    spw = truth.spws[0]
    bdf_paths, time_indices_by_bdf, refs = partition_of_spw(truth, spw.spw_id)

    flags = rldf.load_flags_from_partition_bdfs(
        bdf_paths, spw.bdf_index, time_indices_by_bdf, (slice(1, 3), 4, slice(0, 4), 1)
    )
    assert flags.dtype == np.bool_
    assert flags.shape == (2, 4)
    assert not flags.any()
    vis = rldf.load_visibilities_from_partition_bdfs(
        bdf_paths, spw.bdf_index, time_indices_by_bdf
    )
    np.testing.assert_allclose(
        vis, synth.expected_visibility(truth, spw.spw_id, refs), rtol=1e-6, atol=1e-6
    )


def test_load_spw_not_in_bdf(synth_interferometric):
    """An SPW position beyond the BDF SPWs raises, naming the BDF (F57)."""
    truth = synth_interferometric
    bdf_paths, time_indices_by_bdf, _ = partition_of_spw(truth, truth.spws[0].spw_id)
    with pytest.raises(RuntimeError, match="SPW 3 not found in BDF"):
        rldf.load_visibilities_from_partition_bdfs(bdf_paths, 3, time_indices_by_bdf)
    with pytest.raises(RuntimeError, match="SPW 3 not found in BDF"):
        rldf.load_flags_from_partition_bdfs(bdf_paths, 3, time_indices_by_bdf)


def test_load_selection_beyond_bdf_dims(synth_interferometric):
    """Selections beyond the channels of the SPW in the BDF raise (metadata / BDF
    mismatch), naming the BDF."""
    truth = synth_interferometric
    spw = truth.spws[2]  # 4 channels
    bdf_paths, time_indices_by_bdf, _ = partition_of_spw(truth, spw.spw_id)
    with pytest.raises(RuntimeError, match="exceeds the 4 frequency elements"):
        rldf.load_visibilities_from_partition_bdfs(
            bdf_paths,
            spw.bdf_index,
            time_indices_by_bdf,
            (slice(0, 1), slice(0, 10), slice(0, 8), slice(0, 2)),
        )


def test_make_bdf_description_real_headers(synth_single_dish_simple, synth_full_pol):
    """binary_types lists the components actually present (F58)."""
    from xradio.measurement_set._utils._asdm._utils._bdf.bdf_description_checks import (
        expected_data_sizes,
    )

    for truth, expected_types in [
        (synth_single_dish_simple, {"flags", "autoData"}),
        (
            synth_full_pol,
            {"flags", "actualTimes", "actualDurations", "crossData", "autoData"},
        ),
    ]:
        bdf_reader = pyasdm.bdf.BDFReader()
        bdf_reader.open(truth.bdfs[0].path)
        try:
            bdf_descr = rldf.make_bdf_description(bdf_reader.getHeader())
        finally:
            bdf_reader.close()
        assert set(bdf_descr["binary_types"]) == expected_types
        assert set(bdf_descr["sizes"]) == expected_types
        assert set(bdf_descr["axes"]) == expected_types
        expected_sizes = expected_data_sizes(bdf_descr)
        for component in {"crossData", "autoData"} & expected_types:
            assert bdf_descr["sizes"][component] in expected_sizes[component]
        assert "SPP" in bdf_descr["axes"]["autoData"]


def test_make_bdf_description_empty_header():
    bdf_descr = rldf.make_bdf_description(pyasdm.bdf.BDFHeader())
    assert bdf_descr["binary_types"] == []
    assert bdf_descr["sizes"] == {}
    assert bdf_descr["apc"] == []


# ---------------------------------------------------------------------------
# BDF level, with mocked BDF reader
# ---------------------------------------------------------------------------

BDF_KEY = (slice(0, 2), slice(0, 6), slice(0, 8), slice(0, 2))


@pytest.fixture
def mock_reader():
    with mock.patch("pyasdm.bdf.BDFReader") as mock_bdf_reader:
        reader = mock_bdf_reader.return_value
        configure_header_mock(reader.getHeader.return_value)
        yield reader


@pytest.mark.parametrize("never_reshape", [True, False])
def test_load_visibilities_from_bdf(mock_reader, monkeypatch, never_reshape):
    """The per-BDF loader gets the normalized block selection and its result is
    returned as complex64; the reader is closed."""
    loaded = np.arange(2 * 6 * 8 * 2).reshape(2, 6, 8, 2) * (1 + 2j)
    mock_trees = mock.MagicMock(return_value=loaded)
    mock_reshape = mock.MagicMock(return_value=loaded)
    monkeypatch.setattr(rldf, "load_visibilities_all_subsets_from_trees", mock_trees)
    monkeypatch.setattr(rldf, "load_visibilities_all_subsets", mock_reshape)

    vis = rldf.load_visibilities_from_bdf(
        "/foo", 0, (slice(0, 2), None, slice(None), slice(0, 2, 1)), never_reshape
    )
    assert vis.dtype == np.complex64
    np.testing.assert_array_equal(vis, loaded.astype(np.complex64))
    # uneven channels per SPW => always the trees loader
    mock_reshape.assert_not_called()
    (reader, _shape, baseband_spw_idxs, bdf_descr, key) = mock_trees.call_args[0]
    assert reader is mock_reader
    assert baseband_spw_idxs == (0, 0)
    assert key == (slice(0, 2, 1), slice(0, 6, 1), slice(0, 8, 1), slice(0, 2, 1))
    assert bdf_descr["apc_index"] == 0
    assert bdf_descr["binary_types"] == ["flags", "crossData", "autoData"]
    mock_reader.close.assert_called_once()


@pytest.mark.parametrize(
    "loader_effect, expected_error, match",
    [
        (ValueError("bad shape"), RuntimeError, "Error while loading data"),
        (
            pyasdm.exceptions.BDFReaderException("parse error"),
            RuntimeError,
            "Error while loading data",
        ),
        (NotImplementedError("not supported"), NotImplementedError, "not supported"),
        (None, RuntimeError, "Could not load visibilities from BDF /foo"),
        (np.zeros((1, 6, 8, 2)), RuntimeError, r"shape \(1, 6, 8, 2\), expected"),
    ],
)
def test_load_visibilities_from_bdf_errors(
    mock_reader, monkeypatch, loader_effect, expected_error, match
):
    """Loader errors are raised naming the BDF (NotImplementedError unchanged), a
    loader returning None or a wrong shape is an error (F61); the reader is always
    closed (F62)."""
    if isinstance(loader_effect, Exception):
        mock_loader = mock.MagicMock(side_effect=loader_effect)
    else:
        mock_loader = mock.MagicMock(return_value=loader_effect)
    monkeypatch.setattr(rldf, "load_visibilities_all_subsets_from_trees", mock_loader)

    with pytest.raises(expected_error, match=match) as exc_info:
        rldf.load_visibilities_from_bdf("/foo", 0, BDF_KEY)
    if expected_error is RuntimeError and isinstance(loader_effect, Exception):
        assert "/foo" in str(exc_info.value)
        assert exc_info.value.__cause__ is loader_effect
    mock_reader.close.assert_called_once()


@pytest.mark.parametrize(
    "key, expected_error, match",
    [
        ((slice(None), None, None, None), ValueError, "explicit bounds"),
        ((slice(0, None), None, None, None), ValueError, "explicit"),
        ((0, None, None, None), ValueError, "time selection"),
        ((slice(0, 2), slice(0, 6, 2), None, None), ValueError, "baseline selection"),
        ((slice(0, 2), 3, None, None), ValueError, "baseline selection"),
        ((slice(0, 2), slice(0, 7), None, None), RuntimeError, "exceeds the 6"),
        ((slice(0, 2), None, slice(0, 9), None), RuntimeError, "exceeds the 8"),
        ((slice(0, 2), None, None, slice(0, 3)), RuntimeError, "exceeds the 2"),
    ],
)
def test_load_from_bdf_invalid_selection(
    mock_reader, monkeypatch, key, expected_error, match
):
    mock_loader = mock.MagicMock()
    monkeypatch.setattr(rldf, "load_visibilities_all_subsets_from_trees", mock_loader)
    monkeypatch.setattr(rldf, "load_flags_all_subsets", mock_loader)
    with pytest.raises(expected_error, match=match):
        rldf.load_visibilities_from_bdf("/foo", 0, key)
    with pytest.raises(expected_error, match=match):
        rldf.load_flags_from_bdf("/foo", 0, key)
    mock_loader.assert_not_called()
    assert mock_reader.close.call_count == 2


@pytest.mark.parametrize(
    "header_change, match",
    [
        (
            {"correlation_mode": pyasdm.enumerations.CorrelationMode.CROSS_ONLY},
            "correlation mode CROSS_ONLY in BDF /data/bdf_X9 ",
        ),
        ({"basebands": []}, "number of basebands in BDF /data/bdf_X9: "),
    ],
)
@pytest.mark.parametrize(
    "load_function", [rldf.load_visibilities_from_bdf, rldf.load_flags_from_bdf]
)
def test_load_from_bdf_header_errors_name_the_bdf(
    mock_reader, load_function, header_change, match
):
    configure_header_mock(mock_reader.getHeader.return_value, **header_change)
    with pytest.raises(RuntimeError, match=match):
        load_function("/data/bdf_X9", 0, BDF_KEY)
    mock_reader.close.assert_called_once()


def test_load_flags_from_bdf_unsupported_flags_axes(mock_reader):
    mock_reader.getHeader.return_value.getAxesNames.return_value = ["BAL", "HOL"]
    with pytest.raises(RuntimeError, match="HOL.* in BDF /data/bdf_X9"):
        rldf.load_flags_from_bdf("/data/bdf_X9", 0, BDF_KEY)
    mock_reader.close.assert_called_once()


def test_load_visibilities_from_bdf_missing_cross_data(mock_reader, monkeypatch):
    """A CROSS_AND_AUTO BDF without crossData is rejected (F58)."""
    configure_header_mock(mock_reader.getHeader.return_value, present=("autoData",))
    monkeypatch.setattr(rldf, "load_visibilities_all_subsets_from_trees", mock.Mock())
    with pytest.raises(
        RuntimeError, match="does not have the binary component crossData"
    ):
        rldf.load_visibilities_from_bdf("/foo", 0, BDF_KEY)


def test_load_visibilities_from_bdf_auto_only(mock_reader, monkeypatch):
    """An AUTO_ONLY BDF needs only autoData (no crossData, no flags)."""
    configure_header_mock(
        mock_reader.getHeader.return_value,
        correlation_mode=pyasdm.enumerations.CorrelationMode.AUTO_ONLY,
        present=("autoData",),
    )
    loaded = np.ones((2, 3, 8, 2), dtype=np.float32)
    monkeypatch.setattr(
        rldf,
        "load_visibilities_all_subsets_from_trees",
        mock.MagicMock(return_value=loaded),
    )
    vis = rldf.load_visibilities_from_bdf("/foo", 0, (slice(0, 2), None, None, None))
    assert vis.dtype == np.complex64
    np.testing.assert_array_equal(vis, np.ones((2, 3, 8, 2), dtype=np.complex64))


def test_load_visibilities_from_bdf_packed_full_time(mock_reader, monkeypatch):
    """For packed BDFs the number of integrations is known: slice(None) is allowed
    for time."""
    configure_header_mock(
        mock_reader.getHeader.return_value,
        correlation_mode=pyasdm.enumerations.CorrelationMode.AUTO_ONLY,
        present=("autoData",),
        dimensionality=0,
        num_time=4,
    )
    mock_loader = mock.MagicMock(return_value=np.zeros((4, 3, 8, 2)))
    monkeypatch.setattr(rldf, "load_visibilities_all_subsets_from_trees", mock_loader)
    vis = rldf.load_visibilities_from_bdf("/foo", 0, (slice(None),) * 4)
    assert vis.shape == (4, 3, 8, 2)
    assert mock_loader.call_args[0][4][0] == slice(0, 4, 1)
    with pytest.raises(RuntimeError, match="numTime=4"):
        rldf.load_visibilities_from_bdf("/foo", 0, (slice(2, 5), None, None, None))


@pytest.mark.parametrize(
    "never_reshape, expected_loader", [(False, "reshape"), (True, "trees")]
)
def test_load_flags_from_bdf(mock_reader, monkeypatch, never_reshape, expected_loader):
    """Flags loaded per (time, baseline, polarization) are expanded to the selected
    channels; flagged = any bit set (K2)."""
    configure_header_mock(
        mock_reader.getHeader.return_value,
        basebands=[{"name": "BB_1", "spectralWindows": [make_spw(8)]}],
    )
    words = np.array([[[0, 1], [16, 0], [0, 0], [2**30, -(2**31)], [0, 3], [0, 0]]])
    words = np.concatenate([words, words[:, ::-1]])
    mock_trees = mock.MagicMock(return_value=words)
    mock_reshape = mock.MagicMock(return_value=words)
    monkeypatch.setattr(rldf, "load_flags_all_subsets_from_trees", mock_trees)
    monkeypatch.setattr(rldf, "load_flags_all_subsets", mock_reshape)

    key = (slice(0, 2), slice(0, 6), slice(0, 3), slice(0, 2))
    flags = rldf.load_flags_from_bdf("/foo", 0, key, never_reshape)
    assert flags.dtype == np.bool_
    assert flags.shape == (2, 6, 3, 2)
    expected = np.broadcast_to((words != 0)[:, :, np.newaxis, :], (2, 6, 3, 2))
    np.testing.assert_array_equal(flags, expected)
    called, not_called = (
        (mock_reshape, mock_trees)
        if expected_loader == "reshape"
        else (mock_trees, mock_reshape)
    )
    called.assert_called_once()
    not_called.assert_not_called()
    mock_reader.close.assert_called_once()


def test_load_flags_from_bdf_without_flags(mock_reader, monkeypatch):
    configure_header_mock(
        mock_reader.getHeader.return_value, present=("crossData", "autoData")
    )
    mock_loader = mock.MagicMock()
    monkeypatch.setattr(rldf, "load_flags_all_subsets_from_trees", mock_loader)
    monkeypatch.setattr(rldf, "load_flags_all_subsets", mock_loader)
    flags = rldf.load_flags_from_bdf("/foo", 2, (slice(0, 3), None, None, None))
    assert flags.shape == (3, 6, 4, 2)
    assert flags.dtype == np.bool_
    assert not flags.any()
    mock_loader.assert_not_called()
    mock_reader.getSubset.assert_not_called()


@pytest.mark.parametrize(
    "loader_effect, expected_error, match",
    [
        (RuntimeError("from loader"), RuntimeError, "Error while loading flags"),
        (None, RuntimeError, "returned no flags"),
        (np.zeros((2, 5, 2)), RuntimeError, "flags from BDF /foo with shape"),
        (np.zeros((2, 6)), RuntimeError, "Expected flags with dimensions"),
    ],
)
def test_load_flags_from_bdf_errors(
    mock_reader, monkeypatch, loader_effect, expected_error, match
):
    if isinstance(loader_effect, Exception):
        mock_loader = mock.MagicMock(side_effect=loader_effect)
    else:
        mock_loader = mock.MagicMock(return_value=loader_effect)
    monkeypatch.setattr(rldf, "load_flags_all_subsets_from_trees", mock_loader)
    monkeypatch.setattr(rldf, "load_flags_all_subsets", mock_loader)
    with pytest.raises(expected_error, match=match):
        rldf.load_flags_from_bdf("/foo", 0, BDF_KEY)
    mock_reader.close.assert_called_once()


@pytest.mark.parametrize(
    "input_dims, expected_error",
    [
        ("ANT BAB", None),
        ("BAL ANT BAB", None),
        ("BAL ANT BAB SPW", None),
        ("BAL ANT BAB SPW BIN", None),
        ("BAL ANT BAB SPW STO POL", "Unsupported dimension"),
        ("BAB APC SPP POL", "Unsupported dimension"),
        ("BAL ANT BAB BIN STO", "Unsupported dimension"),
        ("BAL ANT BAB SPW HOL", "Unsupported dimension"),
    ],
)
def test_check_flags_dims(input_dims, expected_error):
    if expected_error is None:
        rldf.check_flags_dims(input_dims.split())
    else:
        with pytest.raises(RuntimeError, match=expected_error):
            rldf.check_flags_dims(input_dims.split())


# ---------------------------------------------------------------------------
# Expansion of the flags along frequency
# ---------------------------------------------------------------------------

#: 2 basebands: BB_1 with SPWs of 1024 and 512 channels, BB_2 with 1024 and 1024
bdf_descr_two_bb = {
    "basebands": [
        {"name": "BB_1", "spectralWindows": [make_spw(1024), make_spw(512)]},
        {"name": "BB_2", "spectralWindows": [make_spw(1024), make_spw(1024)]},
    ]
}


@pytest.mark.parametrize(
    "input_flags_shape, input_baseband_idx, input_spw_idx, input_slice, expected_shape",
    [
        ((1, 6, 2), 0, 0, None, (1, 6, 1024, 2)),
        ((1, 6, 2), 0, 0, (slice(None), slice(None), None, 0), (1, 6, 1024, 2)),
        ((1, 6, 2), 0, 0, (slice(None),) * 4, (1, 6, 1024, 2)),
        ((1, 6, 2), 0, 1, (slice(None),) * 4, (1, 6, 512, 2)),
        ((1, 3, 2), 1, 0, (slice(None),) * 4, (1, 3, 1024, 2)),
        ((1, 3, 2), 1, 1, (slice(None), slice(None), 64, slice(None)), (1, 3, 2)),
        ((1, 3, 2), 1, 1, (slice(None), slice(None), slice(32, 33)), (1, 3, 1, 2)),
        ((1, 3, 2), 1, 1, (slice(None), slice(None), slice(32, 33, 1)), (1, 3, 1, 2)),
        ((1, 3, 2), 1, 1, (slice(None), slice(None), slice(16, 46)), (1, 3, 30, 2)),
        ((1, 3, 2), 1, 1, (slice(None), slice(None), slice(16, 46, 2)), (1, 3, 15, 2)),
        # selections starting at channel 0, with or without steps (F21)
        ((1, 3, 2), 1, 1, (slice(None), slice(None), slice(0, 4)), (1, 3, 4, 2)),
        ((1, 3, 2), 1, 1, (slice(None), slice(None), slice(0, 4, 1)), (1, 3, 4, 2)),
        ((7, 3, 2), 0, 1, (slice(None), slice(None), slice(0, 512, 2)), (7, 3, 256, 2)),
        (
            (7, 3, 2),
            0,
            1,
            (slice(None), slice(None), slice(0, None, 3)),
            (7, 3, 171, 2),
        ),
        ((7, 3, 2), 0, 1, (slice(None), slice(None), slice(None, 10)), (7, 3, 10, 2)),
        ((7, 3, 2), 0, 1, (slice(None), slice(None), slice(-10, None)), (7, 3, 10, 2)),
        ((7, 3, 2), 0, 1, (slice(None), slice(None), slice(0, 0)), (7, 3, 0, 2)),
        ((7, 3, 2), 0, 1, (slice(None), slice(None), slice(500, 600)), (7, 3, 12, 2)),
        # flags that already have a frequency dimension
        ((2, 3, 1, 2), 0, 1, (slice(None), slice(None), slice(0, 5)), (2, 3, 5, 2)),
        ((2, 3, 5, 2), 0, 1, (slice(None), slice(None), slice(0, 5)), (2, 3, 5, 2)),
    ],
)
def test__expand_frequency_in_flags_subset(
    input_flags_shape, input_baseband_idx, input_spw_idx, input_slice, expected_shape
):
    rng = np.random.default_rng(1)
    flags = rng.random(input_flags_shape) > 0.5
    expanded = rldf._expand_frequency_in_flags_subset(
        flags,
        bdf_descr_two_bb,
        input_baseband_idx,
        input_spw_idx,
        input_slice,
    )
    assert isinstance(expanded, np.ndarray)
    assert expanded.dtype == "bool"
    assert expanded.shape == expected_shape
    # the same flags for every channel
    if len(expected_shape) == 4 and len(input_flags_shape) == 3:
        for chan in range(expected_shape[2]):
            np.testing.assert_array_equal(expanded[:, :, chan, :], flags)
    elif len(expected_shape) == 3:
        np.testing.assert_array_equal(expanded, flags)


@pytest.mark.parametrize(
    "input_flags_shape, input_slice",
    [
        ((3, 2), (slice(None),) * 4),
        ((3, 2), (slice(None), slice(None), 5)),
        ((2, 3, 4, 2), (slice(None), slice(None), slice(0, 5))),
        ((2, 3, 4, 2, 1), None),
    ],
)
def test__expand_frequency_in_flags_subset_wrong_dims(input_flags_shape, input_slice):
    """Flags without (time, baseline, polarization) dimensions are not guessed
    (F16)."""
    with pytest.raises(ValueError, match="Expected flags with dimensions"):
        rldf._expand_frequency_in_flags_subset(
            np.zeros(input_flags_shape, dtype=bool), bdf_descr_two_bb, 0, 1, input_slice
        )
