import os
from unittest import mock

import numpy as np
import pandas as pd
import pyasdm
import pytest

#: 2024-01-01T12:00:00 as ASDM ArrayTime (ns since 1858-11-17)
NS_2024 = (60310 * 86400 + 12 * 3600) * 10**9
#: same instant, seconds since 1970-01-01
UNIX_2024 = 1704110400.0


@pytest.fixture(autouse=True)
def empty_bdf_times_cache():
    from xradio.measurement_set._utils._asdm._utils._bdf.load_time import (
        clear_bdf_times_cache,
    )

    clear_bdf_times_cache()
    yield
    clear_bdf_times_cache()


def make_subset(
    midpoint_ns: int,
    interval_ns: int,
    actual_times=None,
    actual_durations=None,
) -> dict:
    return {
        "midpointInNanoSeconds": midpoint_ns,
        "intervalInNanoSeconds": interval_ns,
        "actualTimes": {
            "present": actual_times is not None,
            "arr": None if actual_times is None else np.asarray(actual_times, "int64"),
        },
        "actualDurations": {
            "present": actual_durations is not None,
            "arr": (
                None
                if actual_durations is None
                else np.asarray(actual_durations, "int64")
            ),
        },
    }


def configure_reader_mock(
    mock_bdf_reader, subsets_per_bdf, dimensionality=1, num_time=0
):
    """
    Make the BDFReader mock return the given subsets (one list per BDF opened, in
    order). Every BDF is expected to be opened once.
    """
    has_subset, get_subset = [], []
    for subsets in subsets_per_bdf:
        has_subset.extend([True] * len(subsets) + [False])
        get_subset.extend(subsets)
    reader = mock_bdf_reader.return_value
    reader.hasSubset.side_effect = has_subset
    reader.getSubset.side_effect = get_subset
    reader.getHeader.return_value.getDimensionality.return_value = dimensionality
    reader.getHeader.return_value.getNumTime.return_value = num_time
    return reader


def test_load_times_from_partition_bdfs_empty():
    from xradio.measurement_set._utils._asdm._utils._bdf.load_time import (
        load_times_from_partition_bdfs,
    )

    with pytest.raises(ValueError, match="at least one BDF"):
        load_times_from_partition_bdfs([], pd.DataFrame())


def test_load_times_from_partition_bdfs_non_existent():
    from xradio.measurement_set._utils._asdm._utils._bdf.load_time import (
        load_times_from_partition_bdfs,
    )

    with pytest.raises(
        RuntimeError, match="Cannot open the BDF empty-non-existent.*No such file"
    ) as exc_info:
        load_times_from_partition_bdfs(["empty-non-existent"], pd.DataFrame())
    # still a pyasdm BDFReaderException, for callers that catch those
    assert isinstance(exc_info.value, pyasdm.exceptions.BDFReaderException)


@pytest.mark.parametrize(
    "contents",
    [b"", b"MIME-Version: 1.0\nContent-Type: garbage\n"],
    ids=["empty", "corrupt_header"],
)
def test_load_times_from_partition_bdfs_broken_bdf(
    synth_interferometric, tmp_path, contents
):
    """A BDF of the partition that is empty or has a corrupt header raises a
    RuntimeError that names that BDF (not just the partition). The failure is not
    cached."""
    import shutil

    from xradio.measurement_set._utils._asdm._utils._bdf.load_time import (
        load_times_from_partition_bdfs,
    )

    good_bdfs = [bdf.path for bdf in synth_interferometric.bdfs[:2]]
    bdf_paths = [str(tmp_path / f"bdf_{idx}") for idx in range(2)]
    shutil.copyfile(good_bdfs[0], bdf_paths[0])
    with open(bdf_paths[1], "wb") as bdf_file:
        bdf_file.write(contents)

    with pytest.raises(RuntimeError) as exc_info:
        load_times_from_partition_bdfs(bdf_paths)
    assert f"BDF {bdf_paths[1]} " in str(exc_info.value)
    assert bdf_paths[0] not in str(exc_info.value)

    shutil.copyfile(good_bdfs[1], bdf_paths[1])
    *times, time_indices_by_bdf = load_times_from_partition_bdfs(bdf_paths)
    expected = [load_times_from_partition_bdfs(good_bdfs)[idx] for idx in range(4)]
    for loaded, expected_times in zip(times, expected, strict=True):
        np.testing.assert_array_equal(loaded, expected_times)
    assert time_indices_by_bdf["bdf_names"] == bdf_paths


def test_load_times_from_partition_bdfs():
    """Absolute times are converted to unix seconds from the integer ns (subtracting
    the MJD-unix epoch difference), durations are only converted to seconds."""
    from xradio.measurement_set._utils._asdm._utils._bdf.load_time import (
        load_times_from_partition_bdfs,
    )

    interval = 1_008_000_000
    with mock.patch("pyasdm.bdf.BDFReader") as mock_bdf_reader:
        reader = configure_reader_mock(
            mock_bdf_reader,
            [
                [
                    make_subset(NS_2024, interval),
                    make_subset(NS_2024 + interval, interval),
                ],
                [
                    make_subset(
                        NS_2024 + 10 * interval,
                        interval,
                        actual_times=[NS_2024 + 10 * interval + 10_000] * 3,
                        actual_durations=[interval - 2_000] * 3,
                    )
                ],
            ],
        )

        bdf_paths = np.array(["/no_path/nonexistant/foo", "/no_path/nonexistant/bar"])
        centers, durations, actual_times, actual_durations, time_indices_by_bdf = (
            load_times_from_partition_bdfs(bdf_paths, pd.DataFrame())
        )

    expected_centers = UNIX_2024 + np.array([0, 1, 10]) * interval / 1e9
    np.testing.assert_allclose(centers, expected_centers, rtol=0, atol=1e-6)
    np.testing.assert_allclose(durations, [interval / 1e9] * 3, rtol=1e-12)
    np.testing.assert_allclose(
        actual_times, expected_centers + [0, 0, 1e-5], rtol=0, atol=5e-7
    )
    np.testing.assert_allclose(
        actual_durations, [1.008, 1.008, 1.008 - 2e-6], rtol=1e-12
    )
    for times in centers, durations, actual_times, actual_durations:
        assert times.dtype == np.float64
    assert time_indices_by_bdf == {
        "bdf_names": ["/no_path/nonexistant/foo", "/no_path/nonexistant/bar"],
        "bdf_start": [0, 2, 3],
    }
    # Every BDF opened once (F17), and its header read once
    assert reader.open.call_count == 2
    assert reader.getHeader.call_count == 2
    assert reader.close.call_count == 2
    assert reader.getSubset.call_count == 3
    for call in reader.getSubset.call_args_list:
        assert call.kwargs["loadOnlyComponents"] == {"actualTimes", "actualDurations"}


@pytest.mark.parametrize(
    "error",
    [
        pyasdm.exceptions.BDFReaderException("message from BDFReaderException"),
        ValueError("message from ValueError"),
    ],
)
def test_load_times_from_partition_bdfs_error(error):
    """Errors reading a subset are raised naming the BDF, there is no fallback to
    zeros or to other times (F61)."""
    from xradio.measurement_set._utils._asdm._utils._bdf.load_time import (
        load_times_from_partition_bdfs,
    )

    with mock.patch("pyasdm.bdf.BDFReader") as mock_bdf_reader:
        reader = mock_bdf_reader.return_value
        reader.getHeader.return_value.getDimensionality.return_value = 1
        reader.hasSubset.side_effect = [True, True]
        reader.getSubset.side_effect = [make_subset(NS_2024, 10**9), error]

        bdf_paths = ["/no_path/nonexistant/foo", "/no_path/nonexistant/bar"]
        with pytest.raises(RuntimeError, match=str(error)) as exc_info:
            load_times_from_partition_bdfs(bdf_paths, pd.DataFrame())

    assert "/no_path/nonexistant/foo" in str(exc_info.value)
    assert "subset 1" in str(exc_info.value)
    assert exc_info.value.__cause__ is error
    reader.close.assert_called_once()


def test_load_times_from_partition_bdfs_error_has_subset():
    from xradio.measurement_set._utils._asdm._utils._bdf.load_time import (
        load_times_from_partition_bdfs,
    )

    with mock.patch("pyasdm.bdf.BDFReader") as mock_bdf_reader:
        reader = mock_bdf_reader.return_value
        reader.getHeader.return_value.getDimensionality.return_value = 1
        reader.hasSubset.side_effect = pyasdm.exceptions.BDFReaderException(
            "message from BDFReaderException"
        )

        with pytest.raises(RuntimeError, match="message from BDFReaderException"):
            load_times_from_partition_bdfs(["/no_path/foo"], pd.DataFrame())
    reader.close.assert_called_once()


def test_load_times_bdf_packed():
    """A packed BDF (dimensionality 0, numTime=N, one subset) gives N integrations
    splitting the subset interval in N equal parts (F19, K5)."""
    from xradio.measurement_set._utils._asdm._utils._bdf.load_time import (
        load_times_bdf,
    )

    num_time = 4
    tim_interval = 576_000_000
    subset_interval = num_time * tim_interval
    subset_midpoint = NS_2024 + subset_interval // 2
    with mock.patch("pyasdm.bdf.BDFReader") as mock_bdf_reader:
        configure_reader_mock(
            mock_bdf_reader,
            [[make_subset(subset_midpoint, subset_interval)]],
            dimensionality=0,
            num_time=num_time,
        )
        centers, durations, actual_times, actual_durations = load_times_bdf("/foo")

    expected_centers = (
        UNIX_2024 + (np.arange(num_time) + 0.5) * tim_interval / 1e9
    )  # first TIM sample centered half a sample after the subset start
    np.testing.assert_allclose(centers, expected_centers, rtol=0, atol=1e-6)
    np.testing.assert_allclose(actual_times, expected_centers, rtol=0, atol=1e-6)
    np.testing.assert_allclose(durations, [tim_interval / 1e9] * num_time)
    np.testing.assert_allclose(actual_durations, [tim_interval / 1e9] * num_time)


def test_load_times_bdf_packed_with_actual_times():
    """Per-TIM actualTimes / actualDurations (axes TIM ANT) of a packed BDF."""
    from xradio.measurement_set._utils._asdm._utils._bdf.load_time import (
        load_times_bdf,
    )

    num_time, num_antenna = 3, 2
    tim_interval = 1_000_000_000
    subset_interval = num_time * tim_interval
    tim_actual = NS_2024 + np.array([510, 1_490, 2_520]) * 1_000_000
    actual_durations = np.array([990, 980, 970]) * 1_000_000
    with mock.patch("pyasdm.bdf.BDFReader") as mock_bdf_reader:
        configure_reader_mock(
            mock_bdf_reader,
            [
                [
                    make_subset(
                        NS_2024 + subset_interval // 2,
                        subset_interval,
                        actual_times=np.repeat(tim_actual, num_antenna),
                        actual_durations=np.repeat(actual_durations, num_antenna),
                    )
                ]
            ],
            dimensionality=0,
            num_time=num_time,
        )
        centers, durations, actual_times, actual_durations_s = load_times_bdf("/foo")

    np.testing.assert_allclose(
        centers, UNIX_2024 + np.array([0.5, 1.5, 2.5]), rtol=0, atol=1e-6
    )
    np.testing.assert_allclose(
        actual_times, UNIX_2024 + np.array([0.51, 1.49, 2.52]), rtol=0, atol=1e-6
    )
    np.testing.assert_allclose(durations, [1.0] * 3)
    np.testing.assert_allclose(actual_durations_s, [0.99, 0.98, 0.97])


def test_load_times_bdf_packed_invalid_num_time():
    from xradio.measurement_set._utils._asdm._utils._bdf.load_time import (
        load_times_bdf,
    )

    with mock.patch("pyasdm.bdf.BDFReader") as mock_bdf_reader:
        reader = configure_reader_mock(
            mock_bdf_reader,
            [[make_subset(NS_2024, 10**9)]],
            dimensionality=0,
            num_time=0,
        )
        with pytest.raises(RuntimeError, match="numTime=0"):
            load_times_bdf("/foo")
    reader.close.assert_called_once()


def test_load_times_bdf_implausible_actual_times():
    """actualTimes far from the midpoint (e.g. zeros) are not used."""
    from xradio.measurement_set._utils._asdm._utils._bdf.load_time import (
        load_times_bdf,
    )

    with mock.patch("pyasdm.bdf.BDFReader") as mock_bdf_reader:
        configure_reader_mock(
            mock_bdf_reader,
            [
                [
                    make_subset(
                        NS_2024, 10**9, actual_times=[0, 0], actual_durations=[0, 0]
                    ),
                    make_subset(
                        NS_2024 + 10**9,
                        10**9,
                        actual_times=[NS_2024 + 10**9 + 10_000] * 2,
                        actual_durations=[10**9 - 100] * 2,
                    ),
                ]
            ],
        )
        _centers, _durations, actual_times, actual_durations = load_times_bdf("/foo")

    np.testing.assert_allclose(
        actual_times, [UNIX_2024, UNIX_2024 + 1 + 1e-5], rtol=0, atol=5e-7
    )
    np.testing.assert_allclose(actual_durations, [1.0, 1 - 1e-7], rtol=1e-12)


def test_load_times_bdf_no_subsets():
    from xradio.measurement_set._utils._asdm._utils._bdf.load_time import (
        load_times_bdf,
    )

    with mock.patch("pyasdm.bdf.BDFReader") as mock_bdf_reader:
        configure_reader_mock(mock_bdf_reader, [[]])
        times = load_times_bdf("/foo")

    assert all(arr.shape == (0,) for arr in times)


def test_load_times_bdf_empty():
    from xradio.measurement_set._utils._asdm._utils._bdf.load_time import (
        load_times_bdf,
    )

    with pytest.raises(
        RuntimeError, match="Cannot open the BDF /path/nonexistant/foo.*No such file"
    ):
        load_times_bdf("/path/nonexistant/foo")


def test_load_times_bdf_open_error_closes_reader():
    """When the header parse fails, the reader (file handle) is closed and the
    error names the BDF."""
    from xradio.measurement_set._utils._asdm._utils._bdf.load_time import (
        load_times_bdf,
    )

    with mock.patch("pyasdm.bdf.BDFReader") as mock_bdf_reader:
        reader = mock_bdf_reader.return_value
        reader.open.side_effect = pyasdm.exceptions.BDFReaderException(
            "count not found a boundary definition in 'garbage'."
        )
        with pytest.raises(RuntimeError, match="BDF /some/bdf_X1 .*boundary"):
            load_times_bdf("/some/bdf_X1")
    reader.close.assert_called_once()
    reader.getSubset.assert_not_called()


def test_load_times_bdf_cached(tmp_path):
    """The times of a BDF file are read once and reused (F17), until the file
    changes."""
    from xradio.measurement_set._utils._asdm._utils._bdf import load_time

    bdf_path = tmp_path / "bdf_X1"
    bdf_path.write_bytes(b"0")
    subsets = [make_subset(NS_2024, 10**9), make_subset(NS_2024 + 10**9, 10**9)]
    with mock.patch("pyasdm.bdf.BDFReader") as mock_bdf_reader:
        reader = configure_reader_mock(mock_bdf_reader, [subsets, subsets])

        first = load_time.load_times_from_bdfs([str(bdf_path)])
        first[0][:] = 0.0  # callers get copies
        second = load_time.load_times_from_bdfs([str(bdf_path)])
        assert reader.open.call_count == 1
        assert reader.getSubset.call_count == 2
        np.testing.assert_allclose(
            second[0], [UNIX_2024, UNIX_2024 + 1], rtol=0, atol=1e-6
        )
        assert second[4] == {"bdf_names": [str(bdf_path)], "bdf_start": [0, 2]}

        # modified file => read again
        bdf_path.write_bytes(b"01")
        third = load_time.load_times_from_bdfs([str(bdf_path)])
        assert reader.open.call_count == 2
        np.testing.assert_array_equal(third[0], second[0])


def test_load_times_from_synthetic_bdfs(synth_interleaved, synthetic_asdm_module):
    """Times from real BDFs (non-packed, and packed WVR with numTime=4) equal the
    true integration times and durations (F02, F19, K1, K5)."""
    from xradio.measurement_set._utils._asdm._utils._bdf.load_time import (
        load_times_from_partition_bdfs,
    )

    synth = synthetic_asdm_module
    truth = synth_interleaved
    for spw in truth.spws:
        refs = truth.integrations(spw.spw_id)
        bdf_paths = list(dict.fromkeys(ref.bdf.path for ref in refs))
        centers, durations, actual_times, actual_durations, time_indices_by_bdf = (
            load_times_from_partition_bdfs(bdf_paths)
        )
        expected_times = synth.expected_times(refs)
        expected_intervals = synth.expected_intervals(refs)
        np.testing.assert_allclose(centers, expected_times, rtol=0, atol=1e-6)
        np.testing.assert_allclose(actual_times, expected_times, rtol=0, atol=1e-6)
        np.testing.assert_allclose(durations, expected_intervals, rtol=1e-12)
        np.testing.assert_allclose(actual_durations, expected_intervals, rtol=1e-12)
        expected_counts = [
            sum(1 for ref in refs if ref.bdf.path == path) for path in bdf_paths
        ]
        assert np.diff(time_indices_by_bdf["bdf_start"]).tolist() == expected_counts
        assert time_indices_by_bdf["bdf_names"] == bdf_paths

    # the WVR SPW has packed BDFs with 4 integrations each
    wvr_refs = truth.integrations(truth.spws[0].spw_id)
    assert all(ref.bdf.packed for ref in wvr_refs)
    assert len(wvr_refs) == 4 * len({ref.bdf.path for ref in wvr_refs})


def test_load_times_from_synthetic_bdfs_with_actual_times(
    synth_full_pol, synthetic_asdm_module
):
    from xradio.measurement_set._utils._asdm._utils._bdf.load_time import (
        load_times_from_partition_bdfs,
    )

    truth = synth_full_pol
    refs = truth.integrations(truth.spws[0].spw_id)
    bdf_paths = list(dict.fromkeys(ref.bdf.path for ref in refs))
    centers, _, actual_times, actual_durations, _ = load_times_from_partition_bdfs(
        bdf_paths
    )
    expected_times = synthetic_asdm_module.expected_times(refs)
    np.testing.assert_allclose(centers, expected_times, rtol=0, atol=1e-6)
    np.testing.assert_allclose(actual_times, expected_times, rtol=0, atol=1e-6)
    np.testing.assert_allclose(
        actual_durations, synthetic_asdm_module.expected_intervals(refs)
    )


basebands_one_only = [
    {
        "name": "BB_1",
        "spectralWindows": [
            {
                "crossPolProducts": [],
                "sdPolProducts": [],
                "scaleFactor": 103107.95,
                "numSpectralPoint": 960,
                "numBin": 1,
                "sideband": None,
                "sw": "1",
            }
        ],
    },
]


def make_sufficient_bdf_header_mock(mock_bdf_header):
    mock_bdf_header.getDimensionality.return_value = 1
    mock_bdf_header.getNumTime.return_value = 1
    mock_bdf_header.getProcessorType.return_value = (
        pyasdm.enumerations.ProcessorType.CORRELATOR
    )
    mock_bdf_header.getBinaryTypes.return_value = [
        "flags",
        "actualTimes",
        "actualDurations",
        "zeroLags",
        "crossData",
        "autoData",
    ]
    mock_bdf_header.getCorrelationMode.return_value = (
        pyasdm.enumerations.CorrelationMode.CROSS_AND_AUTO
    )
    mock_bdf_header.getAPClist.return_value = []
    mock_bdf_header.getNumAntenna.return_value = 9
    mock_bdf_header.getBasebandsList.return_value = basebands_one_only


def test_make_blob_info_empty():
    from xradio.measurement_set._utils._asdm._utils._bdf.load_time import make_blob_info

    info = make_blob_info(pyasdm.bdf.BDFHeader())
    assert isinstance(info, pd.DataFrame)
    assert info.shape == (1, 22)


def test_make_blob_info():
    from xradio.measurement_set._utils._asdm._utils._bdf.load_time import make_blob_info

    with mock.patch("pyasdm.bdf.BDFHeader") as mock_bdf_header:
        make_sufficient_bdf_header_mock(mock_bdf_header)

        info = make_blob_info(mock_bdf_header)

        assert isinstance(info, pd.DataFrame)
        assert info.shape == (1, 22)
        assert info["num_antenna"].iloc[0] == 9
        assert info["basebands_spws_points_bins_crossx_sdx"].iloc[0] == (
            "BB_1 spw_1 960 1 0 0"
        )


def test_save_blob_info(tmp_path):
    from xradio.measurement_set._utils._asdm._utils._bdf.load_time import (
        save_blob_info,
    )

    csv_path = tmp_path / "blob_info.csv"
    blob_info = pd.DataFrame({"foo": [1], "bar": ["a"]}).set_index("foo")
    save_blob_info(str(csv_path), blob_info)
    save_blob_info(str(csv_path), blob_info)

    lines = csv_path.read_text().splitlines()
    assert lines == ["foo,bar", "1,a", "1,a"]


def test_load_times_bdf_save_blob_info(tmp_path, monkeypatch):
    """With config.do_save_blob_info the header info is saved, from the same reader
    (the BDF is still opened once)."""
    from xradio.measurement_set._utils._asdm._utils._bdf import config, load_time

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(config, "do_save_blob_info", True)
    with (
        mock.patch("pyasdm.bdf.BDFReader") as mock_bdf_reader,
        mock.patch.object(
            load_time,
            "make_blob_info",
            return_value=pd.DataFrame({"foo": [1]}).set_index("foo"),
        ) as mock_make_blob_info,
    ):
        reader = configure_reader_mock(mock_bdf_reader, [[make_subset(NS_2024, 1)]])
        load_time.load_times_bdf("/foo")

    mock_make_blob_info.assert_called_once_with(reader.getHeader.return_value)
    assert reader.open.call_count == 1
    assert os.path.isfile(tmp_path / load_time.BLOB_INFO_CSV_PATH)
