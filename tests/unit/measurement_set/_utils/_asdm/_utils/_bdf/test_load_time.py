from unittest import mock

import numpy as np
import pandas as pd

import pytest

import pyasdm


def test_get_times_from_bdfs_empty():
    from xradio.measurement_set._utils._asdm._utils._bdf.load_time import get_times_from_bdfs

    with pytest.raises(ValueError, match="at least one"):
        get_times_from_bdfs([], pd.DataFrame())


def test_get_times_from_bdfs_non_existent():
    from xradio.measurement_set._utils._asdm._utils._bdf.load_time import get_times_from_bdfs

    with pytest.raises(
        pyasdm.exceptions.BDFReaderException, match="No such file or directory"
    ):
        get_times_from_bdfs(["empty-non-existent"], pd.DataFrame())


def test_get_times_from_bdfs():
    from xradio.measurement_set._utils._asdm._utils._bdf.load_time import (
        get_times_from_bdfs,
    )

    with mock.patch("pyasdm.bdf.BDFReader") as mock_bdf_reader:
        mock_bdf_reader.return_value.hasSubset.side_effect = [True, False, True, False]
        subset_times = {
            "midpointInNanoSeconds": 1e10,
            "intervalInNanoSeconds": 1e9,
            "actualTimes": {"present": True, "arr": 10.1e9 * np.ones((100))},
            "actualDurations": {"present": True, "arr": 1.01e9 * np.ones((100))},
        }
        mock_bdf_reader.return_value.getSubset.side_effect = [subset_times] * 2

        bdf_paths = ["/no_path/nonexistant/foo", "/no_path/nonexistant/bar"]
        centers, durations, actual_times, actual_durations = get_times_from_bdfs(
            bdf_paths, pd.DataFrame()
        )
        assert (centers == [10] * 2).all()
        assert (durations == [1] * 2).all()
        assert (actual_times == [10.1] * 2).all()
        assert (actual_durations == [1.01] * 2).all()
        assert mock_bdf_reader.return_value.hasSubset.call_count == 4
        assert mock_bdf_reader.return_value.getSubset.call_count == 2


def test_get_times_from_bdfs_error():
    from xradio.measurement_set._utils._asdm._utils._bdf.load_time import (
        get_times_from_bdfs,
    )

    with mock.patch("pyasdm.bdf.BDFReader") as mock_bdf_reader:
        bdf_reader_exception_msg = "message from BDFReaderException"
        mock_bdf_reader.return_value.hasSubset.side_effect = [
            True,
            False,
            pyasdm.exceptions.BDFReaderException(bdf_reader_exception_msg),
            False,
        ]
        subset_times = {
            "midpointInNanoSeconds": 1e10,
            "intervalInNanoSeconds": 1e9,
            "actualTimes": {"present": True, "arr": 10.1e9 * np.ones((100))},
            "actualDurations": {"present": True, "arr": 1.01e9 * np.ones((100))},
        }
        mock_bdf_reader.return_value.getSubset.side_effect = [subset_times] * 2

        bdf_paths = ["/no_path/nonexistant/foo", "/no_path/nonexistant/bar"]
        with pytest.raises(
            pyasdm.exceptions.BDFReaderException,
            match="message from BDFReaderException",
        ):
            _centers, _durations, _actual_times, _actual_durations = (
                get_times_from_bdfs(bdf_paths, pd.DataFrame())
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
    ] * 2
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


def test_load_times_from_bdfs_empty():
    from xradio.measurement_set._utils._asdm._utils._bdf.load_time import (
        load_times_from_bdfs,
    )

    with pytest.raises(
        pyasdm.exceptions.BDFReaderException, match="No such file or directory"
    ):
        load_times_from_bdfs(["/path/nonexistant/foo", "/path/nonexistant/bar"])


def test_load_times_from_bdfs():
    from xradio.measurement_set._utils._asdm._utils._bdf.load_time import (
        load_times_from_bdfs,
    )

    with mock.patch("pyasdm.bdf.BDFReader") as mock_bdf_reader:
        mock_bdf_reader.return_value.hasSubset.side_effect = [True, False, True, False]
        subset_times = {
            "midpointInNanoSeconds": 1e10,
            "intervalInNanoSeconds": 1e9,
            "actualTimes": {"present": True, "arr": 10.1e9 * np.ones((100))},
            "actualDurations": {"present": True, "arr": 1.01e9 * np.ones((100))},
        }
        mock_bdf_reader.return_value.getSubset.side_effect = [subset_times] * 2

        bdf_paths = ["/no_path/nonexistant/foo", "/no_path/nonexistant/bar"]
        centers, durations, actual_times, actual_durations = load_times_from_bdfs(
            bdf_paths
        )
        assert (centers == [10] * 2).all()
        assert (durations == [1] * 2).all()
        assert (actual_times == [10.1] * 2).all()
        assert (actual_durations == [1.01] * 2).all()
        assert mock_bdf_reader.return_value.hasSubset.call_count == 4
        assert mock_bdf_reader.return_value.getSubset.call_count == 2


def test_load_times_bdf_empty():
    from xradio.measurement_set._utils._asdm._utils._bdf.load_time import load_times_bdf

    with pytest.raises(
        pyasdm.exceptions.BDFReaderException, match="No such file or directory"
    ):
        load_times_bdf("/path/nonexistant/foo")


def test_load_times_bdf_without_actual_times():
    from xradio.measurement_set._utils._asdm._utils._bdf.load_time import load_times_bdf

    with mock.patch("pyasdm.bdf.BDFReader") as mock_bdf_reader:
        mock_bdf_reader.return_value.hasSubset.side_effect = [True, False]
        mock_bdf_reader.return_value.getSubset.side_effect = [
            {
                "midpointInNanoSeconds": 1e10,
                "intervalInNanoSeconds": 1e9,
                "actualTimes": {"present": False, "arr": None},
                "actualDurations": {"present": False, "arr": None},
            },
            None,
        ]

        bdf_path = "/path/nonexistant/foo"
        centers, durations, actual_times, actual_durations = load_times_bdf(bdf_path)
        assert centers == [10]
        assert durations == [1]
        assert actual_times == [10]
        assert actual_durations == [1]
        assert mock_bdf_reader.return_value.hasSubset.call_count == 2
        assert mock_bdf_reader.return_value.getSubset.call_count == 1


def test_load_times_bdf():
    from xradio.measurement_set._utils._asdm._utils._bdf.load_time import load_times_bdf

    with mock.patch("pyasdm.bdf.BDFReader") as mock_bdf_reader:
        mock_bdf_reader.return_value.hasSubset.side_effect = [True, False]
        mock_bdf_reader.return_value.getSubset.side_effect = [
            {
                "midpointInNanoSeconds": 1e10,
                "intervalInNanoSeconds": 1e9,
                "actualTimes": {"present": True, "arr": 10.1e9 * np.ones((100))},
                "actualDurations": {"present": True, "arr": 1.01e9 * np.ones((100))},
            },
            None,
        ]

        bdf_path = "/path/nonexistant/foo"
        centers, durations, actual_times, actual_durations = load_times_bdf(bdf_path)
        assert centers == [10]
        assert durations == [1]
        assert actual_times == [10.1]
        assert actual_durations == [1.01]
        assert mock_bdf_reader.return_value.hasSubset.call_count == 2
        assert mock_bdf_reader.return_value.getSubset.call_count == 1


def test_load_times_bdf_error_old_to_be_removed():
    from xradio.measurement_set._utils._asdm._utils._bdf.load_time import load_times_bdf

    with mock.patch("pyasdm.bdf.BDFReader") as mock_bdf_reader:
        mock_bdf_reader.return_value.hasSubset.side_effect = [True, False]
        mock_bdf_reader.return_value.getSubset.side_effect = [ValueError]

        bdf_path = "/path/nonexistant/foo"
        centers, durations, actual_times, actual_durations = load_times_bdf(bdf_path)
        for time_array in [centers, durations, actual_times, actual_durations]:
            assert time_array == [0]


def test_load_times_bdf_pybdfreader_exception():
    from xradio.measurement_set._utils._asdm._utils._bdf.load_time import load_times_bdf

    with mock.patch("pyasdm.bdf.BDFReader") as mock_bdf_reader:
        mock_bdf_reader.return_value.hasSubset.side_effect = [True, False]
        #mock_bdf_reader.exceptions.BDFReaderException = pyasdm.exceptions.BDFReaderException
        bdf_reader_exception_msg = "msg from BDFReader"
        mock_bdf_reader.return_value.getSubset.side_effect = [
            pyasdm.exceptions.BDFReaderException(bdf_reader_exception_msg)
        ]

        bdf_path = "/path/nonexistant/foo"
        with pytest.raises(RuntimeError, match=bdf_reader_exception_msg):
            _centers, _durations, _actual_times, _actual_durations = load_times_bdf(
                bdf_path
            )
