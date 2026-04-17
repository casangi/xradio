from contextlib import nullcontext as no_raises
from unittest import mock

import pytest

import numpy as np

import pyasdm


def test_array_slice_to_msv4_indices():
    from xradio.measurement_set._utils._asdm._utils._bdf.robust_load_data_flags import (
        array_slice_to_msv4_indices,
    )

    slice_def = {
        "time": slice(0, 1),
        "baseline": slice(0, 9),
        "frequency": slice(0, 0.1),
        "polarization": slice(0, 1),
    }
    indices = array_slice_to_msv4_indices(slice_def)
    assert indices == (
        slice_def["time"],
        slice_def["baseline"],
        slice_def["frequency"],
        slice_def["polarization"],
    )


# From uid___A002_Xfd764e_X2197
basebands_example_X2197 = [
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
    {
        "name": "BB_2",
        "spectralWindows": [
            {
                "crossPolProducts": [],
                "sdPolProducts": [],
                "scaleFactor": 103107.95,
                "numSpectralPoint": 960,
                "numBin": 1,
                "sideband": None,
                "sw": "2",
            }
        ],
    },
    {
        "name": "BB_3",
        "spectralWindows": [
            {
                "crossPolProducts": [],
                "sdPolProducts": [],
                "scaleFactor": 36454.168,
                "numSpectralPoint": 480,
                "numBin": 1,
                "sideband": None,
                "sw": "3",
            },
            {
                "crossPolProducts": [],
                "sdPolProducts": [],
                "scaleFactor": 36454.168,
                "numSpectralPoint": 480,
                "numBin": 1,
                "sideband": None,
                "sw": "4",
            },
        ],
    },
    {
        "name": "BB_4",
        "spectralWindows": [
            {
                "crossPolProducts": [],
                "sdPolProducts": [],
                "scaleFactor": 103107.95,
                "numSpectralPoint": 960,
                "numBin": 1,
                "sideband": None,
                "sw": "5",
            }
        ],
    },
]


# Will likely also need a ProcessorType.RADIOMETER/SPECTROMETER
# From uid___A002_Xc33ac1_X136e (AUTO_ONLY)
bdf_descr_X136e = {
    "dimensionality": 1,
    "num_time": 0,
    "processor_type": pyasdm.enumerations.ProcessorType.CORRELATOR,
    "binary_types": [
        "flags",
        "actualTimes",
        "actualDurations",
        "zeroLags",
        "crossData",
        "autoData",
    ],
    "correlation_mode": pyasdm.enumerations.CorrelationMode.AUTO_ONLY,
    "apc": [],
    "num_antenna": 10,
    "basebands": [
        {
            "name": "BB_1",
            "spectralWindows": [
                {
                    "crossPolProducts": [
                        pyasdm.enumerations.StokesParameter.XX,
                        pyasdm.enumerations.StokesParameter.YY,
                    ],
                    "sdPolProducts": [
                        pyasdm.enumerations.StokesParameter.XX,
                        pyasdm.enumerations.StokesParameter.YY,
                    ],
                    "scaleFactor": None,
                    "numSpectralPoint": 1024,
                    "numBin": 1,
                    "sideband": pyasdm.enumerations.NetSideband.LSB,
                    "sw": "1",
                },
                {
                    "crossPolProducts": [],
                    "sdPolProducts": [
                        pyasdm.enumerations.StokesParameter.XX,
                        pyasdm.enumerations.StokesParameter.YY,
                    ],
                    "scaleFactor": None,
                    "numSpectralPoint": 512,
                    "numBin": 1,
                    "sideband": pyasdm.enumerations.NetSideband.USB,
                    "sw": "2",
                },
            ],
        },
        {
            "name": "BB_2",
            "spectralWindows": [
                {
                    "crossPolProducts": [],
                    "sdPolProducts": [
                        pyasdm.enumerations.StokesParameter.XX,
                        pyasdm.enumerations.StokesParameter.YY,
                    ],
                    "scaleFactor": None,
                    "numSpectralPoint": 1024,
                    "numBin": 1,
                    "sideband": pyasdm.enumerations.NetSideband.LSB,
                    "sw": "1",
                },
                {
                    "crossPolProducts": [],
                    "sdPolProducts": [
                        pyasdm.enumerations.StokesParameter.XX,
                        pyasdm.enumerations.StokesParameter.YY,
                    ],
                    "scaleFactor": None,
                    "numSpectralPoint": 1024,
                    "numBin": 1,
                    "sideband": pyasdm.enumerations.NetSideband.USB,
                    "sw": "2",
                },
            ],
        },
        {
            "name": "BB_3",
            "spectralWindows": [
                {
                    "crossPolProducts": [],
                    "sdPolProducts": [
                        pyasdm.enumerations.StokesParameter.XX,
                        pyasdm.enumerations.StokesParameter.YY,
                    ],
                    "scaleFactor": None,
                    "numSpectralPoint": 2048,
                    "numBin": 1,
                    "sideband": pyasdm.enumerations.NetSideband.LSB,
                    "sw": "1",
                }
            ],
        },
        {
            "name": "BB_4",
            "spectralWindows": [
                {
                    "crossPolProducts": [],
                    "sdPolProducts": [
                        pyasdm.enumerations.StokesParameter.XX,
                        pyasdm.enumerations.StokesParameter.YY,
                    ],
                    "scaleFactor": None,
                    "numSpectralPoint": 1024,
                    "numBin": 1,
                    "sideband": pyasdm.enumerations.NetSideband.USB,
                    "sw": "1",
                }
            ],
        },
    ],
}


def test_load_visibilities_from_partition_bdfs_empty():
    from xradio.measurement_set._utils._asdm._utils._bdf.robust_load_data_flags import (
        load_visibilities_from_partition_bdfs,
    )

    bdf_paths = []
    with pytest.raises(ValueError, match="at least one array"):
        load_visibilities_from_partition_bdfs(bdf_paths, 0, {})


def test_load_visibilities_from_partition_bdfs_inexistent():
    from xradio.measurement_set._utils._asdm._utils._bdf.robust_load_data_flags import (
        load_visibilities_from_partition_bdfs,
    )

    with (
        mock.patch("pyasdm.bdf.BDFReader") as mock_bdf_reader,
        mock.patch("pyasdm.bdf.BDFHeader") as mock_bdf_header,
    ):
        mock_bdf_reader.return_value.hasSubset.side_effect = [True, True, False]
        subset_shape = (1, 9, 64, 2)
        mock_bdf_reader.return_value.getNDArrays.side_effect = [
            {
                "visibilities": np.zeros(subset_shape, dtype="complex128"),
                # Will be needed when not using loadOneSPWFunction, etc.
                # "autoData": {"present": True, "arr": np.zeros((71680), dtype="complex128")},
            },
        ] * 2

        make_sufficient_bdf_header_mock(mock_bdf_header)
        mock_bdf_reader.return_value.getHeader.return_value = mock_bdf_header

        bdf_paths = ["/inexistent_path/foo/"]
        visibilities = load_visibilities_from_partition_bdfs(bdf_paths, 0, {})
        assert isinstance(visibilities, np.ndarray)
        assert visibilities.dtype == "complex128"
        assert visibilities.shape == (2, *subset_shape[1:])

        mock_bdf_header.getBasebandsList.assert_called_once()
        assert mock_bdf_reader.return_value.hasSubset.call_count == 3
        assert mock_bdf_reader.return_value.getNDArrays.call_count == 2


@pytest.mark.parametrize(
    "input_never_reshape, input_spw", [(True, 0), (True, 1), (False, 0), (False, 1)]
)
def test_load_visibilities_from_bdf_incomplete_descr(input_never_reshape, input_spw):
    from xradio.measurement_set._utils._asdm._utils._bdf.robust_load_data_flags import (
        load_visibilities_from_bdf,
    )

    with (
        mock.patch("pyasdm.bdf.BDFReader") as mock_bdf_reader,
        mock.patch("pyasdm.bdf.BDFHeader") as mock_bdf_header,
    ):
        # mock_bdf_reader.
        mock_bdf_reader.hasSubset.side_effect = [True, False]
        mock_bdf_reader.getSubset.side_effect = {}
        mock_bdf_reader.getNDArrays.side_effect = {}
        mock_bdf_header.getBasebandsList.side_effect = ["foo", "bar"]
        with pytest.raises(RuntimeError, match="basebands"):
            load_visibilities_from_bdf(
                "/inexistent/foo/path/",
                input_spw,
                {},
                never_reshape_from_all_spws=input_never_reshape,
            )
        mock_bdf_header.getBasebandsList.assert_not_called()
        mock_bdf_reader.hasSubset.assert_not_called()
        mock_bdf_reader.getSubset.assert_not_called()
        mock_bdf_reader.getNDArrays.assert_not_called()


@pytest.mark.parametrize(
    "input_never_reshape, input_spw", [(True, 0), (True, 2), (False, 1), (False, 4)]
)
def test_load_visibilities_from_bdf_error_loading(input_never_reshape, input_spw):
    from xradio.measurement_set._utils._asdm._utils._bdf.robust_load_data_flags import (
        load_visibilities_from_bdf,
    )

    with (
        mock.patch("pyasdm.bdf.BDFReader") as mock_bdf_reader,
        mock.patch("pyasdm.bdf.BDFHeader") as mock_bdf_header,
    ):
        mock_bdf_reader.return_value.hasSubset.side_effect = [True, True, False]
        # Force some error loading
        subset_shape = (1, 3, 64, 2)
        mock_bdf_reader.return_value.getNDArrays.side_effect = [
            {
                "visibilities": np.zeros(subset_shape, dtype="complex128"),
            },
            RuntimeError,
        ]

        make_sufficient_bdf_header_mock(mock_bdf_header)
        mock_bdf_reader.return_value.getHeader.return_value = mock_bdf_header
        with pytest.raises(RuntimeError, match="Error while loading data/visibilities"):
            load_visibilities_from_bdf(
                "/inexistent/foo/path/",
                input_spw,
                {},
                never_reshape_from_all_spws=input_never_reshape,
            )

        mock_bdf_header.getBasebandsList.assert_called_once()
        assert mock_bdf_reader.return_value.hasSubset.call_count == 2
        assert mock_bdf_reader.return_value.getNDArrays.call_count == 2
        assert mock_bdf_reader.return_value.getSubset.call_count == 0


def test_load_flags_from_partition_bdfs_empty():
    from xradio.measurement_set._utils._asdm._utils._bdf.robust_load_data_flags import (
        load_flags_from_partition_bdfs,
    )

    bdf_paths = []
    with pytest.raises(ValueError, match="at least one array"):
        load_flags_from_partition_bdfs(bdf_paths, 0, {})


def test_load_flags_from_partition_bdfs_inexistent():
    from xradio.measurement_set._utils._asdm._utils._bdf.robust_load_data_flags import (
        load_flags_from_partition_bdfs,
    )

    bdf_paths = ["/inexistent_path_to_flags/foo/"]
    with pytest.raises(
        pyasdm.exceptions.BDFReaderException, match="Error while opening"
    ):
        load_flags_from_partition_bdfs(bdf_paths, 0, {})


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
    mock_bdf_header.getBasebandsList.return_value = bdf_descr_X136e["basebands"]


def test_load_flags_from_partition_bdfs():
    from xradio.measurement_set._utils._asdm._utils._bdf.robust_load_data_flags import (
        load_flags_from_partition_bdfs,
    )

    with (
        mock.patch("pyasdm.bdf.BDFHeader") as mock_bdf_header,
        mock.patch("pyasdm.bdf.BDFReader") as mock_bdf_reader,
    ):
        mock_bdf_reader.return_value.hasSubset.side_effect = [True, False]
        mock_bdf_reader.return_value.getSubset.side_effect = [{}, {}]

        make_sufficient_bdf_header_mock(mock_bdf_header)
        mock_bdf_reader.return_value.getHeader.return_value = mock_bdf_header

        bdf_paths = ["/inexistent_path_to_flags/foo/"]
        empty_slice = (slice(None), slice(None), slice(None), slice(None))
        flags = load_flags_from_partition_bdfs(bdf_paths, 0, empty_slice)
        assert isinstance(flags, np.ndarray)
        assert flags.dtype == "bool"
        assert flags.shape == (1, 45, 1024, 2)


@pytest.mark.parametrize(
    "input_dims, expected_error",
    [
        ("ANT BAB", no_raises()),
        ("BAL ANT BAB", no_raises()),
        ("BAL ANT BAB SPW", no_raises()),
        ("BAL ANT BAB SPW BIN", no_raises()),
        (
            "BAL ANT BAB SPW STO POL",
            pytest.raises(RuntimeError, match="Unsupported dimension"),
        ),
        ("BAB APC SPP POL", pytest.raises(RuntimeError, match="Unsupported dimension")),
        (
            "BAL ANT BAB BIN STO",
            pytest.raises(RuntimeError, match="Unsupported dimension"),
        ),
        (
            "BAL ANT BAB SPW HOL",
            pytest.raises(RuntimeError, match="Unsupported dimension"),
        ),
    ],
)
def test_check_flags_dims(input_dims, expected_error):
    from xradio.measurement_set._utils._asdm._utils._bdf.robust_load_data_flags import (
        check_flags_dims,
    )

    with expected_error:
        check_flags_dims(input_dims)


@pytest.mark.parametrize(
    "input_never_reshape, input_spw, expected_shape",
    [
        (True, 2, (1, 45, 1024, 0)),
        (True, 4, (1, 45, 2048, 0)),
        (False, 0, (1, 45, 1024, 2)),
        (False, 1, (1, 45, 512, 0)),
    ],
)
def test_load_flags_from_bdf(input_never_reshape, input_spw, expected_shape):
    from xradio.measurement_set._utils._asdm._utils._bdf.robust_load_data_flags import (
        load_flags_from_bdf,
    )

    bdf_path = "/inexistent_path_to_bdf/foo/"
    with (
        mock.patch("pyasdm.bdf.BDFHeader") as mock_bdf_header,
        mock.patch("pyasdm.bdf.BDFReader") as mock_bdf_reader,
    ):
        mock_bdf_reader.return_value.hasSubset.side_effect = [True, False]
        mock_bdf_reader.return_value.getSubset.side_effect = [{}, {}]

        make_sufficient_bdf_header_mock(mock_bdf_header)
        mock_bdf_reader.return_value.getHeader.return_value = mock_bdf_header

        empty_slice = (slice(None), slice(None), slice(None), slice(None))
        flags = load_flags_from_bdf(
            bdf_path, input_spw, empty_slice, input_never_reshape
        )
        assert isinstance(flags, np.ndarray)
        assert flags.dtype == "bool"
        assert flags.shape == expected_shape


def test_load_flags_from_bdf_with_error():
    from xradio.measurement_set._utils._asdm._utils._bdf.robust_load_data_flags import (
        load_flags_from_bdf,
    )

    bdf_path = "/inexistent_path_to_bdf/foo/"
    with (
        mock.patch("pyasdm.bdf.BDFHeader") as mock_bdf_header,
        mock.patch("pyasdm.bdf.BDFReader") as mock_bdf_reader,
    ):
        mock_bdf_reader.return_value.hasSubset.side_effect = [True, False]
        mock_bdf_reader.return_value.getSubset.side_effect = [RuntimeError, {}]

        make_sufficient_bdf_header_mock(mock_bdf_header)
        mock_bdf_reader.return_value.getHeader.return_value = mock_bdf_header

        with pytest.raises(RuntimeError, match="Error while loading flags from a BDF"):
            _flags = load_flags_from_bdf(bdf_path, 1, {}, True)

@pytest.mark.parametrize(
    "input_flags_shape, input_baseband_idx, input_spw_idx, expected_shape",
    [
        ((1, 6, 2), 0, 0, (1, 6, 1024, 2)),
        ((1, 6, 2), 0, 1, (1, 6, 512, 2)),
        ((1, 3, 2), 0, 1, (1, 3, 512, 2)),
        ((1, 3, 2), 1, 0, (1, 3, 1024, 2)),
        ((1, 3, 2), 1, 1, (1, 3, 1024, 2)),
    ],
)
def test__expand_frequency_in_flags_subset(input_flags_shape, input_baseband_idx, input_spw_idx, expected_shape):
    from xradio.measurement_set._utils._asdm._utils._bdf.robust_load_data_flags import (
        _expand_frequency_in_flags_subset,
    )

    flags = np.ones(input_flags_shape, dtype="bool")
    expanded = _expand_frequency_in_flags_subset(
        flags,
        bdf_descr_X136e,
        input_baseband_idx,
        input_spw_idx,
        (slice(None), slice(None), slice(None), slice(None)),
    )
    assert isinstance(expanded, np.ndarray)
    assert expanded.dtype == "bool"
    assert expanded.shape == expected_shape
