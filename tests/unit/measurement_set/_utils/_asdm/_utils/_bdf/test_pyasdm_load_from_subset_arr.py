from unittest import mock

import pytest

import numpy as np

import pyasdm

basebands_simple = [
    {
        "name": "BB_1",
        "spectralWindows": [
            {
                "crossPolProducts": [],
                "sdPolProducts": [],
                "scaleFactor": 103107.95,
                "numSpectralPoint": 128,
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
                "numSpectralPoint": 128,
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
                "numSpectralPoint": 64,
                "numBin": 1,
                "sideband": None,
                "sw": "3",
            },
            {
                "crossPolProducts": [],
                "sdPolProducts": [],
                "scaleFactor": 36454.168,
                "numSpectralPoint": 64,
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
                "numSpectralPoint": 128,
                "numBin": 1,
                "sideband": None,
                "sw": "5",
            }
        ],
    },
]


def test_load_visibilities_all_subsets():
    from xradio.measurement_set._utils._asdm._utils._bdf.pyasdm_load_from_subset_arr import (
        load_visibilities_all_subsets,
    )

    bdf_descr = {
        "basebands": basebands_simple,
        "processor_type": "CORRELATOR",
    }

    with (
        mock.patch("pyasdm.bdf.BDFReader") as mock_bdf_reader,
        mock.patch("pyasdm.bdf.BDFHeader") as mock_bdf_header,
    ):
        with pytest.raises(RuntimeError, match="autoData not present"):
            load_visibilities_all_subsets(
                mock_bdf_reader,
                (1, 36, 9, 4, 2, 512, 2, 2),
                (0, 0),
                bdf_descr,
                slice(None),
            )
        mock_bdf_header.getBasebandsList.assert_not_called()


def test_load_visibilities_all_subsets_error():
    from xradio.measurement_set._utils._asdm._utils._bdf.pyasdm_load_from_subset_arr import (
        load_visibilities_all_subsets,
    )

    bdf_descr = {
        "basebands": basebands_simple,
        "processor_type": "CORRELATOR",
    }

    with (
        mock.patch("pyasdm.bdf.BDFReader") as mock_bdf_reader,
        mock.patch("pyasdm.bdf.BDFHeader") as mock_bdf_header,
    ):
        mock_bdf_reader.hasSubset.side_effect = [True, False]
        mock_bdf_reader.getSubset.side_effect = {ValueError}
        visibilities = load_visibilities_all_subsets(
            mock_bdf_reader, (1, 36, 9, 4, 2, 512, 2, 2), (0, 0), bdf_descr
        )
        assert visibilities is None
        mock_bdf_header.getBasebandsList.assert_not_called()
        mock_bdf_reader.hasSubset.assert_called_once()
        mock_bdf_reader.getSubset.assert_called_once()


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


def test_load_visibilities_all_subsets_X136e():
    from xradio.measurement_set._utils._asdm._utils._bdf.pyasdm_load_from_subset_arr import (
        load_visibilities_all_subsets,
    )

    with mock.patch("pyasdm.bdf.BDFReader") as mock_bdf_reader:
        # For _load_vis_subset, etc.
        mock_bdf_reader.hasSubset.side_effect = [True, False]
        mock_bdf_reader.getSubset.side_effect = [
            {"autoData": {"present": True, "arr": np.zeros((73728))}}
        ]
        visibilities = load_visibilities_all_subsets(
            mock_bdf_reader, (1, 36, 9, 4, 2, 512, 2, 2), (0, 1), bdf_descr_X136e
        )

        assert isinstance(visibilities, np.ndarray)
        assert visibilities.size == 9216
        assert visibilities.shape == (1, 9, 512, 2)
        assert visibilities.dtype == np.dtype("float64")

        mock_bdf_reader.hasSubset.assert_called()
        assert mock_bdf_reader.hasSubset.call_count == 2
        mock_bdf_reader.getSubset.assert_called()
        assert mock_bdf_reader.getSubset.call_count == 1


def test__load_vis_subset():
    from xradio.measurement_set._utils._asdm._utils._bdf.pyasdm_load_from_subset_arr import (
        _load_vis_subset,
    )

    subset = {
        "autoData": {"present": True, "arr": np.zeros((18432))},
        "crossData": {"present": True, "arr": np.zeros((147456))},
    }
    visibilities = _load_vis_subset(
        subset,
        (2, 36, 9, 4, 2, 64, 2, 2),
        (0, 1),
        12345.6,
        pyasdm.enumerations.ProcessorType.CORRELATOR,
    )

    assert isinstance(visibilities, np.ndarray)
    assert visibilities.size == 11520
    assert visibilities.shape == (2, 45, 64, 2)
    assert visibilities.dtype == np.dtype("complex128")


@pytest.mark.parametrize(
    "input_data_arr, input_guessed_shape, input_baseband_spw_idxs, input_processor_type, expected_size, expected_shape",
    [
        (
            np.zeros((86016)),
            (2, 21, 9, 4, 2, 64, 2, 2),
            (0, 0),
            pyasdm.enumerations.ProcessorType.CORRELATOR,
            5376,
            (2, 21, 64, 2),
        ),
        (
            np.zeros((7168)),
            (1, 7, 4, 4, 2, 32, 2, 2),
            (0, 0),
            pyasdm.enumerations.ProcessorType.CORRELATOR,
            448,
            (1, 7, 32, 2),
        ),
        (
            np.zeros((14336)),
            (1, 7, 4, 4, 2, 32, 4, 2),
            (0, 0),
            pyasdm.enumerations.ProcessorType.CORRELATOR,
            896,
            (1, 7, 32, 4),
        ),
        (
            np.zeros((14336)),
            (1, 7, 4, 4, 2, 32, 4, 2),
            (0, 0),
            pyasdm.enumerations.ProcessorType.RADIOMETER,
            14336,
            (1, 7, 4, 2, 32, 4, 2),
        ),
        (
            np.zeros((14336)),
            (1, 7, 4, 4, 2, 32, 4, 2),
            (0, 0),
            pyasdm.enumerations.ProcessorType.SPECTROMETER,
            14336,
            (1, 7, 4, 2, 32, 4, 2),
        ),
    ],
)
def test__load_vis_subset_cross_data(
    input_data_arr,
    input_guessed_shape,
    input_baseband_spw_idxs,
    input_processor_type,
    expected_size,
    expected_shape,
):
    from xradio.measurement_set._utils._asdm._utils._bdf.pyasdm_load_from_subset_arr import (
        _load_vis_subset_cross_data,
    )

    visibilities = _load_vis_subset_cross_data(
        input_data_arr,
        input_guessed_shape,
        input_baseband_spw_idxs,
        9876.5,
        input_processor_type,
    )

    assert isinstance(visibilities, np.ndarray)
    assert visibilities.size == expected_size
    assert visibilities.shape == expected_shape
    if input_processor_type == pyasdm.enumerations.ProcessorType.CORRELATOR:
        assert visibilities.dtype == np.dtype("complex128")
    else:
        assert visibilities.dtype == np.dtype("float64")


@pytest.mark.parametrize(
    "input_data_arr, input_guessed_shape, input_baseband_spw_idxs, expected_size, expected_shape",
    [
        (np.zeros((18432)), (2, 36, 9, 4, 2, 64, 2, 2), (0, 0), 2304, (2, 9, 64, 2)),
        (np.zeros((2048)), (1, 7, 4, 4, 2, 32, 2, 2), (0, 0), 256, (1, 4, 32, 2)),
        (np.zeros((4096)), (1, 7, 4, 4, 2, 32, 3, 2), (0, 0), 512, (1, 1, 4, 128)),
    ],
)
def test__load_vis_subset_auto_data(
    input_data_arr,
    input_guessed_shape,
    input_baseband_spw_idxs,
    expected_size,
    expected_shape,
):
    from xradio.measurement_set._utils._asdm._utils._bdf.pyasdm_load_from_subset_arr import (
        _load_vis_subset_auto_data,
    )

    visibilities = _load_vis_subset_auto_data(
        input_data_arr,
        input_guessed_shape,
        input_baseband_spw_idxs,
    )

    assert isinstance(visibilities, np.ndarray)
    assert visibilities.size == expected_size
    assert visibilities.shape == expected_shape
    if expected_shape[-1] <= 2:
        assert visibilities.dtype == np.dtype("float64")
    else:
        assert visibilities.dtype == np.dtype("complex128")


@pytest.mark.parametrize(
    "input_bdf_descr, input_baseband_spw_idxs, expected_shape",
    [
        (bdf_descr_X136e, (0, 0), (1, 45, 10, 4, 2, 1024, 2, 2)),
        (bdf_descr_X136e, (0, 1), (1, 45, 10, 4, 2, 512, 2, 2)),
        (bdf_descr_X136e, (1, 0), (1, 45, 10, 4, 2, 1024, 2, 2)),
        (bdf_descr_X136e, (1, 1), (1, 45, 10, 4, 2, 1024, 2, 2)),
        (bdf_descr_X136e, (2, 0), (1, 45, 10, 4, 1, 2048, 2, 2)),
    ],
)
def test_define_visibility_shape(
    input_bdf_descr, input_baseband_spw_idxs, expected_shape
):
    from xradio.measurement_set._utils._asdm._utils._bdf.pyasdm_load_from_subset_arr import (
        define_visibility_shape,
    )

    shape = define_visibility_shape(input_bdf_descr, input_baseband_spw_idxs)
    assert shape == expected_shape


def test_load_flags_all_subsets():
    from xradio.measurement_set._utils._asdm._utils._bdf.pyasdm_load_from_subset_arr import (
        load_flags_all_subsets,
    )

    with (
        mock.patch("pyasdm.bdf.BDFReader") as mock_bdf_reader,
        mock.patch("pyasdm.bdf.BDFHeader") as mock_bdf_header,
    ):
        mock_bdf_reader.hasSubset.side_effect = [True, False]
        mock_bdf_reader.getSubset.side_effect = [
            {"flags": {"present": True, "arr": np.zeros((71680), dtype="bool")}}
        ]
        guessed_shape = {
            "auto": (1, 7, 4, 2, 64, 2),
            "cross": (1, 28, 4, 2, 64, 2, 2),
        }
        flags = load_flags_all_subsets(mock_bdf_reader, guessed_shape, (0, 0))
        assert flags.size == 560
        assert flags.shape == (1, 35, 4, 2, 2)
        assert mock_bdf_reader.hasSubset.call_count == 2
        mock_bdf_reader.getSubset.assert_called_once()
        mock_bdf_header.getBasebandsList.assert_not_called()


def test_load_flags_all_subsets_3pols_only_autodata():
    from xradio.measurement_set._utils._asdm._utils._bdf.pyasdm_load_from_subset_arr import (
        load_flags_all_subsets,
    )

    with (
        mock.patch("pyasdm.bdf.BDFReader") as mock_bdf_reader,
        mock.patch("pyasdm.bdf.BDFHeader") as mock_bdf_header,
    ):
        mock_bdf_reader.hasSubset.side_effect = [True, False]
        mock_bdf_reader.getSubset.side_effect = [
            {"flags": {"present": True, "arr": np.zeros((10752), dtype="bool")}}
        ]
        guessed_shape = {
            "auto": (1, 7, 4, 2, 64, 3),
            "cross": None,
        }
        flags = load_flags_all_subsets(mock_bdf_reader, guessed_shape, (0, 0))
        assert flags.size == 28
        assert flags.shape == (1, 7, 4, 1)
        assert mock_bdf_reader.hasSubset.call_count == 2
        mock_bdf_reader.getSubset.assert_called_once()
        mock_bdf_header.getBasebandsList.assert_not_called()


def test_load_flags_all_subsets_auto_3pol():
    from xradio.measurement_set._utils._asdm._utils._bdf.pyasdm_load_from_subset_arr import (
        load_flags_all_subsets,
    )

    with (
        mock.patch("pyasdm.bdf.BDFReader") as mock_bdf_reader,
        mock.patch("pyasdm.bdf.BDFHeader") as mock_bdf_header,
    ):
        mock_bdf_reader.hasSubset.side_effect = [True, False]
        mock_bdf_reader.getSubset.side_effect = [
            {"flags": {"present": True, "arr": np.zeros((114688 + 10752))}}
        ]
        guessed_shape = {
            "auto": (1, 7, 4, 2, 64, 3),
            "cross": (1, 28, 4, 2, 64, 2, 4),
        }
        with pytest.raises(ValueError, match="number of dimensions"):
            flags = load_flags_all_subsets(mock_bdf_reader, guessed_shape, (0, 0))
            assert flags.size == 560
            assert flags.shape == (1, 35, 4, 2, 2)
            assert mock_bdf_reader.hasSubset.call_count == 2
            mock_bdf_reader.getSubset.assert_called_once()
            mock_bdf_header.getBasebandsList.assert_not_called()


def test_load_flags_all_subsets_error():
    from xradio.measurement_set._utils._asdm._utils._bdf.pyasdm_load_from_subset_arr import (
        load_flags_all_subsets,
    )

    with (
        mock.patch("pyasdm.bdf.BDFReader") as mock_bdf_reader,
        mock.patch("pyasdm.bdf.BDFHeader") as mock_bdf_header,
    ):
        mock_bdf_reader.hasSubset.side_effect = [True, False]
        mock_bdf_reader.getSubset.side_effect = {ValueError}
        flags = load_flags_all_subsets(
            mock_bdf_reader, (1, 36, 9, 4, 2, 512, 2, 2), (0, 0)
        )
        assert flags is None
        mock_bdf_header.getBasebandsList.assert_not_called()
        mock_bdf_reader.hasSubset.assert_called_once()
        mock_bdf_reader.getSubset.assert_called_once()


def test_load_flags_all_subsets_X136e():
    from xradio.measurement_set._utils._asdm._utils._bdf.pyasdm_load_from_subset_arr import (
        load_flags_all_subsets,
    )

    with mock.patch("pyasdm.bdf.BDFReader") as mock_bdf_reader:
        # For _load_vis_subset, etc.
        mock_bdf_reader.hasSubset.side_effect = [True, False]
        mock_bdf_reader.getSubset.side_effect = [
            {"autoData": {"present": True, "arr": np.zeros((73728))}}
        ]
        guessed_shape = {
            "cross": (1, 36, 4, 2, 512, 2, 2),
            "auto": (1, 9, 4, 2, 512, 2),
        }
        flags = load_flags_all_subsets(
            mock_bdf_reader,
            guessed_shape,
            (0, 1),
        )
        assert isinstance(flags, np.ndarray)
        assert flags.size == 90
        assert flags.shape == (1, 45, 2)
        assert flags.dtype == np.dtype("bool")

        mock_bdf_reader.hasSubset.assert_called()
        assert mock_bdf_reader.hasSubset.call_count == 2
        mock_bdf_reader.getSubset.assert_called()
        assert mock_bdf_reader.getSubset.call_count == 1


def test_define_flag_shape_X136e():
    from xradio.measurement_set._utils._asdm._utils._bdf.pyasdm_load_from_subset_arr import (
        define_flag_shape,
    )

    shape = define_flag_shape(bdf_descr_X136e, (0, 0))
    assert shape == {"cross": (), "auto": (1, 10, 4, 2, 2)}


def test__try_alternatives_guessed_shape():
    from xradio.measurement_set._utils._asdm._utils._bdf.pyasdm_load_from_subset_arr import (
        _try_alternatives_guessed_shape,
    )

    guessed_shape = {
        "cross": (1, 36, 4, 512, 2, 2),
        "auto": (1, 9, 4, 512, 2),
    }
    flags_actual_size = 1024
    baseband_spw_idxs = (0, 0)
    new_shape, new_baseband_spw_idxs = _try_alternatives_guessed_shape(
        guessed_shape, flags_actual_size, baseband_spw_idxs
    )
    assert new_baseband_spw_idxs == baseband_spw_idxs
    assert new_shape == {"cross": (1, 36, 1, 1, 2), "auto": (1, 9, 1, 1, 2)}


def test__try_alternatives_guessed_shape_oversized():
    from xradio.measurement_set._utils._asdm._utils._bdf.pyasdm_load_from_subset_arr import (
        _try_alternatives_guessed_shape,
    )

    guessed_shape = {
        "cross": (1, 10, 4, 64, 2, 2),
        "auto": (1, 5, 4, 64, 2),
    }
    flags_actual_size = 16000
    baseband_spw_idxs = (0, 0)
    with pytest.raises(RuntimeError, match="Unexpected large flags array"):
        _new_shape, _new_baseband_spw_idxs = _try_alternatives_guessed_shape(
            guessed_shape, flags_actual_size, baseband_spw_idxs
        )
