from contextlib import nullcontext as no_raises
from unittest import mock

import numpy as np
import pyasdm
import pytest

basebands_example = [
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


@pytest.mark.parametrize(
    "input_slice",
    [
        ((slice(None), slice(None), slice(None), slice(None))),
        ((slice(2, 3), slice(None), slice(None), slice(None))),
        ((slice(None), slice(1, 5), slice(None), slice(None))),
        ((slice(None), slice(None), slice(100, 200), slice(None))),
        ((slice(None), slice(None), slice(None), slice(0, 1))),
        ((slice(1, 2), slice(0, 3), slice(900, 960), slice(0, 1))),
    ],
)
def test_load_visibilities_all_subsets_from_trees(input_slice):
    from xradio.measurement_set._utils._asdm._utils._bdf.load_from_pyasdm_arr_trees import (
        load_visibilities_all_subsets_from_trees,
    )

    bdf_descr = {
        "basebands": basebands_example,
        "correlation_mode": pyasdm.enumerations.CorrelationMode.CROSS_AND_AUTO,
        "processor_type": "CORRELATOR",
        "num_antenna": 44,
    }
    with (
        mock.patch("pyasdm.bdf.BDFReader") as mock_bdf_reader,
        mock.patch("pyasdm.bdf.BDFHeader") as mock_bdf_header,
    ):
        with pytest.raises(RuntimeError, match="not present"):
            # Test with load_one_spw_from_file=False => load_vis_subset_from_tree() (also below)
            load_visibilities_all_subsets_from_trees(
                mock_bdf_reader,
                (2, 45, 2, 64, 2, 2),
                (0, 0),
                bdf_descr,
                input_slice,
                load_one_spw_from_file=False,
            )
        mock_bdf_header.getBasebandsList.assert_not_called()
        mock_bdf_header.getSubset.assert_not_called()
        mock_bdf_header.hasSubset.assert_not_called()


def test_load_subset_with_get_subset():
    from xradio.measurement_set._utils._asdm._utils._bdf.load_from_pyasdm_arr_trees import (
        load_subset_with_get_subset,
    )

    with (
        mock.patch("pyasdm.bdf.BDFReader") as mock_bdf_reader,
        mock.patch("pyasdm.bdf.BDFHeader") as mock_bdf_header,
    ):
        mock_bdf_reader.hasSubset.side_effect = [True, False]
        subset = load_subset_with_get_subset(mock_bdf_reader, ["autoData", "crossData"])
        assert subset
        mock_bdf_reader.hasSubset.assert_not_called()
        mock_bdf_reader.getSubset.assert_called_once()

        mock_bdf_header.getBasebandsList.assert_not_called()
        mock_bdf_header.getSubset.assert_not_called()
        mock_bdf_header.hasSubset.assert_not_called()


def test_load_subset_with_get_ndarrays():
    from xradio.measurement_set._utils._asdm._utils._bdf.load_from_pyasdm_arr_trees import (
        load_subset_with_get_ndarrays,
    )

    guessed_shape = (2, 45, 2, 64, 2, 2)
    bdf_descr = {
        "basebands": basebands_example,
        "processor_type": "CORRELATOR",
    }
    with (
        mock.patch("pyasdm.bdf.BDFReader") as mock_bdf_reader,
        mock.patch("pyasdm.bdf.BDFHeader") as mock_bdf_header,
    ):
        mock_bdf_reader.hasSubset.side_effect = [True, False]
        result_ndarray = np.ndarray(())

        def load_spw_function(x, y):
            return result_ndarray

        load_spw_function_params = (bdf_descr, guessed_shape)
        ndarrays = load_subset_with_get_ndarrays(
            mock_bdf_reader, 0, load_spw_function, load_spw_function_params
        )
        assert ndarrays
        assert mock_bdf_reader.hasSubset.call_count == 0
        mock_bdf_reader.getSubset.assert_not_called()

        mock_bdf_header.getBasebandsList.assert_not_called()
        mock_bdf_header.getSubset.assert_not_called()
        mock_bdf_header.hasSubset.assert_not_called()


def test_load_vis_subset_from_tree():
    from xradio.measurement_set._utils._asdm._utils._bdf.load_from_pyasdm_arr_trees import (
        load_vis_subset_from_tree,
    )

    pyasdm_subset = {}
    guessed_shape = (2, 45, 2, 64, 2, 2)
    bdf_descr = {
        "basebands": basebands_example,
        "correlation_mode": pyasdm.enumerations.CorrelationMode.AUTO_ONLY,
        "num_antenna": 18,
        "processor_type": "CORRELATOR",
    }
    with (
        mock.patch("pyasdm.bdf.BDFReader") as mock_bdf_reader,
        mock.patch("pyasdm.bdf.BDFHeader") as mock_bdf_header,
    ):
        mock_bdf_reader.hasSubset.side_effect = [True, False]
        empty_slice = (slice(None), slice(None), slice(None), slice(None))
        with pytest.raises(RuntimeError, match="not present"):
            load_vis_subset_from_tree(
                pyasdm_subset, guessed_shape, (0, 0), bdf_descr, empty_slice
            )
        mock_bdf_header.getBasebandsList.assert_not_called()
        mock_bdf_header.getSubset.assert_not_called()
        mock_bdf_header.hasSubset.assert_not_called()


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


@pytest.mark.parametrize("input_load_one_spw_from_file", [(True), (False)])
def test_load_visibilities_all_subsets_from_trees_X136e(input_load_one_spw_from_file):
    from xradio.measurement_set._utils._asdm._utils._bdf.load_from_pyasdm_arr_trees import (
        load_visibilities_all_subsets_from_trees,
    )

    with mock.patch("pyasdm.bdf.BDFReader") as mock_bdf_reader:
        # For load_vis_subset, etc.
        mock_bdf_reader.hasSubset.side_effect = [True, False]
        if input_load_one_spw_from_file:
            mock_bdf_reader.getNDArrays.side_effect = [
                {"visibilities": np.ones(shape=(1, 9, 1024, 2), dtype="complex128")}
            ]
        else:
            mock_bdf_reader.getSubset.side_effect = [
                {
                    "autoData": {
                        "present": True,
                        "arr": np.zeros((1000000), dtype="float64"),
                    },
                    "crossData": {
                        "present": False,
                        "arr": None,
                    },
                },
                None,
            ]
        empty_slice = (slice(None), slice(None), slice(None), slice(None))
        visibilities = load_visibilities_all_subsets_from_trees(
            mock_bdf_reader,
            (1, 36, 9, 4, 2, 512, 2, 2),
            (0, 1),
            bdf_descr_X136e,
            empty_slice,
            load_one_spw_from_file=input_load_one_spw_from_file,
        )

        assert isinstance(visibilities, np.ndarray)
        assert visibilities.dtype == np.dtype("complex128")

        assert mock_bdf_reader.hasSubset.call_count == 2
        if input_load_one_spw_from_file:
            assert visibilities.size == 18432
            assert visibilities.shape == (1, 9, 1024, 2)
            assert mock_bdf_reader.hasSubset.call_count == 2
            mock_bdf_reader.getNDArrays.assert_called_once()
            mock_bdf_reader.getSubset.assert_not_called()
        else:
            assert visibilities.size == 9216
            assert visibilities.shape == (1, 9, 512, 2)
            mock_bdf_reader.getSubset.assert_called_once()
            mock_bdf_reader.getNDArrays.assert_not_called()


# From (SD) EB uid___A002_Xac5575_X4086, BDF uid___A002_Xac5575_X4089 (AUTO_ONLY/ALMA WVR Data)
bdf_descr_X4089 = {
    "execBlock": "uid://A002/Xac5575/X4086",
    "dimensionality": 0,
    "num_time": 34,
    "processor_type": pyasdm.enumerations.ProcessorType.RADIOMETER,
    "binary_types": [
        "flags",
        "autoData",
    ],
    "correlation_mode": pyasdm.enumerations.CorrelationMode.AUTO_ONLY,
    "apc": [],
    "num_antenna": 43,
    "basebands": [
        {
            "name": "NOBB",
            "spectralWindows": [
                {
                    "crossPolProducts": [],
                    "sdPolProducts": [
                        pyasdm.enumerations.StokesParameter.I,
                    ],
                    "scaleFactor": None,
                    "numSpectralPoint": 4,
                    "numBin": 1,
                    "sideband": pyasdm.enumerations.NetSideband.DSB,
                    "sw": "1",
                },
            ],
        },
    ],
}


@pytest.mark.parametrize("input_load_one_spw_from_file", [(True), (False)])
def test_load_visibilities_all_subsets_from_trees_X4089_wvr(
    input_load_one_spw_from_file,
):
    from xradio.measurement_set._utils._asdm._utils._bdf.load_from_pyasdm_arr_trees import (
        load_visibilities_all_subsets_from_trees,
    )

    with mock.patch("pyasdm.bdf.BDFReader") as mock_bdf_reader:
        # For load_vis_subset, etc.
        mock_bdf_reader.hasSubset.side_effect = [True, False]
        if input_load_one_spw_from_file:
            mock_bdf_reader.getNDArrays.side_effect = [
                {"visibilities": np.ones(shape=(1, 9, 4, 2), dtype="float64")}
            ]
        else:
            mock_bdf_reader.getSubset.side_effect = [
                {
                    "autoData": {
                        "present": True,
                        "arr": np.zeros((1000000), dtype="float64"),
                    },
                    "crossData": {
                        "present": False,
                        "arr": None,
                    },
                },
                None,
            ]
        empty_slice = (slice(None), slice(None), slice(None), slice(None))
        visibilities = load_visibilities_all_subsets_from_trees(
            mock_bdf_reader,
            (1, 903, 43, 1, 1, 4, 1, 1),
            (0, 0),
            bdf_descr_X4089,
            empty_slice,
            load_one_spw_from_file=input_load_one_spw_from_file,
        )

        assert isinstance(visibilities, np.ndarray)
        assert visibilities.dtype == np.dtype("float64")

        assert mock_bdf_reader.hasSubset.call_count == 2
        if input_load_one_spw_from_file:
            assert visibilities.size == 72
            assert visibilities.shape == (1, 9, 4, 2)
            assert mock_bdf_reader.hasSubset.call_count == 2
            mock_bdf_reader.getNDArrays.assert_called_once()
            mock_bdf_reader.getSubset.assert_not_called()
        else:
            assert visibilities.size == 172
            assert visibilities.shape == (1, 43, 4, 1)
            mock_bdf_reader.getSubset.assert_called_once()
            mock_bdf_reader.getNDArrays.assert_not_called()


# All SPWs have same #chan => uses load_subset_with_get_subset() / load_vis_subset_from_tree()
bdf_descr_X136e_simplified = {
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
                    "crossPolProducts": [],
                    "sdPolProducts": [
                        pyasdm.enumerations.StokesParameter.XX,
                        pyasdm.enumerations.StokesParameter.YY,
                    ],
                    "scaleFactor": None,
                    "numSpectralPoint": 512,
                    "numBin": 1,
                    "sideband": pyasdm.enumerations.NetSideband.LSB,
                    "sw": "1",
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
                    "numSpectralPoint": 512,
                    "numBin": 1,
                    "sideband": pyasdm.enumerations.NetSideband.LSB,
                    "sw": "1",
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
                    "numSpectralPoint": 512,
                    "numBin": 1,
                    "sideband": pyasdm.enumerations.NetSideband.LSB,
                    "sw": "1",
                }
            ],
        },
    ],
}


@pytest.mark.parametrize("input_load_one_spw_from_file", [(True), (False)])
def test_load_visibilities_all_subsets_from_trees_X136e_simplified(
    input_load_one_spw_from_file,
):
    from xradio.measurement_set._utils._asdm._utils._bdf.load_from_pyasdm_arr_trees import (
        load_visibilities_all_subsets_from_trees,
    )

    frequency_len = 512
    polarization_len = 2
    all_basebands_len = 9
    with mock.patch("pyasdm.bdf.BDFReader") as mock_bdf_reader:
        # For load_vis_subset, etc.
        mock_bdf_reader.hasSubset.side_effect = [True, False]
        # Only autoDAta values in this test
        if input_load_one_spw_from_file:
            mock_bdf_reader.getNDArrays.side_effect = [
                {
                    "visibilities": np.ones(
                        shape=(1, all_basebands_len, frequency_len, polarization_len),
                        dtype="complex128",
                    )
                }
            ]
        else:
            mock_bdf_reader.getSubset.side_effect = [
                {
                    "autoData": {
                        "present": True,
                        "arr": np.zeros((50000), dtype="float64"),
                    },
                    "crossData": {"present": False, "arr": None},
                },
                None,
            ]
        empty_slice = (slice(None), slice(None), slice(None), slice(None))
        visibilities = load_visibilities_all_subsets_from_trees(
            mock_bdf_reader,
            (1, 36, 9, 4, 2, frequency_len, polarization_len, 2),
            (1, 0),
            bdf_descr_X136e_simplified,
            empty_slice,
            load_one_spw_from_file=input_load_one_spw_from_file,
        )

        assert isinstance(visibilities, np.ndarray)
        assert visibilities.size == all_basebands_len * frequency_len * polarization_len
        assert visibilities.shape == (
            1,
            all_basebands_len,
            frequency_len,
            polarization_len,
        )
        assert visibilities.dtype == np.dtype("complex128")

        assert mock_bdf_reader.hasSubset.call_count == 2
        if input_load_one_spw_from_file:
            assert mock_bdf_reader.hasSubset.call_count == 2
            mock_bdf_reader.getNDArrays.assert_called_once()
            mock_bdf_reader.getSubset.assert_not_called()
        else:
            mock_bdf_reader.getSubset.assert_called_once()
            mock_bdf_reader.getNDArrays.assert_not_called()


bdf_descr_X136e_simplified_with_cross_data = {
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
    "correlation_mode": pyasdm.enumerations.CorrelationMode.CROSS_AND_AUTO,
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
                    "numSpectralPoint": 256,
                    "numBin": 1,
                    "sideband": pyasdm.enumerations.NetSideband.LSB,
                    "sw": "1",
                },
            ],
        },
        {
            "name": "BB_2",
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
                    "numSpectralPoint": 128,
                    "numBin": 1,
                    "sideband": pyasdm.enumerations.NetSideband.LSB,
                    "sw": "1",
                },
            ],
        },
    ],
}


@pytest.mark.parametrize("input_load_one_spw_from_file", [(True), (False)])
def test_load_visibilities_all_subsets_from_trees_X136e_simplified_with_cross_data(
    input_load_one_spw_from_file,
):
    from xradio.measurement_set._utils._asdm._utils._bdf.load_from_pyasdm_arr_trees import (
        load_visibilities_all_subsets_from_trees,
    )

    frequency_len = 128
    polarization_len = 2
    all_basebands_len = 45
    with mock.patch("pyasdm.bdf.BDFReader") as mock_bdf_reader:
        # For load_vis_subset, etc.
        mock_bdf_reader.hasSubset.side_effect = [True, False]
        # Only autoDAta values in this test
        if input_load_one_spw_from_file:
            mock_bdf_reader.getNDArrays.side_effect = [
                {
                    "visibilities": np.ones(
                        shape=(1, all_basebands_len, frequency_len, polarization_len),
                        dtype="complex128",
                    )
                }
            ]
        else:
            mock_bdf_reader.getSubset.side_effect = [
                {
                    "autoData": {
                        "present": True,
                        "arr": np.zeros((50000), dtype="float64"),
                    },
                    "crossData": {
                        "present": True,
                        "arr": np.zeros((250000), dtype="complex128"),
                    },
                },
                None,
            ]
        empty_slice = (slice(None), slice(None), slice(None), slice(None))
        visibilities = load_visibilities_all_subsets_from_trees(
            mock_bdf_reader,
            (1, 36, 9, 4, 2, frequency_len, polarization_len, 2),
            (1, 0),
            bdf_descr_X136e_simplified_with_cross_data,
            empty_slice,
            load_one_spw_from_file=input_load_one_spw_from_file,
        )

        assert isinstance(visibilities, np.ndarray)
        assert visibilities.size == all_basebands_len * frequency_len * polarization_len
        assert visibilities.shape == (
            1,
            all_basebands_len,
            frequency_len,
            polarization_len,
        )
        assert visibilities.dtype == np.dtype("complex128")

        assert mock_bdf_reader.hasSubset.call_count == 2
        if input_load_one_spw_from_file:
            assert mock_bdf_reader.hasSubset.call_count == 2
            mock_bdf_reader.getNDArrays.assert_called_once()
            mock_bdf_reader.getSubset.assert_not_called()
        else:
            mock_bdf_reader.getSubset.assert_called_once()
            mock_bdf_reader.getNDArrays.assert_not_called()


@pytest.mark.parametrize("input_load_one_spw_from_file", [(True), (False)])
def test_load_visibilities_all_subsets_from_trees_X136e_error(
    input_load_one_spw_from_file,
):
    from xradio.measurement_set._utils._asdm._utils._bdf.load_from_pyasdm_arr_trees import (
        load_visibilities_all_subsets_from_trees,
    )

    with mock.patch("pyasdm.bdf.BDFReader") as mock_bdf_reader:
        # For load_vis_subset, etc.
        mock_bdf_reader.hasSubset.side_effect = [True, False]
        if input_load_one_spw_from_file:
            mock_bdf_reader.getNDArrays.side_effect = [ValueError, None]
        else:
            mock_bdf_reader.getSubset.side_effect = [ValueError, None]

        empty_slice = (slice(None), slice(None), slice(None), slice(None))
        visibilities = load_visibilities_all_subsets_from_trees(
            mock_bdf_reader,
            (1, 36, 9, 4, 2, 512, 2, 2),
            (0, 1),
            bdf_descr_X136e,
            empty_slice,
            load_one_spw_from_file=input_load_one_spw_from_file,
        )
        assert visibilities is None

        mock_bdf_reader.hasSubset.assert_called_once()
        if input_load_one_spw_from_file:
            mock_bdf_reader.getNDArrays.assert_called_once()
            mock_bdf_reader.getSubset.assert_not_called()
        else:
            mock_bdf_reader.getSubset.assert_called_once()
            mock_bdf_reader.getNDArrays.assert_not_called()


# BDF uid___A002_Xb08ef9_X64c6
bdf_descr_X64c6 = {
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
    "correlation_mode": pyasdm.enumerations.CorrelationMode.CROSS_AND_AUTO,
    "apc": pyasdm.enumerations.AtmPhaseCorrection.AP_UNCORRECTED,
    "num_antenna": 9,
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
                    "scaleFactor": np.float32(47062.125),
                    "numSpectralPoint": 2048,
                    "numBin": 1,
                    "sideband": pyasdm.enumerations.NetSideband.LSB,
                    "sw": "1",
                }
            ],
        },
        {
            "name": "BB_2",
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
                    "scaleFactor": np.float32(33277.95),
                    "numSpectralPoint": 2048,
                    "numBin": 1,
                    "sideband": pyasdm.enumerations.NetSideband.USB,
                    "sw": "1",
                },
                {
                    "crossPolProducts": [
                        pyasdm.enumerations.StokesParameter.XX,
                        pyasdm.enumerations.StokesParameter.YY,
                    ],
                    "sdPolProducts": [
                        pyasdm.enumerations.StokesParameter.XX,
                        pyasdm.enumerations.StokesParameter.YY,
                    ],
                    "scaleFactor": np.float32(33277.95),
                    "numSpectralPoint": 2048,
                    "numBin": 1,
                    "sideband": pyasdm.enumerations.NetSideband.LSB,
                    "sw": "2",
                },
            ],
        },
        {
            "name": "BB_3",
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
                    "scaleFactor": np.float32(376497.0),
                    "numSpectralPoint": 128,
                    "numBin": 1,
                    "sideband": pyasdm.enumerations.NetSideband.USB,
                    "sw": "1",
                }
            ],
        },
        {
            "name": "BB_4",
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
                    "scaleFactor": np.float32(376497.0),
                    "numSpectralPoint": 128,
                    "numBin": 1,
                    "sideband": pyasdm.enumerations.NetSideband.USB,
                    "sw": "1",
                }
            ],
        },
    ],
}


@pytest.mark.parametrize(
    "input_cross_data_arr, input_guessed_shape, input_spw_chan_lens, input_overall_spw_idx, input_processor_type, input_array_slice, expected_size, expected_shape",
    [
        (
            np.zeros((15360), dtype="float64"),
            (1, 10, 5, 4, 2, 32, 2, 2),
            [32, 64, 16, 8],
            0,
            pyasdm.enumerations.ProcessorType.CORRELATOR,
            ((slice(None), slice(None), slice(None), slice(None))),
            640,
            (1, 10, 32, 2),
        ),
        (
            np.zeros((15360), dtype="float64"),
            (1, 10, 5, 4, 2, 32, 2, 2),
            [32, 64, 16, 8],
            3,
            pyasdm.enumerations.ProcessorType.RADIOMETER,
            ((slice(None), slice(None), slice(None), slice(None))),
            160,
            (1, 10, 8, 2),
        ),
        (
            np.zeros((15360), dtype="float64"),
            (1, 10, 5, 4, 2, 32, 2, 2),
            [32, 64, 16, 8],
            1,
            pyasdm.enumerations.ProcessorType.CORRELATOR,
            ((slice(None), slice(None), slice(None), slice(None))),
            1280,
            (1, 10, 64, 2),
        ),
        (
            np.zeros((15360), dtype="float64"),
            (1, 10, 5, 4, 2, 32, 2, 2),
            [32, 64, 16, 8],
            2,
            pyasdm.enumerations.ProcessorType.CORRELATOR,
            ((slice(None), slice(None), slice(None), slice(None))),
            320,
            (1, 10, 16, 2),
        ),
        (
            np.zeros((15360), dtype="float64"),
            (1, 10, 5, 4, 2, 32, 2, 2),
            [32, 64, 16, 8],
            3,
            pyasdm.enumerations.ProcessorType.CORRELATOR,
            ((slice(None), slice(None), slice(None), slice(None))),
            160,
            (1, 10, 8, 2),
        ),
        (
            np.zeros((15360), dtype="float64"),
            (3, 10, 5, 4, 2, 32, 2, 2),
            [32, 64, 16, 8],
            3,
            pyasdm.enumerations.ProcessorType.CORRELATOR,
            ((slice(None), slice(None), slice(None), slice(None))),
            160,
            (1, 10, 8, 2),
        ),
        (
            np.zeros((15360), dtype="float64"),
            (3, 10, 5, 4, 2, 32, 2, 2),
            [32, 64, 16, 8],
            3,
            pyasdm.enumerations.ProcessorType.CORRELATOR,
            ((slice(None), slice(None), slice(None), slice(0, 1))),
            80,
            (1, 10, 8, 1),
        ),
        (
            np.zeros((15360), dtype="float64"),
            (3, 10, 5, 4, 2, 32, 2, 2),
            [32, 64, 16, 8],
            3,
            pyasdm.enumerations.ProcessorType.CORRELATOR,
            ((slice(1, 2), slice(2, 4), slice(0, 16), slice(0, 1))),
            16,
            (1, 2, 8, 1),
        ),
    ],
)
def test_load_vis_subset_cross_data_from_tree(
    input_cross_data_arr,
    input_guessed_shape,
    input_spw_chan_lens,
    input_overall_spw_idx,
    input_processor_type,
    input_array_slice,
    expected_size,
    expected_shape,
):
    from xradio.measurement_set._utils._asdm._utils._bdf.load_from_pyasdm_arr_trees import (
        load_vis_subset_cross_data_from_tree,
    )

    visibilities = load_vis_subset_cross_data_from_tree(
        input_cross_data_arr,
        input_guessed_shape,
        input_spw_chan_lens,
        input_overall_spw_idx,
        123456.789,
        input_processor_type,
        input_array_slice,
    )

    assert isinstance(visibilities, np.ndarray)
    assert visibilities.size == expected_size
    assert visibilities.shape == expected_shape
    if input_processor_type == pyasdm.enumerations.ProcessorType.CORRELATOR:
        assert visibilities.dtype == np.dtype("complex128")
    else:
        assert visibilities.dtype == np.dtype("float64")


@pytest.mark.parametrize(
    "input_auto_data_arr, input_guessed_shape, input_spw_chan_lens, input_overall_spw_idx, input_array_slice, expected_size, expected_shape",
    [
        (
            np.zeros((15360), dtype="float64"),
            (1, 10, 5, 4, 2, 32, 2, 2),
            [32, 64, 16, 8],
            0,
            ((slice(None), slice(None), slice(None), slice(None))),
            320,
            (1, 5, 32, 2),
        ),
        (
            np.zeros((15360), dtype="float64"),
            (1, 10, 5, 4, 2, 32, 2, 2),
            [32, 64, 16, 8],
            1,
            ((slice(None), slice(None), slice(None), slice(None))),
            640,
            (1, 5, 64, 2),
        ),
        (
            np.zeros((15360), dtype="float64"),
            (1, 10, 5, 4, 2, 32, 2, 2),
            [32, 64, 16, 8],
            2,
            ((slice(None), slice(None), slice(None), slice(None))),
            160,
            (1, 5, 16, 2),
        ),
        (
            np.zeros((15360), dtype="float64"),
            (1, 10, 5, 4, 2, 32, 2, 2),
            [32, 64, 16, 8],
            3,
            ((slice(None), slice(None), slice(None), slice(None))),
            80,
            (1, 5, 8, 2),
        ),
        (
            np.zeros((15360), dtype="float64"),
            (1, 10, 5, 4, 2, 32, 3, 2),
            [32, 5, 16, 8],
            2,
            ((slice(None), slice(None), slice(None), slice(None))),
            240,
            (1, 5, 16, 3),
        ),
        (
            np.zeros((15360), dtype="float64"),
            (2, 10, 5, 4, 2, 32, 2, 2),
            [32, 5, 16, 8],
            2,
            ((slice(None), slice(None), slice(None), slice(1, 2))),
            160,
            (2, 5, 16, 1),
        ),
        (
            np.zeros((15360), dtype="float64"),
            (2, 10, 5, 4, 2, 32, 2, 2),
            [32, 5, 16, 8],
            2,
            ((slice(None), slice(None), slice(8, 16), slice(1, 2))),
            80,
            (2, 5, 8, 1),
        ),
        (
            np.zeros((15360), dtype="float64"),
            (2, 10, 5, 4, 2, 32, 2, 2),
            [32, 5, 16, 8],
            2,
            ((slice(0, 1), slice(3, 5), slice(8, 16), slice(1, 2))),
            16,
            (1, 2, 8, 1),
        ),
    ],
)
def test_load_vis_subset_auto_data_from_tree(
    input_auto_data_arr,
    input_guessed_shape,
    input_spw_chan_lens,
    input_overall_spw_idx,
    input_array_slice,
    expected_size,
    expected_shape,
):
    from xradio.measurement_set._utils._asdm._utils._bdf.load_from_pyasdm_arr_trees import (
        load_vis_subset_auto_data_from_tree,
    )

    visibilities = load_vis_subset_auto_data_from_tree(
        input_auto_data_arr,
        input_guessed_shape,
        input_spw_chan_lens,
        input_overall_spw_idx,
        input_array_slice,
    )

    assert isinstance(visibilities, np.ndarray)
    assert visibilities.size == expected_size
    assert visibilities.shape == expected_shape
    npol = input_guessed_shape[-2]
    if npol == 3:
        assert visibilities.dtype == np.dtype("complex128")
    else:
        assert visibilities.dtype == np.dtype("float64")


def test_load_flags_all_subsets_from_trees_error():
    from xradio.measurement_set._utils._asdm._utils._bdf.load_from_pyasdm_arr_trees import (
        load_flags_all_subsets_from_trees,
    )

    with (
        mock.patch("pyasdm.bdf.BDFReader") as mock_bdf_reader,
        mock.patch("pyasdm.bdf.BDFHeader") as mock_bdf_header,
    ):
        bdf_descr = {
            "basebands": basebands_example,
            "processor_type": "CORRELATOR",
        }
        mock_bdf_reader.hasSubset.side_effect = [True, False]
        mock_bdf_reader.getSubset.side_effect = [ValueError, None]
        guessed_shape = {
            "auto": (2, 9, 4, 2, 64, 2),
            "cross": (2, 36, 4, 2, 64, 2, 2),
        }
        empty_slice = (slice(None), slice(None), slice(None), slice(None))
        vis = load_flags_all_subsets_from_trees(
            mock_bdf_reader, guessed_shape, (0, 0), bdf_descr, empty_slice
        )
        assert vis is None
        mock_bdf_header.getBasebandsList.assert_not_called()
        assert mock_bdf_reader.getSubset.call_count == 1
        assert mock_bdf_reader.hasSubset.call_count == 1


def test_load_flags_all_subsets_from_trees_X136e():
    from xradio.measurement_set._utils._asdm._utils._bdf.load_from_pyasdm_arr_trees import (
        load_flags_all_subsets_from_trees,
    )

    with mock.patch("pyasdm.bdf.BDFReader") as mock_bdf_reader:
        # For load_vis_subset, etc.
        mock_bdf_reader.hasSubset.side_effect = [True, False]
        mock_bdf_reader.getSubset.side_effect = [
            {"autoData": {"present": True, "arr": np.zeros(73728)}}
        ]
        guessed_shape = {
            "auto": (1, 9, 4, 2, 512, 2),
            "cross": (1, 36, 4, 2, 512, 2, 2),
        }
        empty_slice = (slice(None), slice(None), slice(None), slice(None))
        flags = load_flags_all_subsets_from_trees(
            mock_bdf_reader, guessed_shape, bdf_descr_X136e, (0, 1), empty_slice
        )
        assert isinstance(flags, np.ndarray)
        assert flags.size == 90
        assert flags.shape == (1, 45, 2)
        assert flags.dtype == np.dtype("bool")

        mock_bdf_reader.hasSubset.assert_called()
        assert mock_bdf_reader.hasSubset.call_count == 2
        mock_bdf_reader.getSubset.assert_called()
        assert mock_bdf_reader.getSubset.call_count == 1


bdf_descr_autodata_3pol_pseudo_X136e = {
    "dimensionality": 1,
    "num_time": 0,
    "processor_type": pyasdm.enumerations.ProcessorType.RADIOMETER,
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
                    "crossPolProducts": [],
                    "sdPolProducts": [
                        pyasdm.enumerations.StokesParameter.XX,
                        pyasdm.enumerations.StokesParameter.XY,
                        pyasdm.enumerations.StokesParameter.YY,
                    ],
                    "scaleFactor": None,
                    "numSpectralPoint": 64,
                    "numBin": 1,
                    "sideband": pyasdm.enumerations.NetSideband.LSB,
                    "sw": "1",
                }
            ],
        },
    ],
}
flags_input_guessed_shape_x136e = {
    "cross": (1, 36, 4, 2, 512, 2, 2),
    "auto": (1, 9, 4, 2, 512, 2),
}
flags_input_guessed_shape_2times_x136e = {
    "cross": (2, 36, 4, 2, 512, 2, 2),
    "auto": (2, 9, 4, 2, 512, 2),
}


@pytest.mark.parametrize(
    "input_subset, input_guessed_shape, input_bdf_descr, expected_size, expected_shape, expected_error",
    [
        (
            {},
            flags_input_guessed_shape_x136e,
            bdf_descr_X136e,
            90,
            (1, 45, 2),
            no_raises(),
        ),
        (
            {"flags": {"present": True, "arr": np.zeros(120)}},
            flags_input_guessed_shape_x136e,
            bdf_descr_X136e,
            20,
            (1, 10, 2),
            no_raises(),
        ),
        (
            {"flags": {"present": True, "arr": np.zeros(80)}},
            flags_input_guessed_shape_x136e,
            bdf_descr_X136e,
            20,
            (1, 10, 2),
            no_raises(),
        ),
        (
            {"flags": {"present": True, "arr": np.zeros(120)}},
            flags_input_guessed_shape_2times_x136e,
            bdf_descr_X136e,
            40,
            (2, 10, 2),
            no_raises(),
        ),
        (
            {"flags": {"present": True, "arr": np.zeros(30)}},
            flags_input_guessed_shape_x136e,
            bdf_descr_autodata_3pol_pseudo_X136e,
            40,
            (1, 10, 4),
            no_raises(),
        ),
        (
            {"flags": {"present": True, "arr": np.zeros(3)}},
            flags_input_guessed_shape_x136e,
            bdf_descr_autodata_3pol_pseudo_X136e,
            40,
            (1, 10, 4),
            pytest.raises(RuntimeError, match="Unexpected flags array"),
        ),
        (
            {"flags": {"present": True, "arr": np.zeros(450)}},
            flags_input_guessed_shape_x136e,
            bdf_descr_X64c6,
            90,
            (1, 45, 2),
            no_raises(),
        ),
        (
            {"flags": {"present": True, "arr": np.zeros(360)}},
            flags_input_guessed_shape_x136e,
            bdf_descr_X64c6,
            90,
            (1, 45, 2),
            no_raises(),
        ),
    ],
)
def test_load_flags_subset_from_tree(
    input_subset,
    input_guessed_shape,
    input_bdf_descr,
    expected_size,
    expected_shape,
    expected_error,
):
    from xradio.measurement_set._utils._asdm._utils._bdf.load_from_pyasdm_arr_trees import (
        load_flags_subset_from_tree,
    )

    with expected_error:
        empty_slice = (slice(None), slice(None), slice(None), slice(None))
        flags = load_flags_subset_from_tree(
            input_subset, input_guessed_shape, input_bdf_descr, (0, 0), empty_slice
        )

        assert isinstance(flags, np.ndarray)
        assert flags.dtype == np.dtype("bool")
        assert flags.size == expected_size
        assert flags.shape == expected_shape


class _SubsetsReader:
    def __init__(self, subsets):
        self._subsets = list(subsets)

    def hasSubset(self):
        return bool(self._subsets)

    def getSubset(self, loadOnlyComponents=None):
        return self._subsets.pop(0)

    def getPath(self):
        return "/no_path/nonexistant/foo"


def _two_spw_bdf_descr():
    spw = {
        "crossPolProducts": ["XX", "YY"],
        "sdPolProducts": ["XX", "YY"],
        "scaleFactor": 1,
    }
    return {
        "correlation_mode": pyasdm.enumerations.CorrelationMode.CROSS_AND_AUTO,
        "processor_type": pyasdm.enumerations.ProcessorType.CORRELATOR,
        "num_antenna": 3,
        "dimensionality": 1,
        "num_time": 0,
        "basebands": [
            {
                "spectralWindows": [
                    {**spw, "numSpectralPoint": 4},
                    {**spw, "numSpectralPoint": 6},
                ]
            }
        ],
    }


def _vis_subset(integration):
    cross, auto = [], []
    for row in range(3):
        for spw, channel_len in enumerate([4, 6]):
            for channel in range(channel_len):
                for pol in range(2):
                    value = 10000 * integration + 1000 * row + 100 * spw
                    value += 10 * channel + pol
                    cross += [value, -value]
                    auto.append(value + 0.5)
    return {
        "crossData": {"present": True, "arr": np.array(cross, dtype="float64")},
        "autoData": {"present": True, "arr": np.array(auto, dtype="float64")},
    }


def test_load_visibilities_all_subsets_from_trees_values():
    from xradio.measurement_set._utils._asdm._utils._bdf.load_from_pyasdm_arr_trees import (
        load_visibilities_all_subsets_from_trees,
    )
    from xradio.measurement_set._utils._asdm._utils._bdf.load_from_pyasdm_subset_array import (
        define_visibility_shape,
    )

    bdf_descr = _two_spw_bdf_descr()
    vis = load_visibilities_all_subsets_from_trees(
        _SubsetsReader([_vis_subset(integration) for integration in range(3)]),
        define_visibility_shape(bdf_descr, (0, 1)),
        (0, 1),
        bdf_descr,
        (slice(1, 3), slice(None), slice(2, 5), slice(None)),
        False,
    )

    integration = np.array([1, 2])[:, None, None, None]
    row = np.arange(3)[None, :, None, None]
    channel = np.arange(2, 5)[None, None, :, None]
    pol = np.arange(2)[None, None, None, :]
    value = 10000 * integration + 1000 * row + 100 + 10 * channel + pol
    np.testing.assert_array_equal(
        vis, np.concatenate([value - 1j * value, value + 0.5], axis=1)
    )


def _flags_subset(flagged):
    # rows: cross baselines 0-2, then antennas 0-2; per row: spw 0, 1; per spw: 2 pols
    flags = np.zeros(24, dtype="int32")
    for row, spw, pol, word in flagged:
        flags[row * 4 + spw * 2 + pol] = word
    return {"flags": {"present": True, "arr": flags}}


FLAGGED = [(1, 1, 0, 16), (5, 1, 1, 2**30), (1, 0, 1, 1), (3, 0, 0, 1)]


@pytest.mark.parametrize(
    "input_baseline_slice, input_polarization_slice, expected_true",
    [
        (slice(None), slice(None), [(1, 0), (5, 1)]),
        (slice(1, 6), slice(None), [(0, 0), (4, 1)]),
        (slice(2, 5), slice(None), []),
        (slice(None), slice(1, 2), [(5, 0)]),
    ],
)
def test_load_flags_subset_from_tree_values(
    input_baseline_slice, input_polarization_slice, expected_true
):
    from xradio.measurement_set._utils._asdm._utils._bdf.load_from_pyasdm_arr_trees import (
        load_flags_subset_from_tree,
    )
    from xradio.measurement_set._utils._asdm._utils._bdf.load_from_pyasdm_subset_array import (
        define_flag_shape,
    )

    bdf_descr = _two_spw_bdf_descr()
    flags = load_flags_subset_from_tree(
        _flags_subset(FLAGGED),
        define_flag_shape(bdf_descr, (0, 1)),
        bdf_descr,
        (0, 1),
        (slice(None), input_baseline_slice, slice(None), input_polarization_slice),
    )

    expected = np.zeros(
        (
            1,
            len(range(6)[input_baseline_slice]),
            len(range(2)[input_polarization_slice]),
        ),
        dtype=bool,
    )
    for row, pol in expected_true:
        expected[0, row, pol] = True
    np.testing.assert_array_equal(flags, expected)


def test_load_flags_all_subsets_from_trees_time_selection():
    from xradio.measurement_set._utils._asdm._utils._bdf.load_from_pyasdm_arr_trees import (
        load_flags_all_subsets_from_trees,
    )
    from xradio.measurement_set._utils._asdm._utils._bdf.load_from_pyasdm_subset_array import (
        define_flag_shape,
    )

    bdf_descr = _two_spw_bdf_descr()
    subsets = [_flags_subset([(row, 1, 0, 1)]) for row in range(3)]
    flags = load_flags_all_subsets_from_trees(
        _SubsetsReader(subsets),
        define_flag_shape(bdf_descr, (0, 1)),
        bdf_descr,
        (0, 1),
        (slice(1, 3), slice(None), slice(None), slice(None)),
    )

    expected = np.zeros((2, 6, 2), dtype=bool)
    expected[0, 1, 0] = expected[1, 2, 0] = True
    np.testing.assert_array_equal(flags, expected)


@pytest.mark.parametrize(
    "input_polarization_slice, expected_true",
    [(slice(None), [(1, 0), (2, 1)]), (slice(0, 1), [(1, 0)]), (slice(1, 2), [(2, 0)])],
)
def test_load_flags_subset_from_tree_auto_only_values(
    input_polarization_slice, expected_true
):
    from xradio.measurement_set._utils._asdm._utils._bdf.load_from_pyasdm_arr_trees import (
        load_flags_subset_from_tree,
    )
    from xradio.measurement_set._utils._asdm._utils._bdf.load_from_pyasdm_subset_array import (
        define_flag_shape,
    )

    bdf_descr = _two_spw_bdf_descr()
    bdf_descr["correlation_mode"] = pyasdm.enumerations.CorrelationMode.AUTO_ONLY
    for spw in bdf_descr["basebands"][0]["spectralWindows"]:
        spw["crossPolProducts"] = []
    # rows: antennas 0-2; per row: spw 0, 1; per spw: 2 pols
    flag_words = np.zeros(12, dtype="int32")
    flag_words[[1 * 4 + 2 + 0, 2 * 4 + 2 + 1, 0 * 4 + 0 + 1]] = [16, 2**30, 1]
    flags = load_flags_subset_from_tree(
        {"flags": {"present": True, "arr": flag_words}},
        define_flag_shape(bdf_descr, (0, 1)),
        bdf_descr,
        (0, 1),
        (slice(None), slice(None), slice(None), input_polarization_slice),
    )

    expected = np.zeros((1, 3, len(range(2)[input_polarization_slice])), dtype=bool)
    for row, pol in expected_true:
        expected[0, row, pol] = True
    np.testing.assert_array_equal(flags, expected)
