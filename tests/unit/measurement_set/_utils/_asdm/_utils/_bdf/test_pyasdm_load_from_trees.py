from contextlib import nullcontext as no_raises
from unittest import mock

import numpy as np

import pytest

import pyasdm

example_guessed_shape_auto_only = {
    "auto": (5, 9, 1, 64, 2),
    "cross": None,
}
example_guessed_shape = {
    "cross": (3, 36, 1, 64, 2, 2),
    "auto": (5, 9, 1, 64, 2),
}


@pytest.mark.parametrize(
    "input_guessed_shape, expected_out_shape",
    [
        (example_guessed_shape, (3, 45, 1, 64, 2, 2)),
        (example_guessed_shape_auto_only, (5, 9, 1, 64, 2)),
    ],
)
def test_add_cross_and_auto_flag_shapes_with_cross(
    input_guessed_shape, expected_out_shape
):
    from xradio.measurement_set._utils._asdm._utils._bdf.pyasdm_load_from_trees import (
        add_cross_and_auto_flag_shapes,
    )

    added = add_cross_and_auto_flag_shapes(input_guessed_shape)
    assert added == expected_out_shape


@pytest.mark.parametrize(
    "input_shape, expected_out_shape",
    [
        (example_guessed_shape["auto"], (5, 9, 2)),
        (example_guessed_shape["cross"], (3, 36, 2)),
    ],
)
def test_full_shape_to_output_filled_flags_shape(input_shape, expected_out_shape):
    from xradio.measurement_set._utils._asdm._utils._bdf.pyasdm_load_from_trees import (
        full_shape_to_output_filled_flags_shape,
    )

    out_shape = full_shape_to_output_filled_flags_shape(input_shape)
    assert out_shape == expected_out_shape


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


def test_load_visibilities_all_subsets_from_trees():
    from xradio.measurement_set._utils._asdm._utils._bdf.pyasdm_load_from_trees import (
        load_visibilities_all_subsets_from_trees,
    )

    bdf_descr = {
        "basebands": basebands_example,
        "processor_type": "CORRELATOR",
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
                load_one_spw_from_file=False,
            )
        mock_bdf_header.getBasebandsList.assert_not_called()
        mock_bdf_header.getSubset.assert_not_called()
        mock_bdf_header.hasSubset.assert_not_called()


def test_load_vis_subset_from_tree():
    from xradio.measurement_set._utils._asdm._utils._bdf.pyasdm_load_from_trees import (
        load_vis_subset_from_tree,
    )

    pyasdm_subset = {}
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
        with pytest.raises(RuntimeError, match="not present"):
            load_vis_subset_from_tree(pyasdm_subset, guessed_shape, (0, 0), bdf_descr)
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


@pytest.mark.parametrize(
    "input_basebands, input_baseband_idx, input_spw_idx, expected_overall_spw_idx, expected_error",
    [
        (basebands_example, 0, 0, 0, no_raises()),
        (basebands_example, 0, 1, 0, no_raises()),
        (basebands_example, 1, 0, 1, no_raises()),
        (basebands_example, 3, 0, 4, no_raises()),
        (basebands_example, 4, 1, 5, no_raises()),
        (basebands_example, 6, 1, 5, pytest.raises(IndexError, match="out of range")),
        (bdf_descr_X136e["basebands"], 0, 0, 0, no_raises()),
        (bdf_descr_X136e["basebands"], 0, 1, 0, no_raises()),
        (bdf_descr_X136e["basebands"], 0, 2, 0, no_raises()),
        (bdf_descr_X136e["basebands"], 2, 0, 4, no_raises()),
        (
            bdf_descr_X136e["basebands"],
            5,
            0,
            4,
            pytest.raises(IndexError, match="out of range"),
        ),
    ],
)
def test_calculate_overall_spw_idx(
    input_basebands,
    input_baseband_idx,
    input_spw_idx,
    expected_overall_spw_idx,
    expected_error,
):
    from xradio.measurement_set._utils._asdm._utils._bdf.pyasdm_load_from_trees import (
        calculate_overall_spw_idx,
    )

    with expected_error:
        spw_idx = calculate_overall_spw_idx(
            input_basebands, input_baseband_idx, input_spw_idx
        )
        assert spw_idx == expected_overall_spw_idx


@pytest.mark.parametrize(
    "input_baseband_spw_idxs, input_bdf_descr, expected_overall_spw_idx, expected_error",
    [
        ((0, 0), bdf_descr_X136e, 0, no_raises()),
        ((0, 1), bdf_descr_X136e, 0, no_raises()),
        ((0, 2), bdf_descr_X136e, 0, no_raises()),
        ((2, 0), bdf_descr_X136e, 4, no_raises()),
        ((5, 0), bdf_descr_X136e, 4, pytest.raises(IndexError, match="out of range")),
    ],
)
def test_baseband_spw_to_overall_spw_idx(
    input_baseband_spw_idxs, input_bdf_descr, expected_overall_spw_idx, expected_error
):
    from xradio.measurement_set._utils._asdm._utils._bdf.pyasdm_load_from_trees import (
        baseband_spw_to_overall_spw_idx,
    )

    with expected_error:
        spw_idx = baseband_spw_to_overall_spw_idx(
            input_baseband_spw_idxs, input_bdf_descr
        )
        assert spw_idx == expected_overall_spw_idx


def test_load_visibilities_all_subsets_from_trees_X136e():
    from xradio.measurement_set._utils._asdm._utils._bdf.robust_load_data_flags import (
        load_visibilities_all_subsets_from_trees,
    )

    with (
        mock.patch("pyasdm.bdf.BDFReader") as mock_bdf_reader,
        mock.patch("pyasdm.bdf.BDFHeader") as mock_bdf_header,
    ):
        # For load_vis_subset, etc.
        mock_bdf_reader.hasSubset.side_effect = [True, False]
        mock_bdf_reader.getSubset.side_effect = [
            {
                "autoData": {
                    "present": True,
                    "arr": np.zeros((1000000), dtype="float64"),
                },
                "crossData": {
                    "present": True,
                    "arr": np.zeros((1000000), dtype="float64"),
                },
            },
            None,
        ]
        visibilities = load_visibilities_all_subsets_from_trees(
            mock_bdf_reader,
            (1, 36, 9, 4, 2, 512, 2, 2),
            (0, 1),
            bdf_descr_X136e,
            load_one_spw_from_file=False,
        )

        assert isinstance(visibilities, np.ndarray)
        assert visibilities.size == 92160
        assert visibilities.shape == (1, 45, 1024, 2)
        assert visibilities.dtype == np.dtype("complex128")

        mock_bdf_reader.hasSubset.assert_called()
        assert mock_bdf_reader.hasSubset.call_count == 2
        mock_bdf_reader.getSubset.assert_called()
        assert mock_bdf_reader.getSubset.call_count == 1


def test_load_visibilities_all_subsets_from_trees_X136e_error():
    from xradio.measurement_set._utils._asdm._utils._bdf.robust_load_data_flags import (
        load_visibilities_all_subsets_from_trees,
    )

    with (
        mock.patch("pyasdm.bdf.BDFReader") as mock_bdf_reader,
        mock.patch("pyasdm.bdf.BDFHeader") as mock_bdf_header,
    ):
        # For load_vis_subset, etc.
        mock_bdf_reader.hasSubset.side_effect = [True, False]
        mock_bdf_reader.getSubset.side_effect = [ValueError, None]
        visibilities = load_visibilities_all_subsets_from_trees(
            mock_bdf_reader,
            (1, 36, 9, 4, 2, 512, 2, 2),
            (0, 1),
            bdf_descr_X136e,
            load_one_spw_from_file=False,
        )
        assert visibilities == None

        mock_bdf_reader.hasSubset.assert_called()
        assert mock_bdf_reader.hasSubset.call_count == 1
        mock_bdf_reader.getSubset.assert_called()
        assert mock_bdf_reader.getSubset.call_count == 1


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


# load_vis_subset_cross_data_from_tree
# load_vis_subset_auto_data_from_tree
@pytest.mark.parametrize(
    "input_bdf_descr, input_baseband_idx, input_overall_spw_idx, input_flag_array_len, expected_additions",
    [
        (
            bdf_descr_X136e,
            0,
            0,
            120,
            {
                "cross": {"before": 0, "after": 0},
                "auto": {"before": 0, "after": 12},
            },
        ),
        (
            bdf_descr_X136e,
            1,
            1,
            120,
            {
                "cross": {"before": 0, "after": 0},
                "auto": {"before": 2, "after": 10},
            },
        ),
        (
            bdf_descr_X136e,
            0,
            0,
            80,
            {
                "cross": {"before": 0, "after": 0},
                "auto": {"before": 0, "after": 8},
            },
        ),
        (
            bdf_descr_X136e,
            1,
            1,
            80,
            {
                "cross": {"before": 0, "after": 0},
                "auto": {"before": 2, "after": 6},
            },
        ),
        (
            bdf_descr_X64c6,
            1,
            1,
            450,  # per-SPW: 360 (cross) + 90 (sd)
            {
                "cross": {"before": 2, "after": 8},
                "auto": {"before": 2, "after": 8},
            },
        ),
        (
            bdf_descr_X64c6,
            1,
            1,
            360,  # per-BB: 288 (cross) + 72 (sd)
            {
                "cross": {"before": 2, "after": 6},
                "auto": {"before": 2, "after": 6},
            },
        ),
    ],
)
def test_calculate_offset_additions_cross_sd(
    input_bdf_descr,
    input_baseband_idx,
    input_overall_spw_idx,
    input_flag_array_len,
    expected_additions,
):
    from xradio.measurement_set._utils._asdm._utils._bdf.pyasdm_load_from_trees import (
        calculate_offset_additions_cross_sd,
    )

    additions = calculate_offset_additions_cross_sd(
        input_bdf_descr, input_baseband_idx, input_overall_spw_idx, input_flag_array_len
    )
    assert additions == expected_additions


def test_load_flags_all_subsets_from_trees_error():
    from xradio.measurement_set._utils._asdm._utils._bdf.pyasdm_load_from_trees import (
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
        vis = load_flags_all_subsets_from_trees(
            mock_bdf_reader, guessed_shape, (0, 0), bdf_descr
        )
        assert vis is None
        mock_bdf_header.getBasebandsList.assert_not_called()
        assert mock_bdf_reader.getSubset.call_count == 1
        assert mock_bdf_reader.hasSubset.call_count == 1


def test_load_flags_all_subsets_from_trees_X136e():
    from xradio.measurement_set._utils._asdm._utils._bdf.robust_load_data_flags import (
        load_flags_all_subsets_from_trees,
    )

    with (
        mock.patch("pyasdm.bdf.BDFReader") as mock_bdf_reader,
        mock.patch("pyasdm.bdf.BDFHeader") as mock_bdf_header,
    ):
        # For load_vis_subset, etc.
        mock_bdf_reader.hasSubset.side_effect = [True, False]
        mock_bdf_reader.getSubset.side_effect = [
            {"autoData": {"present": True, "arr": np.zeros((73728))}}
        ]
        guessed_shape = {
            "auto": (1, 9, 4, 2, 512, 2),
            "cross": (1, 36, 4, 2, 512, 2, 2),
        }
        flags = load_flags_all_subsets_from_trees(
            mock_bdf_reader,
            guessed_shape,
            bdf_descr_X136e,
            (0, 1),
        )
        assert isinstance(flags, np.ndarray)
        assert flags.size == 46080
        assert flags.shape == (1, 45, 512, 2)
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
            92160,
            (1, 45, 1024, 2),
            no_raises(),
        ),
        (
            {"flags": {"present": True, "arr": np.zeros((120))}},
            flags_input_guessed_shape_x136e,
            bdf_descr_X136e,
            20480,
            (1, 10, 1024, 2),
            no_raises(),
        ),
        (
            {"flags": {"present": True, "arr": np.zeros((80))}},
            flags_input_guessed_shape_x136e,
            bdf_descr_X136e,
            20480,
            (1, 10, 1024, 2),
            no_raises(),
        ),
        (
            {"flags": {"present": True, "arr": np.zeros((120))}},
            flags_input_guessed_shape_2times_x136e,
            bdf_descr_X136e,
            20480,
            (1, 10, 1024, 2),
            pytest.raises(ValueError, match="must have the same shape"),
        ),
        (
            {"flags": {"present": True, "arr": np.zeros((30))}},
            flags_input_guessed_shape_x136e,
            bdf_descr_autodata_3pol_pseudo_X136e,
            2560,
            (1, 10, 64, 4),
            no_raises(),
        ),
        (
            {"flags": {"present": True, "arr": np.zeros((3))}},
            flags_input_guessed_shape_x136e,
            bdf_descr_autodata_3pol_pseudo_X136e,
            2560,
            (1, 10, 64, 4),
            pytest.raises(RuntimeError, match="Unexpected flags array"),
        ),
        (
            {"flags": {"present": True, "arr": np.zeros((450))}},
            flags_input_guessed_shape_x136e,
            bdf_descr_X64c6,
            184320,
            (1, 45, 2048, 2),
            no_raises(),
        ),
        (
            {"flags": {"present": True, "arr": np.zeros((360))}},
            flags_input_guessed_shape_x136e,
            bdf_descr_X64c6,
            184320,
            (1, 45, 2048, 2),
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
    from xradio.measurement_set._utils._asdm._utils._bdf.pyasdm_load_from_trees import (
        load_flags_subset_from_tree,
    )

    with expected_error:
        flags = load_flags_subset_from_tree(
            input_subset, input_guessed_shape, input_bdf_descr, (0, 0)
        )

        assert isinstance(flags, np.ndarray)
        assert flags.dtype == np.dtype("bool")
        assert flags.size == expected_size
        assert flags.shape == expected_shape
