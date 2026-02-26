from contextlib import nullcontext as no_raises

import pytest

import pyasdm

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


@pytest.mark.parametrize(
    "input_spw_idx, input_basebands, expected_baseband_idx, expected_spw_idx",
    [
        (0, [], 0, 0),
        (1, [], 0, 0),
        (0, basebands_example_X2197, 0, 0),
        (1, basebands_example_X2197, 1, 0),
        (2, basebands_example_X2197, 2, 0),
        (3, basebands_example_X2197, 2, 1),
        (4, basebands_example_X2197, 3, 0),
        (5, basebands_example_X2197, 0, 0),
    ],
)
def test_find_spw_in_basebands_list_empty(
    input_spw_idx, input_basebands, expected_baseband_idx, expected_spw_idx
):
    from xradio.measurement_set._utils._asdm._utils._bdf.robust_load_data_flags import (
        find_spw_in_basebands_list,
    )

    baseband_idx, spw_idx = find_spw_in_basebands_list(
        input_spw_idx, input_basebands, "bogus_path_non_existant.nope"
    )
    assert baseband_idx == expected_baseband_idx
    assert spw_idx == expected_spw_idx
