import numpy as np

import pytest

import pyasdm

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
