from contextlib import nullcontext as no_raises
from unittest import mock

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
                    "numSpectralPoint": 128,
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
                    "numSpectralPoint": 64,
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
                    "numSpectralPoint": 128,
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
                    "numSpectralPoint": 128,
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
                    "numSpectralPoint": 128,
                    "numBin": 1,
                    "sideband": pyasdm.enumerations.NetSideband.USB,
                    "sw": "1",
                }
            ],
        },
    ],
}

# TODO: re-do
bdf_descr_radiometer_pseudo_X136e = {
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


# TODO: re-do
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


# distorted shape: 1 time, 1 baseline, 1 ant for now
guessed_shape_base = (1, 1, 1, 4, 2, 64, 2, 2)
guessed_shape_3pol = (1, 1, 1, 4, 2, 64, 3, 2)
guessed_shape_2times = (2, 1, 1, 4, 2, 64, 2, 2)


@pytest.mark.parametrize(
    "input_bdf_descr, input_component, input_overall_spw_idx, input_elements_count, input_guessed_shape, input_fromfile_array_len, expected_error",
    [
        (
            bdf_descr_X136e,
            "autoData",
            1,
            64,
            guessed_shape_base,
            64 * 2,
            no_raises(),
        ),
        (
            bdf_descr_X136e,
            "autoData",
            1,
            64,
            guessed_shape_2times,
            64 * 2,
            no_raises(),
        ),
        (
            bdf_descr_X136e,
            "crossData",
            1,
            64,
            guessed_shape_base,
            64 * 2 * 2,
            no_raises(),
        ),
        (
            bdf_descr_radiometer_pseudo_X136e,
            "crossData",
            0,
            64,
            guessed_shape_base,
            64 * 2,
            no_raises(),
        ),
        (
            bdf_descr_autodata_3pol_pseudo_X136e,
            "autoData",
            0,
            64,
            guessed_shape_3pol,
            64 * 2 * 2,
            no_raises(),#pytest.raises(StopIteration),
        ),
    ],
)
def test_load_visibilities_one_spw_to_ndarray(
    input_bdf_descr,
    input_component,
    input_overall_spw_idx,
    input_elements_count,
    input_guessed_shape,
    input_fromfile_array_len,
    expected_error,
):
    from xradio.measurement_set._utils._asdm._utils._bdf.pyasdm_get_ndarray import (
        load_visibilities_one_spw_to_ndarray,
    )

    with (
        mock.patch("typing.BinaryIO") as mock_bdf_file,
        mock.patch("numpy.fromfile") as mock_np_fromfile,
    ):
        # mock up to 2 fromfile calls (2 times or 2 baselines/ant)
        mock_np_fromfile.side_effect = [
            np.zeros(input_fromfile_array_len, dtype="float64")
        ] * 2
        with expected_error:
            visibilities = load_visibilities_one_spw_to_ndarray(
                input_component,
                input_overall_spw_idx,
                mock_bdf_file,
                np.float64,
                input_elements_count,
                input_bdf_descr,
                input_guessed_shape,
            )
            assert isinstance(visibilities, np.ndarray)
            if input_component == "crossData":
                assert visibilities.size == input_elements_count * 2
            else:
                assert visibilities.size >= input_elements_count * 2
            assert (visibilities == 0+0j).all()
    
