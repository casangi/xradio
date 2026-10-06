from contextlib import nullcontext as no_raises
from unittest import mock

import numpy as np
import pyasdm
import pytest

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

empty_slice = (slice(None), slice(None), slice(None), slice(None))
slice_with_int_baseline = (slice(None), 0, slice(None), slice(None))
slice_with_int_pol = (slice(None), slice(None), slice(None), 0)


@pytest.mark.parametrize(
    "input_bdf_descr, input_component, input_overall_spw_idx, input_elements_count, input_guessed_shape, input_fromfile_array_len, input_array_slice, expected_error",
    [
        (
            bdf_descr_X136e,
            "autoData",
            1,
            64,
            guessed_shape_base,
            64 * 2,
            empty_slice,
            no_raises(),
        ),
        (
            bdf_descr_X136e,
            "autoData",
            1,
            64,
            guessed_shape_2times,
            64 * 2,
            empty_slice,
            no_raises(),
        ),
        (
            bdf_descr_X136e,
            "crossData",
            1,
            64,
            guessed_shape_base,
            64 * 2 * 2,
            empty_slice,
            no_raises(),
        ),
        (
            bdf_descr_radiometer_pseudo_X136e,
            "crossData",
            0,
            64,
            guessed_shape_base,
            64 * 2,
            empty_slice,
            no_raises(),
        ),
        (
            bdf_descr_radiometer_pseudo_X136e,
            "crossData",
            0,
            64,
            guessed_shape_base,
            64 * 2,
            slice_with_int_pol,
            no_raises(),
        ),
        (
            bdf_descr_radiometer_pseudo_X136e,
            "autoData",
            0,
            64,
            guessed_shape_base,
            64 * 2,
            slice_with_int_pol,
            no_raises(),
        ),
        (
            bdf_descr_autodata_3pol_pseudo_X136e,
            "autoData",
            0,
            64,
            guessed_shape_3pol,
            64 * 2 * 2,
            empty_slice,
            no_raises(),
        ),
        (
            bdf_descr_autodata_3pol_pseudo_X136e,
            "autoData",
            0,
            64,
            guessed_shape_3pol,
            64 * 2 * 2,
            slice_with_int_baseline,
            no_raises(),
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
    input_array_slice,
    expected_error,
):
    from xradio.measurement_set._utils._asdm._utils._bdf.pyasdm_get_ndarray_load_function import (
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
                ["autoData", "crossData"],
                input_guessed_shape,
                input_array_slice,
            )

            assert isinstance(visibilities, np.ndarray)
            polarization_multiplier = 1 if isinstance(input_array_slice[3], int) else 2
            assert visibilities.size >= input_elements_count * polarization_multiplier
            assert (visibilities == 0 + 0j).all()


def _write_component(path, values, offset_bytes):
    # some unrelated bytes before the binary component, which starts at offset_bytes
    np.concatenate(
        [np.full(offset_bytes // 4, -1, dtype="float32"), np.array(values, "float32")]
    ).tofile(path)


@pytest.mark.parametrize("input_baseline_slice", [slice(None), slice(1, 3)])
@pytest.mark.parametrize("input_frequency_slice", [slice(None), slice(2, 5)])
def test_load_vis_one_spw_cross_data_from_tree_values(
    tmp_path, input_baseline_slice, input_frequency_slice
):
    from xradio.measurement_set._utils._asdm._utils._bdf.pyasdm_get_ndarray_load_function import (
        _load_vis_one_spw_cross_data_from_tree,
    )

    spw_chan_lens = [4, 6]
    baseline_len, polarization_len = 3, 2
    values = []
    for baseline in range(baseline_len):
        for spw, channel_len in enumerate(spw_chan_lens):
            for channel in range(channel_len):
                for pol in range(polarization_len):
                    real = 1000 * baseline + 100 * spw + 10 * channel + pol
                    values += [real, -real]
    path = tmp_path / "crossData"
    _write_component(path, values, 20)

    with open(path, "rb") as bdf_file:
        bdf_file.seek(20)
        vis = _load_vis_one_spw_cross_data_from_tree(
            bdf_file,
            (1, baseline_len, 3, 1, 2, 6, polarization_len, 2),
            spw_chan_lens,
            1,
            np.dtype("float32"),
            1,
            pyasdm.enumerations.ProcessorType.CORRELATOR,
            (slice(None), input_baseline_slice, input_frequency_slice, slice(None)),
        )

    baseline = np.arange(baseline_len)[input_baseline_slice][:, None, None]
    channel = np.arange(6)[input_frequency_slice][None, :, None]
    pol = np.arange(polarization_len)[None, None, :]
    real = 1000 * baseline + 100 + 10 * channel + pol
    np.testing.assert_array_equal(vis, (real - 1j * real)[np.newaxis])


@pytest.mark.parametrize("input_antenna_slice", [slice(None), slice(1, 3)])
@pytest.mark.parametrize("input_frequency_slice", [slice(None), slice(2, 5)])
def test_load_vis_one_spw_auto_data_from_tree_values(
    tmp_path, input_antenna_slice, input_frequency_slice
):
    from xradio.measurement_set._utils._asdm._utils._bdf.pyasdm_get_ndarray_load_function import (
        _load_vis_one_spw_auto_data_from_tree,
    )

    spw_chan_lens = [4, 6]
    antenna_len, polarization_len = 3, 2
    values = [
        1000 * antenna + 100 * spw + 10 * channel + pol
        for antenna in range(antenna_len)
        for spw, channel_len in enumerate(spw_chan_lens)
        for channel in range(channel_len)
        for pol in range(polarization_len)
    ]
    path = tmp_path / "autoData"
    _write_component(path, values, 20)

    with open(path, "rb") as bdf_file:
        bdf_file.seek(20)
        vis = _load_vis_one_spw_auto_data_from_tree(
            bdf_file,
            (1, 3, antenna_len, 1, 2, 6, polarization_len, 2),
            spw_chan_lens,
            1,
            np.dtype("float32"),
            None,
            (slice(None), input_antenna_slice, input_frequency_slice, slice(None)),
        )

    antenna = np.arange(antenna_len)[input_antenna_slice][:, None, None]
    channel = np.arange(6)[input_frequency_slice][None, :, None]
    pol = np.arange(polarization_len)[None, None, :]
    np.testing.assert_array_equal(
        vis, (1000 * antenna + 100 + 10 * channel + pol)[np.newaxis]
    )


def test_load_vis_one_spw_cross_data_from_tree_radiometer_values(tmp_path):
    from xradio.measurement_set._utils._asdm._utils._bdf.pyasdm_get_ndarray_load_function import (
        _load_vis_one_spw_cross_data_from_tree,
    )

    spw_chan_lens = [4, 6]
    baseline_len, polarization_len = 3, 2
    values = [
        1000 * baseline + 100 * spw + 10 * channel + pol
        for baseline in range(baseline_len)
        for spw, channel_len in enumerate(spw_chan_lens)
        for channel in range(channel_len)
        for pol in range(polarization_len)
    ]
    path = tmp_path / "crossData"
    _write_component(path, values, 20)

    with open(path, "rb") as bdf_file:
        bdf_file.seek(20)
        vis = _load_vis_one_spw_cross_data_from_tree(
            bdf_file,
            (1, baseline_len, 3, 1, 2, 6, polarization_len, 2),
            spw_chan_lens,
            1,
            np.dtype("float32"),
            1,
            pyasdm.enumerations.ProcessorType.RADIOMETER,
            (slice(None), slice(1, 3), slice(2, 5), slice(None)),
        )

    baseline = np.arange(1, 3)[:, None, None]
    channel = np.arange(2, 5)[None, :, None]
    pol = np.arange(polarization_len)[None, None, :]
    np.testing.assert_array_equal(
        vis, (1000 * baseline + 100 + 10 * channel + pol)[np.newaxis]
    )
