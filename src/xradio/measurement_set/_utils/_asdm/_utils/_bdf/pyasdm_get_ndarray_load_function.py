"""
Module that could be moved to a BDFReader.getNDArray() method in
pyasdm or could stay here, as a function injected to either
BDFReader.getNDArrays() (the "loadOneSPWFunction").

Defines the function responsible for loading visibilities from the
crossData and autoData binary component arrays, to be used in
BDFReader.getNDArray().
"""

import numpy as np
import os
import typing

import pyasdm

from .basebands_spws import find_spw_in_basebands_list


def load_visibilities_one_spw_to_ndarray(
    component_name: str,
    overall_spw_idx: int,
    bdf_file: typing.BinaryIO,
    data_type: np.dtype,  # ???
    elements_count: int,  # ???
    bdf_descr: dict,
    guessed_shape: tuple[int, ...],
    array_slice: tuple[slice, ...],
) -> np.ndarray:
    """
    Function meant to be passed to pyasdm.BDFReader.getNDArrays as
    loader function that does "load data from one SPW only, skipping
    all other SPWs".

    It is expected that pyasdm.BDFReader.getNDArrays will call it for
    the required binary components (crossData and/or autoData) and
    assemble them into an MSv4 style ndarray (with dims time, baseline,
    frequency, polarization).
    """
    spw_chan_lens = [
        bdf_descr["basebands"][bb_idx]["spectralWindows"][spw_idx]["numSpectralPoint"]
        for bb_idx in range(0, len(bdf_descr["basebands"]))
        for spw_idx in range(0, len(bdf_descr["basebands"][bb_idx]["spectralWindows"]))
    ]

    if component_name == "autoData":
        vis_one_spw = _load_vis_one_spw_auto_data_from_tree(
            bdf_file,
            guessed_shape,
            spw_chan_lens,
            overall_spw_idx,
            data_type,
            elements_count,
            array_slice,
        )

    elif component_name == "crossData":
        baseband_spw_idxs = find_spw_in_basebands_list(
            overall_spw_idx, bdf_descr["basebands"], bdf_file.name
        )

        baseband_description = bdf_descr["basebands"][baseband_spw_idxs[0]]
        spw_descr = baseband_description["spectralWindows"][baseband_spw_idxs[1]]
        scale_factor = spw_descr["scaleFactor"] or 1
        processor_type = bdf_descr["processor_type"]

        vis_one_spw = _load_vis_one_spw_cross_data_from_tree(
            bdf_file,
            guessed_shape,
            spw_chan_lens,
            overall_spw_idx,
            data_type,
            scale_factor,
            processor_type,
            array_slice,
        )

    return vis_one_spw


def _load_vis_one_spw_auto_data_from_tree(
    bdf_file: np.ndarray,
    guessed_shape: tuple[int, ...],
    spw_chan_lens: list[int],
    overall_spw_idx: int,
    data_type: np.dtype,
    elements_count: int,
    array_slice: tuple[int, ...],
) -> np.ndarray:

    polarization_len = guessed_shape[-2]
    if polarization_len == 3:
        sd_polarization_len = 4
    else:
        sd_polarization_len = polarization_len
    auto_offset_addition_before = (
        np.sum(spw_chan_lens[0:overall_spw_idx], dtype=int) * sd_polarization_len
    )
    auto_offset_addition_after = (
        np.sum(spw_chan_lens[overall_spw_idx:], dtype=int) * sd_polarization_len
    )
    auto_offset_addition_both = auto_offset_addition_before + auto_offset_addition_after
    antenna_len = guessed_shape[2]
    spw_channel_len = spw_chan_lens[overall_spw_idx]

    time_len = guessed_shape[0]
    vis_subset_integrations = []
    time_min = array_slice[0].start or 0
    time_max = array_slice[0].stop or time_len
    antenna_min = array_slice[1].start or 0
    antenna_max = array_slice[1].stop or antenna_len
    frequency_min = array_slice[2].start or 0
    frequency_max = array_slice[2].stop or spw_channel_len
    polarization_min = array_slice[3].start or 0
    polarization_max = array_slice[3].stop or sd_polarization_len
    component_offset = bdf_file.tell()
    for time_idx in np.arange(time_min, time_max):
        vis_auto_strides = []
        for antenna_idx in np.arange(antenna_min, antenna_max):
            offset = (
                time_idx * antenna_idx * auto_offset_addition_both
                + auto_offset_addition_before
            )
            one_antenna_count = (frequency_max - frequency_min) * sd_polarization_len
            bdf_file.seek(component_offset + offset, os.SEEK_SET)
            spw_floats = np.fromfile(bdf_file, dtype=data_type, count=one_antenna_count)

            first_frequency = offset + (frequency_min * sd_polarization_len)
            last_frequency = offset + (frequency_max * sd_polarization_len)
            if polarization_len != 3:
                spw_values = spw_floats.reshape(
                    (frequency_max - frequency_min, sd_polarization_len)
                )
            else:
                # autoData: "The choice of a real- vs. complex-valued datum is dependent upon the
                # polarization product...parallel-hand polarizations are real-valued, while cross-hand
                # polarizations are complex-valued".
                spw_floats = spw_floats.reshape(
                    (frequency_max - frequency_min, sd_polarization_len)
                )
                spw_values = np.concatenate(
                    [
                        spw_floats[:, [0]],
                        spw_floats[:, [1]] + 1j * spw_floats[:, [2]],
                        spw_floats[:, [3]],
                    ],
                    axis=1,
                )

            vis_auto_strides.append(spw_values)

        vis_subset_integrations.append(np.stack(vis_auto_strides))

    if len(vis_subset_integrations) == 1:
        vis_auto = vis_subset_integrations[0][np.newaxis, :]
    else:
        vis_auto = np.stack(vis_subset_integrations)

    return vis_auto


def _load_vis_one_spw_cross_data_from_tree(
    bdf_file: np.ndarray,
    guessed_shape: tuple[int, ...],
    spw_chan_lens: list[int],
    overall_spw_idx: int,
    data_type: np.dtype,
    scale_factor: float,
    processor_type: pyasdm.enumerations.ProcessorType,
    array_slice: tuple[slice, ...],
) -> np.ndarray:

    polarization_len = guessed_shape[-2]
    cross_offset_addition_before = (
        np.sum(spw_chan_lens[0:overall_spw_idx], dtype=int) * polarization_len * 2
    )
    cross_offset_addition_after = (
        np.sum(spw_chan_lens[overall_spw_idx:], dtype=int) * polarization_len * 2
    )
    cross_offset_both = cross_offset_addition_before + cross_offset_addition_after
    spw_channel_len = spw_chan_lens[overall_spw_idx]
    time_len = guessed_shape[0]
    baseline_len = guessed_shape[1]
    time_min = array_slice[0].start or 0
    time_max = array_slice[0].stop or time_len
    baseline_min = array_slice[1].start or 0
    baseline_max = array_slice[1].stop or baseline_len
    frequency_min = array_slice[2].start or 0
    frequency_max = array_slice[2].stop or spw_channel_len
    polarization_min = array_slice[3].start or 0
    polarization_max = array_slice[3].stop or polarization_len
    component_offset = bdf_file.tell()
    for time_idx in np.arange(time_min, time_max):
        vis_strides = []
        for baseline_idx in np.arange(baseline_min, baseline_max):
            if processor_type == pyasdm.enumerations.ProcessorType.CORRELATOR:
                offset = (
                    time_idx * baseline_idx * cross_offset_both
                    + cross_offset_addition_before
                )
                one_baseline_count = (
                    (frequency_max - frequency_min) * polarization_len * 2
                )
                bdf_file.seek(component_offset + offset, os.SEEK_SET)
                spw_vis = np.fromfile(
                    bdf_file, dtype=data_type, count=one_baseline_count
                )
                spw_vis = spw_vis.reshape((int(spw_vis.size / 2), 2))
                spw_vis = spw_vis[:, 0] + 1j * spw_vis[:, 1]
                spw_vis /= scale_factor
                vis_strides.append(
                    spw_vis.reshape((frequency_max - frequency_min, polarization_len))
                )

            else:
                # radiometer / spectrometer
                offset = int(
                    (
                        time_idx * baseline_idx * cross_offset_both
                        + cross_offset_addition_before
                    )
                    / 2
                )
                one_baseline_count = (frequency_max - frequency_min) * polarization_len
                spw_values = np.fromfile(
                    bdf_file, dtype=data_type, count=one_baseline_count
                )
                spw_values = (
                    spw_values.reshape(
                        (frequency_max - frequency_min, polarization_len)
                    )
                    / scale_factor
                )

                vis_strides.append(spw_values)

    vis_cross = np.stack(vis_strides)
    vis_cross = vis_cross.reshape((1, *vis_cross.shape))

    return vis_cross
