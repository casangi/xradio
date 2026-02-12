"""
Module that could be moved to a BDFReader.getNDArray() method in
pyasdm or could stay here, as a function injected to either
BDFReader.getSubset() or the possible new getNDArray().
"""

import numpy as np
import os
import typing

import pyasdm

import toolviper.utils.logger as logger

from .basebands import find_spw_in_basebands_list


def load_visibilities_one_spw_to_ndarray(
    component_name: str,
    overall_spw_idx: int,
    bdf_file: typing.BinaryIO,
    data_type: np.dtype,  # ???
    elements_count: int,  # ???
    bdf_descr: dict,
    guessed_shape: tuple,
) -> np.ndarray:

    spw_chan_lens = [
        bdf_descr["basebands"][bb_idx]["spectralWindows"][spw_idx]["numSpectralPoint"]
        for bb_idx in range(0, len(bdf_descr["basebands"]))
        for spw_idx in range(0, len(bdf_descr["basebands"][bb_idx]["spectralWindows"]))
    ]

    if component_name == "autoData":
        vis_one_spw = load_vis_one_spw_auto_data_from_tree(
            bdf_file,
            guessed_shape,
            spw_chan_lens,
            overall_spw_idx,
            data_type,
            elements_count,
        )

    elif component_name == "crossData":
        baseband_spw_idxs = find_spw_in_basebands_list(
            overall_spw_idx, bdf_descr["basebands"], bdf_file.name
        )

        baseband_description = bdf_descr["basebands"][baseband_spw_idxs[0]]
        spw_descr = baseband_description["spectralWindows"][baseband_spw_idxs[1]]
        scale_factor = spw_descr["scaleFactor"] or 1
        processor_type = bdf_descr["processor_type"]

        vis_one_spw = load_vis_one_spw_cross_data_from_tree(
            bdf_file,
            guessed_shape,
            spw_chan_lens,
            overall_spw_idx,
            data_type,
            scale_factor,
            processor_type,
        )

    return vis_one_spw


def load_vis_one_spw_auto_data_from_tree(
    bdf_file: np.ndarray,
    guessed_shape: tuple[int, ...],
    spw_chan_lens: list[int],
    overall_spw_idx: int,
    data_type: np.dtype,
    elements_count: int,
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
    antenna_len = guessed_shape[2]
    spw_channel_len = spw_chan_lens[overall_spw_idx]

    offset = 0
    vis_subset_integrations = []
    component_offset = bdf_file.tell()
    for time_idx in np.arange(0, guessed_shape[0]):
        vis_auto_strides = []
        for antenna_idx in np.arange(antenna_len):
            offset += auto_offset_addition_before
            one_antenna_count = spw_channel_len * sd_polarization_len
            bdf_file.seek(component_offset + offset, os.SEEK_SET)
            spw_floats = np.fromfile(bdf_file, dtype=data_type, count=one_antenna_count)
            if polarization_len != 3:
                spw_floats = spw_floats.reshape((spw_channel_len, sd_polarization_len))
            else:
                # autoData: "The choice of a real- vs. complex-valued datum is dependent upon the
                # polarization product...parallel-hand polarizations are real-valued, while cross-hand
                # polarizations are complex-valued".
                spw_floats = spw_floats.reshape((spw_channel_len, sd_polarization_len))
                spw_vis = np.concatenate(
                    [
                        spw_floats[:, [0]],
                        spw_floats[:, [1]] + 1j * spw_floats[:, [2]],
                        spw_floats[:, [3]],
                    ],
                    axis=1,
                )

            vis_auto_strides.append(spw_floats)
            offset += auto_offset_addition_after

        vis_subset_integrations.append(np.stack(vis_auto_strides))

    if len(vis_subset_integrations) == 1:
        vis_auto = vis_subset_integrations[0][np.newaxis, :]
    else:
        vis_auto = np.stack(vis_subset_integrations)

    return vis_auto


def load_vis_one_spw_cross_data_from_tree(
    bdf_file: np.ndarray,
    guessed_shape: tuple[int, ...],
    spw_chan_lens: list[int],
    overall_spw_idx: int,
    data_type: np.dtype,
    scale_factor: float,
    processor_type: pyasdm.enumerations.ProcessorType,
) -> np.ndarray:

    polarization_len = guessed_shape[-2]
    cross_offset_addition_before = (
        np.sum(spw_chan_lens[0:overall_spw_idx], dtype=int) * polarization_len * 2
    )
    cross_offset_addition_after = (
        np.sum(spw_chan_lens[overall_spw_idx:], dtype=int) * polarization_len * 2
    )
    spw_channel_len = spw_chan_lens[overall_spw_idx]
    time_len = guessed_shape[0]
    baseline_len = guessed_shape[1]
    offset = 0
    component_offset = bdf_file.tell()
    for time_idx in np.arange(0, time_len):
        vis_strides = []
        for baseline_idx in np.arange(baseline_len):
            if processor_type == pyasdm.enumerations.ProcessorType.CORRELATOR:
                offset += cross_offset_addition_before
                one_baseline_count = spw_channel_len * polarization_len * 2
                bdf_file.seek(component_offset + offset, os.SEEK_SET)
                spw_vis = np.fromfile(
                    bdf_file, dtype=data_type, count=one_baseline_count
                )
                spw_vis = spw_vis.reshape((int(spw_vis.size / 2), 2))
                spw_vis = spw_vis[:, 0] + 1j * spw_vis[:, 1]
                spw_vis /= scale_factor
                vis_strides.append(spw_vis.reshape((spw_channel_len, polarization_len)))
                offset += cross_offset_addition_after

            else:
                # radiometer / spectrometer
                offset += cross_offset_addition_before / 2
                spw_values = cross_data_arr[
                    offset : offset + spw_channel_len * polarization_len
                ]
                spw_values = (
                    spw_values.reshape((spw_channel_len, polarization_len))
                    / scale_factor
                )

                vis_strides.append(spw_values)
                offset += cross_offset_addition_after / 2

    vis_cross = np.stack(vis_strides)
    vis_cross = vis_cross.reshape((1, *vis_cross.shape))

    return vis_cross
