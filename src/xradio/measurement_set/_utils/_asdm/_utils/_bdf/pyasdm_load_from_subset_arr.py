"""
Loads visibility/flags from the 'arr' arrays produced by pyasdm.BDFReader.getSubset().
These 'arr' 1d arrays contain the data for all the SPWs. These arrays are reshaped and
the relevant SPW is then selected.
"""

import traceback

import numpy as np

import pyasdm

from xradio._utils.logging import xradio_logger
from .shapes import (
    add_cross_and_auto_flag_shapes,
    full_shape_to_output_filled_flags_shape,
)


def load_visibilities_all_subsets(
    bdf_reader: pyasdm.bdf.BDFReader,
    guessed_shape: tuple[int, ...],
    baseband_spw_idxs: tuple[int, int],
    bdf_descr: dict,
) -> np.ndarray:

    baseband_description = bdf_descr["basebands"][baseband_spw_idxs[0]]
    spw_descr = baseband_description["spectralWindows"][baseband_spw_idxs[1]]
    scale_factor = spw_descr["scaleFactor"] or 1
    processor_type = bdf_descr["processor_type"]

    vis_per_subset = []
    while bdf_reader.hasSubset():
        try:
            subset = bdf_reader.getSubset(loadOnlyComponents={"autoData", "crossData"})
        except ValueError as exc:
            trace = traceback.format_exc()
            xradio_logger().warning(
                f"Error in getSubset for {bdf_reader.getPath()=}  when trying to load "
                f"visibilities. {exc=}" + trace
            )
            return None

        vis_subset = _load_vis_subset(
            subset,
            guessed_shape,
            baseband_spw_idxs,
            scale_factor,
            processor_type,
        )

        vis_per_subset.append(vis_subset)

    bdf_vis = np.concatenate(vis_per_subset)
    return bdf_vis


def _load_vis_subset(
    subset: dict,
    guessed_shape: tuple[int, ...],
    baseband_spw_idxs: tuple[int, int],
    scale_factor: float,
    processor_type: pyasdm.enumerations.ProcessorType,
) -> np.ndarray:

    if "autoData" in subset and subset["autoData"]["present"]:
        vis_subset_auto = _load_vis_subset_auto_data(
            subset["autoData"]["arr"], guessed_shape, baseband_spw_idxs
        )

    else:
        # Never allowed for ALMA (BDF doc) and seems so in real life
        raise RuntimeError("autoData not present!")

    vis_subset_cross = None
    if "crossData" in subset and subset["crossData"]["present"]:
        vis_subset_cross = _load_vis_subset_cross_data(
            subset["crossData"]["arr"],
            guessed_shape,
            baseband_spw_idxs,
            scale_factor,
            processor_type,
        )

    if vis_subset_cross is None:
        vis_subset = vis_subset_auto
    else:
        vis_subset = np.concatenate([vis_subset_cross, vis_subset_auto], axis=1)

    return vis_subset


def _load_vis_subset_cross_data(
    cross_data_arr: np.ndarray,
    guessed_shape: tuple[int, ...],
    baseband_spw_idxs: tuple[int, int],
    scale_factor: float,
    processor_type: pyasdm.enumerations.ProcessorType,
) -> np.ndarray:

    cross_shape = guessed_shape[0:2] + guessed_shape[3:]
    cross_len = np.prod(cross_shape)
    cross_values = cross_data_arr[:cross_len].reshape(cross_shape)
    if processor_type == pyasdm.enumerations.ProcessorType.CORRELATOR:
        vis_subset = (
            cross_values[:, :, baseband_spw_idxs[0], baseband_spw_idxs[1], :, :, 0]
            + 1j
            * cross_values[:, :, baseband_spw_idxs[0], baseband_spw_idxs[1], :, :, 1]
        ) / scale_factor
    else:
        # radiometer / spectrometer
        vis_subset = cross_values / scale_factor

    return vis_subset


def _load_vis_subset_auto_data(
    auto_data_arr: np.ndarray,
    guessed_shape: tuple[int, ...],
    baseband_spw_idxs: tuple[int, int],
) -> np.ndarray:

    polarization_len = guessed_shape[-2]
    if polarization_len == 3:
        # autoData: "The choice of a real- vs. complex-valued datum is dependent upon the
        # polarization product...parallel-hand polarizations are real-valued, while cross-hand
        # polarizations are complex-valued".
        auto_shape = guessed_shape[:1] + guessed_shape[2:-2] + (4,)
        auto_len = np.prod(auto_shape)
        auto_floats = (auto_data_arr[:auto_len]).reshape(auto_shape)
        vis_cross_hands = (
            auto_floats[:, :, baseband_spw_idxs[0], baseband_spw_idxs[1], :, [1]]
            + 1j * auto_floats[:, :, baseband_spw_idxs[0], baseband_spw_idxs[1], :, [2]]
        )
        vis_auto = np.concatenate(
            [
                auto_floats[:, :, baseband_spw_idxs[0], baseband_spw_idxs[1], :, [0]],
                vis_cross_hands,
                vis_cross_hands,
                auto_floats[:, :, baseband_spw_idxs[0], baseband_spw_idxs[1], :, [3]],
            ],
            axis=3,
        )
    else:
        auto_shape = guessed_shape[:1] + guessed_shape[2:-1]
        auto_len = np.prod(auto_shape)
        auto_floats = (auto_data_arr[:auto_len]).reshape(auto_shape)
        vis_auto = auto_floats[:, :, baseband_spw_idxs[0], baseband_spw_idxs[1], :, :]

    return vis_auto


def define_visibility_shape(
    bdf_descr: dict, baseband_spw_idxs: tuple[int, int]
) -> tuple:
    # shape of the full crossData/autoData binary component
    baseband_len = len(bdf_descr["basebands"])
    antenna_len = bdf_descr["num_antenna"]
    baseline_len = int(antenna_len * (antenna_len - 1) / 2)
    baseband_descr = bdf_descr["basebands"][baseband_spw_idxs[0]]
    spw_len = len(baseband_descr["spectralWindows"])
    spw_descr = baseband_descr["spectralWindows"][baseband_spw_idxs[1]]
    frequency_len = spw_descr["numSpectralPoint"]
    polarization_len = len(spw_descr["crossPolProducts"]) or len(
        spw_descr["sdPolProducts"]
    )

    # if dimensionality==0, we have TIM dimension / packed format
    time_len = bdf_descr["num_time"] if bdf_descr["dimensionality"] == 0 else 1
    shape = (
        time_len,
        baseline_len,
        antenna_len,
        baseband_len,
        spw_len,
        frequency_len,
        polarization_len,
        2,
    )

    return shape


def load_flags_all_subsets(
    bdf_reader: pyasdm.bdf.BDFReader,
    guessed_shape: dict[str, tuple[int, ...]],
    baseband_spw_idxs: tuple[int, int],
) -> np.ndarray:

    flag_per_subset = []
    while bdf_reader.hasSubset():
        try:
            subset = bdf_reader.getSubset(loadOnlyComponents={"flags"})
        except ValueError as exc:
            xradio_logger().warning(
                f"Error in getSubset for {bdf_reader.getPath()=} when trying to load "
                f"flags. Will use all-False. {exc=}"
            )
            return None

        flag_subset = _load_flags_subset(subset, guessed_shape, baseband_spw_idxs)

        flag_per_subset.append(flag_subset)

    bdf_flag = np.concatenate(flag_per_subset)

    return bdf_flag


def define_flag_shape(
    bdf_descr: dict, baseband_spw_idxs: tuple[int, int]
) -> dict[str, tuple[int, ...]]:

    baseband_len = len(bdf_descr["basebands"])
    antenna_len = bdf_descr["num_antenna"]
    baseline_len = int(antenna_len * (antenna_len - 1) / 2)
    baseband_descr = bdf_descr["basebands"][baseband_spw_idxs[0]]
    spw_len = len(baseband_descr["spectralWindows"])
    spw_descr = baseband_descr["spectralWindows"][baseband_spw_idxs[1]]
    cross_pol_len = len(spw_descr["crossPolProducts"])
    auto_pol_len = len(spw_descr["sdPolProducts"])

    # if dimensionality==0, we have TIM dimension / packed format
    time_len = bdf_descr["num_time"] if bdf_descr["dimensionality"] == 0 else 1

    # shapes of the blocks of flags corresponding to the crossData
    # and autoData binary components
    if bdf_descr["correlation_mode"] == pyasdm.enumerations.CorrelationMode.AUTO_ONLY:
        shape_cross = ()
    else:
        shape_cross = (
            time_len,
            baseline_len,
            baseband_len,
            spw_len,
            cross_pol_len,
        )
    shape_auto = (time_len, antenna_len, baseband_len, spw_len, auto_pol_len)

    return {
        "cross": shape_cross,
        "auto": shape_auto,
    }


def _try_alternatives_guessed_shape(
    guessed_shape: dict[str, tuple[int, ...]],
    flags_actual_size: int,
    baseband_spw_idxs: tuple[int, int],
) -> dict[str, tuple[int, ...]]:

    guessed_size = np.prod(add_cross_and_auto_flag_shapes(guessed_shape))
    if guessed_size > flags_actual_size:
        # try single value for all basebands
        new_shape = {}
        new_shape["cross"] = (
            guessed_shape["cross"][0:2]
            + (
                1,
                1,
            )
            + guessed_shape["cross"][-1:]
        )
        new_shape["auto"] = (
            guessed_shape["auto"][0:2]
            + (
                1,
                1,
            )
            + guessed_shape["auto"][-1:]
        )
        new_baseband_spw_idxs = (0, 0)
    elif guessed_size < flags_actual_size:
        raise RuntimeError(
            f"Unexpected large flags array in a subset. {guessed_size=} {guessed_shape=}, {flags_actual_size=}"
        )
    else:
        new_shape = guessed_shape
        new_baseband_spw_idxs = baseband_spw_idxs

    return new_shape, new_baseband_spw_idxs


def _load_flags_subset(
    subset: dict,
    guessed_shape: dict[str, tuple[int, ...]],
    baseband_spw_idxs: tuple[int, int],
) -> np.ndarray:
    """
    Loads the flags array from one subset in a BDF. The subset includes all the SPWs in the BDF.
    The flag array is reshaped. Then the SPWs we are not interested in are selected out.

    The returned array does not have the frequency dim, as it is not effectively used in the BDFs.
    That will need to be added by the calling code.
    """

    if "flags" in subset and subset["flags"]["present"]:
        shape, baseband_spw_idxs = _try_alternatives_guessed_shape(
            guessed_shape, subset["flags"]["arr"].size, baseband_spw_idxs
        )

        shape = add_cross_and_auto_flag_shapes(guessed_shape)

        # could also check the last dim of guessed_shape["auto"]
        # and guessed_shape["cross"]
        if guessed_shape["auto"][-1] != 3:
            flag_array = subset["flags"]["arr"].reshape(shape)
            flag_subset = flag_array[..., baseband_spw_idxs[0], baseband_spw_idxs[1], :]
        else:
            flag_array = subset["flags"]["arr"]
            if guessed_shape["cross"]:
                # first (bl) block => use directly as is
                # second (ant) block => 3pol, so expand XY to YX
                cross_len = np.prod(guessed_shape["cross"])
                cross_flags = flag_array[:cross_len].reshape(guessed_shape["cross"])
                cross_subset = cross_flags[
                    ..., baseband_spw_idxs[0], baseband_spw_idxs[1], :
                ]
            else:
                cross_len = 0

            auto_flags = flag_array[cross_len:].reshape(guessed_shape["auto"])

            # expand XX XY YY => XX XY YX YY (where XY=YX)
            auto_subset = np.concatenate(
                [
                    auto_flags[..., [0]],
                    auto_flags[..., [1]],
                    auto_flags[..., [1]],
                    auto_flags[..., [2]],
                ],
                axis=4,
            )
            auto_subset = auto_subset[
                ..., baseband_spw_idxs[0], baseband_spw_idxs[1], :
            ]

            if guessed_shape["cross"]:
                flag_subset = np.concatenate([cross_subset, auto_subset], axis=1)
            else:
                flag_subset = auto_subset

    else:
        shape = add_cross_and_auto_flag_shapes(guessed_shape)
        flag_subset = np.full(
            full_shape_to_output_filled_flags_shape(shape), False, dtype="bool"
        )

    return flag_subset
