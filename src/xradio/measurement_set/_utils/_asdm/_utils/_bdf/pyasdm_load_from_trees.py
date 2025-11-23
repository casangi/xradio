import numpy as np

import pyasdm


def add_cross_and_auto_flag_shapes(
    guessed_shape: dict[str, tuple[int, ...]],
) -> tuple[int, ...]:
    guessed_shape_cross = guessed_shape["cross"]
    guessed_shape_auto = guessed_shape["auto"]
    if guessed_shape_cross:
        # second dim is the "BAL ANT"
        shape = (
            guessed_shape_cross[0],
            guessed_shape_cross[1] + guessed_shape_auto[1],
            *guessed_shape_cross[2:],
        )
    else:
        # The axes of flags would be for example "TIM ANT"
        # or something with ANT but not BAL
        shape = guessed_shape_auto

    return shape


def full_shape_to_output_filled_flags_shape(shape: tuple[int, ...]) -> tuple[int, ...]:
    # equivalent to the squeezing that would happen when selecting
    # one baseband / one SPW with int indices.
    return shape[0:2] + shape[-1:]


def calculate_overall_spw_idx(
    basebands_descr: list[dict], baseband_idx: int, spw_idx: int
) -> int:
    overall_spw_idx = sum(
        [
            len(basebands_descr[bb_idx]["spectralWindows"])
            for bb_idx in range(0, baseband_idx)
        ]
    )
    +spw_idx

    return overall_spw_idx


def load_visibilities_all_subsets_from_trees(
    bdf_reader: pyasdm.bdf.BDFReader,
    guessed_shape: tuple[int, ...],
    baseband_spw_idxs: tuple[int, int],
    bdf_descr: dict,
) -> np.ndarray:

    vis_per_subset = []
    while bdf_reader.hasSubset():
        try:
            subset = bdf_reader.getSubset(loadOnlyComponents={"autoData", "crossData"})
        except ValueError as exc:
            logger.warning(f"Error in getSubset for {bdf_path=} {exc=}")
            return None

        vis_subset = load_vis_subset_from_tree(
            subset,
            guessed_shape,
            baseband_spw_idxs,
            bdf_descr,
        )

        vis_per_subset.append(vis_subset)

    bdf_vis = np.concatenate(vis_per_subset)
    return bdf_vis


def load_vis_subset_from_tree(
    subset: dict,
    guessed_shape: tuple,
    baseband_spw_idxs: tuple[int, int],
    bdf_descr: dict,
) -> np.ndarray:

    spw_chan_lens = [
        bdf_descr["basebands"][bb_idx]["spectralWindows"][spw_idx]["numSpectralPoint"]
        for bb_idx in range(0, len(bdf_descr["basebands"]))
        for spw_idx in range(0, len(bdf_descr["basebands"][bb_idx]["spectralWindows"]))
    ]
    baseband_idx, spw_idx = baseband_spw_idxs
    overall_spw_idx = calculate_overall_spw_idx(
        bdf_descr["basebands"], baseband_idx, spw_idx
    )

    if "autoData" in subset and subset["autoData"]["present"]:
        vis_subset_auto = load_vis_subset_auto_data_from_tree(
            subset["autoData"]["arr"],
            guessed_shape,
            spw_chan_lens,
            overall_spw_idx,
        )

    else:
        # Never allowed for ALMA (BDF doc) and seems so in real life
        RuntimeError("autoData not present!")

    vis_subset_cross = None
    if "crossData" in subset and subset["crossData"]["present"]:
        baseband_description = bdf_descr["basebands"][baseband_spw_idxs[0]]
        spw_descr = baseband_description["spectralWindows"][baseband_spw_idxs[1]]
        scale_factor = spw_descr["scaleFactor"] or 1
        processor_type = bdf_descr["processor_type"]

        vis_subset_cross = load_vis_subset_cross_data_from_tree(
            subset["crossData"]["arr"],
            guessed_shape,
            spw_chan_lens,
            overall_spw_idx,
            scale_factor,
            processor_type,
        )

    if vis_subset_cross is None:
        vis_subset = vis_subset_auto
    else:
        vis_subset = np.concatenate([vis_subset_cross, vis_subset_auto], axis=1)

    return vis_subset


def load_vis_subset_cross_data_from_tree(
    cross_data_arr: np.ndarray,
    guessed_shape: tuple[int, ...],
    spw_chan_lens: list[int],
    overall_spw_idx: int,
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
    for time_idx in np.arange(0, time_len):
        vis_strides = []
        for baseline_idx in np.arange(baseline_len):
            if processor_type == pyasdm.enumerations.ProcessorType.CORRELATOR:
                offset += cross_offset_addition_before
                spw_vis = cross_data_arr[
                    offset : offset + spw_channel_len * polarization_len * 2
                ]

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

    vis_subset = np.stack(vis_strides)
    vis_subset = vis_subset.reshape((1, *vis_subset.shape))

    return vis_subset


def load_vis_subset_auto_data_from_tree(
    auto_data_arr: np.ndarray,
    guessed_shape: tuple[int, ...],
    spw_chan_lens: list[int],
    overall_spw_idx: int,
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
    for time_idx in np.arange(0, guessed_shape[0]):
        vis_auto_strides = []
        for antenna_idx in np.arange(antenna_len):
            offset += auto_offset_addition_before
            auto_floats = auto_data_arr
            if polarization_len == 3:
                # autoData: "The choice of a real- vs. complex-valued datum is dependent upon the
                # polarization product...parallel-hand polarizations are real-valued, while cross-hand
                # polarizations are complex-valued".
                spw_floats = auto_floats[
                    offset : offset + spw_channel_len * sd_polarization_len
                ]
                spw_floats = spw_floats.reshape((spw_channel_len, sd_polarization_len))
                spw_vis = np.concatenate(
                    [
                        spw_floats[:, [0]],
                        spw_floats[:, [1]] + 1j * spw_floats[:, [2]],
                        spw_floats[:, [3]],
                    ],
                    axis=1,
                )
                vis_auto_strides.append(spw_vis)
            else:
                spw_floats = auto_floats[
                    offset : offset + spw_channel_len * sd_polarization_len
                ]
                spw_floats = spw_floats.reshape((spw_channel_len, sd_polarization_len))
                vis_auto_strides.append(spw_floats)

            offset += auto_offset_addition_after

        vis_subset_integrations.append(np.stack(vis_auto_strides))

    if len(vis_subset_integrations) == 1:
        vis_auto = vis_subset_integrations[0][np.newaxis, :]
    else:
        vis_auto = np.stack(vis_subset_integrations)

    return vis_auto


def load_flags_all_subsets_from_trees(
    bdf_reader: pyasdm.bdf.BDFReader,
    guessed_shape: tuple[int, ...],
    bdf_descr: dict,
    baseband_spw_idxs: tuple[int, int],
) -> np.ndarray:

    # Load taking pieces from the data trees of the binary components. Needed when the number
    # of SPWs per baseband, or number of channels per SPW are not uniform.
    flag_per_subset = []
    while bdf_reader.hasSubset():
        try:
            subset = bdf_reader.getSubset(loadOnlyComponents={"flags"})
        except ValueError as exc:
            logger.warning(
                f"Error in getSubset for {bdf_reader.getPath()=} when trying to load "
                f"flags. Will use all-False. {exc=}"
            )
            return None

        flag_subset = load_flags_subset_from_tree(
            subset, guessed_shape, bdf_descr, baseband_spw_idxs
        )
        flag_per_subset.append(flag_subset)

    bdf_flag = np.concatenate(flag_per_subset)

    return bdf_flag


def make_per_spw_cross_pol_lens(basebands_descr: list[list]) -> int:
    per_spw_cross_pol_lens = [
        len(basebands_descr[bb_idx]["spectralWindows"][spw_idx]["crossPolProducts"])
        for bb_idx in range(0, len(basebands_descr))
        for spw_idx in range(0, len(basebands_descr[bb_idx]["spectralWindows"]))
    ]
    return per_spw_cross_pol_lens


def make_per_spw_auto_pol_lens(basebands_descr: list[list]) -> int:
    per_spw_auto_pol_len = [
        len(basebands_descr[bb_idx]["spectralWindows"][spw_idx]["sdPolProducts"])
        for bb_idx in range(0, len(basebands_descr))
        for spw_idx in range(0, len(basebands_descr[bb_idx]["spectralWindows"]))
    ]
    return per_spw_auto_pol_len


def make_per_baseband_cross_pol_lens(basebands_descr: list[list]) -> int:
    per_baseband_cross_pol_lens = [
        len(basebands_descr[bb_idx]["spectralWindows"][0]["crossPolProducts"])
        for bb_idx in range(0, len(basebands_descr))
    ]
    return per_baseband_cross_pol_lens


def make_per_baseband_auto_pol_lens(basebands_descr: list[list]) -> int:
    per_baseband_auto_pol_lens = [
        len(basebands_descr[bb_idx]["spectralWindows"][0]["sdPolProducts"])
        for bb_idx in range(0, len(basebands_descr))
    ]
    return per_baseband_auto_pol_lens


def make_flag_tree_offsets_per_spw(
    per_spw_cross_pol_lens: list[int],
    per_spw_auto_pol_lens: list[int],
    overall_spw_idx: int,
) -> dict[str, dict]:

    offset = {"cross": {}, "auto": {}}
    offset["cross"]["before"] = np.sum(
        per_spw_cross_pol_lens[0:overall_spw_idx], dtype=int
    )
    offset["cross"]["after"] = np.sum(
        per_spw_cross_pol_lens[overall_spw_idx:], dtype=int
    )

    offset["auto"]["before"] = np.sum(
        per_spw_auto_pol_lens[0:overall_spw_idx], dtype=int
    )
    offset["auto"]["after"] = np.sum(per_spw_auto_pol_lens[overall_spw_idx:], dtype=int)

    return offset


def make_flag_tree_offsets_per_baseband(
    per_baseband_cross_pol_lens, per_baseband_auto_pol_lens, baseband_idx
) -> dict[str, dict]:

    offset = {"cross": {}, "auto": {}}
    offset["cross"]["before"] = np.sum(
        per_baseband_cross_pol_lens[0:baseband_idx], dtype=int
    )
    offset["cross"]["after"] = np.sum(
        per_baseband_cross_pol_lens[baseband_idx:], dtype=int
    )

    offset["auto"]["before"] = np.sum(
        per_baseband_auto_pol_lens[0:baseband_idx], dtype=int
    )
    offset["auto"]["after"] = np.sum(
        per_baseband_auto_pol_lens[baseband_idx:], dtype=int
    )

    return offset


def calculate_offset_additions_cross_sd(
    bdf_descr: list[dict],
    baseband_idx: int,
    overall_spw_idx: int,
    flag_array_len: int,
) -> dict[str, tuple[int, int]]:

    auto_only = (
        bdf_descr["correlation_mode"] == pyasdm.enumerations.CorrelationMode.AUTO_ONLY
    )

    basebands_descr = bdf_descr["basebands"]
    antenna_len = bdf_descr["num_antenna"]
    baseline_len = int(antenna_len * (antenna_len - 1) / 2)

    per_spw_cross_pol_lens = make_per_spw_cross_pol_lens(basebands_descr)
    if auto_only:
        guessed_cross_len = 0
    else:
        guessed_cross_len = baseline_len * np.sum(per_spw_cross_pol_lens, dtype=int)

    per_spw_auto_pol_lens = make_per_spw_auto_pol_lens(basebands_descr)
    guessed_auto_len = antenna_len * np.sum(per_spw_auto_pol_lens, dtype=int)
    time_len = bdf_descr["num_time"] if bdf_descr["dimensionality"] == 0 else 1

    offset = {"cross": {}, "auto": {}}
    if (guessed_cross_len + guessed_auto_len) * time_len == flag_array_len:
        offset = make_flag_tree_offsets_per_spw(
            per_spw_cross_pol_lens, per_spw_auto_pol_lens, overall_spw_idx
        )
    else:
        # Try per-BB flags
        per_baseband_cross_pol_lens = make_per_baseband_cross_pol_lens(basebands_descr)
        if auto_only:
            second_guessed_cross_len = 0
        else:
            second_guessed_cross_len = baseline_len * np.sum(
                per_baseband_cross_pol_lens, dtype=int
            )

        per_baseband_auto_pol_lens = make_per_baseband_auto_pol_lens(basebands_descr)
        second_guessed_auto_len = antenna_len * np.sum(
            per_baseband_auto_pol_lens, dtype=int
        )

        if (
            second_guessed_cross_len + second_guessed_auto_len
        ) * time_len == flag_array_len:
            offset = make_flag_tree_offsets_per_baseband(
                per_baseband_cross_pol_lens, per_baseband_auto_pol_lens, baseband_idx
            )
        else:
            raise RuntimeError(
                f"Unexpected flags array in a subset. {guessed_cross_len=}, {guessed_auto_len=}, "
                f"{second_guessed_cross_len=}, {second_guessed_auto_len=}, {flag_array_len=} "
                "f{offset=}, {bdf_descr=}"
            )

    return offset


def load_flags_subset_from_tree(
    subset: dict,
    guessed_shape: tuple,
    bdf_descr: dict,
    baseband_spw_idxs: tuple[int, int],
) -> np.ndarray:
    """
    Loads the flags array from one subset in a BDF.
    """

    baseband_idx, spw_idx = baseband_spw_idxs
    if "flags" in subset and subset["flags"]["present"]:
        overall_spw_idx = calculate_overall_spw_idx(
            bdf_descr["basebands"], baseband_idx, spw_idx
        )

        flag_array = subset["flags"]["arr"]
        offset_additions = calculate_offset_additions_cross_sd(
            bdf_descr,
            baseband_idx,
            overall_spw_idx,
            len(flag_array),
        )

        flag_subset = load_flags_subset_cross_and_auto_blocks_from_tree(
            flag_array, bdf_descr, offset_additions, guessed_shape, baseband_spw_idxs
        )
    else:
        shape = add_cross_and_auto_flag_shapes(guessed_shape)
        flag_subset = np.full(
            full_shape_to_output_filled_flags_shape(shape), False, dtype="bool"
        )

    flag_subset = expand_frequency_in_flags_subset(
        flag_subset, bdf_descr, baseband_idx, spw_idx
    )

    return flag_subset


def expand_frequency_in_flags_subset(
    flag_subset: np.ndarray, bdf_descr: dict, baseband_idx: int, spw_idx: int
) -> np.ndarray:
    frequency_len = bdf_descr["basebands"][baseband_idx]["spectralWindows"][spw_idx][
        "numSpectralPoint"
    ]
    flag_subset = np.tile(flag_subset[:, :, np.newaxis, :], (1, 1, frequency_len, 1))

    return flag_subset


def load_flags_subset_cross_and_auto_blocks_from_tree(
    flag_array: np.ndarray,
    bdf_descr: dict,
    offset_additions: dict,
    guessed_shape: tuple[int, ...],
    baseband_spw_idxs: tuple[int, int],
) -> np.ndarray[bool]:

    antenna_len = bdf_descr["num_antenna"]
    baseline_len = int(antenna_len * (antenna_len - 1) / 2)
    baseband_description = bdf_descr["basebands"][baseband_spw_idxs[0]]
    spw_descr = baseband_description["spectralWindows"][baseband_spw_idxs[1]]
    polarization_cross_len = len(spw_descr["crossPolProducts"])
    polarization_auto_len = len(spw_descr["sdPolProducts"])

    offset = 0
    flag_subset_integrations = []
    for time_idx in np.arange(0, guessed_shape["auto"][0]):
        flag_strides = []
        if (
            bdf_descr["correlation_mode"]
            != pyasdm.enumerations.CorrelationMode.AUTO_ONLY
        ):
            cross_offset_addition_before = offset_additions["cross"]["before"]
            cross_offset_addition_after = offset_additions["cross"]["after"]
            for baseline_idx in np.arange(baseline_len):
                offset += cross_offset_addition_before
                stride = flag_array[offset : offset + polarization_cross_len].astype(
                    "bool"
                )  # forgetting the int details (BinaryDataFlags enum)
                flag_strides.append(stride)
                offset += cross_offset_addition_after

        auto_offset_addition_before = offset_additions["auto"]["before"]
        auto_offset_addition_after = offset_additions["auto"]["after"]
        for antenna_idx in np.arange(antenna_len):
            offset += auto_offset_addition_before
            if polarization_auto_len != 3:
                stride = flag_array[offset : offset + polarization_auto_len].astype(
                    "bool"
                )  # forgetting the int details
            else:
                stride = np.array(
                    [
                        flag_array[offset],
                        flag_array[offset + 1],
                        flag_array[offset + 1],
                        flag_array[offset + 2],
                    ],
                    dtype="bool",
                )

            flag_strides.append(stride)
            offset += auto_offset_addition_after

        flag_subset_integrations.append(np.stack(flag_strides))

    if len(flag_subset_integrations) == 1:
        flag_subset = flag_subset_integrations[0][np.newaxis, :]
    else:
        flag_subset = np.stack(flag_subset_integrations)

    return flag_subset
