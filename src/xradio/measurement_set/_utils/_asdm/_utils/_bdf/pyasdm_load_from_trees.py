import numpy as np

import pyasdm


def calculate_overall_spw_idx(
    basebands_descr: list[dict], baseband_idx: int, spw_idx: int
) -> int:
    overall_spw_idx = sum(
        [
            len(basebands_descr[bb_idx]["spectralWindows"])
            for bb_idx in range(0, baseband_idx + 1)
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
            subset = bdf_reader.getSubset()
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

    baseband_description = bdf_descr["basebands"][baseband_spw_idxs[0]]
    spw_descr = baseband_description["spectralWindows"][baseband_spw_idxs[1]]
    scale_factor = spw_descr["scaleFactor"] or 1
    processor_type = bdf_descr["processor_type"]
    spw_chan_lens = [
        bdf_descr["basebands"][bb_idx]["spectralWindows"][spw_idx]["numSpectralPoint"]
        for bb_idx in range(0, len(bdf_descr["basebands"]))
        for spw_idx in range(0, len(bdf_descr["basebands"][bb_idx]["spectralWindows"]))
    ]
    polarization_len = len(spw_descr["crossPolProducts"]) or len(
        spw_descr["sdPolProducts"]
    )
    antenna_len = bdf_descr["num_antenna"]
    baseband_idx, spw_idx = baseband_spw_idxs
    overall_spw_idx = calculate_overall_spw_idx(
        bdf_descr["basebands"], baseband_idx, spw_idx
    )
    spw_channel_len = spw_chan_lens[overall_spw_idx]
    vis_strides = []

    if "crossData" in subset and subset["crossData"]["present"]:
        baseline_len = int(antenna_len * (antenna_len - 1) / 2)
        cross_floats = subset["crossData"]["arr"]
        offset = 0
        for time_idx in np.arange(0, guessed_shape[0]):
            for baseline_idx in np.arange(baseline_len):
                if processor_type == pyasdm.enumerations.ProcessorType.CORRELATOR:
                    offset += (
                        np.sum(spw_chan_lens[0:overall_spw_idx]) * polarization_len * 2
                    )
                    spw_vis = (
                        cross_floats[
                            offset : offset + spw_channel_len * polarization_len * 2
                        ]
                        / scale_factor
                    )
                    spw_vis = spw_vis.reshape((int(spw_vis.size / 2), 2))
                    spw_vis = spw_vis[:, 0] + 1j * spw_vis[:, 1]
                    vis_strides.append(
                        spw_vis.reshape((spw_channel_len, polarization_len))
                    )
                    offset += (
                        np.sum(spw_chan_lens[overall_spw_idx:]) * polarization_len * 2
                    )

                else:
                    # radiometer / spectrometer
                    offset += (
                        np.sum(spw_chan_lens[0:overall_spw_idx]) * polarization_len
                    )
                    spw_floats = (
                        cross_floats[
                            offset : offset + spw_channel_len * polarization_len
                        ]
                        / scale_factor
                    )
                    spw_floats = spw_floats.reshape((spw_channel_len, polarization_len))
                    vis_strides.append(spw_floats)
                    offset += np.sum(spw_chan_lens[overall_spw_idx:]) * polarization_len

    if "autoData" in subset and subset["autoData"]["present"]:
        offset = 0
        for time_idx in np.arange(0, guessed_shape[0]):
            for antenna_idx in np.arange(antenna_len):
                offset += np.sum(spw_chan_lens[0:overall_spw_idx]) * polarization_len
                auto_floats = subset["autoData"]["arr"]
                spw_floats = (
                    auto_floats[offset : offset + spw_channel_len * polarization_len]
                    / scale_factor
                )
                spw_floats = spw_floats.reshape((spw_channel_len, polarization_len))
                vis_strides.append(spw_floats)
                offset += np.sum(spw_chan_lens[overall_spw_idx:]) * polarization_len
    else:
        # Never allowed for ALMA (BDF doc) and seems so in real life
        RuntimeError("autoData not present!")

    vis_subset = np.concatenate(vis_strides)

    return vis_subset


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
            subset = bdf_reader.getSubset()
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


def load_flags_subset_from_tree(
    subset: dict,
    guessed_shape: tuple,
    bdf_descr: dict,
    baseband_spw_idxs: tuple[int, int],
) -> np.ndarray:
    """
    Loads the flags array from one subset in a BDF.
    """

    if "flags" in subset and subset["flags"]["present"]:
        if len(guessed_shape) > 5:
            raise RuntimeError(
                f"Unexpected. Found {guessed_shape=}, {len(guessed_shape)=}, with {flag_array=}"
            )

        antenna_len = bdf_descr["num_antenna"]
        baseline_len = int(antenna_len * (antenna_len - 1) / 2)
        spw_pol_lens = [
            # bdf_descr["basebands"][bb_idx]["spectralWindows"][spw_idx]["numSpectralPoint"]
            bdf_descr["basebands"][bb_idx]["spectralWindows"][spw_idx][
                "crossPolProducts"
            ]
            or bdf_descr["basebands"][bb_idx]["spectralWindows"][spw_idx][
                "sdPolProducts"
            ]
            for spw_idx in range(0, spw_len)
            for bb_idx in range(0, len(bdf_descr["basebands"]))
        ]
        polarization_len = len(spw_descr["crossPolProducts"]) or len(
            spw_descr["sdPolProducts"]
        )
        baseband_idx, spw_idx = baseband_spw_idxs
        overall_spw_idx = calculate_overall_spw_idx(
            bdf_descr["basebands"], baseband_idx, spw_idx
        )
        flag_strides = []
        flag_array = subset["flags"]["arr"]  # .reshape(shape)
        for time_idx in np.arange(0, guessed_shape[0]):
            if (
                bdf_descr["correlation_mode"]
                != pyasdm.enumerations.CorrelationMode.AUTO_ONLY
            ):
                for baseline_idx in np.arange(baseline_len):
                    offset += np.sum(spw_pol_lens[0:overall_spw_idx])
                    flag_strides.append(flag_array[offset : offset + polarization_len])
                    offset += np.sum(spw_pol_lens[overall_spw_idx:])

            for antenna_idx in np.arange(antenna_len):
                offset += np.sum(spw_pol_lens[0:overall_spw_idx])
                flag_strides.append(flag_array[offset : offset + polarization_len])

            offset += np.sum(spw_pol_lens[overall_spw_idx:])

        flag_subset = np.concatenate(flag_strides)
    else:
        shape = guessed_shape[0:2] + (guessed_shape[-1],)
        flag_subset = np.full(shape, False, dtype="bool")

    return flag_subset
