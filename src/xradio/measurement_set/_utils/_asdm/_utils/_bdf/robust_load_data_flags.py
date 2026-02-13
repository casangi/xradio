import traceback

import numpy as np

import pyasdm

import toolviper.utils.logger as logger

from .basebands import find_spw_in_basebands_list
from .pyasdm_load_from_trees import (
    add_cross_and_auto_flag_shapes,
    full_shape_to_output_filled_flags_shape,
    load_flags_all_subsets_from_trees,
    load_visibilities_all_subsets_from_trees,
)


def ensure_presence_binary_components(
    data_array_names: list[str], binary_types: list[str], bdf_path: str
):

    for array_name in data_array_names:
        if array_name not in binary_types:
            raise RuntimeError(
                f"When trying to load visibility data from BDF: {bdf_path}, it does not "
                f"have array {array_name}"
            )


def exclude_unsupported_axis_names(
    dims: list[str], exclude_also_for_flags: bool = False
):

    # This effectively assumes we'll always get "POL" from the last 3 possible axes,
    # from BDF doc: "The final three axes, STO, POL and HOL, also appear at the same
    # level in the axis hierarchy; however, only one of these axes will normally
    # appear for a given binary component type.
    unsupported = ["STO", "HOL"]

    if exclude_also_for_flags:
        # TODO: Consider also "BIN"
        unsupported.extend(["APC", "SPP"])

    bad_found = []
    for bad_dim in unsupported:
        if bad_dim in dims:
            bad_found.append(bad_dim)

    if bad_found:
        raise RuntimeError(f"Unsupported dimension(s) {bad_found=} in {dims=}")


def array_slice_to_msv4_indices(array_slice: dict) -> tuple[range, range, range, range]:

    time_slice = slice(None)
    baseline_slice = slice(None)
    frequency_slice = slice(None)
    polarization_slice = slice(None)

    if "time" in array_slice:
        time_slice = array_slice["time"]
    if "baseline" in array_slice:
        baseline_slice = array_slice["baseline"]
    if "frequency" in array_slice:
        frequency_slice = array_slice["frequency"]
    if "polarization" in array_slice:
        polarization_slice = array_slice["polarization"]

    return (time_slice, baseline_slice, frequency_slice, polarization_slice)


def find_different_basebands_spws(basebands: list[dict]) -> tuple[int, int]:

    all_same = True
    spws_per_baseband = -1
    chans_per_spw = -1
    for bband in basebands:
        if not all_same:
            break
        num_spws = len(bband["spectralWindows"])
        if spws_per_baseband > 0:
            if num_spws != spws_per_baseband:
                all_same = False
                break
        else:
            spws_per_baseband = num_spws

        for spw in bband["spectralWindows"]:
            num_chans = spw["numSpectralPoint"]
            if chans_per_spw > 0:
                if num_chans != chans_per_spw:
                    all_same = False
                    break
            else:
                chans_per_spw = num_chans

    return not all_same


def find_different_basebands_pols(basebands: list[dict]) -> tuple[int, int]:

    all_same = True
    spws_per_baseband = -1
    cross_pols_per_spw = -1
    sd_pols_per_spw = -1
    for bband in basebands:
        if not all_same:
            break
        num_spws = len(bband["spectralWindows"])
        if spws_per_baseband > 0:
            if num_spws != spws_per_baseband:
                all_same = False
                break
        else:
            spws_per_baseband = num_spws

        for spw in bband["spectralWindows"]:
            num_pols_cross = len(spw["crossPolProducts"])
            num_pols_sd = len(spw["sdPolProducts"])
            if cross_pols_per_spw > 0:
                if num_pols_cross != cross_pols_per_spw:
                    all_same = False
                    break
            else:
                cross_pols_per_spw = num_pols_cross
            if sd_pols_per_spw > 0:
                if num_pols_sd != sd_pols_per_spw:
                    all_same = False
                    break
            else:
                sd_pols_per_spw = num_pols_sd

    return not all_same


def load_visibilities_from_partition_bdfs(
    bdf_paths: list[str], spw_id: int, array_slice: dict | None = None
) -> np.ndarray:

    cumulative_vis = []
    for bdf_path in bdf_paths:
        visibility_blob = load_visibilities_from_bdf(bdf_path, spw_id, array_slice)

        if array_slice:
            indices = array_slice_to_msv4_indices(array_slice)
            visibility_blob = visibility_blob[*indices]

        cumulative_vis.append(visibility_blob)

    import time

    start = time.perf_counter()
    visibility = np.concatenate(cumulative_vis)
    end = time.perf_counter()
    logger.info(
        f"Loaded visibility, with {visibility.shape=} from {len(bdf_paths)=} blobs, time: {end-start:.6}"
    )

    return visibility


def make_bdf_description(bdf_header: pyasdm.bdf.BDFHeader) -> dict:
    bdf_descr = {
        # packed/TIM: dimensionality == 0
        "dimensionality": bdf_header.getDimensionality(),
        "num_time": bdf_header.getNumTime(),
        "processor_type": bdf_header.getProcessorType(),
        "binary_types": bdf_header.getBinaryTypes(),
        "correlation_mode": bdf_header.getCorrelationMode(),
        "apc": bdf_header.getAPClist(),
        "num_antenna": bdf_header.getNumAntenna(),
        "basebands": bdf_header.getBasebandsList(),
    }

    return bdf_descr


def check_correlation_mode(correlation_mode: pyasdm.enumerations.CorrelationMode):
    if correlation_mode == pyasdm.enumerations.CorrelationMode.CROSS_ONLY:
        raise RuntimeError(f" Unexpected {correlation_mode=}")


def load_visibilities_from_bdf(
    bdf_path: str,
    spw_id: int,
    array_slice: dict,
    never_reshape_from_all_spws: bool = True,
) -> np.ndarray:

    bdf_reader = pyasdm.bdf.BDFReader()
    bdf_reader.open(bdf_path)
    bdf_header = bdf_reader.getHeader()

    bdf_descr = make_bdf_description(bdf_header)

    check_correlation_mode(bdf_descr["correlation_mode"])
    check_basebands(bdf_descr["basebands"])
    ensure_presence_binary_components(
        ["crossData", "autoData"], bdf_descr["binary_types"], bdf_path
    )

    baseband_spw_idxs = find_spw_in_basebands_list(
        spw_id, bdf_descr["basebands"], bdf_path
    )
    different_channels_per_spw = find_different_basebands_spws(bdf_descr["basebands"])
    guessed_shape = define_visibility_shape(bdf_descr, baseband_spw_idxs)
    try:
        if never_reshape_from_all_spws or different_channels_per_spw:
            # TODO: this is assumed to be slower than the simpler version (simply reshape-based)
            # TO: integrate/replace with pyasdm.
            bdf_vis = load_visibilities_all_subsets_from_trees(
                bdf_reader, guessed_shape, baseband_spw_idxs, bdf_descr
            )
        else:
            bdf_vis = load_visibilities_all_subsets(
                bdf_reader, guessed_shape, baseband_spw_idxs, bdf_descr
            )
    except (RuntimeError, ValueError) as exc:
        trace = traceback.format_exc()
        raise RuntimeError(
            f"Error while loading data/visibilities from a BDF ({bdf_path=}). Details: {exc}."
            + trace
            + "BDF header:\n"
            + str(bdf_header)
        )

    return bdf_vis


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
            logger.warning(
                f"Error in getSubset for {bdf_reader.getPath()=}  when trying to load "
                f"visibilities. {exc=}" + trace
            )
            return None

        vis_subset = load_vis_subset(
            subset,
            guessed_shape,
            baseband_spw_idxs,
            scale_factor,
            processor_type,
        )

        vis_per_subset.append(vis_subset)

    bdf_vis = np.concatenate(vis_per_subset)
    return bdf_vis


def load_vis_subset(
    subset: dict,
    guessed_shape: tuple,
    baseband_spw_idxs: tuple[int, int],
    scale_factor: float,
    processor_type: pyasdm.enumerations.ProcessorType,
) -> np.ndarray:

    if "autoData" in subset and subset["autoData"]["present"]:
        vis_subset_auto = load_vis_subset_auto_data(
            subset["autoData"]["arr"], guessed_shape, baseband_spw_idxs
        )

    else:
        # Never allowed for ALMA (BDF doc) and seems so in real life
        RuntimeError("autoData not present!")

    vis_subset_cross = None
    if "crossData" in subset and subset["crossData"]["present"]:
        vis_subset_cross = load_vis_subset_cross_data(
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


def load_vis_subset_cross_data(
    cross_data_arr: np.ndarray,
    guessed_shape: tuple,
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


def load_vis_subset_auto_data(
    auto_data_arr: np.ndarray,
    guessed_shape: tuple,
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


def load_flags_from_partition_bdfs(
    bdf_paths: list[str], spw_id: int, array_slice: dict | None = None
) -> np.ndarray:

    cumulative_flag = []
    for bdf_path in bdf_paths:
        flag_blob = load_flags_from_bdf(bdf_path, spw_id, array_slice)

        if array_slice:
            indices = array_slice_to_msv4_indices(array_slice)
            flag_blob = flag_blob[*indices]

        cumulative_flag.append(flag_blob)

    return np.concatenate(cumulative_flag)


def check_flags_dims(flags_dims: list[str]):
    exclude_unsupported_axis_names(flags_dims, True)


def check_basebands(basebands: list[dict]):
    # TODO: working check for what's out there...
    # An example of 2 basebands: uid___A002_X9bb85e_Xcb (I think they are rare)
    if len(basebands) not in [1, 2, 3, 4]:
        raise RuntimeError(f" {len(basebands)=}, {basebands=}")


def load_flags_from_bdf(
    bdf_path: list[str],
    spw_id: int,
    array_slice: dict,
    never_reshape_from_all_spws: bool = True,
) -> np.ndarray:

    bdf_reader = pyasdm.bdf.BDFReader()
    bdf_reader.open(bdf_path)
    bdf_header = bdf_reader.getHeader()

    check_flags_dims(bdf_header.getAxesNames("flags"))

    bdf_descr = make_bdf_description(bdf_header)

    check_correlation_mode(bdf_descr["correlation_mode"])
    check_basebands(bdf_descr["basebands"])
    ensure_presence_binary_components(["flags"], bdf_descr["binary_types"], bdf_path)

    baseband_spw_idxs = find_spw_in_basebands_list(
        spw_id, bdf_descr["basebands"], bdf_path
    )
    different_pols_per_spw = find_different_basebands_pols(bdf_descr["basebands"])
    guessed_shape = define_flag_shape(bdf_descr, baseband_spw_idxs)
    try:
        if never_reshape_from_all_spws or different_pols_per_spw:
            # TODO: this is assumed to be slower than the simpler version (simply reshape-based)
            # TO: integrate/replace with in pyasdm.
            bdf_flag = load_flags_all_subsets_from_trees(
                bdf_reader,
                guessed_shape,
                bdf_descr,
                baseband_spw_idxs,
            )
        else:
            bdf_flag = load_flags_all_subsets(
                bdf_reader, guessed_shape, baseband_spw_idxs
            )
    except (RuntimeError, ValueError) as exc:
        trace = traceback.format_exc()
        raise RuntimeError(
            f"Error while loading flags from a BDF ({bdf_path=}). Details: {exc}."
            + trace
            + "BDF header:"
            + str(bdf_header)
        )

    return bdf_flag


def load_flags_all_subsets(
    bdf_reader: pyasdm.bdf.BDFReader,
    guessed_shape: tuple[int, ...],
    baseband_spw_idxs: tuple[int, int],
) -> np.ndarray:

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

        flag_subset = load_flags_subset(subset, guessed_shape, baseband_spw_idxs)

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


def try_alternatives_guessed_shape(
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


def load_flags_subset(
    subset: dict,
    guessed_shape: tuple,
    baseband_spw_idxs: tuple[int, int],
) -> np.ndarray:
    """
    Loads the flags array from one subset in a BDF.
    """

    if "flags" in subset and subset["flags"]["present"]:
        shape, baseband_spw_idxs = try_alternatives_guessed_shape(
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
