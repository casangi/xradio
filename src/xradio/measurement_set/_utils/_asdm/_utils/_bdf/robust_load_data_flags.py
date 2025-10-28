import numpy as np

import pyasdm

import toolviper.utils.logger as logger


def ensure_presence_data_arrays(
    data_array_names: list[str], binary_types: list[str], bdf_path
):

    for array_name in data_array_names:
        if array_name not in binary_types:
            raise RuntimeError(
                f"When trying to load visibility data from BDF: {bdf_path}, it does not "
                f"have array {array_name}"
            )


def exclude_unsupported_axis_names(dims: list[str]):

    # This effectively assumes we'll always get "POL" from the last 3 possible axes,
    # from BDF doc: "The final three axes, STO, POL and HOL, also appear at the same
    # level in the axis hierarchy; however, only one of these axes will normally
    # appear for a given binary component type.
    unsupported = ["STO", "HOL"]

    for bad_dim in unsupported:
        if bad_dim in dims:
            raise RuntimeError(f"Unsupported dimension {bad_dim=} in {dims=}")


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


def find_spw_in_basebands_list(
    bdf_path: str, spw_id: int, basebands: list[dict]
) -> tuple[int, int]:

    baseband_description = {}
    baseband_index = 0
    for bband in basebands:
        for spw in bband["spectralWindows"]:
            if spw_id == int(spw["sw"]) - 1:
                # Not sure if the lists are guaranteed to be sorted by id, so taking the id from name
                if "NOBB" == bband["name"]:
                    spw_index = 0
                else:
                    spw_index = int(bband["name"].split("_")[1]) - 1
                baseband_description = bband["spectralWindows"][0]
        baseband_index += 1

    if not baseband_description:
        # TODO: This is a highly dubious fallback for now...
        # Trying to figure out if there is some mapping from spw_ids
        err_msg = f"SPW {spw_id} not found in this BDF: {bdf_path}"
        # raise RuntimeError(err_msg)
        logger.warning(err_msg)
        spw_index = 1 - 1
        baseband_index = 0
        baseband_description = basebands[0]["spectralWindows"][0]

    return (baseband_index, spw_index)


def load_visibilities_from_partition_bdfs(
    bdf_paths: list[str], spw_id: int, array_slice: dict | None = None
) -> np.ndarray:

    cumulative_vis = []
    for bdf_path in bdf_paths:
        visibility_blob = load_visibilities(bdf_path, spw_id, array_slice)

        if array_slice:
            indices = array_slice_to_msv4_indices(array_slice)
            visibility_blob = visibility_blob[*indices]

        cumulative_vis.append(visibility_blob)

    import time

    start = time.perf_counter()
    visibility = np.concatenate(cumulative_vis)
    end = time.perf_counter()
    logger.info(
        f"Loaded visibiilty, with {visibility.shape=} from {len(bdf_paths)=} blobs, time: {end-start:.6}"
    )

    return visibility


def check_correlation_mode(correlation_mode: pyasdm.enumerations.CorrelationMode):
    if correlation_mode == pyasdm.enumerations.CorrelationMode.CROSS_ONLY:
        raise RuntimeError(f" Unexpected {correlation_mode=} {bdf_header=}")


def load_visibilities(bdf_path: str, spw_id: int, array_slice: dict) -> np.ndarray:

    bdf_reader = pyasdm.bdf.BDFReader()
    bdf_reader.open(bdf_path)
    bdf_header = bdf_reader.getHeader()

    check_correlation_mode(bdf_header.getCorrelationMode())

    bdf_descr = {
        # packed/TIM: dimensionality == 0
        "dimensionality": bdf_header.getDimensionality(),
        "num_times": bdf_header.getNumTime(),
        "processor_type": bdf_header.getProcessorType(),
        "binary_types": bdf_header.getBinaryTypes(),
        "apc": bdf_header.getAPClist(),
        "num_antenna": bdf_header.getNumAntenna(),
        "basebands": bdf_header.getBasebandsList(),
    }

    # TODO: working check for what's out there...
    if len(bdf_descr["basebands"]) not in [1, 3, 4]:
        raise RuntimeError(f"*** {len(basebands)=}, {basebands=}")

    baseband_spw_idxs = find_spw_in_basebands_list(
        bdf_path, spw_id, bdf_descr["basebands"]
    )

    ensure_presence_data_arrays(
        ["crossData", "autoData"], bdf_descr["binary_types"], bdf_path
    )

    baseband_description = bdf_descr["basebands"][baseband_spw_idxs[0]]
    spw_descr = baseband_description["spectralWindows"][baseband_spw_idxs[1]]
    scale_factor = spw_descr["scaleFactor"] or 1
    guessed_shape = define_visibility_shape(bdf_descr, baseband_spw_idxs)
    vis_per_subset = []
    while bdf_reader.hasSubset():
        try:
            subset = bdf_reader.getSubset()
        except ValueError as exc:
            logger.warning(f"Error in getSubset for {bdf_path=} {exc=}")
            return None

        vis_subset = load_vis_subset(
            subset,
            guessed_shape,
            scale_factor,
            baseband_spw_idxs,
            bdf_descr["processor_type"],
        )

        vis_per_subset.append(vis_subset)

    bdf_vis = np.concatenate(vis_per_subset)
    return bdf_vis


def define_visibility_shape(
    bdf_descr: dict, baseband_spw_idxs: tuple[int, int]
) -> tuple:
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


def load_vis_subset(
    subset: dict,
    guessed_shape: tuple,
    scale_factor: float,
    spw_baseband_num: tuple[int, int],
    processor_type: pyasdm.enumerations.ProcessorType,
) -> np.ndarray:

    if "crossData" in subset and subset["crossData"]["present"]:
        shape = guessed_shape[0:2] + guessed_shape[3:]
        # MUSTREMOVE (but to handle the uneven cases of the 'awful_workaround')
        # => cross_floats = (subset["crossData"]["arr"] / scale_factor).reshape(shape)
        cross_floats = (
            subset["crossData"]["arr"][: np.prod(shape)] / scale_factor
        ).reshape(shape)
        if processor_type == pyasdm.enumerations.ProcessorType.CORRELATOR:
            vis_subset = (
                cross_floats[:, :, spw_baseband_num, :, :, 0]
                + 1j * cross_floats[:, :, spw_baseband_num, :, :, 1]
            )
        else:
            # radiometer / spectrometer
            vis_subset = cross_floats
    elif "autoData" in subset and subset["autoData"]["present"]:
        shape = (guessed_shape[0:1],) + guessed_shape[2:-2]
        # MUSTREMOVE (but to handle the uneven cases of the 'awful_workaround')
        # => auto_floats = (subset["autoData"]["arr"] / scale_factor).reshape(shape)
        auto_floats = (
            subset["autoData"]["arr"][: np.prod(shape)] / scale_factor
        ).reshape(shape)
        vis_subset = auto_floats[:, :, spw_baseband_num, :, :]
    else:
        vis_subset = np.zeros(guessed_shape)

    return vis_subset


# ==> dims
# BAL ANT [:, :, spw_baseband_num, :, :]


def load_flags_from_partition_bdfs(
    bdf_paths: list[str], spw_id: int, array_slice: dict | None = None
) -> np.ndarray:

    cumulative_flag = []
    for bdf_path in bdf_paths:
        flag_blob = load_flags(bdf_path, spw_id, array_slice)

        if array_slice:
            indices = array_slice_to_indices(array_slice)
            flag_blob = flag_blob[*indices]

        cumulative_flag.append(flag_blob)

    return np.concatenate(cumulative_flag)


def check_flags_dims(flags_dims: list[str]) -> bool:
    exclude_unsupported_axis_names(flags_dims)


def load_flags(bdf_path: list[str], spw_id: int, array_slice: dict) -> np.ndarray:

    bdf_reader = pyasdm.bdf.BDFReader()
    bdf_reader.open(bdf_path)
    bdf_header = bdf_reader.getHeader()

    check_correlation_mode(bdf_header.getCorrelationMode())

    check_flags_dims(bdf_header.getAxesNames("flags"))

    bdf_descr = {
        # packed/TIM: dimensionality == 0
        "dimensionality": bdf_header.getDimensionality(),
        "num_times": bdf_header.getNumTime(),
        "processor_type": bdf_header.getProcessorType(),
        "binary_types": bdf_header.getBinaryTypes(),
        "apc": bdf_header.getAPClist(),
        "num_antenna": bdf_header.getNumAntenna(),
        "basebands": bdf_header.getBasebandsList(),
    }

    # TODO: working check for what's out there...
    if len(bdf_descr["basebands"]) not in [1, 3, 4]:
        raise RuntimeError(f"*** {len(basebands)=}, {basebands=}")

    baseband_spw_idxs = find_spw_in_basebands_list(
        bdf_path, spw_id, bdf_descr["basebands"]
    )

    ensure_presence_data_arrays(["flags"], bdf_descr["binary_types"], bdf_path)

    guessed_shape = define_flag_shape(bdf_descr, baseband_spw_idxs)

    flag_per_subset = []
    while bdf_reader.hasSubset():
        try:
            subset = bdf_reader.getSubset()
        except ValueError as exc:
            logger.warning(
                f"Error in getSubset for {bdf_path=} when trying to load "
                f"flags. Will use all-False. {exc=}"
            )
            return None

        flag_subset = load_flag_subset(subset, guessed_shape, baseband_spw_idxs)

        flag_per_subset.append(flag_subset)

    bdf_flag = np.concatenate(flag_per_subset)
    return bdf_flag


def define_flag_shape(bdf_descr: dict, baseband_spw_idxs: tuple[int, int]) -> tuple:
    baseband_len = len(bdf_descr["basebands"])
    antenna_len = bdf_descr["num_antenna"]
    baseline_len = int(antenna_len * (antenna_len - 1) / 2)
    baseband_descr = bdf_descr["basebands"][baseband_spw_idxs[0]]
    spw_len = len(baseband_descr["spectralWindows"])
    spw_descr = baseband_descr["spectralWindows"][baseband_spw_idxs[1]]
    # frequency_len = baseband_descr["numSpectralPoint"]
    polarization_len = len(spw_descr["crossPolProducts"]) or len(
        spw_descr["sdPolProducts"]
    )

    # if dimensionality==0, we have TIM dimension / packed format
    time_len = bdf_descr["num_time"] if bdf_descr["dimensionality"] == 0 else 1
    shape = (
        time_len,
        baseline_len + antenna_len,
        baseband_len,
        spw_len,
        polarization_len,
    )

    return shape


def load_flag_subset(
    subset: dict,
    guessed_shape: tuple,
    baseband_spw_idxs: tuple[int, int],
) -> np.ndarray:
    """
    Loads the flags array from one subset in a BDF.
    """

    if "flags" in subset and subset["flags"]["present"]:
        flag_array = subset["flags"]["arr"].reshape(guessed_shape)
        if len(guessed_shape) > 6:
            raise RuntimeError(
                f"Unexpected. Found {guessed_shape=}, {len(guessed_shape)=}, with {flag_array=}"
            )
        flag_subset = flag_array[..., baseband_spw_idxs[0], baseband_spw_idxs[1], :]
    else:
        flag_subset = np.full(guessed_shape, False, dtype="bool")

    return flag_subset
