import numpy as np

import pyasdm

import toolviper.utils.logger as logger


def find_spw_in_basebands_list(
    bdf_path: str, spw_id: int, basebands: list[dict]
) -> tuple[int, dict]:

    baseband_description = {}
    for bband in basebands:
        for spw in bband["spectralWindows"]:
            if spw_id == int(spw["sw"]) - 1:
                # Not sure if the lists are guaranteed to be sorted by id, so taking the id from name
                if "NOBB" == bband["name"]:
                    spw_baseband_num = 0
                else:
                    spw_baseband_num = int(bband["name"].split("_")[1]) - 1
                baseband_description = bband["spectralWindows"][0]

    if not baseband_description:
        # TODO: This is a highly dubious fallback for now...
        # Trying to figure out if there is some mapping from spw_ids
        err_msg = f"SPW {spw_id} not found in this BDF: {bdf_path}"
        # raise RuntimeError(err_msg)
        logger.warning(err_msg)
        spw_baseband_num = 1 - 1
        baseband_description = basebands[0]["spectralWindows"][0]

    return spw_baseband_num, baseband_description


def ensure_presence_data_arrays(
    data_array_names: list[str], bdf_header: pyasdm.bdf.BDFHeader, bdf_path
):

    binary_types = bdf_header.getBinaryTypes()
    for array_name in data_array_names:
        if array_name not in binary_types:
            raise RuntimeError(
                f"When trying to load visibility data from BDF: {bdf_path}, it does not "
                f"have array {array_name}"
            )


def exclude_unsupported_axis_names(dims):

    unsupported = ["APC", "STO", "HOL", "SPW"]

    for bad_dim in unsupported:
        if bad_dim in dims:
            raise RuntimeError(f"Unsupported dimension {bad_dim=} in {dims=}")


def check_cross_and_auto_data_dims(bdf_header: pyasdm.bdf.BDFHeader) -> bool:
    cross_data_dims = bdf_header.getAxesNames("crossData")
    auto_data_dims = bdf_header.getAxesNames("autoData")

    exclude_unsupported_axis_names(cross_data_dims)
    exclude_unsupported_axis_names(auto_data_dims)

    appears_single_dish = False
    # flags: ['BAL', 'ANT', 'BAB', 'POL'] / ['ANT', 'BAB', 'BIN', 'POL']
    # autodata: ['ANT', 'BAB', 'SPP', 'POL'] / ['ANT', 'BAB', 'BIN', 'POL']
    if cross_data_dims not in [["BAL", "BAB", "SPP", "POL"], ["BAL", "BAB", "POL"]]:
        logger.warning(
            f"crossData dims: {cross_data_dims}, autoData dims: {auto_data_dims}"
        )
        appears_single_dish = True
        if bdf_header.getCorrelationMode() != pyasdm.enumerations.CorrelationMode(
            "AUTO_ONLY"
        ):
            raise RuntimeError(
                "I'm confused. There is not crossData in this BDF but the "
                "correlator mode is not AUTO_ONLY, as expected for single-dish "
                f"data. {bdf_heder.getCorrelationMode()=}"
            )

    return appears_single_dish


def define_visibility_shape(
    bdf_header: pyasdm.bdf.BDFHeader,
    baseband_description: dict,
    appears_single_dish: bool,
) -> tuple[tuple, bool]:

    baseband_len = len(bdf_header.getBasebandsList())
    antenna_len = bdf_header.getNumAntenna()
    baseline_len = int(antenna_len * (antenna_len - 1) / 2)
    frequency_len = baseband_description["numSpectralPoint"]
    polarization_len = len(baseband_description["crossPolProducts"]) or len(
        baseband_description["sdPolProducts"]
    )

    cross_data_dims = bdf_header.getAxesNames("crossData")
    auto_data_dims = bdf_header.getAxesNames("autoData")
    if not appears_single_dish:
        shape = (1, baseline_len, baseband_len, frequency_len, polarization_len, 2)

    else:
        # Next ones, with "TIM" dimension, expected for single dish and radiometer,
        # all times in one subset
        num_time = bdf_header.getNumTime()
        if not cross_data_dims and auto_data_dims in [["TIM", "ANT", "SPP"]]:
            shape = (num_time, antenna_len, frequency_len, polarization_len)
        elif not cross_data_dims and auto_data_dims in [["ANT", "BAB", "BIN", "POL"]]:
            shape = (1, antenna_len, baseband_len, frequency_len, polarization_len)
        else:
            shape = (
                num_time,
                antenna_len,
                baseband_len,
                frequency_len,
                polarization_len,
            )

    no_baseband_dim = "BAB" not in cross_data_dims and "BAB" not in auto_data_dims

    return shape, no_baseband_dim


def load_visibilities_from_bdfs(
    bdf_paths: list[str], spw_id: int, array_slice: dict
) -> np.ndarray:

    if not array_slice:
        array_slice = None

    cumulative_vis = []
    for bdf_path in bdf_paths:
        cumulative_vis.append(load_visibilities(bdf_path, spw_id, array_slice))

    import time

    start = time.perf_counter()
    visibility = np.concatenate(cumulative_vis)
    end = time.perf_counter()
    return visibility


def load_visibilities(bdf_path: str, spw_id: int, array_slice: dict) -> np.ndarray:

    bdf_reader = pyasdm.bdf.BDFReader()
    bdf_reader.open(bdf_path)
    bdf_header = bdf_reader.getHeader()

    appears_single_dish = check_cross_and_auto_data_dims(bdf_header)
    basebands = bdf_header.getBasebandsList()
    spw_baseband_num, baseband_description = find_spw_in_basebands_list(
        bdf_path, spw_id, basebands
    )

    ensure_presence_data_arrays(["crossData", "autoData"], bdf_header, bdf_path)

    scale_factor = baseband_description["scaleFactor"] or 1
    shape, no_baseband_dim = define_visibility_shape(
        bdf_header, baseband_description, appears_single_dish
    )
    cumulative_vis = []
    while bdf_reader.hasSubset():
        try:
            subset = bdf_reader.getSubset()
        except ValueError as exc:
            logger.warning(f"Error in getSubset for {bdf_path=} {exc=}")
            return np.zeros(shape)

        vis_subset = load_vis_subset(
            subset, shape, scale_factor, spw_baseband_num, no_baseband_dim
        )

        cumulative_vis.append(vis_subset)

    bdf_vis = np.concatenate(cumulative_vis)
    return bdf_vis


def load_vis_subset(
    subset: dict,
    shape: tuple,
    scale_factor: float,
    spw_baseband_num: int,
    no_baseband_dim: bool,
) -> np.ndarray:

    if "crossData" in subset and subset["crossData"]["present"]:
        # assuming dims ['BAL', 'BAB', 'SPP', 'POL']
        cross_floats = (subset["crossData"]["arr"] / scale_factor).reshape(shape)
        if no_baseband_dim:
            if len(cross_floats.shape[-1]) == 2:
                vis_subset = cross_floats[..., 0] + 1j * cross_floats[..., 1]
            else:
                # radiometer
                vis_subset = cross_floats
        else:
            vis_subset = (
                cross_floats[:, :, spw_baseband_num, :, :, 0]
                + 1j * cross_floats[:, :, spw_baseband_num, :, :, 1]
            )
    elif "autoData" in subset and subset["autoData"]["present"]:
        auto_floats = (subset["autoData"]["arr"] / scale_factor).reshape(shape)
        if no_baseband_dim:
            vis_subset = auto_floats
        else:
            vis_subset = auto_floats[:, :, spw_baseband_num, :, :]
    else:
        vis_subset = np.zeros(shape)

    return vis_subset


def define_flag_shape(
    bdf_header: pyasdm.bdf.BDFHeader,
    baseband_description: dict,
) -> tuple:

    baseband_len = len(bdf_header.getBasebandsList())
    antenna_len = bdf_header.getNumAntenna()
    baseline_len = int(antenna_len * (antenna_len - 1) / 2)
    frequency_len = baseband_description["numSpectralPoint"]
    polarization_len = len(baseband_description["crossPolProducts"]) or len(
        baseband_description["sdPolProducts"]
    )
    flag_dims = bdf_header.getAxesNames("flags")

    exclude_unsupported_axis_names(flag_dims)

    if flag_dims in [["BAL", "BAB", "POL"]]:
        shape = (1, baseline_len, baseband_len, polarization_len)
    elif flag_dims in [["BAL", "ANT", "BAB", "POL"]]:
        shape = (1, baseline_len + antenna_len, baseband_len, polarization_len)
    elif flag_dims in [["ANT", "BAB", "BIN", "POL"]]:
        shape = (1, antenna_len, baseband_len, frequency_len, polarization_len)
    elif flag_dims in [["TIM", "ANT", "BAB", "BIN", "POL"]]:
        num_time = bdf_header.getNumTime()
        shape = (num_time, antenna_len, baseband_len, frequency_len, polarization_len)
    else:
        # flag_dims == [], etc.
        # Typically for radiometer / total power data. No flags. Just fill it.
        # This should also imply 'not subset["flags"]["present"]'
        appears_single_dish = check_cross_and_auto_data_dims(bdf_header)
        shape, _no_baseband_dim = define_visibility_shape(
            bdf_header, baseband_description, appears_single_dish
        )

    return shape


def load_flags_from_bdfs(
    bdf_paths: list[str], spw_id: int, array_slice: dict
) -> np.ndarray:

    if not array_slice:
        array_slice = None

    all_flag = []
    for bdf_path in bdf_paths:
        all_flag.append(load_flags(bdf_path, spw_id, array_slice))

    return np.concatenate(all_flag)


def check_flags_dims(bdf_header: pyasdm.bdf.BDFHeader) -> bool:

    flags_dims = bdf_header.getAxesNames("flags")
    exclude_unsupported_axis_names(flags_dims)


def define_flag_shape_when_not_present(
    bdf_header: pyasdm.bdf.BDFHeader, shape: tuple
) -> tuple[tuple, bool]:

    flag_dims = bdf_header.getAxesNames("flags")
    cross_data_dims = bdf_header.getAxesNames("crossData")
    auto_data_dims = bdf_header.getAxesNames("autoData")

    # This happens for example for dims: ['TIM', 'ANT', 'BAB', 'BIN', 'POL']
    baseband_dim_is_3rd_last = (len(flag_dims) >= 3 and flag_dims[-3] == "BAB") or (
        not flag_dims
        and (
            (len(cross_data_dims) >= 3 and cross_data_dims[-3] == "BAB")
            or (len(auto_data_dims) >= 3 and auto_data_dims[-3] == "BAB")
        )
    )

    # leave out the BAB dim
    if baseband_dim_is_3rd_last:
        shape_when_not_present = shape[0:-3] + shape[-2:]
    else:
        shape_when_not_present = shape[0:-2] + (shape[-1],)

    return shape_when_not_present, baseband_dim_is_3rd_last


def load_flags(bdf_path: list[str], spw_id: int, array_slice: dict) -> np.ndarray:

    bdf_reader = pyasdm.bdf.BDFReader()
    bdf_reader.open(bdf_path)
    bdf_header = bdf_reader.getHeader()

    check_flags_dims(bdf_header)
    basebands = bdf_header.getBasebandsList()
    spw_baseband_num, baseband_description = find_spw_in_basebands_list(
        bdf_path, spw_id, basebands
    )

    ensure_presence_data_arrays(["flags"], bdf_header, bdf_path)

    shape = define_flag_shape(bdf_header, baseband_description)
    shape_when_not_present, baseband_dim_is_3rd_last = (
        define_flag_shape_when_not_present(bdf_header, shape)
    )
    cumulative_flag = []
    while bdf_reader.hasSubset():
        try:
            subset = bdf_reader.getSubset()
        except ValueError as exc:
            logger.warning(
                f"Error in getSubset for {bdf_path=} when trying to load "
                f"flags. Will use all-False. {exc=}"
            )
            # Trying to go to a next subset after this failue will also
            # produce a BDFReaderException.BDFReaderException: BDFReaderException
            # Did not find expected field 'CONTENT-TYPE' in 'MIME-Version: 1.0'
            flag_subset = np.full(
                shape_when_not_present,
                False,
                dtype="bool",
            )
            cumulative_flag.append(flag_subset)
            break

        flag_subset = load_flag_subset(
            subset,
            shape,
            spw_baseband_num,
            baseband_dim_is_3rd_last,
            shape_when_not_present,
        )

        cumulative_flag.append(flag_subset)

    bdf_flag = np.concatenate(cumulative_flag)
    return bdf_flag


def load_flag_subset(
    subset: dict,
    shape: tuple,
    spw_baseband_num: int,
    baseband_dim_is_3rd_last: bool,
    shape_when_not_present: tuple,
) -> np.ndarray:
    """
    Loads the flags array from one subset in a BDF.
    """

    if "flags" in subset and subset["flags"]["present"]:
        flag_array = subset["flags"]["arr"].reshape(shape)

        if baseband_dim_is_3rd_last:
            flag_subset = flag_array[:, :, spw_baseband_num, :, :]
        else:
            if len(shape) == 4:
                flag_subset = flag_array[:, :, spw_baseband_num, :]
            elif len(shape) == 5:
                flag_subset = flag_array[:, :, :, spw_baseband_num, :]
            else:
                raise RuntimeError(f"Found {shape=}, {len(shape)=}, with {flag_array=}")

    else:
        flag_subset = np.full(shape_when_not_present, False, dtype="bool")

    return flag_subset
