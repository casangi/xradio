import time

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

    # This effectively assumes, from BDF doc: "The final three axes, STO, POL and HOL,
    # also appear at the same level in the axis hierarchy; however, only one of these
    # axes will normally appear for a given binary component type.
    unsupported = ["STO", "HOL"]

    for bad_dim in unsupported:
        if bad_dim in dims:
            raise RuntimeError(f"Unsupported dimension {bad_dim=} in {dims=}")


def check_cross_and_auto_data_dims(bdf_header: pyasdm.bdf.BDFHeader) -> bool:
    cross_data_dims = bdf_header.getAxesNames("crossData")
    auto_data_dims = bdf_header.getAxesNames("autoData")

    exclude_unsupported_axis_names(cross_data_dims)
    exclude_unsupported_axis_names(auto_data_dims)

    # could be single dish, radiometer, or even average flows
    appears_not_interferometric = False
    # flags: ['BAL', 'ANT', 'BAB', 'POL'] / ['ANT', 'BAB', 'BIN', 'POL']
    # autodata: ['ANT', 'BAB', 'SPP', 'POL'] / ['ANT', 'BAB', 'BIN', 'POL']
    if cross_data_dims not in [
        ["BAL", "BAB", "SPP"],
        ["BAL", "BAB", "POL"],
        ["BAL", "BAB", "SPP", "POL"],
        ["BAL", "BAB", "SPW", "SPP", "POL"],
        [
            "BAL",
            "BAB",
            "SPW",
            "POL",
        ],  # no explicit SPP, for example: ALMA ACA Correlator Channel Average Data
        ["BAL", "BAB", "APC", "SPP", "POL"],
        ["BAL", "BAB", "SPW", "APC", "SPP", "POL"],
    ]:
        logger.warning(
            f"crossData dims: {cross_data_dims}, autoData dims: {auto_data_dims}"
        )
        if "APC" in cross_data_dims:
            log.warning(f"Found APC dim with {bdf_header.getAPCList()=}")
        appears_not_interferometric = True
        if bdf_header.getCorrelationMode() != pyasdm.enumerations.CorrelationMode(
            "AUTO_ONLY"
        ):
            raise RuntimeError(
                "I'm confused. There is no crossData in this BDF but the "
                "correlator mode is not AUTO_ONLY, as expected for single-dish "
                f"data. {bdf_header.getCorrelationMode()=}"
            )

    return appears_not_interferometric


def define_visibility_shape(
    bdf_header: pyasdm.bdf.BDFHeader,
    baseband_description: dict,
    appears_not_interferometric: bool,
) -> tuple[tuple, bool]:

    def awful_workaround(basebands: dict) -> int:
        """
        Awful workaround to keep us going when the SPWs in a
        same BDF have different number of channels.

        This is wrong but keeps us moving. The data arrays
        will be very scrambled but having the proper shapes. It
        wouldn't be worth re-arranging the arrays properly here
        now.

        Can be:
        - different number of channels in the SPWs, with the same
          number of SPWs per baseband (typically 1)
        - Different number of SPWs in the basebands
        """

        baseband_spws_frequency_len = [
            spw["numSpectralPoint"]
            for baseband in basebands
            for spw in baseband["spectralWindows"]
        ]
        frequency_len = 0
        if baseband_spws_frequency_len.count(baseband_spws_frequency_len[0]) != len(
            baseband_spws_frequency_len
        ):
            logger.warning("awful workaround for now!")
            frequency_len = int(np.mean(baseband_spws_frequency_len))

        return frequency_len

    basebands = bdf_header.getBasebandsList()
    baseband_len = len(basebands)
    antenna_len = bdf_header.getNumAntenna()
    baseline_len = int(antenna_len * (antenna_len - 1) / 2)
    frequency_len = baseband_description["numSpectralPoint"]
    polarization_len = len(baseband_description["crossPolProducts"]) or len(
        baseband_description["sdPolProducts"]
    )

    frequency_len = awful_workaround(basebands) or frequency_len

    cross_data_dims = bdf_header.getAxesNames("crossData")
    auto_data_dims = bdf_header.getAxesNames("autoData")
    if not appears_not_interferometric:
        shape = (1, baseline_len, baseband_len, frequency_len, polarization_len, 2)

    else:
        # Next ones, with "TIM" dimension, expected for single dish and radiometer,
        # all times in one subset
        num_time = bdf_header.getNumTime()
        if not cross_data_dims and auto_data_dims in [["TIM", "ANT", "SPP"]]:
            shape = (num_time, antenna_len, frequency_len, polarization_len)
        elif not cross_data_dims and auto_data_dims in [["ANT", "BAB", "BIN", "POL"]]:
            shape = (1, antenna_len, baseband_len, frequency_len, polarization_len)
        elif not cross_data_dims and auto_data_dims in [
            ["ANT", "BAB", "SPW", "SPP", "POL"]
        ]:
            # TODO: basebands can have different number of SPWs
            # Example: E2E5.1.00026.S/rawdata/uid___A002_Xc3412f_X2a7d
            spw_len = 1
            shape = (
                1,
                antenna_len,
                baseband_len,
                spw_len,
                frequency_len,
                polarization_len,
            )
        else:
            # [ANT, BAB, SPP, POL] and others
            shape = (
                num_time,
                antenna_len,
                baseband_len,
                frequency_len,
                polarization_len,
            )

    no_baseband_dim = "BAB" not in cross_data_dims and "BAB" not in auto_data_dims

    return shape, no_baseband_dim


def array_slice_to_indices(array_slice: dict) -> tuple[range, range, range, range]:

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
        frequency_slice = array_slice["polarization"]


def load_visibilities_from_bdfs(
    bdf_paths: list[str], spw_id: int, array_slice: dict
) -> np.ndarray:

    if not array_slice:
        array_slice = None

    cumulative_vis = []
    for bdf_path in bdf_paths:
        visibility_blob = load_visibilities(bdf_path, spw_id, array_slice)

        if array_slice:
            indices = array_slice_to_indices(array_slice)
            visibility_blob = visibility_blob[*indices]

        cumulative_vis.append(visibility_blob)

    start = time.perf_counter()
    visibility = np.concatenate(cumulative_vis)
    end = time.perf_counter()
    logger.info(
        f"Loaded visibiilty, with {visibility.shape=} from {len(bdf_paths)=} blobs, time: {end-start:.6}"
    )

    return visibility


def load_visibilities(bdf_path: str, spw_id: int, array_slice: dict) -> np.ndarray:

    bdf_reader = pyasdm.bdf.BDFReader()
    bdf_reader.open(bdf_path)
    bdf_header = bdf_reader.getHeader()

    correlation_mode = bdf_header.getCorrelationMode()
    if correlation_mode == pyasdm.enumerations.CorrelationMode.CROSS_ONLY:
        raise RuntimeError(f" {correlation_mode=} {bdf_header=}")

    appears_not_interferometric = check_cross_and_auto_data_dims(bdf_header)
    basebands = bdf_header.getBasebandsList()
    # TODO: working check for what's out there...
    if len(basebands) not in [1, 3, 4]:
        raise RuntimeError(f"*** {len(basebands)=}, {basebands=}")
    spw_baseband_num, baseband_description = find_spw_in_basebands_list(
        bdf_path, spw_id, basebands
    )

    ensure_presence_data_arrays(["crossData", "autoData"], bdf_header, bdf_path)

    scale_factor = baseband_description["scaleFactor"] or 1
    shape, no_baseband_dim = define_visibility_shape(
        bdf_header, baseband_description, appears_not_interferometric
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
        # assuming dims ['BAL', 'BAB', 'SPP', 'POL'] / when ['BAL', 'BAB', 'SPW', 'SPP', 'POL'],
        # truncating if needed

        # MUSTREMOVE (but to handle the uneven cases of the 'awful_workaround')
        # cross_floats = (subset["crossData"]["arr"] / scale_factor).reshape(shape)
        cross_floats = (
            subset["crossData"]["arr"][: np.prod(shape)] / scale_factor
        ).reshape(shape)
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
        # MUSTREMOVE (but to handle the uneven cases of the 'awful_workaround')
        # auto_floats = (subset["autoData"]["arr"] / scale_factor).reshape(shape)
        auto_floats = (
            subset["autoData"]["arr"][: np.prod(shape)] / scale_factor
        ).reshape(shape)
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
        appears_not_interferometric = check_cross_and_auto_data_dims(bdf_header)
        shape, _no_baseband_dim = define_visibility_shape(
            bdf_header, baseband_description, appears_not_interferometric
        )

    return shape


def load_flags_from_bdfs(
    bdf_paths: list[str], spw_id: int, array_slice: dict
) -> np.ndarray:

    if not array_slice:
        array_slice = None

    cumulative_flag = []
    for bdf_path in bdf_paths:
        flag_blob = load_flags(bdf_path, spw_id, array_slice)

        if array_slice:
            indices = array_slice_to_indices(array_slice)
            flag_blob = flag_blob[*indices]

        cumulative_flag.append(flag_blob)

    return np.concatenate(cumulative_flag)


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
