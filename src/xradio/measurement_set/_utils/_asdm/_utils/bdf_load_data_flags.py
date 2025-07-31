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


def check_cross_and_auto_data_dims(bdf_header: pyasdm.bdf.BDFHeader) -> bool:
    cross_data_dims = bdf_header.getAxesNames("crossData")
    auto_data_dims = bdf_header.getAxesNames("autoData")
    appears_single_dish = False
    # flags: ['BAL', 'ANT', 'BAB', 'POL'] / ['ANT', 'BAB', 'BIN', 'POL']
    # autodata: ['ANT', 'BAB', 'SPP', 'POL'] / ['ANT', 'BAB', 'BIN', 'POL']
    if cross_data_dims not in [["BAL", "BAB", "SPP", "POL"], ["BAL", "BAB", "POL"]]:
        logger.warning(
            f"crossData dims: {cross_data_dims}, autoData dims: {auto_data_dims}"
        )
        appears_single_dish = True

    return appears_single_dish


def define_visibility_flag_shape(
    bdf_header: pyasdm.bdf.BDFHeader,
    baseband_description: dict,
    appears_single_dish: bool,
) -> tuple:

    baseband_len = len(bdf_header.getBasebandsList())
    antenna_len = bdf_header.getNumAntenna()
    baseline_len = int(antenna_len * (antenna_len - 1) / 2)
    frequency_len = baseband_description["numSpectralPoint"]
    polarization_len = len(baseband_description["crossPolProducts"]) or len(
        baseband_description["sdPolProducts"]
    )

    if not appears_single_dish:
        shape = (1, baseline_len, baseband_len, frequency_len, polarization_len, 2)
    else:
        shape = (1, antenna_len, baseband_len, frequency_len, polarization_len)

    return shape


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
    shape = define_visibility_flag_shape(
        bdf_header, baseband_description, appears_single_dish
    )
    cumulative_vis = []
    count = 0
    while bdf_reader.hasSubset():
        try:
            subset = bdf_reader.getSubset()
        except ValueError as exc:
            logger.warning(f"Error in getSubset for {bdf_path=} {exc=}")
            return np.zeros(1)

        if "crossData" in subset and subset["crossData"]["present"]:
            # assuming dims ['BAL', 'BAB', 'SPP', 'POL']
            cross_floats = (subset["crossData"]["arr"] / scale_factor).reshape(shape)
            vis_subset = (
                cross_floats[:, :, spw_baseband_num, :, :, 0]
                + 1j * cross_floats[:, :, spw_baseband_num, :, :, 1]
            )
        elif "autoData" in subset and subset["autoData"]["present"]:
            # Support SD a bit for now...
            auto_floats = (subset["autoData"]["arr"] / scale_factor).reshape(shape)
            vis_subset = auto_floats[:, :, spw_baseband_num, :, :]
        else:
            vis_subset = np.zeros(shape)

        cumulative_vis.append(vis_subset)
        count += 1

    bdf_vis = np.concatenate(cumulative_vis)
    return bdf_vis


def load_flags_from_bdfs(
    bdf_paths: list[str], spw_id: int, array_slice: dict
) -> np.ndarray:

    if not array_slice:
        array_slice = None

    all_flag = []
    for bdf_path in bdf_paths:
        all_flag.append(load_vis_flags_from_bdf(bdf_path, spw_id, array_slice))

    return np.concatenate(all_flag)


def load_flags(bdf_path: list[str], spw_id: int, array_slice: dict) -> np.ndarray:

    bdf_reader = pyasdm.bdf.BDFReader()
    bdf_reader.open(bdf_path)
    bdf_header = bdf_reader.getHeader()
    ensure_presence_data_arrays(["flags"], bdf_header, bdf_path)
    while bdf_reader.hasSubset():
        try:
            subset = bdf_reader.getSubset()
        except ValueError as exc:
            logger.warning(f"Error in getSubset for {bdf_path=} {exc=}")
            return np.zeros(1)
