import numpy as np

import pyasdm

import toolviper.utils.logger as logger


def find_spw_in_basebands_list(spw_id: int, basebands: list[dict]) -> tuple[int, dict]:

    for bband in basebands:
        for spw in bband["spectralWindows"]:
            if spw_id == int(spw["sw"]) - 1:
                # Not sure if the lists are guaranteed to be sorted by id, so taking the id from name
                spw_baseband_num = int(bband["name"].split("_")[1])
                return spw_baseband_num, bband["spectralWindows"][0]

    return 0, {}


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


def load_visibilities_from_bdfs(
    bdf_paths: list[str], spw_id: int, array_slice: dict
) -> np.ndarray:

    if not array_slice:
        array_slice = None

    all_vis = []
    for bdf_path in bdf_paths:
        all_vis.append(load_visibilities(bdf_path, spw_id, array_slice))

    return np.concatenate(all_vis)


def load_visibilities(bdf_path: str, spw_id: int, array_slice: dict) -> np.ndarray:

    bdf_reader = pyasdm.bdf.BDFReader()
    bdf_reader.open(bdf_path)
    bdf_header = bdf_reader.getHeader()

    basebands = bdf_header.getBasebandsList()
    spw_baseband_num, baseband_description = find_spw_in_basebands_list(
        spw_id, basebands
    )
    if not baseband_description:
        err_msg = f"SPW {spw_id} not found in this BDF: {bdf_path}"
        # raise RuntimeError(err_msg)
        # Trying to figure out if there is some mapping from spw_ids
        logger.warning(err_msg)
        spw_baseband_num = 1
        baseband_description = basebands[0]["spectralWindows"][0]

    ensure_presence_data_arrays(["crossData", "autoData"], bdf_header, bdf_path)

    scale_factor = baseband_description["scaleFactor"] or 1
    all_subset = []
    antenna_len = bdf_header.getNumAntenna()
    baseline_len = int(antenna_len * (antenna_len - 1) / 2)
    frequency_len = baseband_description["numSpectralPoint"]
    polarization_len = len(baseband_description["crossPolProducts"])
    shape = (baseline_len, frequency_len, polarization_len, 2)
    while bdf_reader.hasSubset():
        try:
            subset = bdf_reader.getSubset()
        except ValueError:
            logger.warning(f"Error in getSubset for {bdf_path=} {exc=}")
            return np.zeros(1)

        # assuming dims ['BAL', 'BAB', 'SPP', 'POL']
        if "crossData" in subset and subset["crossData"]["present"]:
            cross_floats = (subset["crossData"]["arr"] / scale_factor).reshape(shape)
            cross_vis = cross_floats[:, :, :, 0] + 1j * cross_floats[:, :, :, 1]
        else:
            cross_vis = np.zeros(0)
        if False and "autoData" in subset and subset["autoData"]["present"]:
            auto_floats = (subset["autoData"]["arr"] / scale_factor).reshape(shape)
            auto_vis = auto_floats[:, :, :, 0] + 1j * auto_floats[:, :, :, 1]
        else:
            auto_vis = np.zeros(0)

        all_subset.append(cross_vis)

    bdf_vis = np.concatenate(all_subset)
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
        except ValueError:
            logger.warning(f"Error in getSubset for {bdf_path=} {exc=}")
            return np.zeros(1)
