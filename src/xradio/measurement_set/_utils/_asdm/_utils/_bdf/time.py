import os

import numpy as np
import pandas as pd

import toolviper.utils.logger as logger

import pyasdm

from xradio.measurement_set._utils._asdm._utils.time import convert_time_asdm_to_unix


def get_times_from_bdfs(
    bdf_paths: list[str], scans_metadata: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract and convert time information from BDF files and scan metadata.

    This function attempts to read timing information from BDF files and falls back to
    scan metadata if BDF reading fails. All times are converted to Unix timestamp format.

    Parameters
    ----------
    bdf_paths : list[str]
        List of paths to BDF (Binary Data Format) files.
    scans_metadata : pd.DataFrame
        DataFrame containing scan metadata, must include 'startTime' and 'endTime' columns
        for fallback mechanism.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        A tuple containing four numpy arrays:
        - time_centers: Array of center times for each scan
        - durations: Array of scan durations
        - actual_times: Array of actual (measured) times
        - actual_durations: Array of actual (measured) durations

    Notes
    -----
    If BDF reading fails, the function falls back to using scan metadata for time_centers
    and returns zero-filled arrays for durations, actual_times, and actual_durations.

    """

    try:
        time_centers, durations, actual_times, actual_durations = load_times_from_bdfs(
            bdf_paths
        )
        for time_var in time_centers, durations, actual_times, actual_durations:
            time_var = convert_time_asdm_to_unix(time_var)

    except AttributeError as exc:
        logger.warning(
            f"Could not read nominal and actual times and durations from BDFs. {exc=}"
        )
        raise exc
        time_centers = convert_time_asdm_to_unix(
            scans_metadata["startTime"].values
        ) + convert_time_asdm_to_unix(scans_metadata["endTime"].values)
        durations = np.zeros(2)

        actual_times = np.zeros(2)
        actual_durations = np.zeros(2)

    return time_centers, durations, actual_times, actual_durations


def make_blob_info(bdf_header: pyasdm.bdf.BDFHeader) -> dict:
    basebands_info = ""
    baseband_idx = 1
    for baseband in bdf_header.getBasebandsList():
        basebands_info += f"{baseband['name']} "
        for spw in baseband["spectralWindows"]:
            spw_idx = f"spw_{spw['sw']}"
            spectral_points = spw["numSpectralPoint"]
            num_bin = spw["numBin"]
            cross_products = len(spw["crossPolProducts"])
            basebands_info += f"{spw_idx} {spectral_points} {num_bin} {cross_products} "

    bdf_info = {
        "execblock_uid": bdf_header.getExecBlockUID(),
        "dataOID": bdf_header.getDataOID(),
        "title": bdf_header.getTitle(),
        "correlation_mode": bdf_header.getCorrelationMode(),
        "processor_type": bdf_header.getProcessorType(),
        "spectral_resolution_type": bdf_header.getSpectralResolutionType(),
        "apc_list": " ".join(map(str, bdf_header.getAPClist())),
        "dimensionality": bdf_header.getDimensionality(),
        "num_time": bdf_header.getNumTime(),
        "binary_types": bdf_header.getBinaryTypes(),
        "num_antenna": bdf_header.getNumAntenna(),
        "actual_times_size": bdf_header.getSize("actualTimes"),
        "actual_times_axes": " ".join(map(str, bdf_header.getAxes("actualTimes"))),
        "actual_durations_size": bdf_header.getSize("actualDurations"),
        "actual_durations_axes": " ".join(
            map(str, bdf_header.getAxes("actualDurations"))
        ),
        "flags_size": bdf_header.getSize("flags"),
        "flags_axes": " ".join(map(str, bdf_header.getAxes("flags"))),
        "auto_data_size": bdf_header.getSize("autoData"),
        "auto_data_axes": " ".join(map(str, bdf_header.getAxes("autoData"))),
        "cross_data_size": bdf_header.getSize("crossData"),
        "cross_data_axes": " ".join(map(str, bdf_header.getAxes("crossData"))),
        "zero_lags_size": bdf_header.getSize("zeroLags"),
        "zero_lags_axes": " ".join(map(str, bdf_header.getAxes("zeroLags"))),
        "basebands_spws_points_bins_crossx": basebands_info,
    }
    blob_info = pd.DataFrame([bdf_info]).set_index(["execblock_uid", "dataOID"])

    return blob_info


def save_blob_info(csv_path: str, blob_info: pd.DataFrame) -> None:
    do_header = (
        False if os.path.isfile(csv_path) and os.path.getsize(csv_path) > 0 else True
    )
    blob_info.to_csv(csv_path, mode="a", header=do_header)


def load_times_from_bdfs(
    bdf_paths: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Read timing information from Binary Data Format (BDF) files.
    This function processes a list of BDF files to extract timing information for each data subset,
    including center times, durations, and their actual measured values. WVR (Water Vapor Radiometer)
    data is currently not supported and returns zero arrays.

    Parameters
    ----------
    bdf_paths : list[str]
        List of paths to BDF files to be processed.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Returns a tuple containing four numpy arrays:
        - time_centers : Array of midpoint times for each subset (in seconds)
        - durations : Array of nominal durations for each subset (in seconds)
        - actual_times : Array of actual measured times for each subset (in seconds)
        - actual_durations : Array of actual measured durations for each subset (in seconds)

    Notes
    -----
    Times are converted from nanoseconds to seconds (divided by 1e9) in the output.
    For subsets where actual times/durations are not present, the nominal values are used instead.
    WVR data is currently not supported and will return arrays of zeros.
    """

    all_time_centers, all_durations, all_actual_times, all_actual_durations = (
        [],
        [],
        [],
        [],
    )

    for bdf_path in bdf_paths:
        bdf_reader = pyasdm.bdf.BDFReader()
        try:
            bdf_reader.open(bdf_path)
            bdf_header = bdf_reader.getHeader()
            logger.debug(" * In load_times_from_bdf, {bdf_path=} *")
            logger.debug(bdf_header)
            blob_info = make_blob_info(bdf_header)
            save_blob_info("xradio_asdm_blob_time_etc_info.csv", blob_info)
        finally:
            bdf_reader.close()

        midpoint, interval, actual_times, actual_durations = read_times_bdf(bdf_path)
        all_time_centers.append(midpoint)
        all_durations.append(interval)
        all_actual_times.append(actual_times)
        all_actual_durations.append(actual_durations)

    return (
        np.concatenate(all_time_centers),
        np.concatenate(all_durations),
        np.concatenate(all_actual_times),
        np.concatenate(all_actual_durations),
    )


def read_times_bdf(
    bdf_path: str,
) -> tuple[list, list, list, list]:
    bdf_reader = pyasdm.bdf.BDFReader()
    bdf_reader.open(bdf_path)

    # TODO: I'd hope this is not a general problem but for some test datasets WVR SPWs produce
    # failures related to the BDF dims:
    #
    # wvr_title = bdf_header.isWVR()
    # if wvr_title:
    #    return np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1)
    #
    # This anyway seems to happen for other BDFs, so handling it in the except ValueError below...
    all_midpoints, all_actual_times, all_intervals, all_actual_durations = (
        [],
        [],
        [],
        [],
    )
    while bdf_reader.hasSubset():
        try:
            subset = bdf_reader.getSubset()
        except ValueError as exc:
            # Example: (cycle 3, 2015.1.00665.S/uid___A002_Xae4720_X57fe):
            # File... BDFReader.py", line 805, in _requireSDMDataSubsetMIMEPart
            #    intNum = int(projectPathParts[3])
            # ValueError: invalid literal for int() with base 10: ''
            logger.warning(f"Error in getSubset for {bdf_path=} {exc=}")
            bdf_reader.close()
            return np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1)

        # dims:
        # BAL ANT BAB POL / channel avg data
        # ANT BAB BIN POL / radiometer total power /
        # BAL ANT BAB SPW POL
        # BAL ANT BAB?
        # actual_times_dims = bdf_header.getAxesNames("actualTimes")

        midpoint = subset["midpointInNanoSeconds"] / 1e9
        all_midpoints.append(midpoint)
        if subset["actualTimes"]["present"]:
            # note, for now ignoring the unclear/values-dont-add-up baseline? dim, just first
            all_actual_times.append(subset["actualTimes"]["arr"][0] / 1e9)
        else:
            all_actual_times.append(midpoint)

        interval = subset["intervalInNanoSeconds"] / 1e9
        all_intervals.append(interval)
        if subset["actualDurations"]["present"]:
            # note, for now ignoring the unclear baseline? dim, just first
            all_actual_durations.append(subset["actualDurations"]["arr"][0] / 1e9)
        else:
            all_actual_durations.append(interval)

    bdf_reader.close()

    return all_midpoints, all_intervals, all_actual_times, all_actual_durations
