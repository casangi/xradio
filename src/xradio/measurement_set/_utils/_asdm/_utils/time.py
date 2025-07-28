import numpy as np
import pandas as pd

import pyasdm

import toolviper.utils.logger as logger


def convert_time_asdm_to_unix(times_asdm: np.ndarray):
    """Convert ASDM time values to Unix timestamps.

    The ASDM stores time values as nanoseconds since the MJD epoch.
    This function converts these values to Unix timestamps (seconds since the Unix epoch).

    Parameters
    ----------
    times_asdm : (np.ndarray)
        An array of ASDM time values (nanoseconds since MJD epoch). Each element is expected
        to have a `get()` method that returns the time value as a float.

    Returns
    -------
    list
        A list of Unix timestamps (seconds since the Unix epoch), converted from the input ASDM times.

    Notes
    -----
    - The ASDM time scale is TAI, while Unix time is UTC.  This function does not
      explicitly account for the difference between TAI and UTC (leap seconds).
      The comment in the original code suggests this is acceptable because the
      first leap second was added in 1972, which is after the MJD epoch.
    - The function iterates through the input `times_asdm` array, calling the `get()`
      method on each element to retrieve the ASDM time value.
    - It subtracts a constant `MJD_TO_UNIX_TIME_DELTA * 1e9` from the ASDM time value
      to convert it to nanoseconds since the Unix epoch.
    - It divides the result by `1e9` to convert it to seconds since the Unix epoch (Unix timestamp).
    """
    # beware: ArrayTime uses TAI scale, not UTC -> look for UTCTime
    # class in ASDM code?
    # This should be fine wrt the shift done in this function, as
    # the first leap second was added in 1972.30.06?
    #
    # Also,  functions like `asdm_interval.toFITS()` produce tai-referenced
    # values/
    # [((asdm_interval.toFITS()) for asdm_interval in main_df["time"].values]

    MJD_TO_UNIX_TIME_DELTA = 3_506_716_800
    MJD_TO_UNIX_TIME_DELTA_NS = 3_506_716_800 * 1e9

    if isinstance(times_asdm[0], pyasdm.types.ArrayTime):
        asdm_times_float = np.array(
            [asdm_interval.get() for asdm_interval in times_asdm]
        )
    else:
        asdm_times_float = times_asdm

    times_unix = (asdm_times_float - MJD_TO_UNIX_TIME_DELTA_NS) / 1e9

    # alternatively convert via pd:
    # return pd.to_datetime(time_values, unit="s")

    return times_unix


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
        time_centers, durations, actual_times, actual_durations = read_times_from_bdfs(
            bdf_paths
        )
        for time_var in time_centers, durations, actual_times, actual_durations:
            time_var = convert_time_asdm_to_unix(time_var)

    except AttributeError as exc:
        logger.warning(
            f"Could not read nominal and actual times and durations from BDFs. {exc=}"
        )
        time_centers = convert_time_asdm_to_unix(
            scans_metadata["startTime"].values
        ) + convert_time_asdm_to_unix(scans_metadata["endTime"].values)
        durations = np.zeros(2)

        actual_times = np.zeros(2)
        actual_durations = np.zeros(2)

    return time_centers, durations, actual_times, actual_durations


def read_times_from_bdfs(
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
    bdf_header = bdf_reader.getHeader()
    wvr_title = bdf_header.isWVR()

    # TODO: I'd hope this is not a general problem but for some test datasets WVR SPWs produce
    # failures related to the BDF dims:
    # if wvr_title:
    #    return np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1)
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
        except ValueError:
            # Example: (cycle 3, 2015.1.00665.S/uid___A002_Xae4720_X57fe):
            # File... BDFReader.py", line 805, in _requireSDMDataSubsetMIMEPart
            #    intNum = int(projectPathParts[3])
            # ValueError: invalid literal for int() with base 10: ''
            bdf_reader.close()
            return np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1)

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
