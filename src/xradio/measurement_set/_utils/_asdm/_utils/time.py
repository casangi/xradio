import numpy as np

import pyasdm


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
    MJD_TO_UNIX_TIME_DELTA_NS = MJD_TO_UNIX_TIME_DELTA * 1e9

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
