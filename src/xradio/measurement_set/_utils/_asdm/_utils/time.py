"""
Conversion of ASDM time values (ArrayTime) to the time representation used in the
MSv4 datasets produced by the ASDM backend.
"""

import numpy as np
import pyasdm

# Time measure labels for absolute times produced by the ASDM backend (time
# coordinate, TIME_CENTROID, time_pointing, ...). Values are seconds since the
# Unix epoch, obtained with a fixed epoch shift from the ASDM ArrayTime values,
# and labelled "utc" like the MSv2 converter does (importasdm/MSv2 path), so that
# datasets opened from an ASDM and converted from an MSv2 agree.
ASDM_TIME_SCALE = "utc"
ASDM_TIME_FORMAT = "unix"

#: Seconds between the ASDM ArrayTime epoch (MJD 0, 1858-11-17T00:00:00) and the
#: Unix epoch (1970-01-01T00:00:00): 40587 days.
MJD_TO_UNIX_TIME_DELTA = 3_506_716_800
#: Same as MJD_TO_UNIX_TIME_DELTA, in (integer) nanoseconds
MJD_TO_UNIX_TIME_DELTA_NS = MJD_TO_UNIX_TIME_DELTA * 10**9


def _asdm_times_as_ns_array(times_asdm) -> np.ndarray:
    """ASDM time values (ArrayTime objects or numbers) as a numeric numpy array."""
    values = np.asarray(times_asdm)

    if values.dtype == object:
        ns_values = np.array(
            [
                value.get() if isinstance(value, pyasdm.types.ArrayTime) else value
                for value in values.ravel()
            ]
        ).reshape(values.shape)
        if ns_values.dtype == object:
            raise TypeError(
                "Cannot convert ASDM time values: not numbers in the int64 range "
                f"(types: {sorted({type(value).__name__ for value in values.ravel()})})"
            )
        values = ns_values

    return values


def convert_time_asdm_to_datetime64(times_asdm) -> np.ndarray:
    """
    Convert absolute ASDM time values (ArrayTime) to numpy datetime64[ns].

    The epoch difference is subtracted in integer nanoseconds, so no precision
    is lost (same time scale convention as :func:`convert_time_asdm_to_unix`).

    Parameters
    ----------
    times_asdm : array_like
        ASDM time values: pyasdm ArrayTime objects or integers (nanoseconds
        since the MJD epoch). A scalar or an array of any shape.

    Returns
    -------
    np.ndarray
        datetime64[ns] values, with the same shape as the input.

    Raises
    ------
    TypeError
        If the values are not ArrayTime objects or integers.
    """
    values = _asdm_times_as_ns_array(times_asdm)
    if values.size == 0:
        return np.zeros(values.shape, dtype="datetime64[ns]")
    if not np.issubdtype(values.dtype, np.integer):
        raise TypeError(
            f"Cannot convert ASDM time values of dtype {values.dtype} to datetime64 "
            "(integer nanoseconds or ArrayTime objects expected)"
        )
    return (values.astype(np.int64) - MJD_TO_UNIX_TIME_DELTA_NS).astype(
        "datetime64[ns]"
    )


def convert_time_asdm_to_unix(times_asdm) -> np.ndarray:
    """
    Convert absolute ASDM time values (ArrayTime) to seconds since the Unix epoch.

    ASDM ArrayTime values are integer nanoseconds since the MJD epoch
    (1858-11-17T00:00:00). They are converted as
    ``(ns - 3_506_716_800 * 10**9) / 1e9``, subtracting the epoch difference in
    integer nanoseconds before dividing, so that no precision is lost in the
    subtraction. Use this only for absolute times (time stamps), never for
    durations / intervals, which must not be shifted.

    The values produced are labelled with the time measure attributes
    ``format=ASDM_TIME_FORMAT`` ("unix") and ``scale=ASDM_TIME_SCALE`` ("utc").

    Parameters
    ----------
    times_asdm : array_like
        ASDM time values: pyasdm ArrayTime objects, integers (nanoseconds since the
        MJD epoch) or floats (nanoseconds since the MJD epoch, already rounded to
        float64 precision). A scalar or an array of any shape.

    Returns
    -------
    np.ndarray
        float64 seconds since the Unix epoch, with the same shape as the input.

    Raises
    ------
    TypeError
        If the values are not ArrayTime objects or numbers.

    Notes
    -----
    Time scale: the pyasdm ArrayTime docstring describes ArrayTime as TAI, but the
    CASA importasdm/MSv2 path writes these same values into the MS TIME column
    (UTC) with no leap-second correction, and the MSv2 converter labels them
    "utc". The ASDM backend follows that convention (``ASDM_TIME_SCALE``), so
    datasets opened from an ASDM and converted via MSv2 give the same instants.
    A constant epoch shift cannot convert between TAI and UTC (TAI-UTC is 37 s
    since 2017); if ASDM times were truly TAI, the instants would be 37 s late.
    """
    values = _asdm_times_as_ns_array(times_asdm)

    if values.size == 0:
        return np.zeros(values.shape, dtype=np.float64)

    if np.issubdtype(values.dtype, np.integer):
        times_unix = (values.astype(np.int64) - MJD_TO_UNIX_TIME_DELTA_NS) / 1e9
    elif np.issubdtype(values.dtype, np.floating):
        times_unix = (
            values.astype(np.float64) - float(MJD_TO_UNIX_TIME_DELTA_NS)
        ) / 1e9
    else:
        raise TypeError(f"Cannot convert ASDM time values of dtype {values.dtype}")

    return times_unix
