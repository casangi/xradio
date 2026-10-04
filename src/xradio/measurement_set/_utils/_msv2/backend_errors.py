"""
Errors of the MSv2 xarray backend (engine ``xradio_msv2``).

They are re-exported by :mod:`xradio.measurement_set.open_msv2`.
"""


class MSv2ChangedError(RuntimeError):
    """
    The MeasurementSet changed since it was opened: the MAIN table has another
    number of rows, or the rows of a partition give another (time, baseline)
    grid. The lazy arrays of the opened processing set no longer describe
    the MS; open it again.
    """


class MSv2ReadError(RuntimeError):
    """
    Reading the values of a lazy data variable of an MSv2 opened with the
    ``xradio_msv2`` engine failed. The message names the MAIN column, the data
    variable, the MSv4 node and the block that was read.
    """
