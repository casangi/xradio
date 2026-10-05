"""
Errors of the MSv2 xarray backend (engine ``xradio_msv2``).

MSv2ChangedError, MSv2ReadError and PartitionCacheWarning are exported by
:mod:`xradio.measurement_set` (and :mod:`xradio.measurement_set.open_msv2`).
"""


class MSv2ChangedError(RuntimeError):
    """
    The MeasurementSet changed since it was opened: the MAIN table has another
    number of rows, or the rows of a partition give another (time, baseline)
    grid; or the POINTING table has another number of rows, or other rows,
    times or antennas for a partition. The lazy arrays of the opened
    processing set no longer describe the MS; open it again.
    """


class MSv2ReadError(RuntimeError):
    """
    Reading the values of a lazy data variable of an MSv2 opened with the
    ``xradio_msv2`` engine failed. The message names the MAIN (or POINTING)
    column, the data variable, the MSv4 node and the block that was read.
    """


class PartitionCacheWarning(UserWarning):
    """
    The partitions of an MSv2 opened with the ``xradio_msv2`` engine could
    not be stored in the MS (its ``XRADIO_PARTITIONS`` sub-table): they are
    computed in memory, on every open of the MS in a new process. Given once
    per MS and reason in a process; ``partition_cache="read"`` or ``"off"``
    silences it.
    """


class StalePartitionsError(RuntimeError):
    """
    Partitions taken from the partition cache (the XRADIO_PARTITIONS row or
    the per-process memo) do not describe the MAIN rows they point to: the
    MS changed in a way their staleness checks missed, or while it was
    opened. The engine computes the partitions again (internal: not raised
    to users).
    """


class MainRowsChangedError(StalePartitionsError):
    """
    The MAIN table has another number of rows than when the partitions were
    computed: another process added or removed rows while the MS was opened
    (not a defect of the partition cache). The engine computes the
    partitions again (internal: not raised to users).
    """
