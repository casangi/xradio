"""
Open a MeasurementSet v2 directly as a processing set, without converting it:
the ``xradio_msv2`` xarray engine.
"""

import xarray as xr

from _xradio_xarray_backends import MSv2BackendEntrypoint
from xradio.measurement_set._utils._msv2.backend_errors import (
    MSv2ChangedError,
    MSv2ReadError,
)

__all__ = [
    "open_msv2",
    "MSv2BackendEntrypoint",
    "MSv2ChangedError",
    "MSv2ReadError",
]


def open_msv2(
    ms_path: str,
    scan_intents: list | None = None,
    array_backend: str = "dask",
    **engine_kwargs,
) -> xr.DataTree:
    """
    Open a MeasurementSet v2 as a lazy processing set: the DataTree of MSv4s
    that :func:`convert_msv2_to_processing_set` would write, opened as
    :func:`open_processing_set` opens it, without converting the MS. The
    metadata and sub-datasets are read when the MS is opened; the main data
    variables (VISIBILITY*, SPECTRUM*, FLAG, WEIGHT, UVW, TIME_CENTROID,
    EFFECTIVE_INTEGRATION_TIME) are read from the MS when they are indexed or
    computed.

    It is ``xarray.open_datatree(ms_path, engine="xradio_msv2", ...)`` (with
    the engine class, so that it works without installed entry points), with
    the conveniences of :func:`open_processing_set`.

    Parameters
    ----------
    ms_path : str
        Path of the MeasurementSet v2.
    scan_intents : list | None, optional
        Keep only the MSv4s with one of these scan intents (as
        ``open_processing_set``), by default None: all.
    array_backend : str, optional
        "dask" (default): every variable is a dask array (``chunks={}``; the
        main data variables have the chunks of the converted processing set).
        "xarray": lazily indexed arrays (``chunks=None``).
    **engine_kwargs
        Options of the engine, as for ``convert_msv2_to_processing_set``:

        - partition_scheme : list | None. Keys (besides the data description,
          observing mode, observation and ephemeris that every partition has)
          among "FIELD_ID", "SCAN_NUMBER", "STATE_ID", "SOURCE_ID",
          "SUB_SCAN_NUMBER", "ANTENNA1". By default [].
        - partition_filter : Callable[[dict], bool] | None. Open only the
          partitions whose description (a copy) it selects; the MSv4s are
          numbered as by the converter.
        - main_chunksize, with_pointing, pointing_chunksize,
          pointing_interpolate, ephemeris_interpolate, phase_cal_interpolate,
          sys_cal_interpolate: as for the converter. main_chunksize sets the
          chunks of the main data variables (their encoding, and the dask
          chunks with array_backend="dask").
        - drop_variables : str | Iterable[str] | None. Variables left out of
          every node of every MSv4 (and of its data groups).
        - partition_cache : str | None. "auto", "read", "off" or "rebuild";
          by default the environment variable XRADIO_MSV2_PARTITION_CACHE, or
          "auto". The partitions of an MS are kept in memory (per process)
          while the MS is unchanged; "off" computes them on every open.
        - on_partition_error : str. "skip" (default): a partition that cannot
          be opened is left out (logged with its traceback, and a
          RuntimeWarning); a RuntimeError is raised only if none can be
          opened. "raise": raise a RuntimeError for the first one.

    Returns
    -------
    xr.DataTree
        The processing set.

    Raises
    ------
    MSv2ChangedError
        From a lazy read, if the MS changed since it was opened.
    MSv2ReadError
        From a lazy read that failed.
    """
    if array_backend == "xarray":
        chunks, chunked_array_type = None, None
    elif array_backend == "dask":
        chunks, chunked_array_type = {}, "dask"
    else:
        raise ValueError("array_backend must be either 'dask' or 'xarray'")
    ps_xdt = xr.open_datatree(
        ms_path,
        engine=MSv2BackendEntrypoint,
        chunks=chunks,
        chunked_array_type=chunked_array_type,
        **engine_kwargs,
    )
    if scan_intents is None:
        return ps_xdt
    return ps_xdt.xr_ps.query(scan_intents=scan_intents)
