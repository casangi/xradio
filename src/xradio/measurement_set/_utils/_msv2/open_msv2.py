"""
Open a MeasurementSet v2 directly as a processing set, without converting it:
the ``xradio_msv2`` xarray engine. The public functions and classes are
exported by :mod:`xradio.measurement_set` (this module is private, as the
ASDM backend's ``open_asdm`` module, so that ``xradio.measurement_set.open_msv2``
is only the function).
"""

import os

import xarray as xr

from _xradio_xarray_backends import MSv2BackendEntrypoint
from xradio.measurement_set._utils._msv2.backend_errors import (
    MSv2ChangedError,
    MSv2ReadError,
    PartitionCacheWarning,
)

__all__ = [
    "open_msv2",
    "remove_msv2_partition_cache",
    "MSv2ChangedError",
    "MSv2ReadError",
    "PartitionCacheWarning",
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
    EFFECTIVE_INTEGRATION_TIME) and the data variables of the pointing_xds
    (POINTING_BEAM, POINTING_DISH_MEASURED, POINTING_OVER_THE_TOP) are read
    from the MS when they are indexed or computed (a selection reads only its
    rows). For the pointing_xds, the POINTING TIME and ANTENNA_ID columns are
    read at open (its time and antenna coordinates), the shapes of the cells
    of its array columns (stored in the MS by the first open, see
    partition_cache below), and one cell of each data column. Only with
    pointing_interpolate=True is it built at open, as by the converter. A
    partition whose POINTING rows have cells of several shapes or without a
    value gets the pointing_xds of the converter (which reads the rows of
    every partition on its own, and leaves out or pads such a column) from
    the shapes of its cells: its variables are read lazily (1,000 POINTING
    rows or more) or built by the converter's code when read (fewer). Every
    partition of a POINTING table that cannot be described without reading
    it (no DIRECTION column, unusual value types) has its pointing_xds built
    at open by the converter's code and again when it is read. An MS without
    MAIN rows opens as an empty processing set.

    The first open of a writable MS stores its partitions, and the shapes of
    the cells of its POINTING table, in the MS (see partition_cache below).
    See the API documentation for the costs of an open, the chunks to use
    and the changes of an MS after its open.

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
        - partition_cache : str | None. By default the environment variable
          XRADIO_MSV2_PARTITION_CACHE, or "auto". The partitions of an MS
          (``create_partitions``) are computed on its first open and stored
          inside the MS: a sub-table XRADIO_PARTITIONS, linked by a MAIN
          keyword of that name, and a HISTORY row. Later opens use them while
          the MS is unchanged (a fingerprint of the tables they are computed
          from, and no newer HISTORY rows). "auto": use, compute and store
          (an MS that cannot be written is opened with partitions computed in
          memory, with a PartitionCacheWarning once per MS and reason);
          "read": use, never write; "rebuild": compute again and store; "off":
          compute, neither use nor store. Partitions are also kept in memory
          (per process) while the MS is unchanged (not with "off", nor for an
          MS whose MAIN rows live in other tables: a reference or
          concatenated MS). The shapes of the cells of the POINTING array
          columns, which the first open scans, are stored and used the same
          way (a row of XRADIO_PARTITIONS, no HISTORY row), while the
          POINTING table is unchanged (its fingerprint).
        - on_partition_error : str. "skip" (default): a partition that cannot
          be opened is left out (logged with its traceback, and a
          RuntimeWarning); a RuntimeError is raised only if none can be
          opened. "raise": raise a RuntimeError for the first one.
        - skip_columns : str | Iterable[str] | None. MAIN columns to treat as
          unreadable, as the converter treats a column whose read fails: it
          is left out, with the data group (and field_and_source_xds) it
          makes; for WEIGHT_SPECTRUM, WEIGHT is read from the WEIGHT column.
          Some columns can only be checked by reading them (e.g. cells of
          several shapes in a StandardStMan column): their read then raises
          an MSv2ReadError that names this remedy.

    Returns
    -------
    xr.DataTree
        The processing set.

    Raises
    ------
    MSv2ChangedError
        From a lazy read, if the MS changed since it was opened (MAIN or
        POINTING rows added or removed, rows moved to or from another
        partition, or the time and baseline grid of a partition changed).
    MSv2ReadError
        From a lazy read that failed (see skip_columns for columns whose
        cells only a read can check).
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


def remove_msv2_partition_cache(ms_path: str | os.PathLike) -> bool:
    """
    Remove the partitions that the ``xradio_msv2`` engine stored in a
    MeasurementSet v2 (the MAIN keyword XRADIO_PARTITIONS, then the
    sub-table XRADIO_PARTITIONS). Its HISTORY rows are kept. A sub-table
    XRADIO_PARTITIONS that xradio did not write is left alone.

    Parameters
    ----------
    ms_path : str | os.PathLike
        Path of the MeasurementSet v2.

    Returns
    -------
    bool
        Whether anything was removed.

    Raises
    ------
    FileNotFoundError
        If there is no MeasurementSet at ``ms_path``.
    PermissionError
        If the MeasurementSet cannot be written.
    ValueError
        If its XRADIO_PARTITIONS is not a partition cache of xradio (nothing
        is removed).
    RuntimeError
        If another process holds a lock on its MAIN table or on the
        sub-table (nothing is removed).
    """
    from xradio.measurement_set._utils._msv2.partition_cache import (
        remove_partition_cache,
    )

    return remove_partition_cache(ms_path)
