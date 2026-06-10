"""Lazy MSv2 backend: open an MSv2 as MSv4-schema xarray DataTree(s) backed by dask.

This module provides :func:`open_msv2` which reads metadata eagerly but
defers the bulk data column reads (VISIBILITY, FLAG, WEIGHT, UVW ...) as
lazy dask arrays via casacore.  The result is a dict-like
:class:`~xradio.measurement_set.ProcessingSetXdt` that matches the schema
produced by :func:`convert_msv2_to_processing_set`, but without any
intermediate Zarr write.
"""

import datetime
import importlib
import pathlib
import time
import threading

import numpy as np
import toolviper.utils.logger as logger
import xarray as xr

from xradio._utils.list_and_array import check_if_consistent, unique_1d
from xradio.measurement_set.schema import MSV4_SCHEMA_VERSION
from xradio.measurement_set._utils._msv2.partition_queries import create_partitions
from xradio.measurement_set._utils._msv2.conversion import (
    add_data_groups,
    add_missing_data_var_attrs,
    antenna_ids_to_names,
    calc_indx_for_row_split,
    create_coordinates,
    create_data_variables,
    create_taql_query_where,
    find_min_max_times,
    fix_uvw_frame,
    parse_chunksize,
)
from xradio.measurement_set._utils._msv2._tables.table_query import TableManager
from xradio.measurement_set._utils._msv2._tables.read import load_generic_table
from xradio.measurement_set._utils._msv2.create_antenna_xds import (
    create_antenna_xds,
    create_gain_curve_xds,
    create_phase_calibration_xds,
)
from xradio.measurement_set._utils._msv2.create_field_and_source_xds import (
    create_field_and_source_xds,
)
from xradio.measurement_set._utils._msv2.msv4_sub_xdss import (
    create_pointing_xds,
    create_system_calibration_xds,
    create_weather_xds,
    create_phased_array_xds,
)
from xradio.measurement_set._utils._msv2.msv4_info_dicts import create_info_dicts

# ---------------------------------------------------------------------------
# Partition cache — avoid re-scanning the MS for repeated open_msv2 calls.
# ---------------------------------------------------------------------------
_PARTITION_CACHE_TTL = 300  # seconds
_partition_cache: dict[tuple, tuple[float, list]] = {}
_partition_cache_lock = threading.Lock()


def _get_partitions_cached(in_file: str, partition_scheme: list) -> list[dict]:
    """Return cached partitions for *in_file* or compute them fresh."""
    key = (str(pathlib.Path(in_file).resolve()), tuple(partition_scheme))
    now = time.monotonic()
    with _partition_cache_lock:
        if key in _partition_cache:
            ts, partitions = _partition_cache[key]
            if now - ts < _PARTITION_CACHE_TTL:
                return partitions
            del _partition_cache[key]
    partitions = create_partitions(in_file, partition_scheme=partition_scheme)
    with _partition_cache_lock:
        _partition_cache[key] = (time.monotonic(), partitions)
    return partitions


def _build_partition_lazy(
    in_file: str,
    partition_info: dict,
    partition_scheme: list,
    main_chunksize: dict | float | None,
    with_pointing: bool,
    pointing_interpolate: bool,
    ephemeris_interpolate: bool,
    phase_cal_interpolate: bool,
    sys_cal_interpolate: bool,
) -> xr.DataTree | None:
    """Build a single MSv4 partition as a lazily-loaded DataTree.

    All bulk data variables are backed by dask arrays that read from the
    MSv2 on demand via casacore.
    """
    ms_xdt = xr.DataTree()

    taql_where = create_taql_query_where(partition_info)
    table_manager = TableManager(in_file, taql_where)
    ddi = partition_info["DATA_DESC_ID"][0]
    scan_intents = str(partition_info["OBS_MODE"][0]).split(",")

    with table_manager.get_table() as tb_tool:
        if tb_tool.nrows() == 0:
            return None

        (
            tidxs,
            bidxs,
            didxs,
            baseline_ant1_id,
            baseline_ant2_id,
            utime,
        ) = calc_indx_for_row_split(tb_tool, taql_where)
        time_baseline_shape = (len(utime), len(baseline_ant1_id))

        observation_id = check_if_consistent(
            tb_tool.getcol("OBSERVATION_ID"), "OBSERVATION_ID"
        )

        try:
            generic_observation_xds = load_generic_table(
                in_file,
                "OBSERVATION",
                taql_where=f" where (ROWID() IN [{str(observation_id)}])",
            )
            telescope_name = generic_observation_xds["TELESCOPE_NAME"].values[0]
        except (IndexError, KeyError, ValueError) as exc:
            logger.warning(
                "Could not read OBSERVATION subtable row "
                + str(observation_id)
                + ": "
                + str(exc)
                + ". Using UNKNOWN telescope name."
            )
            telescope_name = "UNKNOWN"

        xds = xr.Dataset(
            attrs={
                "schema_version": MSV4_SCHEMA_VERSION,
                "creator": {
                    "software_name": "xradio",
                    "version": importlib.metadata.version("xradio"),
                },
                "creation_date": (
                    datetime.datetime.now(datetime.timezone.utc).isoformat()
                ),
                "type": "visibility",
            }
        )

        interval = tb_tool.getcol("INTERVAL")
        interval_unique = unique_1d(interval)
        if len(interval_unique) > 1:
            interval = np.median(interval)
        else:
            interval = interval_unique[0]

        scan_id = np.full(time_baseline_shape, -42, dtype=int)
        scan_id[tidxs, bidxs] = tb_tool.getcol("SCAN_NUMBER")
        scan_id = np.max(scan_id, axis=1)

        xds, spectral_window_id = create_coordinates(
            xds,
            in_file,
            ddi,
            utime,
            interval,
            baseline_ant1_id,
            baseline_ant2_id,
            scan_id,
            scan_intents,
        )

        # Force dask-backed lazy reads by setting parallel_mode='time'.
        # If no time chunksize provided, default to the full time axis
        # (single chunk — still lazy but no fragmentation).
        main_chunksize = parse_chunksize(main_chunksize, "main", xds)
        if main_chunksize is None:
            main_chunksize = {"time": time_baseline_shape[0]}

        if "time" not in main_chunksize:
            main_chunksize["time"] = time_baseline_shape[0]

        # read_col_conversion_dask assumes dense data (every time has
        # every baseline). Use sparse-aware dask reader for sparse data.
        total_rows = tb_tool.nrows()
        is_dense = total_rows == time_baseline_shape[0] * time_baseline_shape[1]
        parallel_mode = "time" if is_dense else "sparse"

        create_data_variables(
            in_file,
            xds,
            table_manager,
            time_baseline_shape,
            tidxs,
            bidxs,
            didxs,
            use_table_iter=False,
            parallel_mode=parallel_mode,
            main_chunksize=main_chunksize,
        )

        xds, is_single_dish = add_data_groups(xds)
        xds = add_missing_data_var_attrs(xds)

        if "WEIGHT" not in xds.data_vars:
            if is_single_dish:
                xds["WEIGHT"] = xr.DataArray(
                    np.ones(xds.SPECTRUM.shape, dtype=np.float64),
                    dims=xds.SPECTRUM.dims,
                )
            else:
                xds["WEIGHT"] = xr.DataArray(
                    np.ones(xds.VISIBILITY.shape, dtype=np.float64),
                    dims=xds.VISIBILITY.dims,
                )

        time_min_max = find_min_max_times(tb_tool, taql_where)

        # --- Secondary sub-datasets (eagerly loaded, they are small) ---
        feed_id = unique_1d(
            np.concatenate(
                [
                    unique_1d(tb_tool.getcol("FEED1")),
                    unique_1d(tb_tool.getcol("FEED2")),
                ]
            )
        )
        antenna_id = unique_1d(
            np.concatenate(
                [
                    xds["baseline_antenna1_id"].data,
                    xds["baseline_antenna2_id"].data,
                ]
            )
        )

        ant_xds = create_antenna_xds(
            in_file,
            spectral_window_id,
            antenna_id,
            feed_id,
            telescope_name,
            xds.polarization,
        )

        gain_curve_xds = create_gain_curve_xds(in_file, spectral_window_id, ant_xds)

        phase_cal_interp_time = xds.time.values if phase_cal_interpolate else None
        try:
            phase_calibration_xds = create_phase_calibration_xds(
                in_file,
                spectral_window_id,
                ant_xds,
                time_min_max,
                phase_cal_interp_time,
            )
        except (AssertionError, AttributeError, KeyError):
            phase_calibration_xds = None

        sys_cal_interp_time = xds.time.values if sys_cal_interpolate else None
        system_calibration_xds = create_system_calibration_xds(
            in_file,
            spectral_window_id,
            xds.frequency,
            ant_xds,
            sys_cal_interp_time,
        )

        with_antenna_partitioning = "ANTENNA1" in partition_info
        xds = antenna_ids_to_names(
            xds,
            ant_xds,
            is_single_dish,
            with_antenna_partitioning,
        )
        ant_xds_name_ids = ant_xds["antenna_name"].set_xindex("antenna_id")
        ant_position_xds_with_ids = ant_xds["ANTENNA_POSITION"].set_xindex("antenna_id")
        ant_xds = ant_xds.drop_vars("antenna_id")

        weather_xds = create_weather_xds(in_file, ant_position_xds_with_ids)

        pointing_xds = xr.Dataset()
        if with_pointing:
            pointing_interp_time = xds.time if pointing_interpolate else None
            pointing_xds = create_pointing_xds(
                in_file,
                ant_xds_name_ids,
                time_min_max,
                pointing_interp_time,
            )

        phased_array_xds = create_phased_array_xds(
            in_file,
            ant_xds.antenna_name,
            ant_xds.receptor_label,
            ant_xds.polarization_type,
        )

        # Ensure frequency and time are increasing
        if len(xds.frequency) > 1 and xds.frequency[1] - xds.frequency[0] < 0:
            xds = xds.sel(frequency=slice(None, None, -1))
        if len(xds.time) > 1 and xds.time[1] - xds.time[0] < 0:
            xds = xds.sel(time=slice(None, None, -1))

        # Field and source
        field_id = np.full(time_baseline_shape, -42, dtype=int)
        field_id[tidxs, bidxs] = tb_tool.getcol("FIELD_ID")
        field_id = np.max(field_id, axis=1)
        field_times = xds.time.values

        try:
            field_and_source_xds, source_id, _num_lines, field_names = (
                create_field_and_source_xds(
                    in_file,
                    field_id,
                    spectral_window_id,
                    field_times,
                    is_single_dish,
                    time_min_max,
                    ephemeris_interpolate,
                )
            )
        except (AssertionError, IndexError, KeyError, ValueError) as exc:
            logger.warning(
                "Could not build field_and_source sub-dataset: "
                + str(exc)
                + ". Creating minimal placeholder."
            )
            n_fields = len(unique_1d(field_id))
            field_names = ["UNKNOWN"] * len(field_times)
            field_and_source_xds = xr.Dataset(attrs={"type": "field_and_source"})

        xds = fix_uvw_frame(xds, field_and_source_xds, is_single_dish)
        xds = xds.assign_coords({"field_name": ("time", field_names)})

        partition_info_misc_fields = {
            "scan_name": xds.coords["scan_name"].data,
            "taql_where": taql_where,
        }
        if with_antenna_partitioning:
            partition_info_misc_fields["antenna_name"] = xds.coords[
                "antenna_name"
            ].data[0]
        info_dicts = create_info_dicts(
            in_file,
            xds,
            field_and_source_xds,
            partition_info_misc_fields,
            tb_tool,
        )
        xds.attrs.update(info_dicts)

        if is_single_dish:
            xds.attrs["type"] = "spectrum"
            xds = xds.drop_vars("UVW")
            xds = xds.drop_dims("uvw_label")
        else:
            if xds.attrs["processor_info"]["type"] == "RADIOMETER":
                xds.attrs["type"] = "radiometer"
            else:
                xds.attrs["type"] = "visibility"

        # Assemble DataTree
        ms_xdt.ds = xds
        ms_xdt["/antenna_xds"] = ant_xds
        for group_name in xds.attrs["data_groups"]:
            ms_xdt[f"/field_and_source_{group_name}_xds"] = field_and_source_xds

        if with_pointing and len(pointing_xds.data_vars) > 0:
            ms_xdt["/pointing_xds"] = pointing_xds
        if system_calibration_xds:
            ms_xdt["/system_calibration_xds"] = system_calibration_xds
        if gain_curve_xds:
            ms_xdt["/gain_curve_xds"] = gain_curve_xds
        if phase_calibration_xds:
            ms_xdt["/phase_calibration_xds"] = phase_calibration_xds
        if weather_xds:
            ms_xdt["/weather_xds"] = weather_xds
        if phased_array_xds:
            ms_xdt["/phased_array_xds"] = phased_array_xds

    return ms_xdt


def open_msv2(
    in_file: str,
    partition_scheme: list | None = None,
    main_chunksize: dict | float | None = None,
    with_pointing: bool = True,
    pointing_interpolate: bool = False,
    ephemeris_interpolate: bool = False,
    phase_cal_interpolate: bool = False,
    sys_cal_interpolate: bool = False,
) -> xr.DataTree:
    """Open an MSv2 as a lazy MSv4-schema Processing Set.

    This function reads an MSv2 and returns a :class:`xarray.DataTree` that
    matches the schema produced by :func:`convert_msv2_to_processing_set`,
    but *without* writing to Zarr.  Bulk data variables (VISIBILITY, FLAG,
    WEIGHT, UVW, etc.) are backed by dask arrays that read from the MSv2
    on demand via casacore.

    Parameters
    ----------
    in_file
        Path to an MSv2 on disk.
    partition_scheme
        Partitioning keys (same as in :func:`convert_msv2_to_processing_set`).
        By default ``[]``.
    main_chunksize
        Chunk sizes for the main dataset.  If a dict, keys are dimension
        names (``time``, ``baseline_id``, ``frequency``, ``polarization``).
        If a float, gives the chunk size in GiB.  ``None`` (default) uses a
        single chunk per partition.
    with_pointing
        Whether to include the pointing sub-dataset.
    pointing_interpolate
        Whether to interpolate pointing to the main time axis.
    ephemeris_interpolate
        Whether to interpolate ephemeris to the main time axis.
    phase_cal_interpolate
        Whether to interpolate phase calibration to the main time axis.
    sys_cal_interpolate
        Whether to interpolate system calibration to the main time axis.

    Returns
    -------
    xr.DataTree
        A processing-set DataTree with one child per MSv4 partition.
        Data variables are lazy dask arrays backed by casacore reads.
    """
    if partition_scheme is None:
        partition_scheme = []

    partitions = _get_partitions_cached(in_file, partition_scheme)
    logger.info("Number of partitions: " + str(len(partitions)))

    ps_dt = xr.DataTree()
    ps_dt.attrs["type"] = "processing_set"

    for ms_v4_id, partition_info in enumerate(partitions):
        ms_v4_id_str = f"{ms_v4_id:0>{len(str(len(partitions) - 1))}}"

        logger.info(
            "OBSERVATION_ID "
            + str(partition_info["OBSERVATION_ID"])
            + ", DDI "
            + str(partition_info["DATA_DESC_ID"])
            + ", STATE "
            + str(partition_info["STATE_ID"])
            + ", FIELD "
            + str(partition_info["FIELD_ID"])
            + ", SCAN "
            + str(partition_info["SCAN_NUMBER"])
            + (
                ", EPHEMERIS " + str(partition_info["EPHEMERIS_ID"])
                if "EPHEMERIS_ID" in partition_info
                else ""
            )
            + (
                ", ANTENNA " + str(partition_info["ANTENNA1"])
                if "ANTENNA1" in partition_info
                else ""
            )
        )

        start = time.time()
        ms_v4_name = pathlib.Path(in_file).name.replace(".ms", "") + "_" + ms_v4_id_str

        ms_xdt = _build_partition_lazy(
            in_file,
            partition_info,
            partition_scheme,
            main_chunksize,
            with_pointing,
            pointing_interpolate,
            ephemeris_interpolate,
            phase_cal_interpolate,
            sys_cal_interpolate,
        )

        if ms_xdt is not None:
            ps_dt[ms_v4_name] = ms_xdt
            logger.debug(
                "Time to build lazy partition "
                + ms_v4_id_str
                + ": "
                + str(time.time() - start)
                + "s"
            )

    return ps_dt
