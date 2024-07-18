import numcodecs
import math
import time
from .._zarr.encoding import add_encoding
from typing import Dict, Union
import graphviper.utils.logger as logger
import os

import numpy as np
import xarray as xr

from casacore import tables
from .msv4_sub_xdss import create_ant_xds, create_pointing_xds, create_weather_xds
from xradio.vis._vis_utils._ms._tables.create_field_and_source_xds import (
    create_field_and_source_xds,
)
from .msv2_to_msv4_meta import (
    column_description_casacore_to_msv4_measure,
    create_attribute_metadata,
    col_to_data_variable_names,
    col_dims,
)

from .subtables import subt_rename_ids
from ._tables.table_query import open_table_ro, open_query
from ._tables.read import (
    convert_casacore_time,
    extract_table_attributes,
    read_col_conversion,
    read_generic_table,
)
from ._tables.read_main_table import get_baselines, get_baseline_indices, get_utimes_tol
from .._utils.stokes_types import stokes_types
from xradio._utils.list_and_array import check_if_consistent, unique_1d, to_list


def parse_chunksize(
    chunksize: Union[Dict, float, None], xds_type: str, xds: xr.Dataset
) -> Dict[str, int]:
    """
    Parameters
    ----------
    chunksize : Union[Dict, float, None]
        Desired maximum size of the chunks, either as a dict of per-dimension sizes or as
        an amount of memory
    xds_type : str
        whether to use chunking logic for main or pointing datasets
    xds : xr.Dataset
        dataset to calculate best chunking

    Returns
    -------
    Dict[str, int]
        dictionary of chunk sizes (as dim->size)
    """
    if isinstance(chunksize, dict):
        check_chunksize(chunksize, xds_type)
    elif isinstance(chunksize, float):
        chunksize = mem_chunksize_to_dict(chunksize, xds_type, xds)
    elif chunksize is not None:
        raise ValueError(
            f"Chunk size expected as a dict or a float, got: "
            f" {chunksize} (of type {type(chunksize)}"
        )

    return chunksize


def check_chunksize(chunksize: dict, xds_type: str) -> None:
    """
    Rudimentary check of the chunksize parameters to catch obvious errors early before
    more work is done.
    """
    # perphaps start using some TypeDict or/and validator like pydantic?
    if xds_type == "main":
        allowed_dims = [
            "time",
            "baseline_id",
            "antenna_id",
            "frequency",
            "polarization",
        ]
    elif xds_type == "pointing":
        allowed_dims = ["time", "antenna"]

    msg = ""
    for dim in chunksize.keys():
        if dim not in allowed_dims:
            msg += f"dimension {dim} not allowed in {xds_type} dataset:\n"
    if msg:
        raise ValueError(f"Wrong keys found in chunksize: {msg}")


def mem_chunksize_to_dict(
    chunksize: float, xds_type: str, xds: xr.Dataset
) -> Dict[str, int]:
    """
    Given a desired 'chunksize' as amount of memory in GB, calculate best chunk sizes
    for every dimension of an xds.

    Parameters
    ----------
    chunksize : float
        Desired maximum size of the chunks
    xds_type : str
        whether to use chunking logic for main or pointing datasets
    xds : xr.Dataset
        dataset to auto-calculate chunking of its dimensions

    Returns
    -------
    Dict[str, int]
        dictionary of chunk sizes (as dim->size)
    """

    if xds_type == "pointing":
        sizes = mem_chunksize_to_dict_pointing(chunksize, xds)
    elif xds_type == "main":
        sizes = mem_chunksize_to_dict_main(chunksize, xds)
    else:
        raise RuntimeError(f"Unexpected type: {xds_type=}")

    return sizes


GiBYTES_TO_BYTES = 1024 * 1024 * 1024


def mem_chunksize_to_dict_main(chunksize: float, xds: xr.Dataset) -> Dict[str, int]:
    """
    Checks the assumption that all polarizations can be held in memory, at least for one
    data point (one time, one freq, one channel).

    It presently relies on the logic of mem_chunksize_to_dict_main_balanced() to find a
    balanced list of dimension sizes for the chunks

    Assumes these relevant dims: (time, antenna_id/baseline_id, frequency,
    polarization).
    """

    sizeof_vis = itemsize_vis_spec(xds)
    size_all_pols = sizeof_vis * xds.sizes["polarization"]
    if size_all_pols / GiBYTES_TO_BYTES > chunksize:
        raise RuntimeError(
            "Cannot calculate chunk sizes when memory bound ({chunksize}) does not even allow all polarizations in one chunk"
        )

    baseline_or_antenna_id = find_baseline_or_antenna_var(xds)
    total_size = calc_used_gb(xds.sizes, baseline_or_antenna_id, sizeof_vis)

    ratio = chunksize / total_size
    chunked_dims = ["time", baseline_or_antenna_id, "frequency", "polarization"]
    if ratio >= 1:
        result = {dim: xds.sizes[dim] for dim in chunked_dims}
        logger.debug(
            f"{chunksize=} GiB is enough to fully hold {total_size=} GiB (for {xds.sizes=}) in memory in one chunk"
        )
    else:
        xds_dim_sizes = {k: xds.sizes[k] for k in chunked_dims}
        result = mem_chunksize_to_dict_main_balanced(
            chunksize, xds_dim_sizes, baseline_or_antenna_id, sizeof_vis
        )

    return result


def mem_chunksize_to_dict_main_balanced(
    chunksize: float, xds_dim_sizes: dict, baseline_or_antenna_id: str, sizeof_vis: int
) -> Dict[str, int]:
    """
    Assumes the ratio is <1 and all pols can fit in memory (from
    mem_chunksize_to_dict_main()).

    What is kept balanced is the fraction of the total size of every dimension included in a
    chunk. For example, time: 10, baseline: 100, freq: 1000, if we can afford about 33% in
    one chunk, the chunksize will be ~ time: 3, baseline: 33, freq: 333.
    The polarization axis is excluded from the calculations.
    Because this can leave a leftover (below or above the desired chunksize limit) and
    adjustment is done to get the final memory use below but as close as possible to
    'chunksize'. This adjustment alters the balance.

    Parameters
    ----------
    chunksize : float
        Desired maximum size of the chunks
    xds_dim_sizes : dict
        Dataset dimension sizes as dim_name->size
    sizeof_vis : int
        Size in bytes of a data point (one visibility / spectrum value)

    Returns
    -------
    Dict[str, int]
        dictionary of chunk sizes (as dim->size)
    """

    dim_names = [name for name in xds_dim_sizes.keys()]
    dim_sizes = [size for size in xds_dim_sizes.values()]
    # Fix fourth dim (polarization) to all (not free to auto-calculate)
    free_dims_mask = np.array([True, True, True, False])

    total_size = np.prod(dim_sizes) * sizeof_vis / GiBYTES_TO_BYTES
    ratio = chunksize / total_size

    dim_chunksizes = np.array(dim_sizes, dtype="int64")
    factor = ratio ** (1 / np.sum(free_dims_mask))
    dim_chunksizes[free_dims_mask] = np.maximum(
        dim_chunksizes[free_dims_mask] * factor, 1
    )
    used = np.prod(dim_chunksizes) * sizeof_vis / GiBYTES_TO_BYTES

    logger.debug(
        f"Auto-calculating main chunk sizes. First order approximation {dim_chunksizes=}, used total: {used} GiB (with {chunksize=} GiB)"
    )

    # Iterate through the dims, starting from the dims with lower chunk size
    #  (=bigger impact of a +1)
    # Note the use of math.floor, this iteration can either increase or decrease sizes,
    #  if increasing sizes we want to keep mem use below the upper limit, floor(2.3) = +2
    #  if decreasing sizes we want to take mem use below the upper limit, floor(-2.3) = -3
    indices = np.argsort(dim_chunksizes[free_dims_mask])
    for idx in indices:
        left = chunksize - used
        other_dims_mask = np.ones(free_dims_mask.shape, dtype=bool)
        other_dims_mask[idx] = False
        delta = np.divide(
            left,
            np.prod(dim_chunksizes[other_dims_mask]) * sizeof_vis / GiBYTES_TO_BYTES,
        )
        int_delta = np.floor(delta)
        if abs(int_delta) > 0 and int_delta + dim_chunksizes[idx] > 0:
            dim_chunksizes[idx] += int_delta
        used = np.prod(dim_chunksizes) * sizeof_vis / GiBYTES_TO_BYTES

    chunked_dim_names = ["time", baseline_or_antenna_id, "frequency", "polarization"]
    dim_chunksizes_int = [int(v) for v in dim_chunksizes]
    result = dict(zip(chunked_dim_names, dim_chunksizes_int))

    logger.debug(
        f"Auto-calculated main chunk sizes with {chunksize=}, {total_size=} GiB (for {dim_sizes=}): {result=} which uses {used} GiB."
    )

    return result


def mem_chunksize_to_dict_pointing(chunksize: float, xds: xr.Dataset) -> Dict[str, int]:
    """
    Equivalent to mem_chunksize_to_dict_main adapted to pointing xdss.
    Assumes these relevant dims: (time, antenna, direction).
    """

    if not xds.sizes:
        return {}

    sizeof_pointing = itemsize_pointing_spec(xds)
    chunked_dim_names = [name for name in xds.sizes.keys()]
    dim_sizes = [size for size in xds.sizes.values()]
    total_size = np.prod(dim_sizes) * sizeof_pointing / GiBYTES_TO_BYTES

    # Fix third dim (direction) to all
    free_dims_mask = np.array([True, True, False])

    ratio = chunksize / total_size
    if ratio >= 1:
        logger.debug(
            f"Pointing chunsize: {chunksize=} GiB is enough to fully hold {total_size=} GiB (for {xds.sizes=}) in memory in one chunk"
        )
        dim_chunksizes = dim_sizes
    else:
        # balanced
        dim_chunksizes = np.array(dim_sizes, dtype="int")
        factor = ratio ** (1 / np.sum(free_dims_mask))
        dim_chunksizes[free_dims_mask] = np.maximum(
            dim_chunksizes[free_dims_mask] * factor, 1
        )
        used = np.prod(dim_chunksizes) * sizeof_pointing / GiBYTES_TO_BYTES

        logger.debug(
            f"Auto-calculating pointing chunk sizes. First order approximation: {dim_chunksizes=}, used total: {used=} GiB (with {chunksize=} GiB"
        )

        indices = np.argsort(dim_chunksizes[free_dims_mask])
        # refine dim_chunksizes
        for idx in indices:
            left = chunksize - used
            other_dims_mask = np.ones(free_dims_mask.shape, dtype=bool)
            other_dims_mask[idx] = False
            delta = np.divide(
                left,
                np.prod(dim_chunksizes[other_dims_mask])
                * sizeof_pointing
                / GiBYTES_TO_BYTES,
            )
            int_delta = np.floor(delta)
            if abs(int_delta) > 0 and int_delta + dim_chunksizes[idx] > 0:
                dim_chunksizes[idx] += int_delta

            used = np.prod(dim_chunksizes) * sizeof_pointing / GiBYTES_TO_BYTES

    dim_chunksizes_int = [int(v) for v in dim_chunksizes]
    result = dict(zip(chunked_dim_names, dim_chunksizes_int))

    if ratio < 1:
        logger.debug(
            f"Auto-calculated pointing chunk sizes with {chunksize=}, {total_size=} GiB (for {xds.sizes=}): {result=} which uses {used} GiB."
        )

    return result


def find_baseline_or_antenna_var(xds: xr.Dataset) -> str:
    if "baseline_id" in xds.coords:
        baseline_or_antenna_id = "baseline_id"
    elif "antenna_id" in xds.coords:
        baseline_or_antenna_id = "antenna_id"

    return baseline_or_antenna_id


def itemsize_vis_spec(xds: xr.Dataset) -> int:
    """
    Size in bytes of one visibility (or spectrum) value.
    """
    names = ["SPECTRUM", "VISIBILITY"]
    itemsize = 8
    for var in names:
        if var in xds.data_vars:
            var_name = var
            itemsize = np.dtype(xds.data_vars[var_name].dtype).itemsize
            break

    return itemsize


def itemsize_pointing_spec(xds: xr.Dataset) -> int:
    """
    Size in bytes of one pointing (or spectrum) value.
    """
    pnames = ["BEAM_POINTING"]
    itemsize = 8
    for var in pnames:
        if var in xds.data_vars:
            var_name = var
            itemsize = np.dtype(xds.data_vars[var_name].dtype).itemsize
            break

    return itemsize


def calc_used_gb(
    chunksizes: dict, baseline_or_antenna_id: str, sizeof_vis: int
) -> float:
    return (
        chunksizes["time"]
        * chunksizes[baseline_or_antenna_id]
        * chunksizes["frequency"]
        * chunksizes["polarization"]
        * sizeof_vis
        / GiBYTES_TO_BYTES
    )


# TODO: if the didxs are not used in read_col_conversion, remove didxs from here (and convert_and_write_partition)
def calc_indx_for_row_split(tb_tool, taql_where):
    baselines = get_baselines(tb_tool)
    col_names = tb_tool.colnames()
    cshapes = [
        np.array(tb_tool.getcell(col, 0)).shape
        for col in col_names
        if tb_tool.iscelldefined(col, 0)
    ]

    freq_cnt, pol_cnt = [(cc[0], cc[1]) for cc in cshapes if len(cc) == 2][0]
    utimes, tol = get_utimes_tol(tb_tool, taql_where)

    tidxs = np.searchsorted(utimes, tb_tool.getcol("TIME"))

    ts_ant1, ts_ant2 = (
        tb_tool.getcol("ANTENNA1"),
        tb_tool.getcol("ANTENNA2"),
    )

    ts_bases = np.column_stack((ts_ant1, ts_ant2))
    bidxs = get_baseline_indices(baselines, ts_bases)

    # some antenna 2"s will be out of bounds for this chunk, store rows that are in bounds

    didxs = np.where((bidxs >= 0) & (bidxs < len(baselines)))[0]

    baseline_ant1_id = baselines[:, 0]
    baseline_ant2_id = baselines[:, 1]

    return (
        tidxs,
        bidxs,
        didxs,
        baseline_ant1_id,
        baseline_ant2_id,
        convert_casacore_time(utimes, False),
    )


def create_coordinates(
    xds, in_file, ddi, utime, interval, baseline_ant1_id, baseline_ant2_id
):
    coords = {
        "time": utime,
        "baseline_antenna1_id": ("baseline_id", baseline_ant1_id),
        "baseline_antenna2_id": ("baseline_id", baseline_ant2_id),
        "uvw_label": ["u", "v", "w"],
        "baseline_id": np.arange(len(baseline_ant1_id)),
    }

    ddi_xds = read_generic_table(in_file, "DATA_DESCRIPTION").sel(row=ddi)
    pol_setup_id = ddi_xds.polarization_id.values
    spectral_window_id = int(ddi_xds.spectral_window_id.values)

    spectral_window_xds = read_generic_table(
        in_file,
        "SPECTRAL_WINDOW",
        rename_ids=subt_rename_ids["SPECTRAL_WINDOW"],
    ).sel(spectral_window_id=spectral_window_id)
    coords["frequency"] = spectral_window_xds["chan_freq"].data[
        ~(np.isnan(spectral_window_xds["chan_freq"].data))
    ]

    pol_xds = read_generic_table(
        in_file,
        "POLARIZATION",
        rename_ids=subt_rename_ids["POLARIZATION"],
    )
    num_corr = int(pol_xds["num_corr"][pol_setup_id].values)
    coords["polarization"] = np.vectorize(stokes_types.get)(
        pol_xds["corr_type"][pol_setup_id, :num_corr].values
    )

    xds = xds.assign_coords(coords)

    ###### Create Frequency Coordinate ######
    freq_column_description = spectral_window_xds.attrs["other"]["msv2"]["ctds_attrs"][
        "column_descriptions"
    ]

    msv4_measure = column_description_casacore_to_msv4_measure(
        freq_column_description["CHAN_FREQ"],
        ref_code=spectral_window_xds["meas_freq_ref"].data,
    )
    xds.frequency.attrs.update(msv4_measure)

    xds.frequency.attrs["spectral_window_name"] = str(spectral_window_xds.name.values)
    msv4_measure = column_description_casacore_to_msv4_measure(
        freq_column_description["REF_FREQUENCY"],
        ref_code=spectral_window_xds["meas_freq_ref"].data,
    )
    xds.frequency.attrs["reference_frequency"] = {
        "dims": [],
        "data": float(spectral_window_xds.ref_frequency.values),
        "attrs": msv4_measure,
    }
    xds.frequency.attrs["spectral_window_id"] = spectral_window_id

    # xds.frequency.attrs["effective_channel_width"] = "EFFECTIVE_CHANNEL_WIDTH"
    # Add if doppler table is present
    # xds.frequency.attrs["doppler_velocity"] =
    # xds.frequency.attrs["doppler_type"] =

    unique_chan_width = unique_1d(
        spectral_window_xds.chan_width.data[
            np.logical_not(np.isnan(spectral_window_xds.chan_width.data))
        ]
    )
    # assert len(unique_chan_width) == 1, "Channel width varies for spectral_window."
    # xds.frequency.attrs["channel_width"] = spectral_window_xds.chan_width.data[
    #    ~(np.isnan(spectral_window_xds.chan_width.data))
    # ]  # unique_chan_width[0]
    msv4_measure = column_description_casacore_to_msv4_measure(
        freq_column_description["CHAN_WIDTH"],
        ref_code=spectral_window_xds["meas_freq_ref"].data,
    )
    if not msv4_measure:
        msv4_measure["type"] = "quantity"
        msv4_measure["units"] = ["Hz"]
    xds.frequency.attrs["channel_width"] = {
        "dims": [],
        "data": np.abs(unique_chan_width[0]),
        "attrs": msv4_measure,
    }

    ###### Create Time Coordinate ######
    main_table_attrs = extract_table_attributes(in_file)
    main_column_descriptions = main_table_attrs["column_descriptions"]
    msv4_measure = column_description_casacore_to_msv4_measure(
        main_column_descriptions["TIME"]
    )
    xds.time.attrs.update(msv4_measure)

    msv4_measure = column_description_casacore_to_msv4_measure(
        main_column_descriptions["INTERVAL"]
    )
    if not msv4_measure:
        msv4_measure["type"] = "quantity"
        msv4_measure["units"] = ["s"]
    xds.time.attrs["integration_time"] = {
        "dims": [],
        "data": interval,
        "attrs": msv4_measure,
    }
    xds.time.attrs["effective_integration_time"] = "EFFECTIVE_INTEGRATION_TIME"
    return xds


def find_min_max_times(tb_tool: tables.table, taql_where: str) -> tuple:
    """
    Find the min/max times in an MSv4, for constraining pointing.

    To avoid numerical comparison issues (leaving out some times at the edges),
    it substracts/adds a tolerance from/to the min and max values. The tolerance
    is a fraction of the difference between times / interval of the MS (see
    get_utimes_tol()).

    Parameters
    ----------
    tb_tool : tables.table
        table (query) opened with an MSv4 query

    taql_where : str
        TaQL where that defines the partition of this MSv4

    Returns
    -------
    tuple
        min/max times (raw time values from the Msv2 table)
    """
    utimes, tol = get_utimes_tol(tb_tool, taql_where)
    time_min = utimes.min() - tol
    time_max = utimes.max() + tol
    return (time_min, time_max)


def create_data_variables(
    in_file, xds, tb_tool, time_baseline_shape, tidxs, bidxs, didxs
):
    # Create Data Variables
    col_names = tb_tool.colnames()

    main_table_attrs = extract_table_attributes(in_file)
    main_column_descriptions = main_table_attrs["column_descriptions"]
    for col in col_names:
        if col in col_to_data_variable_names:
            if (col == "WEIGHT") and ("WEIGHT_SPECTRUM" in col_names):
                continue
            try:
                start = time.time()
                if col == "WEIGHT":
                    xds[col_to_data_variable_names[col]] = xr.DataArray(
                        np.tile(
                            read_col_conversion(
                                tb_tool,
                                col,
                                time_baseline_shape,
                                tidxs,
                                bidxs,
                            )[:, :, None, :],
                            (1, 1, xds.sizes["frequency"], 1),
                        ),
                        dims=col_dims[col],
                    )

                else:
                    xds[col_to_data_variable_names[col]] = xr.DataArray(
                        read_col_conversion(
                            tb_tool,
                            col,
                            time_baseline_shape,
                            tidxs,
                            bidxs,
                        ),
                        dims=col_dims[col],
                    )
                    logger.debug(
                        "Time to read column "
                        + str(col)
                        + " : "
                        + str(time.time() - start)
                    )
            except:
                # logger.debug("Could not load column",col)
                # print("Could not load column", col)
                continue

            xds[col_to_data_variable_names[col]].attrs.update(
                create_attribute_metadata(col, main_column_descriptions)
            )


def create_taql_query(partition_info):
    main_par_table_cols = [
        "DATA_DESC_ID",
        "STATE_ID",
        "FIELD_ID",
        "SCAN_NUMBER",
        "STATE_ID",
    ]

    taql_where = "WHERE "
    for col_name in main_par_table_cols:
        if col_name in partition_info:
            taql_where = (
                taql_where
                + f"({col_name} IN [{','.join(map(str, partition_info[col_name]))}]) AND"
            )
    taql_where = taql_where[:-3]

    return taql_where


def convert_and_write_partition(
    in_file: str,
    out_file: str,
    ms_v4_id: int,
    partition_info: Dict,
    partition_scheme: str = "ddi_intent_field",
    main_chunksize: Union[Dict, float, None] = None,
    with_pointing: bool = True,
    pointing_chunksize: Union[Dict, float, None] = None,
    pointing_interpolate: bool = False,
    ephemeris_interpolate: bool = False,
    compressor: numcodecs.abc.Codec = numcodecs.Zstd(level=2),
    storage_backend="zarr",
    overwrite: bool = False,
):
    """_summary_

    Parameters
    ----------
    in_file : str
        _description_
    out_file : str
        _description_
    obs_mode : str
        _description_
    ddi : int, optional
        _description_, by default 0
    state_ids : _type_, optional
        _description_, by default None
    field_id : int, optional
        _description_, by default None
    main_chunksize : Union[Dict, float, None], optional
        _description_, by default None
    with_pointing: bool, optional
        _description_, by default True
    pointing_chunksize : Union[Dict, float, None], optional
        _description_, by default None
    pointing_interpolate : bool, optional
        _description_, by default None
    ephemeris_interpolate : bool, optional
        _description_, by default None
    compressor : numcodecs.abc.Codec, optional
        _description_, by default numcodecs.Zstd(level=2)
    storage_backend : str, optional
        _description_, by default "zarr"
    overwrite : bool, optional
        _description_, by default False

    Returns
    -------
    _type_
        _description_
    """

    taql_where = create_taql_query(partition_info)
    ddi = partition_info["DATA_DESC_ID"][0]
    obs_mode = str(partition_info["OBS_MODE"][0])

    start = time.time()
    with open_table_ro(in_file) as mtable:
        taql_main = f"select * from $mtable {taql_where}"
        with open_query(mtable, taql_main) as tb_tool:

            if tb_tool.nrows() == 0:
                tb_tool.close()
                mtable.close()
                return xr.Dataset(), {}, {}

            logger.debug("Starting a real convert_and_write_partition")
            (
                tidxs,
                bidxs,
                didxs,
                baseline_ant1_id,
                baseline_ant2_id,
                utime,
            ) = calc_indx_for_row_split(tb_tool, taql_where)
            time_baseline_shape = (len(utime), len(baseline_ant1_id))
            logger.debug("Calc indx for row split " + str(time.time() - start))

            start = time.time()
            xds = xr.Dataset()
            # interval = check_if_consistent(tb_tool.getcol("INTERVAL"), "INTERVAL")
            interval = tb_tool.getcol("INTERVAL")

            interval_unique = unique_1d(interval)
            if len(interval_unique) > 1:
                logger.debug(
                    "Integration time (interval) not consitent in partition, using median."
                )
                interval = np.median(interval)
            else:
                interval = interval_unique[0]

            xds = create_coordinates(
                xds, in_file, ddi, utime, interval, baseline_ant1_id, baseline_ant2_id
            )
            logger.debug("Time create coordinates " + str(time.time() - start))

            start = time.time()
            create_data_variables(
                in_file, xds, tb_tool, time_baseline_shape, tidxs, bidxs, didxs
            )
            logger.debug("Time create data variables " + str(time.time() - start))

            # Create ant_xds
            start = time.time()
            ant_xds = create_ant_xds(in_file)
            logger.debug("Time ant xds  " + str(time.time() - start))

            # Create weather_xds
            start = time.time()
            weather_xds = create_weather_xds(in_file)
            logger.debug("Time weather " + str(time.time() - start))

            # To constrain the time range to load (in pointing, ephemerides data_vars)
            time_min_max = find_min_max_times(tb_tool, taql_where)

            if with_pointing:
                start = time.time()
                if pointing_interpolate:
                    pointing_interp_time = xds.time
                else:
                    pointing_interp_time = None
                pointing_xds = create_pointing_xds(
                    in_file, time_min_max, pointing_interp_time
                )
                pointing_chunksize = parse_chunksize(
                    pointing_chunksize, "pointing", pointing_xds
                )
                add_encoding(
                    pointing_xds, compressor=compressor, chunks=pointing_chunksize
                )
                logger.debug(
                    "Time pointing (with add compressor and chunking) "
                    + str(time.time() - start)
                )

            start = time.time()

            # Time and frequency should always be increasing
            if len(xds.frequency) > 1 and xds.frequency[1] - xds.frequency[0] < 0:
                xds = xds.sel(frequency=slice(None, None, -1))

            if len(xds.time) > 1 and xds.time[1] - xds.time[0] < 0:
                xds = xds.sel(time=slice(None, None, -1))

            # Add data_groups and field_info
            xds, is_single_dish = add_data_groups(xds)

            # Create field_and_source_xds (combines field, source and ephemeris data into one super dataset)
            start = time.time()
            if ephemeris_interpolate:
                ephemeris_interp_time = xds.time.values
            else:
                ephemeris_interp_time = None

            scan_id = np.full(time_baseline_shape, -42, dtype=int)
            scan_id[tidxs, bidxs] = tb_tool.getcol("SCAN_NUMBER")
            scan_id = np.max(scan_id, axis=1)

            if "FIELD_ID" not in partition_scheme:
                field_id = np.full(time_baseline_shape, -42, dtype=int)
                field_id[tidxs, bidxs] = tb_tool.getcol("FIELD_ID")
                field_id = np.max(field_id, axis=1)
                field_times = utime
            else:
                field_id = check_if_consistent(tb_tool.getcol("FIELD_ID"), "FIELD_ID")
                field_times = None

            # col_unique = unique_1d(col)
            # assert len(col_unique) == 1, col_name + " is not consistent."
            # return col_unique[0]

            field_and_source_xds, source_id = create_field_and_source_xds(
                in_file,
                field_id,
                xds.frequency.attrs["spectral_window_id"],
                field_times,
                is_single_dish,
                time_min_max,
                ephemeris_interp_time,
            )
            logger.debug("Time field_and_source_xds " + str(time.time() - start))

            # Fix UVW frame
            # From CASA fixvis docs: clean and the im tool ignore the reference frame claimed by the UVW column (it is often mislabelled as ITRF when it is really FK5 (J2000)) and instead assume the (u, v, w)s are in the same frame as the phase tracking center. calcuvw does not yet force the UVW column and field centers to use the same reference frame! Blank = use the phase tracking frame of vis.
            # print('##################',field_and_source_xds)
            if is_single_dish:
                xds.UVW.attrs["frame"] = field_and_source_xds[
                    "FIELD_REFERENCE_CENTER"
                ].attrs["frame"]
            else:
                xds.UVW.attrs["frame"] = field_and_source_xds[
                    "FIELD_PHASE_CENTER"
                ].attrs["frame"]

            if overwrite:
                mode = "w"
            else:
                mode = "w-"

            main_chunksize = parse_chunksize(main_chunksize, "main", xds)
            add_encoding(xds, compressor=compressor, chunks=main_chunksize)
            logger.debug("Time add compressor and chunk " + str(time.time() - start))

            file_name = os.path.join(
                out_file,
                out_file.replace(".vis.zarr", "").replace(".zarr", "").split("/")[-1]
                + "_"
                + str(ms_v4_id),
            )

            xds.attrs["partition_info"] = {
                "spectral_window_id": xds.frequency.attrs["spectral_window_id"],
                "spectral_window_name": xds.frequency.attrs["spectral_window_name"],
                "field_id": to_list(unique_1d(field_id)),
                "field_name": to_list(
                    np.unique(field_and_source_xds.field_name.values)
                ),
                "source_id": to_list(unique_1d(source_id)),
                "source_name": to_list(
                    np.unique(field_and_source_xds.source_name.values)
                ),
                "polarization_setup": to_list(xds.polarization.values),
                "obs_mode": obs_mode,
                "taql": taql_where,
            }

            start = time.time()
            if storage_backend == "zarr":
                xds.to_zarr(store=os.path.join(file_name, "MAIN"), mode=mode)
                ant_xds.to_zarr(store=os.path.join(file_name, "ANTENNA"), mode=mode)
                for group_name in xds.attrs["data_groups"]:
                    field_and_source_xds.to_zarr(
                        store=os.path.join(
                            file_name, f"FIELD_AND_SOURCE_{group_name.upper()}"
                        ),
                        mode=mode,
                    )

                if with_pointing:
                    pointing_xds.to_zarr(store=file_name + "/POINTING", mode=mode)

                if weather_xds:
                    weather_xds.to_zarr(
                        store=os.path.join(file_name, "WEATHER"), mode=mode
                    )

            elif storage_backend == "netcdf":
                # xds.to_netcdf(path=file_name+"/MAIN", mode=mode) #Does not work
                raise
            logger.debug("Write data  " + str(time.time() - start))

    # logger.info("Saved ms_v4 " + file_name + " in " + str(time.time() - start_with) + "s")


def add_data_groups(xds):
    xds.attrs["data_groups"] = {}
    if "VISIBILITY" in xds:
        xds.attrs["data_groups"]["base"] = {
            "visibility": "VISIBILITY",
            "flag": "FLAG",
            "weight": "WEIGHT",
            "uvw": "UVW",
        }

    if "VISIBILITY_CORRECTED" in xds:
        xds.attrs["data_groups"]["corrected"] = {
            "visibility": "VISIBILITY_CORRECTED",
            "flag": "FLAG",
            "weight": "WEIGHT",
            "uvw": "UVW",
        }

    if "VISIBILITY_MODEL" in xds:
        xds.attrs["data_groups"]["model"] = {
            "visibility": "VISIBILITY_MODEL",
            "flag": "FLAG",
            "weight": "WEIGHT",
            "uvw": "UVW",
        }

    is_single_dish = False
    if "SPECTRUM" in xds:
        xds.attrs["data_groups"]["base"] = {
            "spectrum": "SPECTRUM",
            "flag": "FLAG",
            "weight": "WEIGHT",
            "uvw": "UVW",
        }
        is_single_dish = True

    if "SPECTRUM_CORRECTED" in xds:
        xds.attrs["data_groups"]["corrected"] = {
            "spectrum": "SPECTRUM_CORRECTED",
            "flag": "FLAG",
            "weight": "WEIGHT",
            "uvw": "UVW",
        }
        is_single_dish = True

    return xds, is_single_dish
