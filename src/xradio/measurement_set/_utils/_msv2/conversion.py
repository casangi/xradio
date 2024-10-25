import datetime
import importlib
import numcodecs
import os
import pathlib
import time
from typing import Dict, Union

import numpy as np
import xarray as xr

import toolviper.utils.logger as logger
from casacore import tables
from xradio.measurement_set._utils._msv2.msv4_sub_xdss import (
    create_pointing_xds,
    create_system_calibration_xds,
    create_weather_xds,
)
from .msv4_info_dicts import create_info_dicts
from xradio.measurement_set._utils._msv2.create_antenna_xds import (
    create_antenna_xds,
    create_gain_curve_xds,
    create_phase_calibration_xds,
)
from xradio.measurement_set._utils._msv2.create_field_and_source_xds import (
    create_field_and_source_xds,
)
from xradio._utils.schema import column_description_casacore_to_msv4_measure
from .msv2_to_msv4_meta import (
    create_attribute_metadata,
    col_to_data_variable_names,
    col_dims,
)

from .._zarr.encoding import add_encoding
from .subtables import subt_rename_ids
from ._tables.table_query import open_table_ro, open_query
from ._tables.read import (
    convert_casacore_time,
    extract_table_attributes,
    read_col_conversion,
    load_generic_table,
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
            "antenna_name",
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

    Assumes these relevant dims: (time, antenna_name/baseline_id, frequency,
    polarization).
    """

    sizeof_vis = itemsize_spec(xds)
    size_all_pols = sizeof_vis * xds.sizes["polarization"]
    if size_all_pols / GiBYTES_TO_BYTES > chunksize:
        raise RuntimeError(
            "Cannot calculate chunk sizes when memory bound ({chunksize}) does not even allow all polarizations in one chunk"
        )

    baseline_or_antenna_name = find_baseline_or_antenna_var(xds)
    total_size = calc_used_gb(xds.sizes, baseline_or_antenna_name, sizeof_vis)

    ratio = chunksize / total_size
    chunked_dims = ["time", baseline_or_antenna_name, "frequency", "polarization"]
    if ratio >= 1:
        result = {dim: xds.sizes[dim] for dim in chunked_dims}
        logger.debug(
            f"{chunksize=} GiB is enough to fully hold {total_size=} GiB (for {xds.sizes=}) in memory in one chunk"
        )
    else:
        xds_dim_sizes = {k: xds.sizes[k] for k in chunked_dims}
        result = mem_chunksize_to_dict_main_balanced(
            chunksize, xds_dim_sizes, baseline_or_antenna_name, sizeof_vis
        )

    return result


def mem_chunksize_to_dict_main_balanced(
    chunksize: float,
    xds_dim_sizes: dict,
    baseline_or_antenna_name: str,
    sizeof_vis: int,
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
    # Note the use of np.floor, this iteration can either increase or decrease sizes,
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

    chunked_dim_names = ["time", baseline_or_antenna_name, "frequency", "polarization"]
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
        baseline_or_antenna_name = "baseline_id"
    elif "antenna_name" in xds.coords:
        baseline_or_antenna_name = "antenna_name"

    return baseline_or_antenna_name


def itemsize_spec(xds: xr.Dataset) -> int:
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
    chunksizes: dict, baseline_or_antenna_name: str, sizeof_vis: int
) -> float:
    return (
        chunksizes["time"]
        * chunksizes[baseline_or_antenna_name]
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
    xds, in_file, ddi, utime, interval, baseline_ant1_id, baseline_ant2_id, scan_id
):
    coords = {
        "time": utime,
        "baseline_antenna1_id": ("baseline_id", baseline_ant1_id),
        "baseline_antenna2_id": ("baseline_id", baseline_ant2_id),
        "baseline_id": np.arange(len(baseline_ant1_id)),
        "scan_number": ("time", scan_id),
        "uvw_label": ["u", "v", "w"],
    }

    ddi_xds = load_generic_table(in_file, "DATA_DESCRIPTION").sel(row=ddi)
    pol_setup_id = ddi_xds.POLARIZATION_ID.values
    spectral_window_id = int(ddi_xds.SPECTRAL_WINDOW_ID.values)

    spectral_window_xds = load_generic_table(
        in_file,
        "SPECTRAL_WINDOW",
        rename_ids=subt_rename_ids["SPECTRAL_WINDOW"],
    ).sel(spectral_window_id=spectral_window_id)
    coords["frequency"] = spectral_window_xds["CHAN_FREQ"].data[
        ~(np.isnan(spectral_window_xds["CHAN_FREQ"].data))
    ]

    pol_xds = load_generic_table(
        in_file,
        "POLARIZATION",
        rename_ids=subt_rename_ids["POLARIZATION"],
    )
    num_corr = int(pol_xds["NUM_CORR"][pol_setup_id].values)
    coords["polarization"] = np.vectorize(stokes_types.get)(
        pol_xds["CORR_TYPE"][pol_setup_id, :num_corr].values
    )

    xds = xds.assign_coords(coords)

    ###### Create Frequency Coordinate ######
    freq_column_description = spectral_window_xds.attrs["other"]["msv2"]["ctds_attrs"][
        "column_descriptions"
    ]

    msv4_measure = column_description_casacore_to_msv4_measure(
        freq_column_description["CHAN_FREQ"],
        ref_code=spectral_window_xds["MEAS_FREQ_REF"].data,
    )
    xds.frequency.attrs.update(msv4_measure)

    spw_name = spectral_window_xds.NAME.values.item()
    if (spw_name is None) or (spw_name == "none") or (spw_name == ""):
        spw_name = "spw_" + str(spectral_window_id)
    else:
        # spw_name = spectral_window_xds.NAME.values.item()
        spw_name = spw_name + "_" + str(spectral_window_id)

    xds.frequency.attrs["spectral_window_name"] = spw_name
    msv4_measure = column_description_casacore_to_msv4_measure(
        freq_column_description["REF_FREQUENCY"],
        ref_code=spectral_window_xds["MEAS_FREQ_REF"].data,
    )
    xds.frequency.attrs["reference_frequency"] = {
        "dims": [],
        "data": float(spectral_window_xds.REF_FREQUENCY.values),
        "attrs": msv4_measure,
    }
    xds.frequency.attrs["spectral_window_id"] = spectral_window_id

    # Add if doppler table is present
    # xds.frequency.attrs["doppler_velocity"] =
    # xds.frequency.attrs["doppler_type"] =

    unique_chan_width = unique_1d(
        spectral_window_xds["CHAN_WIDTH"].data[
            np.logical_not(np.isnan(spectral_window_xds["CHAN_WIDTH"].data))
        ]
    )
    # assert len(unique_chan_width) == 1, "Channel width varies for spectral_window."
    # xds.frequency.attrs["channel_width"] = spectral_window_xds.chan_width.data[
    #    ~(np.isnan(spectral_window_xds.chan_width.data))
    # ]  # unique_chan_width[0]
    msv4_measure = column_description_casacore_to_msv4_measure(
        freq_column_description["CHAN_WIDTH"],
        ref_code=spectral_window_xds["MEAS_FREQ_REF"].data,
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
    in_file, xds, tb_tool, time_baseline_shape, tidxs, bidxs, didxs, use_table_iter
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
                    xds = get_weight(
                        xds,
                        col,
                        tb_tool,
                        time_baseline_shape,
                        tidxs,
                        bidxs,
                        use_table_iter,
                        main_column_descriptions,
                    )
                else:
                    xds[col_to_data_variable_names[col]] = xr.DataArray(
                        read_col_conversion(
                            tb_tool,
                            col,
                            time_baseline_shape,
                            tidxs,
                            bidxs,
                            use_table_iter,
                        ),
                        dims=col_dims[col],
                    )

                xds[col_to_data_variable_names[col]].attrs.update(
                    create_attribute_metadata(col, main_column_descriptions)
                )

                logger.debug(
                    "Time to read column " + str(col) + " : " + str(time.time() - start)
                )
            except Exception as exc:
                logger.debug(f"Could not load column {col}, exception: {exc}")

                if ("WEIGHT_SPECTRUM" == col) and (
                    "WEIGHT" in col_names
                ):  # Bogus WEIGHT_SPECTRUM column, need to use WEIGHT.
                    xds = get_weight(
                        xds,
                        "WEIGHT",
                        tb_tool,
                        time_baseline_shape,
                        tidxs,
                        bidxs,
                        use_table_iter,
                        main_column_descriptions,
                    )


def add_missing_data_var_attrs(xds):
    """
    Adds in the xds attributes expected metadata that cannot be found in the input MSv2.
    For now:
    - missing single-dish/SPECTRUM metadata
    - missing interferometry/VISIBILITY_MODEL metadata
    """
    data_var_names = ["SPECTRUM", "SPECTRUM_CORRECTED"]
    for var_name in data_var_names:
        if var_name in xds.data_vars:
            xds.data_vars[var_name].attrs["units"] = [""]

    vis_var_names = ["VISIBILITY_MODEL"]
    for var_name in vis_var_names:
        if var_name in xds.data_vars and "units" not in xds.data_vars[var_name].attrs:
            # Assume MODEL uses the same units
            if "VISIBILITY" in xds.data_vars:
                xds.data_vars[var_name].attrs["units"] = xds.data_vars[
                    "VISIBILITY"
                ].attrs["units"]
            else:
                xds.data_vars[var_name].attrs["units"] = [""]

    return xds


def get_weight(
    xds,
    col,
    tb_tool,
    time_baseline_shape,
    tidxs,
    bidxs,
    use_table_iter,
    main_column_descriptions,
):
    xds[col_to_data_variable_names[col]] = xr.DataArray(
        np.tile(
            read_col_conversion(
                tb_tool,
                col,
                time_baseline_shape,
                tidxs,
                bidxs,
                use_table_iter,
            )[:, :, None, :],
            (1, 1, xds.sizes["frequency"], 1),
        ),
        dims=col_dims[col],
    )

    xds[col_to_data_variable_names[col]].attrs.update(
        create_attribute_metadata(col, main_column_descriptions)
    )
    return xds


def create_taql_query(partition_info):
    main_par_table_cols = [
        "DATA_DESC_ID",
        "OBSERVATION_ID",
        "STATE_ID",
        "FIELD_ID",
        "SCAN_NUMBER",
        "STATE_ID",
        "ANTENNA1",
    ]

    taql_where = "WHERE "
    for col_name in main_par_table_cols:
        if col_name in partition_info:
            taql_where = (
                taql_where
                + f"({col_name} IN [{','.join(map(str, partition_info[col_name]))}]) AND"
            )
            if col_name == "ANTENNA1":
                taql_where = (
                    taql_where
                    + f"(ANTENNA2 IN [{','.join(map(str, partition_info[col_name]))}]) AND"
                )
    taql_where = taql_where[:-3]

    return taql_where


def fix_uvw_frame(
    xds: xr.Dataset, field_and_source_xds: xr.Dataset, is_single_dish: bool
) -> xr.Dataset:
    """
    Fix UVW frame

    From CASA fixvis docs: clean and the im tool ignore the reference frame claimed by the UVW column (it is often
    mislabelled as ITRF when it is really FK5 (J2000)) and instead assume the (u, v, w)s are in the same frame as the phase
    tracking center. calcuvw does not yet force the UVW column and field centers to use the same reference frame!
    Blank = use the phase tracking frame of vis.
    """
    if xds.UVW.attrs["frame"] == "ITRF":
        if is_single_dish:
            center_var = "FIELD_REFERENCE_CENTER"
        else:
            center_var = "FIELD_PHASE_CENTER"

        xds.UVW.attrs["frame"] = field_and_source_xds[center_var].attrs["frame"]

    return xds


def convert_and_write_partition(
    in_file: str,
    out_file: str,
    ms_v4_id: Union[int, str],
    partition_info: Dict,
    use_table_iter: bool,
    partition_scheme: str = "ddi_intent_field",
    main_chunksize: Union[Dict, float, None] = None,
    with_pointing: bool = True,
    pointing_chunksize: Union[Dict, float, None] = None,
    pointing_interpolate: bool = False,
    ephemeris_interpolate: bool = False,
    phase_cal_interpolate: bool = False,
    sys_cal_interpolate: bool = False,
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
    intents : str
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
    phase_cal_interpolate : bool, optional
        _description_, by default None
    sys_cal_interpolate : bool, optional
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
    intents = str(partition_info["OBS_MODE"][0])

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

            observation_id = check_if_consistent(
                tb_tool.getcol("OBSERVATION_ID"), "OBSERVATION_ID"
            )

            def get_observation_info(in_file, observation_id, intents):
                generic_observation_xds = load_generic_table(
                    in_file,
                    "OBSERVATION",
                    taql_where=f" where (ROWID() IN [{str(observation_id)}])",
                )

                if intents == "None":
                    intents = "obs_" + str(observation_id)

                return generic_observation_xds["TELESCOPE_NAME"].values[0], intents

            telescope_name, intents = get_observation_info(
                in_file, observation_id, intents
            )

            start = time.time()
            xds = xr.Dataset(
                attrs={
                    "creation_date": datetime.datetime.utcnow().isoformat(),
                    "xradio_version": importlib.metadata.version("xradio"),
                    "schema_version": "4.0.-9994",
                    "type": "visibility",
                }
            )

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

            scan_id = np.full(time_baseline_shape, -42, dtype=int)
            scan_id[tidxs, bidxs] = tb_tool.getcol("SCAN_NUMBER")
            scan_id = np.max(scan_id, axis=1)

            xds = create_coordinates(
                xds,
                in_file,
                ddi,
                utime,
                interval,
                baseline_ant1_id,
                baseline_ant2_id,
                scan_id,
            )
            logger.debug("Time create coordinates " + str(time.time() - start))

            start = time.time()
            create_data_variables(
                in_file,
                xds,
                tb_tool,
                time_baseline_shape,
                tidxs,
                bidxs,
                didxs,
                use_table_iter,
            )

            # Add data_groups
            xds, is_single_dish = add_data_groups(xds)
            xds = add_missing_data_var_attrs(xds)

            if (
                "WEIGHT" not in xds.data_vars
            ):  # Some single dish datasets don't have WEIGHT.
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

            logger.debug("Time create data variables " + str(time.time() - start))

            # To constrain the time range to load (in pointing, ephemerides, phase_cal data_vars)
            time_min_max = find_min_max_times(tb_tool, taql_where)

            # Create ant_xds
            start = time.time()
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
                    [xds["baseline_antenna1_id"].data, xds["baseline_antenna2_id"].data]
                )
            )
            if phase_cal_interpolate:
                phase_cal_interp_time = xds.time.values
            else:
                phase_cal_interp_time = None

            ant_xds = create_antenna_xds(
                in_file,
                xds.frequency.attrs["spectral_window_id"],
                antenna_id,
                feed_id,
                telescope_name,
                xds.polarization,
            )
            logger.debug("Time antenna xds  " + str(time.time() - start))

            start = time.time()
            gain_curve_xds = create_gain_curve_xds(
                in_file, xds.frequency.attrs["spectral_window_id"], ant_xds
            )
            logger.debug("Time gain_curve xds  " + str(time.time() - start))

            start = time.time()
            phase_calibration_xds = create_phase_calibration_xds(
                in_file,
                xds.frequency.attrs["spectral_window_id"],
                ant_xds,
                time_min_max,
                phase_cal_interp_time,
            )
            logger.debug("Time phase_calibration xds  " + str(time.time() - start))

            # Change antenna_ids to antenna_names
            with_antenna_partitioning = "ANTENNA1" in partition_info
            xds = antenna_ids_to_names(
                xds, ant_xds, is_single_dish, with_antenna_partitioning
            )
            # but before, keep the name-id arrays, we need them for the pointing and weather xds
            ant_xds_name_ids = ant_xds["antenna_name"].set_xindex("antenna_id")
            ant_xds_station_name_ids = ant_xds["station"].set_xindex("antenna_id")
            # No longer needed after converting to name.
            ant_xds = ant_xds.drop_vars("antenna_id")

            # Create system_calibration_xds
            start = time.time()
            if sys_cal_interpolate:
                sys_cal_interp_time = xds.time.values
            else:
                sys_cal_interp_time = None
            system_calibration_xds = create_system_calibration_xds(
                in_file,
                xds.frequency,
                ant_xds_name_ids,
                sys_cal_interp_time,
            )
            logger.debug("Time system_calibation " + str(time.time() - start))

            # Create weather_xds
            start = time.time()
            weather_xds = create_weather_xds(in_file, ant_xds_station_name_ids)
            logger.debug("Time weather " + str(time.time() - start))

            # Create pointing_xds
            pointing_xds = xr.Dataset()
            if with_pointing:
                start = time.time()
                if pointing_interpolate:
                    pointing_interp_time = xds.time
                else:
                    pointing_interp_time = None
                pointing_xds = create_pointing_xds(
                    in_file, ant_xds_name_ids, time_min_max, pointing_interp_time
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

            # Create field_and_source_xds (combines field, source and ephemeris data into one super dataset)
            start = time.time()
            if ephemeris_interpolate:
                ephemeris_interp_time = xds.time.values
            else:
                ephemeris_interp_time = None

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

            field_and_source_xds, source_id, _num_lines = create_field_and_source_xds(
                in_file,
                field_id,
                xds.frequency.attrs["spectral_window_id"],
                field_times,
                is_single_dish,
                time_min_max,
                ephemeris_interp_time,
            )
            logger.debug("Time field_and_source_xds " + str(time.time() - start))

            xds = fix_uvw_frame(xds, field_and_source_xds, is_single_dish)

            partition_info_misc_fields = {
                "scan_id": scan_id,
                "intents": intents,
                "taql_where": taql_where,
            }
            info_dicts = create_info_dicts(
                in_file, xds, field_and_source_xds, partition_info_misc_fields, tb_tool
            )
            xds.attrs.update(info_dicts)

            # xds ready, prepare to write
            start = time.time()
            main_chunksize = parse_chunksize(main_chunksize, "main", xds)
            add_encoding(xds, compressor=compressor, chunks=main_chunksize)
            logger.debug("Time add compressor and chunk " + str(time.time() - start))

            file_name = os.path.join(
                out_file,
                pathlib.Path(in_file).name.replace(".ms", "") + "_" + str(ms_v4_id),
            )

            if overwrite:
                mode = "w"
            else:
                mode = "w-"

            if is_single_dish:
                xds.attrs["type"] = "spectrum"
                xds = xds.drop_vars(["UVW"])
                del xds["uvw_label"]
            else:
                if any("WVR" in s for s in intents):
                    xds.attrs["type"] = "wvr"
                else:
                    xds.attrs["type"] = "visibility"

            start = time.time()
            if storage_backend == "zarr":
                xds.to_zarr(store=os.path.join(file_name, "correlated_xds"), mode=mode)
                ant_xds.to_zarr(store=os.path.join(file_name, "antenna_xds"), mode=mode)
                for group_name in xds.attrs["data_groups"]:
                    field_and_source_xds.to_zarr(
                        store=os.path.join(
                            file_name, f"field_and_source_xds_{group_name}"
                        ),
                        mode=mode,
                    )

                if with_pointing and len(pointing_xds.data_vars) > 0:
                    pointing_xds.to_zarr(
                        store=os.path.join(file_name, "pointing_xds"), mode=mode
                    )

                if system_calibration_xds:
                    system_calibration_xds.to_zarr(
                        store=os.path.join(file_name, "system_calibration_xds"),
                        mode=mode,
                    )

                if gain_curve_xds:
                    gain_curve_xds.to_zarr(
                        store=os.path.join(file_name, "gain_curve_xds"), mode=mode
                    )

                if phase_calibration_xds:
                    phase_calibration_xds.to_zarr(
                        store=os.path.join(file_name, "phase_calibration_xds"),
                        mode=mode,
                    )

                if weather_xds:
                    weather_xds.to_zarr(
                        store=os.path.join(file_name, "weather_xds"), mode=mode
                    )

            elif storage_backend == "netcdf":
                # xds.to_netcdf(path=file_name+"/MAIN", mode=mode) #Does not work
                raise
            logger.debug("Write data  " + str(time.time() - start))

    # logger.info("Saved ms_v4 " + file_name + " in " + str(time.time() - start_with) + "s")


def antenna_ids_to_names(
    xds: xr.Dataset,
    ant_xds: xr.Dataset,
    is_single_dish: bool,
    with_antenna_partitioning,
) -> xr.Dataset:
    """
    Turns the antenna_ids that we get from MSv2 into MSv4 antenna_name

    Parameters
    ----------
    xds: xr.Dataset
        A main xds (MSv4)
    ant_xds: xr.Dataset
        The antenna_xds for this MSv4
    is_single_dish: bool
        Whether a single-dish ("spectrum" data) dataset
    with_antenna_partitioning: bool
        Whether the MSv4 partitions include the antenna axis => only
        one antenna (and implicitly one 'baseline' - auto-correlation)

    Returns
    ----------
    xr.Dataset
        The main xds with antenna_id replaced with antenna_name
    """
    ant_xds = ant_xds.set_xindex(
        "antenna_id"
    )  # Allows for non-dimension coordinate selection.

    if not is_single_dish:  # Interferometer
        xds["baseline_antenna1_id"].data = ant_xds["antenna_name"].sel(
            antenna_id=xds["baseline_antenna1_id"].data
        )
        xds["baseline_antenna2_id"].data = ant_xds["antenna_name"].sel(
            antenna_id=xds["baseline_antenna2_id"].data
        )
        xds = xds.rename(
            {
                "baseline_antenna1_id": "baseline_antenna1_name",
                "baseline_antenna2_id": "baseline_antenna2_name",
            }
        )
    else:
        if not with_antenna_partitioning:
            # baseline_antenna1_id will be removed soon below, but it is useful here to know the actual antenna_ids,
            # as opposed to the baseline_ids which can mismatch when data is missing for some antennas
            xds["baseline_id"] = ant_xds["antenna_name"].sel(
                antenna_id=xds["baseline_antenna1_id"]
            )
        else:
            xds["baseline_id"] = ant_xds["antenna_name"]

        unwanted_coords_from_ant_xds = [
            "antenna_id",
            "antenna_name",
            "mount",
            "station",
        ]
        for unwanted_coord in unwanted_coords_from_ant_xds:
            xds = xds.drop_vars(unwanted_coord)
        xds = xds.rename({"baseline_id": "antenna_name"})

        # drop more vars that seem unwanted in main_sd_xds, but there shouuld be a better way
        # of not creating them in the first place
        unwanted_coords_sd = ["baseline_antenna1_id", "baseline_antenna2_id"]
        for unwanted_coord in unwanted_coords_sd:
            xds = xds.drop_vars(unwanted_coord)

    return xds


def add_data_groups(xds):
    xds.attrs["data_groups"] = {}
    if "VISIBILITY" in xds:
        xds.attrs["data_groups"]["base"] = {
            "correlated_data": "VISIBILITY",
            "flag": "FLAG",
            "weight": "WEIGHT",
            "uvw": "UVW",
        }

    if "VISIBILITY_CORRECTED" in xds:
        xds.attrs["data_groups"]["corrected"] = {
            "correlated_data": "VISIBILITY_CORRECTED",
            "flag": "FLAG",
            "weight": "WEIGHT",
            "uvw": "UVW",
        }

    if "VISIBILITY_MODEL" in xds:
        xds.attrs["data_groups"]["model"] = {
            "correlated_data": "VISIBILITY_MODEL",
            "flag": "FLAG",
            "weight": "WEIGHT",
            "uvw": "UVW",
        }

    is_single_dish = False
    if "SPECTRUM" in xds:
        xds.attrs["data_groups"]["base"] = {
            "correlated_data": "SPECTRUM",
            "flag": "FLAG",
            "weight": "WEIGHT",
            "uvw": "UVW",
        }
        is_single_dish = True

    if "SPECTRUM_MODEL" in xds:
        xds.attrs["data_groups"]["model"] = {
            "correlated_data": "SPECTRUM_MODEL",
            "flag": "FLAG",
            "weight": "WEIGHT",
            "uvw": "UVW",
        }
        is_single_dish = True

    if "SPECTRUM_CORRECTED" in xds:
        xds.attrs["data_groups"]["corrected"] = {
            "correlated_data": "SPECTRUM_CORRECTED",
            "flag": "FLAG",
            "weight": "WEIGHT",
            "uvw": "UVW",
        }
        is_single_dish = True

    return xds, is_single_dish
