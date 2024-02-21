import graphviper.utils.logger as logger
from typing import Dict, List, Tuple, Union

import pandas as pd
import numpy as np
import xarray as xr

from casacore import tables

from .load import load_col_chunk
from .read_main_table import get_partition_ids, redim_id_data_vars, rename_vars
from .read import add_units_measures, convert_casacore_time, extract_table_attributes
from .write import revert_time
from .table_query import open_query, open_table_ro
from xradio.vis._vis_utils._ms._tables.read_main_table import (
    get_baselines,
    get_baseline_indices,
)
from xradio.vis._vis_utils._ms.optimised_functions import unique_1d


def load_expanded_main_table_chunk(
    infile: str,
    ddi: int,
    chunk: Dict[str, slice],
    ignore_msv2_cols: Union[list, None] = None,
) -> xr.Dataset:
    """
    Load a chunk of data from main table into memory, with expanded
    dims: (time, baseline, freq, pols)

    :param infile: Input MS path
    :param ddi: DDI to load chunk from
    :param chunk: specification of chunk to load
    :param ignore_msv2_cols: cols that should not be loaded (deprecated MSv2 or similar)

    :return: Xarray datasets with chunk of visibility data, one per DDI (spw_id, pol_setup_id) pair
    """

    taql_where = f"where DATA_DESC_ID = {ddi}"
    taql_ddi = f"select * from $mtable {taql_where}"

    with open_table_ro(infile) as mtable:
        with open_query(mtable, taql_ddi) as tb_tool:
            if tb_tool.nrows() == 0:
                return xr.Dataset()

            xds, part_ids, attrs = load_expanded_ddi_chunk(
                infile, tb_tool, taql_where, chunk, ignore_msv2_cols
            )

    return xds, part_ids, attrs


def load_expanded_ddi_chunk(
    infile: str,
    tb_tool: tables.table,
    taql_pre: str,
    chunk: Dict[str, slice],
    ignore_msv2_cols: Union[list, None] = None,
) -> xr.Dataset:
    """
    Helper function to effectively load the chunk and produce an
    xr.Dataset from a DII once the table and initial query(ies) have
    been opened successfully.

    :param infile: Input MS path
    :param tb_tool: table query contrained to one DDI and chunk time
    range
    :param taql_pre: TaQL query used for tb_tool, with some
    pre-selection of rows and columns
    :param chunk: specification of data chunk to load
    :param ignore_msv2_cols: propagated from calling funtions

    :return: An Xarray datasets with data variables as plain numpy
    arrays loaded directly from the MS columns
    """

    # read the specified chunk of data, figure out indices and lens
    utimes, times = get_chunk_times(taql_pre, chunk)
    baselines, blines = get_chunk_baselines(tb_tool, chunk)
    tidxs, bidxs, didxs, taql_where_chunk = get_chunk_data_indices(
        taql_pre, chunk, utimes, times, baselines, blines
    )

    ctlen = min(len(utimes), times[1] - times[0] + 1)
    cblen = min(len(baselines), blines[1] - blines[0] + 1)
    mvars = load_ddi_cols_chunk(
        ctlen, cblen, tidxs, bidxs, didxs, tb_tool, chunk, ignore_msv2_cols
    )

    mcoords = {
        "time": xr.DataArray(convert_casacore_time(utimes[:ctlen]), dims=["time"]),
        "baseline": xr.DataArray(np.arange(cblen), dims=["baseline"]),
    }

    # add xds global attributes
    cc_attrs = extract_table_attributes(infile)
    attrs = {"other": {"msv2": {"ctds_attrs": cc_attrs, "bad_cols": ignore_msv2_cols}}}
    # add per data var attributes
    mvars = add_units_measures(mvars, cc_attrs)
    mcoords = add_units_measures(mcoords, cc_attrs)

    mvars = rename_vars(mvars)
    mvars = redim_id_data_vars(mvars)
    xds = xr.Dataset(mvars, coords=mcoords)

    part_ids = get_partition_ids(tb_tool, taql_where_chunk)

    # needs an ~equivalent to add_partition_attrs?
    return xds, part_ids, attrs


def load_ddi_cols_chunk(
    ctlen: int,
    cblen: int,
    tidxs: np.ndarray,
    bidxs: np.ndarray,
    didxs: np.ndarray,
    tb_tool: tables.table,
    chunk: Dict[str, slice],
    ignore_msv2_cols: Union[list, None] = None,
) -> Dict[str, np.ndarray]:
    """
    For a given chunk and DDI, load all the MSv2 columns

    :param ctlen: length of the time axis/dim of the chunk
    :param cblen: length of the baseline axis of the chunk
    :param tidxs: time axis indices
    :param bidxs: baseline axis indices
    :param didxs: (effective) data indices, excluding missing baselines
    :param tb_tool: a table/TaQL query open and being used to load columns
    :param chunk: data chunk specification
    :param ignore_msv2_cols: propagated from calling funtions

    :return: columns loaded into memory as np arrays
    """
    cols = tb_tool.colnames()

    cshapes = [
        np.array(tb_tool.getcell(col, 0)).shape
        for col in cols
        if tb_tool.iscelldefined(col, 0)
    ]
    # Assumes shapes are consistent across columns - MSv2
    chan_cnt, pol_cnt = [(csh[0], csh[1]) for csh in cshapes if len(csh) == 2][0]

    dims = ["time", "baseline", "freq", "pol"]
    mvars = {}
    # loop over each column and load data
    for col in cols:
        if (col in ignore_msv2_cols + ["TIME"]) or not tb_tool.iscelldefined(col, 0):
            continue

        cdata = tb_tool.getcol(col, 0, 1)[0]
        cell_shape = cdata.shape
        if len(cell_shape) == 0:
            col_dims = dims[:2]
            mvars[col.lower()] = xr.DataArray(
                load_col_chunk(
                    tb_tool, col, (ctlen, cblen), tidxs, bidxs, didxs, None, None
                ),
                dims=col_dims,
            )

        elif col == "UVW":
            col_dims = dims[:2] + ["uvw_coords"]
            mvars[col.lower()] = xr.DataArray(
                load_col_chunk(
                    tb_tool, col, (ctlen, cblen, 3), tidxs, bidxs, didxs, None, None
                ),
                dims=col_dims,
            )

        elif len(cell_shape) == 1:
            pols, col_dims = get_col_1d_pols(cell_shape, dims, chan_cnt, pol_cnt, chunk)
            cshape = (ctlen, cblen) + (pols[1] - pols[0] + 1,)
            mvars[col.lower()] = xr.DataArray(
                load_col_chunk(tb_tool, col, cshape, tidxs, bidxs, didxs, pols, None),
                dims=col_dims,
            )

        elif len(cell_shape) == 2:
            chans, pols = get_col_2d_chans_pols(cell_shape, chan_cnt, pol_cnt, chunk)
            cshape = (ctlen, cblen) + (chans[1] - chans[0] + 1, pols[1] - pols[0] + 1)
            col_dims = dims
            mvars[col.lower()] = xr.DataArray(
                load_col_chunk(tb_tool, col, cshape, tidxs, bidxs, didxs, chans, pols),
                dims=col_dims,
            )

    return mvars


def get_chunk_times(
    taql_pre: str, chunk: Dict[str, slice]
) -> Tuple[np.ndarray, Tuple[int, int], int]:
    """
    Produces time col/axis related values for a chunk: unique times,
    start/stop times.

    :param chunk: specification of data chunk to load
    :param taql_pre: TaQL query used for tb_tool, with some pre-selection
    of rows and columns.

    :return: array of unique times + (firsr, last) time in the chunk
    """

    taql_utimes = f"select DISTINCT TIME from $mtable {taql_pre}"
    with open_query(None, taql_utimes) as query_utimes:
        utimes = unique_1d(query_utimes.getcol("TIME", 0, -1))
        # add a tol around the time ranges returned by taql
        if len(utimes) < 2:
            tol = 1e-5
        else:
            tol = np.diff(utimes).min() / 4

    if "time" in chunk:
        time_slice = chunk["time"]
        if (
            type(time_slice.start) == pd.Timestamp
            and type(time_slice.stop) == pd.Timestamp
        ):
            times = (
                revert_time(time_slice.start) - tol,
                revert_time(time_slice.stop) + tol,
            )
        elif (
            int(time_slice.start) == time_slice.start
            and int(time_slice.stop) == time_slice.stop
        ):
            # could be operator.index(time_slice.start):
            nutimes = len(utimes)
            times = (
                min(nutimes, int(time_slice.start)),
                min(nutimes, int(time_slice.stop)) - 1,
            )
        else:
            raise ValueError(
                f"Invalid time type. Not a timestamp and Cannot use as"
                f" index: {time_slice.start} (type: {type(time_slice.start)})"
            )
    else:
        times = (utimes[0], utimes[-1])

    return utimes, times


def get_chunk_baselines(tb_tool, chunk) -> Tuple[np.ndarray, Tuple[int, int], int]:
    """
    Produces the basline col/axis related values for a chunk: an array
    of baselines and the start/stop baseline indices.

    :param tb_tool: table/query opened with prev selections (time)
    :param chunk: specification of data chunk to load

    :return: array of baselines + (first, last) baseline in the chunk
    """
    baselines = get_baselines(tb_tool)

    if "baseline" in chunk:
        baseline_chunk = chunk["baseline"]
        baseline_boundaries = (int(baseline_chunk.start), int(baseline_chunk.stop))
    else:
        baseline_boundaries = (baselines[0][0], baselines[-1][0] - 1)

    return baselines, baseline_boundaries


def get_chunk_data_indices(
    taql_pre: str,
    chunk: Dict[str, slice],
    utimes: np.ndarray,
    times: Tuple[int, int],
    baselines: np.ndarray,
    blines: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Produces indices to pass to the casacore getcol(slice) functions
    to load the chunk of data. tidxs (time), bidxs (baseline),
    didxs (effective data indices, considering present/absent
    baselines).

    Time selection is added on top of that.

    :param utimes: array of times in the chunk
    :param times: start, stop time indices
    :param baselines: array of baselines inthe chunk
    :param blines: start, stop baseline indices

    :return: indices along the time, baseline and data (time/baseline)
    axes + the full where... defined for this chunk
    """

    taql_time = f"TIME BETWEEN {utimes[times[0]]} AND {utimes[times[1]]}"
    taql_ant = f"ANTENNA1 BETWEEN {blines[0]} and {blines[1]}"
    taql_where_chunk = f"{taql_pre} AND {taql_time} AND {taql_ant}"
    taql_chunk = f"select * from $mtable {taql_where_chunk}"
    with open_query(None, taql_chunk) as query_times_ants:
        logger.debug(
            f"Opened chunk query, with {query_times_ants.nrows()} rows. Query: {taql_chunk}"
        )
        tidxs = (
            np.searchsorted(utimes, query_times_ants.getcol("TIME", 0, -1)) - times[0]
        )
        ts_ant1, ts_ant2 = (
            query_times_ants.getcol("ANTENNA1", 0, -1),
            query_times_ants.getcol("ANTENNA2", 0, -1),
        )

        ts_bases = np.column_stack((ts_ant1, ts_ant2))

        bidxs = get_baseline_indices(baselines, ts_bases) - blines[0]

    # some antenna 2's will be out of bounds for this chunk, store rows that are in bounds
    didxs = np.where(
        (bidxs >= 0)
        & (bidxs < min(blines[1] - blines[0] + 1, len(baselines) - blines[0]))
    )[0]

    return tidxs, bidxs, didxs, taql_where_chunk


def get_col_1d_pols(
    cell_shape: Tuple[int],
    dims: List[str],
    chan_cnt: int,
    pol_cnt: int,
    chunk: Dict[str, slice],
) -> Tuple[Tuple[int, int], List[str]]:
    """
    For a column with 1d array values, calculate the start/stop
    indices for the last dimension (either pol or freq).
    It also produces the adequate dimension names.

    :param cell_shape: shape of the column
    :param dims: full list of dataset dimensions
    :param chan_cnt: number of channels
    :param pol_cnt: number of pols
    :param chunk: data chunk specification

    :return: first and last pol/freq index of the chunk, and its
    dimension name
    """
    if cell_shape == chan_cnt:
        # chan/freq
        col_dims = dims[:2] + ["freq"]
        if "freq" in chunk:
            pols = (
                min(chan_cnt, chunk["freq"].start),
                min(chan_cnt, chunk["freq"].stop) - 1,
            )
        else:
            pols = (0, cell_shape[0])
    else:
        # pol
        col_dims = dims[:2] + ["pol"]
        if "pol" in chunk:
            pols = (
                min(pol_cnt, chunk["pol"].start),
                min(pol_cnt, chunk["pol"].stop) - 1,
            )
        else:
            pols = (0, cell_shape[0])

    return pols, col_dims


def get_col_2d_chans_pols(
    cell_shape: Tuple[int],
    chan_cnt: int,
    pol_cnt: int,
    chunk: Dict[str, slice],
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """
    For a column with 2d array values (FLAG, DATA, WEIGHT_SPECTRUM,
    etc., calculate the the start/stop indices for the last two
    dimensions of the chunk (freq and pol).
    The dimension names can be assumed to be the full list of dims in
    visibilities (time, baseline, freq, pol).

    :param cell_shape: shape of the column
    :param chan_cnt: number of channels
    :param pol_cnt: number of pols
    :param chunk: data chunk specification

    :return: first and last index for freq (channel) and pol axes of
    the chunk
    """
    if "freq" in chunk:
        chans = (
            min(chan_cnt, chunk["freq"].start),
            min(chan_cnt, chunk["freq"].stop) - 1,
        )
    else:
        chans = (0, cell_shape[0])

    if "pol" in chunk:
        pols = (
            min(pol_cnt, chunk["pol"].start),
            min(pol_cnt, chunk["pol"].stop) - 1,
        )
    else:
        pols = (0, cell_shape[1])

    return chans, pols
