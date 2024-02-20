import graphviper.utils.logger as logger
from typing import Any, Dict, List, Tuple, Union

import dask, dask.array
import numpy as np
import xarray as xr
import pandas as pd

from casacore import tables

from .read import (
    read_flat_col_chunk,
    read_col_chunk,
    convert_casacore_time,
    extract_table_attributes,
    add_units_measures,
)

from .table_query import open_table_ro, open_query
from xradio.vis._vis_utils._ms.optimised_functions import (
    unique_1d,
    pairing_function,
    inverse_pairing_function,
)

rename_msv2_cols = {
    "antenna1": "antenna1_id",
    "antenna2": "antenna2_id",
    "feed1": "feed1_id",
    "feed2": "feed2_id",
    # optional cols:
    "weight_spectrum": "weight",
    "corrected_data": "vis_corrected",
    "data": "vis",
    "model_data": "vis_model",
    "float_data": "autocorr",
}


def rename_vars(mvars: Dict[str, xr.DataArray]) -> Dict[str, xr.DataArray]:
    """Apply rename rules. Also preserve ordering of data_vars

    Note: not using xr.DataArray.rename because we have optional
    column renames and rename complains if some of the names passed
    are not present in the dataset

    :param mvars: dictionary of data_vars to be used to create an xr.Dataset
    :return: similar dictionary after applying MSv2 => MSv3/ngCASA renaming rules
    """
    renamed = {
        rename_msv2_cols[name] if name in rename_msv2_cols else name: var
        for name, var in mvars.items()
    }

    return renamed


def redim_id_data_vars(mvars: Dict[str, xr.DataArray]) -> Dict[str, xr.DataArray]:
    """
    Changes:
    Several id data variables to drop its baseline dim
    The antenna id data vars:
     From MS (antenna1_id(time, baseline), antenna2_id(time,baseline)
     To cds (baseline_ant1_id(baseline), baseline_ant2_id(baseline)
    :param mvars: data variables being prepared for a partition xds
    :return: data variables with the ant id ones modified to cds type
    """
    # Vars to drop baseline dim
    var_names = [
        "array_id",
        "observation_id",
        "processor_id",
        "scan_number",
        "state_id",
    ]
    for vname in var_names:
        mvars[vname] = mvars[vname].sel(baseline=0, drop=True)

    for idx in ["1", "2"]:
        new_name = f"baseline_ant{idx}_id"
        mvars[new_name] = mvars.pop(f"antenna{idx}_id")
        mvars[new_name] = mvars[new_name].sel(time=0, drop=True)

    return mvars


def get_partition_ids(mtable: tables.table, taql_where: str) -> Dict:
    """Get some of the partition IDs that we have to retrieve from some
    of the top level ID/sorting cols of the main table of the MS.

    :param mtable: MS main table
    :param taql_where: where part that defines the partition in TaQL
    :return: ids of array, observation, and processor
    """

    taql_ids = f"select DISTINCT ARRAY_ID, OBSERVATION_ID, PROCESSOR_ID from $mtable {taql_where}"
    with open_query(mtable, taql_ids) as query:
        # array_id, observation_id, processor_id
        array_id = unique_1d(query.getcol("ARRAY_ID"))
        obs_id = unique_1d(query.getcol("OBSERVATION_ID"))
        proc_id = unique_1d(query.getcol("PROCESSOR_ID"))
        check_vars = [
            (array_id, "array_id"),
            (obs_id, "observation_id"),
            (proc_id, "processor_id"),
        ]
        for var, var_name in check_vars:
            if len(var) != 1:
                logger.warning(
                    f"Did not get exactly one {var_name} (got {var} for this partition. TaQL: {taql_where}"
                )

    pids = {
        "array_id": list(array_id),
        "observation_id": list(obs_id),
        "processor_id": list(proc_id),
    }
    return pids


def read_expanded_main_table(
    infile: str,
    ddi: int = 0,
    scan_state: Union[Tuple[int, int], None] = None,
    ignore_msv2_cols: Union[list, None] = None,
    chunks: Tuple[int, ...] = (400, 200, 100, 2),
) -> Tuple[xr.Dataset, Dict[str, Any], Dict[str, Any]]:
    """
    Reads one partition from the main table, all columns.
    This is the expanded version (time, baseline) dims.

    Chunk tuple: (time, baseline, freq, pol)
    """
    if ignore_msv2_cols is None:
        ignore_msv2_cols = []

    taql_where = f"where DATA_DESC_ID = {ddi}"
    if scan_state:
        # get partitions by scan/state
        scan, state = scan_state
        if type(state) == np.ndarray:
            state_ids_or = " OR STATE_ID = ".join(np.char.mod("%d", state))
            taql_where += f" AND (STATE_ID = {state_ids_or})"
        elif state:
            taql_where += f" AND SCAN_NUMBER = {scan} AND STATE_ID = {state}"
        elif scan:
            # scan can also be None, when partition_scheme='intent'
            # but the STATE table is empty!
            taql_where += f" AND SCAN_NUMBER = {scan}"

    with open_table_ro(infile) as mtable:
        # one partition, select just the specified ddi (+ scan/subscan)
        taql_main = f"select * from $mtable {taql_where}"
        with open_query(mtable, taql_main) as tb_tool:
            if tb_tool.nrows() == 0:
                tb_tool.close()
                mtable.close()
                return xr.Dataset(), {}, {}

            xds, attrs = read_main_table_chunks(
                infile, tb_tool, taql_where, ignore_msv2_cols, chunks
            )
            part_ids = get_partition_ids(tb_tool, taql_where)

    return xds, part_ids, attrs


def read_main_table_chunks(
    infile: str,
    tb_tool: tables.table,
    taql_where: str,
    ignore_msv2_cols: Union[list, None] = None,
    chunks: Tuple[int, ...] = (400, 200, 100, 2),
) -> Tuple[xr.Dataset, Dict[str, Any]]:
    """
    Iterates through the time,baseline chunks and reads slices from
    all the data columns.
    """
    baselines = get_baselines(tb_tool)

    col_names = tb_tool.colnames()
    cshapes = [
        np.array(tb_tool.getcell(col, 0)).shape
        for col in col_names
        if tb_tool.iscelldefined(col, 0)
    ]
    chan_cnt, pol_cnt = [(cc[0], cc[1]) for cc in cshapes if len(cc) == 2][0]

    unique_times, tol = get_utimes_tol(tb_tool, taql_where)

    tvars = {}
    n_baselines = len(baselines)
    n_unique_times = len(unique_times)
    n_time_chunks = chunks[0]
    n_baseline_chunks = chunks[1]
    # loop over time chunks
    for time_chunk in range(0, n_unique_times, n_time_chunks):
        time_start = (unique_times[time_chunk] - tol,)
        time_end = (
            unique_times[min(n_unique_times, time_chunk + n_time_chunks) - 1] + tol
        )

        # chunk time length
        ctlen = min(n_unique_times, time_chunk + n_time_chunks) - time_chunk

        bvars = {}
        # loop over baseline chunks
        for baseline_chunk in range(0, n_baselines, n_baseline_chunks):
            cblen = min(n_baselines - baseline_chunk, n_baseline_chunks)

            # read the specified chunk of data
            # def read_chunk(infile, ddi, times, blines, chans, pols):
            ttql = f"TIME BETWEEN {time_start} and {time_end}"
            ant1_start = baselines[baseline_chunk][0]
            ant1_end = baselines[cblen + baseline_chunk - 1][0]
            atql = f"ANTENNA1 BETWEEN {ant1_start} and {ant1_end}"
            ts_taql = f"select * from $mtable {taql_where} AND {ttql} AND {atql}"
            with open_query(None, ts_taql) as query_times_ants:
                tidxs = (
                    np.searchsorted(
                        unique_times, query_times_ants.getcol("TIME", 0, -1)
                    )
                    - time_chunk
                )
                ts_ant1, ts_ant2 = (
                    query_times_ants.getcol("ANTENNA1", 0, -1),
                    query_times_ants.getcol("ANTENNA2", 0, -1),
                )

            ts_bases = np.column_stack((ts_ant1, ts_ant2))

            bidxs = get_baseline_indices(baselines, ts_bases) - baseline_chunk

            # some antenna 2's will be out of bounds for this chunk, store rows that are in bounds
            didxs = np.where(
                (bidxs >= 0) & (bidxs < min(chunks[1], n_baselines - baseline_chunk))
            )[0]

            delayed_params = (infile, ts_taql, (ctlen, cblen), tidxs, bidxs, didxs)

            read_all_cols_bvars(
                tb_tool, chunks, chan_cnt, ignore_msv2_cols, delayed_params, bvars
            )

        concat_bvars_update_tvars(bvars, tvars)

    dims = ["time", "baseline", "freq", "pol"]
    mvars = concat_tvars_to_mvars(dims, tvars, pol_cnt, chan_cnt)
    mcoords = {
        "time": xr.DataArray(convert_casacore_time(unique_times), dims=["time"]),
        "baseline": xr.DataArray(np.arange(n_baselines), dims=["baseline"]),
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

    return xds, attrs


def get_utimes_tol(mtable: tables.table, taql_where: str) -> Tuple[np.ndarray, float]:
    taql_utimes = f"select DISTINCT TIME from $mtable {taql_where}"
    with open_query(mtable, taql_utimes) as query_utimes:
        utimes = unique_1d(query_utimes.getcol("TIME", 0, -1))
        # add a tol around the time ranges returned by taql
        if len(utimes) < 2:
            tol = 1e-5
        else:
            tol = np.diff(utimes).min() / 4

    return utimes, tol


def get_baselines(tb_tool: tables.table) -> np.ndarray:
    """Gets the unique baselines from antenna 1 and antenna 2 ids.

    Uses a pairing function and inverse pairing function to decrease the
    computation time of finding unique values.

    Args:
        tb_tool (tables.table): MeasurementSet table to get the antenna ids.

    Returns:
        unique_baselines (np.ndarray): a 2D array of unique antenna pairs
        (baselines) from the MeasurementSet table provided.
    """
    ant1, ant2 = tb_tool.getcol("ANTENNA1", 0, -1), tb_tool.getcol("ANTENNA2", 0, -1)

    baselines = np.column_stack((ant1, ant2))

    # Using pairing function to reduce the computation time of finding unique values.
    baselines_paired = pairing_function(baselines)
    unique_baselines_paired = pd.unique(baselines_paired)
    unique_baselines = inverse_pairing_function(unique_baselines_paired)

    # Sorting the unique baselines.
    unique_baselines = unique_baselines[unique_baselines[:, 1].argsort()]
    unique_baselines = unique_baselines[
        unique_baselines[:, 0].argsort(kind="mergesort")
    ]

    return unique_baselines


def get_baseline_indices(
    unique_baselines: np.ndarray, baseline_set: np.ndarray
) -> np.ndarray:
    """Finds the baseline indices of a set of baselines using the unique baselines.

    Uses a pairing function to reduce the number of values so it's more
    efficient to find the indices.

    Args:
        unique_baselines (np.ndarray): a 2D array of unique antenna pairs
        (baselines).
        baseline_set (np.ndarray): a 2D array of antenna pairs (baselines). This
        array may contain duplicates.

    Returns:
        baseline_indices (np.ndarray): the indices of the baseline set that
        correspond to the unique baselines.
    """
    unique_baselines_paired = pairing_function(unique_baselines)
    baseline_set_paired = pairing_function(baseline_set)

    # Pairing function doesn't preserve order so they need to be sorted.
    unique_baselines_sorted = np.argsort(unique_baselines_paired)
    sorted_indices = np.searchsorted(
        unique_baselines_paired[unique_baselines_sorted],
        baseline_set_paired,
    )
    baseline_indices = unique_baselines_sorted[sorted_indices]

    return baseline_indices


def read_all_cols_bvars(
    tb_tool: tables.table,
    chunks: Tuple[int, ...],
    chan_cnt: int,
    ignore_msv2_cols,
    delayed_params: Tuple,
    bvars: Dict[str, xr.DataArray],
) -> None:
    """
    Loops over each column and create delayed dask arrays
    """

    col_names = tb_tool.colnames()
    for col in col_names:
        if (col in ignore_msv2_cols + ["TIME"]) or (not tb_tool.iscelldefined(col, 0)):
            continue
        if col not in bvars:
            bvars[col] = []

        cdata = tb_tool.getcol(col, 0, 1)[0]

        if len(cdata.shape) == 0:
            infile, ts_taql, (ctlen, cblen), tidxs, bidxs, didxs = delayed_params
            cshape = (ctlen, cblen)
            delayed_col = infile, ts_taql, col, cshape, tidxs, bidxs, didxs
            delayed_array = dask.delayed(read_col_chunk)(*delayed_col, None, None)
            bvars[col] += [dask.array.from_delayed(delayed_array, cshape, cdata.dtype)]

        elif col == "UVW":
            infile, ts_taql, (ctlen, cblen), tidxs, bidxs, didxs = delayed_params
            cshape = (ctlen, cblen, 3)
            delayed_3 = infile, ts_taql, col, cshape, tidxs, bidxs, didxs
            delayed_array = dask.delayed(read_col_chunk)(*delayed_3, None, None)
            bvars[col] += [dask.array.from_delayed(delayed_array, cshape, cdata.dtype)]

        elif len(cdata.shape) == 1:
            pol_list = []
            dd = 2 if cdata.shape == chan_cnt else 3
            for pc in range(0, cdata.shape[0], chunks[dd]):
                pols = (pc, min(cdata.shape[0], pc + chunks[dd]) - 1)
                infile, ts_taql, (ctlen, cblen), tidxs, bidxs, didxs = delayed_params
                cshape = (
                    ctlen,
                    cblen,
                ) + (pols[1] - pols[0] + 1,)
                delayed_cs = infile, ts_taql, col, cshape, tidxs, bidxs, didxs
                delayed_array = dask.delayed(read_col_chunk)(*delayed_cs, pols, None)
                pol_list += [
                    dask.array.from_delayed(delayed_array, cshape, cdata.dtype)
                ]
            bvars[col] += [dask.array.concatenate(pol_list, axis=2)]

        elif len(cdata.shape) == 2:
            chan_list = []
            for cc in range(0, cdata.shape[0], chunks[2]):
                chans = (cc, min(cdata.shape[0], cc + chunks[2]) - 1)
                pol_list = []
                for pc in range(0, cdata.shape[1], chunks[3]):
                    pols = (pc, min(cdata.shape[1], pc + chunks[3]) - 1)
                    (
                        infile,
                        ts_taql,
                        (ctlen, cblen),
                        tidxs,
                        bidxs,
                        didxs,
                    ) = delayed_params
                    cshape = (
                        ctlen,
                        cblen,
                    ) + (chans[1] - chans[0] + 1, pols[1] - pols[0] + 1)
                    delayed_cs = infile, ts_taql, col, cshape, tidxs, bidxs, didxs
                    delayed_array = dask.delayed(read_col_chunk)(
                        *delayed_cs, chans, pols
                    )
                    pol_list += [
                        dask.array.from_delayed(delayed_array, cshape, cdata.dtype)
                    ]
                chan_list += [dask.array.concatenate(pol_list, axis=3)]
            bvars[col] += [dask.array.concatenate(chan_list, axis=2)]


def concat_bvars_update_tvars(
    bvars: Dict[str, xr.DataArray], tvars: Dict[str, xr.DataArray]
) -> None:
    """
    concats all the dask chunks from each baseline. This is intended to
    be called iteratively, for every time chunk iteration, once all the
    baseline chunks have been read.
    """
    for kk in bvars.keys():
        if len(bvars[kk]) == 0:
            continue
        if kk not in tvars:
            tvars[kk] = []
        tvars[kk] += [dask.array.concatenate(bvars[kk], axis=1)]


def concat_tvars_to_mvars(
    dims: List[str], tvars: Dict[str, xr.DataArray], pol_cnt: int, chan_cnt: int
) -> Dict[str, xr.DataArray]:
    """
    Concat into a single dask array all the dask arrays from each time
    chunk to make the final arrays of the xds.

    :param dims: dimension names
    :param tvars: variables as lists of dask arrays per time chunk
    :param pol_cnt: len of pol axis/dim
    :param chan_cnt: len of freq axis/dim (chan indices)

    :return: variables as concated dask arrays
    """

    mvars = {}
    for tvr in tvars.keys():
        data_var = tvr.lower()
        if tvr == "UVW":
            mvars[data_var] = xr.DataArray(
                dask.array.concatenate(tvars[tvr], axis=0),
                dims=dims[:2] + ["uvw_coords"],
            )
        elif len(tvars[tvr][0].shape) == 3 and (tvars[tvr][0].shape[-1] == pol_cnt):
            mvars[data_var] = xr.DataArray(
                dask.array.concatenate(tvars[tvr], axis=0), dims=dims[:2] + ["pol"]
            )
        elif len(tvars[tvr][0].shape) == 3 and (tvars[tvr][0].shape[-1] == chan_cnt):
            mvars[data_var] = xr.DataArray(
                dask.array.concatenate(tvars[tvr], axis=0), dims=dims[:2] + ["freq"]
            )
        else:
            mvars[data_var] = xr.DataArray(
                dask.array.concatenate(tvars[tvr], axis=0),
                dims=dims[: len(tvars[tvr][0].shape)],
            )

    return mvars


def read_flat_main_table(
    infile: str,
    ddi: Union[int, None] = None,
    scan_state: Union[Tuple[int, int], None] = None,
    rowidxs=None,
    ignore_msv2_cols: Union[List, None] = None,
    chunks: Tuple[int, ...] = (22000, 512, 2),
) -> Tuple[xr.Dataset, Dict[str, Any], Dict[str, Any]]:
    """
    Read main table using flat structure: no baseline dimension. Vis
    dimensions are: row, freq, pol
    (experimental, perhaps to be deprecated/removed). Works but some
    features may be missing and/or flaky.

    Chunk tuple: (row, freq, pol)
    """
    taql_where = f"where DATA_DESC_ID = {ddi}"
    if scan_state:
        # TODO: support additional intent/scan/subscan conditions if
        # we keep this read_flat functionality
        _scans, states = scan_state
        # get row indices relative to full main table
        if states:
            taql_where += (
                f" AND SCAN_NUMBER = {scan_state[0]} AND STATE_ID = {scan_state[1]}"
            )
        else:
            taql_where += f" AND SCAN_NUMBER = {scan_state[0]}"

    mtable = tables.table(
        infile, readonly=True, lockoptions={"option": "usernoread"}, ack=False
    )

    # get row indices relative to full main table
    if rowidxs is None:
        taql_rowid = f"select rowid() as ROWS from $mtable {taql_where}"
        with open_query(mtable, taql_rowid) as query_rows:
            rowidxs = query_rows.getcol("ROWS")
            mtable.close()

    nrows = len(rowidxs)
    if nrows == 0:
        return xr.Dataset()

    part_ids = get_partition_ids(mtable, taql_where)

    taql_cols = f"select * from $mtable {taql_where}"
    with open_query(mtable, taql_cols) as query_cols:
        cols = query_cols.colnames()
        ignore = [
            col
            for col in cols
            if (not query_cols.iscelldefined(col, 0))
            or (query_cols.coldatatype(col) == "record")
        ]
        cdata = dict(
            [
                (col, query_cols.getcol(col, 0, 1))
                for col in cols
                if (col not in ignore)
                and (ignore_msv2_cols and (col not in ignore_msv2_cols))
            ]
        )
        chan_cnt, pol_cnt = [
            (cdata[cc].shape[1], cdata[cc].shape[2])
            for cc in cdata
            if len(cdata[cc].shape) == 3
        ][0]

    mtable.close()

    mvars, mcoords, bvars, xds = {}, {}, {}, xr.Dataset()
    # loop over row chunks
    for rc in range(0, nrows, chunks[0]):
        crlen = min(chunks[0], nrows - rc)  # chunk row length
        rcidxs = rowidxs[rc : rc + chunks[0]]

        # loop over each column and create delayed dask arrays
        for col in cdata.keys():
            if col not in bvars:
                bvars[col] = []

            if len(cdata[col].shape) == 1:
                delayed_array = dask.delayed(read_flat_col_chunk)(
                    infile, col, (crlen,), rcidxs, None, None
                )
                bvars[col] += [
                    dask.array.from_delayed(delayed_array, (crlen,), cdata[col].dtype)
                ]

            elif col == "UVW":
                delayed_array = dask.delayed(read_flat_col_chunk)(
                    infile, col, (crlen, 3), rcidxs, None, None
                )
                bvars[col] += [
                    dask.array.from_delayed(delayed_array, (crlen, 3), cdata[col].dtype)
                ]

            elif len(cdata[col].shape) == 2:
                pol_list = []
                dd = 1 if cdata[col].shape[1] == chan_cnt else 2
                for pc in range(0, cdata[col].shape[1], chunks[dd]):
                    plen = min(chunks[dd], cdata[col].shape[1] - pc)
                    delayed_array = dask.delayed(read_flat_col_chunk)(
                        infile, col, (crlen, plen), rcidxs, None, pc
                    )
                    pol_list += [
                        dask.array.from_delayed(
                            delayed_array, (crlen, plen), cdata[col].dtype
                        )
                    ]
                bvars[col] += [dask.array.concatenate(pol_list, axis=1)]

            elif len(cdata[col].shape) == 3:
                chan_list = []
                for cc in range(0, chan_cnt, chunks[1]):
                    clen = min(chunks[1], chan_cnt - cc)
                    pol_list = []
                    for pc in range(0, cdata[col].shape[2], chunks[2]):
                        plen = min(chunks[2], cdata[col].shape[2] - pc)
                        delayed_array = dask.delayed(read_flat_col_chunk)(
                            infile, col, (crlen, clen, plen), rcidxs, cc, pc
                        )
                        pol_list += [
                            dask.array.from_delayed(
                                delayed_array, (crlen, clen, plen), cdata[col].dtype
                            )
                        ]
                    chan_list += [dask.array.concatenate(pol_list, axis=2)]
                bvars[col] += [dask.array.concatenate(chan_list, axis=1)]

    # now concat all the dask chunks from each time to make the xds
    mvars = {}
    for kk in bvars.keys():
        # from uppercase MS col names to lowercase xds var names:
        data_var = kk.lower()
        if len(bvars[kk]) == 0:
            ignore += [kk]
            continue
        if kk == "UVW":
            mvars[data_var] = xr.DataArray(
                dask.array.concatenate(bvars[kk], axis=0), dims=["row", "uvw_coords"]
            )
        elif len(bvars[kk][0].shape) == 2 and (bvars[kk][0].shape[-1] == pol_cnt):
            mvars[data_var] = xr.DataArray(
                dask.array.concatenate(bvars[kk], axis=0), dims=["row", "pol"]
            )
        elif len(bvars[kk][0].shape) == 2 and (bvars[kk][0].shape[-1] == chan_cnt):
            mvars[data_var] = xr.DataArray(
                dask.array.concatenate(bvars[kk], axis=0), dims=["row", "chan"]
            )
        else:
            mvars[data_var] = xr.DataArray(
                dask.array.concatenate(bvars[kk], axis=0),
                dims=["row", "freq", "pol"][: len(bvars[kk][0].shape)],
            )

    mvars["time"] = xr.DataArray(
        convert_casacore_time(mvars["TIME"].values), dims=["row"]
    ).chunk({"row": chunks[0]})

    # add xds global attributes
    cc_attrs = extract_table_attributes(infile)
    attrs = {"other": {"msv2": {"ctds_attrs": cc_attrs, "bad_cols": ignore}}}
    # add per data var attributes
    mvars = add_units_measures(mvars, cc_attrs)
    mcoords = add_units_measures(mcoords, cc_attrs)

    mvars = rename_vars(mvars)
    mvars = redim_id_data_vars(mvars)
    xds = xr.Dataset(mvars, coords=mcoords)

    return xds, part_ids, attrs
