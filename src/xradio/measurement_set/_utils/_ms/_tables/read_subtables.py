import toolviper.utils.logger as logger
from pathlib import Path
from typing import Dict, Tuple, Union

import dask
import numpy as np
import pandas as pd
import xarray as xr

from casacore import tables

from .table_query import open_query, open_table_ro
from .read import (
    read_col_chunk,
    convert_casacore_time,
    extract_table_attributes,
    add_units_measures,
    table_exists,
    load_generic_table,
)
from .write import revert_time
from xradio._utils.list_and_array import unique_1d


def read_ephemerides(
    infile: str,
) -> Union[xr.Dataset, None]:
    """
    Read ephemerides info from MSv2 FIELD/EPHEMi_....tab subtables

    Parameters
    ----------
    infile : str
        path to MS

    Returns
    -------
    Union[xr.Dataset, None]
        ephemerides xds with metainfo as in the MSv3/EPHEMERIDES subtable
    """
    field_subt = Path(infile, "FIELD")
    subdirs = [
        sdir
        for sdir in field_subt.iterdir()
        if "EPHEM" in sdir.name and sdir.is_dir() and table_exists(str(sdir))
    ]
    ephem = []
    for sdir in subdirs:
        logger.debug(f"Reading ephemerides info from: FIELD / {sdir.name}")
        # One "EPHEM_*.tab" (each with a difference ephemeris_id) to concatenate
        ephem.append(
            load_generic_table(infile, str(Path(*sdir.parts[-2:])), timecols=["MJD"])
        )

    if ephem:
        ephem = xr.concat(ephem, dim="ephemeris_id")
    else:
        ephem = None

    return ephem


def read_delayed_pointing_table(
    infile: str,
    rename_ids: Dict[str, str] = None,
    chunks: Tuple = (10000, 100, 2, 20),
    time_slice=None,
) -> xr.Dataset:
    """
    Read MS pointing subtable in delayed arrays into an xr.Dataset

    Parameters
    ----------
    infile : str
        path to pointing table
    rename_ids : Dict[str, str] (Default value = None)
        dict with dimension renaming mapping
    chunks : Tuple (Default value = (10000, 100, 2, 20))
        chunks for the arrays. Chunks tuple: time, antenna, data_vars_dim_1, data_vars_dim_2
    time_slice: slice
        time bounds

    Returns
    -------
    xr.Dataset
        pointing dataset
    """

    with open_table_ro(infile) as mtable:
        taql_time = ""
        if time_slice:
            times = normalize_time_slice(mtable, time_slice)
            taql_time = f"where TIME BETWEEN {times.start} AND {times.stop}"
        else:
            times = None
        taql_all = f"select * from $mtable {taql_time}"
        with open_query(mtable, taql_all) as query_all:
            if query_all.nrows() == 0:
                mtable.close()
                note = ""
                if taql_time:
                    note = (
                        " within the selected time range: {times.start} - {times.stop}"
                    )
                logger.warning(f"POINTING subtable has no data{note}")
                return xr.Dataset()

            # pointing table uses time x antenna_id
            antennas = unique_1d(query_all.getcol("ANTENNA_ID", 0, -1))
            taql_times = f"select DISTINCT TIME from $mtable {taql_time}"
            with open_query(None, taql_times) as query_times:
                utimes = unique_1d(query_times.getcol("TIME", 0, -1))

            tvars = read_delayed_pointing_times(
                infile, antennas, utimes, chunks, query_all, times
            )

    dims = ["time", "antenna_id", "dim_2", "dim_3"]

    # now concat all the dask chunks from each time
    mvars = {}
    for var in tvars.keys():
        mvars[var] = xr.DataArray(
            dask.array.concatenate(tvars[var], axis=0),
            dims=dims[: len(tvars[var][0].shape)],
        )

    mcoords = {}
    mcoords["time"] = xr.DataArray(convert_casacore_time(utimes), dims=["time"])
    mcoords["antenna_id"] = xr.DataArray(np.arange(len(antennas)), dims=["antenna_id"])

    cc_attrs = extract_table_attributes(infile)
    attrs = {"other": {"msv2": {"ctds_attrs": cc_attrs}}}
    mvars = add_units_measures(mvars, cc_attrs)
    mcoords = add_units_measures(mcoords, cc_attrs)

    xds = xr.Dataset(mvars, coords=mcoords)
    if rename_ids:
        rename_ids = {k: v for k, v in rename_ids.items() if k in xds.sizes}
    xds = xds.rename_dims(rename_ids)
    xds = xds.assign_attrs(attrs)

    return xds


def normalize_time_slice(mtable: tables.table, time_slice: slice) -> slice:
    """
    If we get indices, produce the TIME column time value for the
    start/top indices. If we get timestamps, convert them to casacore
    refeference.

    Parameters
    ----------
    mtable : tables.table
        a casacore table from which we are reading a TIME
        column
    time_slice : slice
        slice giving start/stop time. Can be given as
        integer indices or as timestamps (Xarray / pandas reference)

    Returns
    -------
    slice
        a (start, stop) slice with times in casacore ref frame
    """
    if type(time_slice.start) == pd.Timestamp and type(time_slice.stop) == pd.Timestamp:
        # Add tol?
        eps = np.finfo(float).eps
        times = slice(
            revert_time(time_slice.start) - eps, revert_time(time_slice.stop) + eps
        )

    elif (
        int(time_slice.start) == time_slice.start
        and int(time_slice.stop) == time_slice.stop
    ):
        # instead of int cast could be operator.index(time_slice.start)
        taql_utimes = "select DISTINCT TIME from $mtable"
        with open_query(mtable, taql_utimes) as query_utimes:
            utimes = unique_1d(query_utimes.getcol("TIME", 0, -1))
            # add a tol around the time ranges returned by taql
            if len(utimes) < 2:
                tol = 1e-5
            else:
                tol = np.diff(utimes).min() / 4

        nutimes = len(utimes)
        if nutimes == 0:
            times = slice(0, 0)
        else:
            tidxs = slice(
                min(nutimes, int(time_slice.start)),
                min(nutimes, int(time_slice.stop)) - 1,
            )
            times = slice(utimes[tidxs.start] - tol, utimes[tidxs.stop] + tol)

    else:
        raise ValueError(
            f"Invalid time type. Not a timestamp and cannot use as"
            f" index: {time_slice.start} (type: {type(time_slice.start)})"
        )

    return times


def read_delayed_pointing_times(
    infile: str,
    antennas: np.ndarray,
    utimes: np.array,
    chunks: tuple,
    query_all: tables.table,
    time_slice: slice,
) -> Dict[str, xr.DataArray]:
    """
    Read pointing table in delayed time / antenna chunks. Loops over
    time chunks

    Parameters
    ----------
    infile : str
        path to pointing table
    antennas : np.ndarray
        antenna ids
    utimes : np.ndarray
        unique times from table
    chunks : tuple
        chunks for the arrays
    query_all : tables.table
        table to read columns
    time_slice: slice :
        time bounds

    Returns
    -------
    Dict[str, xr.DataArray]
        dictionary of columns=>variables (read as dask.delayed)
    """

    antenna_chunks = range(0, len(antennas), chunks[1])

    # loop over time chunks
    if time_slice:
        time_chunks = [0]
        logger.debug(
            f"reading single chunk from pointing, with times {time_slice.start} - {time_slice.stop}"
        )
    else:
        time_chunks = range(0, len(utimes), chunks[0])
        logger.debug(
            f"reading pointing table into {len(time_chunks)} time x {len(antenna_chunks)} antenna chunks"
        )

    tvars = {}
    for tc in time_chunks:
        bvars = read_delayed_pointing_chunks(
            infile, antennas, chunks, utimes, tc, query_all
        )

        # now concat all the dask chunks from each antenna
        for var in bvars.keys():
            if len(bvars[var]) == 0:
                continue
            if var not in tvars:
                tvars[var] = []
            tvars[var] += [dask.array.concatenate(bvars[var], axis=1)]

    return tvars


def read_delayed_pointing_chunks(
    infile: str,
    antennas: np.ndarray,
    chunks: tuple,
    utimes: np.ndarray,
    tc: int,
    tb_tool: tables.table,
) -> Dict[str, xr.DataArray]:
    """
    For one time chunk, read the baseline/antenna chunks. Loops over
    antenna_id chunks and reads all columns as dask.delayed calls.

    Parameters
    ----------
    infile : str
        path to pointing table
    antennas : np.ndarray
        antenna ids
    chunks : tuple
        chunks for the arrays
    utimes : np.ndarray
        unique times from table
    tc : int
        time index
    tb_tool : tables.table
        table to read columns

    Returns
    -------
    Dict[str, xr.DataArray]
        dictionary of columns=>variables (read as dask.delayed)
    """

    # add a tol around the time ranges returned by taql, for the next taql queries
    if len(utimes) < 2:
        tol = 1e-5
    else:
        tol = np.diff(utimes).min() / 4

    times = (
        utimes[tc] - tol,
        utimes[min(len(utimes) - 1, tc + chunks[0] - 1)] + tol,
    )
    ctlen = min(len(utimes), tc + chunks[0]) - tc  # chunk time length

    antenna_chunks = range(0, len(antennas), chunks[1])
    cols = tb_tool.colnames()

    bvars = {}
    for bc in antenna_chunks:
        blines = (
            antennas[bc],
            antennas[min(len(antennas) - 1, bc + chunks[1] - 1)],
        )
        cblen = min(len(antennas) - bc, chunks[1])

        # read the specified chunk of data
        ttql = "TIME BETWEEN %f and %f" % times
        atql = "ANTENNA_ID BETWEEN %i and %i" % blines
        ts_taql = f"select * from $mtable where {ttql} AND {atql}"
        with open_query(None, ts_taql) as ts_tb:
            tidxs = np.searchsorted(utimes, ts_tb.getcol("TIME", 0, -1)) - tc
            bidxs = np.searchsorted(antennas, ts_tb.getcol("ANTENNA_ID", 0, -1)) - bc
            didxs = np.arange(len(bidxs))

            # loop over each column and create delayed dask arrays
            for col in cols:
                if (col in ["TIME", "ANTENNA_ID"]) or (
                    not tb_tool.iscelldefined(col, 0)
                ):
                    continue
                if col not in bvars:
                    bvars[col] = []

                cdata = tb_tool.getcol(col, 0, 1)[0]
                if isinstance(cdata, str):
                    cdata = np.array(cdata)
                if len(cdata.shape) == 0:
                    delayed_array = dask.delayed(read_col_chunk)(
                        infile,
                        ts_taql,
                        col,
                        (ctlen, cblen),
                        tidxs,
                        bidxs,
                        didxs,
                        None,
                        None,
                    )
                    bvars[col] += [
                        dask.array.from_delayed(
                            delayed_array, (ctlen, cblen), cdata.dtype
                        )
                    ]

                elif len(cdata.shape) == 2:
                    d1_list = []
                    for cc in range(0, cdata.shape[0], chunks[2]):
                        d1s = (cc, min(cdata.shape[0], cc + chunks[2]) - 1)
                        d2_list = []
                        for pc in range(0, cdata.shape[1], chunks[3]):
                            d2s = (pc, min(cdata.shape[1], pc + chunks[3]) - 1)
                            cshape = (
                                ctlen,
                                cblen,
                            ) + (d1s[1] - d1s[0] + 1, d2s[1] - d2s[0] + 1)
                            delayed_array = dask.delayed(read_col_chunk)(
                                infile,
                                ts_taql,
                                col,
                                cshape,
                                tidxs,
                                bidxs,
                                didxs,
                                d1s,
                                d2s,
                            )
                            d2_list += [
                                dask.array.from_delayed(
                                    delayed_array, cshape, cdata.dtype
                                )
                            ]
                        d1_list += [dask.array.concatenate(d2_list, axis=3)]
                    bvars[col] += [dask.array.concatenate(d1_list, axis=2)]

    return bvars
