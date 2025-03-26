import os
import time
from typing import Any, Optional, Union

import dask
import numpy as np
import xarray as xr

import toolviper.utils.logger as logger
from .write import write_generic_table, write_main_table_slice
from .write import create_table, revert_time

from casacore import tables


# TODO: this should be consolidated with the equivalent in read_main_table,
# if we keep this mapping
rename_to_msv2_cols = {
    "antenna1_id": "ANTENNA1",
    "antenna2_id": "ANTENNA2",
    "feed1_id": "FEED1",
    "feed2_id": "FEED2",
    # optional cols:
    # "WEIGHT": "WEIGHT_SPECTRUM",
    "VIS_CORRECTED": "CORRECTED_DATA",
    "VIS": "DATA",
    "VIS_MODEL": "MODEL_DATA",
    "AUTOCORR": "FLOAT_DATA",
}
# cols added in xds not in MSv2
cols_not_in_msv2 = ["baseline_ant1_id", "baseline_ant2_id"]


def cols_from_xds_to_ms(cols: list[str]) -> list[str]:
    """
    Translates between lowercase/uppercase convention
    Rename some MS_colum_names <-> xds_data_var_names
    Excludes the pointing_ vars that are in the xds but should not be written to MS
    """
    return {
        rename_to_msv2_cols.get(col, col).upper(): col
        for col in cols
        if (col and col not in cols_not_in_msv2 and not col.startswith("pointing_"))
    }


def flatten_xds(xds: xr.Dataset) -> xr.Dataset:
    """
    flatten (time, baseline) dimensions of xds back to single dimension (row)

    Parameters
    ----------
    xds : xr.Dataset


    Returns
    -------
    xr.Dataset
        Dataset in flat form (back to 'row' dimension as read by casacore tables)
    """
    txds = xds.copy()

    # flatten the time x baseline dimensions of main table
    if ("time" in xds.sizes) and ("baseline" in xds.sizes):
        txds = xds.stack({"row": ("time", "baseline")}).transpose("row", ...)
        # compute for issue https://github.com/hainegroup/oceanspy/issues/332
        # drop=True silently does compute (or at least used to)

        fill_value_int32 = get_pad_value(np.int32)
        txds = txds.where(
            (
                (txds.STATE_ID != fill_value_int32)
                & (txds.FIELD_ID != fill_value_int32)
            ).compute(),
            drop=True,
        )  # .unify_chunks()

        # re-assigning (implicitly dropping index coords) one by one produces
        # DeprecationWarnings: https://github.com/pydata/xarray/issues/6505
        astyped_data_vars = dict(xds.data_vars)
        for dv in list(txds.data_vars):
            if txds[dv].dtype != xds[dv].dtype:
                astyped_data_vars[dv] = txds[dv].astype(xds[dv].dtype)
            else:
                astyped_data_vars[dv] = txds[dv]

        flat_xds = xr.Dataset(astyped_data_vars, coords=txds.coords, attrs=txds.attrs)
        flat_xds = flat_xds.reset_index(["time", "baseline"])

    else:
        flat_xds = txds

    return flat_xds


def vis_xds_packager_mxds(
    partitions: dict[Any, xr.Dataset],
    subtables: list[tuple[str, xr.Dataset]],
    add_global_coords: bool = True,
) -> xr.Dataset:
    """
    Takes a dictionary of data partition xds datasets and a list of
    subtable xds datasets and packages them as a dataset of datasets
    (mxds)

    Parameters
    ----------
    partitions : dict[Any, xr.Dataset]
        data partiions as xds datasets
    subtables : list[tuple[str, xr.Dataset]]
        subtables as xds datasets
    add_global_coords: bool (Default value = True)
        whether to add coords to the output mxds

    Returns
    -------
    xr.Dataset
        A "mxds" - xr.dataset of datasets
    """
    mxds = xr.Dataset(attrs={"metainfo": subtables, "partitions": partitions})

    if add_global_coords:
        mxds = mxds.assign_coords(make_global_coords(mxds))

    return mxds


def make_global_coords(mxds: xr.Dataset) -> dict[str, xr.DataArray]:
    coords = {}
    metainfo = mxds.attrs["metainfo"]
    if "antenna" in metainfo:
        coords["antenna_name"] = metainfo["antenna"].antenna_name.values
        coords["antennas"] = xr.DataArray(
            metainfo["antenna"].antenna_name.values, dims=["antenna_name"]
        )
    if "field" in metainfo:
        coords["field_name"] = metainfo["field"].field_name.values
        coords["fields"] = xr.DataArray(
            metainfo["field"].field_name.values, dims=["field_name"]
        )
    if "feed" in mxds.attrs:
        coords["feed_ids"] = metainfo["feed"].FEED_ID.values
    if "observation" in metainfo:
        coords["observation_ids"] = metainfo["observation"].observation_id.values
        coords["observations"] = xr.DataArray(
            metainfo["observation"].PROJECT.values, dims=["observation_ids"]
        )
    if "polarization" in metainfo:
        coords["polarization_ids"] = metainfo["polarization"].pol_setup_id.values
    if "source" in metainfo:
        coords["source_ids"] = metainfo["source"].SOURCE_ID.values
        coords["sources"] = xr.DataArray(
            metainfo["source"].NAME.values, dims=["source_ids"]
        )
    if "spectral_window" in metainfo:
        coords["spw_ids"] = metainfo["spectral_window"].spw_id.values
    if "state" in metainfo:
        coords["state_ids"] = metainfo["state"].STATE_ID.values

    return coords


def write_ms(
    mxds: xr.Dataset,
    outfile: str,
    infile: str = None,
    subtables: bool = False,
    modcols: Union[list[str], None] = None,
    verbose: bool = False,
    execute: bool = True,
) -> Optional[list]:
    """
    Write ms format xds contents back to casacore MS (CTDS - casacore Table Data System) format on disk

    Parameters
    ----------
    mxds : xr.Dataset,
        Source multi-xarray dataset (originally created by read_ms)
    outfile : str
        Destination filename
    infile : Union[str, None] (Default value = None)
        Source filename to copy subtables from. Generally faster than reading/writing through mxds via the subtables parameter. Default None
        does not copy subtables to output.
    subtables : bool (Default value = False)
        Also write subtables from mxds. Default of False only writes mxds attributes that begin with xdsN to the MS main table.
        Setting to True will write all other mxds attributes to subtables of the main table.  This is probably going to be SLOW!
        Use infile instead whenever possible.
    modcols : Union[list[str], None] (Default value = None)
        list of strings indicating what column(s) were modified (aka xds data_vars). Different logic can be applied to speed up processing when
        a data_var has not been modified from the input. Default None assumes everything has been modified (SLOW)
    verbose : bool (Default value = False)
        Whether or not to print output progress. Since writes will typically execute the DAG, if something is
        going to go wrong, it will be here.  Default False
    execute : bool (Default value = True)
        Whether or not to actually execute the DAG, or just return it with write steps appended. Default True will execute it

    Returns
    -------
    Optional[list]
        delayed write functions
    """
    outfile = os.path.expanduser(outfile)
    if verbose:
        print("initializing output...")
    start = time.time()

    xds_list = [flatten_xds(xds) for _key, xds in mxds.partitions.items()]

    cols = cols_from_xds_to_ms(
        list(set([dv for dx in xds_list for dv in dx.data_vars]))
    )
    if modcols is None:
        modcols = cols

    # create an empty main table with enough space for all desired xds partitions
    # the first selected xds partition will be passed to create_table to provide a definition of columns and table keywords
    # we first need to add in additional keywords for the selected subtables that will be written as well
    max_rows = np.sum([dx.row.shape[0] for dx in xds_list])
    create_table(
        outfile, xds_list[0], max_rows=max_rows, infile=infile, cols=cols, generic=False
    )

    # start a list of dask delayed writes to disk (to be executed later)
    # the SPECTRAL_WINDOW, POLARIZATION, and DATA_DESCRIPTION tables must always be present and will always be written
    delayed_writes = [
        dask.delayed(write_generic_table)(
            mxds.metainfo["spectral_window"], outfile, "SPECTRAL_WINDOW", cols=None
        )
    ]
    delayed_writes += [
        dask.delayed(write_generic_table)(
            mxds.metainfo["polarization"], outfile, "POLARIZATION", cols=None
        )
    ]
    # should data_description be kept somewhere (in attrs?) or rebuilt?
    # delayed_writes += [
    #     dask.delayed(write_generic_table)(
    #         mxds.metainfo["data_description"], outfile, "DATA_DESCRIPTION", cols=None
    #     )
    # ]
    if subtables:  # also write the rest of the subtables
        for subtable in list(mxds.attrs.keys()):
            if (
                subtable.startswith("xds")
                or (subtable in ["spectral_window", "polarization", "data_description"])
                or not isinstance(subtable, xr.Dataset)
            ):
                continue

            if verbose:
                print("writing subtable %s..." % subtable)
            delayed_writes += [
                dask.delayed(write_generic_table)(
                    mxds.attrs[subtable], outfile, subtable, cols=None
                )
            ]

    ddi_row_start = 0  # output rows will be ordered by DDI
    for xds in xds_list:
        txds = xds.copy().unify_chunks()
        # TODO: carry over or rebuild?
        ddi = 0  # txds.data_desc_id[:1].values[0]

        # serial write entire DDI column first so subsequent delayed writes can find their spot
        if verbose:
            print("setting up DDI %i..." % ddi)

        # write each chunk of each modified data_var, triggering the DAG along the way
        for col in modcols:
            if col not in txds:
                continue  # this can happen with bad_cols, should still be created in create_table()

            if col in cols_not_in_msv2:
                continue

            chunks = txds[col].chunks
            dims = txds[col].dims
            for d0 in range(len(chunks[0])):
                d0start = ([0] + list(np.cumsum(chunks[0][:-1])))[d0]

                for d1 in range(len(chunks[1]) if len(chunks) > 1 else 1):
                    d1start = (
                        ([0] + list(np.cumsum(chunks[1][:-1])))[d1]
                        if len(chunks) > 1
                        else 0
                    )

                    for d2 in range(len(chunks[2]) if len(chunks) > 2 else 1):
                        d2start = (
                            ([0] + list(np.cumsum(chunks[2][:-1])))[d2]
                            if len(chunks) > 2
                            else 0
                        )

                        starts = [d0start, d1start, d2start]
                        lengths = [
                            chunks[0][d0],
                            (chunks[1][d1] if len(chunks) > 1 else 0),
                            (chunks[2][d2] if len(chunks) > 2 else 0),
                        ]
                        slices = [
                            slice(starts[0], starts[0] + lengths[0]),
                            slice(starts[1], starts[1] + lengths[1]),
                            slice(starts[2], starts[2] + lengths[2]),
                        ]
                        txda = txds[col].isel(
                            dict(zip(dims, slices)), missing_dims="ignore"
                        )
                        starts[0] = starts[0] + ddi_row_start  # offset to end of table
                        delayed_writes += [
                            dask.delayed(write_main_table_slice)(
                                txda,
                                outfile,
                                ddi=ddi,
                                col=col,
                                full_shape=txds[col].shape[1:],
                                starts=starts,
                            )
                        ]

        # now write remaining data_vars from the xds that weren't modified
        # this can be done faster by collapsing the chunking to maximum size (minimum #) possible
        max_chunk_size = np.prod(
            [txds.chunks[kk][0] for kk in txds.chunks if kk in ["row", "freq", "pol"]]
        )
        for col in list(np.setdiff1d(list(cols), modcols)):
            if col not in txds:
                continue  # this can happen with bad_cols, should still be created in create_table()

            if col in cols_not_in_msv2:
                continue

            col_chunk_size = np.prod([kk[0] for kk in txds[col].chunks])
            if max_chunk_size <= 0:
                max_chunk_size = 19200
            if col_chunk_size <= 0:
                col_rows = max_chunk_size
            else:
                col_rows = (
                    int(np.ceil(max_chunk_size / col_chunk_size))
                    * txds[col].chunks[0][0]
                )
            for rr in range(0, txds[col].row.shape[0], col_rows):
                txda = txds[col].isel(row=slice(rr, rr + col_rows))
                delayed_writes += [
                    dask.delayed(write_main_table_slice)(
                        txda,
                        outfile,
                        ddi=ddi,
                        col=rename_to_msv2_cols.get(col, col).upper(),
                        full_shape=txda.shape[1:],
                        starts=(rr + ddi_row_start,) + (0,) * (len(txda.shape) - 1),
                    )
                ]

        ddi_row_start += txds.row.shape[0]  # next xds will be appended after this one

    if execute:
        if verbose:
            print("triggering DAG...")
        zs = dask.compute(delayed_writes)
        if verbose:
            print(
                "execution time %0.2f sec. Compute result len: %d"
                % ((time.time() - start), len(zs))
            )
    else:
        if verbose:
            print("returning delayed task list")
        return delayed_writes


def calc_optimal_ms_chunk_shape(
    memory_available_in_bytes, shape, element_size_in_bytes, column_name
) -> int:
    """
    Calculates the max number of rows (1st dim in shape) of a variable
    that can be fit in the memory for a thread.

    Parameters
    ----------
    memory_available_in_bytes :

    shape :

    element_size_in_bytes :

    column_name :


    Returns
    -------
    int
    """
    factor = 0.8  # Account for memory used by other objects in thread.
    # total_mem = np.prod(shape)*element_size_in_bytes
    single_row_mem = np.prod(shape[1:]) * element_size_in_bytes

    if not single_row_mem < factor * memory_available_in_bytes:
        msg = (
            "Not engough memory in a thread to contain a row of "
            f"{column_name}. Need at least {single_row_mem / factor}"
            " bytes."
        )
        raise RuntimeError(msg)

    rows_chunk_size = int((factor * memory_available_in_bytes) / single_row_mem)

    if rows_chunk_size > shape[0]:
        rows_chunk_size = shape[0]

    logger.debug(
        "Numbers of rows in chunk for " + column_name + ": " + str(rows_chunk_size)
    )

    return rows_chunk_size


def write_ms_serial(
    mxds: xr.Dataset,
    outfile: str,
    infile: str = None,
    subtables: bool = False,
    verbose: bool = False,
    execute: bool = True,
    memory_available_in_bytes: int = 500000000000,
):
    """
    Write ms format xds contents back to casacore table format on disk

    Parameters
    ----------
    mxds : xr.Dataset
        Source multi-xarray dataset (originally created by read_ms)
    outfile : str
        Destination filename
    infile : str (Default value = None)
        Source filename to copy subtables from. Generally faster than reading/writing through mxds via the subtables parameter. Default None
        does not copy subtables to output.
    subtables : bool (Default value = False)
        Also write subtables from mxds. Default of False only writes mxds attributes that begin with xdsN to the MS main table.
        Setting to True will write all other mxds attributes to subtables of the main table.  This is probably going to be SLOW!
        Use infile instead whenever possible.
    verbose : bool (Default value = False)
        Whether or not to print output progress. Since writes will typically execute the DAG, if something is
        going to go wrong, it will be here.  Default False

    execute : bool (Default value = True)
        Whether or not to actually execute the DAG, or just return it with write steps appended. Default True will execute it
    memory_available_in_bytes : (Default value = 500000000000)

    Returns
    -------

    """

    print("*********************")
    outfile = os.path.expanduser(outfile)
    if verbose:
        print("initializing output...")
    # start = time.time()

    xds_list = [flatten_xds(xds) for _key, xds in mxds.partitions.items()]
    cols = list(set([dv for dx in xds_list for dv in dx.data_vars]))
    cols = cols_from_xds_to_ms(list(np.atleast_1d(cols)))

    # create an empty main table with enough space for all desired xds partitions
    # the first selected xds partition will be passed to create_table to provide a definition of columns and table keywords
    # we first need to add in additional keywords for the selected subtables that will be written as well
    max_rows = np.sum([dx.row.shape[0] for dx in xds_list])
    create_table(
        outfile, xds_list[0], max_rows=max_rows, infile=infile, cols=cols, generic=False
    )

    # start a list of dask delayed writes to disk (to be executed later)
    # the SPECTRAL_WINDOW, POLARIZATION, and DATA_DESCRIPTION tables must always be present and will always be written
    write_generic_table(
        mxds.metainfo["spectral_window"], outfile, "SPECTRAL_WINDOW", cols=None
    )
    write_generic_table(
        mxds.metainfo["polarization"], outfile, "POLARIZATION", cols=None
    )
    # should data_description be kept somewhere (in attrs?) or rebuilt?
    # write_generic_table(mxds.metainfo.data_description, outfile, "DATA_DESCRIPTION", cols=None)

    if subtables:  # also write the rest of the subtables
        # for subtable in list(mxds.attrs.keys()):
        #'OBSERVATION','STATE'
        # ['FEED','OBSERVATION','FIELD','ANTENNA','HISTORY','STATE']
        # ['FEED','FIELD','ANTENNA','HISTORY']
        # ,'FIELD','ANTENNA'
        # for subtable in ['OBSERVATION']:
        for subtable in list(mxds.metainfo.keys()):
            if subtable.startswith("xds") or (
                subtable in ["spectral_window", "polarization", "data_description"]
            ):
                continue
            if verbose:
                print("writing subtable %s..." % subtable)
            # print(subtable)
            # print(mxds.attrs[subtable])
            try:
                write_generic_table(
                    mxds.metainfo[subtable], outfile, subtable.upper(), cols=None
                )
            except (RuntimeError, KeyError) as exc:
                print(f"Exception writing subtable {subtable}: {exc}")

    part_key0 = next(iter(mxds.partitions))
    vis_data_shape = mxds.partitions[part_key0].VIS.shape
    rows_chunk_size = calc_optimal_ms_chunk_shape(
        memory_available_in_bytes, vis_data_shape, 16, "DATA"
    )

    # print(rows_chunk_size)
    # rows_chunk_size = 200000000
    # write each chunk of each modified data_var, triggering the DAG along the way
    tbs = tables.table(
        outfile, readonly=False, lockoptions={"option": "permanentwait"}, ack=True
    )

    start_main = time.time()
    for col, var_name in cols.items():
        xda = mxds.partitions[part_key0][var_name]
        # print(col,xda.dtype)

        for start_row in np.arange(0, vis_data_shape[0], rows_chunk_size):
            end_row = start_row + rows_chunk_size
            if end_row > vis_data_shape[0]:
                end_row = vis_data_shape[0]

            # start = time.time()
            values = xda[start_row:end_row,].compute().values
            if xda.dtype == "datetime64[ns]":
                values = revert_time(values)
            # print('1. Time', time.time()-start, values.shape)

            # start = time.time()
            try:
                tbs.putcol(col, values, start_row, len(values))
                # print('2. Time', time.time()-start)
            except RuntimeError as exc:
                print(f"Exception writing main table column {col}: {exc}")

    print("3. Time", time.time() - start_main)

    tbs.unlock()
    tbs.close()
