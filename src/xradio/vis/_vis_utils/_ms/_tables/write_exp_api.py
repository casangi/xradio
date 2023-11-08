import os, time
from typing import Optional

import dask
import numpy as np


from ..._utils.xds_helper import flatten_xds, calc_optimal_ms_chunk_shape
from .write import write_generic_table, write_main_table_slice
from .write import create_table, revert_time

from casacore import tables


def write_ms(
    mxds,
    outfile,
    infile=None,
    subtables=False,
    modcols=None,
    verbose=False,
    execute=True,
) -> Optional[list]:
    """
    Write ms format xds contents back to casacore MS (CTDS - casacore Table Data System) format on disk

    Parameters
    ----------
    mxds : xarray.Dataset
        Source multi-xarray dataset (originally created by read_ms)
    outfile : str
        Destination filename
    infile : str
        Source filename to copy subtables from. Generally faster than reading/writing through mxds via the subtables parameter. Default None
        does not copy subtables to output.
    subtables : bool
        Also write subtables from mxds. Default of False only writes mxds attributes that begin with xdsN to the MS main table.
        Setting to True will write all other mxds attributes to subtables of the main table.  This is probably going to be SLOW!
        Use infile instead whenever possible.
    modcols : list
        List of strings indicating what column(s) were modified (aka xds data_vars). Different logic can be applied to speed up processing when
        a data_var has not been modified from the input. Default None assumes everything has been modified (SLOW)
    verbose : bool
        Whether or not to print output progress. Since writes will typically execute the DAG, if something is
        going to go wrong, it will be here.  Default False
    execute : bool
        Whether or not to actually execute the DAG, or just return it with write steps appended. Default True will execute it
    """
    outfile = os.path.expanduser(outfile)
    if verbose:
        print("initializing output...")
    start = time.time()

    xds_list = [
        flatten_xds(mxds.attrs[kk]) for kk in mxds.attrs if kk.startswith("xds")
    ]
    cols = list(set([dv for dx in xds_list for dv in dx.data_vars]))
    if modcols is None:
        modcols = cols
    modcols = list(np.atleast_1d(modcols))

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
            mxds.SPECTRAL_WINDOW, outfile, "SPECTRAL_WINDOW", cols=None
        )
    ]
    delayed_writes += [
        dask.delayed(write_generic_table)(
            mxds.POLARIZATION, outfile, "POLARIZATION", cols=None
        )
    ]
    delayed_writes += [
        dask.delayed(write_generic_table)(
            mxds.DATA_DESCRIPTION, outfile, "DATA_DESCRIPTION", cols=None
        )
    ]
    if subtables:  # also write the rest of the subtables
        for subtable in list(mxds.attrs.keys()):
            if subtable.startswith("xds") or (
                subtable in ["SPECTRAL_WINDOW", "POLARIZATION", "DATA_DESCRIPTION"]
            ):
                continue
            if verbose:
                print("writing subtable %s..." % subtable)
            delayed_writes += [
                dask.delayed(write_generic_table)(
                    mxds.attrs[subtable], outfile, subtable, cols=None, verbose=verbose
                )
            ]

    ddi_row_start = 0  # output rows will be ordered by DDI
    for xds in xds_list:
        txds = xds.copy().unify_chunks()
        ddi = txds.data_desc_id[:1].values[0]

        # serial write entire DDI column first so subsequent delayed writes can find their spot
        if verbose:
            print("setting up DDI %i..." % ddi)

        # write each chunk of each modified data_var, triggering the DAG along the way
        for col in modcols:
            if col not in txds:
                continue  # this can happen with bad_cols, should still be created in create_table()
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
        for col in list(np.setdiff1d(cols, modcols)):
            if col not in txds:
                continue  # this can happen with bad_cols, should still be created in create_table()
            col_chunk_size = np.prod([kk[0] for kk in txds[col].chunks])
            col_rows = (
                int(np.ceil(max_chunk_size / col_chunk_size)) * txds[col].chunks[0][0]
            )
            for rr in range(0, txds[col].row.shape[0], col_rows):
                txda = txds[col].isel(row=slice(rr, rr + col_rows))
                delayed_writes += [
                    dask.delayed(write_main_table_slice)(
                        txda,
                        outfile,
                        ddi=ddi,
                        col=col,
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


def write_ms_serial(
    mxds,
    outfile,
    infile=None,
    subtables=False,
    verbose=False,
    execute=True,
    memory_available_in_bytes=500000000000,
):
    """
    Write ms format xds contents back to casacore table format on disk

    Parameters
    ----------
    mxds : xarray.Dataset
        Source multi-xarray dataset (originally created by read_ms)
    outfile : str
        Destination filename
    infile : str
        Source filename to copy subtables from. Generally faster than reading/writing through mxds via the subtables parameter. Default None
        does not copy subtables to output.
    subtables : bool
        Also write subtables from mxds. Default of False only writes mxds attributes that begin with xdsN to the MS main table.
        Setting to True will write all other mxds attributes to subtables of the main table.  This is probably going to be SLOW!
        Use infile instead whenever possible.
    modcols : list
        List of strings indicating what column(s) were modified (aka xds data_vars). Different logic can be applied to speed up processing when
        a data_var has not been modified from the input. Default None assumes everything has been modified (SLOW)
    verbose : bool
        Whether or not to print output progress. Since writes will typically execute the DAG, if something is
        going to go wrong, it will be here.  Default False
    execute : bool
        Whether or not to actually execute the DAG, or just return it with write steps appended. Default True will execute it
    """

    print("*********************")
    outfile = os.path.expanduser(outfile)
    if verbose:
        print("initializing output...")
    # start = time.time()

    xds_list = [
        flatten_xds(mxds.attrs[kk]) for kk in mxds.attrs if kk.startswith("xds")
    ]
    cols = list(set([dv for dx in xds_list for dv in dx.data_vars]))
    cols = list(np.atleast_1d(cols))

    # create an empty main table with enough space for all desired xds partitions
    # the first selected xds partition will be passed to create_table to provide a definition of columns and table keywords
    # we first need to add in additional keywords for the selected subtables that will be written as well
    max_rows = np.sum([dx.row.shape[0] for dx in xds_list])
    create_table(
        outfile, xds_list[0], max_rows=max_rows, infile=infile, cols=cols, generic=False
    )

    # start a list of dask delayed writes to disk (to be executed later)
    # the SPECTRAL_WINDOW, POLARIZATION, and DATA_DESCRIPTION tables must always be present and will always be written
    write_generic_table(mxds.SPECTRAL_WINDOW, outfile, "SPECTRAL_WINDOW", cols=None)
    write_generic_table(mxds.POLARIZATION, outfile, "POLARIZATION", cols=None)
    write_generic_table(mxds.DATA_DESCRIPTION, outfile, "DATA_DESCRIPTION", cols=None)

    if subtables:  # also write the rest of the subtables
        # for subtable in list(mxds.attrs.keys()):
        #'OBSERVATION','STATE'
        # ['FEED','OBSERVATION','FIELD','ANTENNA','HISTORY','STATE']
        # ['FEED','FIELD','ANTENNA','HISTORY']
        # ,'FIELD','ANTENNA'
        # for subtable in ['OBSERVATION']:
        for subtable in list(mxds.attrs.keys()):
            if subtable.startswith("xds") or (
                subtable in ["SPECTRAL_WINDOW", "POLARIZATION", "DATA_DESCRIPTION"]
            ):
                continue
            if verbose:
                print("writing subtable %s..." % subtable)
            # print(subtable)
            # print(mxds.attrs[subtable])
            write_generic_table(
                mxds.attrs[subtable], outfile, subtable, cols=None, verbose=verbose
            )

    vis_data_shape = mxds.xds0.data.shape
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
    for col in cols:
        xda = mxds.xds0[col]
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
            tbs.putcol(col, values, start_row, len(values))
            # print('2. Time', time.time()-start)

    print("3. Time", time.time() - start_main)

    tbs.unlock()
    tbs.close()
