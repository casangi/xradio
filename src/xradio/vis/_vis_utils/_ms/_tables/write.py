import toolviper.utils.logger as logger, os
from typing import Tuple

import numpy as np
import xarray as xr

from casacore import tables


def revert_time(datetimes: np.ndarray) -> np.ndarray:
    """
    Convert time back from pandas datetime ref to casacore ref
    (reverse of read.convert_casacore_time).

    Parameters
    ----------
    datetimes : np.ndarray
        times in pandas reference

    Returns
    -------
    np.ndarray
        times converted to casacore reference

    """
    return (datetimes.astype(float) / 10**9) + 3506716800.0


#####################################
# translate numpy dtypes to casacore type strings
def type_converter(npdtype: str) -> str:
    cctype = "bad"
    if (npdtype == "int64") or (npdtype == "int32"):
        cctype = "int"
    elif npdtype == "bool":
        cctype = "bool"
    elif npdtype == "float32":
        cctype = "float"
    elif (npdtype == "float64") or (npdtype == "datetime64[ns]"):
        cctype = "double"
    elif npdtype == "complex64":
        cctype = "complex"
    elif npdtype == "complex128":
        cctype = "dcomplex"
    elif str(npdtype).startswith("<U"):
        cctype = "string"

    return cctype


####################################
# create and initialize new output table
def create_table(
    outfile: str,
    xds: xr.Dataset,
    max_rows: int,
    infile=None,
    cols=None,
    generic=False,
):
    if os.path.isdir(outfile):
        os.system("rm -fr %s" % outfile)

    # create column descriptions for table description
    ctds_attrs = {}
    try:
        ctds_attrs = xds.attrs["other"]["msv2"]["ctds_attrs"]
    except KeyError as exc:
        pass

    if cols is None:
        if ctds_attrs and "column_descriptions" in ctds_attrs:
            cols = {col: col for col in ctds_attrs["column_descriptions"]}
        else:
            cols = {var: var for var in xds.data_vars}
            # Would add all xds data vars regardless of description availability
            # +
            # list(xds.data_vars) +

    tabledesc = {}
    for col, var_name in cols.items():
        if ("column_descriptions" in ctds_attrs) and (
            col in ctds_attrs["column_descriptions"]
        ):
            coldesc = ctds_attrs["column_descriptions"][col]
            # col not in ignore_msv2_cols
            if (
                not generic
                and "DATA" in col
                and "shape" not in coldesc
                and var_name in xds.data_vars
            ):
                coldesc["shape"] = tuple(np.clip(xds[var_name].shape[1:], 1, None))

            if col == "UVW" or (
                (not "shape" in coldesc or type(coldesc["shape"]) == str)
                and var_name in xds.data_vars
            ):
                coldesc["shape"] = tuple(np.clip(xds[var_name].shape[1:], 1, None))
        else:
            coldesc = {"valueType": type_converter(xds[col].dtype)}
            if generic or (
                col == "UVW" or col == "DATA"
            ):  # will be statically shaped even if not originally
                coldesc = {"shape": tuple(np.clip(xds[col].shape[1:], 1, None))}
            elif xds[col].ndim > 1:  # make variably shaped
                coldesc = {"ndim": xds[col].ndim - 1}
            coldesc["name"] = col
            coldesc["desc"] = col
        tabledesc[col] = coldesc

        # fix the fun set of edge cases from casatestdata that cause errors
        if (
            "dataManagerType" in tabledesc[col]
            and tabledesc[col]["dataManagerType"] == "TiledShapeStMan"
        ) and (tabledesc[col]["ndim"] == 1):
            tabledesc[col]["dataManagerType"] = ""

    if generic:
        tb_tool = tables.table(
            outfile,
            tabledesc=tabledesc,
            nrow=max_rows,
            readonly=False,
            lockoptions={"option": "permanentwait"},
            ack=False,
        )
    else:
        tb_tool = tables.default_ms(outfile, tabledesc)
        tb_tool.addrows(max_rows)
        # if 'DATA_DESC_ID' in cols: tb_tool.putcol('DATA_DESC_ID',
        #    np.zeros((max_rows), dtype='int32') - 1, 0, max_rows)

    # write xds attributes to table keywords, skipping certain reserved attributes
    existing_keywords = tb_tool.getkeywords()
    for attr in ctds_attrs:
        if attr in [
            "other",
            "history",
            "info",
        ] + list(existing_keywords.keys()):
            continue
        tb_tool.putkeyword(attr, ctds_attrs[attr])
    if "info" in ctds_attrs:
        tb_tool.putinfo(ctds_attrs["info"])

    # copy subtables and add to main table
    if infile:
        subtables = [
            ss.path
            for ss in os.scandir(infile)
            if ss.is_dir() and ("SORTED_TABLE" not in ss.path)
        ]
        os.system("cp -r %s %s" % (" ".join(subtables), outfile))
        for subtable in subtables:
            if not tables.tableexists(
                os.path.join(outfile, subtable[subtable.rindex("/") + 1 :])
            ):
                continue
            sub_tbl = tables.table(
                os.path.join(outfile, subtable[subtable.rindex("/") + 1 :]),
                readonly=False,
                lockoptions={"option": "permanentwait"},
                ack=False,
            )
            tb_tool.putkeyword(
                subtable[subtable.rindex("/") + 1 :], sub_tbl, makesubrecord=True
            )
            sub_tbl.close()

    tb_tool.close()


############################################################################
##
## write_generic_table() - write any xds to generic casacore table format
##
############################################################################
def write_generic_table(xds: xr.Dataset, outfile: str, subtable="", cols=None):
    """
    Write generic xds contents back to casacore table format on disk

    Parameters
    ----------
    xds : xr.Dataset

    outfile : str

    subtable : str (Default value = "")

    cols : List[str] (Default value = None)

    Returns
    -------

    """
    outfile = os.path.expanduser(outfile)
    logger.debug("writing {os.path.join(outfile, subtable)}")

    try:
        ctds_attrs = {}
        ctds_attrs = xds.attrs["other"]["msv2"]["ctds_attrs"]
    except KeyError as exc:
        pass

    if cols is None:
        cols = {var.upper(): var for var in xds.data_vars}
        cols.update({coo.upper(): coo for coo in xds.coords if coo not in xds.dims})
        # Would add cols with a description regardless of presence in xds
        # + (
        #     list(
        #         ctds_attrs["column_descriptions"].keys()
        #         if "column_descriptions" in ctds_attrs
        #         else []
        #     )
        # )
    max_rows = xds.row.shape[0] if "row" in xds.dims else 0
    create_table(
        os.path.join(outfile, subtable),
        xds,
        max_rows,
        infile=None,
        cols=cols,
        generic=True,
    )

    tb_tool = tables.table(
        os.path.join(outfile, subtable),
        readonly=False,
        lockoptions={"option": "permanentwait"},
        ack=False,
    )
    try:
        for dv, col in cols.items():
            if (dv not in xds) or (np.prod(xds[dv].shape) == 0):
                continue
            values = (
                xds[dv].values
                if xds[dv].dtype != "datetime64[ns]"
                else revert_time(xds[dv].values)
            )
            tb_tool.putcol(col, values, 0, values.shape[0], 1)
    except Exception:
        print(
            "ERROR: exception in write generic table - %s, %s, %s, %s"
            % (os.path.join(outfile, subtable), dv, str(values.shape), tb_tool.nrows())
        )

    # now we have to add this subtable to the main table keywords (assuming a main table already exists)
    if len(subtable) > 0:
        main_tbl = tables.table(
            outfile, readonly=False, lockoptions={"option": "permanentwait"}, ack=False
        )
        main_tbl.putkeyword(subtable, tb_tool, makesubrecord=True)
        main_tbl.done()
    tb_tool.close()


###################################
# local helper
def write_main_table_slice(
    xda: xr.DataArray,
    outfile: str,
    ddi: int,
    col: str,
    full_shape: Tuple,
    starts: Tuple,
):
    """
    Write an xds row chunk to the corresponding main table slice

    Parameters
    ----------
    xda : xr.DataArray

    outfile : str

    ddi : int

    col : str

    full_shape : Tuple

    starts : Tuple


    Returns
    -------

    """
    # trigger the DAG for this chunk and return values while the table is unlocked
    values = xda.compute().values
    if xda.dtype == "datetime64[ns]":
        values = revert_time(values)

    tbs = tables.table(
        outfile, readonly=False, lockoptions={"option": "permanentwait"}, ack=True
    )

    # try:
    if (
        (values.ndim == 1) or (col == "UVW") or (values.shape[1:] == full_shape)
    ):  # scalar columns
        tbs.putcol(col, values, starts[0], len(values))
    else:
        if not tbs.iscelldefined(col, starts[0]):
            tbs.putcell(col, starts[0] + np.arange(len(values)), np.zeros((full_shape)))
        tbs.putcolslice(
            col,
            values,
            starts[1 : values.ndim],
            tuple(np.array(starts[1 : values.ndim]) + np.array(values.shape[1:]) - 1),
            [],
            starts[0],
            len(values),
            1,
        )
    # except:
    #    print("ERROR: write exception - %s, %s, %s" % (col, str(values.shape), str(starts)))
    tbs.close()
