import graphviper.utils.logger as logger
import os
from pathlib import Path
import re
from typing import Any, List, Dict, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr

import astropy.units
from casacore import tables

from .table_query import open_query, open_table_ro

CASACORE_TO_PD_TIME_CORRECTION = 3506716800.0
SECS_IN_DAY = 86400


def table_exists(path: str) -> bool:
    return tables.tableexists(path)


def convert_casacore_time(rawtimes: np.ndarray, convert_to_datetime=True) -> np.ndarray:
    """
    Read time columns to datetime format
    pandas datetimes are referenced against a 0 of 1970-01-01
    CASA's modified julian day reference time is (of course) 1858-11-17

    This requires a correction of 3506716800 seconds which is hardcoded to save time

    :param rawtimes: times in casacore ref
    :return: times converted to pandas reference
    """

    if convert_to_datetime:
        return pd.to_datetime(
            np.array(rawtimes) - CASACORE_TO_PD_TIME_CORRECTION, unit="s"
        ).values
    else:
        return np.array(rawtimes) - CASACORE_TO_PD_TIME_CORRECTION
    # dt = pd.to_datetime(np.atleast_1d(rawtimes) - correction, unit='s').values
    # if len(np.array(rawtimes).shape) == 0: dt = dt[0]
    # return dt


def convert_mjd_time(rawtimes: np.ndarray) -> np.ndarray:
    """Different time conversion needed for the MJD col of EPHEM{i}_*.tab
    files (only, as far as I've seen)

    :param rawtimes: MJD times for example from the MJD col of ephemerides tables
    :return: times converted to pandas reference and datetime type
    """
    return pd.to_datetime(
        rawtimes * SECS_IN_DAY - CASACORE_TO_PD_TIME_CORRECTION, unit="s"
    ).values


def extract_table_attributes(infile: str) -> Dict[str, Dict]:
    """
    return a dictionary of table attributes created from MS keywords and column descriptions
    """
    with open_table_ro(infile) as tb_tool:
        kwd = tb_tool.getkeywords()
        attrs = dict([(kk, kwd[kk]) for kk in kwd if kk not in os.listdir(infile)])
        cols = tb_tool.colnames()
        column_descriptions = {}
        for col in cols:
            column_descriptions[col] = tb_tool.getcoldesc(col)
        attrs["column_descriptions"] = column_descriptions
        attrs["info"] = tb_tool.info()

    return attrs


def add_units_measures(
    mvars: Dict[str, xr.DataArray], cc_attrs: Dict[str, Any]
) -> Dict[str, xr.DataArray]:
    """
    Add attributes with units and measure metainfo to the variables passed in the input dictionary

    :param mvars: data variables where to populate units
    :param cc_attrs: dictionary with casacore table attributes (from extract_table_attributes)
    :return: variables with units added in their attributes
    """
    col_descrs = cc_attrs["column_descriptions"]
    # TODO: Should probably loop the other way around, over mvars
    for col in col_descrs:
        var_name = col.lower()
        if var_name in mvars and "keywords" in col_descrs[col]:
            if "QuantumUnits" in col_descrs[col]["keywords"]:
                cc_units = col_descrs[col]["keywords"]["QuantumUnits"]
                if not isinstance(cc_units, list) or not cc_units:
                    logger.warning(
                        f"Invalid units found for column/variable {col}: {cc_units}"
                    )
                mvars[var_name].attrs["units"] = cc_units[0]
                try:
                    astropy.units.Unit(cc_units[0])
                except Exception as exc:
                    logger.warning(
                        f"Unsupported units found for column/variable {col}: "
                        f"{cc_units}. Cannot create an astropy.units.Units object from it: {exc}"
                    )

            if "MEASINFO" in col_descrs[col]["keywords"]:
                cc_meas = col_descrs[col]["keywords"]["MEASINFO"]
                mvars[var_name].attrs["measure"] = {"type": cc_meas["type"]}
                # VarRefCol used for several cols of:
                # - SPECTRAL_WINDOW (MEAS_FREG_REF, in MSv2 std)
                # - FIELD  (PhseDir_Ref, DelayDir_Ref, RefDir_Ref, not in MSv2 std)
                # - POINTING - to be split
                if "VarRefCol" not in cc_meas:
                    mvars[var_name].attrs["measure"]["ref_frame"] = cc_meas["Ref"]
                else:
                    mvars[var_name].attrs["measure"]["ref_frame_data_var"] = cc_meas[
                        "VarRefCol"
                    ]
                    if "TabRefTypes" in cc_meas:
                        mvars[var_name].attrs["measure"].update(
                            {
                                "ref_frame_types": list(cc_meas["TabRefTypes"]),
                                "ref_frame_codes": list(cc_meas["TabRefCodes"]),
                            }
                        )

    return mvars


def make_freq_attrs(spw_xds: xr.Dataset, spw_id: int) -> Dict[str, Any]:
    """Grab the units/measure metainfo for the xds.freq dimension of a
    parttion from the SPECTRAL_WINDOW subtable CTDS attributes.

    Has to read xds_spw.meas_freq_ref and use it as index in the CTDS
    'VarRefCol' attrs of CHAN_FREQ and REF_FREQUENCY to give a
    reference frame to xds_spw.ref_frequency and xds_spw.chan_freq
    (then the ref frame from the second will be pulled to
    xds.freq.attrs)

    :param spw_xds: (metainfo) SPECTRAL_WINDOW xds
    :param spw_id: SPW id of a partition
    :return: attributes (units/measure) for the freq dim of a partition

    """
    fallback_TabRefTypes = [
        "REST",
        "LSRK",
        "LSRD",
        "BARY",
        "GEO",
        "TOPO",
        "GALACTO",
        "LGROUP",
        "CMB",
    ]

    ctds_cols = spw_xds.attrs["other"]["msv2"]["ctds_attrs"]["column_descriptions"]
    cfreq = ctds_cols["CHAN_FREQ"]

    cf_attrs = spw_xds.chan_freq.attrs
    if "MEASINFO" in cfreq["keywords"] and "VarRefCol" in cfreq["keywords"]["MEASINFO"]:
        fattrs = cfreq["keywords"]["MEASINFO"]
        var_ref_col = fattrs["VarRefCol"]
        # This should point to the SPW/MEAS_FREQ_REF col
        meas_freq_ref_idx = spw_xds.data_vars[var_ref_col.lower()].values[spw_id]

        if "TabRefCodes" not in fattrs or "TabRefTypes" not in fattrs:
            # Datasets like vla/ic2233_1.ms say "VarRefCol" but "TabRefTypes" is missing
            ref_frame = fallback_TabRefTypes[meas_freq_ref_idx]
        else:
            ref_type_code = fattrs["TabRefCodes"][meas_freq_ref_idx]
            ref_frame = fattrs["TabRefTypes"][ref_type_code]

        cf_attrs["measure"] = {
            "type": fattrs["type"],
            "ref_frame": ref_frame,
        }

        # Also set the 'VarRefCol' for CHAN_FREQ and REF_FREQUENCEY
        spw_xds.data_vars["chan_freq"].attrs.update(cf_attrs)
        spw_xds.data_vars["ref_frequency"].attrs.update(cf_attrs)

    return cf_attrs


def get_pad_nan(col: np.ndarray) -> np.ndarray:
    """
    Produce a padding/nan value appropriate for a data column
    (for when we need to pad data vars coming from columns with rows of
    variable size array values)

    Parameters
    ----------
    col : np.ndarray
        data being loaded from a table column
    """
    # This is causing frequent warnings for integers. Cast of nan to "int nan"
    # produces -2147483648 but also seems to trigger a
    # "RuntimeWarning: invalid value encountered in cast" (new in numpy>=1.24)
    policy = "warn"
    col_type = np.array(col).dtype
    if np.issubdtype(col_type, np.integer):
        policy = "ignore"
    with np.errstate(invalid=policy):
        pad_nan = np.array([np.nan]).astype(col_type)[0]

    return pad_nan


def redimension_ms_subtable(xds: xr.Dataset, subt_name: str) -> xr.Dataset:
    """Expand a MeasurementSet subtable xds from single dimension (row)
    to multiple dimensions (such as (source_id, time, spectral_window)

    WIP: only works for source, experimenting

    :param xds: dataset to change the dimensions
    :param subt_name: subtable name (SOURCE, etc.)
    :return: transformed xds with data dimensions representing the MS subtable key
    (one dimension for every columns)
    """
    subt_key_cols = {
        "DOPPLER": ["doppler_id", "source_id"],
        "FREQ_OFFSET": [
            "antenna1",
            "antenna2",
            "feed_id",
            "spectral_window_id",
            "time",
        ],
        "POINTING": ["time", "antenna_id"],
        "SOURCE": ["source_id", "time", "spectral_window_id"],
        "SYSCAL": ["antenna_id", "feed_id", "spectral_window_id", "time"],
        "WEATHER": ["antenna_id", "time"],
        # added tables (MSv3 but not preent in MSv2). Build it from "EPHEMi_... tables
        # Not clear what to do about 'time' var/dim:  , "time"],
        "EPHEMERIDES": ["ephemeris_row_id", "ephemeris_id"],
    }
    key_dims = subt_key_cols[subt_name]

    rxds = xds.copy()
    try:
        with np.errstate(invalid="ignore"):
            rxds = rxds.set_index(row=key_dims).unstack("row").transpose(*key_dims, ...)
        # unstack changes type to float when it needs to introduce NaNs, so
        # we need to reset to the original type.
        for var in rxds.data_vars:
            if rxds[var].dtype != xds[var].dtype:
                rxds[var] = rxds[var].astype(xds[var].dtype)
    except Exception as exc:
        logger.warning(
            f"Cannot expand rows to {key_dims}, possibly duplicate values in those coordinates. Exception: {exc}"
        )
        rxds = xds.copy()

    return rxds


def is_ephem_subtable(tname: str) -> bool:
    return "EPHEM" in tname and Path(tname).name.startswith("EPHEM")


def add_ephemeris_vars(tname: str, xds: xr.Dataset) -> xr.Dataset:
    fname = Path(tname).name
    pattern = r"EPHEM(\d+)_"
    match = re.match(pattern, fname)
    if match:
        ephem_id = match.group(1)
    else:
        ephem_id = 0

    xds["ephemeris_id"] = np.uint32(ephem_id) * xr.ones_like(
        xds["mjd"], dtype=np.uint32
    )
    xds = xds.rename({"mjd": "time"})
    xds["ephemeris_row_id"] = (
        xr.zeros_like(xds["time"], dtype=np.uint32) + xds["row"].values
    )

    return xds


def is_nested_ms(attrs: Dict) -> bool:
    ctds_attrs = attrs["other"]["msv2"]["ctds_attrs"]
    return (
        "MS_VERSION" in ctds_attrs
        and "column_descriptions" in ctds_attrs
        and all(
            col in ctds_attrs["column_descriptions"]
            for col in (
                "UVW",
                "ANTENNA1",
                "ANTENNA2",
                "FEED1",
                "FEED2",
                "OBSERVATION_ID",
            )
        )
    )


def read_generic_table(
    inpath: str,
    tname: str,
    timecols: Union[List[str], None] = None,
    ignore: Union[List[str], None] = None,
    rename_ids: Dict[str, str] = None,
    taql_where: str = None,
) -> xr.Dataset:
    """
    load generic casacore (sub)table into memory resident xds (xarray wrapped
    numpy arrays). This reads through the table columns and loads the data.

    TODO: change read_ name to load_ name (and most if not all this module)

    Parameters
    ----------
    :param inpath: path to the MS or directory containing the table
    :param tname: (sub)table name, for example 'SOURCE' for myms.ms/SOURCE

    :param timecols: column names to convert to numpy datetime format.
    leaves times as their original casacore format.
    :param ignore: list of column names to ignore and not try to read.
    :rename_ids: dict with dimension renaming mapping
    :taql_where: TaQL string to optionally constain the rows/columns to read

    :return: table loaded as XArray dataset
    """
    if timecols is None:
        timecols = []
    if ignore is None:
        ignore = []

    infile = Path(inpath, tname)
    infile = str(infile.expanduser())
    if not os.path.isdir(infile):
        raise ValueError(
            f"invalid input filename to read_generic_table: {infile} table {tname}"
        )

    cc_attrs = extract_table_attributes(infile)
    attrs: Dict[str, Any] = {"other": {"msv2": {"ctds_attrs": cc_attrs}}}
    if is_nested_ms(attrs):
        logger.warning(
            f"Skipping subtable that looks like a MeasurementSet main table: {inpath} {tname}"
        )
        return xr.Dataset()

    with open_table_ro(infile) as gtable:
        if gtable.nrows() == 0:
            logger.debug(f"table is empty: {inpath} {tname}")
            return xr.Dataset(attrs=attrs)

        taql_gtable = f"select * from $gtable {taql_where}"
        with open_query(gtable, taql_gtable) as tb_tool:
            if tb_tool.nrows() == 0:
                logger.debug(
                    f"table query is empty: {inpath} {tname}, with where {taql_where}"
                )
                return xr.Dataset(attrs=attrs)

            colnames = tb_tool.colnames()
            mcoords, mvars = load_cols_into_coords_data_vars(
                infile, tb_tool, timecols, ignore
            )

    mvars = add_units_measures(mvars, cc_attrs)
    mcoords = add_units_measures(mcoords, cc_attrs)

    xds = xr.Dataset(mvars, coords=mcoords)

    dim_prefix = "dim"
    dims = ["row"] + [f"{dim_prefix}_{i}" for i in range(1, 20)]
    xds = xds.rename(dict([(dv, dims[di]) for di, dv in enumerate(xds.sizes)]))
    if rename_ids:
        rename_ids = {k: v for k, v in rename_ids.items() if k in xds.sizes}
    xds = xds.rename_dims(rename_ids)

    attrs["other"]["msv2"]["bad_cols"] = list(
        np.setdiff1d(
            [dv for dv in colnames],
            [dv for dv in list(xds.data_vars) + list(xds.coords)],
        )
    )

    if tname in ["DOPPLER", "FREQ_OFFSET", "POINTING", "SOURCE", "SYSCAL", "WEATHER"]:
        xds = redimension_ms_subtable(xds, tname)

    if is_ephem_subtable(tname):
        xds = add_ephemeris_vars(tname, xds)
        xds = redimension_ms_subtable(xds, "EPHEMERIDES")

    xds = xds.assign_attrs(attrs)

    return xds


def load_cols_into_coords_data_vars(
    inpath: str,
    tb_tool: tables.table,
    timecols: Union[List[str], None] = None,
    ignore: Union[List[str], None] = None,
) -> Tuple[Dict[str, xr.Dataset], Dict[str, xr.Dataset]]:
    """
    Produce a set of coordinate xarrays and a set of data variables xarrays
    from the columns of a table.
    """
    columns_loader = find_best_col_loader(inpath, tb_tool.nrows())

    mcoords, mvars = columns_loader(inpath, tb_tool, timecols, ignore)

    return mcoords, mvars


def find_best_col_loader(inpath: str, nrows: int):
    """
    Simple heuristic: for any tables other than POINTING, use the generic_load_cols
    function that is able to deal with variable size columns. For POINTING (and if it has
    more rows than an arbitrary "small" threshold) use a more efficient load function that
    loads the data by column (but is not able to deal with any generic table).
    For now, all other subtables are loaded using the generic column loader.

    Background: the POINTING subtable can have a very large number of rows. For example in
    ALMA it is sampled at ~50ms intervals which typically produces of the order of
    [10^5, 10^7] rows. This becomes a serious performance bottleneck when loading the
    table using row() (and one dict allocated per row).
    This function chooses an alternative "by-column" load function to load in the columns
    when the table is POINTING. See xradio issue #128 for now this distinction is made
    solely for performance reasons.

    :param inpath: path name of the MS table
    :param nrows: number of rows found in the table

    :return: function best suited to load the data from the columns of this table
    """
    # do not give up generic by-row() loading if nrows is (arbitrary) small
    ARBITRARY_MIN_ROWS = 1000

    if inpath.endswith("POINTING") and nrows >= ARBITRARY_MIN_ROWS:
        columns_loader = load_fixed_size_cols
    else:
        columns_loader = load_generic_cols

    return columns_loader


def load_generic_cols(
    inpath: str,
    tb_tool: tables.table,
    timecols: Union[List[str], None],
    ignore: Union[List[str], None],
) -> Tuple[Dict[str, xr.Dataset], Dict[str, xr.Dataset]]:
    """Loads data for each MS column (loading the data in memory) into Xarray datasets

    This function is generic in that it can load variable size array columns. See also
    load_fixed_size_cols() as a simpler and much better performing alternative
    for tables that are large and expected/guaranteed to not have columns with variable
    size cells.

    :param inpath: path name of the MS table
    :param tb_tool: table to load the columns
    :param timecols: columns names to convert to datetime format
    :param ignore: list of column names to skip and not try to load.

    :return: dict of coordinates and dict of data vars.
    """

    col_cells = find_loadable_filled_cols(tb_tool, ignore)

    trows = tb_tool.row(ignore, exclude=True)[:]

    # Produce coords and data vars from MS columns
    mcoords, mvars = {}, {}
    for col in col_cells.keys():
        try:
            # TODO
            # benchmark np.stack() performance
            data = np.stack(
                [row[col] for row in trows]
            )  # .astype(col_cells[col].dtype)
            if isinstance(trows[0][col], dict):
                # TODO
                # benchmark np.stack() performance
                data = np.stack(
                    [
                        (
                            row[col]["array"].reshape(row[col]["shape"])
                            if len(row[col]["array"]) > 0
                            else np.array([""])
                        )
                        for row in trows
                    ]
                )
        except Exception:
            # sometimes the cols are variable, so we need to standardize to the largest sizes

            if len(set([isinstance(row[col], dict) for row in trows])) > 1:
                continue  # can't deal with this case

            data = handle_variable_col_issues(inpath, col, col_cells, trows)

        if len(data) == 0:
            continue

        array_type, array_data = raw_col_data_to_coords_vars(
            inpath, tb_tool, col, data, timecols
        )
        if array_type == "coord":
            mcoords[col.lower()] = array_data
        elif array_type == "data_var":
            mvars[col.lower()] = array_data

    return mcoords, mvars


def load_fixed_size_cols(
    inpath: str,
    tb_tool: tables.table,
    timecols: Union[List[str], None],
    ignore: Union[List[str], None],
) -> Tuple[Dict[str, xr.Dataset], Dict[str, xr.Dataset]]:
    """
    Loads columns into memory via the table tool getcol() function, as opposed to
    load_generic_cols() which loads on a per-row basis via row().
    This function is 2+ orders of magnitude faster for large tables (pointing tables with
    the order of >=10^5 rows)
    Prefer this function for performance reasons when all rows can be assumed to be fixed
    size (even if they are of array type).
    This is performance-critical for the POINTING subtable.

    :param inpath: path name of the MS
    :param tb_tool: table to red the columns
    :param timecols: columns names to convert to datetime format
    :param ignore: list of column names to skip and not try to load.

    :return: dict of coordinates and dict of data vars, ready to construct
    an xr.Dataset
    """

    loadable_cols = find_loadable_filled_cols(tb_tool, ignore)

    # Produce coords and data vars from MS columns
    mcoords, mvars = {}, {}
    for col in loadable_cols.keys():
        try:
            data = tb_tool.getcol(col)
            if isinstance(data, dict):
                data = data["array"].reshape(data["shape"])
        except Exception as exc:
            logger.warning(
                f"{inpath}: failed to load data with getcol for column {col}: {exc}"
            )
            data = []

        if len(data) == 0:
            continue

        array_type, array_data = raw_col_data_to_coords_vars(
            inpath, tb_tool, col, data, timecols
        )
        if array_type == "coord":
            mcoords[col.lower()] = array_data
        elif array_type == "data_var":
            mvars[col.lower()] = array_data

    return mcoords, mvars


def find_loadable_filled_cols(tb_tool: tables.table, ignore: Union[List[str], None]):
    """
    For a table, finds the columns that are:
    - loadable = not of record type, and not to be ignored
    - filled = the column cells are populated.

    :param tb_tool: table to red the columns
    :param ignore: list of column names to skip and not try to load.

    :return: dict of {column name => first cell} for columns that can/should be loaded
    """

    colnames = tb_tool.colnames()
    # columns that are not populated are skipped. record columns are not supported
    loadable_cols = {
        col: tb_tool.getcell(col, 0)
        for col in colnames
        if (col not in ignore)
        and (tb_tool.iscelldefined(col, 0))
        and tb_tool.coldatatype(col) != "record"
    }
    return loadable_cols


def raw_col_data_to_coords_vars(
    inpath: str,
    tb_tool: tables.table,
    col: str,
    data: np.ndarray,
    timecols: Union[List[str], None],
) -> Tuple[str, xr.DataArray]:
    """
    From a raw np array of data (freshly loaded from a table column), prepares either a
    coord or a data_var ready to be added to an xr.Dataset

    :return: whether this column is a 'coord' or a 'data_var', DataArray
    with column data/coord values ready to be added to the table xds
    """

    # Almost sure that when TIME is present (in a standard MS subt) it
    # is part of the key. But what about non-std subtables, ASDM subts?
    subts_with_time_key = (
        "FEED",
        "FLAG_CMD",
        "FREQ_OFFSET",
        "HISTORY",
        "POINTING",
        "SOURCE",
        "SYSCAL",
        "WEATHER",
    )
    dim_prefix = "dim"

    if col in timecols:
        if col == "MJD":
            data = convert_mjd_time(data)
        else:
            try:
                data = convert_casacore_time(data)
            except pd.errors.OutOfBoundsDatetime as exc:
                if inpath.endswith("WEATHER"):
                    logger.error(
                        f"Exception when converting WEATHER/TIME: {exc}. TIME data: {data}"
                    )
                else:
                    raise
    # should also probably add INTERVAL not only TIME
    if col.endswith("_ID") or (inpath.endswith(subts_with_time_key) and col == "TIME"):
        # weather table: importasdm produces very wrong "-1" ANTENNA_ID
        if (
            inpath.endswith("WEATHER")
            and col == "ANTENNA_ID"
            and "NS_WX_STATION_ID" in tb_tool.colnames()
        ):
            data = tb_tool.getcol("NS_WX_STATION_ID")

        array_type = "coord"
        array_data = xr.DataArray(
            data,
            dims=[
                f"{dim_prefix}_{di}_{ds}" for di, ds in enumerate(np.array(data).shape)
            ],
        )
    else:
        array_type = "data_var"
        array_data = xr.DataArray(
            data,
            dims=[
                f"{dim_prefix}_{di}_{ds}" for di, ds in enumerate(np.array(data).shape)
            ],
        )

    return array_type, array_data


def handle_variable_col_issues(
    inpath: str, col: str, col_cells: dict, trows: tables.tablerow
) -> np.ndarray:
    """
    load variable-size array columns, padding with nans wherever
    needed. This happens for example often in the SPECTRAL_WINDOW
    table (CHAN_WIDTH, EFFECTIVE_BW, etc.).
    Also handle exceptions gracefully when trying to load the rows.

    :param inpath: path name of the MS
    :param col: column being loaded
    :param col_cells: {col: cell} values
    :param trows: rows from a table as loaded by tables.row()

    :return: array with column values (possibly padded if rows vary in size)
    """

    # Optional cols known to sometimes have inconsistent values
    known_misbehaving_cols = ["ASSOC_NATURE"]

    mshape = np.array(max([np.array(row[col]).shape for row in trows]))
    try:
        pad_nan = get_pad_nan(col_cells[col])

        # TODO
        # benchmark np.stack() performance
        data = np.stack(
            [
                np.pad(
                    row[col]
                    if len(row[col]) > 0
                    else np.array(row[col]).reshape(np.arange(len(mshape)) * 0),
                    [(0, ss) for ss in mshape - np.array(row[col]).shape],
                    "constant",
                    constant_values=pad_nan,
                )
                for row in trows
            ]
        )
    except Exception as exc:
        msg = f"{inpath}: failed to load data for column {col}: {exc}"
        if col in known_misbehaving_cols:
            logger.debug(msg)
        else:
            logger.warning(msg)
        data = np.empty(0)

    return data


def read_flat_col_chunk(infile, col, cshape, ridxs, cstart, pstart) -> np.ndarray:
    """
    Extract data chunk for each table col, this is fed to dask.delayed
    """

    with open_table_ro(infile) as tb_tool:
        rgrps = [
            (rr[0], rr[-1])
            for rr in np.split(ridxs, np.where(np.diff(ridxs) > 1)[0] + 1)
        ]
        # try:
        if (len(cshape) == 1) or (col == "UVW"):  # all the scalars and UVW
            data = np.concatenate(
                [tb_tool.getcol(col, rr[0], rr[1] - rr[0] + 1) for rr in rgrps], axis=0
            )
        elif len(cshape) == 2:  # WEIGHT, SIGMA
            data = np.concatenate(
                [
                    tb_tool.getcolslice(
                        col,
                        pstart,
                        pstart + cshape[1] - 1,
                        [],
                        rr[0],
                        rr[1] - rr[0] + 1,
                    )
                    for rr in rgrps
                ],
                axis=0,
            )
        elif len(cshape) == 3:  # DATA and FLAG
            data = np.concatenate(
                [
                    tb_tool.getcolslice(
                        col,
                        (cstart, pstart),
                        (cstart + cshape[1] - 1, pstart + cshape[2] - 1),
                        [],
                        rr[0],
                        rr[1] - rr[0] + 1,
                    )
                    for rr in rgrps
                ],
                axis=0,
            )
        # except:
        #    print('ERROR reading chunk: ', col, cshape, cstart, pstart)

    return data


def read_col_chunk(
    infile: str,
    ts_taql: str,
    col: str,
    cshape: Tuple[int],
    tidxs: np.ndarray,
    bidxs: np.ndarray,
    didxs: np.ndarray,
    d1: Tuple[int, int],
    d2: Tuple[int, int],
) -> np.ndarray:
    """
    Function to perform delayed reads from table columns.
    """
    # TODO: consider calling load_col_chunk() from inside the withs
    # for read_delayed_pointing_table and read_expanded_main_table
    with open_table_ro(infile) as mtable:
        with open_query(mtable, ts_taql) as query:
            if (len(cshape) == 2) or (col == "UVW"):  # all the scalars and UVW
                data = np.array(query.getcol(col, 0, -1))
            elif len(cshape) == 3:  # WEIGHT, SIGMA
                data = query.getcolslice(col, d1[0], d1[1], [], 0, -1)
            elif len(cshape) == 4:  # DATA and FLAG
                data = query.getcolslice(col, (d1[0], d2[0]), (d1[1], d2[1]), [], 0, -1)

    # full data is the maximum of the data shape and chunk shape dimensions
    fulldata = np.full(cshape, np.nan, dtype=data.dtype)
    if len(didxs) > 0:
        fulldata[tidxs[didxs], bidxs[didxs]] = data[didxs]

    return fulldata


def read_col_conversion(
    tb_tool,
    col: str,
    cshape: Tuple[int],
    tidxs: np.ndarray,
    bidxs: np.ndarray,
):
    """
    Function to perform delayed reads from table columns when converting
    (no need for didxs)
    """

    # Workaround for https://github.com/casacore/python-casacore/issues/130
    # WARNING: Assumes tb_tool is a single measurement set not an MMS.
    # WARNING: Assumes the num_frequencies * num_polarisations > 2**29. If false,
    # https://github.com/casacore/python-casacore/issues/130 isn't mitigated.

    # Use casacore to get the shape of a row for this column

    #################################################################################

    # Get the total number of rows in the base measurement set
    nrows_total = tb_tool.nrows()

    # getcolshapestring() only works on columns where a row element is an
    # array (ie fails for TIME, etc)
    # Assumes RuntimeError is because the column is a scalar
    try:

        shape_string = tb_tool.getcolshapestring(col)[0]
        extra_dimensions = tuple(
            [
                int(idx)
                for idx in shape_string.replace("[", "").replace("]", "").split(", ")
            ]
        )
        full_shape = tuple(
            [nrows_total]
            + [
                int(idx)
                for idx in shape_string.replace("[", "").replace("]", "").split(", ")
            ]
        )
    except RuntimeError:
        extra_dimensions = ()
        full_shape = (nrows_total,)

    #################################################################################

    # Get dtype of the column. Only read first row from disk
    col_dtype = np.array(tb_tool.col(col)[0]).dtype

    # Construct the numpy array to populate with data
    data = np.empty(full_shape, dtype=col_dtype)

    # Use built-in casacore table iterator to populate the data column by unique times.
    start_row = 0
    for ts in tb_tool.iter("TIME", sort=False):
        num_rows = ts.nrows()
        # Note don't use getcol() because it's less safe. See:
        # https://github.com/casacore/python-casacore/issues/130#issuecomment-463202373
        ts.getcolnp(col, data[start_row : start_row + num_rows])
        start_row += num_rows

    # TODO
    # Can we return a view of `data` instead of copying?
    fulldata = np.full(cshape + extra_dimensions, np.nan, dtype=col_dtype)
    fulldata[tidxs, bidxs] = data
    return fulldata
