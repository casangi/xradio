import toolviper.utils.logger as logger
import os
from pathlib import Path
import re
from typing import Any, Callable, Dict, List, Tuple, Union

import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr

import astropy.units
from casacore import tables

from .table_query import open_query, open_table_ro, TableManager
from xradio._utils.list_and_array import get_pad_value

CASACORE_TO_PD_TIME_CORRECTION = 3_506_716_800.0
SECS_IN_DAY = 86400
MJD_DIF_UNIX = 40587


def table_exists(path: str) -> bool:
    """
    Whether a casacore table exists on disk (in the casacore.tables.tableexists sense)
    """
    return tables.tableexists(path)


def table_has_column(path: str, column_name: str) -> bool:
    """
    Whether a column is present in a casacore table
    """
    with open_table_ro(path) as tb_tool:
        if column_name in tb_tool.colnames():
            return True
        else:
            return False


def convert_casacore_time(
    rawtimes: np.ndarray, convert_to_datetime: bool = True
) -> np.ndarray:
    """
    Read time columns to datetime format
    pandas datetimes are referenced against a 0 of 1970-01-01
    CASA's modified julian day reference time is (of course) 1858-11-17

    This requires a correction of 3506716800 seconds which is hardcoded to save time

    Parameters
    ----------
    rawtimes : np.ndarray
        times in casacore ref
    convert_to_datetime : bool (Default value = True)
        whether to produce pandas style datetime

    Returns
    -------
    np.ndarray
        times converted to pandas reference
    """
    times_reref = np.array(rawtimes) - CASACORE_TO_PD_TIME_CORRECTION
    if convert_to_datetime:
        return pd.to_datetime(times_reref, unit="s").values
    else:
        return times_reref
    # dt = pd.to_datetime(np.atleast_1d(rawtimes) - correction, unit='s').values
    # if len(np.array(rawtimes).shape) == 0: dt = dt[0]
    # return dt


def convert_mjd_time(rawtimes: np.ndarray) -> np.ndarray:
    """
    Different time conversion needed for the MJD col of EPHEM{i}_*.tab
    files (only, as far as I've seen)

    Parameters
    ----------
    rawtimes : np.ndarray
        MJD times for example from the MJD col of ephemerides tables

    Returns
    -------
    np.ndarray
        times converted to pandas reference and datetime type
    """
    times_reref = pd.to_datetime(
        (rawtimes - MJD_DIF_UNIX) * SECS_IN_DAY, unit="s"
    ).values

    return times_reref


def convert_casacore_time_to_mjd(rawtimes: np.ndarray) -> np.ndarray:
    """
    From CASA/casacore time (as used in the TIME column of the main table) to MJD
    (as used in the EPHEMi*.tab ephemeris tables). As the epochs are the same, this
    is just a conversion of units.

    Parameters
    ----------
    rawtimes : np.ndarray
        times from a TIME column (seconds, casacore time epoch)

    Returns
    -------
    np.ndarray
        times converted to (ephemeris) MJD (days since casacore time epoch (1858-11-17))
    """
    return rawtimes / SECS_IN_DAY


def make_taql_where_between_min_max(
    min_max: Tuple[np.float64, np.float64],
    path: str,
    table_name: str,
    colname="TIME",
) -> Union[str, None]:
    """
    From a numerical min/max range, produce a TaQL string to select between
    those min/max values (example: times) in a table.
    The table can be for example a POINTING subtable or an EPHEM* ephemeris
    table.
    This is meant to be used on MSv2 table columns that will be loaded as a
    coordinate in MSv4s and their sub-xdss (example: POINTING/TIME ephemeris/MJD).

    This can be used for example to produce a TaQL string to constraing loading of:
    - POINTING rows (based on the min/max from the time coordinate of the main MSv4)
    - ephemeris rows, from EPHEM* tables ((based on the MJD column and the min/max
      from the main MSv4 time coordinate).

    Parameters
    ----------
    min_max : Tuple[np.float64, np.float64]
        min / max values of time or other column used as coordinate
        (assumptions: float values, sortable, typically: time coord from MSv4)
    path :
        Path to input MS or location of the table
    table_name :
        Name of the table where to load a column (example: 'POINTING')
    colname :
        Name of the column to search for min/max values (examples: 'TIME', 'MJD')

    Returns
    -------
    taql_where : str
        TaQL (sub)string with the min/max time 'WHERE' constraint
    """

    min_max_range = find_projected_min_max_table(min_max, path, table_name, colname)
    if min_max_range is None:
        taql = None
    else:
        (min_val, max_val) = min_max_range
        taql = f"where {colname} >= {min_val} AND {colname} <= {max_val}"

    return taql


def find_projected_min_max_table(
    min_max: Tuple[np.float64, np.float64], path: str, table_name: str, colname: str
) -> Union[Tuple[np.float64, np.float64], None]:
    """
    We have: min/max values that define a range (for example of time)
    We want: to project that min/max range on a sortable column (for example a
    range of times onto a TIME column), and find min and max values
    derived from that table column such that the range between those min and max
    values includes at least the input min/max range.

    The returned min/max can then be used in a data selection TaQL query to
    select at least the values within the input range (possibly extended if
    the input range overlaps only partially or not at all with the column
    values). A tolerance is added to the min/max to prevent numerical issues in
    comparisons and conversios between numerical types and strings.

    When the range given as input is wider than the range of values found in
    the column, use the input range, as it is sufficient and more inclusive.

    When the range given as input is narrow (projected on the target table/column)
    and falls between two points of the column values, or overlaps with only one
    point, the min/max are extended to include at least the two column values that
    define a range within which the input range is included.
    Example scenario: an ephemeris table is sampled at a coarse interval
    (20 min) and we want to find a min/max range projected from the time min/max
    of a main MSv4 time coordinate sampled at ~1s for a field-scan/intent
    that spans ~2 min. Those ~2 min will typically fall between ephemeris samples.

    Parameters
    ----------
    min_max : Tuple[np.float64, np.float64]
        min / max values of time or other column used as coordinate
        (assumptions: float values, sortable)
    path :
        Path to input MS or location of the table
    table_name :
        Name of the table where to load a column (example: 'POINTING')
    colname :
        Name of the column to search for min/max values (example: 'TIME')

    Returns
    -------
    output_min_max : Union[Tuple[np.float64, np.float64], None]
        min/max values derived from the input min/max and the column values
    """
    with open_table_ro(os.path.join(path, table_name)) as tb_tool:
        if tb_tool.nrows() == 0:
            return None
        col = tb_tool.getcol(colname)

    out_min_max = find_projected_min_max_array(min_max, col)
    return out_min_max


def find_projected_min_max_array(
    min_max: Tuple[np.float64, np.float64], array: np.array
) -> Tuple[np.float64, np.float64]:
    """Does the min/max checks and search for find_projected_min_max_table()"""

    sorted_array = np.sort(array)
    (range_min, range_max) = min_max
    if len(sorted_array) < 2:
        tol = np.finfo(sorted_array.dtype).eps * 4
    else:
        tol = np.diff(sorted_array[np.nonzero(sorted_array)]).min() / 4

    if range_max > sorted_array[-1]:
        projected_max = range_max + tol
    else:
        max_idx = sorted_array.size - 1
        max_array_idx = min(
            max_idx, np.searchsorted(sorted_array, range_max, side="right")
        )
        projected_max = sorted_array[max_array_idx] + tol

    if range_min < sorted_array[0]:
        projected_min = range_min - tol
    else:
        min_array_idx = max(
            0, np.searchsorted(sorted_array, range_min, side="left") - 1
        )
        # ensure 'sorted_array[min_array_idx] < range_min' when values ==
        if sorted_array[min_array_idx] == range_min:
            min_array_idx = max(0, min_array_idx - 1)
        projected_min = sorted_array[min_array_idx] - tol

    return (projected_min, projected_max)


def extract_table_attributes(infile: str) -> Dict[str, Dict]:
    """
    Return a dictionary of table attributes created from MS keywords and column descriptions

    Parameters
    ----------
    infile : str
        table file path

    Returns
    -------
    Dict[str, Dict]
        table attributes as a dictionary
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

    Parameters
    ----------
    mvars : Dict[str, xr.DataArray]
        data variables where to populate units
    cc_attrs : Dict[str, Any]
        dictionary with casacore table attributes (from extract_table_attributes)

    Returns
    -------
    Dict[str, xr.DataArray]
        variables with units added in their attributes
    """
    col_descrs = cc_attrs["column_descriptions"]
    # TODO: Should probably loop the other way around, over mvars
    for col in col_descrs:
        if col == "TIME":
            var_name = "time"
        else:
            var_name = col
        if var_name in mvars and "keywords" in col_descrs[col]:
            if "QuantumUnits" in col_descrs[col]["keywords"]:
                cc_units = col_descrs[col]["keywords"]["QuantumUnits"]

                if isinstance(
                    cc_units, str
                ):  # Little fix for Meerkat data where the units are a string.
                    cc_units = [cc_units]

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
    """
    Grab the units/measure metainfo for the xds.freq dimension of a
    parttion from the SPECTRAL_WINDOW subtable CTDS attributes.

    Has to read xds_spw.meas_freq_ref and use it as index in the CTDS
    'VarRefCol' attrs of CHAN_FREQ and REF_FREQUENCY to give a
    reference frame to xds_spw.ref_frequency and xds_spw.chan_freq
    (then the ref frame from the second will be pulled to
    xds.freq.attrs)

    Parameters
    ----------
    spw_xds : xr.Dataset
        (metainfo) SPECTRAL_WINDOW xds
    spw_id : int
        SPW id of a partition

    Returns
    -------
    Dict[str, Any]
        attributes (units/measure) for the freq dim of a partition
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

    cf_attrs = spw_xds.data_vars["CHAN_FREQ"].attrs
    if "MEASINFO" in cfreq["keywords"] and "VarRefCol" in cfreq["keywords"]["MEASINFO"]:
        fattrs = cfreq["keywords"]["MEASINFO"]
        var_ref_col = fattrs["VarRefCol"]
        # This should point to the SPW/MEAS_FREQ_REF col
        meas_freq_ref_idx = spw_xds.data_vars[var_ref_col].values[spw_id]

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
        spw_xds.data_vars["CHAN_FREQ"].attrs.update(cf_attrs)
        spw_xds.data_vars["REF_FREQUENCY"].attrs.update(cf_attrs)

    return cf_attrs


def redimension_ms_subtable(xds: xr.Dataset, subt_name: str) -> xr.Dataset:
    """
    Expand a MeasurementSet subtable xds from single dimension (row)
    to multiple dimensions (such as (source_id, time, spectral_window)

    WIP: only works for source, experimenting

    Parameters
    ----------
    xds : xr.Dataset
        dataset to change the dimensions
    subt_name : str
        subtable name (SOURCE, etc.)

    Returns
    -------
    xr.Dataset
        transformed xds with data dimensions representing the MS subtable key
        (one dimension for every columns)
    """
    subt_key_cols = {
        "DOPPLER": ["DOPPLER_ID", "SOURCE_ID"],
        "FREQ_OFFSET": [
            "ANTENNA1",
            "ANTENNA2",
            "FEED_ID",
            "SPECTRAL_WINDOW_ID",
            "TIME",
        ],
        "POINTING": ["TIME", "ANTENNA_ID"],
        "SOURCE": ["SOURCE_ID", "TIME", "SPECTRAL_WINDOW_ID"],
        "SYSCAL": ["ANTENNA_ID", "FEED_ID", "SPECTRAL_WINDOW_ID", "TIME"],
        "WEATHER": ["ANTENNA_ID", "TIME"],
        "PHASE_CAL": ["ANTENNA_ID", "TIME", "SPECTRAL_WINDOW_ID"],
        "GAIN_CURVE": ["ANTENNA_ID", "TIME", "SPECTRAL_WINDOW_ID"],
        "FEED": ["ANTENNA_ID", "SPECTRAL_WINDOW_ID"],
        # added tables (MSv3 but not preent in MSv2). Build it from "EPHEMi_... tables
        # Not clear what to do about 'time' var/dim:  , "time"],
        "EPHEMERIDES": ["ephemeris_row_id", "ephemeris_id"],
    }
    key_dims = subt_key_cols[subt_name]

    rxds = xds.copy()
    try:
        # drop_duplicates() needed (https://github.com/casangi/xradio/issues/185). Examples:
        # - Some early ALMA datasets have bogus WEATHER tables with many/most rows with
        #   (ANTENNA_ID=0, TIME=0) and no other columns to figure out the right IDs, such
        #   as "NS_WX_STATION_ID" or similar. (example: X425.pm04.scan4.ms)
        # - Some GBT MSs have duplicated (ANTENNA_ID=0, TIME=xxx). (example: analytic_variable.ms)
        rxds = (
            rxds.set_index(row=key_dims)
            .drop_duplicates("row")
            .unstack("row")
            .transpose(*key_dims, ...)
        )
        # unstack changes type to float when it needs to introduce NaNs, so
        # we need to reset to the original type.
        for var in rxds.data_vars:
            if rxds[var].dtype != xds[var].dtype:
                # beware of gaps/empty==nan values when redimensioning
                with np.errstate(invalid="ignore"):
                    rxds[var] = rxds[var].astype(xds[var].dtype)
    except Exception as exc:
        logger.warning(
            f"Cannot expand rows in table {subt_name} to {key_dims}, possibly duplicate values in those coordinates. "
            f"Exception: {exc}"
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
        xds["MJD"], dtype=np.uint32
    )
    xds = xds.rename({"MJD": "time"})
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


def load_generic_table(
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
    inpath : str
        path to the MS or directory containing the table
    tname : str
        (sub)table name, for example 'SOURCE' for myms.ms/SOURCE
    timecols : Union[List[str], None] (Default value = None)
        column names to convert to numpy datetime format.
        leaves times as their original casacore format.
    ignore : Union[List[str], None] (Default value = None)
        list of column names to ignore and not try to read.
    rename_ids : Dict[str, str] (Default value = None)
        dict with dimension renaming mapping
    taql_where : str (Default value = None)
         TaQL string to optionally constain the rows/columns to read
         (Default value = None)

    Returns
    -------
    xr.Dataset
        table loaded as XArray dataset
    """
    if timecols is None:
        timecols = []
    if ignore is None:
        ignore = []

    infile = Path(inpath, tname)
    infile = str(infile.expanduser())
    if not os.path.isdir(infile):
        raise ValueError(
            f"invalid input filename to load_generic_table: {infile} table {tname}"
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

        # if len(ignore) > 0: #This is needed because some SOURCE tables have a SOURCE_MODEL column that is corrupted and this causes the open_query to fail.
        #     select_columns = gtable.colnames()
        #     select_columns_str = str([item for item in select_columns if item not in ignore])[1:-1].replace("'", "") #Converts an array to a comma sepearted string. For example ['a', 'b', 'c'] to 'a, b, c'.
        #     taql_gtable = f"select " + select_columns_str + f" from $gtable {taql_where}"
        # else:
        #     taql_gtable = f"select * from $gtable {taql_where}"

        # relatively often broken columns that we do not need
        exclude_pattern = ", !~p/SOURCE_MODEL/"
        taql_gtable = f"select *{exclude_pattern} from $gtable {taql_where or ''}"

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

    if tname in [
        "DOPPLER",
        "FREQ_OFFSET",
        "POINTING",
        "SOURCE",
        "SYSCAL",
        "WEATHER",
        "PHASE_CAL",
        "GAIN_CURVE",
        "FEED",
    ]:
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

    Parameters
    ----------
    inpath : str
        input path
    tb_tool: tables.table
        tool being used to load data
    timecols: Union[List[str], None] (Default value = None)
        list of columns to be considered as TIME-related
    ignore: Union[List[str], None] (Default value = None)
        columns to ignore

    Returns
    -------
    Tuple[Dict[str, xr.Dataset], Dict[str, xr.Dataset]]
        coordinates dictionary + variables dictionary
    """
    columns_loader = find_best_col_loader(inpath, tb_tool.nrows())

    mcoords, mvars = columns_loader(inpath, tb_tool, timecols, ignore)

    return mcoords, mvars


def find_best_col_loader(inpath: str, nrows: int) -> Callable:
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

    Parameters
    ----------
    inpath : str
        path name of the MS table
    nrows : int
        number of rows found in the table

    Returns
    -------
    Callable
        function best suited to load the data from the columns of this table
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
    """
    Loads data for each MS column (loading the data in memory) into Xarray datasets

    This function is generic in that it can load variable size array columns. See also
    load_fixed_size_cols() as a simpler and much better performing alternative
    for tables that are large and expected/guaranteed to not have columns with variable
    size cells.

    Parameters
    ----------
    inpath : str
        path name of the MS table
    tb_tool : tables.table
        table to load the columns
    timecols : Union[List[str], None]
        columns names to convert to datetime format
    ignore : Union[List[str], None]
        list of column names to skip and not try to load.

    Returns
    -------
    Tuple[Dict[str, xr.Dataset], Dict[str, xr.Dataset]]
        dict of coordinates and dict of data vars.
    """

    col_types = find_loadable_cols(tb_tool, ignore)

    trows = tb_tool.row(ignore, exclude=True)[:]

    # Produce coords and data vars from MS columns
    mcoords, mvars = {}, {}
    for col in col_types.keys():
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

            data = handle_variable_col_issues(inpath, col, col_types[col], trows)

        if len(data) == 0:
            continue

        array_type, array_data = raw_col_data_to_coords_vars(
            inpath, tb_tool, col, data, timecols
        )
        if array_type == "coord":
            mcoords[col] = array_data
        elif array_type == "data_var":
            mvars[col] = array_data

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

    Parameters
    ----------
    inpath : str
        path name of the MS
    tb_tool : tables.table
        table to red the columns
    timecols : Union[List[str], None]
        columns names to convert to datetime format
    ignore : Union[List[str], None]
        list of column names to skip and not try to load.

    Returns
    -------
    Tuple[Dict[str, xr.Dataset], Dict[str, xr.Dataset]]
        dict of coordinates and dict of data vars, ready to construct an xr.Dataset
    """

    loadable_cols = find_loadable_cols(tb_tool, ignore)

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
            mcoords[col] = array_data
        elif array_type == "data_var":
            mvars[col] = array_data

    return mcoords, mvars


def find_loadable_cols(
    tb_tool: tables.table, ignore: Union[List[str], None]
) -> Dict[str, str]:
    """
    For a table, finds the columns that are loadable = not of record type,
    and not to be ignored
    In extreme cases of variable size columns, it can happen that all the
    cells are empty (iscelldefined() == false). This is still considered a
    loadable column, even though all values of the resulting data var will
    be empty.

    Parameters
    ----------
    tb_tool : tables.table
        table to red the columns
    ignore : Union[List[str], None]
        list of column names to skip and not try to load.

    Returns
    -------
    Dict
        dict of {column name: column type} for columns that can/should be loaded
    """

    colnames = tb_tool.colnames()
    table_desc = tb_tool.getdesc()
    loadable_cols = {
        col: table_desc[col]["valueType"]
        for col in colnames
        if (col not in ignore) and tb_tool.coldatatype(col) != "record"
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

    Parameters
    ----------
    inpath: str
        input table path
    tb_tool: tables.table :
        table toold being used to load data
    col: str :
        column
    data: np.ndarray :
        column data
    timecols: Union[List[str], None]
        columns to be treated as TIME-related

    Returns
    -------
    Tuple[str, xr.DataArray]
        array type string (whether this column is a 'coord' or a 'data_var') + DataArray
        with column  data/coord values ready to be added to the table xds
    """

    # Almost sure that when TIME is present (in a standard MS subt) it
    # is part of the key. But what about non-std subtables, ASDM subts?
    subts_with_time_key = (
        "FLAG_CMD",
        "FREQ_OFFSET",
        "HISTORY",
        "POINTING",
        "SOURCE",
        "SYSCAL",
        "WEATHER",
        "PHASE_CAL",
        "GAIN_CURVE",
        "FEED",
    )
    dim_prefix = "dim"

    if col in timecols:
        if col == "MJD":
            data = convert_mjd_time(data).astype("float64") / 1e9
        else:
            try:
                data = convert_casacore_time(data)
            except pd.errors.OutOfBoundsDatetime as exc:
                if inpath.endswith("WEATHER"):
                    # intentionally not callling logging.exception
                    logger.warning(
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


def get_pad_value_in_tablerow_column(trows: tables.tablerow, col: str) -> object:
    """
    Gets the pad value for the type of a column (IMPORTANTLY) as froun in the
    the type specified in the row / column value dict returned by tablerow.
    This can differ from the type of the column as given in the casacore
    column descriptions. See https://github.com/casangi/xradio/issues/242.

    Parameters
    ----------
    trows : tables.tablerow
        list of rows from a table as loaded by tables.row()
    col: str
        get the pad value for this column

    Returns
    -------
    object
        pad value as produced by get_pad_value for the appropriate data type from
    tablerow
    """
    col_value = trows[0][col]
    if isinstance(col_value, np.ndarray):
        col_dtype = col_value.dtype
    elif isinstance(col_value, list):
        col_dtype = type(col_value[0])
    else:
        raise RuntimeError(
            "Found unexpected type (not np.array or list) in column value of "
            f"first row of column {col}: {col_value}"
        )

    return get_pad_value(col_dtype)


def handle_variable_col_issues(
    inpath: str, col: str, col_type: str, trows: tables.tablerow
) -> np.ndarray:
    """
    load variable-size array columns, padding with missing/fill/nans
    wherever needed. This happens for example often in the
    SPECTRAL_WINDOW table (CHAN_WIDTH, EFFECTIVE_BW, etc.).
    Also handle exceptions gracefully when trying to load the rows.

    Parameters
    ----------
    inpath : str
        path name of the MS
    col : str
        column being loaded
    col_type : str
        type of the column cell values (as numpy dtype string)
    trows : tables.tablerow
        rows from a table as loaded by tables.row()

    Returns
    -------
    np.ndarray
        array with column values (possibly padded if rows vary in size)
    """

    # Optional cols known to sometimes have inconsistent values
    known_misbehaving_cols = ["ASSOC_NATURE"]

    mshape = np.array(max([np.array(row[col]).shape for row in trows]))
    try:
        pad_val = None
        pad_val = get_pad_value_in_tablerow_column(trows, col)

        # TODO
        # benchmark np.stack() performance
        data = np.stack(
            [
                np.pad(
                    (
                        row[col]
                        if len(row[col]) > 0
                        else np.array(row[col]).reshape(np.arange(len(mshape)) * 0)
                    ),
                    [(0, ss) for ss in mshape - np.array(row[col]).shape],
                    "constant",
                    constant_values=pad_val,
                )
                for row in trows
            ]
        )
    except Exception as exc:
        msg = f"{inpath}: failed to load data for column {col}, with {pad_val=}: {exc}"
        if col in known_misbehaving_cols:
            logger.debug(msg)
        else:
            logger.warning(msg)
        data = np.empty(0)

    return data


def read_flat_col_chunk(infile, col, cshape, ridxs, cstart, pstart) -> np.ndarray:
    """
    Extract data chunk for each table col, this is fed to dask.delayed

    Parameters
    ----------
    infile :

    col :

    cshape :

    ridxs :

    cstart :

    pstart :

    Returns
    -------
    np.ndarray
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

    Parameters
    ----------
    infile : str

    ts_taql : str

    col : str

    cshape : Tuple[int]

    tidxs : np.ndarray

    bidxs : np.ndarray

    didxs : np.ndarray

    d1: Tuple[int, int]

    d2: Tuple[int, int]

    Returns
    -------
    np.ndarray
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

    fill_value = get_pad_value(data.dtype)
    fulldata = np.full(cshape, fill_value, dtype=data.dtype)

    if len(didxs) > 0:
        fulldata[tidxs[didxs], bidxs[didxs]] = data[didxs]

    return fulldata


def read_col_conversion_numpy(
    table_manager: TableManager,
    col: str,
    cshape: Tuple[int],
    tidxs: np.ndarray,
    bidxs: np.ndarray,
    use_table_iter: bool,
    time_chunksize: int,
) -> np.ndarray:
    """
    Function to perform delayed reads from table columns when converting
    (no need for didxs)

    Parameters
    ----------
    table_manager : TableManager

    col : str

    cshape : Tuple[int]

    tidxs : np.ndarray

    bidxs : np.ndarray

    use_table_iter : bool

    Returns
    -------
    np.ndarray
    """

    # Workaround for https://github.com/casacore/python-casacore/issues/130
    # WARNING: Assumes tb_tool is a single measurement set not an MMS.
    # WARNING: Assumes the num_frequencies * num_polarizations < 2**29. If false,
    # https://github.com/casacore/python-casacore/issues/130 isn't mitigated.

    with table_manager.get_table() as tb_tool:

        # Use casacore to get the shape of a row for this column
        #################################################################################

        # getcolshapestring() only works on columns where a row element is an
        # array ie. fails for TIME
        # Assumes the RuntimeError is because the column is a scalar
        try:
            shape_string = tb_tool.getcolshapestring(col)[0]
            # Convert `shape_string` into a tuple that numpy understands
            extra_dimensions = tuple(
                [
                    int(idx)
                    for idx in shape_string.replace("[", "")
                    .replace("]", "")
                    .split(", ")
                ]
            )
        except RuntimeError:
            extra_dimensions = ()

        #################################################################################

        # Get dtype of the column. Only read first row from disk
        col_dtype = np.array(tb_tool.col(col)[0]).dtype
        # Use a custom/safe fill value (https://github.com/casangi/xradio/issues/219)
        fill_value = get_pad_value(col_dtype)

        # Construct a numpy array to populate. `data` has shape (n_times, n_baselines, n_frequencies, n_polarizations)
        data = np.full(cshape + extra_dimensions, fill_value, dtype=col_dtype)

        # Use built-in casacore table iterator to populate the data column by unique times.
        if use_table_iter:
            start_row = 0
            for ts in tb_tool.iter("TIME", sort=False):
                num_rows = ts.nrows()

                # Create small temporary array to store the partial column
                tmp_arr = np.full(
                    (num_rows,) + extra_dimensions, fill_value, dtype=col_dtype
                )

                # Note we don't use `getcol()` because it's less safe. See:
                # https://github.com/casacore/python-casacore/issues/130#issuecomment-463202373
                ts.getcolnp(col, tmp_arr)

                # Get the slice of rows contained in `tmp_arr`.
                # Used to get the relevant integer indexes from `tidxs` and `bidxs`
                tmp_slice = slice(start_row, start_row + num_rows)

                # Copy `tmp_arr` into correct elements of `tmp_arr`
                data[tidxs[tmp_slice], bidxs[tmp_slice]] = tmp_arr
                start_row += num_rows
        else:
            data[tidxs, bidxs] = tb_tool.getcol(col)

    return data


def read_col_conversion_dask(
    table_manager: TableManager,
    col: str,
    cshape: Tuple[int],
    tidxs: np.ndarray,
    bidxs: np.ndarray,
    use_table_iter: bool,
    time_chunksize: int,
) -> da.Array:
    """
    Function to perform delayed reads from table columns when converting
    (no need for didxs)

    Parameters
    ----------
    tb_tool : tables.table

    col : str

    cshape : Tuple[int]

    tidxs : np.ndarray

    bidxs : np.ndarray

    Returns
    -------
    da.Array
    """

    # Use casacore to get the shape of a row for this column
    #################################################################################

    with table_manager.get_table() as tb_tool:
        first_row = tb_tool.row(col)[0][col]

    if isinstance(first_row, np.ndarray):
        extra_dimensions = first_row.shape

    else:
        extra_dimensions = ()

    # Use dask primitives to lazily read chunks of data from the MeasurementSet
    # Takes inspiration from dask_image https://image.dask.org/en/latest/
    #################################################################################

    # Get dtype of the column. Wrap in numpy array in case of scalar column
    col_dtype = np.array(first_row).dtype

    # Get the number of rows for a single TIME value
    num_utimes = cshape[0]
    rows_per_time = cshape[1]

    # Calculate the chunks of unique times that gives the target chunk sizes
    tmp_chunks = da.core.normalize_chunks(time_chunksize, (num_utimes,))[0]

    sum = 0
    arr_start_end_rows = []
    for chunk in tmp_chunks:
        start = (sum) * rows_per_time
        end = (sum + chunk) * rows_per_time

        arr_start_end_rows.append((start, end))
        sum += chunk

    # Store the start and end rows that should be read for the chunk
    arr_start_end_rows = da.from_array(arr_start_end_rows, chunks=(1, 2))

    # Specify the output shape `load_col_chunk`
    output_chunkshape = (tmp_chunks, cshape[1]) + extra_dimensions

    # Apply `load_col_chunk` to each chunk
    data = arr_start_end_rows.map_blocks(
        load_col_chunk,
        table_manager=table_manager,
        col_name=col,
        col_dtype=col_dtype,
        tidxs=tidxs,
        bidxs=bidxs,
        rows_per_time=rows_per_time,
        cshape=cshape,
        extra_dimensions=extra_dimensions,
        drop_axis=[1],
        new_axis=list(range(1, len(cshape + extra_dimensions))),
        meta=np.array([], dtype=col_dtype),
        chunks=output_chunkshape,
    )

    return data


def load_col_chunk(
    x,
    table_manager,
    col_name,
    col_dtype,
    tidxs,
    bidxs,
    rows_per_time,
    cshape,
    extra_dimensions,
):
    start_row = x[0][0]
    end_row = x[0][1]
    num_rows = end_row - start_row
    assert (num_rows % rows_per_time) == 0
    num_utimes = num_rows // rows_per_time

    # Create memory buffer to populate with data from disk
    row_data = np.full((num_rows,) + extra_dimensions, np.nan, dtype=col_dtype)

    # Load data from the column
    # Release the casacore table as soon as possible
    with table_manager.get_table() as tb_tool:
        tb_tool.getcolnp(col_name, row_data, startrow=start_row, nrow=num_rows)

    # Initialise reshaped numpy array
    reshaped_data = np.full(
        (num_utimes, cshape[1]) + extra_dimensions, np.nan, dtype=col_dtype
    )

    # Create slice object for readability
    slc = slice(start_row, end_row)
    tidxs_slc = tidxs[slc]

    tidxs_slc = (
        tidxs_slc - tidxs_slc[0]
    )  # Indices of reshaped_data along time differ from values in tidxs. Assumes first time is earliest time
    bidxs_slc = bidxs[slc]

    # Populate `reshaped_data` with `row_data`
    reshaped_data[tidxs_slc, bidxs_slc] = row_data

    return reshaped_data
