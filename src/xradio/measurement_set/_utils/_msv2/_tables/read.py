import contextlib
import os
import re
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import astropy.units
import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr

try:
    from casacore import tables
except ImportError:
    import xradio._utils._casacore.casacore_from_casatools as tables

from xradio._utils.list_and_array import get_pad_value
from xradio._utils.logging import xradio_logger
from xradio.measurement_set._utils._msv2._tables.read_rows import (
    CASACORE_TO_NUMPY_DTYPE,
    DEFAULT_MAX_ELEMS,
    MainTableRows,
    TimeChunkRows,
    backend_has_in_place_reads,
    getcol_chunks,
    parse_shape_string,
    read_column_rows,
    read_grid,
    read_time_chunk,
)
from xradio.measurement_set._utils._msv2._tables.subtable_cache import (
    active_subtable_cache,
    is_memoized_table,
)
from xradio.measurement_set._utils._msv2._tables.table_query import (
    open_query,
    open_table_ro,
)

CASACORE_TO_PD_TIME_CORRECTION = 3_506_716_800.0
# Elements per read call of the vectorized sub-table loads (python-casacore #130)
SUBTABLE_READ_MAX_ELEMS = DEFAULT_MAX_ELEMS
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
    Convert data from casacore time columns to a different format, either:
    a) pandas style datetime,
    b) simply seconds from 1970-01-01 00:00:00 UTC (as used in the Unix scale of
       astropy).

    Pandas datetimes and Unix times are referenced against a 0 of 1970-01-01.
    CASA's (casacore) modified julian day reference time is (of course) 1858-11-17.

    This requires a correction of 3506716800 seconds which is hardcoded to save time

    Parameters
    ----------
    rawtimes : np.ndarray
        time values wrt casacore reference
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


def casacore_numpy_to_json_safe_type(value: object) -> object:
    """
    This funciton converts values with numpy types (commonly found in "raw" xds loaded with load_generic_table()) to
    native types that can be safely written to JSON.

    This is handy when loading values from table columns that one wants to put in attributes. Starting
    with Zarr 3, numpy types are not converted/encoded before writing to JSON, and an exception is raised.

    Parameters
    ----------
    value : object
        A value or numpy array of values, of presumably numpy type

    Returns
    -------
    object
        The same value converted to a JSON safe type (for example int(a_numpy_int32_value))
    """
    if isinstance(value, np.ndarray):
        return ",".join([scalar for scalar in value])
    elif isinstance(value, np.integer):
        return int(value)
    elif isinstance(value, np.floating):
        return float(value)
    else:
        return value


def make_taql_where_between_min_max(
    min_max: tuple[np.float64, np.float64],
    path: str,
    table_name: str,
    colname="TIME",
) -> str | None:
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
        min_val, max_val = min_max_range
        taql = f"where {colname} >= {min_val} AND {colname} <= {max_val}"

    return taql


def find_projected_min_max_table(
    min_max: tuple[np.float64, np.float64], path: str, table_name: str, colname: str
) -> tuple[np.float64, np.float64] | None:
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
    subtable_cache = active_subtable_cache()
    if subtable_cache is not None:
        table_path = os.path.join(path, table_name)
        sorted_column = subtable_cache.get_or_build(
            ("sorted_column", table_path, colname),
            lambda: load_sorted_column(table_path, colname),
            amortized=True,
        )
        # None: not cacheable (or not worth building yet), the uncached code
        # below runs (and fails) as before
        if sorted_column is not None:
            sorted_array, tol = sorted_column
            if sorted_array.size == 0:
                return None
            range_min, range_max = min_max
            return project_min_max_sorted(range_min, range_max, sorted_array, tol)

    with open_table_ro(os.path.join(path, table_name)) as tb_tool:
        if tb_tool.nrows() == 0:
            return None
        col = tb_tool.getcol(colname)

    out_min_max = find_projected_min_max_array(min_max, col)
    return out_min_max


def load_sorted_column(
    table_path: str, colname: str
) -> tuple[np.ndarray, np.float64 | None] | None:
    """
    Reads and sorts a column once, for find_projected_min_max_table() with an
    active sub-table cache.

    Parameters
    ----------
    table_path : str
        Path of the table.
    colname : str
        Name of the (sortable) column, for example TIME or MJD.

    Returns
    -------
    tuple[np.ndarray, np.float64 | None] | None
        The sorted column values and their projection tolerance (an empty array
        and None if the table has no rows). None if anything fails: the caller
        then reads the column itself, and fails the same way as without cache.
        A MemoryError is raised.
    """
    try:
        with open_table_ro(table_path) as tb_tool:
            nrows = tb_tool.nrows()
            if nrows == 0:
                return np.empty(0), None
            # bounded reads in the column's own dtype (no full-size temporary)
            col = read_column_rows(
                tb_tool, colname, np.arange(nrows), max_elems=SUBTABLE_READ_MAX_ELEMS
            )
        sorted_array = np.sort(col)
        return sorted_array, projection_tolerance(sorted_array)
    except MemoryError:
        raise
    except Exception as exc:
        xradio_logger().debug(
            f"Not caching the sorted column {colname} of {table_path}: {exc}"
        )
        return None


def projection_tolerance(sorted_array: np.ndarray) -> np.float64:
    """
    The tolerance find_projected_min_max_array() adds to the projected min/max
    of a sorted column: a quarter of the smallest difference between its
    non-zero values (4 eps for fewer than two values).
    """
    if len(sorted_array) < 2:
        tol = np.finfo(sorted_array.dtype).eps * 4
    else:
        tol = np.diff(sorted_array[np.nonzero(sorted_array)]).min() / 4
    return tol


def find_projected_min_max_array(
    min_max: tuple[np.float64, np.float64], array: np.array
) -> tuple[np.float64, np.float64]:
    """Does the min/max checks and search for find_projected_min_max_table()"""

    sorted_array = np.sort(array)
    range_min, range_max = min_max
    tol = projection_tolerance(sorted_array)
    return project_min_max_sorted(range_min, range_max, sorted_array, tol)


def project_min_max_sorted(
    range_min: np.float64,
    range_max: np.float64,
    sorted_array: np.ndarray,
    tol: np.float64,
) -> tuple[np.float64, np.float64]:
    """
    The search of find_projected_min_max_array() on an already sorted, non-empty
    array and its projection_tolerance().
    """
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


def extract_table_attributes(infile: str) -> dict[str, dict]:
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
        attrs = {kk: kwd[kk] for kk in kwd if kk not in os.listdir(infile)}
        cols = tb_tool.colnames()
        column_descriptions = {}
        for col in cols:
            column_descriptions[col] = tb_tool.getcoldesc(col)
        attrs["column_descriptions"] = column_descriptions
        attrs["info"] = tb_tool.info()

    return attrs


def add_units_measures(
    mvars: dict[str, xr.DataArray], cc_attrs: dict[str, Any]
) -> dict[str, xr.DataArray]:
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

                if isinstance(cc_units, np.ndarray):
                    cc_units = cc_units.tolist()
                if not isinstance(cc_units, list) or not cc_units:
                    xradio_logger().warning(
                        f"Invalid units found for column/variable {col}: {cc_units}"
                    )
                mvars[var_name].attrs["units"] = cc_units[0]
                try:
                    astropy.units.Unit(cc_units[0])
                except Exception as exc:
                    xradio_logger().warning(
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
        xradio_logger().warning(
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


def is_nested_ms(attrs: dict) -> bool:
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
    timecols: list[str] | None = None,
    ignore: list[str] | None = None,
    rename_ids: dict[str, str] = None,
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
        Names of time column(s), to convert from casacore times to 1970-01-01 scale
        An empty list leaves times as their original casacore format.
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

    subtable_cache = active_subtable_cache()
    if subtable_cache is not None and is_memoized_table(tname):
        # Partitions load these sub-tables again and again with the same arguments
        key = (
            "load_generic_table",
            str(Path(inpath, tname).expanduser()),
            tuple(timecols),
            tuple(ignore),
            tuple(sorted(rename_ids.items())) if rename_ids else None,
            taql_where,
        )
        return subtable_cache.memo_dataset(
            key,
            lambda: _load_generic_table(
                inpath, tname, timecols, ignore, rename_ids, taql_where
            ),
        )
    return _load_generic_table(inpath, tname, timecols, ignore, rename_ids, taql_where)


def _load_generic_table(
    inpath: str,
    tname: str,
    timecols: list[str],
    ignore: list[str],
    rename_ids: dict[str, str] | None,
    taql_where: str | None,
) -> xr.Dataset:
    """load_generic_table() without the memo (timecols and ignore are lists)."""
    infile = Path(inpath, tname)
    infile = str(infile.expanduser())
    if not os.path.isdir(infile):
        raise ValueError(
            f"invalid input filename to load_generic_table: {infile} table {tname}"
        )

    cc_attrs = extract_table_attributes(infile)
    attrs: dict[str, Any] = {"other": {"msv2": {"ctds_attrs": cc_attrs}}}
    if is_nested_ms(attrs):
        xradio_logger().warning(
            f"Skipping subtable that looks like a MeasurementSet main table: {inpath} {tname}"
        )
        return xr.Dataset()

    with open_table_ro(infile) as gtable:
        if gtable.nrows() == 0:
            xradio_logger().debug(f"table is empty: {inpath} {tname}")
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
                xradio_logger().debug(
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
    xds = xds.rename({dv: dims[di] for di, dv in enumerate(xds.sizes)})
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
    timecols: list[str] | None = None,
    ignore: list[str] | None = None,
) -> tuple[dict[str, xr.Dataset], dict[str, xr.Dataset]]:
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
    if columns_loader is load_generic_cols and active_subtable_cache() is not None:
        # same result, without one row() dict per table row
        columns_loader = load_generic_cols_vectorized

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
    timecols: list[str] | None,
    ignore: list[str] | None,
) -> tuple[dict[str, xr.Dataset], dict[str, xr.Dataset]]:
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
        column names to convert from casacore time format
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
        data = stack_tablerow_column(inpath, col, col_types[col], trows)
        if data is None or len(data) == 0:
            continue

        array_type, array_data = raw_col_data_to_coords_vars(
            inpath, tb_tool, col, data, timecols
        )
        if array_type == "coord":
            mcoords[col] = array_data
        elif array_type == "data_var":
            mvars[col] = array_data

    return mcoords, mvars


def stack_tablerow_column(
    inpath: str, col: str, col_type: str, trows: list[dict]
) -> np.ndarray | None:
    """
    The values of one column, from the per-row dicts returned by tables.row(),
    stacked into one array (padded when the cells vary in shape), as
    load_generic_cols() loads them.

    Parameters
    ----------
    inpath : str
        path name of the MS table
    col : str
        column name
    col_type : str
        value type of the column (as in the column description)
    trows : list[dict]
        rows from tables.row() (with at least the column ``col``)

    Returns
    -------
    np.ndarray | None
        column data, or None if the column cannot be loaded (mixed cell types)
    """
    try:
        # TODO
        # benchmark np.stack() performance
        data = np.stack([row[col] for row in trows])  # .astype(col_cells[col].dtype)
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

        if len({isinstance(row[col], dict) for row in trows}) > 1:
            return None  # can't deal with this case

        data = handle_variable_col_issues(inpath, col, col_type, trows)

    return data


# dtype of np.stack() of the Python scalars that tables.row() returns for the
# cells of a scalar column, by column value type (strings are stacked as is).
_TABLEROW_SCALAR_STACK_DTYPES = {
    "boolean": np.dtype(np.bool_),
    "uchar": np.asarray(0).dtype,
    "short": np.asarray(0).dtype,
    "ushort": np.asarray(0).dtype,
    "int": np.asarray(0).dtype,
    "uint": np.asarray(0).dtype,
    "int64": np.asarray(0).dtype,
    "float": np.dtype(np.float64),
    "double": np.dtype(np.float64),
    "complex": np.dtype(np.complex128),
    "dcomplex": np.dtype(np.complex128),
}
# Storage managers whose array columns are read with one getcol() by
# load_generic_cols_vectorized(): a getcol() on a cell that is not defined raises
# there (tiled storage managers are left to tables.row(): a getcol() covering
# their undefined cells can crash the process).
_GETCOL_ARRAY_STORAGE_MANAGERS = ("StandardStMan", "IncrementalStMan")


def getcol_as_tablerow_stack(
    tb_tool: tables.table, col: str, col_type: str, storage_manager: str | None
) -> np.ndarray | None:
    """
    Reads a column with bounded column reads into exactly the array that
    stack_tablerow_column() builds from tables.row() (same values, shape and
    dtype), when that is possible without reading the rows one by one.

    - Scalar columns: tables.row() gives Python scalars, so np.stack() makes
      int columns int64, float columns float64, complex columns complex128.
    - Array columns of a StandardStMan / IncrementalStMan: the cells stacked
      when they all have the same shape (reading raises when they differ or
      are undefined, for which the caller uses tables.row()).

    Every read call covers at most SUBTABLE_READ_MAX_ELEMS elements
    (python-casacore #130) and, for the numeric value types, reads in place
    into an array of the column dtype (no getcol full-size temporary).

    Parameters
    ----------
    tb_tool : tables.table
        table (or selection) to read
    col : str
        column name
    col_type : str
        value type of the column (as in the column description)
    storage_manager : str | None
        type of the storage manager of the column

    Returns
    -------
    np.ndarray | None
        column data, or None if the column has to be read row by row

    Raises
    ------
    MemoryError
        Not turned into a row-by-row read (that needs more memory).
    """
    max_elems = SUBTABLE_READ_MAX_ELEMS
    try:
        rows = np.arange(tb_tool.nrows())
        if tb_tool.isscalarcol(col):
            if col_type == "string":
                values = []
                for part in getcol_chunks(tb_tool, col, rows, (), max_elems):
                    values.extend(part)
                return np.stack(values)
            dtype = _TABLEROW_SCALAR_STACK_DTYPES.get(col_type)
            if dtype is None:
                return None
            if col_type in CASACORE_TO_NUMPY_DTYPE:
                return read_column_rows(tb_tool, col, rows, max_elems).astype(dtype)
            parts = getcol_chunks(tb_tool, col, rows, (), max_elems)
            if not all(isinstance(part, np.ndarray) for part in parts):
                return None
            return np.concatenate(parts).astype(dtype)

        if (
            storage_manager not in _GETCOL_ARRAY_STORAGE_MANAGERS
            or col_type not in _TABLEROW_SCALAR_STACK_DTYPES
        ):
            return None
        if col_type in CASACORE_TO_NUMPY_DTYPE:
            # raises on cells of another shape than the first or undefined cells
            data = read_column_rows(tb_tool, col, rows, max_elems)
        else:
            cell_shape = parse_shape_string(tb_tool.getcolshapestring(col, 0, 1)[0])
            parts = getcol_chunks(tb_tool, col, rows, cell_shape, max_elems)
            if not all(
                isinstance(part, np.ndarray) and part.shape[1:] == cell_shape
                for part in parts
            ):
                return None
            data = np.concatenate(parts)
    except MemoryError:
        raise
    except Exception:
        # undefined cells, cells of different shapes, ...
        return None
    if not isinstance(data, np.ndarray) or data.ndim < 2:
        return None
    return data


def load_generic_cols_vectorized(
    inpath: str,
    tb_tool: tables.table,
    timecols: list[str] | None,
    ignore: list[str] | None,
) -> tuple[dict[str, xr.Dataset], dict[str, xr.Dataset]]:
    """
    Same result as load_generic_cols(), but every column that can be is read
    with one getcol() (see getcol_as_tablerow_stack()) instead of one row()
    dict per table row. Only the remaining columns (variable-shape or undefined
    cells, string arrays, tiled storage managers) are read with tables.row().

    Parameters
    ----------
    inpath : str
        path name of the MS table
    tb_tool : tables.table
        table to load the columns
    timecols : Union[List[str], None]
        column names to convert from casacore time format
    ignore : Union[List[str], None]
        list of column names to skip and not try to load.

    Returns
    -------
    Tuple[Dict[str, xr.Dataset], Dict[str, xr.Dataset]]
        dict of coordinates and dict of data vars.
    """
    col_types = find_loadable_cols(tb_tool, ignore)
    storage_managers = {
        col: dm_info["TYPE"]
        for dm_info in tb_tool.getdminfo().values()
        for col in dm_info["COLUMNS"]
    }

    col_data = {
        col: getcol_as_tablerow_stack(tb_tool, col, col_type, storage_managers.get(col))
        for col, col_type in col_types.items()
    }
    row_cols = [col for col, data in col_data.items() if data is None]
    if row_cols:
        trows = tb_tool.row(row_cols)[:]
        for col in row_cols:
            col_data[col] = stack_tablerow_column(inpath, col, col_types[col], trows)
        del trows

    # Produce coords and data vars from MS columns, in the same order as
    # load_generic_cols()
    mcoords, mvars = {}, {}
    for col, data in col_data.items():
        if data is None or len(data) == 0:
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
    timecols: list[str] | None,
    ignore: list[str] | None,
) -> tuple[dict[str, xr.Dataset], dict[str, xr.Dataset]]:
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
        column names to convert from casacore time format
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
            xradio_logger().warning(
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
    tb_tool: tables.table, ignore: list[str] | None
) -> dict[str, str]:
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
    timecols: list[str] | None,
) -> tuple[str, xr.DataArray]:
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
        columns to be treated as TIME-related (they are coordinate, need conversion from
        casacore time format.

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
            # data = convert_mjd_time(data).astype("float64") / 1e9
            data = convert_mjd_time(data).astype("datetime64[ns]").view("int64") / 1e9
        else:
            try:
                data = convert_casacore_time(data, False)
            except pd.errors.OutOfBoundsDatetime as exc:
                if inpath.endswith("WEATHER"):
                    # intentionally not callling logging.exception
                    xradio_logger().warning(
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
    Gets the pad value for the type of a column (IMPORTANTLY) as found in the
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
            xradio_logger().debug(msg)
        else:
            xradio_logger().warning(msg)
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


def _partition_cell_shape_and_dtype(
    main_rows: MainTableRows, col: str
) -> tuple[tuple[int, ...], np.dtype]:
    """
    Cell shape and output dtype of a column, taken from the first row of the
    partition (shape string of the first row, dtype of the first cell as the
    casacore bindings return it: scalar cells come back as Python scalars, so
    for example an int column gives int64). Raises if that cell is undefined.
    """
    table = main_rows.table
    first_row = int(main_rows.rows[0])
    if table.isscalarcol(col):
        extra_dimensions = ()
    else:
        extra_dimensions = parse_shape_string(
            table.getcolshapestring(col, first_row, 1)[0]
        )
    col_dtype = np.array(table.getcell(col, first_row)).dtype
    return extra_dimensions, col_dtype


def read_col_conversion_numpy(
    main_rows: MainTableRows,
    col: str,
    cshape: tuple[int, int],
    tidxs: np.ndarray,
    bidxs: np.ndarray,
) -> np.ndarray:
    """
    Reads a column of the partition rows from the base MAIN table into the
    dense (time, baseline, ...) grid, with bounded read calls of ascending
    rows (straight into the grid where rows map to consecutive cells,
    otherwise through a bounded temporary, see ``read_rows_to_grid``). No
    TaQL selection of the MAIN table is made.

    The grid has the dtype of the partition's first cell. Cells without a row
    are padded with get_pad_value (NaN, FLAG=False, ...) and, for duplicated
    (time, baseline) rows, the last row wins. A column that cannot be read
    (undefined cells, varying cell shapes) raises.

    Parameters
    ----------
    main_rows : MainTableRows
        MAIN table and the partition rows.
    col : str
        Column name.
    cshape : tuple[int, int]
        (n_times, n_baselines) of the grid.
    tidxs : np.ndarray
        Time index of every partition row.
    bidxs : np.ndarray
        Baseline index of every partition row.

    Returns
    -------
    np.ndarray
        The column values on the (time, baseline, ...) grid.
    """
    extra_dimensions, col_dtype = _partition_cell_shape_and_dtype(main_rows, col)
    plan = main_rows.grid_plan(tidxs, bidxs, cshape)
    shape = tuple(cshape) + extra_dimensions
    # padded with get_pad_value (https://github.com/casangi/xradio/issues/219)
    return read_grid(
        main_rows.table, col, plan, shape, col_dtype, max_elems=main_rows.max_elems
    )


def read_col_conversion_dask(
    main_rows: MainTableRows,
    col: str,
    cshape: tuple[int, int],
    tidxs: np.ndarray,
    bidxs: np.ndarray,
    time_chunksize: int,
) -> da.Array:
    """
    The lazy version of read_col_conversion_numpy (parallel_mode="time"): a
    dask array with one block per chunk of times, each block reading the rows
    of its times from the base MAIN table (bounded reads, as
    read_col_conversion_numpy).

    Any row order and missing or duplicated (time, baseline) rows give the
    values of read_col_conversion_numpy (cells without a row are padded with
    get_pad_value, FLAG=False). With casatools (no python-casacore) the blocks
    of a process are read one at a time (``_CASATOOLS_READ_LOCK``): casatools
    tables cannot be read from several threads at once.

    Parameters
    ----------
    main_rows : MainTableRows
        MAIN table and the partition rows.
    col : str
        Column name.
    cshape : tuple[int, int]
        (n_times, n_baselines) of the grid.
    tidxs : np.ndarray
        Time index of every partition row.
    bidxs : np.ndarray
        Baseline index of every partition row.
    time_chunksize : int
        Number of times per block (as dask chunks along time).

    Returns
    -------
    da.Array
        Lazy (time, baseline, ...) array; every block opens the MAIN table by
        name when computed.
    """
    import dask

    extra_dimensions, col_dtype = _partition_cell_shape_and_dtype(main_rows, col)
    in_file = main_rows.name()
    num_utimes, num_baselines = int(cshape[0]), int(cshape[1])
    time_chunks = da.core.normalize_chunks(time_chunksize, (num_utimes,))[0]

    # The rows of every time chunk, computed once per partition and shared by
    # the blocks of every column: the graph references the partition's index
    # arrays (one graph key, so a distributed scheduler sends it once per
    # worker) instead of holding copies per block and column.
    chunk_rows = main_rows.time_chunk_rows(tidxs, bidxs, time_chunks, num_baselines)
    shared_rows = main_rows.time_chunk_rows_delayed(chunk_rows)

    blocks = []
    for k, ntimes in enumerate(time_chunks):
        block_shape = (int(ntimes), num_baselines) + extra_dimensions
        block = dask.delayed(_load_rows_time_chunk, pure=False)(
            in_file,
            col,
            shared_rows,
            k,
            block_shape,
            col_dtype,
            main_rows.max_elems,
        )
        blocks.append(da.from_delayed(block, shape=block_shape, dtype=col_dtype))

    return da.concatenate(blocks, axis=0)


# casatools tables (the shim, used where python-casacore is not installed) must
# not be used from several threads at once: with dask's threaded scheduler the
# concurrent block reads of read_col_conversion_dask crashed, hung or returned
# wrong values (casatools 6.7.0.31). Those block reads open, read, close and
# release their table holding this lock.
_CASATOOLS_READ_LOCK = threading.Lock()


def _load_rows_time_chunk(
    in_file: str,
    col: str,
    chunk_rows: TimeChunkRows,
    k: int,
    shape: tuple[int, ...],
    dtype: np.dtype,
    max_elems: int,
) -> np.ndarray:
    """Read time chunk (block) ``k`` of read_col_conversion_dask."""
    cell_shape = tuple(shape[2:])
    if chunk_rows.chunk_n_rows(k) == 0:  # only padding: no table needed
        return read_time_chunk(None, col, chunk_rows, k, cell_shape, dtype, max_elems)
    one_thread_at_a_time = (
        contextlib.nullcontext()
        if backend_has_in_place_reads()  # python-casacore
        else _CASATOOLS_READ_LOCK
    )
    with one_thread_at_a_time:
        # Opened in the thread/process that computes the block
        with open_table_ro(in_file) as tb_tool:
            values = read_time_chunk(
                tb_tool, col, chunk_rows, k, cell_shape, dtype, max_elems
            )
        del tb_tool  # the table object is destroyed holding the lock
    return values
