import toolviper.utils.logger as logger
import os
import time
from typing import Tuple, Union

import numpy as np
import xarray as xr

from xradio._utils.coord_math import convert_to_si_units
from xradio._utils.schema import (
    column_description_casacore_to_msv4_measure,
    convert_generic_xds_to_xradio_schema,
)
from .subtables import subt_rename_ids
from ._tables.read import (
    load_generic_table,
    make_taql_where_between_min_max,
    table_exists,
    table_has_column,
)


standard_time_coord_attrs = {
    "type": "time",
    "units": ["s"],
    "scale": "utc",
    "format": "unix",
}


def rename_and_interpolate_to_time(
    xds: xr.Dataset,
    time_initial_name: str,
    interp_time: Union[xr.DataArray, None],
    message_prefix: str,
) -> xr.Dataset:
    """
    This function interpolates the time dimension and renames it:

    - interpolates a time_* dimension to values given in interp_time (presumably the time
    axis of the main xds)
    - rename/replace that time_* dimension to "time", where time_* is a (sub)xds specific
    time axis
    (for example "time_pointing", "time_ephemeris", "time_syscal", "time_phase_cal").

    If interp_time is None this will simply return the input xds without modificaitons.
    Uses interpolate_to_time() for interpolation.
    ...

    Parameters:
    ----------
    xds : xr.Dataset
        Xarray dataset to interpolate (presumably a pointing_xds or an xds of
        ephemeris variables)
    time_initial_name: str = None
        Name of time to be renamed+interpolated. Expected an existing time_* coordinate in the
        dataset
    interp_time:
        Time axis to interpolate the dataset to (usually main MSv4 time)
    message_prefix:
        A prefix for info/debug/etc. messages about the specific xds being interpolated/
        time-renamed

    Returns:
    -------
    renamed_interpolated_xds : xr.Dataset
        xarray dataset with time axis renamed to "time" (from time_name, for example
        "time_ephemeris") and interpolated to interp_time.
    """
    if interp_time is None:
        return xds

    interpolated_xds = interpolate_to_time(
        xds,
        interp_time,
        message_prefix,
        time_name=time_initial_name,
    )

    # rename the time_* axis to time.
    time_coord = {"time": (time_initial_name, interp_time.data)}
    renamed_time_xds = interpolated_xds.assign_coords(time_coord)
    renamed_time_xds.coords["time"].attrs.update(standard_time_coord_attrs)
    renamed_time_xds = renamed_time_xds.swap_dims({time_initial_name: "time"})
    if time_initial_name != "time":
        renamed_time_xds = renamed_time_xds.drop_vars(time_initial_name)

    return renamed_time_xds


def interpolate_to_time(
    xds: xr.Dataset,
    interp_time: Union[xr.DataArray, None],
    message_prefix: str,
    time_name: str = "time",
) -> xr.Dataset:
    """
    Interpolate the time coordinate of the input xarray dataset to the
    a data array. This can be used for example to interpolate a pointing_xds
    to the time coord of the (main) MSv4, or similarly the ephemeris
    data variables of a field_and_source_xds.

    Uses interpolation method "linear", unless the source number of points is
    1 in which case "nearest" is used, to avoid divide-by-zero issues.

    Parameters:
    ----------
    xds : xr.Dataset
        Xarray dataset to interpolate (presumably a pointing_xds or an xds of
        ephemeris variables)
    interp_time : Union[xr.DataArray, None]
        Time axis to interpolate the dataset to (usually main MSv4 time)
    message_prefix: str
        A prefix for info/debug/etc. messages

    Returns:
    -------
    interpolated_xds : xr.Dataset
        xarray dataset with time axis interpolated to interp_time.
    """
    if interp_time is not None:
        points_before = xds[time_name].size
        if points_before > 1:
            method = "linear"
        else:
            method = "nearest"
        xds = xds.interp(
            {time_name: interp_time.data}, method=method, assume_sorted=True
        )
        # scan_name sneaks in as a coordinate of the main time axis, drop it
        if "scan_name" in xds.coords:
            xds = xds.drop_vars("scan_name")
        points_after = xds[time_name].size
        logger.debug(
            f"{message_prefix}: interpolating the time coordinate "
            f"from {points_before} to {points_after} points"
        )

    return xds


def make_taql_where_weather(
    in_file: str, ant_xds_station_name_ids: xr.DataArray
) -> str:
    """
    The use of taql_where with WEATHER is complicated because of ALMA and its
    (ab)use of NS_WX_STATION_ID vs. ANTENNA_ID (all -1) (see read.py). We
    cannot simply use a 'WHERE ANTENNA_ID=...'. This function produces a TaQL
    where string that uses ANTENNA_ID/NS_WX_STATION_ID depending on waht is
    found in the WEATHER subtable.

    Parameters
    ----------
    in_file : str
        Input MS name.
    ant_xds_station_name_ids : xr.DataArray
        station name data array from antenna_xds, with name/id information

    Returns
    -------
    taql_where: str
        WHERE substring safe to use in taql_where when loading WEATHER subtable
    """
    weather_path = os.path.join(in_file, "WEATHER")
    if table_exists(weather_path) and table_has_column(
        weather_path, "NS_WX_STATION_ID"
    ):
        taql_where = f"WHERE (NS_WX_STATION_ID IN [{','.join(map(str, ant_xds_station_name_ids.antenna_id.values))}])"
    else:
        taql_where = f"WHERE (ANTENNA_ID IN [{','.join(map(str, ant_xds_station_name_ids.antenna_id.values))}])"

    return taql_where


def create_weather_xds(in_file: str, ant_xds_station_name_ids: xr.DataArray):
    """
    Creates a Weather Xarray Dataset from a MS v2 WEATHER table.

    Parameters
    ----------
    in_file : str
        Input MS name.
    ant_xds_station_name_ids : xr.DataArray
        station name data array from antenna_xds, with name/id information

    Returns
    -------
    xr.Dataset
        Weather Xarray Dataset.
    """

    try:
        taql_where = make_taql_where_weather(in_file, ant_xds_station_name_ids)
        generic_weather_xds = load_generic_table(
            in_file,
            "WEATHER",
            rename_ids=subt_rename_ids["WEATHER"],
            taql_where=taql_where,
        )
    except ValueError as _exc:
        return None

    if not generic_weather_xds.data_vars:
        # for example when the weather subtable only has info for antennas/stations
        # not present in the MSv4 (no overlap between antennas loaded in ant_xds and weather)
        return None

    weather_xds = xr.Dataset(attrs={"type": "weather"})
    stations_present = ant_xds_station_name_ids.sel(
        antenna_id=generic_weather_xds["ANTENNA_ID"]
    )
    coords = {
        "station_name": stations_present.data,
        "antenna_name": stations_present.coords["antenna_name"].data,
    }
    weather_xds = weather_xds.assign_coords(coords)

    dims_station_time = ["station_name", "time_weather"]
    to_new_data_variables = {
        "H20": ["H2O", dims_station_time],
        "IONOS_ELECTRON": ["IONOS_ELECTRON", dims_station_time],
        "PRESSURE": ["PRESSURE", dims_station_time],
        "REL_HUMIDITY": ["REL_HUMIDITY", dims_station_time],
        "TEMPERATURE": ["TEMPERATURE", dims_station_time],
        "DEW_POINT": ["DEW_POINT", dims_station_time],
        "WIND_DIRECTION": ["WIND_DIRECTION", dims_station_time],
        "WIND_SPEED": ["WIND_SPEED", dims_station_time],
    }

    to_new_coords = {
        "TIME": ["time_weather", ["time_weather"]],
    }

    weather_xds = convert_generic_xds_to_xradio_schema(
        generic_weather_xds, weather_xds, to_new_data_variables, to_new_coords
    )

    # TODO: option to interpolate to main time

    # PRESSURE: hPa in MSv2 specs and some MSs => Pa
    weather_xds = convert_to_si_units(weather_xds)

    # correct expected types (for example "IONOS_ELECTRON", "PRESSURE" can be float32)
    for data_var in weather_xds:
        if weather_xds.data_vars[data_var].dtype != np.float64:
            weather_xds[data_var] = weather_xds[data_var].astype(np.float64)

    return weather_xds


def correct_generic_pointing_xds(
    generic_pointing_xds: xr.Dataset, to_new_data_variables: dict[str, list]
) -> xr.Dataset:
    """
    Takes a (generic) pointing_xds as read from a POINTING subtable of an MSv2
    and tries to correct several deviations from the MSv2 specs seen in
    common test data.
    The problems fixed here include wrong dimensions:
    - for example transposed dimensions with respect to the MSv2 specs (output
    from CASA simulator),
    - missing/additional unexpected dimensions when some of the columns are
    empty (in the sense of "empty casacore cells").

    This function modifies the data arrays of the data vars affected by such
    issues.

    Parameters
    ----------
    generic_pointing_xds: xr.Dataset
        The generic pointing dataset (loaded from MSv2) to be fixed
    to_new_data_variables: dict
        The dict used for convert_generic_xds_to_xradio_schema, which gives all
        the data variables relevant for the final MSv4 dataset.

    Returns:
    --------
    xr.Dataset
        Corrected dataset with dimensions conforming to MSv2 specs.
    """

    correct_pointing_xds = generic_pointing_xds.copy()

    for data_var_name in generic_pointing_xds:
        if data_var_name in to_new_data_variables:
            # Corrects dim sizes of "empty cell" variables, such as empty DIRECTION, TARGET, etc.
            if (
                "dim_2" in generic_pointing_xds.sizes
                and generic_pointing_xds.sizes["dim_2"] == 0
            ):
                # When all direction variables are "empty"
                data_var_data = xr.DataArray(
                    [[[[np.nan, np.nan]]]],
                    dims=generic_pointing_xds.dims,
                ).isel(n_polynomial=0, drop=True)
                correct_pointing_xds[data_var_name].data = data_var_data

            elif (
                "dir" in generic_pointing_xds.sizes
                and generic_pointing_xds.sizes["dir"] == 0
            ):
                # When some direction variables are "empty" but some are populated properly
                if "dim_2" in generic_pointing_xds[key].sizes:
                    data_var_data = xr.DataArray(
                        generic_pointing_xds[key].values,
                        dims=generic_pointing_xds[key].dims,
                    )
                else:
                    shape = tuple(
                        generic_pointing_xds.sizes[dim]
                        for dim in ["TIME", "ANTENNA_ID"]
                    ) + (2,)
                    data_var_data = xr.DataArray(
                        np.full(shape, np.nan),
                        dims=generic_pointing_xds[key].dims,
                    )
                correct_pointing_xds[data_var_name].data = data_var_data

    return correct_pointing_xds


def create_pointing_xds(
    in_file: str,
    ant_xds_name_ids: xr.DataArray,
    time_min_max: Union[Tuple[np.float64, np.float64], None],
    interp_time: Union[xr.DataArray, None] = None,
) -> xr.Dataset:
    """
    Creates a Pointing Xarray Dataset from an MS v2 POINTING (sub)table.

    WIP: details of a few direction variables (and possibly moving some to attributes) to be
    settled (see MSv4 spreadsheet).

    Parameters
    ----------
    in_file : str
        Input MS name.
    ant_xds_name_ids : xr.DataArray
        antenna_name data array from antenna_xds, with name/id information
    time_min_max : tuple
        min / max times values to constrain loading (from the TIME column)
    interp_time : Union[xr.DataArray, None] (Default value = None)
        interpolate time to this (presumably main dataset time)

    Returns
    -------
    xr.Dataset
         Pointing Xarray dataset
    """
    start = time.time()

    taql_time_range = make_taql_where_between_min_max(
        time_min_max, in_file, "POINTING", "TIME"
    )

    if taql_time_range is None:
        taql_where = f"WHERE (ANTENNA_ID IN [{','.join(map(str, ant_xds_name_ids.antenna_id.values))}])"
    else:
        taql_where = (
            taql_time_range
            + f" AND (ANTENNA_ID IN [{','.join(map(str, ant_xds_name_ids.antenna_id.values))}])"
        )
    # Read POINTING table into a Xarray Dataset.
    generic_pointing_xds = load_generic_table(
        in_file,
        "POINTING",
        rename_ids=subt_rename_ids["POINTING"],
        taql_where=taql_where,
    )

    if not generic_pointing_xds.data_vars:
        # apparently empty MS/POINTING table => produce empty xds
        return xr.Dataset()

    # Checking a simple way of using only the one single coefficient of the polynomials
    if "n_polynomial" in generic_pointing_xds.sizes:
        size = generic_pointing_xds.sizes["n_polynomial"]
        if size == 1:
            generic_pointing_xds = generic_pointing_xds.sel({"n_polynomial": 0})
        elif size == 0:
            generic_pointing_xds = generic_pointing_xds.drop_dims("n_polynomial")

    time_ant_dims = ["time_pointing", "antenna_name"]
    time_ant_dir_dims = time_ant_dims + ["local_sky_dir_label"]
    to_new_data_variables = {
        "DIRECTION": ["POINTING_BEAM", time_ant_dir_dims],
        "ENCODER": ["POINTING_DISH_MEASURED", time_ant_dir_dims],
        "OVER_THE_TOP": ["POINTING_OVER_THE_TOP", time_ant_dims],
    }

    to_new_coords = {
        "TIME": ["time_pointing", ["time_pointing"]],
        "dim_2": ["local_sky_dir_label", ["local_sky_dir_label"]],
    }

    generic_pointing_xds = correct_generic_pointing_xds(
        generic_pointing_xds, to_new_data_variables
    )
    pointing_xds = xr.Dataset(attrs={"type": "pointing"})
    coords = {
        "antenna_name": ant_xds_name_ids.sel(
            antenna_id=generic_pointing_xds["ANTENNA_ID"]
        ).data,
        "local_sky_dir_label": ["az", "alt"],
    }
    pointing_xds = pointing_xds.assign_coords(coords)
    pointing_xds = convert_generic_xds_to_xradio_schema(
        generic_pointing_xds, pointing_xds, to_new_data_variables, to_new_coords
    )

    pointing_xds = rename_and_interpolate_to_time(
        pointing_xds, "time_pointing", interp_time, "pointing_xds"
    )

    logger.debug(f"create_pointing_xds() execution time {time.time() - start:0.2f} s")

    return pointing_xds


def prepare_generic_sys_cal_xds(generic_sys_cal_xds: xr.Dataset) -> xr.Dataset:
    """
    A generic_sys_cal_xds loaded with load_generic_table() cannot be easily
    used in convert_generic_xds_to_xradio_schema() to produce an MSv4
    sys_cal_xds dataset, as their structure differs in dimensions and order
    of dimensions.
    This function performs various prepareation steps, such as:
    - filter out dimensions not neeed for an individual MSv4 (SPW, FEED),
    - drop variables loaded from columns with all items set to empty array,
    - transpose the dimensions frequency,receptor
    - fix dimension names (and order) when needed.

    Parameters
    ----------
    generic_sys_cal_xds : xr.Dataset
        generic dataset read from an MSv2 SYSCAL subtable

    Returns
    -------
    generic_sys_cal_xds: xr.Dataset
        System calibration Xarray Dataset prepared for generic conversion
        to MSv4.
    """

    # drop SPW and feed dims
    generic_sys_cal_xds = generic_sys_cal_xds.isel(SPECTRAL_WINDOW_ID=0, drop=True)
    generic_sys_cal_xds = generic_sys_cal_xds.isel(FEED_ID=0, drop=True)

    # Often some of the T*_SPECTRUM are present but all the cells are populated
    # with empty arrays
    empty_arrays_vars = []
    for data_var in generic_sys_cal_xds.data_vars:
        if generic_sys_cal_xds[data_var].size == 0:
            empty_arrays_vars.append(data_var)
    if empty_arrays_vars:
        generic_sys_cal_xds = generic_sys_cal_xds.drop_vars(empty_arrays_vars)

    # Re-arrange receptor and frequency dims depending on input structure
    if (
        "receptor" in generic_sys_cal_xds.sizes
        and "frequency" in generic_sys_cal_xds.sizes
    ):
        # dim_3 can be created for example when the T*_SPECTRUM have varying # channels!
        # more generaly, could transpose with ... to avoid errors with additional spurious dimensions
        if "dim_3" in generic_sys_cal_xds.dims:
            generic_sys_cal_xds = generic_sys_cal_xds.drop_dims("dim_3")
        # From MSv2 tables we get (...,frequency, receptor)
        #  -> transpose to (...,receptor,frequency) ready for MSv4 sys_cal_xds
        generic_sys_cal_xds = generic_sys_cal_xds.transpose(
            "ANTENNA_ID", "TIME", "receptor", "frequency"
        )
    elif (
        "frequency" in generic_sys_cal_xds.sizes
        and not "dim_3" in generic_sys_cal_xds.sizes
    ):
        # because order is (...,frequency,receptor), when frequency is missing
        # receptor can get wrongly labeled as frequency
        generic_sys_cal_xds = generic_sys_cal_xds.rename_dims({"frequency": "receptor"})
    elif (
        "frequency" not in generic_sys_cal_xds.sizes
        and "receptor" in generic_sys_cal_xds.sizes
        and "dim_3" in generic_sys_cal_xds.sizes
    ):
        # different *_SPECTRUM array sizes + some empty arrays can create an additional spurious
        # generic dimension, which should have been "receptor"
        generic_sys_cal_xds = generic_sys_cal_xds.rename_dims({"receptor": "frequency"})
        generic_sys_cal_xds = generic_sys_cal_xds.rename_dims({"dim_3": "receptor"})
        generic_sys_cal_xds = generic_sys_cal_xds.transpose(
            "ANTENNA_ID", "TIME", "receptor", "frequency"
        )
    else:
        raise RuntimeError(
            "Cannot understand the arrangement of dimensions of {generic_sys_cal_xds=}"
        )

    return generic_sys_cal_xds


def create_system_calibration_xds(
    in_file: str,
    main_xds_frequency: xr.DataArray,
    ant_xds: xr.DataArray,
    sys_cal_interp_time: Union[xr.DataArray, None] = None,
):
    """
    Creates a system calibration Xarray Dataset from a MSv2 SYSCAL table.

    Parameters
    ----------
    in_file: str
        Input MS name.
    main_xds_frequency: xr.DataArray
        frequency array of the main xds (MSv4), containing among other things
        spectral_window_id and measures metadata
    ant_xds : xr.Dataset
        The antenna_xds that has information such as names, stations, etc., for coordinates
    sys_cal_interp_time: Union[xr.DataArray, None] = None,
        Time axis to interpolate the data vars to (usually main MSv4 time)

    Returns
    -------
    sys_cal_xds: xr.Dataset
        System calibration Xarray Dataset.
    """

    spectral_window_id = main_xds_frequency.attrs["spectral_window_id"]
    try:
        generic_sys_cal_xds = load_generic_table(
            in_file,
            "SYSCAL",
            rename_ids=subt_rename_ids["SYSCAL"],
            taql_where=(
                f" where (SPECTRAL_WINDOW_ID = {spectral_window_id})"
                f" AND (ANTENNA_ID IN [{','.join(map(str, ant_xds.antenna_id.values))}])"
            ),
        )
    except ValueError as _exc:
        return None

    if not generic_sys_cal_xds.data_vars:
        # even though SYSCAL is an optional subtable, some write it empty
        return None

    generic_sys_cal_xds = prepare_generic_sys_cal_xds(generic_sys_cal_xds)

    mandatory_dimensions = ["antenna_name", "time_system_cal", "receptor_label"]
    if "frequency" not in generic_sys_cal_xds.sizes:
        dims_all = mandatory_dimensions
    else:
        dims_all = mandatory_dimensions + ["frequency_system_cal"]

    to_new_data_variables = {
        "PHASE_DIFF": ["PHASE_DIFFERENCE", ["antenna_name", "time_system_cal"]],
        "TCAL": ["TCAL", dims_all],
        "TCAL_SPECTRUM": ["TCAL", dims_all],
        "TRX": ["TRX", dims_all],
        "TRX_SPECTRUM": ["TRX", dims_all],
        "TSKY": ["TSKY", dims_all],
        "TSKY_SPECTRUM": ["TSKY", dims_all],
        "TSYS": ["TSYS", dims_all],
        "TSYS_SPECTRUM": ["TSYS", dims_all],
        "TANT": ["TANT", dims_all],
        "TANT_SPECTRUM": ["TANT", dims_all],
        "TAN_TSYS": ["TAN_TSYS", dims_all],
        "TANT_SYS_SPECTRUM": ["TANT_TSYS", dims_all],
    }

    to_new_coords = {
        "TIME": ["time_system_cal", ["time_system_cal"]],
        "receptor": ["receptor_label", ["receptor_label"]],
        "frequency": ["frequency_system_cal", ["frequency_system_cal"]],
    }

    sys_cal_xds = xr.Dataset(attrs={"type": "system_calibration"})
    ant_borrowed_coords = {
        "antenna_name": ant_xds.coords["antenna_name"],
        "receptor_label": ant_xds.coords["receptor_label"],
        "polarization_type": ant_xds.coords["polarization_type"],
    }
    sys_cal_xds = sys_cal_xds.assign_coords(ant_borrowed_coords)
    sys_cal_xds = convert_generic_xds_to_xradio_schema(
        generic_sys_cal_xds, sys_cal_xds, to_new_data_variables, to_new_coords
    )

    # Add frequency coord and its measures data, if present
    if "frequency_system_cal" in dims_all:
        frequency_coord = {
            "frequency_system_cal": generic_sys_cal_xds.coords["frequency"].data
        }
        sys_cal_xds = sys_cal_xds.assign_coords(frequency_coord)
        frequency_measure = {
            "type": main_xds_frequency.attrs["type"],
            "units": main_xds_frequency.attrs["units"],
            "observer": main_xds_frequency.attrs["observer"],
        }
        sys_cal_xds.coords["frequency_system_cal"].attrs.update(frequency_measure)

    sys_cal_xds = rename_and_interpolate_to_time(
        sys_cal_xds, "time_system_cal", sys_cal_interp_time, "system_calibration_xds"
    )

    # correct expected types
    for data_var in sys_cal_xds:
        if sys_cal_xds.data_vars[data_var].dtype != np.float64:
            sys_cal_xds[data_var] = sys_cal_xds[data_var].astype(np.float64)

    return sys_cal_xds
