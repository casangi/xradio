import toolviper.utils.logger as logger
import time
from typing import Tuple, Union

import numpy as np
import xarray as xr

from xradio._utils.schema import (
    column_description_casacore_to_msv4_measure,
    convert_generic_xds_to_xradio_schema,
)
from .subtables import subt_rename_ids
from ._tables.read import make_taql_where_between_min_max, load_generic_table


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
        xds = xds.interp({time_name: interp_time}, method=method, assume_sorted=True)
        points_after = xds[time_name].size
        logger.debug(
            f"{message_prefix}: interpolating the time coordinate "
            f"from {points_before} to {points_after} points"
        )

    return xds


def create_weather_xds(in_file: str):
    """
    Creates a Weather Xarray Dataset from a MS v2 WEATHER table.

    Parameters
    ----------
    in_file : str
        Input MS name.

    Returns
    -------
    xr.Dataset
        Weather Xarray Dataset.
    """

    dims_station_time = ["station_id", "time"]
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
        "ANTENNA_ID": ["station_id", ["station_id"]],
        "TIME": ["time", ["time"]],
    }

    # Read WEATHER table into a Xarray Dataset.
    try:
        generic_weather_xds = load_generic_table(
            in_file,
            "WEATHER",
            rename_ids=subt_rename_ids["WEATHER"],
        )
    except ValueError as _exc:
        return None

    weather_xds = xr.Dataset(attrs={"type": "weather"})
    weather_xds = convert_generic_xds_to_xradio_schema(
        generic_weather_xds, weather_xds, to_new_data_variables, to_new_coords
    )

    # correct expected types
    weather_xds["station_id"] = weather_xds["station_id"].astype(np.int64)

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

    for key in generic_pointing_xds:
        if key in to_new_data_variables:
            data_var_name = to_new_data_variables[key]
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
    ant_xds_name_ids : xr.Dataset
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

    time_ant_dims = ["time", "antenna_name"]
    time_ant_dir_dims = time_ant_dims + ["sky_dir_label"]
    to_new_data_variables = {
        "DIRECTION": ["BEAM_POINTING", time_ant_dir_dims],
        "ENCODER": ["DISH_MEASURED_POINTING", time_ant_dir_dims],
        # => attribute?
        "TARGET": ["TARGET", time_ant_dir_dims],
        "POINTING_OFFSET": ["POINTING_OFFSET", time_ant_dir_dims],
        "SOURCE_OFFSET": ["SOURCE_OFFSET", time_ant_dir_dims],
        "OVER_THE_TOP": ["OVER_THE_TOP", time_ant_dims],
    }

    to_new_coords = {
        "TIME": ["time", ["time"]],
        # "ANTENNA_ID": ["antenna_name", ["antenna_name"]],
        "dim_2": ["sky_dir_label", ["sky_dir_label"]],
    }

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

    generic_pointing_xds = correct_generic_pointing_xds(
        generic_pointing_xds, to_new_data_variables
    )

    pointing_xds = xr.Dataset(attrs={"type": "pointing"})
    coords = {
        "antenna_name": ant_xds_name_ids.sel(
            antenna_id=generic_pointing_xds["ANTENNA_ID"]
        ).data,
        "sky_dir_label": ["ra", "dec"],
    }
    pointing_xds = pointing_xds.assign_coords(coords)
    pointing_xds = convert_generic_xds_to_xradio_schema(
        generic_pointing_xds, pointing_xds, to_new_data_variables, to_new_coords
    )

    # Add attributes specific to pointing_xds
    if "TRACKING" in generic_pointing_xds.data_vars:
        pointing_xds.attrs["tracking"] = generic_pointing_xds.data_vars[
            "TRACKING"
        ].values[0, 0]

    # Move target from data_vars to attributes?
    move_target_as_attr = False
    if move_target_as_attr:
        pointing_column_descriptions = generic_pointing_xds.attrs["other"]["msv2"][
            "ctds_attrs"
        ]["column_descriptions"]

        target = generic_pointing_xds.data_vars["TARGET"]
        pointing_xds.attrs["target"] = {
            "dims": ["sky_dir_label"],
            "data": target.values[0, 0].tolist(),
            "attrs": column_description_casacore_to_msv4_measure(
                pointing_column_descriptions["TARGET"]
            ),
        }
    # TODO: move also source_offset/pointing_offset from data_vars to attrs?

    pointing_xds = interpolate_to_time(pointing_xds, interp_time, "pointing_xds")

    logger.debug(f"create_pointing_xds() execution time {time.time() - start:0.2f} s")

    return pointing_xds
