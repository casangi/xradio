import graphviper.utils.logger as logger
import time
from typing import Tuple, Union

import numpy as np
import xarray as xr

from .msv2_to_msv4_meta import column_description_casacore_to_msv4_measure
from .subtables import subt_rename_ids
from ._tables.read import make_taql_where_between_min_max, read_generic_table


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


def create_ant_xds(in_file: str):
    """
    Creates an Antenna Xarray Dataset from a MS v2 ANTENNA table.

    Parameters
    ----------
    in_file : str
        Input MS name.

    Returns
    -------
    xr.Dataset
        Antenna Xarray Dataset.
    """
    # Dictionaries that define the conversion from MSv2 to MSv4:
    to_new_data_variable_names = {
        "position": "POSITION",
        "offset": "FEED_OFFSET",
        "dish_diameter": "DISH_DIAMETER",
    }
    data_variable_dims = {
        "position": ["antenna_id", "xyz_label"],
        "offset": ["antenna_id", "xyz_label"],
        "dish_diameter": ["antenna_id"],
    }
    to_new_coord_names = {
        "name": "name",
        "station": "station",
        "type": "type",
        "mount": "mount",
        "phased_array_id": "phased_array_id",
    }
    coord_dims = {
        "name": ["antenna_id"],
        "station": ["antenna_id"],
        "type": ["antenna_id"],
        "mount": ["antenna_id"],
        "phased_array_id": ["antenna_id"],
    }

    # Read ANTENNA table into a Xarray Dataset.
    generic_ant_xds = read_generic_table(
        in_file,
        "ANTENNA",
        rename_ids=subt_rename_ids["ANTENNA"],
    )

    ant_column_description = generic_ant_xds.attrs["other"]["msv2"]["ctds_attrs"][
        "column_descriptions"
    ]
    # ['OFFSET', 'POSITION', 'TYPE', 'DISH_DIAMETER', 'FLAG_ROW', 'MOUNT', 'NAME', 'STATION']
    ant_xds = xr.Dataset()

    coords = {
        "antenna_id": np.arange(generic_ant_xds.sizes["antenna_id"]),
        "xyz_label": ["x", "y", "z"],
    }
    for key in generic_ant_xds:
        msv4_measure = column_description_casacore_to_msv4_measure(
            ant_column_description[key.upper()]
        )
        if key in to_new_data_variable_names:
            ant_xds[to_new_data_variable_names[key]] = xr.DataArray(
                generic_ant_xds[key].data, dims=data_variable_dims[key]
            )

            if msv4_measure:
                ant_xds[to_new_data_variable_names[key]].attrs.update(msv4_measure)

            if key in ["dish_diameter"]:
                ant_xds[to_new_data_variable_names[key]].attrs.update(
                    {"units": ["m"], "type": "quantity"}
                )

        if key in to_new_coord_names:
            coords[to_new_coord_names[key]] = (
                coord_dims[key],
                generic_ant_xds[key].data,
            )

    ant_xds = ant_xds.assign_coords(coords)
    return ant_xds


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
    # Dictionaries that define the conversion from MSv2 to MSv4:
    to_new_data_variable_names = {
        "h2o": "H2O",
        "ionos_electron": "IONOS_ELECTRON",
        "pressure": "PRESSURE",
        "rel_humidity": "REL_HUMIDITY",
        "temperature": "TEMPERATURE",
        "dew_point": "DEW_POINT",
        "wind_direction": "WIND_DIRECTION",
        "wind_speed": "WIND_SPEED",
    }
    data_variable_dims = {
        "h2o": ["station_id", "time"],
        "ionos_electron": ["station_id", "time"],
        "pressure": ["station_id", "time"],
        "rel_humidity": ["station_id", "time"],
        "temperature": ["station_id", "time"],
        "dew_point": ["station_id", "time"],
        "wind_direction": ["station_id", "time"],
        "wind_speed": ["station_id", "time"],
    }
    to_new_coord_names = {
        # No MS data cols are turned into xds coords
    }
    coord_dims = {
        # No MS data cols are turned into xds coords
    }
    to_new_dim_names = {
        "antenna_id": "station_id",
    }

    # Read WEATHER table into a Xarray Dataset.
    try:
        generic_weather_xds = read_generic_table(
            in_file,
            "WEATHER",
            rename_ids=subt_rename_ids["WEATHER"],
        )
    except ValueError as _exc:
        return None

    generic_weather_xds = generic_weather_xds.rename_dims(to_new_dim_names)

    weather_column_description = generic_weather_xds.attrs["other"]["msv2"][
        "ctds_attrs"
    ]["column_descriptions"]
    # ['ANTENNA_ID', 'TIME', 'INTERVAL', 'H2O', 'IONOS_ELECTRON',
    #  'PRESSURE', 'REL_HUMIDITY', 'TEMPERATURE', 'DEW_POINT',
    #  'WIND_DIRECTION', 'WIND_SPEED']
    weather_xds = xr.Dataset()

    coords = {
        "station_id": generic_weather_xds["station_id"].data,
        "time": generic_weather_xds["time"].data,
    }
    for key in generic_weather_xds:
        msv4_measure = column_description_casacore_to_msv4_measure(
            weather_column_description[key.upper()]
        )
        if key in to_new_data_variable_names:
            var_name = to_new_data_variable_names[key]
            weather_xds[var_name] = xr.DataArray(
                generic_weather_xds[key].data, dims=data_variable_dims[key]
            )

            if msv4_measure:
                weather_xds[var_name].attrs.update(msv4_measure)

            if key in ["interval"]:
                weather_xds[var_name].attrs.update({"units": ["s"], "type": "quantity"})
            elif key in ["h2o"]:
                weather_xds[var_name].attrs.update(
                    {"units": ["/m^2"], "type": "quantity"}
                )
            elif key in ["ionos_electron"]:
                weather_xds[var_name].attrs.update(
                    {"units": ["/m^2"], "type": "quantity"}
                )
            elif key in ["pressure"]:
                weather_xds[var_name].attrs.update(
                    {"units": ["Pa"], "type": "quantity"}
                )
            elif key in ["rel_humidity"]:
                weather_xds[var_name].attrs.update({"units": ["%"], "type": "quantity"})
            elif key in ["temperature"]:
                weather_xds[var_name].attrs.update({"units": ["K"], "type": "quantity"})
            elif key in ["dew_point"]:
                weather_xds[var_name].attrs.update({"units": ["K"], "type": "quantity"})
            elif key in ["wind_direction"]:
                weather_xds[var_name].attrs.update(
                    {"units": ["rad"], "type": "quantity"}
                )
            elif key in ["wind_speed"]:
                weather_xds[var_name].attrs.update(
                    {"units": ["m/s"], "type": "quantity"}
                )

        if key in to_new_coord_names:
            coords[to_new_coord_names[key]] = (
                coord_dims[key],
                generic_weather_xds[key].data,
            )

    weather_xds = weather_xds.assign_coords(coords)
    return weather_xds


def create_pointing_xds(
    in_file: str,
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

    # Dictionaries that define the conversion from MSv2 to MSv4:
    to_new_data_variable_names = {
        # "name": "NAME",   # removed
        # "time_origin": "TIME_ORIGIN",  # removed?
        "direction": "BEAM_POINTING",
        "encoder": "DISH_MEASURED_POINTING",
        "target": "TARGET",  # => attribute?
        "pointing_offset": "POINTING_OFFSET",
        "source_offset": "SOURCE_OFFSET",
        # "pointing_model_id": "POINTING_MODEL_ID",   # removed
        # "tracking": "TRACKING",   # => attribute
        # "on_source": "ON_SOURCE",   # removed
        "over_the_top": "OVER_THE_TOP",
    }
    time_ant_ids = ["time", "antenna_id"]
    data_variable_dims = {
        # "name": ["time", "antenna_id"],   # removed
        # "time_origin": ["time", "antenna_id"],   # removed?
        "direction": ["time", "antenna_id", "direction"],
        "encoder": ["time", "antenna_id", "direction"],
        "target": ["time", "antenna_id", "direction"],
        "pointing_offset": ["time", "antenna_id", "direction"],
        "source_offset": ["time", "antenna_id", "direction"],
        # "pointing_model_id": ["time", "antenna_id"],   # removed
        # "tracking": ["time", "antenna_id"],   # => attribute
        # "on_source": ["time", "antenna_id"],  # removed
        "over_the_top": ["time", "antenna_id"],
    }
    to_new_coord_names = {"ra/dec": "direction"}
    coord_dims = {}

    taql_time_range = make_taql_where_between_min_max(
        time_min_max, in_file, "POINTING", "TIME"
    )
    # Read POINTING table into a Xarray Dataset.
    generic_pointing_xds = read_generic_table(
        in_file,
        "POINTING",
        rename_ids=subt_rename_ids["POINTING"],
        taql_where=taql_time_range,
    )
    if not generic_pointing_xds.data_vars:
        # apparently empty MS/POINTING table => produce empty xds
        return xr.Dataset()

    # Checking a simple way of using only the one single coefficient of the polynomials
    if "n_polynomial" in generic_pointing_xds.sizes:
        size = generic_pointing_xds.sizes["n_polynomial"]
        if size == 1:
            generic_pointing_xds = generic_pointing_xds.sel({"n_polynomial": 0})

    pointing_column_descriptions = generic_pointing_xds.attrs["other"]["msv2"][
        "ctds_attrs"
    ]["column_descriptions"]

    pointing_xds = xr.Dataset()
    for key in generic_pointing_xds:
        if key in to_new_data_variable_names:
            data_var_name = to_new_data_variable_names[key]
            pointing_xds[data_var_name] = xr.DataArray(
                generic_pointing_xds[key].data, dims=data_variable_dims[key]
            )

            msv4_measure = column_description_casacore_to_msv4_measure(
                pointing_column_descriptions[key.upper()]
            )
            if msv4_measure:
                pointing_xds[data_var_name].attrs.update(msv4_measure)

    coords = {
        "time": generic_pointing_xds["time"].values,
        "antenna_id": np.arange(generic_pointing_xds.sizes["antenna_id"]),
        "direction": ["ra", "dec"],
    }
    pointing_xds = pointing_xds.assign_coords(coords)

    # missing attributes
    pointing_xds["time"].attrs.update({"units": ["s"], "type": "quantity"})

    if "tracking" in generic_pointing_xds.data_vars:
        pointing_xds.attrs["tracking"] = generic_pointing_xds.data_vars[
            "tracking"
        ].values[0, 0]

    # Move target from data_vars to attributes?
    move_target_as_attr = False
    if move_target_as_attr:
        target = generic_pointing_xds.data_vars["target"]
        pointing_xds.attrs["target"] = {
            "dims": ["direction"],
            "data": target.values[0, 0].tolist(),
            "attrs": column_description_casacore_to_msv4_measure(
                pointing_column_descriptions["TARGET"]
            ),
        }
    # TODO: move also source_offset/pointing_offset from data_vars to attrs?

    pointing_xds = interpolate_to_time(pointing_xds, interp_time, "pointing_xds")

    logger.debug(f"create_pointing_xds() execution time {time.time() - start:0.2f} s")

    return pointing_xds
