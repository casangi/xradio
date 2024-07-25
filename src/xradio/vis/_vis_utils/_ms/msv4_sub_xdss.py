import graphviper.utils.logger as logger
import time
from typing import Tuple, Union

import numpy as np
import xarray as xr

from .msv2_to_msv4_meta import column_description_casacore_to_msv4_measure
from .subtables import subt_rename_ids
from ._tables.read import make_taql_where_between_min_max, load_generic_table

from xradio._utils.list_and_array import (
    check_if_consistent,
    unique_1d,
    to_list,
    to_np_array,
)


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


def create_ant_xds(
    in_file: str, spectral_window_id: int, antenna_id: list, feed_id: list
):
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
        "POSITION": "ANTENNA_POSITION",
        "OFFSET": "ANTENNA_FEED_OFFSET",
        "DISH_DIAMETER": "ANTENNA_DISH_DIAMETER",
    }
    data_variable_dims = {
        "POSITION": ["antenna_id", "cartesian_pos_label"],
        "OFFSET": ["antenna_id", "cartesian_pos_label"],
        "DISH_DIAMETER": ["antenna_id"],
    }
    to_new_coord_names = {
        "NAME": "name",
        "STATION": "station",
        "MOUNT": "mount",
        "PHASED_ARRAY_ID": "phased_array_id",
    }

    # coord_dims = {
    #     "name": ["antenna_id"],
    #     "station": ["antenna_id"],
    #     "mount": ["antenna_id"],
    #     "phased_array_id": ["antenna_id"],
    #     "POSITION": "POSITION",
    #     "OFFSET": "FEED_OFFSET",
    #     "DISH_DIAMETER": "DISH_DIAMETER",
    # }
    # data_variable_dims = {
    #     "POSITION": ["antenna_id", "xyz_label"],
    #     "OFFSET": ["antenna_id", "xyz_label"],
    #     "DISH_DIAMETER": ["antenna_id"],
    # }
    # to_new_coord_names = {
    #     "NAME": "name",
    #     "STATION": "station",
    #     "TYPE": "type",
    #     "MOUNT": "mount",
    #     "PHASED_ARRAY_ID": "phased_array_id",
    # }
    coord_dims = {
        "NAME": ["antenna_id"],
        "STATION": ["antenna_id"],
        "TYPE": ["antenna_id"],
        "MOUNT": ["antenna_id"],
        "PHASED_ARRAY_ID": ["antenna_id"],
    }

    # Read ANTENNA table into a Xarray Dataset.
    unique_antenna_id = unique_1d(
        antenna_id
    )  # Also ensures that it is sorted otherwise TaQL will give wrong results.

    generic_ant_xds = load_generic_table(
        in_file,
        "ANTENNA",
        rename_ids=subt_rename_ids["ANTENNA"],
        taql_where=f" where (ROWID() IN [{','.join(map(str,unique_antenna_id))}])",  # order is not guaranteed
    )
    generic_ant_xds = generic_ant_xds.assign_coords({"antenna_id": unique_antenna_id})
    generic_ant_xds = generic_ant_xds.sel(
        antenna_id=antenna_id, drop=False
    )  # Make sure the antenna_id order is correct.

    ant_column_description = generic_ant_xds.attrs["other"]["msv2"]["ctds_attrs"][
        "column_descriptions"
    ]

    # ['OFFSET', 'POSITION', 'DISH_DIAMETER', 'FLAG_ROW', 'MOUNT', 'NAME', 'STATION']
    ant_xds = xr.Dataset()
    ant_xds = ant_xds.assign_coords(
        {"antenna_id": antenna_id, "cartesian_pos_label": ["x", "y", "z"]}
    )

    coords = {}
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

            if key in ["DISH_DIAMETER"]:
                ant_xds[to_new_data_variable_names[key]].attrs.update(
                    {"units": ["m"], "type": "quantity"}
                )

        if key in to_new_coord_names:
            coords[to_new_coord_names[key]] = (
                coord_dims[key],
                generic_ant_xds[key].data,
            )

    ant_xds["ANTENNA_FEED_OFFSET"].attrs["type"] = "earth_location_offset"
    ant_xds["ANTENNA_FEED_OFFSET"].attrs["coordinate_system"] = "geocentric"
    ant_xds["ANTENNA_POSITION"].attrs["coordinate_system"] = "geocentric"

    # Extract feed information
    generic_feed_xds = load_generic_table(
        in_file,
        "FEED",
        rename_ids=subt_rename_ids["FEED"],
        taql_where=f" where (ANTENNA_ID IN [{','.join(map(str, unique_antenna_id))}]) AND (FEED_ID IN [{','.join(map(str, feed_id))}])",
    )  # Some Lofar and MeerKAT data have the spw column set to -1 so we can't use '(SPECTRAL_WINDOW_ID = {spectral_window_id})'

    if "SPECTRAL_WINDOW_ID" in generic_feed_xds and not all(
        generic_feed_xds.SPECTRAL_WINDOW_ID == -1
    ):
        generic_feed_xds = generic_feed_xds.where(
            generic_feed_xds.SPECTRAL_WINDOW_ID == spectral_window_id, drop=True
        )
    if "row" in generic_feed_xds and len(generic_feed_xds.row) > 0:
        # Some times the feed table is empty (this is the case with ALMA spw WVR#NOMINAL).
        assert len(generic_feed_xds.ANTENNA_ID) == len(
            ant_xds.antenna_id
        ), "Can only process feed table with a single time entry for an antenna and spectral_window_id."
        generic_feed_xds = generic_feed_xds.set_xindex(
            "ANTENNA_ID"
        )  # Allows for non-dimension coordinate selection.
        generic_feed_xds = generic_feed_xds.sel(
            ANTENNA_ID=ant_xds.antenna_id
        )  # Make sure the antenna_id is in the same order as the xds.

        num_receptors = np.ravel(generic_feed_xds.NUM_RECEPTORS)
        num_receptors = unique_1d(num_receptors[~np.isnan(num_receptors)])

        assert (
            len(num_receptors) == 1
        ), "The number of receptors must be constant in feed table."

        feed_column_description = generic_feed_xds.attrs["other"]["msv2"]["ctds_attrs"][
            "column_descriptions"
        ]

        to_new_data_variable_names = {
            "BEAM_OFFSET": "BEAM_OFFSET",
            "RECEPTOR_ANGLE": "RECEPTOR_ANGLE",
            "POLARIZATION_TYPE": "POLARIZATION_TYPE",
            # "pol_response": "POLARIZATION_RESPONSE", ?repeated dim creates problems.
            "FOCUS_LENGTH": "FOCUS_LENGTH",  # optional
            # "position": "ANTENNA_FEED_OFFSET" #Will be added to the existing position in ant_xds
        }

        data_variable_dims = {
            "BEAM_OFFSET": ["antenna_id", "receptor_name", "sky_dir_label"],
            "RECEPTOR_ANGLE": ["antenna_id", "receptor_name"],
            "POLARIZATION_TYPE": ["antenna_id", "receptor_name"],
            # "pol_response": ["antenna_id", "receptor_name", "receptor_name_"],
            "FOCUS_LENGTH": ["antenna_id"],
            # "position": ["antenna_id", "cartesian_pos_label"],
        }

        for key in generic_feed_xds:
            msv4_measure = column_description_casacore_to_msv4_measure(
                feed_column_description[key.upper()]
            )
            if key in to_new_data_variable_names:
                ant_xds[to_new_data_variable_names[key]] = xr.DataArray(
                    generic_feed_xds[key].data, dims=data_variable_dims[key]
                )

                if msv4_measure:
                    ant_xds[to_new_data_variable_names[key]].attrs.update(msv4_measure)

            if key in to_new_coord_names:
                coords[to_new_coord_names[key]] = (
                    coord_dims[key],
                    generic_feed_xds[key].data,
                )

        ant_xds["ANTENNA_FEED_OFFSET"] = (
            ant_xds["ANTENNA_FEED_OFFSET"] + generic_ant_xds["POSITION"].data
        )

        coords["receptor_name"] = np.arange(ant_xds.sizes["receptor_name"]).astype(str)

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
    # Dict from col/data_var names in generic_weather_xds (from MSv2) to MSV4
    # weather_xds produced here
    to_new_data_variable_names = {
        "H2O": "H2O",
        "IONOS_ELECTRON": "IONOS_ELECTRON",
        "PRESSURE": "PRESSURE",
        "REL_HUMIDITY": "REL_HUMIDITY",
        "TEMPERATURE": "TEMPERATURE",
        "DEW_POINT": "DEW_POINT",
        "WIND_DIRECTION": "WIND_DIRECTION",
        "WIND_SPEED": "WIND_SPEED",
    }
    data_variable_dims = {
        "H2O": ["station_id", "time"],
        "IONOS_ELECTRON": ["station_id", "time"],
        "PRESSURE": ["station_id", "time"],
        "REL_HUMIDITY": ["station_id", "time"],
        "TEMPERATURE": ["station_id", "time"],
        "DEW_POINT": ["station_id", "time"],
        "WIND_DIRECTION": ["station_id", "time"],
        "WIND_SPEED": ["station_id", "time"],
    }
    to_new_coord_names = {
        # No MS data cols are turned into xds coords
    }
    coord_dims = {
        # No MS data cols are turned into xds coords
    }
    to_new_dim_names = {
        "ANTENNA_ID": "STATION_ID",
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

    generic_weather_xds = generic_weather_xds.rename_dims(to_new_dim_names)

    weather_column_description = generic_weather_xds.attrs["other"]["msv2"][
        "ctds_attrs"
    ]["column_descriptions"]
    # ['ANTENNA_ID', 'TIME', 'INTERVAL', 'H2O', 'IONOS_ELECTRON',
    #  'PRESSURE', 'REL_HUMIDITY', 'TEMPERATURE', 'DEW_POINT',
    #  'WIND_DIRECTION', 'WIND_SPEED']
    weather_xds = xr.Dataset()

    coords = {
        "station_id": generic_weather_xds["STATION_ID"].data,
        "time": generic_weather_xds["TIME"].data,
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

            if key in ["INTERVAL"]:
                weather_xds[var_name].attrs.update({"units": ["s"], "type": "quantity"})
            elif key in ["H2O"]:
                weather_xds[var_name].attrs.update(
                    {"units": ["/m^2"], "type": "quantity"}
                )
            elif key in ["IONOS_ELECTRON"]:
                weather_xds[var_name].attrs.update(
                    {"units": ["/m^2"], "type": "quantity"}
                )
            elif key in ["PRESSURE"]:
                weather_xds[var_name].attrs.update(
                    {"units": ["Pa"], "type": "quantity"}
                )
            elif key in ["REL_HUMIDITY"]:
                weather_xds[var_name].attrs.update({"units": ["%"], "type": "quantity"})
            elif key in ["TEMPERATURE"]:
                weather_xds[var_name].attrs.update({"units": ["K"], "type": "quantity"})
            elif key in ["DEW_POINT"]:
                weather_xds[var_name].attrs.update({"units": ["K"], "type": "quantity"})
            elif key in ["WIND_DIRECTION"]:
                weather_xds[var_name].attrs.update(
                    {"units": ["rad"], "type": "quantity"}
                )
            elif key in ["WIND_SPEED"]:
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
        "DIRECTION": "BEAM_POINTING",
        "ENCODER": "DISH_MEASURED_POINTING",
        "TARGET": "TARGET",  # => attribute?
        "POINTING_OFFSET": "POINTING_OFFSET",
        "SOURCE_OFFSET": "SOURCE_OFFSET",
        # "pointing_model_id": "POINTING_MODEL_ID",   # removed
        # "tracking": "TRACKING",   # => attribute
        # "on_source": "ON_SOURCE",   # removed
        "OVER_THE_TOP": "OVER_THE_TOP",
    }
    time_ant_dims = ["time", "antenna_id"]
    time_ant_dir_dims = time_ant_dims + ["direction"]
    data_variable_dims = {
        # "name": ["time", "antenna_id"],   # removed
        # "time_origin": ["time", "antenna_id"],   # removed?
        "DIRECTION": time_ant_dir_dims,
        "ENCODER": time_ant_dir_dims,
        "TARGET": time_ant_dir_dims,
        "POINTING_OFFSET": time_ant_dir_dims,
        "SOURCE_OFFSET": time_ant_dir_dims,
        # "pointing_model_id": ["time", "antenna_id"],   # removed
        # "tracking": ["time", "antenna_id"],   # => attribute
        # "on_source": ["time", "antenna_id"],  # removed
        "OVER_THE_TOP": time_ant_dims,
    }
    # Unused here
    # to_new_coord_names = {"ra/dec": "direction"}
    # coord_dims = {}

    taql_time_range = make_taql_where_between_min_max(
        time_min_max, in_file, "POINTING", "TIME"
    )
    # Read POINTING table into a Xarray Dataset.
    generic_pointing_xds = load_generic_table(
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
            # Corrects dim sizes of "empty cell" variables, such as empty DIRECTION, TARGET, etc.
            # TODO: this should be moved to a function when/if stable - perhaps 'correct_generic_pointing_xds'
            if (
                "dim_2" in generic_pointing_xds.sizes
                and generic_pointing_xds.sizes["dim_2"] == 0
            ):
                # When all direction variables are "empty"
                data_var_data = xr.DataArray(
                    [[[[np.nan, np.nan]]]],
                    dims=generic_pointing_xds.dims,
                ).isel(n_polynomial=0, drop=True)
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
            else:
                data_var_data = generic_pointing_xds[key].data

            pointing_xds[data_var_name] = xr.DataArray(
                data_var_data, dims=data_variable_dims[key]
            )

            msv4_measure = column_description_casacore_to_msv4_measure(
                pointing_column_descriptions[key.upper()]
            )
            if msv4_measure:
                pointing_xds[data_var_name].attrs.update(msv4_measure)

    coords = {
        "time": generic_pointing_xds["TIME"].values,
        "antenna_id": np.arange(generic_pointing_xds.sizes["ANTENNA_ID"]),
        "direction": ["ra", "dec"],
    }
    pointing_xds = pointing_xds.assign_coords(coords)

    # missing attributes
    pointing_xds["time"].attrs.update({"units": ["s"], "type": "quantity"})

    if "TRACKING" in generic_pointing_xds.data_vars:
        pointing_xds.attrs["tracking"] = generic_pointing_xds.data_vars[
            "TRACKING"
        ].values[0, 0]

    # Move target from data_vars to attributes?
    move_target_as_attr = False
    if move_target_as_attr:
        target = generic_pointing_xds.data_vars["TARGET"]
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
