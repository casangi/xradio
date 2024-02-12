import numpy as np
import xarray as xr

from .msv2_to_msv4_meta import column_description_casacore_to_msv4_measure
from .subtables import subt_rename_ids
from ._tables.read import read_generic_table


def create_ant_xds(in_file: str):
    """Creates an Antenna Xarray Dataset from a MS v2 ANTENNA table.

    Parameters
    ----------
    in_file : str
        Input MS name.

    Returns
    -------
    ant_xds : xarray.Dataset
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
    """Creates a Weather Xarray Dataset from a MS v2 WEATHER table.

    Parameters
    ----------
    in_file : str
        Input MS name.

    Returns
    -------
    weather_xds : xarray.Dataset
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
        "station_id": generic_weather_xds["station_id"],
        "time": generic_weather_xds["time"],
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
