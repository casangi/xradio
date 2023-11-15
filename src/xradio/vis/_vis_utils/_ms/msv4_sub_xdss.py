import numpy as np
import xarray as xr

from .msv2_to_msv4_meta import column_description_casacore_to_msv4_measure
from .subtables import subt_rename_ids
from ._tables.read import read_generic_table


def create_ant_xds(in_file):
    to_new_data_variable_names = {
        "position": "POSITION",
        "offset": "FEED_OFFSET",
        "dish_diameter": "DISH_DIAMETER",
    }
    ant_dims = {
        "position": ["antenna_id", "coordinate"],
        "offset": ["antenna_id", "coordinate"],
        "dish_diameter": ["antenna_id"],
    }
    to_new_coord_names = {
        "name": "name",
        "station": "station",
        "type": "type",
        "mount": "mount",
        "phased_array_id": "phased_array_id",
    }

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

    coords = {"antenna_id": np.arange(generic_ant_xds.dims["antenna_id"])}
    for key in generic_ant_xds:
        msv4_measure = column_description_casacore_to_msv4_measure(
            ant_column_description[key.upper()]
        )
        if key in to_new_data_variable_names:
            ant_xds[to_new_data_variable_names[key]] = xr.DataArray(
                generic_ant_xds[key].data, dims=ant_dims[key]
            )

            if msv4_measure:
                ant_xds[to_new_data_variable_names[key]].attrs.update(msv4_measure)

            if key in ["dish_diameter"]:
                ant_xds[to_new_data_variable_names[key]].attrs.update(
                    {"units": ["m"], "type": "quantity"}
                )

        if key in to_new_coord_names:
            coords[to_new_coord_names[key]] = ("antenna_id", generic_ant_xds[key].data)

    ant_xds = ant_xds.assign_coords(coords)
    return ant_xds
