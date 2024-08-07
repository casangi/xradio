import graphviper.utils.logger as logger
import time
from typing import Tuple, Union

import numpy as np
import xarray as xr
import os

from .msv2_to_msv4_meta import column_description_casacore_to_msv4_measure
from .subtables import subt_rename_ids
from ._tables.read import load_generic_table

from xradio._utils.list_and_array import (
    check_if_consistent,
    unique_1d,
    to_list,
    to_np_array,
)


def create_antenna_xds(
    in_file: str,
    spectral_window_id: int,
    antenna_id: list,
    feed_id: list,
    telescope_name: str,
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

    ant_xds = xr.Dataset()

    ant_xds = extract_antenna_info(ant_xds, in_file, antenna_id, telescope_name)

    ant_xds = extract_feed_info(
        ant_xds, in_file, antenna_id, feed_id, spectral_window_id
    )

    ant_xds = extract_phase_cal_info(
        ant_xds, in_file, spectral_window_id
    )  # Only used in VLBI.
    ant_xds = extract_gain_curve_info(
        ant_xds, in_file, spectral_window_id
    )  # Only used in VLBI.

    ant_xds.attrs["overall_telescope_name"] = telescope_name
    return ant_xds


def extract_feed_info(ant_xds, in_file, antenna_id, feed_id, spectral_window_id):
    # Extract feed information
    generic_feed_xds = load_generic_table(
        in_file,
        "FEED",
        rename_ids=subt_rename_ids["FEED"],
        taql_where=f" where (ANTENNA_ID IN [{','.join(map(str, ant_xds.antenna_id.values))}]) AND (FEED_ID IN [{','.join(map(str, feed_id))}]) AND (SPECTRAL_WINDOW_ID = {spectral_window_id})",
    )  # Some Lofar and MeerKAT data have the spw column set to -1 so we can't use '(SPECTRAL_WINDOW_ID = {spectral_window_id})'

    if (
        generic_feed_xds.data_vars
    ):  # Some times the feed table is empty (this is the case with ALMA spw WVR#NOMINAL).
        assert (
            len(generic_feed_xds.SPECTRAL_WINDOW_ID) == 1
        ), "Only one spectral window is supported."
        generic_feed_xds = generic_feed_xds.isel(
            SPECTRAL_WINDOW_ID=0, drop=True
        )  # Only one spectral window is present.

        assert len(generic_feed_xds.TIME) == len(
            antenna_id
        ), "Can only process feed table with a single time entry for an feed, antenna and spectral_window_id."
        generic_feed_xds = generic_feed_xds.sel(
            ANTENNA_ID=antenna_id, drop=False
        )  # Make sure the antenna_id is in the same order as the xds.

        num_receptors = np.ravel(generic_feed_xds.NUM_RECEPTORS)
        num_receptors = unique_1d(num_receptors[~np.isnan(num_receptors)])

        assert (
            len(num_receptors) == 1
        ), "The number of receptors must be constant in feed table."

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

        ant_xds = convert_generic_xds_to_msv4_xds(
            generic_feed_xds,
            ant_xds,
            to_new_data_variable_names,
            data_variable_dims,
            coord_dims={},
            to_new_coord_names={},
        )

        ant_xds["ANTENNA_FEED_OFFSET"] = (
            ant_xds["ANTENNA_FEED_OFFSET"] + generic_feed_xds["POSITION"].data
        )
        coords = {}
        coords["receptor_name"] = np.arange(ant_xds.sizes["receptor_name"]).astype(str)
        ant_xds = ant_xds.assign_coords(coords)
    return ant_xds


def extract_antenna_info(ant_xds, in_file, antenna_id, telescope_name):

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

    # ['OFFSET', 'POSITION', 'DISH_DIAMETER', 'FLAG_ROW', 'MOUNT', 'NAME', 'STATION']
    ant_xds = xr.Dataset()
    ant_xds = ant_xds.assign_coords(
        {"antenna_id": antenna_id, "cartesian_pos_label": ["x", "y", "z"]}
    )

    ant_xds = convert_generic_xds_to_msv4_xds(
        generic_ant_xds,
        ant_xds,
        to_new_data_variable_names,
        data_variable_dims,
        coord_dims,
        to_new_coord_names,
    )

    ant_xds["ANTENNA_DISH_DIAMETER"].attrs.update({"units": ["m"], "type": "quantity"})

    ant_xds["ANTENNA_FEED_OFFSET"].attrs["type"] = "earth_location_offset"
    ant_xds["ANTENNA_FEED_OFFSET"].attrs["coordinate_system"] = "geocentric"
    ant_xds["ANTENNA_POSITION"].attrs["coordinate_system"] = "geocentric"

    if telescope_name in ["ALMA", "VLA", "NOEMA", "EVLA"]:
        # ant_name = ant_xds["name"].values + "_" + ant_xds["station"].values
        # works on laptop but fails in github test runner with error:
        # numpy.core._exceptions._UFuncNoLoopError: ufunc 'add' did not contain a loop with signature matching types (dtype('<U4'), dtype('<U4')) -> None

        # Have to use private numpy functions to get around this.
        antenna_name = ant_xds["name"].values
        antenna_name = np._core.defchararray.add(antenna_name, "_")
        antenna_name = np._core.defchararray.add(
            antenna_name,
            ant_xds["station"].values,
        )

        ant_xds["name"] = xr.DataArray(antenna_name, dims=["antenna_id"])
        ant_xds.attrs["relocatable_antennas"] = True
    else:
        ant_xds.attrs["relocatable_antennas"] = False

    return ant_xds


def extract_gain_curve_info(ant_xds, path, spectral_window_id):

    if os.path.exists(os.path.join(path, "GAIN_CURVE")):
        generic_gain_curve_xds = load_generic_table(
            path,
            "GAIN_CURVE",
            taql_where=f" where (ANTENNA_ID IN [{','.join(map(str,ant_xds.antenna_id.values))}]) AND (SPECTRAL_WINDOW_ID = {spectral_window_id})",
        )

        if (
            generic_gain_curve_xds.data_vars
        ):  # Some times the gain_curve table is empty (this is the case with ngEHT simulation).

            assert (
                len(generic_gain_curve_xds.SPECTRAL_WINDOW_ID) == 1
            ), "Only one spectral window is supported."
            generic_gain_curve_xds = generic_gain_curve_xds.isel(
                SPECTRAL_WINDOW_ID=0, drop=True
            )  # Drop the spectral window dimension as it is singleton.

            generic_gain_curve_xds = generic_gain_curve_xds.sel(
                ANTENNA_ID=ant_xds.antenna_id, drop=False
            )  # Make sure the antenna_id is in the same order as the xds .

            to_new_data_variable_names = {
                "INTERVAL": "GAIN_CURVE_INTERVAL",
                "GAIN": "GAIN_CURVE",
                "GAIN_CURVE_SENSITIVITY": "SENSITIVITY",
            }

            data_variable_dims = {
                "INTERVAL": ["antenna_id", "gain_curve_time"],
                "GAIN": ["antenna_id", "gain_curve_time", "poly_term", "receptor_name"],
                "SENSITIVITY": ["antenna_id", "gain_curve_time", "receptor_name"],
            }

            to_new_coord_names = {
                "TIME": "gain_curve_time",
            }

            ant_xds = convert_generic_xds_to_msv4_xds(
                generic_gain_curve_xds,
                ant_xds,
                to_new_data_variable_names,
                data_variable_dims,
                coord_dims={},
                to_new_coord_names=to_new_coord_names,
            )
        return ant_xds

    else:
        return ant_xds


def extract_phase_cal_info(ant_xds, path, spectral_window_id):

    if os.path.exists(os.path.join(path, "PHASE_CAL")):
        generic_phase_cal_xds = load_generic_table(
            path,
            "PHASE_CAL",
            taql_where=f" where (ANTENNA_ID IN [{','.join(map(str,ant_xds.antenna_id.values))}]) AND (SPECTRAL_WINDOW_ID = {spectral_window_id})",
        )

        assert (
            len(generic_phase_cal_xds.SPECTRAL_WINDOW_ID) == 1
        ), "Only one spectral window is supported."
        generic_phase_cal_xds = generic_phase_cal_xds.isel(
            SPECTRAL_WINDOW_ID=0, drop=True
        )  # Drop the spectral window dimension as it is singleton.

        generic_phase_cal_xds = generic_phase_cal_xds.sel(
            ANTENNA_ID=ant_xds.antenna_id, drop=False
        )  # Make sure the antenna_id is in the same order as the xds.

        to_new_data_variable_names = {
            "INTERVAL": "PHASE_CAL_INTERVAL",
            "TONE_FREQUENCY": "PHASE_CAL_TONE_FREQUENCY",
            "PHASE_CAL": "PHASE_CAL",
            "CABLE_CAL": "PHASE_CAL_CABLE_CAL",
        }

        data_variable_dims = {
            "INTERVAL": ["antenna_id", "phase_cal_time"],
            "TONE_FREQUENCY": [
                "antenna_id",
                "phase_cal_time",
                "receptor_name",
                "tone_label",
            ],
            "PHASE_CAL": [
                "antenna_id",
                "phase_cal_time",
                "receptor_name",
                "tone_label",
            ],
            "CABLE_CAL": ["antenna_id", "phase_cal_time"],
        }

        to_new_coord_names = {
            "TIME": "phase_cal_time",
        }

        ant_xds = convert_generic_xds_to_msv4_xds(
            generic_phase_cal_xds,
            ant_xds,
            to_new_data_variable_names,
            data_variable_dims,
            coord_dims={},
            to_new_coord_names=to_new_coord_names,
        )

        return ant_xds

    else:
        return ant_xds


def convert_generic_xds_to_msv4_xds(
    generic_xds,
    msv4_xds,
    to_new_data_variable_names,
    data_variable_dims,
    coord_dims,
    to_new_coord_names,
):
    column_description = generic_xds.attrs["other"]["msv2"]["ctds_attrs"][
        "column_descriptions"
    ]
    coords = {}
    for key in generic_xds:
        msv4_measure = column_description_casacore_to_msv4_measure(
            column_description[key.upper()]
        )
        if key in to_new_data_variable_names:
            msv4_xds[to_new_data_variable_names[key]] = xr.DataArray(
                generic_xds[key].data, dims=data_variable_dims[key]
            )

            if msv4_measure:
                msv4_xds[to_new_data_variable_names[key]].attrs.update(msv4_measure)

        if key in to_new_coord_names:
            coords[to_new_coord_names[key]] = (
                coord_dims[key],
                generic_xds[key].data,
            )
    msv4_xds = msv4_xds.assign_coords(coords)
    return msv4_xds