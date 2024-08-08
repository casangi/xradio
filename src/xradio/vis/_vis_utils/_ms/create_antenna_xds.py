import graphviper.utils.logger as logger
import time
from typing import Tuple, Union

import numpy as np
import xarray as xr
import os

from .subtables import subt_rename_ids
from ._tables.read import load_generic_table
from xradio._utils.zarr.common import convert_generic_xds_to_xradio_schema

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
) -> xr.Dataset:
    """
    Create an Xarray Dataset containing antenna information.

    Parameters
    ----------
    in_file : str
        Path to the input MSv2.
    spectral_window_id : int
        Spectral window ID.
    antenna_id : list
        List of antenna IDs.
    feed_id : list
        List of feed IDs.
    telescope_name : str
        Name of the telescope.

    Returns
    ----------
        xr.Dataset: Xarray Dataset containing the antenna information.
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


def extract_antenna_info(
    ant_xds: xr.Dataset, in_file: str, antenna_id: list, telescope_name: str
) -> xr.Dataset:
    """Reformats MSv2 Antenna table content to MSv4 schema.

    Parameters
    ----------
    ant_xds : xr.Dataset
        The dataset that will be updated with antenna information.
    in_file : str
        Path to the input MSv2.
    antenna_id : list
        A list of antenna IDs to extract information for.
    telescope_name : str
        The name of the telescope.

    Returns
    -------
    xr.Dataset
        Dataset updated to contain the antenna information.
    """
    to_new_data_variables = {
        "POSITION": ["ANTENNA_POSITION", ["name", "cartesian_pos_label"]],
        "OFFSET": ["ANTENNA_FEED_OFFSET", ["name", "cartesian_pos_label"]],
        "DISH_DIAMETER": ["ANTENNA_DISH_DIAMETER", ["name"]],
    }

    to_new_coords = {
        "NAME": ["name", ["name"]],
        "STATION": ["station", ["name"]],
        "MOUNT": ["mount", ["name"]],
        "PHASED_ARRAY_ID": ["phased_array_id", ["name"]],
        "antenna_id": ["antenna_id", ["name"]],
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
    ant_xds = ant_xds.assign_coords({"cartesian_pos_label": ["x", "y", "z"]})

    ant_xds = convert_generic_xds_to_xradio_schema(
        generic_ant_xds, ant_xds, to_new_data_variables, to_new_coords
    )

    ant_xds["ANTENNA_DISH_DIAMETER"].attrs.update({"units": ["m"], "type": "quantity"})

    ant_xds["ANTENNA_FEED_OFFSET"].attrs["type"] = "earth_location_offset"
    ant_xds["ANTENNA_FEED_OFFSET"].attrs["coordinate_system"] = "geocentric"
    ant_xds["ANTENNA_POSITION"].attrs["coordinate_system"] = "geocentric"

    if telescope_name in ["ALMA", "VLA", "NOEMA", "EVLA"]:
        # antenna_name = ant_xds["name"].values + "_" + ant_xds["station"].values
        # works on laptop but fails in github test runner with error:
        # numpy.core._exceptions._UFuncNoLoopError: ufunc 'add' did not contain a loop with signature matching types (dtype('<U4'), dtype('<U4')) -> None

        # Also doesn't work on github test runner:
        # antenna_name = ant_xds["name"].values
        # antenna_name = np._core.defchararray.add(antenna_name, "_")
        # antenna_name = np._core.defchararray.add(
        #     antenna_name,
        #     ant_xds["station"].values,
        # )

        # None of the native numpy functions work on the github test runner.
        antenna_name = ant_xds["name"].values
        station = ant_xds["station"].values
        antenna_name = np.array(
            list(map(lambda x, y: x + "_" + y, antenna_name, station))
        )

        ant_xds["name"] = xr.DataArray(antenna_name, dims=["name"])
        ant_xds.attrs["relocatable_antennas"] = True
    else:
        ant_xds.attrs["relocatable_antennas"] = False

    return ant_xds


def extract_feed_info(
    ant_xds: xr.Dataset,
    in_file: str,
    antenna_id: list,
    feed_id: int,
    spectral_window_id: int,
) -> xr.Dataset:
    """
    Reformats MSv2 Feed table content to MSv4 schema.

    Parameters
    ----------
    ant_xds : xr.Dataset
        Xarray Dataset containing antenna information.
    in_file : str
        Path to the input MSv2.
    antenna_id : list
        List of antenna IDs.
    feed_id : int
        Feed ID.
    spectral_window_id : int
        Spectral window ID.

    Returns
    -------
    xr.Dataset
        Dataset updated to contain the feed information.
    """

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

        to_new_data_variables = {
            "BEAM_OFFSET": ["BEAM_OFFSET", ["name", "receptor_name", "sky_dir_label"]],
            "RECEPTOR_ANGLE": ["RECEPTOR_ANGLE", ["name", "receptor_name"]],
            "POLARIZATION_TYPE": ["POLARIZATION_TYPE", ["name", "receptor_name"]],
            # "pol_response": ["POLARIZATION_RESPONSE", ["name", "receptor_name", "receptor_name_"]] #repeated dim creates problems.
            "FOCUS_LENGTH": ["FOCUS_LENGTH", ["name"]],  # optional
            # "position": ["ANTENNA_FEED_OFFSET",["name", "cartesian_pos_label"]] #Will be added to the existing position in ant_xds
        }

        ant_xds = convert_generic_xds_to_xradio_schema(
            generic_feed_xds,
            ant_xds,
            to_new_data_variables,
            to_new_coords={},
        )

        ant_xds["ANTENNA_FEED_OFFSET"] = (
            ant_xds["ANTENNA_FEED_OFFSET"] + generic_feed_xds["POSITION"].data
        )
        coords = {}
        coords["receptor_name"] = np.arange(ant_xds.sizes["receptor_name"]).astype(str)
        coords["sky_dir_label"] = ["ra", "dec"]
        ant_xds = ant_xds.assign_coords(coords)
    return ant_xds


def extract_gain_curve_info(
    ant_xds: xr.Dataset, in_file: str, spectral_window_id: int
) -> xr.Dataset:
    """
    Reformats MSv2 GAIN CURVE table content to MSv4 schema.

    Parameters
    ----------
    ant_xds : xr.Dataset
        The dataset that will be updated with gain curve information.
    in_file : str
        Path to the input MSv2.
    spectral_window_id : int
        The ID of the spectral window.

    Returns
    -------
    xr.Dataset
        The updated antenna dataset with gain curve information.
    """
    if os.path.exists(
        os.path.join(in_file, "GAIN_CURVE")
    ):  # Check if the table exists.
        generic_gain_curve_xds = load_generic_table(
            in_file,
            "GAIN_CURVE",
            taql_where=f" where (ANTENNA_ID IN [{','.join(map(str,ant_xds.antenna_id.values))}]) AND (SPECTRAL_WINDOW_ID = {spectral_window_id})",
        )

        if (
            generic_gain_curve_xds.data_vars
        ):  # Some times the gain_curve table is empty (this is the case with ngEHT simulation data we have).

            assert (
                len(generic_gain_curve_xds.SPECTRAL_WINDOW_ID) == 1
            ), "Only one spectral window is supported."
            generic_gain_curve_xds = generic_gain_curve_xds.isel(
                SPECTRAL_WINDOW_ID=0, drop=True
            )  # Drop the spectral window dimension as it is singleton.

            generic_gain_curve_xds = generic_gain_curve_xds.sel(
                ANTENNA_ID=ant_xds.antenna_id, drop=False
            )  # Make sure the antenna_id is in the same order as the xds .

            to_new_data_variables = {
                "INTERVAL": ["GAIN_CURVE_INTERVAL", ["antenna_id", "gain_curve_time"]],
                "GAIN": [
                    "GAIN_CURVE",
                    ["antenna_id", "gain_curve_time", "poly_term", "receptor_name"],
                ],
                "GAIN_CURVE_SENSITIVITY": [
                    "SENSITIVITY",
                    ["antenna_id", "gain_curve_time", "receptor_name"],
                ],
            }

            to_new_coords = {
                "TIME": ["gain_curve_time", ["gain_curve_time"]],
            }

            ant_xds = convert_generic_xds_to_xradio_schema(
                generic_gain_curve_xds,
                ant_xds,
                to_new_data_variables,
                to_new_coords,
            )
        return ant_xds

    else:
        return ant_xds


def extract_phase_cal_info(ant_xds, path, spectral_window_id):
    """
    Reformats MSv2 Phase Cal table content to MSv4 schema.

    Parameters
    ----------
    ant_xds : xr.Dataset
        The dataset that will be updated with phase cal information.
    in_file : str
        Path to the input MSv2.
    spectral_window_id : int
        The ID of the spectral window.

    Returns
    -------
    xr.Dataset
        The updated antenna dataset with phase cal information.
    """

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

        to_new_data_variables = {
            "INTERVAL": ["PHASE_CAL_INTERVAL", ["antenna_id", "phase_cal_time"]],
            "TONE_FREQUENCY": [
                "PHASE_CAL_TONE_FREQUENCY",
                ["antenna_id", "phase_cal_time", "receptor_name", "tone_label"],
            ],
            "PHASE_CAL": [
                "PHASE_CAL",
                ["antenna_id", "phase_cal_time", "receptor_name", "tone_label"],
            ],
            "CABLE_CAL": ["PHASE_CAL_CABLE_CAL", ["antenna_id", "phase_cal_time"]],
        }

        to_new_coords = {
            "TIME": ["phase_cal_time", ["phase_cal_time"]],
        }

        ant_xds = convert_generic_xds_to_xradio_schema(
            generic_phase_cal_xds, ant_xds, to_new_data_variables, to_new_coords
        )
        return ant_xds

    else:
        return ant_xds
