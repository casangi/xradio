import graphviper.utils.logger as logger
import time
from typing import Tuple, Union

import numpy as np
import xarray as xr
import os

from xradio.vis._vis_utils._ms.subtables import subt_rename_ids
from xradio.vis._vis_utils._ms._tables.read import (
    load_generic_table,
    convert_casacore_time_to_mjd,
    make_taql_where_between_min_max,
)
from xradio._utils.schema import convert_generic_xds_to_xradio_schema
from xradio.vis._vis_utils._ms.msv4_sub_xdss import interpolate_to_time

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
    time_min_max: Tuple[np.float64, np.float64],
    phase_cal_interp_time: Union[xr.DataArray, None] = None,
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
    time_min_max : Tuple[np.float46, np.float64]
        Min / max times to constrain loading (usually to the time range relevant to an MSv4)
    phase_cal_interp_time : Union[xr.DataArray, None]
        Time axis to interpolate the data vars to (usually main MSv4 time)

    Returns
    ----------
        xr.Dataset: Xarray Dataset containing the antenna information.
    """
    ant_xds = xr.Dataset(attrs={"type": "antenna"})

    ant_xds = extract_antenna_info(ant_xds, in_file, antenna_id, telescope_name)

    ant_xds = extract_feed_info(
        ant_xds, in_file, antenna_id, feed_id, spectral_window_id
    )

    ant_xds = extract_phase_cal_info(
        ant_xds, in_file, spectral_window_id, time_min_max, phase_cal_interp_time
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
        "POSITION": ["ANTENNA_POSITION", ["antenna_name", "cartesian_pos_label"]],
        "OFFSET": ["ANTENNA_FEED_OFFSET", ["antenna_name", "cartesian_pos_label"]],
        "DISH_DIAMETER": ["ANTENNA_DISH_DIAMETER", ["antenna_name"]],
    }

    to_new_coords = {
        "NAME": ["antenna_name", ["antenna_name"]],
        "STATION": ["station", ["antenna_name"]],
        "MOUNT": ["mount", ["antenna_name"]],
        "PHASED_ARRAY_ID": ["phased_array_id", ["antenna_name"]],
        "antenna_id": ["antenna_id", ["antenna_name"]],
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
        # antenna_name = ant_xds["antenna_name"].values + "_" + ant_xds["station"].values
        # works on laptop but fails in github test runner with error:
        # numpy.core._exceptions._UFuncNoLoopError: ufunc 'add' did not contain a loop with signature matching types (dtype('<U4'), dtype('<U4')) -> None

        # Also doesn't work on github test runner:
        # antenna_name = ant_xds["antenna_name"].values
        # antenna_name = np._core.defchararray.add(antenna_name, "_")
        # antenna_name = np._core.defchararray.add(
        #     antenna_name,
        #     ant_xds["station"].values,
        # )

        # None of the native numpy functions work on the github test runner.
        antenna_name = ant_xds["antenna_name"].values
        station = ant_xds["station"].values
        antenna_name = np.array(
            list(map(lambda x, y: x + "_" + y, antenna_name, station))
        )

        ant_xds["antenna_name"] = xr.DataArray(antenna_name, dims=["antenna_name"])
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
        taql_where=f" where (ANTENNA_ID IN [{','.join(map(str, ant_xds.antenna_id.values))}]) AND (FEED_ID IN [{','.join(map(str, feed_id))}])",
    )  # Some Lofar and MeerKAT data have the spw column set to -1 so we can't use '(SPECTRAL_WINDOW_ID = {spectral_window_id})'

    feed_spw = np.unique(generic_feed_xds.SPECTRAL_WINDOW_ID)
    if len(feed_spw) == 1 and feed_spw[0] == -1:
        generic_feed_xds = generic_feed_xds.isel(SPECTRAL_WINDOW_ID=0, drop=True)
    else:
        if spectral_window_id not in feed_spw:
            return ant_xds  # For some spw the feed table is empty (this is the case with ALMA spw WVR#NOMINAL).
        else:
            generic_feed_xds = generic_feed_xds.sel(
                SPECTRAL_WINDOW_ID=spectral_window_id, drop=True
            )

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
        "BEAM_OFFSET": [
            "BEAM_OFFSET",
            ["antenna_name", "receptor_label", "sky_dir_label"],
        ],
        "RECEPTOR_ANGLE": ["RECEPTOR_ANGLE", ["antenna_name", "receptor_label"]],
        # "pol_response": ["POLARIZATION_RESPONSE", ["antenna_name", "receptor_label", "receptor_name_"]] #repeated dim creates problems.
        "FOCUS_LENGTH": ["FOCUS_LENGTH", ["antenna_name"]],  # optional
        # "position": ["ANTENNA_FEED_OFFSET",["antenna_name", "cartesian_pos_label"]] #Will be added to the existing position in ant_xds
    }

    to_new_coords = {
        "POLARIZATION_TYPE": ["polarization_type", ["antenna_name", "receptor_label"]]
    }

    ant_xds = convert_generic_xds_to_xradio_schema(
        generic_feed_xds,
        ant_xds,
        to_new_data_variables,
        to_new_coords=to_new_coords,
    )

    # print('ant_xds["ANTENNA_FEED_OFFSET"]',ant_xds["ANTENNA_FEED_OFFSET"].data)
    # print('generic_feed_xds["POSITION"].data',generic_feed_xds["POSITION"].data)
    ant_xds["ANTENNA_FEED_OFFSET"] = (
        ant_xds["ANTENNA_FEED_OFFSET"] + generic_feed_xds["POSITION"].data
    )
    coords = {}
    # coords["receptor_label"] = "pol_" + np.arange(ant_xds.sizes["receptor_label"]).astype(str) #Works on laptop but fails in github test runner.
    coords["receptor_label"] = np.array(
        list(
            map(
                lambda x, y: x + "_" + y,
                ["pol"] * ant_xds.sizes["receptor_label"],
                np.arange(ant_xds.sizes["receptor_label"]).astype(str),
            )
        )
    )

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

            assert (
                len(generic_gain_curve_xds.TIME) == 1
            ), "Only one gain curve measurement per antenna is supported."
            generic_gain_curve_xds = generic_gain_curve_xds.isel(TIME=0, drop=True)

            generic_gain_curve_xds = generic_gain_curve_xds.sel(
                ANTENNA_ID=ant_xds.antenna_id, drop=False
            )  # Make sure the antenna_id is in the same order as the xds .

            to_new_data_variables = {
                "INTERVAL": ["GAIN_CURVE_INTERVAL", ["antenna_name"]],
                "GAIN": [
                    "GAIN_CURVE",
                    ["antenna_name", "poly_term", "receptor_label"],
                ],
                "SENSITIVITY": [
                    "GAIN_CURVE_SENSITIVITY",
                    ["antenna_name", "receptor_label"],
                ],
            }

            to_new_coords = {
                "TYPE": ["gain_curve_type", ["antenna_name"]],
            }

            # print(generic_gain_curve_xds)

            ant_xds = convert_generic_xds_to_xradio_schema(
                generic_gain_curve_xds,
                ant_xds,
                to_new_data_variables,
                to_new_coords,
            )
            ant_xds["GAIN_CURVE"] = ant_xds["GAIN_CURVE"].transpose(
                "antenna_name", "receptor_label", "poly_term"
            )

        return ant_xds

    else:
        return ant_xds


def extract_phase_cal_info(
    ant_xds, path, spectral_window_id, time_min_max, phase_cal_interp_time
):
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
    time_min_max : Tuple[np.float46, np.float64]
        Min / max times to constrain loading (usually to the time range relevant to an MSv4)
    interp_time : Union[xr.DataArray, None]
        Time axis to interpolate the data vars to (usually main MSv4 time)

    Returns
    -------
    xr.Dataset
        The updated antenna dataset with phase cal information.
    """

    if os.path.exists(os.path.join(path, "PHASE_CAL")):

        # Only read data between the min and max times of the visibility data in the MSv4.
        taql_time_range = make_taql_where_between_min_max(
            time_min_max, path, "PHASE_CAL", "TIME"
        )
        generic_phase_cal_xds = load_generic_table(
            path,
            "PHASE_CAL",
            timecols=["TIME"],
            taql_where=f" {taql_time_range} AND (ANTENNA_ID IN [{','.join(map(str,ant_xds.antenna_id.values))}]) AND (SPECTRAL_WINDOW_ID = {spectral_window_id})",
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
            "INTERVAL": ["PHASE_CAL_INTERVAL", ["antenna_name", "time_phase_cal"]],
            "TONE_FREQUENCY": [
                "PHASE_CAL_TONE_FREQUENCY",
                ["antenna_name", "time_phase_cal", "tone_label", "receptor_label"],
            ],
            "PHASE_CAL": [
                "PHASE_CAL",
                ["antenna_name", "time_phase_cal", "tone_label", "receptor_label"],
            ],
            "CABLE_CAL": ["PHASE_CAL_CABLE_CAL", ["antenna_name", "time_phase_cal"]],
        }

        to_new_coords = {
            "TIME": ["time_phase_cal", ["time_phase_cal"]],
        }

        ant_xds = convert_generic_xds_to_xradio_schema(
            generic_phase_cal_xds, ant_xds, to_new_data_variables, to_new_coords
        )
        ant_xds["PHASE_CAL"] = ant_xds["PHASE_CAL"].transpose(
            "antenna_name", "time_phase_cal", "receptor_label", "tone_label"
        )

        ant_xds["PHASE_CAL_TONE_FREQUENCY"] = ant_xds[
            "PHASE_CAL_TONE_FREQUENCY"
        ].transpose("antenna_name", "time_phase_cal", "receptor_label", "tone_label")

        # ant_xds = ant_xds.assign_coords({"tone_label" : "freq_" + np.arange(ant_xds.sizes["tone_label"]).astype(str)}) #Works on laptop but fails in github test runner.
        ant_xds = ant_xds.assign_coords(
            {
                "tone_label": np.array(
                    list(
                        map(
                            lambda x, y: x + "_" + y,
                            ["freq"] * ant_xds.sizes["tone_label"],
                            np.arange(ant_xds.sizes["tone_label"]).astype(str),
                        )
                    )
                )
            }
        )

        ant_xds["time_phase_cal"] = (
            ant_xds.time_phase_cal.astype("float64").astype("float64") / 10**9
        )

        ant_xds = interpolate_to_time(
            ant_xds, phase_cal_interp_time, "antenna_xds", time_name="time_phase_cal"
        )

        time_coord_attrs = {
            "type": "time",
            "units": ["s"],
            "scale": "UTC",
            "format": "UNIX",
        }

        # If we interpolate rename the time_ephemeris_axis axis to time.
        if phase_cal_interp_time is not None:
            time_coord = {"time": ("time_phase_cal", phase_cal_interp_time.data)}
            ant_xds = ant_xds.assign_coords(time_coord)
            ant_xds.coords["time"].attrs.update(time_coord_attrs)
            ant_xds = ant_xds.swap_dims({"time_phase_cal": "time"}).drop_vars(
                "time_phase_cal"
            )

        return ant_xds

    else:
        return ant_xds
