import toolviper.utils.logger as logger
import time
from typing import Tuple, Union

import numpy as np
import xarray as xr
import os

from xradio.measurement_set._utils._msv2.subtables import subt_rename_ids
from xradio.measurement_set._utils._msv2._tables.read import (
    load_generic_table,
    convert_casacore_time,
    convert_casacore_time_to_mjd,
    make_taql_where_between_min_max,
    table_exists,
)
from xradio._utils.schema import convert_generic_xds_to_xradio_schema
from xradio.measurement_set._utils._msv2.msv4_sub_xdss import (
    rename_and_interpolate_to_time,
)

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
    partition_polarization: xr.DataArray,
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
    partition_polarization: xr.DataArray
        Polarization labels of this partition, needed if that info is not present in FEED

    Returns
    ----------
        xr.Dataset: Xarray Dataset containing the antenna information.
    """
    ant_xds = xr.Dataset(attrs={"type": "antenna"})

    ant_xds = extract_antenna_info(ant_xds, in_file, antenna_id, telescope_name)

    ant_xds = extract_feed_info(
        ant_xds, in_file, antenna_id, feed_id, spectral_window_id
    )
    # Needed for special SPWs such as ALMA WVR or CHANNEL_AVERAGE data (have no feed info)
    if "polarization_type" not in ant_xds:
        pols_chars = list(partition_polarization.values[0])
        pols_labels = [f"pol_{idx}" for idx in np.arange(0, len(pols_chars))]
        ant_xds = ant_xds.assign_coords(receptor_label=pols_labels)
        pol_type_values = [pols_chars] * len(ant_xds.antenna_name)
        ant_xds = ant_xds.assign_coords(
            polarization_type=(
                ["antenna_name", "receptor_label"],
                pol_type_values,
            )
        )

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
        "DISH_DIAMETER": ["ANTENNA_DISH_DIAMETER", ["antenna_name"]],
    }

    to_new_coords = {
        "NAME": ["antenna_name", ["antenna_name"]],
        "STATION": ["station", ["antenna_name"]],
        "MOUNT": ["mount", ["antenna_name"]],
        # "PHASED_ARRAY_ID": ["phased_array_id", ["antenna_name"]],
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
    ant_xds = ant_xds.assign_coords({"cartesian_pos_label": ["x", "y", "z"]})

    ant_xds = convert_generic_xds_to_xradio_schema(
        generic_ant_xds, ant_xds, to_new_data_variables, to_new_coords
    )

    ant_xds["ANTENNA_DISH_DIAMETER"].attrs.update({"units": ["m"], "type": "quantity"})

    ant_xds["ANTENNA_POSITION"].attrs["coordinate_system"] = "geocentric"
    ant_xds["ANTENNA_POSITION"].attrs["origin_object_name"] = "earth"

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

    ant_xds = ant_xds.assign_coords(
        {
            "telescope_name": (
                "antenna_name",
                np.array([telescope_name for ant in ant_xds["antenna_name"]]),
            )
        }
    )

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

    if not generic_feed_xds:
        # Some MSv2 have a FEED table that does not cover all antenna_id (and feed_id)
        return ant_xds

    feed_spw = np.unique(generic_feed_xds.SPECTRAL_WINDOW_ID)
    if len(feed_spw) == 1 and feed_spw[0] == -1:
        generic_feed_xds = generic_feed_xds.isel(SPECTRAL_WINDOW_ID=0, drop=True)
    else:
        if spectral_window_id not in feed_spw:
            # For some spw the feed table is empty (this is the case with ALMA spw WVR#NOMINAL).
            return ant_xds
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
        "RECEPTOR_ANGLE": [
            "ANTENNA_RECEPTOR_ANGLE",
            ["antenna_name", "receptor_label"],
        ],
        "FOCUS_LENGTH": [
            "ANTENNA_FOCUS_LENGTH",
            ["antenna_name"],
        ],  # optional
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

    # coords["receptor_label"] = "pol_" + np.arange(ant_xds.sizes["receptor_label"]).astype(str) #Works on laptop but fails in github test runner.
    coords = {
        "receptor_label": np.array(
            list(
                map(
                    lambda x, y: x + "_" + y,
                    ["pol"] * ant_xds.sizes["receptor_label"],
                    np.arange(ant_xds.sizes["receptor_label"]).astype(str),
                )
            ),
            dtype=str,
        )
    }

    ant_xds = ant_xds.assign_coords(coords)

    # Correct to expected types. Some ALMA-SD (at least) leave receptor_label, polarization_type columns
    # in the MS empty, causing a type mismatch
    if (
        "polarization_type" in ant_xds.coords
        and ant_xds.coords["polarization_type"].dtype != str
    ):
        ant_xds.coords["polarization_type"] = ant_xds.coords[
            "polarization_type"
        ].astype(str)
    return ant_xds


def create_gain_curve_xds(
    in_file: str, spectral_window_id: int, ant_xds: xr.Dataset
) -> xr.Dataset:
    """
    Produces a gain_curve_xds, reformats MSv2 GAIN CURVE table content to MSv4 schema.

    Parameters
    ----------
    in_file : str
        Path to the input MSv2.
    spectral_window_id : int
        The ID of the spectral window.
    ant_xds : xr.Dataset
        The antenna_xds that has information such as names, stations, etc., for coordinates

    Returns
    -------
    xr.Dataset
        The updated antenna dataset with gain curve information.
    """

    gain_curve_xds = None
    if not table_exists(os.path.join(in_file, "GAIN_CURVE")):
        return gain_curve_xds

    generic_gain_curve_xds = load_generic_table(
        in_file,
        "GAIN_CURVE",
        taql_where=f" where (ANTENNA_ID IN [{','.join(map(str,ant_xds.antenna_id.values))}]) AND (SPECTRAL_WINDOW_ID = {spectral_window_id})",
    )

    if not generic_gain_curve_xds.data_vars:
        # Some times the gain_curve table is empty (this is the case with ngEHT simulation data we have).
        return gain_curve_xds

    assert (
        len(generic_gain_curve_xds.SPECTRAL_WINDOW_ID) == 1
    ), "Only one spectral window is supported."
    generic_gain_curve_xds = generic_gain_curve_xds.isel(
        SPECTRAL_WINDOW_ID=0, drop=True
    )  # Drop the spectral window dimension as it is singleton.

    assert (
        len(generic_gain_curve_xds.TIME) == 1
    ), "Only one gain curve measurement per antenna is supported."
    measured_time = generic_gain_curve_xds.coords["TIME"].values[0]
    generic_gain_curve_xds = generic_gain_curve_xds.isel(TIME=0, drop=True)

    generic_gain_curve_xds = generic_gain_curve_xds.sel(
        ANTENNA_ID=ant_xds.antenna_id, drop=False
    )  # Make sure the antenna_id is in the same order as the xds .

    gain_curve_xds = xr.Dataset(attrs={"type": "gain_curve"})

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

    gain_curve_xds = convert_generic_xds_to_xradio_schema(
        generic_gain_curve_xds,
        gain_curve_xds,
        to_new_data_variables,
        to_new_coords,
    )

    ant_borrowed_coords = {
        "antenna_name": ant_xds.coords["antenna_name"],
        "station": ant_xds.coords["station"],
        "mount": ant_xds.coords["mount"],
        "telescope_name": ant_xds.coords["telescope_name"],
        "receptor_label": ant_xds.coords["receptor_label"],
        "polarization_type": ant_xds.coords["polarization_type"],
    }
    gain_curve_xds = gain_curve_xds.assign_coords(ant_borrowed_coords)

    gain_curve_xds.attrs.update(
        {
            "measured_date": np.datetime_as_string(
                convert_casacore_time([measured_time])[0]
            )
        }
    )

    # correct expected types (for example "GAIN_CURVE" can be float32)
    for data_var in gain_curve_xds:
        if gain_curve_xds.data_vars[data_var].dtype != np.float64:
            gain_curve_xds[data_var] = gain_curve_xds[data_var].astype(np.float64)

    return gain_curve_xds


def create_phase_calibration_xds(
    in_file: str,
    spectral_window_id: int,
    ant_xds: xr.Dataset,
    time_min_max: Tuple[np.float64, np.float64],
    phase_cal_interp_time: Union[xr.DataArray, None] = None,
) -> xr.Dataset:
    """
    Produces a phase_calibration_xds, reformats MSv2 Phase Cal table content to MSv4 schema.

    Parameters
    ----------
    in_file : str
        Path to the input MSv2.
    spectral_window_id : int
        The ID of the spectral window.
    ant_xds : xr.Dataset
        The antenna_xds that has information such as names, stations, etc., for coordinates
    time_min_max : Tuple[np.float46, np.float64]
        Min / max times to constrain loading (usually to the time range relevant to an MSv4)
    interp_time : Union[xr.DataArray, None]
        Time axis to interpolate the data vars to (usually main MSv4 time)

    Returns
    -------
    xr.Dataset
        The updated antenna dataset with phase cal information.
    """

    phase_cal_xds = None
    if not table_exists(os.path.join(in_file, "PHASE_CAL")):
        return phase_cal_xds

    # Only read data between the min and max times of the visibility data in the MSv4.
    taql_time_range = make_taql_where_between_min_max(
        time_min_max, in_file, "PHASE_CAL", "TIME"
    )
    generic_phase_cal_xds = load_generic_table(
        in_file,
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

    phase_cal_xds = xr.Dataset(attrs={"type": "phase_calibration"})
    phase_cal_xds = convert_generic_xds_to_xradio_schema(
        generic_phase_cal_xds, phase_cal_xds, to_new_data_variables, to_new_coords
    )

    phase_cal_xds["PHASE_CAL"] = phase_cal_xds["PHASE_CAL"].transpose(
        "antenna_name", "time_phase_cal", "receptor_label", "tone_label"
    )
    phase_cal_xds["PHASE_CAL_TONE_FREQUENCY"] = phase_cal_xds[
        "PHASE_CAL_TONE_FREQUENCY"
    ].transpose("antenna_name", "time_phase_cal", "receptor_label", "tone_label")

    ant_borrowed_coords = {
        "antenna_name": ant_xds.coords["antenna_name"],
        "station": ant_xds.coords["station"],
        "mount": ant_xds.coords["mount"],
        "telescope_name": ant_xds.coords["telescope_name"],
        "receptor_label": ant_xds.coords["receptor_label"],
        "polarization_type": ant_xds.coords["polarization_type"],
    }
    # phase_cal_xds = phase_cal_xds.assign_coords({"tone_label" : "freq_" + np.arange(phase_cal_xds.sizes["tone_label"]).astype(str)}) #Works on laptop but fails in github test runner.
    tone_label_coord = {
        "tone_label": np.array(
            list(
                map(
                    lambda x, y: x + "_" + y,
                    ["freq"] * phase_cal_xds.sizes["tone_label"],
                    np.arange(phase_cal_xds.sizes["tone_label"]).astype(str),
                )
            )
        )
    }
    phase_cal_xds = phase_cal_xds.assign_coords(ant_borrowed_coords | tone_label_coord)

    # Adjust expected types
    phase_cal_xds["time_phase_cal"] = (
        phase_cal_xds.time_phase_cal.astype("float64").astype("float64") / 10**9
    )

    phase_cal_xds = rename_and_interpolate_to_time(
        phase_cal_xds, "time_phase_cal", phase_cal_interp_time, "phase_cal_xds"
    )

    return phase_cal_xds
