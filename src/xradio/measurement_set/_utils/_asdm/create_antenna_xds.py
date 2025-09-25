import numpy as np
import pandas as pd
import xarray as xr


import pyasdm

import toolviper.utils.logger as logger

from xradio._utils.dict_helpers import make_quantity_attrs
from xradio.measurement_set._utils._asdm._utils.metadata_tables import (
    exp_asdm_table_to_df,
)


def create_antenna_xds(
    asdm: pyasdm.ASDM,
    num_antenna: int,
    spectral_window_id: int,
    polarization: xr.DataArray,
) -> xr.Dataset:
    """
    Create an xarray Dataset with antenna metadata from ASDM.
    This function extracts antenna-related information from an ASDM (ALMA Science Data Model)
    and creates an xarray Dataset containing antenna metadata including positions, dish diameters,
    station information and mount types.

    Parameters
    ----------
    asdm : pyasdm.ASDM
        The ASDM object containing the source data
    num_antenna : int
        Number of antennas in the array
    polarization : xr.DataArray
        DataArray containing polarization information (needed if
        that info is not present in the Feed table)

    Returns
    -------
    xr.Dataset
        Dataset containing antenna metadata with the following variables and coordinates:
        - ANTENNA_DISH_DIAMETER: dish diameter in meters for each antenna
        - ANTENNA_POSITION: cartesian position coordinates (x,y,z) in ITRS frame
        Coordinates:
        - antenna_name: names of the antennas
        - cartesian_pos_label: position coordinate labels (x,y,z)
        - station: station names for each antenna
        - mount: mount type for each antenna

    Notes
    -----
    The function currently assumes relocatable antennas and ALT-AZ mount types.
    Some features like receptor angles and focus lengths are not yet implemented.
    """

    xds = xr.Dataset(
        attrs={
            "type": "antenna",
            # SDM: EVLA and ALMA -> so assume alwasy true?
            "relocatable_antennas": True,
        },
    )

    sdm_antenna_attrs = [
        "antennaId",
        "name",
        "dishDiameter",
        "position",
        "offset",
        "stationId",
    ]
    antenna_df = exp_asdm_table_to_df(asdm, "Antenna", sdm_antenna_attrs)
    if num_antenna != antenna_df.shape[0]:
        raise RuntimeError(
            f"When creating antenna_xds, the expected {num_antenna=}, while "
            f"the antennas found in the Antenna table are {antenna_df.shape[0]=}"
        )

    sdm_station_attrs = ["stationId", "name"]
    station_df = exp_asdm_table_to_df(asdm, "Station", sdm_station_attrs)
    antenna_df = pd.merge(
        antenna_df, station_df, on="stationId", suffixes=("_antenna", "_station")
    )

    antenna_name = ("antenna_name", antenna_df["name_antenna"].values.astype("str"))
    cartesian_pos_label = ("cartesian_pos_label", ["x", "y", "z"])
    station_name = ("antenna_name", antenna_df["name_station"].values.astype("str"))
    mount = ("antenna_name", np.repeat(["ALT-AZ"], len(antenna_name[1])))
    telescope_name = get_telescope_name(asdm)
    telescope_name_by_antenna = [telescope_name] * len(antenna_name[1])
    xds = xds.assign_coords(
        {
            "antenna_name": antenna_name,
            "cartesian_pos_label": cartesian_pos_label,
            "station_name": station_name,
            "mount": mount,
            "telescope_name": (["antenna_name"], telescope_name_by_antenna),
            # Later, from Feed table/(polarizationTypes:
            # "receptor_label"
            # "polarization_type"
        }
    )

    diameter_attrs = {
        "type": "quantity",
        "units": "m",
    }
    xds["ANTENNA_DISH_DIAMETER"] = (
        "antenna_name",
        [val.get() for val in antenna_df["dishDiameter"].values],
        diameter_attrs,
    )
    # TODO: np.map or alike:
    position_values = [
        [pos[0].get(), pos[1].get(), pos[2].get()] for pos in antenna_df["position"]
    ]  # + antenna_df["offset"]
    position_attrs = {
        "type": "location",
        "units": "m",
        "frame": "ITRS",
        "coordinate_system": "geocentric",
        "origin_object_name": "earth",
    }
    # TODO:position attrs
    xds["ANTENNA_POSITION"] = (
        ["antenna_name", "cartesian_pos_label"],
        position_values,
        position_attrs,
    )

    xds.attrs.update({"overall_telescope_name": telescope_name})

    feed_xds = create_feed_xds(asdm, antenna_df, spectral_window_id, polarization)
    xds = xr.merge([xds, feed_xds])

    return xds


def create_feed_xds(
    asdm: pyasdm.ASDM,
    antenna_df: pd.DataFrame,
    spectral_window_id: int,
    polarization: xr.DataArray,
) -> xr.Dataset:
    """
    Create an xarray Dataset with feed data from an ASDM table.
    This function extracts feed-related information from an ASDM Feed table and creates
    an xarray Dataset containing polarization types, receptor angles and focus length
    for each antenna.
    Parameters
    ----------
    asdm : pyasdm.ASDM
        The ASDM object containing the feed table
    antenna_df : pd.DataFrame
        DataFrame containing antenna information
    spectral_window_id : int
        ID of the spectral window to filter feed data
    polarization : xr.DataArray
        DataArray containing polarization information (needed if
        that info is not present in the Feed table)

    Returns
    -------
    xr.Dataset
        Dataset with the following data variables:
        - ANTENNA_RECEPTOR_ANGLE (antenna_name, receptor_label) [rad]: Receptor angles
        - ANTENNA_FOCUS_LENGTH (antenna_name) [m]: Focus length
        And coordinates:
        - receptor_label: Labels for each receptor
        - polarization_type (antenna_name, receptor_label): Polarization types
    """

    sdm_feed_attrs = [
        "antennaId",
        "spectralWindowId",
        "timeInterval",
        "feedId",
        "focusReference",
        "polarizationTypes",
        "receptorAngle",
    ]
    feed_df = exp_asdm_table_to_df(asdm, "Feed", sdm_feed_attrs)
    feed_df = feed_df.loc[feed_df["spectralWindowId"] == spectral_window_id]

    feed_info_available = not feed_df.empty
    if not feed_info_available:
        # This happens typically for ALMA WVR SPWs - no feed info
        logger.warning(
            f"No feed info found for spectral window ID {spectral_window_id}"
        )
        # TODO: this should be shared with MSv2, same logic
        polarization_types = list(polarization.values[0])
        receptor_label = [f"pol_{idx}" for idx in np.arange(0, len(polarization_types))]
        pol_type_values = [polarization_types] * antenna_df.shape[0]
        polarization_type_df = pd.DataFrame(
            pol_type_values, columns=receptor_label, index=antenna_df["name_antenna"]
        )
    else:
        antenna_feed_df = pd.merge(
            antenna_df, feed_df, on="antennaId", suffixes=("_antenna", "_feed")
        )
        polarization_types_len = len(antenna_feed_df["polarizationTypes"][0])
        receptor_label = [f"pol_{idx}" for idx in np.arange(polarization_types_len)]
        polarization_type_df = pd.DataFrame(
            antenna_feed_df["polarizationTypes"].to_list(),
            columns=receptor_label,
            index=antenna_feed_df["name_antenna"],
        ).astype(str)

    feed_coords = {
        "receptor_label": (["receptor_label"], receptor_label),
        "polarization_type": (["antenna_name", "receptor_label"], polarization_type_df),
    }
    feed_xds = xr.Dataset(coords=feed_coords)
    feed_xds["polarization_type"] = feed_xds["polarization_type"].astype(str)

    if not feed_info_available:
        return feed_xds

    receptor_angle_values = [
        [angle[0].get(), angle[1].get()] for angle in antenna_feed_df["receptorAngle"]
    ]
    feed_xds["ANTENNA_RECEPTOR_ANGLE"] = (
        ["antenna_name", "receptor_label"],
        receptor_angle_values,
        make_quantity_attrs("rad"),
    )

    # first dim is receptor, which we don't have in MSv4
    focus_reference_values = [
        ref[0][2].get() for ref in antenna_feed_df["focusReference"]
    ]
    # TODO: double-check what is the "focus length" in ASDM/MSv2/MSv4
    feed_xds["ANTENNA_FOCUS_LENGTH"] = (
        ["antenna_name"],
        focus_reference_values,
        make_quantity_attrs("m"),
    )

    return feed_xds


def get_telescope_name(asdm: pyasdm.ASDM) -> str:
    """
    Get the telescope name from an ASDM dataset.
    This function extracts the telescope name from the ExecBlock table of an ASDM
    dataset and verifies that there is only one unique telescope name.

    Parameters
    ----------
    asdm : pyasdm.ASDM
        The ASDM dataset object containing the ExecBlock table.

    Returns
    -------
    str
        The unique telescope name found in the dataset.

    Raises
    ------
    RuntimeError
        If more than one unique telescope name is found in the ExecBlock table.
    Notes
    -----
    The function assumes that the ExecBlock table contains a 'telescopeName' column
    and that all entries in this column should refer to the same telescope.
    """

    sdm_execblock_attrs = ["execBlockId", "telescopeName"]
    execblock_df = exp_asdm_table_to_df(asdm, "ExecBlock", sdm_execblock_attrs)

    telescope_name = execblock_df["telescopeName"].unique()
    if len(telescope_name) != 1:
        raise RuntimeError(
            f"Issue with telescopeName. It should be one: {telescope_name}"
        )

    return telescope_name[0]
