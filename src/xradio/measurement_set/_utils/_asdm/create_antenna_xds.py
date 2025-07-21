import numpy as np
import pandas as pd
import xarray as xr


import pyasdm

from xradio.measurement_set._utils._asdm._utils.metadata_tables import (
    exp_asdm_table_to_df,
)


def create_antenna_xds(
    asdm: pyasdm.ASDM,
    num_antenna: int,
    partition_descr: dict,
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
    partition_descr : dict
        Dictionary containing partition description (not used in current implementation)
    polarization : xr.DataArray
        DataArray containing polarization information (not used in current implementation)

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
    station = ("antenna_name", antenna_df["name_station"].values)
    mount = ("antenna_name", np.repeat(["ALT-AZ"], len(station[1])))
    xds = xds.assign_coords(
        {
            "antenna_name": antenna_name,
            "cartesian_pos_src/xradio/measurement_set/_utils/_asdm/open_partition.pylabel": cartesian_pos_label,
            "station": station,
            "mount": mount,
            # TODO: Feed info
            # From Feed table, numReceptor, polarizationTypes, receptorAngle
            # "receptor_label"
            # "polarization_type"
        }
    )

    diameter_attrs = {
        "type": "quantity",
        "units": ["m"],
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
        "units": ["m"],
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

    # TODO: xds["ANTENNA_RECEPTOR_ANGLE"] - from feed_df
    # xds["ANTENNA_FOCUS_LENGTH"] - some calculation needed?

    telescope_name = get_telescope_name(asdm)
    xds.attrs.update({"overall_telescope_name": telescope_name})

    return xds


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
