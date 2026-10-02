import astropy.units as u
import numpy as np
import pandas as pd
import pyasdm
import xarray as xr
from astropy.coordinates import EarthLocation

from xradio._utils.dict_helpers import make_quantity_attrs
from xradio._utils.logging import xradio_logger
from xradio.measurement_set._utils._asdm._utils.metadata_tables import (
    exp_asdm_table_to_df,
)


def create_antenna_xds(
    asdm: pyasdm.ASDM,
    num_antenna: int,
    spectral_window_id: int,
    polarization: xr.DataArray,
    antenna_id: list[int] | np.ndarray | None = None,
) -> xr.Dataset:
    """
    Create an xarray Dataset with antenna metadata from ASDM.
    This function extracts antenna-related information from an ASDM (ALMA Science Data Model)
    and creates an xarray Dataset containing antenna metadata including positions, dish diameters,
    station information, mount types and feed (receptor) information.

    Parameters
    ----------
    asdm : pyasdm.ASDM
        The ASDM object containing the source data
    num_antenna : int
        Number of antennas in the array
    spectral_window_id : int
        ID of the spectral window, used to select the Feed table rows
    polarization : xr.DataArray
        Correlation products of the partition (polarization coordinate). Used to
        derive the receptor polarization types when the Feed table has no rows
        for the spectral window.
    antenna_id : list[int] | np.ndarray | None, optional
        Antenna ids (integer values of the Antenna table tags) of the antennas to
        include, in the order to use for the antenna_name dimension (for example
        the order of the antennas in ConfigDescription.antennaId). When None (the
        default), all the antennas of the Antenna table are included, in table
        order.

    Returns
    -------
    xr.Dataset
        Dataset containing antenna metadata with the following variables and coordinates:
        - ANTENNA_DISH_DIAMETER: dish diameter in meters for each antenna
        - ANTENNA_POSITION: cartesian position coordinates (x,y,z) in ITRS frame
        - ANTENNA_RECEPTOR_ANGLE: (when the Feed table has rows for the spectral
          window) receptor angles
        Coordinates:
        - antenna_name: names of the antennas
        - cartesian_pos_label: position coordinate labels (x,y,z)
        - station_name: station names for each antenna
        - mount: mount type for each antenna
        - telescope_name: telescope name for each antenna
        - receptor_label, polarization_type: receptors and their polarization types

    Raises
    ------
    RuntimeError
        If the number of antennas does not match num_antenna, an antenna id or
        station is not found, or the feed information is not supported.

    Notes
    -----
    The function currently assumes relocatable antennas and ALT-AZ mount types.

    ANTENNA_POSITION is the Station position (ITRF, geocentric) plus the
    Antenna position and offset, which are taken as vectors in the local
    topocentric (East, North, Up) frame of the station and rotated into ITRF.
    ANTENNA_FOCUS_LENGTH is not produced (see :func:`create_feed_xds`).
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
    antenna_df = _select_antennas(antenna_df, num_antenna, antenna_id)

    sdm_station_attrs = ["stationId", "name", "position"]
    station_df = exp_asdm_table_to_df(asdm, "Station", sdm_station_attrs)
    antenna_df = _merge_antenna_station(antenna_df, station_df)

    antenna_name = ("antenna_name", antenna_df["name_antenna"].to_numpy(dtype="str"))
    cartesian_pos_label = ("cartesian_pos_label", ["x", "y", "z"])
    station_name = ("antenna_name", antenna_df["name_station"].to_numpy(dtype="str"))
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
            # Later/below, from Feed table/(polarizationTypes:
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
        np.array([val.get() for val in antenna_df["dishDiameter"].values], dtype=float),
        diameter_attrs,
    )

    position_attrs = {
        "type": "location",
        "units": "m",
        "frame": "ITRS",
        "coordinate_system": "geocentric",
        "origin_object_name": "earth",
    }
    xds["ANTENNA_POSITION"] = (
        ["antenna_name", "cartesian_pos_label"],
        _antenna_positions_itrf(antenna_df),
        position_attrs,
    )

    xds.attrs.update({"overall_telescope_name": telescope_name})

    feed_xds = create_feed_xds(asdm, antenna_df, spectral_window_id, polarization)
    xds = xr.merge([xds, feed_xds], join="exact", combine_attrs="override")

    return xds


def _select_antennas(
    antenna_df: pd.DataFrame,
    num_antenna: int,
    antenna_id: list[int] | np.ndarray | None,
) -> pd.DataFrame:
    """
    Select the Antenna table rows to include in the antenna_xds, in order.

    Parameters
    ----------
    antenna_df : pd.DataFrame
        Antenna table (with an "antennaId" column)
    num_antenna : int
        Expected number of antennas
    antenna_id : list[int] | np.ndarray | None
        Antenna ids to select, in order. None selects all the rows (table order).

    Returns
    -------
    pd.DataFrame
        The selected rows, in the requested order, with a fresh index.
    """
    if antenna_id is None:
        if num_antenna != antenna_df.shape[0]:
            raise RuntimeError(
                f"When creating antenna_xds, the expected {num_antenna=}, while "
                f"the antennas found in the Antenna table are {antenna_df.shape[0]=}"
            )
        return antenna_df.reset_index(drop=True)

    antenna_id = [int(ant_id) for ant_id in np.ravel(antenna_id)]
    if num_antenna != len(antenna_id):
        raise RuntimeError(
            f"When creating antenna_xds, the expected {num_antenna=}, while "
            f"{len(antenna_id)} antenna ids were given: {antenna_id}"
        )
    indexed_df = antenna_df.set_index("antennaId", drop=False)
    missing = [ant_id for ant_id in antenna_id if ant_id not in indexed_df.index]
    if missing:
        raise RuntimeError(
            f"When creating antenna_xds, antenna ids {missing} were not found in "
            f"the Antenna table (ids present: {indexed_df.index.to_list()})"
        )
    return indexed_df.loc[antenna_id].reset_index(drop=True)


def _merge_antenna_station(
    antenna_df: pd.DataFrame, station_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Join the Station rows to the Antenna rows (keeping the antenna order). Common
    columns get the suffixes "_antenna" and "_station".
    """
    missing = set(antenna_df["stationId"]) - set(station_df["stationId"])
    if missing:
        raise RuntimeError(
            f"When creating antenna_xds, station ids {sorted(missing)} used in the "
            "Antenna table were not found in the Station table"
        )
    return pd.merge(
        antenna_df,
        station_df,
        how="left",
        on="stationId",
        suffixes=("_antenna", "_station"),
        validate="many_to_one",
    )


def _enu_to_itrf_matrices(station_positions: np.ndarray) -> np.ndarray:
    """
    Rotation matrices from the local topocentric (East, North, Up) frame of each
    station to ITRF (geocentric x, y, z).

    Parameters
    ----------
    station_positions : np.ndarray
        Station positions (ITRF, geocentric), shape (n, 3), in meters.

    Returns
    -------
    np.ndarray
        Shape (n, 3, 3). The columns of each matrix are the East, North and Up
        unit vectors (geodetic, WGS84 ellipsoid) expressed in ITRF.
    """
    location = EarthLocation.from_geocentric(
        station_positions[:, 0],
        station_positions[:, 1],
        station_positions[:, 2],
        unit=u.m,
    )
    geodetic = location.to_geodetic("WGS84")
    lon = np.atleast_1d(geodetic.lon.to_value(u.rad))
    lat = np.atleast_1d(geodetic.lat.to_value(u.rad))
    sin_lon, cos_lon = np.sin(lon), np.cos(lon)
    sin_lat, cos_lat = np.sin(lat), np.cos(lat)
    zeros = np.zeros_like(lon)
    east = np.stack([-sin_lon, cos_lon, zeros], axis=-1)
    north = np.stack([-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat], axis=-1)
    up = np.stack([cos_lat * cos_lon, cos_lat * sin_lon, sin_lat], axis=-1)
    return np.stack([east, north, up], axis=-1)


def _antenna_positions_itrf(antenna_df: pd.DataFrame) -> np.ndarray:
    """
    Compute the ITRF (geocentric) positions of the antennas.

    ASDM Station.position is the ITRF position of the pad. Antenna.position (the
    antenna reference point relative to the pad, about (0, 0, 7.5) m for ALMA
    antennas, i.e. a height above the pad) and Antenna.offset are taken as
    vectors in the local topocentric (East, North, Up) frame of the station, so
    they are rotated into ITRF (geodetic East/North/Up of the station, WGS84)
    before being added to the station position.

    This convention is to be confirmed against CASA importasdm (and checked
    separately for VLA SDMs, whose Antenna.position may be ITRF-aligned). In
    particular, using the geocentric instead of the geodetic latitude of the
    station for the local frame moves ALMA antennas (7.5 m above the pad) by
    about 1.8 cm, but changes baselines by less than 0.1 mm. The treatment of
    Antenna.offset (zero in the ALMA data seen so far) as an additional
    pad-relative vector is also to be confirmed.

    Parameters
    ----------
    antenna_df : pd.DataFrame
        Antenna rows joined with their station, with the columns
        "position_station", "position_antenna" and "offset" (pyasdm Length
        triplets).

    Returns
    -------
    np.ndarray
        Antenna positions, shape (n, 3), in meters.
    """

    def lengths_to_array(column: str) -> np.ndarray:
        return np.array(
            [[length.get() for length in triplet] for triplet in antenna_df[column]],
            dtype=np.float64,
        ).reshape(-1, 3)

    station_positions = lengths_to_array("position_station")
    antenna_positions = lengths_to_array("position_antenna")
    antenna_offsets = lengths_to_array("offset")

    if np.any(antenna_offsets != 0.0):
        xradio_logger().warning(
            "Non-zero Antenna.offset values found. They are added to the antenna "
            "positions as local (East, North, Up) vectors relative to the station "
            "(convention to be confirmed)."
        )

    local_vectors = antenna_positions + antenna_offsets
    positions = station_positions.copy()
    needs_rotation = np.any(local_vectors != 0.0, axis=1)
    if np.any(needs_rotation):
        at_geocenter = needs_rotation & (np.linalg.norm(station_positions, axis=1) == 0)
        if np.any(at_geocenter):
            raise RuntimeError(
                "Cannot compute ANTENNA_POSITION: stations at the geocenter (0, 0, 0) "
                "have antennas with non-zero pad-relative positions, so no local frame "
                f"can be defined. Antennas: {antenna_df['name_antenna'][at_geocenter].to_list()}"
            )
        rotations = _enu_to_itrf_matrices(station_positions[needs_rotation])
        positions[needs_rotation] += np.einsum(
            "nij,nj->ni", rotations, local_vectors[needs_rotation]
        )
    return positions


def _receptor_types_from_correlations(polarization: xr.DataArray) -> list[str]:
    """
    Derive the receptor polarization types from the correlation products.

    For example XX,YY or XX,XY,YX,YY give [X, Y], RR,LL give [R, L], XX gives [X].

    Parameters
    ----------
    polarization : xr.DataArray
        Correlation products (polarization coordinate values, e.g. "XX", "YY")

    Returns
    -------
    list[str]
        Receptor polarization types, in order of first appearance.
    """
    if polarization is None:
        raise RuntimeError(
            "Cannot derive the receptor polarization types: there is no Feed "
            "information and no correlation products were given."
        )
    products = [
        str(prod) for prod in np.ravel(getattr(polarization, "values", polarization))
    ]
    receptors = []
    for product in products:
        if len(product) != 2:
            continue
        for receptor in product:
            if receptor not in receptors:
                receptors.append(receptor)
    if not receptors:
        xradio_logger().warning(
            f"Cannot derive receptor polarization types from the correlation products "
            f"{products}. Using the products as receptor types."
        )
        receptors = list(dict.fromkeys(products))
    return receptors


def _select_feed_rows(feed_df: pd.DataFrame, spectral_window_id: int) -> pd.DataFrame:
    """
    Select one Feed row per antenna: lowest feedId, then earliest timeInterval.
    Logs a warning when there is more than one candidate row for an antenna.
    """
    feed_df = feed_df.assign(
        time_start=[interval.getStart().get() for interval in feed_df["timeInterval"]]
    ).sort_values(["antennaId", "feedId", "time_start"], kind="stable")
    rows_per_antenna = feed_df.groupby("antennaId").size()
    multi = rows_per_antenna[rows_per_antenna > 1]
    if not multi.empty:
        xradio_logger().warning(
            f"The Feed table has several rows (feedId / timeInterval) for antenna "
            f"ids {multi.index.to_list()} and spectral window {spectral_window_id}. "
            "This is not currently supported. Only the row with the lowest feedId "
            "and the earliest time interval will be used."
        )
    return feed_df.drop_duplicates("antennaId", keep="first").set_index("antennaId")


def create_feed_xds(
    asdm: pyasdm.ASDM,
    antenna_df: pd.DataFrame,
    spectral_window_id: int,
    polarization: xr.DataArray,
) -> xr.Dataset:
    """
    Create an xarray Dataset with feed data from an ASDM table.
    This function extracts feed-related information from an ASDM Feed table and creates
    an xarray Dataset containing polarization types and receptor angles for each antenna.
    The Feed rows are matched to the antennas by antennaId.

    Parameters
    ----------
    asdm : pyasdm.ASDM
        The ASDM object containing the feed table
    antenna_df : pd.DataFrame
        DataFrame containing antenna information, with (at least) the columns
        "antennaId" and "name_antenna", one row per antenna in the order of the
        antenna_name dimension
    spectral_window_id : int
        ID of the spectral window to filter feed data
    polarization : xr.DataArray
        DataArray containing the correlation products (needed if
        that info is not present in the Feed table)

    Returns
    -------
    xr.Dataset
        Dataset with the following data variables:
        - ANTENNA_RECEPTOR_ANGLE (antenna_name, receptor_label) [rad]: Receptor angles
          (only when the Feed table has rows for the spectral window, NaN for
          antennas without Feed row)
        And coordinates:
        - antenna_name: antenna names (from antenna_df)
        - receptor_label: Labels for each receptor
        - polarization_type (antenna_name, receptor_label): Polarization types

    Raises
    ------
    RuntimeError
        If the Feed rows have different numbers of receptors, or the
        polarization types of the antennas cannot be determined.

    Notes
    -----
    One Feed row is used per antenna (the lowest feedId, then the earliest
    timeInterval, with a warning when there are several). When no Feed row exists
    for the spectral window (typically ALMA WVR spectral windows), the receptor
    types are derived from the correlation products. ANTENNA_FOCUS_LENGTH is not
    produced: Feed.focusReference is a focus reference position per receptor
    (often the -99999 placeholder in ALMA data), not the focal length along the
    optical axis that the MSv4 schema defines.
    """

    antenna_names = antenna_df["name_antenna"].to_numpy(dtype="str")
    antenna_ids = antenna_df["antennaId"].to_numpy(dtype=int)

    sdm_feed_attrs = [
        "antennaId",
        "spectralWindowId",
        "timeInterval",
        "feedId",
        "numReceptor",
        "polarizationTypes",
        "receptorAngle",
    ]
    feed_df = exp_asdm_table_to_df(asdm, "Feed", sdm_feed_attrs)
    feed_df = feed_df.loc[
        (feed_df["spectralWindowId"] == spectral_window_id)
        & feed_df["antennaId"].isin(antenna_ids)
    ]

    if feed_df.empty:
        # This happens typically for ALMA WVR SPWs - no feed info
        xradio_logger().warning(
            f"No feed info found for spectral window ID {spectral_window_id}"
        )
        # TODO: this should be shared with MSv2, same logic
        polarization_types = _receptor_types_from_correlations(polarization)
        receptor_label = [f"pol_{idx}" for idx in range(len(polarization_types))]
        return xr.Dataset(
            coords={
                "antenna_name": ("antenna_name", antenna_names),
                "receptor_label": ("receptor_label", receptor_label),
                "polarization_type": (
                    ("antenna_name", "receptor_label"),
                    np.array([polarization_types] * len(antenna_names), dtype=str),
                ),
            }
        )

    feed_rows = _select_feed_rows(feed_df, spectral_window_id)

    num_receptors = feed_rows["numReceptor"].unique()
    if len(num_receptors) != 1:
        raise RuntimeError(
            f"The Feed rows of spectral window {spectral_window_id} have different "
            f"numbers of receptors ({sorted(num_receptors)}), which is not supported."
        )
    num_receptor = int(num_receptors[0])
    receptor_label = [f"pol_{idx}" for idx in range(num_receptor)]

    pol_types_by_antenna = {}
    angles_by_antenna = {}
    for ant_id, row in feed_rows.iterrows():
        pol_types = [
            pol.getName() if hasattr(pol, "getName") else str(pol)
            for pol in row["polarizationTypes"]
        ]
        angles = [angle.get() for angle in row["receptorAngle"]]
        if len(pol_types) < num_receptor or len(angles) < num_receptor:
            raise RuntimeError(
                f"Feed row of antenna id {ant_id}, spectral window "
                f"{spectral_window_id} has numReceptor={num_receptor} but "
                f"polarizationTypes={pol_types} and {len(angles)} receptor angles"
            )
        pol_types_by_antenna[ant_id] = pol_types[:num_receptor]
        angles_by_antenna[ant_id] = angles[:num_receptor]

    missing_ids = [ant_id for ant_id in antenna_ids if ant_id not in feed_rows.index]
    if missing_ids:
        unique_pol_types = {tuple(types) for types in pol_types_by_antenna.values()}
        if len(unique_pol_types) != 1:
            raise RuntimeError(
                f"Antenna ids {missing_ids} have no Feed row for spectral window "
                f"{spectral_window_id}, and the polarization types of the other "
                f"antennas differ ({unique_pol_types}): cannot determine theirs."
            )
        missing_names = antenna_names[np.isin(antenna_ids, missing_ids)].tolist()
        xradio_logger().warning(
            f"Antennas {missing_names} have no Feed row for spectral window "
            f"{spectral_window_id}. Their receptor angles are set to NaN and their "
            "polarization types to those of the other antennas."
        )
        common_pol_types = list(unique_pol_types.pop())
        for ant_id in missing_ids:
            pol_types_by_antenna[ant_id] = common_pol_types
            angles_by_antenna[ant_id] = [np.nan] * num_receptor

    feed_xds = xr.Dataset(
        coords={
            "antenna_name": ("antenna_name", antenna_names),
            "receptor_label": ("receptor_label", receptor_label),
            "polarization_type": (
                ("antenna_name", "receptor_label"),
                np.array(
                    [pol_types_by_antenna[ant_id] for ant_id in antenna_ids], dtype=str
                ),
            ),
        }
    )
    feed_xds["ANTENNA_RECEPTOR_ANGLE"] = (
        ["antenna_name", "receptor_label"],
        np.array([angles_by_antenna[ant_id] for ant_id in antenna_ids], dtype=float),
        make_quantity_attrs("rad"),
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
            f"Issue with telescopeName. It should be one string from: {telescope_name}"
        )

    return telescope_name[0]
