import time

import numpy as np
import xarray as xr

import astropy.units as u
from astropy.coordinates import (
    SkyCoord,
    CartesianRepresentation,
)

import pyasdm

from xradio._utils.dict_helpers import make_time_measure_attrs
from xradio._utils.logging import xradio_logger
from xradio.measurement_set._utils._asdm._utils.metadata_tables import (
    exp_asdm_table_to_df,
)
from xradio.measurement_set._utils._asdm._utils.sky_coord_dict_helper import (
    make_sky_coord_measure_attrs,
)
from xradio.measurement_set._utils._asdm._utils.time import convert_time_asdm_to_unix


def create_pointing_xds(asdm: pyasdm.ASDM) -> xr.Dataset:
    """
    Create an xarray Dataset with pointing data from an ASDM.

    """

    time_start = time.time()
    xds = xr.Dataset(
        attrs={
            "type": "pointing",
        },
    )

    sdm_antenna_attrs = ["antennaId", "name"]
    antenna_df = exp_asdm_table_to_df(asdm, "Antenna", sdm_antenna_attrs)

    # overTheTop is optional, and not present in common ALMA ASDMs.
    sdm_pointing_attrs = [
        "antennaId",
        "timeInterval",
        "numTerm",
        "numSample",
        "target",
        "offset",
        "encoder",
        "pointingDirection",
    ]
    pointing_df = exp_asdm_table_to_df(asdm, "Pointing", sdm_pointing_attrs)

    # Definitions for coords/time_pointing,
    #  if no sampled timeInterval: getStart() + interval/2 + idx * (interval), (interval = getDuration() / numSamples)
    #  if sampled timeInterval: the per sample ASDM timeInterval getStart()+getDuration()/2
    num_samples = int(pointing_df["numSample"].values[0])
    # time_interval_all_rows = pointing_df["timeInterval"]
    antenna_groups = pointing_df.groupby("antennaId")
    antenna_names_coord = []
    time_interval_all_rows = {}
    for antenna_id, group in antenna_groups:
        antenna_name = antenna_df.loc[
            antenna_df["antennaId"] == antenna_id, "name"
        ].values[0]
        antenna_names_coord.append(antenna_name)
        time_interval_all_rows[antenna_name] = group["timeInterval"]

    # This makes the time coord
    rows_per_antenna = min(
        [
            len(time_interval_all_rows[antenna_name])
            for antenna_name in antenna_names_coord
        ]
    )
    all_time_centers = np.zeros((rows_per_antenna * num_samples), dtype="float64")
    all_row_idx = np.arange(0, rows_per_antenna)
    first_antenna_name = antenna_names_coord[0]
    for row_idx in all_row_idx:
        time_interval = time_interval_all_rows[first_antenna_name][row_idx]
        # For MSv4 only the centers are kept (no INTERVAL col as in MSv2)
        # ASDM always in ns
        sample_interval = (time_interval.getDuration() / num_samples).get()
        first_center = ((time_interval.getStart() + int(sample_interval)) / 2).get()
        last_center = first_center + (num_samples) * sample_interval
        time_centers_from_row = np.arange(first_center, last_center, sample_interval)
        index_time_centers = row_idx * num_samples
        all_time_centers[index_time_centers : index_time_centers + num_samples] = (
            time_centers_from_row
        )

    time_pointing_values = convert_time_asdm_to_unix(all_time_centers)
    time_attrs = make_time_measure_attrs("s", "tai", time_format="unix")
    time_pointing_coord = ("time_pointing", time_pointing_values, time_attrs)
    # Could take: antenna_df["name"].values.astype("str"))
    antenna_name_coord = ("antenna_name", antenna_names_coord)
    xds = xds.assign_coords(
        {
            "time_pointing": time_pointing_coord,
            "antenna_name": antenna_name_coord,
            "local_sky_dir_label": ["az", "alt"],
        }
    )

    # The direction arrays are broken into time/interval rows with a subset
    # of samples within that interval. Concatenate them:
    empty_direction = np.array([]).reshape(0, len(time_pointing_values), 2)
    direction_vars = {
        "target": empty_direction,
        "offset": empty_direction,
        "encoder": empty_direction,
        "pointingDirection": empty_direction,
    }
    for _antenna_id, group in antenna_groups:
        for dvar in direction_vars:
            # First concatenate all the rows for the antenna (along time_pointing coord)
            direction_var_antenna_values = np.concatenate(group[dvar].values)
            # Then concatenate this antenna to all the antennas (along antenna_name coord)
            direction_vars[dvar] = np.concatenate(
                (direction_vars[dvar], [direction_var_antenna_values])
            )

    # How the MSv4 data_vars are derived from the attributes of the ASDM pointing table:
    # MSv4/POINTING_BEAM = rotate(ASDM/target, ASDM/offset) + correction
    #  where correction = (ASDM/encoder - ASDM/pointingDirection) is applied when
    #
    # MSV4/(POINTING_DISH_MEASURED) = ASDM/encoder
    #
    # MSv4/(POINTING_OVER_THE_TOP) = ASDM/overTheTop (optional attribute apparently not present in usual ALMA ASDMs

    rotated_target = _rotate_offset_to_target(
        direction_vars["target"], direction_vars["offset"]
    )
    rotated_target = direction_vars["target"]
    corrected_rotated_target = (
        rotated_target + direction_vars["encoder"] - direction_vars["pointingDirection"]
    )

    # Using antenna/pointing order first, as it is closer to what we get from the ASDM/Pointing rows
    # time_antenna_dir_dims = ["time_pointing", "antenna_name", "local_sky_dir_label"]
    antenna_time_dir_dims = ["antenna_name", "time_pointing", "local_sky_dir_label"]
    direction_attrs = make_sky_coord_measure_attrs("rad", "altaz")
    data_vars = {
        "DIRECTION": (antenna_time_dir_dims, direction_vars["target"], direction_attrs),
        "POINTING_DISH_MEASURED": (
            antenna_time_dir_dims,
            direction_vars["encoder"],
            direction_attrs,
        ),
        # "POINTING_OVER_THE_TOP": (antenna_time_dir_dims, over_the_top),
    }
    xds = xds.assign(data_vars)
    xds = xds.transpose("time_pointing", "antenna_name", "local_sky_dir_label")

    xradio_logger().info(
        f"create_pointing_xds() took {time.time() - time_start:0.2f} s"
    )

    return xds


def _rotate_offset_to_target(target: np.ndarray, offset: np.ndarray) -> np.ndarray:
    """
    Rotates alt-az offset directions ('offset' values from an ASDM pointing table) into the frame defined by the
    corresponding target alt-az offset directions.

    Parameters
    ----------
    target : Array with target AltAz coordinates (in the last dimension), arbitrary shape (...). Typically 3
             dimensions. The second dimension is the samples over time for a time interval and an antenna. The
             first dimension is for the groups of rows of the pointing table for every antenna.
    offset : Array with offset AltAz coordinates, same shape as target.

    Returns
    -------
    np.ndarray
        Rotated directions in the same frame.
    """
    target_coord = SkyCoord(
        az=target[..., 0] * u.rad,
        alt=target[..., 1] * u.rad,
        frame="altaz",
    )

    offset_coord = SkyCoord(
        az=offset[..., 0] * u.rad,
        alt=offset[..., 1] * u.rad,
        frame="altaz",
    )

    rotated_offset_coords = rotate_sky_coords_offset_to_target(
        target_coord, offset_coord
    )

    rotated_offset = np.stack(
        (
            rotated_offset_coords.az.rad,
            rotated_offset_coords.alt.rad,
        ),
        axis=-1,
    )

    # If astropy's AltAz.az could return values in [0, 2pi), normalize => [-pi, pi]?
    # rotated_target[:, 0] = np.mod(rotated_target[:, 0] + np.pi, 2*np.pi) - np.pi
    return rotated_offset


def altaz_local_basis(target: SkyCoord):
    """
    Compute the local East-North-Up basis for each AltAz coordinate.

    Parameters
    ----------
    target : SkyCoord
        AltAz coordinates of arbitrary shape (...).

    Returns
    -------
    east, north, up : ndarray
        Arrays of shape (..., 3).
    """

    az = target.az.rad

    # Cartesian pointing vector, shape (..., N, 3)
    # Astropy stores Cartesian coordinates as (3, ..., N). Moving the Cartesian axis to the end gives (..., N, 3)
    # Up vector (pointing direction)
    up = np.moveaxis(target.cartesian.xyz.value, 0, -1)

    # East vector
    east = np.stack(
        (
            -np.sin(az),
            np.cos(az),
            np.zeros_like(az),
        ),
        axis=-1,
    )

    north = np.cross(up, east)

    return east, north, up


def rotate_sky_coords_offset_to_target(target: SkyCoord, offset: SkyCoord) -> SkyCoord:
    """
    Rotate offset directions from the local ENU frame defined by each
    target direction into the global AltAz frame.

    Parameters
    ----------
    target : SkyCoord
        AltAz coordinates, arbitrary shape (...).

    offset : SkyCoord
        AltAz coordinates, same shape as target.

    Returns
    -------
    SkyCoord
        Rotated directions in the same AltAz frame.
    """

    east, north, up = altaz_local_basis(target)

    # Offset vectors in Cartesian coordinates
    xyz = np.moveaxis(offset.cartesian.xyz.value, 0, -1)

    # Avoid explicitly building the full 3×3 rotation matrix (einsum):
    #
    # Since the matrix columns are just the basis vectors, you can apply the rotation directly:
    # This is mathematically identical to the matrix multiplication because
    # v_rot = v_x e_east + v_y e_north + v_z e_up
    # Expand local ENU coordinates into the global frame
    xyz_rot = (
        xyz[..., 0, None] * east + xyz[..., 1, None] * north + xyz[..., 2, None] * up
    )

    cartesian_rot = CartesianRepresentation(
        x=xyz_rot[..., 0],
        y=xyz_rot[..., 1],
        z=xyz_rot[..., 2],
    )
    result_sky_direction = SkyCoord(cartesian_rot, frame=target.frame)

    return result_sky_direction
