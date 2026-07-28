"""Utilities for rotating ASDM pointing-direction offsets into a target AltAz frame.

This module provides helpers for converting/rotating local offset directions expressed
in an AltAz-based East-North-Up (ENU) basis into the global coordinate frame of a
corresponding target direction. It is meant to be used when working with pointing
information from ASDM tables, where offsets are stored in the local frame of the
pointing target.
"""

import numpy as np

import astropy.units as u
from astropy.coordinates import (
    SkyCoord,
    CartesianRepresentation,
)


def rotate_offset_to_target(target: np.ndarray, offset: np.ndarray) -> np.ndarray:
    """
    Rotates alt-az offset directions ('offset' values from an ASDM pointing table) into the frame defined by the
    corresponding target alt-az offset directions.

    Parameters
    ----------
    target : np.ndarray
        Target AltAz coordinates with shape (..., 2). The trailing axis stores
        ``(az, alt)`` in radians. The array may have arbitrary leading shape,
        such as ``(n_antenna, n_samples, 2)``. Typically 3
             dimensions. The second dimension is the samples over time for a time interval and an antenna. The
             first dimension is for the groups of rows of the pointing table for every antenna.
    offset : np.ndarray
        Offset AltAz coordinates, same shape as ``target``, expressed as
        ``(az, alt)`` in radians.

    Returns
    -------
    np.ndarray
        Rotated offset directions in the same AltAz frame as ``target``, with
        shape ``target.shape``.
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
    Build the local East-North-Up basis for one or more AltAz directions.

    Parameters
    ----------
    target : SkyCoord
        AltAz coordinates of arbitrary shape ``(...)``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        ``(east, north, up)`` arrays, each with shape ``(..., 3)``. The vectors
        are expressed in the global Cartesian frame and form the local ENU basis
        attached to each target direction.
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
    Rotate offset vectors from the local ENU basis of each target into global AltAz.

    Parameters
    ----------
    target : SkyCoord
        AltAz coordinates defining the local reference directions. The input may
        have arbitrary shape ``(...)``.
    offset : SkyCoord
        Offset directions expressed in the local ENU frame attached to each
        corresponding target coordinate. Must have the same shape as ``target``.

    Returns
    -------
    SkyCoord
        A ``SkyCoord`` object in the same frame as ``target`` containing the
        rotated directions expressed in the global AltAz reference frame.
    """

    east, north, up = altaz_local_basis(target)

    # Offset vectors in Cartesian coordinates
    xyz = np.moveaxis(offset.cartesian.xyz.value, 0, -1)

    # Emulates the 'eulmat', 'matvec' calculations of the CASA sdm tool
    # But avoids explicitly building the full 3×3 rotation matrix (einsum):
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
