"""Utilities for rotating ASDM pointing-direction offsets into a target AltAz frame.

This module provides helpers to apply the offsets of an ASDM Pointing table
(``offset`` column) to the corresponding target directions (``target`` column).
The offsets are small angles expressed in the local frame of each target: the
first component is along increasing azimuth and the second along increasing
elevation, and a zero offset designates the target itself. This is the
``eulmat(az, -el, 0) * rect(offset)`` construction used by the CASA ``sdm``
tool (importasdm) to compute the MSv2 POINTING DIRECTION, and it is equivalent
to astropy's ``SkyOffsetFrame`` centered on the target.
"""

import astropy.units as u
import numpy as np
from astropy.coordinates import (
    CartesianRepresentation,
    SkyCoord,
)


def rotate_offset_to_target(target: np.ndarray, offset: np.ndarray) -> np.ndarray:
    """
    Apply alt-az offsets ('offset' values from an ASDM Pointing table) to their
    target alt-az directions.

    The offset ``(d_az, d_alt)`` is interpreted as a spherical direction in the
    local frame of the target, whose origin ``(0, 0)`` is the target: a zero
    offset returns the target, and the angular separation between the result
    and the target is the angular length of the offset.

    Parameters
    ----------
    target : np.ndarray
        Target AltAz coordinates with shape (..., 2). The trailing axis stores
        ``(az, alt)`` in radians. The array may have arbitrary leading shape,
        such as ``(n_time, n_antenna, 2)``.
    offset : np.ndarray
        Offsets with the same shape as ``target``, ``(d_az, d_alt)`` in
        radians, in the local frame of the corresponding target.

    Returns
    -------
    np.ndarray
        Offset directions ``(az, alt)`` in radians in the same AltAz frame as
        ``target``, with shape ``target.shape``. The azimuth is returned within
        pi of the target azimuth, so that it follows the convention (range) of
        the input azimuths (for example [-pi, pi] or the extended azimuth range
        of the antenna mount) instead of being wrapped to [0, 2 pi).

    Raises
    ------
    TypeError
        If ``target`` or ``offset`` is None.
    ValueError
        If the shapes of ``target`` and ``offset`` differ or their trailing
        dimension is not 2.
    """
    if target is None or offset is None:
        raise TypeError(
            "target and offset must be arrays of (az, alt) values, got "
            f"{type(target).__name__} and {type(offset).__name__} (NoneType "
            "is not supported)"
        )
    target = np.asarray(target, dtype=np.float64)
    offset = np.asarray(offset, dtype=np.float64)
    if target.shape != offset.shape or target.ndim < 1 or target.shape[-1] != 2:
        raise ValueError(
            "target and offset must have the same shape (..., 2), got "
            f"{target.shape} and {offset.shape}"
        )

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

    rotated_az = rotated_offset_coords.az.rad
    # astropy wraps azimuths to [0, 2 pi). Bring them back within pi of the
    # target azimuth to keep the convention of the input (and of the encoder
    # and pointingDirection values the correction is combined with).
    target_az = target[..., 0]
    rotated_az = target_az + np.mod(rotated_az - target_az + np.pi, 2 * np.pi) - np.pi

    rotated_offset = np.stack(
        (
            rotated_az,
            rotated_offset_coords.alt.rad,
        ),
        axis=-1,
    )

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
        ``(east, north, up)`` arrays, each with shape ``(..., 3)``, expressed
        in the global (astropy AltAz) Cartesian frame: ``up`` is the unit
        vector of the target direction, ``east`` the unit vector along
        increasing azimuth and ``north`` the unit vector along increasing
        elevation at the target.
    """

    az = target.az.rad

    # Cartesian pointing vector, shape (..., 3)
    # Astropy stores Cartesian coordinates as (3, ...). Moving the Cartesian axis to the end gives (..., 3)
    # Up vector (pointing direction)
    up = np.moveaxis(target.cartesian.xyz.value, 0, -1)

    # East vector (direction of increasing azimuth)
    east = np.stack(
        (
            -np.sin(az),
            np.cos(az),
            np.zeros_like(az),
        ),
        axis=-1,
    )

    # North vector (direction of increasing elevation)
    north = np.cross(up, east)

    return east, north, up


def rotate_sky_coords_offset_to_target(target: SkyCoord, offset: SkyCoord) -> SkyCoord:
    """
    Rotate offsets from the local frame of each target into global AltAz.

    Parameters
    ----------
    target : SkyCoord
        AltAz coordinates defining the local reference directions. The input may
        have arbitrary shape ``(...)``.
    offset : SkyCoord
        Offset directions expressed in the local frame attached to each
        corresponding target coordinate: ``offset.az`` is the offset along
        increasing azimuth and ``offset.alt`` the offset along increasing
        elevation, so that ``(az, alt) = (0, 0)`` designates the target. Must
        have the same shape as ``target``.

    Returns
    -------
    SkyCoord
        A ``SkyCoord`` object in the same frame as ``target`` containing the
        rotated directions expressed in the global AltAz reference frame.
    """

    east, north, up = altaz_local_basis(target)

    # Offset directions as unit vectors in the local frame of the target, shape
    # (..., 3): x is the boresight (the target itself for a zero offset), y is
    # along increasing azimuth (east) and z along increasing elevation (north).
    xyz = np.moveaxis(offset.cartesian.xyz.value, 0, -1)

    # Emulates the 'eulmat', 'matvec' calculations of the CASA sdm tool, but
    # avoids explicitly building the full 3x3 rotation matrix: its columns are
    # the basis vectors (up, east, north), so the rotation is
    #   v_rot = v_x e_up + v_y e_east + v_z e_north
    xyz_rot = (
        xyz[..., 0, None] * up + xyz[..., 1, None] * east + xyz[..., 2, None] * north
    )

    cartesian_rot = CartesianRepresentation(
        x=xyz_rot[..., 0],
        y=xyz_rot[..., 1],
        z=xyz_rot[..., 2],
    )
    result_sky_direction = SkyCoord(cartesian_rot, frame=target.frame)

    return result_sky_direction
