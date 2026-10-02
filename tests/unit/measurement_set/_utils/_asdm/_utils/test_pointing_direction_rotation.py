import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import AltAz, SkyCoord

from xradio.measurement_set._utils._asdm._utils.pointing_direction_rotation import (
    altaz_local_basis,
    rotate_offset_to_target,
)


def skyoffset_reference(target: np.ndarray, offset: np.ndarray) -> np.ndarray:
    """Independent reference: astropy SkyOffsetFrame centered on the target,
    with (lon, lat) = (d_az, d_alt)."""
    out = []
    for (az, alt), (d_az, d_alt) in zip(
        target.reshape(-1, 2), offset.reshape(-1, 2), strict=True
    ):
        origin = SkyCoord(az=az * u.rad, alt=alt * u.rad, frame=AltAz())
        point = SkyCoord(
            lon=d_az * u.rad, lat=d_alt * u.rad, frame=origin.skyoffset_frame()
        ).transform_to(AltAz())
        out.append([point.az.rad, point.alt.rad])
    return np.array(out).reshape(target.shape)


def separation(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Great-circle separation (rad) between (az, alt) arrays."""
    return (
        SkyCoord(az=first[..., 0] * u.rad, alt=first[..., 1] * u.rad, frame=AltAz())
        .separation(
            SkyCoord(
                az=second[..., 0] * u.rad, alt=second[..., 1] * u.rad, frame=AltAz()
            )
        )
        .rad
    )


@pytest.mark.parametrize(
    "input_target, input_offset, expected_error",
    [
        (None, None, pytest.raises(TypeError, match="NoneType")),
        (np.zeros((1, 2)), None, pytest.raises(TypeError, match="NoneType")),
        (np.zeros((2, 2)), np.zeros((3, 2)), pytest.raises(ValueError, match="shape")),
        (np.zeros((2, 3)), np.zeros((2, 3)), pytest.raises(ValueError, match="shape")),
    ],
)
def test_rotate_offset_to_target_invalid_input(
    input_target, input_offset, expected_error
):
    with expected_error:
        rotate_offset_to_target(input_target, input_offset)


@pytest.mark.parametrize(
    "target",
    [
        (np.pi / 2, np.pi / 2),
        (0.90, 0.35),
        (-1.46, 1.14),
        (2.0, 0.8),
        (4.5, 1.2),
        (0.1, 0.05),
        (-4.0, 0.3),
    ],
)
def test_zero_offset_gives_the_target(target):
    target = np.array([[target]])
    result = rotate_offset_to_target(target, np.zeros_like(target))
    assert result.shape == target.shape
    np.testing.assert_allclose(result, target, rtol=0, atol=1e-12)


def test_offset_along_elevation():
    """A pure elevation offset only changes the elevation, by that amount."""
    result = rotate_offset_to_target(np.array([0.9, 0.35]), np.array([0.0, 1e-3]))
    np.testing.assert_allclose(result, [0.9, 0.351], rtol=0, atol=1e-12)
    result = rotate_offset_to_target(np.array([-2.5, 1.0]), np.array([0.0, -0.2]))
    np.testing.assert_allclose(result, [-2.5, 0.8], rtol=0, atol=1e-12)


@pytest.mark.parametrize("d_az", [1e-4, -3e-3, 0.1])
@pytest.mark.parametrize("target", [(0.9, 0.35), (-1.46, 1.14), (3.0, 0.05)])
def test_offset_along_azimuth(target, d_az):
    """A pure azimuth offset d moves along the great circle through the target
    perpendicular to its meridian: alt' = asin(cos(d) sin(alt)),
    az' = az + atan2(sin(d), cos(d) cos(alt)). For small d, az' ~ az + d / cos(alt)."""
    az, alt = target
    result = rotate_offset_to_target(np.array(target), np.array([d_az, 0.0]))
    expected_alt = np.arcsin(np.cos(d_az) * np.sin(alt))
    expected_az = az + np.arctan2(np.sin(d_az), np.cos(d_az) * np.cos(alt))
    np.testing.assert_allclose(result, [expected_az, expected_alt], rtol=0, atol=1e-12)
    if abs(d_az) < 1e-3:
        np.testing.assert_allclose(result[0], az + d_az / np.cos(alt), rtol=1e-5)


def test_matches_astropy_skyoffset_frame():
    rng = np.random.default_rng(1234)
    target = np.stack(
        [rng.uniform(-np.pi, np.pi, 60), rng.uniform(0.05, 1.5, 60)], axis=-1
    ).reshape(3, 20, 2)
    offset = rng.uniform(-0.05, 0.05, target.shape)
    result = rotate_offset_to_target(target, offset)
    assert result.shape == target.shape
    reference = skyoffset_reference(target, offset)
    np.testing.assert_allclose(separation(result, reference), 0.0, rtol=0, atol=1e-12)
    # the separation from the target is the angular length of the offset
    np.testing.assert_allclose(
        separation(result, target),
        np.arccos(np.cos(offset[..., 0]) * np.cos(offset[..., 1])),
        rtol=0,
        atol=1e-12,
    )


def test_azimuth_follows_the_target_convention():
    """The azimuth is not wrapped to [0, 2 pi): it stays within pi of the target
    azimuth, also when the offset crosses +-pi."""
    target = np.array([[-1.46, 1.14], [3.1, 0.2], [-3.1, 0.2], [4.0, 0.5]])
    offset = np.array([[0.01, 0.0], [0.1, 0.0], [-0.1, 0.0], [0.0, 0.0]])
    result = rotate_offset_to_target(target, offset)
    assert np.all(np.abs(result[:, 0] - target[:, 0]) < 0.2)
    assert result[1, 0] > np.pi
    assert result[2, 0] < -np.pi
    np.testing.assert_allclose(result[3], target[3], rtol=0, atol=1e-12)


def test_accepts_lists():
    result = rotate_offset_to_target([[0.5, 0.5]], [[0.0, 0.0]])
    np.testing.assert_allclose(result, [[0.5, 0.5]], rtol=0, atol=1e-12)


def test_altaz_local_basis_is_orthonormal_enu():
    target = SkyCoord(
        az=np.array([0.0, 0.9, -2.0]) * u.rad,
        alt=np.array([0.0, 0.35, 1.2]) * u.rad,
        frame="altaz",
    )
    east, north, up = altaz_local_basis(target)
    for vec in (east, north, up):
        np.testing.assert_allclose(np.linalg.norm(vec, axis=-1), 1.0, atol=1e-12)
    np.testing.assert_allclose(np.sum(east * north, axis=-1), 0.0, atol=1e-12)
    np.testing.assert_allclose(np.sum(east * up, axis=-1), 0.0, atol=1e-12)
    np.testing.assert_allclose(np.sum(north * up, axis=-1), 0.0, atol=1e-12)
    # at az=0, alt=0 (horizon, towards north): east = +y, north = up = +z
    np.testing.assert_allclose(up[0], [1.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(east[0], [0.0, 1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(north[0], [0.0, 0.0, 1.0], atol=1e-12)
