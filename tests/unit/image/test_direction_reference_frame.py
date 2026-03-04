import numpy as np
import pytest

from xradio.image import make_empty_sky_image
from xradio.image._util.common import _compute_world_sph_dims


def _base_args():
    """
    Provide shared constructor arguments for empty-sky-image tests.

    Parameters
    ----------
    None

    Returns
    -------
    dict
        Keyword arguments for ``make_empty_sky_image`` excluding frame options.
    """
    return dict(
        phase_center=[1.5, 0.2],
        image_size=[8, 8],
        cell_size=[np.radians(1 / 3600), np.radians(1 / 3600)],
        frequency_coords=np.array([1.4e9]),
        pol_coords=["I"],
        time_coords=np.array([59000.0]),
        projection="SIN",
        spectral_reference="lsrk",
    )


def test_make_empty_sky_image_galactic_uses_galactic_coords_and_reference_direction():
    """
    Verify Galactic frame requests emit Galactic coords and metadata labels.

    Parameters
    ----------
    None

    Returns
    -------
    None
        This test asserts expected coordinate and attribute behavior.
    """
    xds = make_empty_sky_image(
        **_base_args(),
        direction_reference="galactic",
        do_sky_coords=True,
    )

    assert "galactic_longitude" in xds.coords
    assert "galactic_latitude" in xds.coords
    assert "right_ascension" not in xds.coords
    assert "declination" not in xds.coords

    ref_dir = xds.attrs["coordinate_system_info"]["reference_direction"]
    assert ref_dir["attrs"]["frame"] == "galactic"
    assert ref_dir["coords"]["sky_dir_label"]["data"] == ["lon", "lat"]
    assert "equinox" not in ref_dir["attrs"]


def test_make_empty_sky_image_equatorial_keeps_radec_and_equinox():
    """
    Verify equatorial frame requests keep RA/Dec coords and equinox metadata.

    Parameters
    ----------
    None

    Returns
    -------
    None
        This test asserts expected coordinate and attribute behavior.
    """
    xds = make_empty_sky_image(
        **_base_args(),
        direction_reference="fk5",
        do_sky_coords=True,
    )

    assert "right_ascension" in xds.coords
    assert "declination" in xds.coords

    ref_dir = xds.attrs["coordinate_system_info"]["reference_direction"]
    assert ref_dir["attrs"]["frame"] == "fk5"
    assert ref_dir["coords"]["sky_dir_label"]["data"] == ["ra", "dec"]
    assert ref_dir["attrs"]["equinox"] == "j2000.0"


def test_compute_world_sph_dims_raises_for_unhandled_axis_name():
    """
    Verify unknown spherical axis names raise a clear runtime error.

    Parameters
    ----------
    None

    Returns
    -------
    None
        This test asserts error handling for unsupported axis-name inputs.
    """
    with pytest.raises(RuntimeError, match="Unhandled sky axis name"):
        _compute_world_sph_dims(
            projection="SIN",
            shape=[8, 8],
            ctype=["INVALID_AXIS", "DEC"],
            crpix=[4, 4],
            crval=[1.5, 0.2],
            cdelt=[-np.radians(1 / 3600), np.radians(1 / 3600)],
            cunit=["rad", "rad"],
        )
