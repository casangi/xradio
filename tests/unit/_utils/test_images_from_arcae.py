"""
The python-casacore images API of the arcae backend
(xradio._utils._casacore.images_from_arcae): coordinate systems, pixel and
mask access and the FITS export of images it creates.
"""

import numpy as np
import pytest

pytest.importorskip("arcae")

from astropy.io import fits  # noqa: E402

from xradio._utils._casacore import casacore_from_arcae as tables  # noqa: E402
from xradio._utils._casacore import images_from_arcae as images  # noqa: E402


def test_coordinates_of_a_new_image(tmp_path):
    """A new image has casacore's default coordinate system: direction,
    Stokes and spectral coordinates (in C order the other way round)."""
    im = images.image(str(tmp_path / "new.im"), shape=[3, 1, 4, 6])
    csys = im.coordinates()
    assert csys.get_names() == ["spectral", "stokes", "direction"]
    assert [type(coord) for coord in csys] == [
        images.spectralcoordinate,
        images.stokescoordinate,
        images.directioncoordinate,
    ]
    assert csys["stokes"].get_stokes() == ["I"]
    assert csys["stokes"].dict() == csys.dict()["stokes1"]
    assert csys["direction"].get_frame() == "J2000"
    assert csys["spectral"].get_frame() == "LSRK"
    assert csys["spectral"].get_restfrequency() == pytest.approx(1420405751.786)
    assert csys.get_observer() == "Karl Jansky"
    assert csys.get_telescope() == "ALMA"
    obsdate = csys.get_obsdate()
    assert obsdate["refer"] == "UTC"
    assert obsdate["m0"]["value"] == pytest.approx(51544.0, abs=1e-6)
    assert im.ndim() == 4
    assert im.miscinfo() == {}
    im.close()


def test_coordinate_system_without_coordinates():
    with pytest.raises(LookupError, match="valid coordinates"):
        images.coordinatesystem({"telescope": "ALMA", "observer": "me"})


def test_native_pole_of_a_projection_wcslib_does_not_know():
    coords = images._fill_direction_defaults(
        {
            "direction0": {
                "crval": np.array([0.1, 0.2]),
                "units": ["rad", "rad"],
                "projection": "XYZ",
            }
        }
    )
    assert coords["direction0"]["longpole"] == 180.0
    assert coords["direction0"]["latpole"] == pytest.approx(np.degrees(0.2))


def test_image_shapes_without_a_default_coordinate_system(tmp_path):
    """casacore gives the axes of other images linear coordinates, which
    the arcae backend does not create."""
    with pytest.raises(ValueError, match="without a coordinate system"):
        images.image(str(tmp_path / "linear.im"), shape=[5, 7, 6, 8])


def test_image_of_a_value_of_another_type(tmp_path):
    """An image filled with a value of a type without an image pixel type
    holds that value."""
    im = images.image(str(tmp_path / "int.im"), shape=[2, 3], values=np.int16(7))
    np.testing.assert_array_equal(im.getdata(), np.full((2, 3), 7))


def test_pixels_and_mask_of_an_image_without_a_mask(tmp_path):
    path = str(tmp_path / "nomask.im")
    im = images.image(path, shape=[4, 6])
    assert im.name() == path and im.name(strippath=True) == "nomask.im"
    data = np.arange(24, dtype=np.float32).reshape(4, 6)
    im.put(data)
    with pytest.raises(ValueError, match="value has 1 dimensions, image has 2"):
        im.putdata(np.zeros(6, dtype=np.float32))

    im.putmask(np.zeros((2, 3), dtype=bool))  # all good: no mask is made
    with tables.table(path) as tb:
        assert "Image_defaultmask" not in tb.keywordnames()
    masked = np.zeros((2, 3), dtype=bool)
    masked[1, 2] = True
    im.putmask(masked, blc=[1, 2])
    expected = np.zeros((4, 6), dtype=bool)
    expected[2, 4] = True  # the pixels not put stay good
    for image in (im, images.image(path)):
        values = image.get()
        np.testing.assert_array_equal(values.data, data, strict=True)
        np.testing.assert_array_equal(values.mask, expected, strict=True)
    with tables.table(path) as tb:
        assert tb.getkeyword("Image_defaultmask") == "mask0"


@pytest.mark.parametrize(
    "system, radesys, equinox",
    [("J2000", "FK5", 2000.0), ("B1950", "FK4", 1950.0), ("ICRS", "ICRS", None)],
)
def test_tofits_direction_frame_and_restoring_beam(tmp_path, system, radesys, equinox):
    path = str(tmp_path / "beam.im")
    images.image(path, shape=[1, 1, 4, 4]).close()
    with tables.table(path, readonly=False) as tb:
        coords = tb.getkeyword("coords")
        coords["direction0"]["system"] = system
        tb.putkeyword("coords", coords)
        beam = {
            "major": {"value": 2.0, "unit": "arcsec"},
            "minor": {"value": 1.0, "unit": "arcsec"},
            "positionangle": {"value": 30.0, "unit": "deg"},
        }
        tb.putkeyword("imageinfo", {"imagetype": "Intensity", "restoringbeam": beam})
    fits_path = str(tmp_path / "beam.fits")
    images.image(path).tofits(fits_path)

    header = fits.getheader(fits_path)
    assert header["RADESYS"] == radesys
    assert header.get("EQUINOX") == equinox
    assert header["BMAJ"] == pytest.approx(2.0 / 3600)
    assert header["BMIN"] == pytest.approx(1.0 / 3600)
    assert header["BPA"] == pytest.approx(30.0)
