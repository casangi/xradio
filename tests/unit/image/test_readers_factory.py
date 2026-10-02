"""Tests for the CASA image reader, the image factory and their helpers.

* reference pixels of images whose reference direction lies outside the grid
* spectral frames, units and channel widths of CASA images, and images
  without a spectral axis
* time coordinates of CASA images and of the ``make_empty_*`` factories
* the factories' spectral references, tuple inputs and u/v attributes
* the image type and data group roles detected from store names
* sub types derived from the casacore image type
* opening images whose coordinates differ only by round-off
* the upgrade of image zarr stores written by xradio 1.2.3 and earlier

The CASA images are created in ``tmp_path`` (casacore table locking does not
work on synced folders).
"""

import datetime
import os
import shutil
import warnings
from typing import get_args

import dask
import numpy as np
import pytest
from astropy.time import Time

try:
    from casacore import tables
except ImportError:
    import xradio._utils._casacore.casacore_from_casatools as tables

from xradio._utils.dict_helpers import make_spectral_coord_reference_dict
from xradio._utils.schema import casacore_to_msv4_measure_type
from xradio.image import (
    load_image,
    make_empty_aperture_image,
    make_empty_lmuv_image,
    make_empty_sky_image,
    open_image,
    write_image,
)
from xradio.image._util import image_factory
from xradio.image._util._casacore.common import _create_new_image
from xradio.image._util.common import (
    _compute_sky_reference_pixel,
    _l_m_attr_notes,
    _linear_axis_reference_pixel,
)
from xradio.image._util.conventions import (
    CASACORE_SPECTRAL_FRAMES,
    spectral_frame_to_observer,
)
from xradio.image._util.image_factory import (
    create_image_xds_from_store,
    create_store_dict,
    detect_image_type,
)
from xradio.image._util.legacy import _LEGACY_L_M_NOTES, upgrade_legacy_image_attrs
from xradio.image.schema import AllowedSkyImageSubTypes, check_image

pytestmark = pytest.mark.usefixtures("dask_client_module")

_CELL = np.pi / 180 / 60  # 1 arcmin
_SHAPE = (3, 2, 6, 8)  # frequency, polarization, m (dec), l (ra)


def _make_casa_image(path, shape=_SHAPE, edit_coords=None, imageinfo=None):
    """Create a CASA image with casacore's default coordinate system for
    ``shape`` (in numpy order), then edit its coordinate system record and
    image info."""
    data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    with _create_new_image(str(path), shape=list(shape)) as im:
        # a masked array, as the casatools backend requires
        im.put(np.ma.masked_array(data, np.zeros(shape, dtype=bool)))
    if edit_coords is not None or imageinfo is not None:
        with tables.table(str(path), readonly=False, ack=False) as tb:
            if edit_coords is not None:
                coords = tb.getkeyword("coords")
                edit_coords(coords)
                tb.putkeyword("coords", coords)
            if imageinfo is not None:
                tb.putkeyword("imageinfo", imageinfo)
    return str(path), data


def _spectral_key(coords):
    return next(k for k in coords if k.startswith("spectral"))


def _to_aperture_coords(coords):
    """Replace the direction coordinate by a linear UU/VV coordinate."""
    del coords["direction0"]
    coords["linear0"] = {
        "axes": ["UU", "VV"],
        "cdelt": np.array([-10.0, 10.0]),
        "crpix": np.array([4.0, 3.0]),
        "crval": np.array([0.0, 0.0]),
        "pc": np.array([[1.0, 0.0], [0.0, 1.0]]),
        "units": ["lambda", "lambda"],
    }


# --------------------------------------------------------------------------- #
# Reference pixels                                                             #
# --------------------------------------------------------------------------- #


class TestReferencePixel:
    """The reference pixel is exact on the grid and extrapolated outside it."""

    @pytest.mark.parametrize("sign", [-1.0, 1.0])
    def test_reference_pixel_on_the_grid(self, sign):
        values = (np.arange(10) - 4) * sign * 1e-5
        assert _linear_axis_reference_pixel(values) == 4.0

    @pytest.mark.parametrize(
        "crpix,n",
        [(15.0, 10), (-2.0, 8), (-4.0, 5), (7.25, 5), (-0.5, 3)],
    )
    @pytest.mark.parametrize("sign", [-1.0, 1.0])
    def test_reference_pixel_outside_the_grid(self, crpix, n, sign):
        cdelt = sign * 3.7e-6
        values = (np.arange(n) - crpix) * cdelt
        assert _linear_axis_reference_pixel(values) == pytest.approx(crpix, abs=1e-12)

    def test_extrapolated_integer_pixel_is_exact(self):
        values = (np.arange(10) + 12 - 10) * 2.9088820866572157e-4
        assert _linear_axis_reference_pixel(values) == -2.0

    def test_reference_value_other_than_zero(self):
        values = 100.0 + (np.arange(4) - 6) * 2.0
        assert _linear_axis_reference_pixel(values, 100.0) == 6.0

    def test_single_pixel_axis(self):
        assert _linear_axis_reference_pixel([0.0]) == 0.0
        assert _linear_axis_reference_pixel([3e-5], increment=1e-5) == -3.0
        with pytest.raises(ValueError, match="single pixel"):
            _linear_axis_reference_pixel([3e-5])

    def test_degenerate_axes_raise(self):
        with pytest.raises(ValueError, match="empty"):
            _linear_axis_reference_pixel([])
        with pytest.raises(ValueError, match="equal"):
            _linear_axis_reference_pixel([1.0, 1.0, 1.0])

    def test_cutout_reference_pixel(self):
        xds = make_empty_sky_image(
            [0.2, -0.5], [30, 20], [_CELL, _CELL], [1.4e9], ["I"], [54000.0]
        )
        assert list(_compute_sky_reference_pixel(xds)) == [15.0, 10.0]
        cutout = xds.isel(l=slice(0, 10), m=slice(12, 20))
        assert list(_compute_sky_reference_pixel(cutout)) == [15.0, -2.0]

    def test_single_pixel_cutout(self):
        xds = make_empty_sky_image(
            [0.2, -0.5], [30, 20], [_CELL, _CELL], [1.4e9], ["I"], [54000.0]
        )
        crpix = _compute_sky_reference_pixel(
            xds.isel(l=[3], m=[10]), cdelt=[-_CELL, _CELL]
        )
        np.testing.assert_allclose(crpix, [12.0, 0.0], atol=1e-12)
        with pytest.raises(ValueError, match="Reference pixel of the l axis"):
            _compute_sky_reference_pixel(xds.isel(l=[3]))

    @pytest.mark.parametrize("out_format", ["casa", "fits"])
    def test_cutout_round_trip_keeps_sky_positions(self, tmp_path, out_format):
        """A cutout that does not contain the reference pixel is written with
        the extrapolated reference pixel, so its sky positions survive."""
        path, _ = _make_casa_image(tmp_path / "field.im", shape=(2, 1, 20, 30))
        xds = open_image(path)
        cutout = xds.isel(l=slice(0, 10), m=slice(12, 20))
        out = str(tmp_path / ("cutout.fits" if out_format == "fits" else "cutout.im"))
        write_image(cutout, out, out_format=out_format)
        rt = open_image(out)
        for name in ("l", "m", "right_ascension", "declination"):
            np.testing.assert_allclose(
                rt[name].values, cutout[name].values, rtol=0, atol=1e-10
            )


# --------------------------------------------------------------------------- #
# CASA reader: spectral axis, time, sub type                                   #
# --------------------------------------------------------------------------- #


class TestCasaReaderSpectral:
    """Frequency coordinate of CASA images."""

    @pytest.mark.parametrize("frame", CASACORE_SPECTRAL_FRAMES)
    def test_spectral_frame(self, tmp_path, frame):
        def edit(coords):
            sd = coords[_spectral_key(coords)]
            sd["system"] = frame
            if "conversion" in sd:
                # no conversion layer (converting TOPO or GEO needs a position)
                sd["conversion"]["system"] = frame

        path, _ = _make_casa_image(tmp_path / "frame.im", edit_coords=edit)
        xds = open_image(path)
        attrs = xds.frequency.attrs
        assert attrs["frame"] == frame
        assert attrs["reference_frequency"]["attrs"]["observer"] == (
            spectral_frame_to_observer(frame)
        )
        assert not check_image(xds)

    def test_frequency_axis_in_ghz_is_converted_to_hz(self, tmp_path):
        """Values, reference, rest frequency and channel width come in Hz; the
        velocities follow the image's (optical) Doppler convention."""

        def edit(coords):
            sd = coords[_spectral_key(coords)]
            sd["unit"] = "GHz"
            sd["wcs"]["crval"] = 100.0
            sd["wcs"]["cdelt"] = 0.002
            sd["wcs"]["crpix"] = 1.0
            sd["restfreq"] = 100.1
            sd["restfreqs"] = np.array([100.1])
            sd["velType"] = 1  # optical

        path, _ = _make_casa_image(tmp_path / "ghz.im", edit_coords=edit)
        for xds in (open_image(path), load_image(path)):
            attrs = xds.frequency.attrs
            np.testing.assert_allclose(
                xds.frequency.values, [99.998e9, 100.0e9, 100.002e9], rtol=1e-12
            )
            assert attrs["units"] == "Hz"
            assert attrs["reference_frequency"]["attrs"]["units"] == "Hz"
            assert attrs["reference_frequency"]["data"] == pytest.approx(100e9)
            assert attrs["rest_frequency"]["attrs"]["units"] == "Hz"
            assert attrs["rest_frequency"]["data"] == pytest.approx(100.1e9)
            assert attrs["channel_width"]["data"] == pytest.approx(2e6)
            assert attrs["channel_width"]["attrs"]["units"] == "Hz"
            c = 299792458.0
            expected = (100.1e9 / xds.frequency.values - 1) * c
            assert xds.velocity.attrs["doppler_type"] == "z"
            np.testing.assert_allclose(xds.velocity.values, expected, rtol=1e-9)
            assert not check_image(xds)

    def test_channel_width_from_increment(self, tmp_path):
        def edit(coords):
            coords[_spectral_key(coords)]["wcs"]["cdelt"] = -250e3

        path, _ = _make_casa_image(
            tmp_path / "one_channel.im", shape=(1, 1, 6, 8), edit_coords=edit
        )
        xds = open_image(path)
        assert xds.sizes["frequency"] == 1
        assert xds.frequency.attrs["channel_width"]["data"] == 250e3

    @pytest.mark.parametrize("reader", [open_image, load_image])
    def test_image_without_spectral_and_stokes_axes(self, tmp_path, reader):
        path, data = _make_casa_image(tmp_path / "plane.im", shape=(6, 8))
        xds = reader(path)
        assert dict(xds.sizes)["frequency"] == 1
        assert xds.polarization.values.tolist() == ["I"]
        assert xds.frequency.values.tolist() == [1.415e9]
        attrs = xds.frequency.attrs
        assert attrs["frame"] == "LSRK"
        assert attrs["units"] == "Hz"
        assert attrs["reference_frequency"]["attrs"]["observer"] == "lsrk"
        assert "waveUnit" not in attrs
        np.testing.assert_array_equal(xds.SKY.values[0, 0, 0], data.T)
        assert not check_image(xds)


def _casacore_sky_positions(path):
    """Sky positions of every pixel from casacore, shape (l, m, 2) in
    radians (the direction axes are in radians)."""
    images = pytest.importorskip("casacore.images")
    image = images.image(path)
    n_m, n_l = image.shape()[-2:]
    world = np.array(
        [
            [
                image.toworld([0] * (image.ndim() - 2) + [y, x])[-2:][::-1]
                for y in range(n_m)
            ]
            for x in range(n_l)
        ]
    )
    del image
    return world


def _angle_difference(a, b):
    return (np.asarray(a) - np.asarray(b) + np.pi) % (2 * np.pi) - np.pi


class TestCasaReaderDirection:
    """Sky coordinates of CASA images: projection parameters, rotation and
    galactic axes."""

    @staticmethod
    def _direction(coords, crval, system="J2000", axes=None):
        direction = coords["direction0"]
        direction["system"] = system
        direction["conversionSystem"] = system
        if axes is not None:
            direction["axes"] = axes
        direction["units"] = ["rad", "rad"]
        direction["crval"] = np.radians(crval)
        direction["cdelt"] = np.array([-_CELL, _CELL])
        direction["crpix"] = np.array([3.0, 2.0])
        return direction

    @pytest.mark.parametrize("reader", [open_image, load_image])
    def test_projection_parameters_and_rotation(self, tmp_path, reader):
        """A slant orthographic (NCP-like) SIN image with a rotated pixel grid
        gets the sky positions casacore computes."""
        dec0 = np.radians(-40.0)
        rho = np.radians(20.0)

        def edit(coords):
            direction = self._direction(coords, [105.0, -40.0])
            direction["projection_parameters"] = np.array([0.0, 1 / np.tan(dec0)])
            direction["pc"] = np.array(
                [[np.cos(rho), -np.sin(rho)], [np.sin(rho), np.cos(rho)]]
            )

        path, _ = _make_casa_image(tmp_path / "slant.im", edit_coords=edit)
        world = _casacore_sky_positions(path)
        xds = reader(path)
        # 1e-9 arcsec; ignoring the projection parameters and the rotation
        # gives arcseconds to arcminutes
        tolerance = np.radians(1e-9 / 3600)
        assert np.abs(_angle_difference(xds.right_ascension, world[..., 0])).max() < (
            tolerance
        )
        assert np.abs(xds.declination.values - world[..., 1]).max() < tolerance

    @pytest.mark.parametrize("reader", [open_image, load_image])
    def test_galactic_image(self, tmp_path, reader):
        """casacore names the axes of galactic images Longitude and Latitude."""

        def edit(coords):
            self._direction(
                coords, [30.0, 2.0], system="GALACTIC", axes=["Longitude", "Latitude"]
            )

        path, data = _make_casa_image(tmp_path / "galactic.im", edit_coords=edit)
        world = _casacore_sky_positions(path)
        xds = reader(path)
        assert xds.SKY.dims == ("time", "frequency", "polarization", "l", "m")
        np.testing.assert_array_equal(
            xds.SKY.values[0], np.transpose(data, (0, 1, 3, 2))
        )
        reference = xds.attrs["coordinate_system_info"]["reference_direction"]
        assert reference["attrs"]["frame"] == "galactic"
        tolerance = np.radians(1e-9 / 3600)
        assert np.abs(
            _angle_difference(xds.galactic_longitude, world[..., 0])
        ).max() < (tolerance)
        assert np.abs(xds.galactic_latitude.values - world[..., 1]).max() < tolerance
        assert not check_image(xds)

    def test_galactic_image_round_trip(self, tmp_path):
        xds = make_empty_sky_image(
            phase_center=[0.5, 0.03],
            image_size=[8, 6],
            cell_size=[_CELL, _CELL],
            frequency_coords=[1.4e9],
            pol_coords=["I"],
            time_coords=[59000.5],
            direction_reference="galactic",
            do_sky_coords=True,
        )
        xds["SKY"] = (
            ("time", "frequency", "polarization", "l", "m"),
            np.arange(48, dtype=np.float32).reshape(1, 1, 1, 8, 6),
            {"type": "sky", "units": "Jy/beam"},
        )
        xds.attrs["data_groups"]["base"]["sky"] = "SKY"
        path = str(tmp_path / "galactic_rt.im")
        write_image(xds, path, out_format="casa")
        rt = open_image(path)
        np.testing.assert_array_equal(rt.SKY.values, xds.SKY.values)
        for name in ("galactic_longitude", "galactic_latitude"):
            np.testing.assert_allclose(
                rt[name].values, xds[name].values, rtol=0, atol=1e-12
            )

    def test_image_without_rest_frequency_has_no_velocity(self, tmp_path):
        """casacore records 0 for an unknown rest frequency: there are then no
        velocities (as in the FITS reader), instead of infinite ones."""

        def edit(coords):
            spectral = coords[_spectral_key(coords)]
            spectral["restfreq"] = 0.0
            spectral["restfreqs"] = np.array([0.0])

        path, _ = _make_casa_image(tmp_path / "no_rest.im", edit_coords=edit)
        for xds in (open_image(path), load_image(path)):
            assert "velocity" not in xds.coords
            assert xds.frequency.attrs["rest_frequency"]["data"] == 0.0
        assert not check_image(open_image(path))


class TestCasaReaderTime:
    """Time coordinate and observation date of CASA images."""

    @pytest.mark.parametrize(
        "refer,scale", [("UTC", "utc"), ("TT", "tt"), ("IAT", "tai"), ("TDB", "tdb")]
    )
    def test_epoch_reference_to_scale(self, tmp_path, refer, scale):
        def edit(coords):
            coords["obsdate"] = {
                "type": "epoch",
                "refer": refer,
                "m0": {"value": 59000.25, "unit": "d"},
            }

        path, _ = _make_casa_image(tmp_path / "epoch.im", edit_coords=edit)
        xds = open_image(path)
        assert xds.time.values.tolist() == [59000.25]
        assert xds.time.attrs == {
            "type": "time",
            "units": "d",
            "scale": scale,
            "format": "mjd",
        }
        obsdate = xds.SKY.attrs["obsdate"]
        assert obsdate["data"] == 59000.25
        assert obsdate["attrs"]["scale"] == scale
        assert obsdate["attrs"]["format"] == "mjd"
        assert not check_image(xds)

    @pytest.mark.parametrize("refer", ["LAST", "GMST1"])
    def test_sidereal_or_unset_epoch(self, tmp_path, refer):
        """An unset observation date (casacore's default, 0 d LAST) has no
        astropy scale: the time is labelled UTC and no obsdate is kept."""

        def edit(coords):
            coords["obsdate"] = {
                "type": "epoch",
                "refer": refer,
                "m0": {"value": 0.0, "unit": "d"},
            }

        path, _ = _make_casa_image(tmp_path / "unset.im", edit_coords=edit)
        for xds in (open_image(path), load_image(path)):
            assert xds.time.attrs["scale"] == "utc"
            assert xds.time.attrs["format"] == "mjd"
            assert "obsdate" not in xds.SKY.attrs
            assert not check_image(xds)

    def test_old_epoch_in_days_is_mjd(self, tmp_path):
        """Day valued epochs are MJD whatever their value."""

        def edit(coords):
            coords["obsdate"] = {
                "type": "epoch",
                "refer": "UTC",
                "m0": {"value": 30000.0, "unit": "d"},
            }

        path, _ = _make_casa_image(tmp_path / "old.im", edit_coords=edit)
        xds = open_image(path)
        assert xds.time.attrs["format"] == "mjd"
        assert xds.SKY.attrs["obsdate"]["attrs"]["format"] == "mjd"


class TestSubType:
    """The casacore image type becomes the sky image sub_type."""

    @pytest.mark.parametrize(
        "imagetype,sub_type",
        [
            ("Intensity", "Intensity"),
            ("Spectral Index", "SpectralIndex"),
            ("Column Density", "ColumnDensity"),
            ("Rotation Measure", "RotationMeasure"),
        ],
    )
    def test_casacore_image_type(self, tmp_path, imagetype, sub_type):
        path, _ = _make_casa_image(
            tmp_path / "typed.im",
            imageinfo={"imagetype": imagetype, "objectname": "src"},
        )
        xds = open_image(path)
        assert xds.SKY.attrs["sub_type"] == sub_type
        assert xds.SKY.attrs["type"] == "sky"
        assert not check_image(xds)

    @pytest.mark.parametrize(
        "native,sub_type",
        [
            ("intensity", "Intensity"),
            ("SPECTRAL_INDEX", "SpectralIndex"),
            ("Jy/beam", None),
            ("Undefined", None),
            ("", None),
            # kept only if the schema allows it for sky images
            ("Beam", "Beam" if "Beam" in get_args(AllowedSkyImageSubTypes) else None),
        ],
    )
    def test_native_type_spellings(self, tmp_path, native, sub_type):
        stores = {"sky": _fake_store(tmp_path, "a.fits")}
        xds = _open_fake(stores, {stores["sky"]: _fake_image(native_type=native)})
        assert xds.SKY.attrs.get("sub_type") == sub_type
        assert xds.SKY.attrs["type"] == "sky"
        assert not check_image(xds)

    def test_sub_type_of_other_images(self, tmp_path):
        """Images other than sky images keep any casacore image type."""
        stores = {"point_spread_function": _fake_store(tmp_path, "a.fits")}
        xds = _open_fake(
            stores, {stores["point_spread_function"]: _fake_image(native_type="Beam")}
        )
        assert xds.POINT_SPREAD_FUNCTION.attrs["sub_type"] == "Beam"
        assert xds.POINT_SPREAD_FUNCTION.attrs["type"] == "point_spread_function"


# --------------------------------------------------------------------------- #
# Image types and data groups detected from store names                        #
# --------------------------------------------------------------------------- #


class TestDetectImageType:
    """The image type is taken from the last token of the store name when it
    names a role, else from the role names the name contains (as in xradio
    1.2.4 and earlier)."""

    @pytest.mark.parametrize(
        "name,image_type",
        [
            ("target.image", "SKY"),
            ("target.im", "SKY"),
            ("target.psf", "POINT_SPREAD_FUNCTION"),
            ("target.pb", "PRIMARY_BEAM"),
            ("target.residual", "SKY_RESIDUAL"),
            ("target.model", "SKY_MODEL"),
            ("target.dirty", "SKY_DIRTY"),
            ("target.mask", "MASK"),
            ("target.sumwt", "VISIBILITY_NORMALIZATION"),
            ("target.image.pbcor", "SKY"),
            ("target.residual.tt0", "SKY_RESIDUAL"),
            ("target.image.fits", "SKY"),
            ("target.PSF.FITS", "POINT_SPREAD_FUNCTION"),
            ("out.base.sky.fits", "SKY"),
            ("out.base.point_spread_function.fits", "POINT_SPREAD_FUNCTION"),
            ("out.base.deconvolved.primary_beam", "PRIMARY_BEAM"),
            ("out.base.mask.fits", "MASK"),
            ("out.base.visibility_normalization", "VISIBILITY_NORMALIZATION"),
            ("out.base.uv_sampling", "UV_SAMPLING"),
            ("ngc_uvcontsub.cube.mask", "MASK"),
            ("my_uvsub.im", "SKY"),
            ("pbcor_test.fits", "SKY"),
            ("ngc1234_cont", "SKY"),
            ("ngc1234.integrated", "SKY"),
            ("mosaic_image_psf_test", "SKY"),
            ("cube.zarr", "ALL"),
            # role names inside words or before a generic extension, as
            # xradio 1.2.4 classified them
            ("psf.im", "POINT_SPREAD_FUNCTION"),
            ("my.psf.im", "POINT_SPREAD_FUNCTION"),
            ("target_psf.im", "POINT_SPREAD_FUNCTION"),
            ("ngc1234_psf", "POINT_SPREAD_FUNCTION"),
            ("mypsf", "POINT_SPREAD_FUNCTION"),
            ("target.pb.im", "PRIMARY_BEAM"),
            ("target_pb", "PRIMARY_BEAM"),
            ("target_primary_beam.im", "PRIMARY_BEAM"),
            ("model.im", "SKY_MODEL"),
            ("field_model", "SKY_MODEL"),
            ("dirty.im", "SKY_DIRTY"),
            ("cube_residual", "SKY_RESIDUAL"),
            ("sumwt.im", "VISIBILITY_NORMALIZATION"),
            ("t_mask", "MASK"),
            ("simulation_mask", "MASK"),
            # a name containing "image", "sky" or "fits" is a sky image unless
            # its last token names another role
            ("my_psf.image", "SKY"),
            ("x.psf.image", "SKY"),
            ("my.image.psf", "POINT_SPREAD_FUNCTION"),
            ("target_psf.fits", "SKY"),
            ("out.model.sky", "SKY"),
            ("my_mask.im", "SKY"),
        ],
    )
    def test_names(self, name, image_type):
        # the directories of the path take no part in the classification
        assert detect_image_type(os.path.join("/data/pb/model", name)) == image_type

    def test_non_string_store(self):
        assert detect_image_type({"a": 1}) == "ALL"

    def test_name_without_a_role_is_a_sky_image_with_a_warning(self, monkeypatch):
        messages = []

        class _Logger:
            def warning(self, message, *args, **kwargs):
                messages.append(str(message))

            def __getattr__(self, name):
                return lambda *args, **kwargs: None

        monkeypatch.setattr(image_factory, "xradio_logger", _Logger)
        assert detect_image_type("/data/ngc1234_cont") == "SKY"
        assert len(messages) == 1 and "names no image role" in messages[0]
        messages.clear()
        assert detect_image_type("/data/ngc1234_psf") == "POINT_SPREAD_FUNCTION"
        assert detect_image_type("/data/ngc1234.image") == "SKY"
        assert not messages

    @pytest.mark.parametrize("name", ["target.image", "ngc1234_cont", "plane.sky"])
    def test_aperture_by_coordinates(self, tmp_path, name):
        path, _ = _make_casa_image(tmp_path / name, edit_coords=_to_aperture_coords)
        assert detect_image_type(path) == "APERTURE"
        xds = open_image(path)
        assert "APERTURE" in xds.data_vars
        assert xds.attrs["data_groups"] == {"base": {"aperture": "APERTURE"}}

    def test_sky_by_coordinates(self, tmp_path):
        path, _ = _make_casa_image(tmp_path / "my_uvsub.aperture")
        assert detect_image_type(path) == "SKY"

    def test_unrecognized_name_opens_as_sky(self, tmp_path):
        """Names without a role token no longer need a store dict (as the
        xarray backends cannot pass one)."""
        path, _ = _make_casa_image(tmp_path / "ngc1234_cont")
        xds = open_image(path)
        assert "SKY" in xds.data_vars
        assert not check_image(xds)

    def test_tclean_products_get_schema_roles(self, tmp_path):
        stores = {}
        for product in ("image", "psf", "pb", "model", "residual", "mask"):
            stores[product], _ = _make_casa_image(tmp_path / f"target.{product}")
        xds = open_image(list(stores.values()))
        assert set(xds.data_vars) >= {
            "SKY",
            "POINT_SPREAD_FUNCTION",
            "PRIMARY_BEAM",
            "SKY_MODEL",
            "SKY_RESIDUAL",
            "MASK",
        }
        groups = xds.attrs["data_groups"]
        assert set(groups) == {"base", "model", "residual"}
        assert groups["base"]["sky"] == "SKY"
        assert groups["model"]["sky"] == "SKY_MODEL"
        assert groups["residual"]["sky"] == "SKY_RESIDUAL"
        for group in groups.values():
            assert group["point_spread_function"] == "POINT_SPREAD_FUNCTION"
            assert group["primary_beam"] == "PRIMARY_BEAM"
            assert group["mask"] == "MASK"
        assert xds.SKY_MODEL.attrs["type"] == "sky"
        assert xds.MASK.attrs["type"] == "mask"
        assert not check_image(xds)

    def test_store_dict_aliases(self, tmp_path):
        stores = {}
        for product in ("image", "psf", "residual"):
            stores[product], _ = _make_casa_image(tmp_path / f"t.{product}")
        xds = open_image(stores)
        assert {"SKY", "POINT_SPREAD_FUNCTION", "SKY_RESIDUAL"} <= set(xds.data_vars)
        assert set(xds.attrs["data_groups"]) == {"base", "residual"}
        assert not check_image(xds)

    def test_image_without_sky_has_a_data_group(self, tmp_path):
        path, _ = _make_casa_image(tmp_path / "target.psf")
        xds = open_image(path)
        assert xds.attrs["data_groups"] == {
            "base": {"point_spread_function": "POINT_SPREAD_FUNCTION"}
        }
        assert not check_image(xds)

    def test_duplicate_types_raise(self, tmp_path):
        a, _ = _make_casa_image(tmp_path / "a.image")
        b, _ = _make_casa_image(tmp_path / "b.image")
        with pytest.raises(ValueError, match="Duplicate image type SKY"):
            create_store_dict([a, b])


# --------------------------------------------------------------------------- #
# Combining images whose coordinates differ by round-off                       #
# --------------------------------------------------------------------------- #


def _fake_store(tmp_path, name):
    """An empty file that the store type detection takes for a FITS image."""
    path = tmp_path / name
    path.touch()
    return str(path)


def _fake_image(
    l_shift=0.0,
    freq_shift=0.0,
    time_shift=0.0,
    pols=("I", "Q"),
    native_type="Intensity",
):
    """Coordinates of a small sky image (the variable is added by the fake
    reader), with shifted coordinates and the polarizations in the order
    given (the factory takes them in canonical order only)."""
    canonical = sorted(pols, key=" IQUV".index)
    xds = make_empty_sky_image(
        [0.2, -0.5],
        [8, 6],
        [_CELL, _CELL],
        [1.4e9, 1.401e9, 1.402e9],
        canonical,
        [59000.5],
    )
    xds = xds.isel(polarization=[canonical.index(p) for p in pols])
    xds = xds.assign_coords(
        l=("l", xds.l.values + l_shift * _CELL, xds.l.attrs),
        frequency=("frequency", xds.frequency.values + freq_shift, xds.frequency.attrs),
        time=("time", xds.time.values + time_shift, xds.time.attrs),
    )
    xds.attrs["native_type"] = native_type
    return xds


def _open_fake(stores, coords_by_store):
    """Open ``stores`` with a fake reader that returns, for each store, an
    image on the coordinates of ``coords_by_store[store]`` whose pixel values
    encode the polarization (1 for I, 2 for Q, ...)."""

    def read(store, image_type, **kwargs):
        coords = coords_by_store[store]
        shape = tuple(coords.sizes[d] for d in ("time", "frequency", "polarization"))
        codes = np.array(
            [" IQUV".index(p) for p in coords.polarization.values], dtype=np.float32
        )
        data = np.broadcast_to(
            codes[None, None, :, None, None],
            shape + (coords.sizes["l"], coords.sizes["m"]),
        ).copy()
        xds = coords.copy()
        native = xds.attrs.pop("native_type")
        xds[image_type] = (
            ("time", "frequency", "polarization", "l", "m"),
            data,
            {"type": native},
        )
        return xds

    return create_image_xds_from_store(stores, read, {}, read, {}, None, {})


class TestCombineImages:
    """Images opened together are aligned on shared coordinates without
    NaN filling; real coordinate differences raise."""

    def _stores(self, tmp_path):
        return {
            "sky": _fake_store(tmp_path, "a.fits"),
            "point_spread_function": _fake_store(tmp_path, "b.fits"),
        }

    def test_round_off_differences_are_snapped(self, tmp_path):
        stores = self._stores(tmp_path)
        first = _fake_image()
        second = _fake_image(l_shift=3e-9, freq_shift=1e-4, time_shift=1e-4 / 86400.0)
        xds = _open_fake(
            stores, {stores["sky"]: first, stores["point_spread_function"]: second}
        )
        psf = xds.POINT_SPREAD_FUNCTION.values
        assert not np.isnan(psf).any()
        np.testing.assert_array_equal(xds.l.values, first.l.values)
        np.testing.assert_array_equal(xds.frequency.values, first.frequency.values)
        np.testing.assert_array_equal(xds.time.values, first.time.values)
        np.testing.assert_array_equal(psf, xds.SKY.values)
        assert not check_image(xds)

    @pytest.mark.parametrize(
        "shift,dim",
        [
            ({"l_shift": 0.5}, "l"),
            ({"l_shift": 1e-5}, "l"),
            ({"freq_shift": 10.0}, "frequency"),
            ({"time_shift": 1.0 / 86400.0}, "time"),
            ({"pols": ("I", "V")}, "polarization"),
        ],
    )
    def test_real_differences_raise(self, tmp_path, shift, dim):
        stores = self._stores(tmp_path)
        coords = {
            stores["sky"]: _fake_image(),
            stores["point_spread_function"]: _fake_image(**shift),
        }
        with pytest.raises(ValueError, match=f"its {dim} coordinate"):
            _open_fake(stores, coords)

    def test_different_lengths_raise(self, tmp_path):
        stores = self._stores(tmp_path)
        coords = {
            stores["sky"]: _fake_image(),
            stores["point_spread_function"]: _fake_image(pols=("I",)),
        }
        with pytest.raises(ValueError, match="polarization coordinate has 1 values"):
            _open_fake(stores, coords)

    def test_polarizations_are_aligned_by_label(self, tmp_path):
        stores = self._stores(tmp_path)
        coords = {
            stores["sky"]: _fake_image(pols=("I", "Q")),
            stores["point_spread_function"]: _fake_image(pols=("Q", "I")),
        }
        xds = _open_fake(stores, coords)
        assert xds.polarization.values.tolist() == ["I", "Q"]
        np.testing.assert_array_equal(xds.POINT_SPREAD_FUNCTION.values, xds.SKY.values)

    def test_casa_and_fits_images_of_the_same_field(self, tmp_path):
        """A CASA image and its FITS export (frequency axis) open together:
        coordinates that differ by round-off do not NaN-fill the FITS image."""
        from astropy.io import fits

        path, data = _make_casa_image(tmp_path / "field.image")
        casa = open_image(path)
        # a FITS twin of the image, with a FREQ axis, written with astropy
        header = fits.Header()
        sky_values = casa.SKY.values[0]  # frequency, polarization, l, m
        header["CTYPE1"] = "RA---SIN"
        header["CTYPE2"] = "DEC--SIN"
        header["CTYPE3"] = "STOKES"
        header["CTYPE4"] = "FREQ"
        csys = casa.attrs["coordinate_system_info"]
        crpix = _compute_sky_reference_pixel(casa)
        header["CRVAL1"] = np.degrees(csys["reference_direction"]["data"][0])
        header["CRVAL2"] = np.degrees(csys["reference_direction"]["data"][1])
        header["CDELT1"] = np.degrees(casa.l.values[1] - casa.l.values[0])
        header["CDELT2"] = np.degrees(casa.m.values[1] - casa.m.values[0])
        header["CRPIX1"] = crpix[0] + 1
        header["CRPIX2"] = crpix[1] + 1
        header["CUNIT1"] = header["CUNIT2"] = "deg"
        header["CRVAL3"], header["CDELT3"], header["CRPIX3"] = 1.0, 1.0, 1.0
        freq = casa.frequency.values
        header["CRVAL4"], header["CDELT4"], header["CRPIX4"] = (
            freq[0],
            freq[1] - freq[0],
            1.0,
        )
        header["CUNIT4"] = "Hz"
        header["SPECSYS"] = "LSRK"
        header["RESTFRQ"] = casa.frequency.attrs["rest_frequency"]["data"]
        header["RADESYS"] = "FK5"
        header["EQUINOX"] = 2000.0
        header["BUNIT"] = "Jy/beam"
        header["BTYPE"] = "Intensity"
        header["TELESCOP"] = "ALMA"
        header["DATE-OBS"] = Time(casa.time.values[0], format="mjd", scale="utc").isot
        header["TIMESYS"] = "UTC"
        header["LONPOLE"] = 180.0
        header["LATPOLE"] = np.degrees(csys["reference_direction"]["data"][1])
        fits_path = str(tmp_path / "field.psf.fits")
        fits.PrimaryHDU(
            data=np.transpose(sky_values, (0, 1, 3, 2)), header=header
        ).writeto(fits_path)
        fits_alone = open_image(fits_path)
        assert "POINT_SPREAD_FUNCTION" in fits_alone.data_vars
        xds = open_image({"sky": path, "point_spread_function": fits_path})
        psf = xds.POINT_SPREAD_FUNCTION.values
        assert not np.isnan(psf).any()
        np.testing.assert_array_equal(psf, xds.SKY.values)


# --------------------------------------------------------------------------- #
# Image factories                                                              #
# --------------------------------------------------------------------------- #


_FACTORY_ARGS = (
    [0.2, -0.5],
    [10, 10],
    [_CELL, _CELL],
    [1.412e9, 1.413e9],
    ["I", "Q", "U"],
    [54000.1],
)


class TestFactorySpectralReference:
    """The factories accept casacore, FITS and observer frame names."""

    @pytest.mark.parametrize(
        "spectral_reference,frame,observer",
        [
            ("lsrk", "LSRK", "lsrk"),
            ("LSRK", "LSRK", "lsrk"),
            ("lsrd", "LSRD", "lsrd"),
            ("bary", "BARY", "BARY"),
            ("BARYCENT", "BARY", "BARY"),
            ("heliocen", "BARY", "BARY"),
            ("topo", "TOPO", "TOPO"),
            ("TOPOCENT", "TOPO", "TOPO"),
            ("rest", "REST", "REST"),
            ("SOURCE", "REST", "REST"),
            ("geo", "GEO", "gcrs"),
            ("gcrs", "GEO", "gcrs"),
            ("GEOCENTR", "GEO", "gcrs"),
            ("galacto", "GALACTO", "GALACTO"),
            ("GALACTOC", "GALACTO", "GALACTO"),
            ("LGROUP", "LGROUP", "LGROUP"),
            ("LOCALGRP", "LGROUP", "LGROUP"),
            ("cmb", "CMB", "CMB"),
            ("CMBDIPOL", "CMB", "CMB"),
        ],
    )
    @pytest.mark.parametrize(
        "factory",
        [make_empty_sky_image, make_empty_aperture_image, make_empty_lmuv_image],
    )
    def test_spectral_reference(self, factory, spectral_reference, frame, observer):
        xds = factory(*_FACTORY_ARGS, spectral_reference=spectral_reference)
        attrs = xds.frequency.attrs
        assert attrs["frame"] == frame
        assert attrs["observer"] == observer
        assert attrs["reference_frequency"]["attrs"]["observer"] == observer
        assert not check_image(xds)

    @pytest.mark.parametrize("spectral_reference", ["icrs", "HCRS", "lsr"])
    def test_frames_without_casacore_equivalent_are_kept(
        self, tmp_path, spectral_reference
    ):
        """The astropy frames without a casacore equivalent build an image
        dataset, as before; only the CASA and FITS writers refuse them."""
        xds = make_empty_sky_image(
            *_FACTORY_ARGS, spectral_reference=spectral_reference
        )
        name = spectral_reference.lower()
        attrs = xds.frequency.attrs
        assert attrs["frame"] == name
        assert attrs["observer"] == name
        assert attrs["reference_frequency"]["attrs"]["observer"] == name
        assert not check_image(xds)
        xds["SKY"] = (
            ("time", "frequency", "polarization", "l", "m"),
            np.zeros(
                tuple(
                    xds.sizes[d]
                    for d in ("time", "frequency", "polarization", "l", "m")
                ),
                np.float32,
            ),
        )
        xds.attrs["data_groups"]["base"]["sky"] = "SKY"
        for out_format in ("casa", "fits"):
            path = tmp_path / f"out.{out_format}"
            with pytest.raises(ValueError, match="no casacore equivalent"):
                write_image(xds, str(path), out_format=out_format)
            assert not path.exists()

    def test_unknown_frame_raises(self):
        with pytest.raises(ValueError, match="spectral_reference"):
            make_empty_sky_image(*_FACTORY_ARGS, spectral_reference="nonsense")


class TestFactoryInputs:
    """Time coordinates and tuple inputs of the factories."""

    @pytest.mark.parametrize(
        "time_coords",
        [
            [59000.5],
            59000.5,
            np.array([59000.5]),
            "59000.5",
            np.datetime64("2020-05-31T12:00:00"),
            np.array(["2020-05-31T12:00:00"], dtype="datetime64[ns]"),
            np.array(["2020-05-31T12:00:00"], dtype="datetime64[s]"),
            "2020-05-31T12:00:00",
            ["2020-05-31T12:00:00"],
            datetime.datetime(2020, 5, 31, 12, 0, 0),
            Time("2020-05-31T12:00:00", scale="utc"),
            Time(59000.5, format="mjd", scale="utc").tt,
            [Time(59000.5, format="mjd", scale="utc")],
        ],
    )
    def test_time_coords_are_mjd(self, time_coords):
        xds = make_empty_sky_image(*_FACTORY_ARGS[:5], time_coords)
        np.testing.assert_allclose(xds.time.values, [59000.5], rtol=0, atol=1e-9)
        assert xds.time.attrs["format"] == "mjd"
        assert xds.time.attrs["scale"] == "utc"

    @pytest.mark.parametrize(
        "time_coords",
        [
            [np.timedelta64(1, "D")],
            [True],
            [1 + 2j],
            [{"mjd": 59000.5}],
        ],
    )
    def test_non_time_values_raise(self, time_coords):
        with pytest.raises(TypeError, match="time_coords"):
            make_empty_sky_image(*_FACTORY_ARGS[:5], time_coords)

    def test_unparsable_time_string_raises(self):
        with pytest.raises(ValueError, match="cannot be parsed"):
            make_empty_sky_image(*_FACTORY_ARGS[:5], ["not a time"])

    @pytest.mark.parametrize(
        "factory",
        [make_empty_sky_image, make_empty_aperture_image, make_empty_lmuv_image],
    )
    def test_tuple_inputs(self, factory):
        xds = factory(
            (0.2, -0.5),
            (10, 10),
            (_CELL, _CELL),
            (1.412e9, 1.413e9),
            ("I", "Q"),
            (54000.1,),
        )
        reference = xds.attrs["coordinate_system_info"]["reference_direction"]
        assert reference["data"] == [0.2, -0.5]
        assert xds.sizes["frequency"] == 2
        assert xds.polarization.values.tolist() == ["I", "Q"]
        assert xds.time.values.tolist() == [54000.1]
        assert not check_image(xds)

    @pytest.mark.parametrize(
        "factory", [make_empty_aperture_image, make_empty_lmuv_image]
    )
    def test_uv_attrs(self, factory):
        xds = factory(*_FACTORY_ARGS)
        cdelt = 1 / (_CELL * 10)
        for name in ("u", "v"):
            assert xds[name].attrs == {
                "units": "lambda",
                "crval": 0.0,
                "cdelt": pytest.approx(cdelt),
                "type": "quantity",
            }
            assert xds[name].values[5] == 0.0
            np.testing.assert_allclose(np.diff(xds[name].values), cdelt)


# --------------------------------------------------------------------------- #
# Spectral frame vocabulary shared with the measurement set converter          #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("frame", ["GALACTO", "LGROUP", "CMB"])
def test_casacore_named_observers(frame):
    ref_map = casacore_to_msv4_measure_type["frequency"]["Ref_map"]
    assert ref_map[frame] == frame
    reference = make_spectral_coord_reference_dict(1e9, "Hz", frame)
    assert reference["attrs"]["observer"] == frame
    assert make_spectral_coord_reference_dict(1e9, "Hz", "LSRK")["attrs"][
        "observer"
    ] == ("lsrk")


# --------------------------------------------------------------------------- #
# Upgrade of image stores written by xradio <= 1.2.3                           #
# --------------------------------------------------------------------------- #


def _legacy_image_xds():
    """An lmuv image with a sky image, with the attributes xradio 1.2.3 wrote."""
    xds = make_empty_lmuv_image(*_FACTORY_ARGS, spectral_reference="bary")
    xds["SKY"] = (
        ("time", "frequency", "polarization", "l", "m"),
        np.ones((1, 2, 3, 10, 10), dtype=np.float32),
        {
            "type": "sky",
            "units": "Jy/beam",
            "obsdate": {
                "attrs": {"format": "", "scale": "last", "type": "time", "units": "d"},
                "data": 0.0,
                "dims": [],
            },
        },
    )
    xds.attrs["data_groups"] = {"base": {"sky": "SKY"}}
    xds.attrs["type"] = "image"
    frequency = xds.frequency.attrs
    del frequency["frame"]
    frequency["observer"] = "bary"
    frequency["reference_frequency"]["attrs"]["observer"] = "bary"
    del xds.time.attrs["type"]
    for name in ("u", "v"):
        xds[name].attrs = {
            "attrs": {"type": "quantity", "units": "lambda"},
            "data": 0.0,
            "dims": [],
        }
    for name, note in _LEGACY_L_M_NOTES.items():
        xds[name].attrs = {**xds[name].attrs, "note": note}
    return xds


def _write_legacy_store(tmp_path):
    """Write the legacy image dataset to a zarr store with xarray."""
    store = str(tmp_path / "legacy.zarr")
    with warnings.catch_warnings():
        # zarr warns about the fixed width string dtypes of the coordinates
        warnings.simplefilter("ignore")
        _legacy_image_xds().to_zarr(store)
    return store


class TestLegacyUpgrade:
    """Image datasets written by xradio <= 1.2.3 are upgraded on read."""

    def test_upgrade(self):
        legacy = _legacy_image_xds()
        assert check_image(legacy)
        xds = upgrade_legacy_image_attrs(legacy)
        assert xds.attrs["type"] == "image_dataset"
        assert xds.frequency.attrs["frame"] == "BARY"
        assert xds.frequency.attrs["observer"] == "BARY"
        assert xds.frequency.attrs["reference_frequency"]["attrs"]["observer"] == (
            "BARY"
        )
        assert xds.time.attrs["type"] == "time"
        assert "obsdate" not in xds.SKY.attrs
        cdelt = 1 / (_CELL * 10)
        assert xds.u.attrs == {
            "units": "lambda",
            "crval": 0.0,
            "cdelt": pytest.approx(cdelt),
            "type": "quantity",
        }
        # l and m are documented as projection plane coordinates, not angles
        notes = _l_m_attr_notes()
        assert xds.l.attrs["note"] == notes["l"]
        assert xds.m.attrs["note"] == notes["m"]
        assert not check_image(xds)
        # the input is not modified
        assert legacy.attrs["type"] == "image"
        assert "frame" not in legacy.frequency.attrs
        assert "obsdate" in legacy.SKY.attrs
        assert legacy.l.attrs["note"] == _LEGACY_L_M_NOTES["l"]

    def test_current_datasets_are_unchanged(self):
        xds = make_empty_lmuv_image(*_FACTORY_ARGS)
        assert upgrade_legacy_image_attrs(xds) is xds
        upgraded = upgrade_legacy_image_attrs(_legacy_image_xds())
        assert upgrade_legacy_image_attrs(upgraded) is upgraded

    def test_observers_and_frames_are_normalized(self):
        xds = make_empty_sky_image(*_FACTORY_ARGS)
        attrs = xds.frequency.attrs
        attrs["reference_frequency"]["attrs"]["observer"] = "geo"
        attrs["observer"] = "geo"
        attrs["frame"] = "GEOCENTR"
        xds = upgrade_legacy_image_attrs(xds)
        assert xds.frequency.attrs["frame"] == "GEO"
        assert xds.frequency.attrs["observer"] == "gcrs"
        assert xds.frequency.attrs["reference_frequency"]["attrs"]["observer"] == (
            "gcrs"
        )

    def test_frequencies_in_ghz(self):
        xds = make_empty_sky_image(*_FACTORY_ARGS)
        attrs = dict(xds.frequency.attrs)
        attrs["units"] = "GHz"
        attrs["reference_frequency"] = make_spectral_coord_reference_dict(
            1.413, "GHz", "lsrk"
        )
        xds = xds.assign_coords(
            frequency=("frequency", xds.frequency.values / 1e9, attrs)
        )
        xds = upgrade_legacy_image_attrs(xds)
        np.testing.assert_allclose(xds.frequency.values, [1.412e9, 1.413e9])
        assert xds.frequency.attrs["units"] == "Hz"
        reference = xds.frequency.attrs["reference_frequency"]
        assert reference["data"] == pytest.approx(1.413e9)
        assert reference["attrs"]["units"] == "Hz"

    def test_frame_without_casacore_equivalent_is_left_missing(self):
        xds = make_empty_sky_image(*_FACTORY_ARGS)
        attrs = xds.frequency.attrs
        del attrs["frame"]
        attrs["observer"] = "icrs"
        attrs["reference_frequency"]["attrs"]["observer"] = "icrs"
        xds = upgrade_legacy_image_attrs(xds)
        assert "frame" not in xds.frequency.attrs
        assert xds.frequency.attrs["observer"] == "icrs"

    @pytest.mark.parametrize("reader", [open_image, load_image])
    def test_zarr_stores_are_upgraded_on_read(self, tmp_path, reader):
        store = _write_legacy_store(tmp_path)
        xds = reader(store)
        assert xds.attrs["type"] == "image_dataset"
        assert xds.frequency.attrs["frame"] == "BARY"
        assert "obsdate" not in xds.SKY.attrs
        assert not check_image(xds)

    def test_zarr_store_in_a_list(self, tmp_path):
        store = _write_legacy_store(tmp_path)
        xds = open_image([store])
        assert xds.attrs["type"] == "image_dataset"


def test_copied_casa_image_keeps_its_role(tmp_path):
    """The role token of a copy decides its type, not that of the original."""
    path, _ = _make_casa_image(tmp_path / "target.image")
    copy_path = str(tmp_path / "copy.residual")
    shutil.copytree(path, copy_path)
    xds = open_image(copy_path)
    assert "SKY_RESIDUAL" in xds.data_vars
    assert xds.attrs["data_groups"]["residual"]["sky"] == "SKY_RESIDUAL"
    assert not check_image(xds)


def test_combined_images_stay_lazy(tmp_path):
    """Combining images (and matching their coordinates) keeps their data lazy."""
    stores = {}
    for product in ("image", "psf"):
        stores[product], _ = _make_casa_image(tmp_path / f"lazy.{product}")
    xds = open_image(list(stores.values()))
    assert dask.is_dask_collection(xds.SKY.data)
    assert dask.is_dask_collection(xds.POINT_SPREAD_FUNCTION.data)


# --------------------------------------------------------------------------- #
# Canonical polarization order                                                 #
# --------------------------------------------------------------------------- #


def _to_fits_correlation_order(coords):
    """Give a CASA image the correlations in the order of FITS files."""
    coords["stokes1"]["stokes"] = ["RR", "LL", "RL", "LR"]


class TestCanonicalPolarizationOrder:
    """In memory the polarization axis is always in canonical (Jones) order."""

    @pytest.mark.parametrize(
        "factory", [make_empty_sky_image, make_empty_aperture_image]
    )
    def test_factories_reject_other_orders(self, factory):
        args = list(_FACTORY_ARGS)
        args[4] = ["RR", "LL", "RL", "LR"]
        with pytest.raises(ValueError, match="canonical") as error:
            factory(*args)
        assert "['RR', 'RL', 'LR', 'LL']" in str(error.value)

    def test_casa_image_in_fits_order(self, tmp_path):
        """A CASA image converted from FITS (importfits) keeps the FITS order
        RR, LL, RL, LR; it opens in canonical order, pixels, masks and
        selections included."""
        shape = (3, 4, 6, 8)
        path, data = _make_casa_image(
            tmp_path / "rrll.image", shape=shape, edit_coords=_to_fits_correlation_order
        )
        # stored order RR, LL, RL, LR -> canonical RR, RL, LR, LL
        order = [0, 2, 3, 1]
        expected = np.transpose(data, (0, 1, 3, 2))[:, order]
        for xds in (open_image(path), load_image(path)):
            assert xds.polarization.values.tolist() == ["RR", "RL", "LR", "LL"]
            np.testing.assert_array_equal(xds.SKY.values[0], expected)
            assert not check_image(xds)
        part = load_image(path, {"polarization": slice(1, 3), "frequency": 1})
        assert part.polarization.values.tolist() == ["RL", "LR"]
        np.testing.assert_array_equal(part.SKY.values[0], expected[1:2, 1:3])
        single = load_image(path, {"polarization": 3})
        assert single.polarization.values.tolist() == ["LL"]
        np.testing.assert_array_equal(single.SKY.values[0], expected[:, 3:4])

    def test_zarr_store_in_another_order(self, tmp_path):
        xds = make_empty_sky_image(
            [0.2, -0.5],
            [4, 3],
            [_CELL, _CELL],
            [1.4e9],
            ["XX", "XY", "YX", "YY"],
            [59000.5],
        )
        data = np.arange(4 * 4 * 3, dtype=np.float32).reshape(1, 1, 4, 4, 3)
        xds["SKY"] = (("time", "frequency", "polarization", "l", "m"), data)
        xds.attrs["data_groups"]["base"]["sky"] = "SKY"
        other = xds.isel(polarization=[3, 0, 2, 1])
        assert check_image(other)
        store = str(tmp_path / "xyyx.zarr")
        write_image(other, store, out_format="zarr")
        for reader in (open_image, load_image):
            back = reader(store)
            assert back.polarization.values.tolist() == ["XX", "XY", "YX", "YY"]
            np.testing.assert_array_equal(back.SKY.values, data)


# --------------------------------------------------------------------------- #
# More upgrades of image stores written by xradio <= 1.2.3                     #
# --------------------------------------------------------------------------- #


_DIMS5 = ("time", "frequency", "polarization", "l", "m")


def _tclean_list_dataset():
    """A dataset as open_image of a list of tclean products gave it in
    xradio <= 1.2.3: the deconvolution products in the sky image's group."""
    xds = make_empty_sky_image(*_FACTORY_ARGS)
    shape = tuple(xds.sizes[d] for d in _DIMS5)
    for value, (name, image_type) in enumerate(
        [
            ("SKY", "sky"),
            ("MODEL", "model"),
            ("RESIDUAL", "residual"),
            ("MASK_DECONVOLVE", "mask_deconvolve"),
            ("POINT_SPREAD_FUNCTION", "point_spread_function"),
            ("PRIMARY_BEAM", "primary_beam"),
        ]
    ):
        xds[name] = (_DIMS5, np.full(shape, value, np.float32), {"type": image_type})
    xds["FLAG_SKY"] = (_DIMS5, np.zeros(shape, bool), {"type": "flag"})
    xds["FLAG_MODEL"] = (_DIMS5, np.zeros(shape, bool), {"type": "flag"})
    xds.attrs["data_groups"] = {
        "base": {
            "sky": "SKY",
            "flag": "FLAG_SKY",
            "model": "MODEL",
            "residual": "RESIDUAL",
            "mask_deconvolve": "MASK_DECONVOLVE",
            "point_spread_function": "POINT_SPREAD_FUNCTION",
            "primary_beam": "PRIMARY_BEAM",
        }
    }
    return xds


class TestLegacyRolesAndAttributes:
    def test_tclean_list_roles(self):
        legacy = _tclean_list_dataset()
        assert any("Unknown data group role" in i.message for i in check_image(legacy))
        xds = upgrade_legacy_image_attrs(legacy)
        shared = {
            "mask": "MASK_DECONVOLVE",
            "point_spread_function": "POINT_SPREAD_FUNCTION",
            "primary_beam": "PRIMARY_BEAM",
        }
        assert xds.attrs["data_groups"] == {
            "base": {"sky": "SKY", "flag": "FLAG_SKY", **shared},
            "model": {**shared, "sky": "MODEL", "flag": "FLAG_MODEL"},
            "residual": {**shared, "sky": "RESIDUAL"},
        }
        assert xds.MODEL.attrs["type"] == "sky"
        assert xds.RESIDUAL.attrs["type"] == "sky"
        assert xds.MASK_DECONVOLVE.attrs["type"] == "mask"
        assert not check_image(xds)
        # the input is not modified, and the upgrade is done once
        assert "model" in legacy.attrs["data_groups"]["base"]
        assert legacy.MODEL.attrs["type"] == "model"
        assert upgrade_legacy_image_attrs(xds) is xds

    def test_upgraded_tclean_list_is_written(self, tmp_path):
        store = str(tmp_path / "legacy_list.zarr")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _tclean_list_dataset().to_zarr(store)
        xds = open_image(store)
        out = str(tmp_path / "out")
        write_image(xds, out, out_format="casa")
        written = sorted(os.listdir(tmp_path))
        assert "out.model.sky" in written and "out.residual.sky" in written

    def test_missing_types_and_beam_units(self):
        """As in AstroVIPER stores: variables without type, beams without
        units (the writers take beams without units as radians)."""
        xds = make_empty_sky_image(*_FACTORY_ARGS)
        shape = tuple(xds.sizes[d] for d in _DIMS5)
        xds["SKY"] = (_DIMS5, np.ones(shape, np.float32))
        xds["BEAM_FIT_PARAMS_SKY"] = (
            ("time", "frequency", "polarization", "beam_params_label"),
            np.full(shape[:3] + (3,), 1e-5),
        )
        xds = xds.assign_coords(beam_params_label=["major", "minor", "pa"])
        xds.attrs["data_groups"] = {
            "base": {"sky": "SKY", "beam_fit_params_sky": "BEAM_FIT_PARAMS_SKY"}
        }
        assert check_image(xds)
        upgraded = upgrade_legacy_image_attrs(xds)
        assert upgraded.SKY.attrs["type"] == "sky"
        assert upgraded.BEAM_FIT_PARAMS_SKY.attrs == {
            "type": "beam_fit_params_sky",
            "units": "rad",
        }
        assert not check_image(upgraded)
        assert "type" not in xds.SKY.attrs

    def test_factory_fk4_equinox(self, tmp_path):
        """make_empty_* of xradio <= 1.2.3 stamped equinox j2000.0 on fk4."""
        xds = make_empty_sky_image(*_FACTORY_ARGS, direction_reference="fk4")
        attrs = xds.attrs["coordinate_system_info"]["reference_direction"]["attrs"]
        assert attrs["equinox"] == "b1950.0"
        attrs["equinox"] = "j2000.0"
        # only stores of the old factories (dataset type "image") are changed
        assert upgrade_legacy_image_attrs(xds) is xds
        xds.attrs["type"] = "image"
        upgraded = upgrade_legacy_image_attrs(xds)
        reference = upgraded.attrs["coordinate_system_info"]["reference_direction"]
        assert reference["attrs"]["equinox"] == "b1950.0"
        assert attrs["equinox"] == "j2000.0"
        shape = tuple(upgraded.sizes[d] for d in _DIMS5)
        upgraded["SKY"] = (_DIMS5, np.ones(shape, np.float32), {"type": "sky"})
        upgraded.attrs["data_groups"]["base"]["sky"] = "SKY"
        path = tmp_path / "fk4.im"
        write_image(upgraded, str(path), out_format="casa")
        with tables.table(str(path), ack=False) as tb:
            assert tb.getkeyword("coords")["direction0"]["system"] == "B1950"

    def test_uv_coordinates_without_units(self):
        """add_uv_coordinates of xradio <= 1.2.3 gave u and v in wavelengths
        without units."""
        xds = make_empty_sky_image(*_FACTORY_ARGS)
        xds = xds.assign_coords(
            u=("u", np.arange(4.0) * 10), v=("v", np.arange(3.0) * 10)
        )
        upgraded = upgrade_legacy_image_attrs(xds)
        assert upgraded.u.attrs == {"units": "lambda"}
        assert upgraded.v.attrs == {"units": "lambda"}
        u_lambda, v_lambda = upgraded.xr_img.get_uv_in_lambda(1.4e9)
        np.testing.assert_array_equal(u_lambda.values, xds.u.values)
        np.testing.assert_array_equal(v_lambda.values, xds.v.values)


# --------------------------------------------------------------------------- #
# Images without an observation date                                           #
# --------------------------------------------------------------------------- #


def _write_dated_fits(path, date_obs):
    from astropy.io import fits

    header = fits.Header()
    for key, value in {
        "CTYPE1": "RA---SIN",
        "CRVAL1": 10.0,
        "CDELT1": -1 / 3600,
        "CRPIX1": 2.0,
        "CUNIT1": "deg",
        "CTYPE2": "DEC--SIN",
        "CRVAL2": -30.0,
        "CDELT2": 1 / 3600,
        "CRPIX2": 2.0,
        "CUNIT2": "deg",
        "CTYPE3": "STOKES",
        "CRVAL3": 1.0,
        "CDELT3": 1.0,
        "CRPIX3": 1.0,
        "CTYPE4": "FREQ",
        "CRVAL4": 1e11,
        "CDELT4": 1e6,
        "CRPIX4": 1.0,
        "CUNIT4": "Hz",
        "RESTFRQ": 1e11,
        "SPECSYS": "LSRK",
        "RADESYS": "FK5",
        "EQUINOX": 2000.0,
        "TELESCOP": "ALMA",
    }.items():
        header[key] = value
    if date_obs is not None:
        header["DATE-OBS"] = date_obs
        header["TIMESYS"] = "UTC"
    data = np.ones((2, 1, 3, 4), np.float32)
    fits.PrimaryHDU(data=data, header=header).writeto(path)
    return str(path)


class TestImagesWithoutObservationDate:
    """An image without a date (placeholder time MJD 0) takes the date of
    the images it is opened with, in either order."""

    @pytest.mark.parametrize("dated_first", [True, False])
    def test_undated_image_takes_the_shared_date(self, tmp_path, dated_first):
        dated = _write_dated_fits(tmp_path / "dated.fits", "2020-01-01T00:00:00")
        undated = _write_dated_fits(tmp_path / "undated.fits", None)
        assert open_image(undated).time.values.tolist() == [0.0]
        stores = (
            {"sky": dated, "point_spread_function": undated}
            if dated_first
            else {"point_spread_function": undated, "sky": dated}
        )
        xds = open_image(stores)
        np.testing.assert_allclose(xds.time.values, [58849.0])
        assert not np.isnan(xds.POINT_SPREAD_FUNCTION.values).any()
        assert not np.isnan(xds.SKY.values).any()

    def test_different_dates_still_raise(self, tmp_path):
        first = _write_dated_fits(tmp_path / "a.fits", "2020-01-01T00:00:00")
        second = _write_dated_fits(tmp_path / "b.fits", "2020-01-02T00:00:00")
        with pytest.raises(ValueError, match="time coordinate differs"):
            open_image({"sky": first, "point_spread_function": second})
