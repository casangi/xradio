"""Unit tests for the FITS image writer (``xradio.image._util._fits.xds_to_fits``).

* ``TestPolarizationAxis`` -> FITS STOKES codes, plane order, round trips
* ``TestSpectralAxis``     -> SPECSYS, VELREF, spectral frame round trips
* ``TestFrequencyAxis``    -> linear FREQ axis, units, single channel width
* ``TestDirectionAxes``    -> reference pixel, single pixel axes, PV cards,
                              EQUINOX, pointing center (OBSRA/OBSDEC)
* ``TestPixelDataTypes``   -> BITPIX for each numpy data type
* ``TestBeams``            -> single beam cards, BEAMS table, beam labels
* ``TestHeaderText``       -> non-ASCII text, user keywords
* ``TestObservationDate``  -> DATE-OBS, MJD-OBS, TIMESYS
* ``TestBlockWiseWrite``   -> chunk-wise writing, no overwrite, cleanup
* ``TestWithoutCasacore``  -> FITS write and read without casacore

The images are synthetic (no downloads). Each test writes the single image
dataset that the FITS driver passes to the writer (SKY, FLAG and
BEAM_FIT_PARAMS); the round trip tests read the file back with
``open_image``.
"""

import os
import pickle
import subprocess
import sys
import textwrap

import dask
import dask.array as da
import numpy as np
import pytest
import xarray as xr
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS

from xradio._utils.dict_helpers import make_quantity
from xradio.image import open_image, write_image
from xradio.image._util import conventions
from xradio.image._util._fits import xds_to_fits
from xradio.image._util._fits.xds_to_fits import (
    _fits_image_header,
    _xds_to_fits_image,
)

_FITS_DIMS = ("frequency", "polarization", "m", "l")


def make_image_xds(
    pols=("I", "Q", "U", "V"),
    nchan=3,
    shape=(6, 5),
    crpix=(2.0, 2.0),
    cell=(-1e-5, 1e-5),
    frequencies=None,
    frame="LSRK",
    doppler="radio",
    dtype=np.float32,
) -> xr.Dataset:
    """Build a small sky image dataset as ``open_image`` returns it.

    Every pixel value is distinct (it encodes channel, polarization, l and m),
    a few pixels are flagged and every plane has its own beam.
    """
    nl, nm = shape
    npol = len(pols)
    if frequencies is None:
        frequencies = 1.0e11 + 1.0e6 * np.arange(nchan)
    frequencies = np.asarray(frequencies, dtype=float)
    nchan = frequencies.size
    rest_frequency = float(frequencies[0])
    dec0 = -0.5
    l_values = (np.arange(nl) - crpix[0]) * cell[0]
    m_values = (np.arange(nm) - crpix[1]) * cell[1]

    chan, pol, li, mi = np.meshgrid(
        np.arange(nchan), np.arange(npol), np.arange(nl), np.arange(nm), indexing="ij"
    )
    sky = (1000.0 * chan + 100.0 * pol + 10.0 * li + mi + 0.5)[np.newaxis]
    flag = ((chan + pol + li + mi) % 7 == 0)[np.newaxis]
    major = 1e-5 * (1.0 + 0.1 * np.arange(nchan)[:, None] + 0.01 * np.arange(npol))
    beams = np.stack(
        [major, 0.5 * major, 0.1 + 0.0 * major + 0.01 * np.arange(npol)], axis=-1
    )[np.newaxis]

    telescope = {
        "name": "ALMA",
        "direction": {
            "attrs": {
                "coordinate_system": "geocentric",
                "frame": "ITRF",
                "origin_object_name": "earth",
                "type": "location",
                "units": "rad",
            },
            "data": [-1.18, -0.4],
            "dims": ["ellipsoid_dir_label"],
            "coords": {
                "ellipsoid_dir_label": {
                    "dims": ["ellipsoid_dir_label"],
                    "data": ["lon", "lat"],
                }
            },
        },
        "distance": {
            "attrs": {
                "coordinate_system": "geocentric",
                "frame": "ITRF",
                "origin_object_name": "earth",
                "type": "location",
                "units": "m",
            },
            "data": [6379946.0],
            "dims": ["ellipsoid_dis_label"],
            "coords": {
                "ellipsoid_dis_label": {
                    "dims": ["ellipsoid_dis_label"],
                    "data": ["dist"],
                }
            },
        },
    }
    sky_dir = {
        "attrs": {"frame": "fk5", "type": "sky_coord", "units": "rad"},
        "data": [0.2, dec0],
        "dims": "sky_dir_label",
        "coords": {"sky_dir_label": {"data": ["ra", "dec"], "dims": "sky_dir_label"}},
    }
    coords = {
        "time": (
            "time",
            [54000.123456789],
            {"type": "time", "units": "d", "scale": "utc", "format": "mjd"},
        ),
        "frequency": (
            "frequency",
            frequencies,
            {
                "rest_frequency": make_quantity(rest_frequency, "Hz"),
                "reference_frequency": {
                    "attrs": {
                        "units": "Hz",
                        "observer": conventions.spectral_frame_to_observer(frame),
                        "type": "spectral_coord",
                    },
                    "data": float(frequencies[0]),
                    "dims": [],
                },
                "type": "spectral_coord",
                "units": "Hz",
                "frame": frame,
                "wave_units": "mm",
            },
        ),
        "velocity": (
            "frequency",
            (1 - frequencies / rest_frequency) * 299792458.0,
            {"doppler_type": doppler, "units": "m/s", "type": "doppler"},
        ),
        "polarization": ("polarization", list(pols)),
        "l": ("l", l_values, {"note": "l"}),
        "m": ("m", m_values, {"note": "m"}),
        "beam_params_label": ("beam_params_label", ["major", "minor", "pa"]),
    }
    dims = ("time", "frequency", "polarization", "l", "m")
    sky_attrs = {
        "type": "sky",
        "units": "Jy/beam",
        "telescope": telescope,
        "obsdate": {
            "attrs": {"units": "d", "scale": "utc", "format": "mjd", "type": "time"},
            "data": 54000.123456789,
            "dims": [],
        },
        "pointing_center": {**sky_dir, "data": [0.2001, dec0 + 0.0002]},
        "object_name": "test_object",
        "observer": "Karl Jansky",
        "beam_fit_params": "BEAM_FIT_PARAMS_SKY",
        "sub_type": "Intensity",
    }
    attrs = {
        "coordinate_system_info": {
            "reference_direction": {
                **sky_dir,
                "attrs": {**sky_dir["attrs"], "equinox": "j2000.0"},
            },
            "native_pole_direction": {
                "attrs": {
                    "frame": "NATIVE_PROJECTION",
                    "type": "location",
                    "units": "rad",
                },
                "data": [np.pi, dec0],
                "dims": "ellipsoid_dir_label",
                "coords": {
                    "ellipsoid_dir_label": {
                        "data": ["lon", "lat"],
                        "dims": "ellipsoid_dir_label",
                    }
                },
            },
            "projection": "SIN",
            "projection_parameters": [0.0, 0.0],
            "pixel_coordinate_transformation_matrix": [[1.0, 0.0], [0.0, 1.0]],
        },
        "type": "image_dataset",
        "data_groups": {
            "base": {
                "sky": "SKY",
                "flag": "FLAG_SKY",
                "beam_fit_params_sky": "BEAM_FIT_PARAMS_SKY",
            }
        },
    }
    return xr.Dataset(
        {
            "SKY": (dims, sky.astype(dtype), sky_attrs),
            "FLAG_SKY": (dims, flag, {"type": "flag"}),
            "BEAM_FIT_PARAMS_SKY": (
                ("time", "frequency", "polarization", "beam_params_label"),
                beams,
                {"units": "rad", "type": "beam_fit_params_sky"},
            ),
        },
        coords=coords,
        attrs=attrs,
    )


def single_image(xds: xr.Dataset) -> xr.Dataset:
    """The single image dataset the FITS driver passes to the writer."""
    image = xr.Dataset(attrs=dict(xds.attrs))
    image["SKY"] = xds["SKY"]
    if "FLAG_SKY" in xds.data_vars:
        image["FLAG"] = xds["FLAG_SKY"]
    if "BEAM_FIT_PARAMS_SKY" in xds.data_vars:
        image["BEAM_FIT_PARAMS"] = xds["BEAM_FIT_PARAMS_SKY"]
    return image


def write(xds: xr.Dataset, path) -> str:
    """Write the sky image of ``xds`` to ``path`` with the FITS writer."""
    _xds_to_fits_image(single_image(xds), str(path))
    return str(path)


def header_of(xds: xr.Dataset) -> fits.Header:
    """The primary header the writer would write for ``xds``."""
    return _fits_image_header(single_image(xds))[0]


def expected_cube(xds: xr.Dataset, fits_order) -> np.ndarray:
    """The FITS data cube (frequency, FITS polarization, m, l) of ``xds``,
    with flagged pixels as NaN."""
    sky = xds["SKY"].isel(time=0).transpose(*_FITS_DIMS).values
    flag = xds["FLAG_SKY"].isel(time=0).transpose(*_FITS_DIMS).values
    return np.where(flag, np.nan, sky)[:, list(fits_order)]


class _RecordingLogger:
    """Stand-in for the xradio logger that records its messages."""

    def __init__(self):
        self.messages = {"warning": [], "info": [], "debug": []}

    def warning(self, message):
        self.messages["warning"].append(message)

    def info(self, message):
        self.messages["info"].append(message)

    def debug(self, message):
        self.messages["debug"].append(message)


@pytest.fixture
def recorded_log(monkeypatch):
    logger = _RecordingLogger()
    monkeypatch.setattr(xds_to_fits, "xradio_logger", lambda: logger)
    return logger.messages


class TestPolarizationAxis:
    """F2: FITS STOKES codes on a linear axis; the dataset keeps its order."""

    @pytest.mark.parametrize(
        "pols, fits_order, crval, cdelt",
        [
            (("I", "Q", "U", "V"), [0, 1, 2, 3], 1, 1),
            (("RR", "RL", "LR", "LL"), [0, 3, 1, 2], -1, -1),
            (("XX", "XY", "YX", "YY"), [0, 3, 1, 2], -5, -1),
            (("XX", "YY"), [0, 1], -5, -1),
            (("RR", "LL"), [0, 1], -1, -1),
            (("I", "V"), [0, 1], 1, 3),
            (("RR",), [0], -1, 1),
        ],
    )
    def test_fits_codes_and_plane_order(self, tmp_path, pols, fits_order, crval, cdelt):
        xds = make_image_xds(pols=pols)
        before = xds.copy(deep=True)
        path = write(xds, tmp_path / "img.fits")
        with fits.open(path) as hdul:
            hdul.verify("exception")
            header = hdul[0].header
            assert header["CTYPE3"] == "STOKES"
            assert (header["CRVAL3"], header["CDELT3"], header["CRPIX3"]) == (
                crval,
                cdelt,
                1,
            )
            labels = conventions.fits_stokes_labels(
                header["CRVAL3"], header["CDELT3"], header["CRPIX3"], header["NAXIS3"]
            )
            assert labels == [pols[i] for i in fits_order]
            # planes and their flags (NaN) follow the FITS axis order
            np.testing.assert_array_equal(hdul[0].data, expected_cube(xds, fits_order))
            # so do the per plane beams
            table = hdul["BEAMS"].data
            beams = xds["BEAM_FIT_PARAMS_SKY"].values[0]
            for row in table:
                expected = beams[row["CHAN"], fits_order[row["POL"]]]
                np.testing.assert_allclose(
                    row["BMAJ"], np.degrees(expected[0]) * 3600, rtol=1e-6
                )
                np.testing.assert_allclose(
                    row["BPA"], np.degrees(expected[2]), rtol=1e-6
                )
        # the caller's dataset is not reordered or otherwise changed
        xr.testing.assert_identical(xds, before)

    @pytest.mark.parametrize(
        "pols",
        [
            ("RR", "RL", "LR", "LL"),
            ("XX", "XY", "YX", "YY"),
            ("I", "Q", "U", "V"),
            ("XX", "YY"),
            ("RR", "LL"),
            ("I", "V"),
        ],
    )
    def test_round_trip_restores_canonical_order(self, tmp_path, pols):
        xds = make_image_xds(pols=pols)
        rt = open_image(write(xds, tmp_path / "img.fits"))
        assert list(rt.polarization.values) == list(pols)
        flag = xds["FLAG_SKY"].values
        np.testing.assert_array_equal(rt["FLAG_SKY"].values, flag)
        np.testing.assert_array_equal(
            rt["SKY"].values, np.where(flag, np.nan, xds["SKY"].values)
        )
        np.testing.assert_allclose(
            rt["BEAM_FIT_PARAMS_SKY"].values,
            xds["BEAM_FIT_PARAMS_SKY"].values,
            rtol=1e-6,
        )

    @pytest.mark.parametrize(
        "pols", [("I", "Q", "V"), ("RR", "XX", "YY"), ("PP", "QQ")]
    )
    def test_unrepresentable_polarizations_raise_before_writing(self, tmp_path, pols):
        path = tmp_path / "img.fits"
        with pytest.raises(ValueError, match="Cannot write polarizations"):
            write(make_image_xds(pols=pols), path)
        assert not path.exists()


class TestSpectralAxis:
    """F3 and F10: SPECSYS and VELREF use the FITS and AIPS conventions."""

    @pytest.mark.parametrize(
        "frame, specsys", sorted(conventions.CASACORE_TO_FITS_SPECSYS.items())
    )
    def test_specsys_from_frame(self, frame, specsys):
        assert header_of(make_image_xds(frame=frame))["SPECSYS"] == specsys

    @pytest.mark.parametrize(
        "observer, specsys",
        [
            ("BARY", "BARYCENT"),
            ("TOPO", "TOPOCENT"),
            ("gcrs", "GEOCENTR"),
            ("lsrk", "LSRK"),
        ],
    )
    def test_specsys_from_observer_without_frame(self, observer, specsys):
        """Stores written by xradio 1.2.3 and earlier have no frame attribute:
        the reference frequency's observer gives the frame, not an LSRK
        default."""
        xds = make_image_xds()
        del xds["frequency"].attrs["frame"]
        xds["frequency"].attrs["reference_frequency"]["attrs"]["observer"] = observer
        assert header_of(xds)["SPECSYS"] == specsys

    def test_unknown_spectral_frame_raises_before_writing(self, tmp_path):
        xds = make_image_xds()
        del xds["frequency"].attrs["frame"]
        del xds["frequency"].attrs["reference_frequency"]["attrs"]["observer"]
        path = tmp_path / "img.fits"
        with pytest.raises(ValueError, match="spectral reference frame"):
            write(xds, path)
        assert not path.exists()

    @pytest.mark.parametrize("observer", ["icrs", "hcrs", "lsr"])
    def test_frames_without_fits_equivalent_raise(self, tmp_path, observer):
        xds = make_image_xds()
        del xds["frequency"].attrs["frame"]
        xds["frequency"].attrs["reference_frequency"]["attrs"]["observer"] = observer
        path = tmp_path / "img.fits"
        with pytest.raises(ValueError, match="no FITS SPECSYS equivalent"):
            write(xds, path)
        assert not path.exists()

    def test_frame_and_observer_disagree_warns(self, recorded_log):
        xds = make_image_xds(frame="BARY")
        xds["frequency"].attrs["reference_frequency"]["attrs"]["observer"] = "lsrk"
        assert header_of(xds)["SPECSYS"] == "BARYCENT"
        assert any("disagree" in m for m in recorded_log["warning"])

    @pytest.mark.parametrize(
        "frame, doppler, velref",
        [
            ("LSRK", "radio", 257),
            ("BARY", "z", 2),
            ("BARY", "optical", 2),
            ("TOPO", "radio", 259),
            ("LSRD", "radio", 260),
            ("GEO", "radio", 261),
            ("REST", "optical", 6),
            ("GALACTO", "radio", 263),
            ("LGROUP", "radio", None),
            ("CMB", "optical", None),
        ],
    )
    def test_velref(self, frame, doppler, velref):
        header = header_of(make_image_xds(frame=frame, doppler=doppler))
        assert header.get("VELREF") == velref

    @pytest.mark.parametrize("frame", ["BARY", "TOPO", "GEO", "LSRD", "CMB"])
    def test_round_trip_spectral_frame(self, tmp_path, frame):
        rt = open_image(write(make_image_xds(frame=frame), tmp_path / "img.fits"))
        attrs = rt["frequency"].attrs
        assert attrs["frame"] == frame
        assert attrs["reference_frequency"]["attrs"][
            "observer"
        ] == conventions.spectral_frame_to_observer(frame)

    @pytest.mark.parametrize("doppler", ["radio", "z"])
    def test_round_trip_velocity_convention(self, tmp_path, doppler):
        xds = make_image_xds(doppler=doppler)
        rt = open_image(write(xds, tmp_path / "img.fits"))
        written = rt["velocity"].attrs["doppler_type"]
        if doppler == "radio":
            assert written == "radio"
        else:
            assert written in ("z", "optical")


class TestFrequencyAxis:
    """F4 and F12: the FREQ axis is linear, in Hz, with a channel width."""

    def test_non_uniform_frequencies_raise_before_writing(self, tmp_path):
        xds = make_image_xds(frequencies=[1.000e9, 1.001e9, 1.005e9])
        path = tmp_path / "img.fits"
        with pytest.raises(ValueError, match="not uniformly spaced"):
            write(xds, path)
        assert not path.exists()

    def test_channel_subset_raises(self, tmp_path):
        xds = make_image_xds(nchan=7).isel(frequency=[0, 2, 5, 6])
        with pytest.raises(ValueError, match="not uniformly spaced"):
            write(xds, tmp_path / "img.fits")

    def test_nearly_linear_axis_is_written(self, tmp_path):
        """Frequencies derived from an optical velocity axis deviate from a
        linear axis by about 5e-5 of a channel."""
        frequencies = 1.0e11 + 1.0e6 * np.arange(5)
        frequencies[2] += 5e-5 * 1.0e6
        path = write(make_image_xds(frequencies=frequencies), tmp_path / "img.fits")
        header = fits.getheader(path)
        np.testing.assert_allclose(header["CDELT4"], 1.0e6, rtol=1e-12)

    def test_world_frequencies(self, tmp_path):
        xds = make_image_xds(nchan=4)
        # reference value off the first channel
        xds["frequency"].attrs["reference_frequency"]["data"] = 1.0e11 + 2.5e6
        header = fits.getheader(write(xds, tmp_path / "img.fits"))
        assert header["CUNIT4"] == "Hz"
        world = header["CRVAL4"] + header["CDELT4"] * (
            np.arange(1, 5) - header["CRPIX4"]
        )
        np.testing.assert_allclose(world, xds["frequency"].values, rtol=0, atol=1e-3)

    def test_non_hz_units_are_converted(self, tmp_path):
        xds = make_image_xds(nchan=4)
        attrs = xds["frequency"].attrs
        hz = xds["frequency"].values
        xds = xds.assign_coords(frequency=("frequency", hz / 1e9, attrs))
        xds["frequency"].attrs["units"] = "GHz"
        xds["frequency"].attrs["reference_frequency"]["data"] = hz[0] / 1e9
        xds["frequency"].attrs["reference_frequency"]["attrs"]["units"] = "GHz"
        header = fits.getheader(write(xds, tmp_path / "img.fits"))
        assert header["CUNIT4"] == "Hz"
        np.testing.assert_allclose(header["CRVAL4"], hz[0], rtol=1e-15)
        np.testing.assert_allclose(header["CDELT4"], 1.0e6, rtol=1e-9)

    def test_single_channel_width_from_attribute(self):
        xds = make_image_xds(nchan=1)
        xds["frequency"].attrs["channel_width"] = make_quantity(2.5e6, "Hz")
        assert header_of(xds)["CDELT4"] == 2.5e6
        xds["frequency"].attrs["channel_width"] = make_quantity(2.5, "MHz")
        assert header_of(xds)["CDELT4"] == 2.5e6

    def test_single_channel_width_fallback(self):
        header = header_of(make_image_xds(nchan=1))
        assert header["CDELT4"] == conventions.SINGLE_CHANNEL_WIDTH_FALLBACK_HZ
        assert header["CRPIX4"] == 1.0

    @pytest.mark.parametrize("width", [0.0, np.nan])
    def test_unusable_single_channel_width_falls_back(self, width, recorded_log):
        """As in the CASA writer, an unusable channel_width gives the
        conventional width, with a warning."""
        xds = make_image_xds(nchan=1)
        xds["frequency"].attrs["channel_width"] = make_quantity(width, "Hz")
        header = header_of(xds)
        assert header["CDELT4"] == conventions.SINGLE_CHANNEL_WIDTH_FALLBACK_HZ
        assert any("channel_width" in m for m in recorded_log["warning"])


class TestDirectionAxes:
    """F5, F12 and F14: celestial axes, projection and pointing center."""

    def test_reference_pixel_outside_image(self, tmp_path):
        """A cutout without the reference direction keeps its reference pixel
        (CRPIX outside the image) instead of a clamped one."""
        xds = make_image_xds(shape=(8, 7), crpix=(2.0, 3.0))
        cutout = xds.isel(l=slice(4, 8), m=slice(4, 7))
        full = fits.getheader(write(xds, tmp_path / "full.fits"))
        cut = fits.getheader(write(cutout, tmp_path / "cut.fits"))
        assert (cut["CRPIX1"], cut["CRPIX2"]) == (3.0 - 4.0, 4.0 - 4.0)
        world_full = WCS(full).celestial.pixel_to_world_values([4, 7], [4, 6])
        world_cut = WCS(cut).celestial.pixel_to_world_values([0, 3], [0, 2])
        np.testing.assert_allclose(world_cut, world_full, rtol=0, atol=1e-12)

    def test_single_pixel_axis_takes_the_other_increment(self, tmp_path):
        """Without a cdelt attribute, a single pixel axis takes the other
        axis' pixel size, as in the CASA writer."""
        xds = make_image_xds(shape=(1, 5), crpix=(0.0, 2.0))
        header = header_of(xds)
        assert header["CDELT1"] == -abs(header["CDELT2"])
        assert header["CRPIX1"] == 1.0
        rt = open_image(write(xds, tmp_path / "img.fits"))
        np.testing.assert_allclose(rt.l.values, [0.0], atol=1e-15)
        np.testing.assert_allclose(rt.m.values, xds.m.values, rtol=1e-12)

    def test_one_by_one_image_needs_increments(self, tmp_path):
        xds = make_image_xds(shape=(1, 1), crpix=(0.0, 0.0))
        path = tmp_path / "img.fits"
        with pytest.raises(ValueError, match="1 x 1 pixel"):
            write(xds, path)
        assert not path.exists()

    def test_single_pixel_axis_increment_from_attribute(self):
        xds = make_image_xds(shape=(5, 1), crpix=(2.0, -3.0))
        xds["m"].attrs["cdelt"] = 1e-5
        header = header_of(xds)
        np.testing.assert_allclose(header["CDELT2"], np.degrees(1e-5), rtol=1e-12)
        # m = 3e-5 at the only pixel: the reference pixel is 3 pixels below
        np.testing.assert_allclose(header["CRPIX2"], 1.0 - 3.0, rtol=0, atol=1e-9)
        np.testing.assert_allclose(header["CRPIX1"], 3.0, rtol=0, atol=1e-9)

    def test_projection_parameters(self):
        xds = make_image_xds()
        assert "PV2_1" not in header_of(xds)
        xds.attrs["coordinate_system_info"]["projection_parameters"] = [0.0, -1.83]
        header = header_of(xds)
        assert (header["PV2_1"], header["PV2_2"]) == (0.0, -1.83)

    def test_zpn_projection_parameters_round_trip(self, tmp_path):
        """ZPN parameters are numbered from PV2_0 (FITS WCS paper II), as the
        reader numbers them, so a ZPN image reads, writes and reads back with
        the same parameters and sky positions."""
        xds = make_image_xds(shape=(9, 7), crpix=(4.0, 3.0), cell=(-1e-3, 1e-3))
        xds.attrs["coordinate_system_info"]["projection"] = "ZPN"
        xds.attrs["coordinate_system_info"]["projection_parameters"] = [
            0.0,
            1.0,
            0.0,
            0.05,
        ]
        path = write(xds, tmp_path / "zpn.fits")
        header = fits.getheader(path)
        assert header["CTYPE1"] == "RA---ZPN"
        assert [header.get(f"PV2_{m}") for m in range(5)] == [
            0.0,
            1.0,
            0.0,
            0.05,
            None,
        ]
        # astropy (wcslib) accepts the parameters and the polynomial is used
        wcs = WCS(header).celestial
        corner_ra, corner_dec = wcs.pixel_to_world_values(0, 0)
        assert np.isfinite(corner_ra) and np.isfinite(corner_dec)
        back = open_image(path)
        np.testing.assert_allclose(
            back.attrs["coordinate_system_info"]["projection_parameters"],
            [0.0, 1.0, 0.0, 0.05],
        )
        path2 = write(back, tmp_path / "zpn2.fits")
        header2 = fits.getheader(path2)
        assert [header2.get(f"PV2_{m}") for m in range(5)] == [
            0.0,
            1.0,
            0.0,
            0.05,
            None,
        ]
        np.testing.assert_allclose(
            WCS(header2).celestial.pixel_to_world_values(0, 0),
            (corner_ra, corner_dec),
            rtol=0,
            atol=1e-10,
        )

    def test_slant_orthographic_positions_match_casacore(self, tmp_path):
        """An NCP-like SIN image (projection parameters [0, cot(dec0)]) has the
        same sky positions through FITS as in casacore."""
        images = pytest.importorskip("casacore.images")
        shape = (2, 1, 20, 30)
        template = images.image(str(tmp_path / "template.im"), shape=shape)
        coords = template.coordinates().dict()
        del template
        dec0 = np.radians(-40.0)
        direction = coords["direction0"]
        direction["units"] = ["rad", "rad"]
        direction["crval"] = np.array([np.radians(105.0), dec0])
        direction["cdelt"] = np.array([-np.radians(1 / 60), np.radians(1 / 60)])
        direction["crpix"] = np.array([15.0, 10.0])
        direction["projection_parameters"] = np.array([0.0, 1 / np.tan(dec0)])
        casa_path = str(tmp_path / "slant.im")
        image = images.image(
            casa_path,
            shape=shape,
            coordsys=images.coordinates.coordinatesystem(coords),
        )
        image.putdata(np.ones(shape, dtype=np.float32))
        corners = [(0, 0), (29, 0), (0, 19), (29, 19)]
        casa_world = [image.toworld([0, 0, y, x])[2:][::-1] for x, y in corners]
        del image
        xds = open_image(casa_path)
        fits_path = tmp_path / "slant.fits"
        _xds_to_fits_image(
            xr.Dataset({"SKY": xds["SKY"]}, attrs=dict(xds.attrs)), str(fits_path)
        )
        wcs = WCS(fits.getheader(fits_path)).celestial
        for (x, y), (ra, dec) in zip(corners, casa_world, strict=True):
            fits_ra, fits_dec = wcs.pixel_to_world_values(x, y)
            assert abs(np.degrees(ra) % 360 - fits_ra) * 3600 < 1e-6
            assert abs(np.degrees(dec) - fits_dec) * 3600 < 1e-6

    @pytest.mark.parametrize(
        "frame, equinox, radesys, fits_equinox",
        [
            ("fk5", "j2000.0", "FK5", 2000.0),
            ("fk5", None, "FK5", 2000.0),
            ("fk4", None, "FK4", 1950.0),
            ("fk4", "b1950.0", "FK4", 1950.0),
            ("icrs", None, "ICRS", None),
        ],
    )
    def test_equinox(self, frame, equinox, radesys, fits_equinox):
        xds = make_image_xds()
        attrs = xds.attrs["coordinate_system_info"]["reference_direction"]["attrs"]
        attrs["frame"] = frame
        attrs.pop("equinox")
        if equinox is not None:
            attrs["equinox"] = equinox
        xds["SKY"].attrs.pop("pointing_center")
        header = header_of(xds)
        assert header["RADESYS"] == radesys
        assert header.get("EQUINOX") == fits_equinox

    @pytest.mark.parametrize(
        "frame, equinox, expected_equinox",
        [("fk4", None, "b1950.0"), ("fk5", None, "j2000.0"), ("icrs", None, None)],
    )
    def test_round_trip_direction_frame(
        self, tmp_path, frame, equinox, expected_equinox
    ):
        xds = make_image_xds()
        attrs = xds.attrs["coordinate_system_info"]["reference_direction"]["attrs"]
        attrs["frame"] = frame
        attrs.pop("equinox")
        xds["SKY"].attrs["pointing_center"]["attrs"]["frame"] = frame
        rt = open_image(write(xds, tmp_path / "img.fits"))
        reference = rt.attrs["coordinate_system_info"]["reference_direction"]
        assert reference["attrs"]["frame"] == frame
        assert reference["attrs"].get("equinox") == expected_equinox
        np.testing.assert_allclose(reference["data"], [0.2, -0.5], rtol=0, atol=1e-14)

    def test_non_equatorial_frame_raises(self, tmp_path):
        xds = make_image_xds()
        reference = xds.attrs["coordinate_system_info"]["reference_direction"]
        reference["attrs"]["frame"] = "galactic"
        path = tmp_path / "img.fits"
        with pytest.raises(RuntimeError, match="equatorial"):
            write(xds, path)
        assert not path.exists()

    def test_pointing_center(self):
        xds = make_image_xds()
        xds["SKY"].attrs["pointing_center"]["data"] = [-0.3, -0.45]
        header = header_of(xds)
        np.testing.assert_allclose(header["OBSRA"], np.degrees(-0.3) + 360, rtol=1e-14)
        np.testing.assert_allclose(header["OBSDEC"], np.degrees(-0.45), rtol=1e-14)

    def test_pointing_center_in_another_frame_is_converted(self):
        from astropy.coordinates import FK5, SkyCoord

        xds = make_image_xds()
        pointing = xds["SKY"].attrs["pointing_center"]
        pointing["attrs"]["frame"] = "icrs"
        expected = SkyCoord(*pointing["data"], unit="rad", frame="icrs").transform_to(
            FK5(equinox="J2000")
        )
        header = header_of(xds)
        np.testing.assert_allclose(header["OBSRA"], expected.ra.deg, rtol=0, atol=1e-10)
        np.testing.assert_allclose(
            header["OBSDEC"], expected.dec.deg, rtol=0, atol=1e-10
        )
        # the frames differ by about 20 mas, so the conversion is visible
        assert abs(expected.ra.deg - np.degrees(pointing["data"][0])) > 1e-6

    def test_round_trip_pointing_center(self, tmp_path):
        xds = make_image_xds()
        rt = open_image(write(xds, tmp_path / "img.fits"))
        np.testing.assert_allclose(
            rt["SKY"].attrs["pointing_center"]["data"],
            xds["SKY"].attrs["pointing_center"]["data"],
            rtol=0,
            atol=1e-12,
        )


class TestPixelDataTypes:
    """F8: the FITS data type keeps the precision of the pixels."""

    @pytest.mark.parametrize(
        "dtype, bitpix",
        [
            (">f8", -64),
            ("<f8", -64),
            (">f4", -32),
            ("<f4", -32),
            ("f2", -32),
            ("bool", -32),
            ("i2", -32),
            ("u1", -32),
            ("i4", -64),
            ("i8", -64),
        ],
    )
    def test_bitpix(self, tmp_path, dtype, bitpix):
        xds = make_image_xds()
        xds["SKY"] = xds["SKY"].astype(dtype)
        before = xds["SKY"].dtype
        path = write(xds, tmp_path / "img.fits")
        with fits.open(path) as hdul:
            assert hdul[0].header["BITPIX"] == bitpix
            expected = expected_cube(xds, range(xds.sizes["polarization"]))
            np.testing.assert_array_equal(hdul[0].data, expected)
        assert xds["SKY"].dtype == before

    def test_big_endian_float64_keeps_precision(self, tmp_path):
        xds = make_image_xds()
        values = xds["SKY"].values.astype(">f8")
        values[0, 0, 1, 1, 1] = 1 + 1e-12
        values[0, 0, 1, 1, 2] = 1e-50
        xds["SKY"] = xds["SKY"].copy(data=values)
        with fits.open(write(xds, tmp_path / "img.fits")) as hdul:
            assert hdul[0].data[0, 1, 1, 1] == 1 + 1e-12
            assert hdul[0].data[0, 1, 2, 1] == 1e-50

    @pytest.mark.parametrize("dtype", ["complex64", "complex128"])
    def test_complex_raises_before_writing(self, tmp_path, dtype):
        xds = make_image_xds()
        xds["SKY"] = xds["SKY"].astype(dtype)
        path = tmp_path / "img.fits"
        with pytest.raises(ValueError, match="complex"):
            write(xds, path)
        assert not path.exists()

    def test_driver_error_names_the_variable_once(self, tmp_path):
        """write_image's error names the dataset variable, without repeating
        the writer's own description of the image."""
        xds = make_image_xds()
        psf = xds["SKY"].astype("complex64")
        psf.attrs = {"type": "point_spread_function"}
        xds["POINT_SPREAD_FUNCTION"] = psf
        xds.attrs["data_groups"]["base"]["point_spread_function"] = (
            "POINT_SPREAD_FUNCTION"
        )
        with pytest.raises(ValueError) as error:
            write_image(xds, str(tmp_path / "out.fits"), out_format="fits")
        message = str(error.value)
        assert message.startswith(
            "Cannot write POINT_SPREAD_FUNCTION to FITS: its data are complex"
        )
        assert message.count("Cannot write") == 1
        assert not list(tmp_path.iterdir())


class TestBeams:
    """F9 and R14(b): single and per plane beams."""

    def test_per_plane_beams_differing_by_less_than_a_milliarcsecond(self, tmp_path):
        """Circular beams with one position angle whose sizes differ by
        about 0.2 mas are per plane beams, not one beam."""
        xds = make_image_xds(nchan=4, pols=("I",))
        size = 1e-6 * (1 + 1e-3 * np.arange(4))  # 0.2 arcsec + k * 0.2 mas
        beams = np.zeros(xds["BEAM_FIT_PARAMS_SKY"].shape)
        beams[0, :, 0, 0] = size
        beams[0, :, 0, 1] = size
        xds["BEAM_FIT_PARAMS_SKY"].values[:] = beams
        with fits.open(write(xds, tmp_path / "img.fits")) as hdul:
            assert hdul[0].header["CASAMBM"]
            assert "BMAJ" not in hdul[0].header
            np.testing.assert_allclose(
                hdul["BEAMS"].data["BMAJ"], np.degrees(size) * 3600, rtol=1e-6
            )

    def test_equal_beams_written_as_one_beam(self, tmp_path):
        xds = make_image_xds()
        xds["BEAM_FIT_PARAMS_SKY"].values[:] = [3e-5, 2e-5, 0.25]
        with fits.open(write(xds, tmp_path / "img.fits")) as hdul:
            header = hdul[0].header
            assert len(hdul) == 1
            assert "CASAMBM" not in header
            np.testing.assert_allclose(
                [header["BMAJ"], header["BMIN"], header["BPA"]],
                np.degrees([3e-5, 2e-5, 0.25]),
                rtol=1e-14,
            )

    def test_beams_selected_by_label(self):
        xds = make_image_xds()
        xds["BEAM_FIT_PARAMS_SKY"].values[:] = [3e-5, 2e-5, 0.25]
        reordered = xds["BEAM_FIT_PARAMS_SKY"].isel(beam_params_label=[1, 0, 2])
        xds = xds.drop_vars("BEAM_FIT_PARAMS_SKY").drop_vars("beam_params_label")
        xds["BEAM_FIT_PARAMS_SKY"] = reordered
        assert list(xds["beam_params_label"].values) == ["minor", "major", "pa"]
        header = header_of(xds)
        np.testing.assert_allclose(header["BMAJ"], np.degrees(3e-5), rtol=1e-14)
        np.testing.assert_allclose(header["BMIN"], np.degrees(2e-5), rtol=1e-14)

    def test_missing_beam_label_raises_before_writing(self, tmp_path):
        xds = make_image_xds()
        two = xds["BEAM_FIT_PARAMS_SKY"].isel(beam_params_label=[0, 1])
        xds = xds.drop_vars("BEAM_FIT_PARAMS_SKY").drop_vars("beam_params_label")
        xds["BEAM_FIT_PARAMS_SKY"] = two
        path = tmp_path / "img.fits"
        with pytest.raises(ValueError, match="lacks"):
            write(xds, path)
        assert not path.exists()


class TestHeaderText:
    """K3: header text, and the user keywords that are copied."""

    def test_non_ascii_text_is_replaced_with_one_warning(self, tmp_path, recorded_log):
        xds = make_image_xds()
        xds["SKY"].attrs["object_name"] = "ρ Oph A"
        xds["SKY"].attrs["observer"] = "Karl\tJansky"
        xds["SKY"].attrs["user"] = {"place": "café"}
        with fits.open(write(xds, tmp_path / "img.fits")) as hdul:
            hdul.verify("exception")
            header = hdul[0].header
            assert header["OBJECT"] == "? Oph A"
            assert header["OBSERVER"] == "Karl?Jansky"
            assert header["PLACE"] == "caf?"
        replaced = [m for m in recorded_log["warning"] if "printable ASCII" in m]
        assert len(replaced) == 1
        assert "OBJECT" in replaced[0] and "OBSERVER" in replaced[0]

    def test_user_keywords(self, tmp_path):
        xds = make_image_xds()
        # one beam for all planes: no CASAMBM of the writer's own
        xds["BEAM_FIT_PARAMS_SKY"].values[:] = [3e-5, 2e-5, 0.25]
        xds["SKY"].attrs["user"] = {
            # stale or conflicting with what the writer manages
            "bscale": 2.0,
            "naxis1": 7,
            "datamin": -1.0,
            "checksum": "abc",
            "casambm": True,
            "crota2": 5.0,
            "pc001001": 2.0,
            "mjd-obs": 1.0,
            # copied
            "origin": "test suite",
            "instrume": "ALMA-BAND6",
            "comment": "a comment",
            # not representable
            "toolongkeyword": 1,
            "nested": {"a": 1},
        }
        with fits.open(write(xds, tmp_path / "img.fits")) as hdul:
            hdul.verify("exception")
            header = hdul[0].header
            for keyword in ("BSCALE", "DATAMIN", "CHECKSUM", "CASAMBM", "CROTA2"):
                assert keyword not in header
            assert header["NAXIS1"] == xds.sizes["l"]
            assert header["MJD-OBS"] != 1.0
            assert header["ORIGIN"] == "test suite"
            assert header["INSTRUME"] == "ALMA-BAND6"
            assert "a comment" in list(header["COMMENT"])
            assert "NESTED" not in header
            # the pixels are not scaled
            np.testing.assert_array_equal(
                hdul[0].data, expected_cube(xds, range(xds.sizes["polarization"]))
            )

    def test_btype_uses_casacore_spelling(self):
        xds = make_image_xds()
        xds["SKY"].attrs["sub_type"] = "SpectralIndex"
        assert header_of(xds)["BTYPE"] == "Spectral Index"

    def test_invalid_value_names_the_card(self, tmp_path):
        xds = make_image_xds()
        reference = xds.attrs["coordinate_system_info"]["reference_direction"]
        reference["data"] = [np.nan, -0.5]
        path = tmp_path / "img.fits"
        with pytest.raises(ValueError, match="CRVAL1"):
            write(xds, path)
        assert not path.exists()

    def test_malformed_optional_metadata_is_skipped(self, recorded_log):
        xds = make_image_xds()
        xds["SKY"].attrs["pointing_center"] = [0.2, -0.5]
        xds["SKY"].attrs["telescope"] = "ALMA"
        header = header_of(xds)
        assert "OBSRA" not in header
        assert header["TELESCOP"] == "UNKNOWN"
        assert any("OBSRA" in m for m in recorded_log["warning"])

    def test_empty_image_raises(self, tmp_path):
        xds = make_image_xds().isel(frequency=slice(0, 0))
        path = tmp_path / "img.fits"
        with pytest.raises(RuntimeError, match="empty image"):
            write(xds, path)
        assert not path.exists()


class TestObservationDate:
    """N2: DATE-OBS from the time coordinate's units, format and scale."""

    def test_full_precision(self):
        header = header_of(make_image_xds())
        date = Time(header["DATE-OBS"], format="isot", scale="utc")
        assert abs(date.mjd - 54000.123456789) * 86400 < 1e-6
        np.testing.assert_allclose(
            header["MJD-OBS"], 54000.123456789, rtol=0, atol=1e-11
        )
        assert header["TIMESYS"] == "UTC"

    @pytest.mark.parametrize(
        "value, units, time_format, scale, expected",
        [
            (54000.5 * 86400, "s", "mjd", "utc", "2006-09-22T12:00:00"),
            (1158926400.0, "s", "unix", "utc", "2006-09-22T12:00:00"),
            (1158926400.0 / 86400, "d", "unix", "utc", "2006-09-22T12:00:00"),
            (54000.5, "d", "mjd", "tt", "2006-09-22T12:00:00"),
        ],
    )
    def test_units_format_and_scale(self, value, units, time_format, scale, expected):
        xds = make_image_xds()
        xds = xds.assign_coords(
            time=(
                "time",
                [value],
                {"type": "time", "units": units, "format": time_format, "scale": scale},
            )
        )
        header = header_of(xds)
        assert header["DATE-OBS"].startswith(expected)
        assert header["TIMESYS"] == scale.upper()

    def test_unset_date_is_not_written(self, tmp_path, recorded_log):
        """casacore's unset observation date (0 d, sidereal LAST) has no
        astropy time scale: the image is written without a date."""
        xds = make_image_xds()
        xds = xds.assign_coords(
            time=("time", [0.0], {"type": "time", "units": "d", "scale": "last"})
        )
        with fits.open(write(xds, tmp_path / "img.fits")) as hdul:
            assert "DATE-OBS" not in hdul[0].header
            assert "MJD-OBS" not in hdul[0].header
        assert any("DATE-OBS" in m for m in recorded_log["warning"])

    def test_placeholder_date_is_not_written(self, tmp_path):
        """MJD 0, the readers' placeholder for an unknown date, is written as
        no date (as casacore does), and reopens as the same placeholder."""
        xds = make_image_xds()
        xds = xds.assign_coords(
            time=(
                "time",
                [0.0],
                {"type": "time", "units": "d", "format": "mjd", "scale": "utc"},
            )
        )
        path = write(xds, tmp_path / "img.fits")
        with fits.open(path) as hdul:
            for keyword in ("DATE-OBS", "MJD-OBS", "TIMESYS"):
                assert keyword not in hdul[0].header
        reopened = open_image(str(path))
        assert reopened.time.values.tolist() == [0.0]
        assert "obsdate" not in reopened.SKY.attrs

    def test_datetime64_time(self):
        """A datetime64 time coordinate gives its date, in its scale."""
        xds = make_image_xds()
        xds = xds.assign_coords(
            time=(
                "time",
                np.array(["2006-09-22T12:00:00"], dtype="datetime64[ns]"),
                {"type": "time", "scale": "utc"},
            )
        )
        header = header_of(xds)
        assert header["DATE-OBS"].startswith("2006-09-22T12:00:00")
        np.testing.assert_allclose(header["MJD-OBS"], 54000.5, rtol=0, atol=1e-9)


def _counting(array: da.Array, calls: list) -> da.Array:
    """Wrap a dask array so that every computed block is recorded."""

    def record(block, block_id=None):
        calls.append((block_id, block.shape))
        return block

    return array.map_blocks(record, dtype=array.dtype, meta=np.array((), array.dtype))


class TestBlockWiseWrite:
    """F7 and decision 7: pixels are written chunk by chunk, files are never
    replaced and a failed write leaves nothing behind."""

    def test_each_chunk_is_computed_once(self, tmp_path):
        xds = make_image_xds(pols=("RR", "RL", "LR", "LL"), nchan=4, shape=(6, 5))
        sky_calls, flag_calls = [], []
        sky = _counting(
            da.from_array(xds["SKY"].values, chunks=(1, 1, 2, 3, 5)), sky_calls
        )
        flag = _counting(
            da.from_array(xds["FLAG_SKY"].values, chunks=(1, 2, 1, 6, 5)), flag_calls
        )
        lazy = xds.copy()
        lazy["SKY"] = lazy["SKY"].copy(data=sky)
        lazy["FLAG_SKY"] = lazy["FLAG_SKY"].copy(data=flag)
        with dask.config.set(scheduler="synchronous"):
            path = write(lazy, tmp_path / "img.fits")
        assert sorted(c[0] for c in sky_calls) == sorted(np.ndindex(*sky.numblocks)), (
            "every SKY chunk exactly once"
        )
        assert sorted(c[0] for c in flag_calls) == sorted(np.ndindex(*flag.numblocks))
        with fits.open(path) as hdul:
            np.testing.assert_array_equal(
                hdul[0].data, expected_cube(xds, [0, 3, 1, 2])
            )

    def test_flags_derived_from_pixels_compute_pixels_once(self, tmp_path):
        """Like the FITS reader's NaN flags: one computation serves both."""
        xds = make_image_xds(nchan=4)
        calls = []
        sky = _counting(da.from_array(xds["SKY"].values, chunks=(1, 2, 4, 6, 5)), calls)
        lazy = xds.copy()
        lazy["SKY"] = lazy["SKY"].copy(data=sky)
        lazy["FLAG_SKY"] = lazy["FLAG_SKY"].copy(data=da.isnan(sky))
        with dask.config.set(scheduler="synchronous"):
            write(lazy, tmp_path / "img.fits")
        assert len(calls) == sky.npartitions

    def test_cube_is_not_loaded_at_once(self, tmp_path):
        xds = make_image_xds(nchan=6)
        calls = []
        chunks = (1, 1, 4, 6, 5)
        sky = _counting(da.from_array(xds["SKY"].values, chunks=chunks), calls)
        lazy = xds.copy()
        lazy["SKY"] = lazy["SKY"].copy(data=sky)
        lazy["FLAG_SKY"] = lazy["FLAG_SKY"].chunk(
            dict(zip(lazy.FLAG_SKY.dims, chunks, strict=False))
        )
        with dask.config.set(scheduler="synchronous"):
            write(lazy, tmp_path / "img.fits")
        assert {shape for _, shape in calls} == {chunks}

    def test_header_is_validated_before_any_pixel_is_computed(self, tmp_path):
        xds = make_image_xds(pols=("I", "Q", "V"))
        calls = []
        lazy = xds.copy()
        lazy["SKY"] = lazy["SKY"].copy(
            data=_counting(da.from_array(xds["SKY"].values, chunks=1), calls)
        )
        path = tmp_path / "img.fits"
        with dask.config.set(scheduler="synchronous"):
            with pytest.raises(ValueError):
                write(lazy, path)
        assert calls == []
        assert not path.exists()

    def test_prepared_header_is_used(self, tmp_path):
        """A caller that validated the header passes it on."""
        image = single_image(make_image_xds())
        prepared = _fits_image_header(image)
        prepared[0]["ORIGIN"] = "prepared"
        path = tmp_path / "img.fits"
        _xds_to_fits_image(image, str(path), prepared=prepared)
        assert fits.getheader(path)["ORIGIN"] == "prepared"

    def test_existing_file_is_not_replaced(self, tmp_path):
        path = tmp_path / "img.fits"
        path.write_bytes(b"original")
        with pytest.raises(FileExistsError):
            write(make_image_xds(), path)
        assert path.read_bytes() == b"original"

    def test_failed_write_leaves_no_file(self, tmp_path):
        xds = make_image_xds(nchan=4)

        def fail_on_third_channel(block, block_id=None):
            if block_id[1] == 2:
                raise RuntimeError("read error")
            return block

        lazy = xds.copy()
        lazy["SKY"] = lazy["SKY"].copy(
            data=da.from_array(xds["SKY"].values, chunks=(1, 1, 4, 6, 5)).map_blocks(
                fail_on_third_channel, dtype=np.float32, meta=np.array((), np.float32)
            )
        )
        path = tmp_path / "img.fits"
        with dask.config.set(scheduler="synchronous"):
            with pytest.raises(RuntimeError, match="read error"):
                write(lazy, path)
        assert not path.exists()

    def test_numpy_and_dask_inputs_give_identical_files(self, tmp_path):
        xds = make_image_xds(pols=("XX", "XY", "YX", "YY"))
        numpy_path = write(xds, tmp_path / "numpy.fits")
        with dask.config.set(scheduler="synchronous"):
            dask_path = write(
                xds.chunk({"frequency": 1, "l": 4}), tmp_path / "dask.fits"
            )
        with open(numpy_path, "rb") as a, open(dask_path, "rb") as b:
            assert a.read() == b.read()


class TestWithoutCasacore:
    """X4: FITS images are written and read with astropy only."""

    def test_fits_write_and_read_without_casacore(self, tmp_path):
        dataset = tmp_path / "xds.pickle"
        with open(dataset, "wb") as f:
            pickle.dump(make_image_xds(pols=("RR", "RL", "LR", "LL")), f)
        script = textwrap.dedent(
            """
            import pickle
            import sys

            # block casacore and casatools before anything imports them
            for name in ("casacore", "casatools", "casaconfig"):
                sys.modules[name] = None

            import dask

            dask.config.set(scheduler="synchronous")
            from xradio.image import open_image, write_image

            with open(sys.argv[1], "rb") as f:
                xds = pickle.load(f)
            write_image(xds, sys.argv[2], out_format="fits")
            rt = open_image(sys.argv[2])
            assert rt["SKY"].shape == xds["SKY"].shape, rt["SKY"].shape
            loaded = [
                name
                for name, module in sys.modules.items()
                if name.split(".")[0] in ("casacore", "casatools")
                and module is not None
            ]
            assert not loaded, loaded
            print("OK")
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", script, str(dataset), str(tmp_path / "img.fits")],
            capture_output=True,
            text=True,
            cwd=tmp_path,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            timeout=300,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert result.stdout.strip().endswith("OK")
