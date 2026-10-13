"""Unit tests for the CASA image writer
(``xradio.image._util._casacore.xds_to_casacore``).

The images are small synthetic datasets, written the way the multi-image
driver hands them over (one image per dataset, with optional ``MASK_0`` flag
and ``BEAM_FIT_PARAMS`` variables), and checked with casacore directly.

* ``TestDirectionSystem``  -> direction frames, reference pixels, l/m axes
* ``TestSpectralAxis``     -> spectral frames, doppler types, frequency axes
* ``TestObservationDate``  -> time units, formats and scales
* ``TestImageInfo``        -> image types and beams
* ``TestPixelsAndMasks``   -> pixel types, byte order, masks, single pass
* ``TestLinearAxes``       -> u/v axes of aperture images
* ``TestWriteImage``       -> write_image end to end: every written image reopens

No test data is downloaded and no dask cluster is used.
"""

import os
import shutil
from glob import glob

import dask
import dask.array as da
import numpy as np
import pytest
import xarray as xr
from astropy import units as u
from astropy.time import Time

from xradio._utils._casacore.tables import open_table_ro
from xradio._utils.dict_helpers import (
    make_direction_location_dict,
    make_quantity,
    make_skycoord_dict,
)
from xradio.image import (
    load_image,
    make_empty_aperture_image,
    make_empty_lmuv_image,
    make_empty_sky_image,
    open_image,
    write_image,
)
from xradio.image._util import _blocks, conventions
from xradio.image._util._casacore.common import _open_image_ro as open_image_ro
from xradio.image._util._casacore.xds_to_casacore import (
    _block_bounds,
    _write_casa_data,
)
from xradio.image._util.casacore import _xds_to_casa_image

_C = 299792458.0
_CELL = np.radians(1.0 / 3600.0)
_DIMS = ("time", "frequency", "polarization", "l", "m")


def _observer(frame: str) -> str:
    try:
        return conventions.spectral_frame_to_observer(frame)
    except ValueError:
        return frame.lower()


def make_sky_xds(
    nchan: int = 3,
    pols=("I", "Q"),
    nl: int = 6,
    nm: int = 5,
    frame: str = "fk5",
    equinox: str | None = "j2000.0",
    spectral_frame: str = "LSRK",
    dtype=np.float32,
    frequencies=None,
    l_values=None,
    m_values=None,
    time_value: float = 59000.5,
    time_attrs: dict | None = None,
    seed: int = 0,
) -> xr.Dataset:
    """Build a one image dataset as the CASA multi-image driver hands it to
    the writer (a SKY variable and the shared coordinates)."""
    frequencies = (
        1.4e9 + 1e6 * np.arange(nchan)
        if frequencies is None
        else np.asarray(frequencies, dtype=float)
    )
    l_values = (
        (np.arange(nl) - nl // 2) * -_CELL
        if l_values is None
        else np.asarray(l_values, dtype=float)
    )
    m_values = (
        (np.arange(nm) - nm // 2) * _CELL
        if m_values is None
        else np.asarray(m_values, dtype=float)
    )
    rest = 1.42e9
    shape = (1, frequencies.size, len(pols), l_values.size, m_values.size)
    rng = np.random.default_rng(seed)
    if np.dtype(dtype).kind == "c":
        sky = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    else:
        sky = rng.normal(size=shape)
    sky = sky.astype(dtype)
    reference_direction = make_skycoord_dict([1.0, -0.5], "rad", frame)
    if equinox is not None:
        reference_direction["attrs"]["equinox"] = equinox
    coords = {
        "time": (
            "time",
            [time_value],
            time_attrs
            or {"type": "time", "units": "d", "scale": "utc", "format": "mjd"},
        ),
        "frequency": (
            "frequency",
            frequencies,
            {
                "type": "spectral_coord",
                "units": "Hz",
                "frame": spectral_frame,
                "rest_frequency": make_quantity(rest, "Hz"),
                "reference_frequency": {
                    "data": float(frequencies[frequencies.size // 2]),
                    "dims": [],
                    "attrs": {
                        "type": "spectral_coord",
                        "units": "Hz",
                        "observer": _observer(spectral_frame),
                    },
                },
                "wave_units": "mm",
            },
        ),
        "velocity": (
            "frequency",
            (1 - frequencies / rest) * _C,
            {"type": "doppler", "units": "m/s", "doppler_type": "radio"},
        ),
        "polarization": ("polarization", list(pols)),
        "l": ("l", l_values),
        "m": ("m", m_values),
        "beam_params_label": ("beam_params_label", ["major", "minor", "pa"]),
    }
    xds = xr.Dataset(
        {
            "SKY": (
                _DIMS,
                sky,
                {
                    "type": "sky",
                    "units": "Jy/beam",
                    "sub_type": "Intensity",
                    "object_name": "test source",
                },
            )
        },
        coords=coords,
    )
    xds.attrs = {
        "type": "image_dataset",
        "data_groups": {"base": {"sky": "SKY"}},
        "coordinate_system_info": {
            "reference_direction": reference_direction,
            "native_pole_direction": make_direction_location_dict(
                [np.pi, 0.0], "rad", "native_projection"
            ),
            "pixel_coordinate_transformation_matrix": [[1.0, 0.0], [0.0, 1.0]],
            "projection": "SIN",
            "projection_parameters": [0.0, 0.0],
        },
    }
    return xds


def add_flag(xds: xr.Dataset, flag: np.ndarray) -> xr.Dataset:
    """Attach a flag the way the driver does (MASK_0, named by SKY's flag)."""
    xds["MASK_0"] = (_DIMS, flag, {"type": "flag"})
    xds["SKY"].attrs["flag"] = "MASK_0"
    return xds


def add_beams(xds: xr.Dataset, beams: np.ndarray, labels=("major", "minor", "pa")):
    """Attach per plane beams (time, frequency, polarization, 3) in radians."""
    xds = xds.assign_coords(beam_params_label=list(labels))
    xds["BEAM_FIT_PARAMS"] = (
        ("time", "frequency", "polarization", "beam_params_label"),
        beams,
        {"type": "beam_fit_params_sky", "units": "rad"},
    )
    return xds


def coords_of(path) -> dict:
    with open_table_ro(str(path)) as tb:
        return tb.getkeyword("coords")


def keywords_of(path) -> dict:
    with open_table_ro(str(path)) as tb:
        return tb.getkeywords()


def imageinfo_of(path) -> dict:
    with open_image_ro(str(path)) as im:
        return im.imageinfo()


def pixels_of(path) -> np.ndarray:
    """Image pixels in the dataset's (frequency, polarization, l, m) order."""
    with open_image_ro(str(path)) as im:
        return np.transpose(im.getdata(), (0, 1, 3, 2))


def mask_table_of(path, name) -> np.ndarray:
    """Mask table values (True means good) in (frequency, polarization, l, m)."""
    with open_table_ro(os.path.join(str(path), name)) as tb:
        return np.transpose(tb.getcell(tb.colnames()[0], 0), (0, 1, 3, 2))


def count_chunks(xda: xr.DataArray, chunks) -> tuple[xr.DataArray, dict]:
    """Return a dask backed copy of xda that counts its chunk computations."""
    counter = {"n": 0}

    def count(block):
        counter["n"] += 1
        return block

    values = xda.values
    arr = da.from_array(values, chunks=chunks)
    counted = arr.map_blocks(
        count,
        dtype=values.dtype,
        meta=np.empty((0,) * values.ndim, dtype=values.dtype),
    )
    return xr.DataArray(counted, dims=xda.dims, attrs=xda.attrs), {
        "counter": counter,
        "nchunks": arr.npartitions,
    }


class TestDirectionSystem:
    """The direction system comes from the reference direction's frame."""

    @pytest.mark.parametrize(
        "frame, equinox, system, axes",
        [
            ("icrs", None, "ICRS", ["Right Ascension", "Declination"]),
            # an equinox on an icrs frame (older factories) is ignored
            ("icrs", "j2000.0", "ICRS", ["Right Ascension", "Declination"]),
            ("fk5", "j2000.0", "J2000", ["Right Ascension", "Declination"]),
            ("fk5", None, "J2000", ["Right Ascension", "Declination"]),
            ("FK5", "J2000", "J2000", ["Right Ascension", "Declination"]),
            ("fk4", None, "B1950", ["Right Ascension", "Declination"]),
            ("fk4", "b1950.0", "B1950", ["Right Ascension", "Declination"]),
            ("galactic", None, "GALACTIC", ["Longitude", "Latitude"]),
        ],
    )
    def test_frame_to_casacore_system(self, tmp_path, frame, equinox, system, axes):
        path = tmp_path / "dir.im"
        xds = make_sky_xds(frame=frame, equinox=equinox)
        _xds_to_casa_image(xds, str(path))
        direction = coords_of(path)["direction0"]
        assert direction["system"] == system
        assert direction["conversionSystem"] == system
        assert list(direction["axes"]) == axes
        np.testing.assert_allclose(direction["crval"], [1.0, -0.5])
        with open_image_ro(str(path)) as im:
            # casacore accepts the coordinate system
            assert im.coordinates().dict()["direction0"]["system"] == system
            np.testing.assert_array_equal(
                np.transpose(im.getdata(), (0, 1, 3, 2)), xds.SKY.values[0]
            )

    @pytest.mark.parametrize(
        "frame, equinox",
        [
            ("fk5", "b1950.0"),
            ("fk4", "j2000.0"),
            ("altaz", None),
            ("supergalactic", None),
        ],
    )
    def test_unsupported_frame_raises_before_writing(self, tmp_path, frame, equinox):
        path = tmp_path / "bad_dir.im"
        xds = make_sky_xds(frame=frame, equinox=equinox)
        with pytest.raises(ValueError, match="direction|equinox"):
            _xds_to_casa_image(xds, str(path))
        assert not path.exists()

    @pytest.mark.parametrize(
        "direction_reference, system, equinox",
        [
            ("icrs", "ICRS", None),
            ("galactic", "GALACTIC", None),
            ("fK5", "J2000", "j2000.0"),
            ("fk4", "B1950", "b1950.0"),
        ],
    )
    def test_factory_frames(self, tmp_path, direction_reference, system, equinox):
        """C0: make_empty_sky_image frames are not all written as J2000, and
        only fk5 and fk4 reference directions get an equinox."""
        path = tmp_path / "factory_dir.im"
        xds = make_empty_sky_image(
            phase_center=[1.0, -0.5],
            image_size=[6, 5],
            cell_size=[_CELL, _CELL],
            frequency_coords=[1.4e9, 1.401e9],
            pol_coords=["I"],
            time_coords=[59000.5],
            direction_reference=direction_reference,
        )
        xds["SKY"] = (
            _DIMS,
            np.ones((1, 2, 1, 6, 5), dtype=np.float32),
            {"type": "sky", "units": "Jy/beam"},
        )
        reference = xds.attrs["coordinate_system_info"]["reference_direction"]
        assert reference["attrs"].get("equinox") == equinox
        _xds_to_casa_image(xds, str(path))
        direction = coords_of(path)["direction0"]
        assert direction["system"] == system
        np.testing.assert_allclose(direction["crval"], [1.0, -0.5])
        np.testing.assert_allclose(direction["crpix"], [3.0, 2.0])

    def test_frame_without_equinox_does_not_need_it(self, tmp_path):
        """ICRS images as read from CASA carry no equinox (C0)."""
        path = tmp_path / "icrs.im"
        xds = make_sky_xds(frame="icrs", equinox=None)
        _xds_to_casa_image(xds, str(path))
        rt = open_image(str(path))
        assert (
            rt.attrs["coordinate_system_info"]["reference_direction"]["attrs"]["frame"]
            == "icrs"
        )
        np.testing.assert_array_equal(rt.SKY.values, xds.SKY.values)

    def test_reference_pixel_outside_grid(self, tmp_path):
        """A cutout away from the reference direction keeps its position (F5)."""
        path = tmp_path / "cutout.im"
        l_values = (np.arange(6) + 5) * -_CELL  # reference pixel at -5
        m_values = (np.arange(5) - 12) * _CELL  # reference pixel at 12
        xds = make_sky_xds(l_values=l_values, m_values=m_values)
        _xds_to_casa_image(xds, str(path))
        direction = coords_of(path)["direction0"]
        np.testing.assert_allclose(direction["crpix"], [-5.0, 12.0], atol=1e-9)
        np.testing.assert_allclose(direction["cdelt"], [-_CELL, _CELL], rtol=1e-12)

    def test_single_pixel_axis(self, tmp_path):
        """A single pixel axis takes the other axis' pixel size and keeps the
        pixel at its l value."""
        path = tmp_path / "single_l.im"
        l0 = 3 * -_CELL
        xds = make_sky_xds(l_values=[l0])
        _xds_to_casa_image(xds, str(path))
        direction = coords_of(path)["direction0"]
        np.testing.assert_allclose(direction["cdelt"], [-_CELL, _CELL], rtol=1e-12)
        # the world offset of pixel 0 is l0
        np.testing.assert_allclose(
            (0 - direction["crpix"][0]) * direction["cdelt"][0], l0, rtol=1e-12
        )
        np.testing.assert_allclose(direction["crpix"][1], 2.0)

    def test_single_pixel_axis_cdelt_attr(self, tmp_path):
        """A cdelt attribute sets the increment of a single pixel axis."""
        path = tmp_path / "single_l_cdelt.im"
        xds = make_sky_xds(l_values=[0.0])
        xds["l"].attrs["cdelt"] = make_quantity(-2.0, "arcsec")
        _xds_to_casa_image(xds, str(path))
        direction = coords_of(path)["direction0"]
        np.testing.assert_allclose(direction["cdelt"], [-2 * _CELL, _CELL], rtol=1e-12)
        np.testing.assert_allclose(direction["crpix"], [0.0, 2.0])

    def test_one_by_one_image_raises(self, tmp_path):
        path = tmp_path / "one_pixel.im"
        xds = make_sky_xds(l_values=[0.0], m_values=[0.0])
        with pytest.raises(ValueError, match="1 x 1"):
            _xds_to_casa_image(xds, str(path))
        assert not path.exists()

    def test_non_uniform_l_raises(self, tmp_path):
        path = tmp_path / "bad_l.im"
        l_values = np.array([0, -1, -2, -4, -5, -6]) * _CELL
        xds = make_sky_xds(l_values=l_values)
        with pytest.raises(ValueError, match="not uniformly spaced"):
            _xds_to_casa_image(xds, str(path))
        assert not path.exists()

    def test_missing_coordinate_system_info_raises(self, tmp_path):
        path = tmp_path / "no_csys.im"
        xds = make_sky_xds()
        del xds.attrs["coordinate_system_info"]
        with pytest.raises(ValueError, match="coordinate_system_info"):
            _xds_to_casa_image(xds, str(path))
        assert not path.exists()


class TestSpectralAxis:
    """Spectral frames and doppler types are translated to casacore."""

    @pytest.mark.parametrize(
        "frame, system",
        [
            ("LSRK", "LSRK"),
            ("LSRD", "LSRD"),
            ("BARY", "BARY"),
            ("TOPO", "TOPO"),
            ("GEO", "GEO"),
            ("REST", "REST"),
            ("GALACTO", "GALACTO"),
            ("LGROUP", "LGROUP"),
            ("CMB", "CMB"),
            # astropy and FITS names
            ("gcrs", "GEO"),
            ("GCRS", "GEO"),
            ("BARYCENT", "BARY"),
            ("lsrk", "LSRK"),
        ],
    )
    def test_spectral_frame(self, tmp_path, frame, system):
        path = tmp_path / "spec.im"
        xds = make_sky_xds(spectral_frame=frame)
        _xds_to_casa_image(xds, str(path))
        assert coords_of(path)["spectral2"]["system"] == system
        with open_image_ro(str(path)) as im:
            assert im.coordinates().dict()["spectral2"]["system"] == system

    @pytest.mark.parametrize("frame", ["icrs", "ICRS", "hcrs", "lsr", "LSR"])
    def test_frames_without_casacore_equivalent_raise(self, tmp_path, frame):
        path = tmp_path / "bad_spec.im"
        xds = make_sky_xds(spectral_frame=frame)
        with pytest.raises(ValueError, match="no casacore equivalent"):
            _xds_to_casa_image(xds, str(path))
        assert not path.exists()

    def test_observer_when_frame_is_missing(self, tmp_path):
        """Datasets without a frame attribute fall back to the observer."""
        path = tmp_path / "observer.im"
        xds = make_sky_xds()
        del xds.frequency.attrs["frame"]
        xds.frequency.attrs["reference_frequency"]["attrs"]["observer"] = "BARY"
        _xds_to_casa_image(xds, str(path))
        assert coords_of(path)["spectral2"]["system"] == "BARY"

    def test_frame_takes_precedence_over_observer(self, tmp_path):
        path = tmp_path / "frame_wins.im"
        xds = make_sky_xds(spectral_frame="TOPO")
        xds.frequency.attrs["reference_frequency"]["attrs"]["observer"] = "lsrk"
        _xds_to_casa_image(xds, str(path))
        assert coords_of(path)["spectral2"]["system"] == "TOPO"

    @pytest.mark.parametrize(
        "doppler_type, vel_type",
        [
            ("radio", 0),
            ("z", 1),
            ("optical", 1),
            ("OPTICAL", 1),
            ("ratio", 2),
            ("beta", 3),
            ("true", 3),
            ("relativistic", 3),
            ("gamma", 4),
        ],
    )
    def test_doppler_type(self, tmp_path, doppler_type, vel_type):
        path = tmp_path / "doppler.im"
        xds = make_sky_xds()
        xds.velocity.attrs["doppler_type"] = doppler_type
        _xds_to_casa_image(xds, str(path))
        assert coords_of(path)["spectral2"]["velType"] == vel_type
        with open_image_ro(str(path)) as im:
            assert im.coordinates().dict()["spectral2"]["velType"] == vel_type

    def test_unknown_doppler_type_raises_before_writing(self, tmp_path):
        path = tmp_path / "bad_doppler.im"
        xds = make_sky_xds()
        xds.velocity.attrs["doppler_type"] = "sideways"
        with pytest.raises(ValueError, match="doppler_type"):
            _xds_to_casa_image(xds, str(path))
        assert not path.exists()

    def test_missing_velocity_and_wave_units(self, tmp_path):
        """Optional velocity information is derived (R13)."""
        path = tmp_path / "no_velocity.im"
        xds = make_sky_xds().drop_vars("velocity")
        del xds.frequency.attrs["wave_units"]
        _xds_to_casa_image(xds, str(path))
        spectral = coords_of(path)["spectral2"]
        assert spectral["velType"] == 0
        assert spectral["velUnit"] == "m/s"
        assert spectral["waveUnit"] == "mm"

    def test_non_uniform_frequency_raises_before_writing(self, tmp_path):
        """F4: a frequency axis that is not linear cannot be described."""
        path = tmp_path / "non_uniform.im"
        xds = make_sky_xds(frequencies=[1.000e9, 1.001e9, 1.005e9])
        with pytest.raises(ValueError, match="not uniformly spaced"):
            _xds_to_casa_image(xds, str(path))
        assert not path.exists()

    def test_channel_subset_raises(self, tmp_path):
        path = tmp_path / "subset.im"
        xds = make_sky_xds(nchan=8).isel(frequency=[0, 2, 5, 6])
        with pytest.raises(ValueError, match="not uniformly spaced"):
            _xds_to_casa_image(xds, str(path))
        assert not path.exists()

    def test_nearly_uniform_frequency_is_written(self, tmp_path):
        """Round off far below the tolerance (optical velocity axes) is fine."""
        path = tmp_path / "nearly_uniform.im"
        frequencies = 1.4e9 + 1e6 * np.arange(5)
        frequencies[2] += 1e6 * 5e-5
        xds = make_sky_xds(frequencies=frequencies)
        _xds_to_casa_image(xds, str(path))
        wcs = coords_of(path)["spectral2"]["wcs"]
        np.testing.assert_allclose(wcs["cdelt"], 1e6)
        np.testing.assert_allclose(wcs["crval"], frequencies[2])
        np.testing.assert_allclose(wcs["crpix"], 2.0, atol=1e-3)

    def test_frequency_units_converted_to_hz(self, tmp_path):
        path = tmp_path / "ghz.im"
        xds = make_sky_xds()
        hz = xds.frequency.values
        xds = xds.assign_coords(
            frequency=("frequency", hz / 1e9, {**xds.frequency.attrs, "units": "GHz"})
        )
        reference = xds.frequency.attrs["reference_frequency"]
        reference["data"] = reference["data"] / 1e9
        reference["attrs"]["units"] = "GHz"
        _xds_to_casa_image(xds, str(path))
        spectral = coords_of(path)["spectral2"]
        assert spectral["unit"] == "Hz"
        np.testing.assert_allclose(spectral["wcs"]["crval"], hz[1])
        np.testing.assert_allclose(spectral["wcs"]["cdelt"], 1e6)
        np.testing.assert_allclose(spectral["wcs"]["crpix"], 1.0)
        np.testing.assert_allclose(spectral["restfreq"], 1.42e9)

    @pytest.mark.parametrize(
        "channel_width, expected",
        [
            (make_quantity(2.5e6, "Hz"), 2.5e6),
            (make_quantity(250.0, "kHz"), 2.5e5),
            ({"data": 3e6, "dims": [], "attrs": {"units": ["Hz"]}}, 3e6),
            (None, conventions.SINGLE_CHANNEL_WIDTH_FALLBACK_HZ),
        ],
    )
    def test_single_channel_increment(self, tmp_path, channel_width, expected):
        path = tmp_path / "single_channel.im"
        xds = make_sky_xds(nchan=1)
        if channel_width is not None:
            xds.frequency.attrs["channel_width"] = channel_width
        _xds_to_casa_image(xds, str(path))
        wcs = coords_of(path)["spectral2"]["wcs"]
        np.testing.assert_allclose(wcs["cdelt"], expected)
        np.testing.assert_allclose(wcs["crval"], 1.4e9)
        np.testing.assert_allclose(wcs["crpix"], 0.0)

    def test_unknown_polarization_raises_before_writing(self, tmp_path):
        path = tmp_path / "bad_pol.im"
        xds = make_sky_xds(pols=("I", "Bogus"))
        with pytest.raises(ValueError, match="Stokes"):
            _xds_to_casa_image(xds, str(path))
        assert not path.exists()

    def test_correlation_order_is_kept(self, tmp_path):
        """CASA images store the canonical (Jones) order as it is."""
        path = tmp_path / "corr.im"
        xds = make_sky_xds(pols=("RR", "RL", "LR", "LL"))
        _xds_to_casa_image(xds, str(path))
        assert list(coords_of(path)["stokes1"]["stokes"]) == ["RR", "RL", "LR", "LL"]
        np.testing.assert_array_equal(pixels_of(path), xds.SKY.values[0])


class TestObservationDate:
    """N1: obsdate honours the time units, format and scale."""

    @staticmethod
    def _obsdate(tmp_path, time_value, time_attrs):
        path = tmp_path / "obsdate.im"
        xds = make_sky_xds(time_value=time_value, time_attrs=time_attrs)
        _xds_to_casa_image(xds, str(path))
        return coords_of(path)["obsdate"]

    def test_mjd_days_unchanged(self, tmp_path):
        obsdate = self._obsdate(
            tmp_path,
            51544.00000000116,
            {"type": "time", "units": "d", "scale": "utc", "format": "mjd"},
        )
        assert obsdate["refer"] == "UTC"
        assert obsdate["m0"]["unit"] == "d"
        assert obsdate["m0"]["value"] == 51544.00000000116

    @pytest.mark.parametrize(
        "time_value, units, time_format",
        [
            # 2000-01-01T12:00:00 UTC in each representation
            (946728000.0, "s", "unix"),
            (946728000.0 / 86400.0, "d", "unix"),
            (51544.5 * 86400.0, "s", "mjd"),
            (Time("2000-01-01T12:00:00", scale="utc").gps, "s", "gps"),
            (Time("2000-01-01T12:00:00", scale="utc").cxcsec, "s", "cxcsec"),
        ],
    )
    def test_units_and_formats(self, tmp_path, time_value, units, time_format):
        obsdate = self._obsdate(
            tmp_path,
            time_value,
            {"type": "time", "units": units, "scale": "utc", "format": time_format},
        )
        assert obsdate["refer"] == "UTC"
        assert obsdate["m0"]["unit"] == "d"
        np.testing.assert_allclose(obsdate["m0"]["value"], 51544.5, atol=1e-8)

    def test_time_scale(self, tmp_path):
        path = tmp_path / "tt.im"
        xds = make_sky_xds(
            time_value=51544.5,
            time_attrs={"type": "time", "units": "d", "scale": "tt", "format": "mjd"},
        )
        _xds_to_casa_image(xds, str(path))
        with open_image_ro(str(path)) as im:
            obsdate = im.coordinates().dict()["obsdate"]
        # casacore calls TT by its other name, TDT
        assert obsdate["refer"] in ("TT", "TDT")
        np.testing.assert_allclose(obsdate["m0"]["value"], 51544.5)

    def test_unset_sidereal_date_is_kept(self, tmp_path):
        """casacore's unset observation date (0 d LAST) is written as it is."""
        obsdate = self._obsdate(
            tmp_path, 0.0, {"type": "time", "units": "d", "scale": "last", "format": ""}
        )
        assert obsdate["refer"] == "LAST"
        assert obsdate["m0"]["value"] == 0.0

    def test_unknown_scale_raises_before_writing(self, tmp_path):
        path = tmp_path / "bad_scale.im"
        xds = make_sky_xds(
            time_attrs={"type": "time", "units": "d", "scale": "xyz", "format": "mjd"}
        )
        with pytest.raises(ValueError, match="time scale"):
            _xds_to_casa_image(xds, str(path))
        assert not path.exists()


class TestImageInfo:
    """Image types (R4) and beams (C10, R14)."""

    @pytest.mark.parametrize(
        "sub_type, imagetype",
        [
            ("SpectralIndex", "Spectral Index"),
            ("ColumnDensity", "Column Density"),
            ("RotationMeasure", "Rotation Measure"),
            ("Intensity", "Intensity"),
            (None, "Intensity"),
        ],
    )
    def test_sub_type(self, tmp_path, sub_type, imagetype):
        path = tmp_path / "subtype.im"
        xds = make_sky_xds()
        if sub_type is None:
            del xds.SKY.attrs["sub_type"]
        else:
            xds.SKY.attrs["sub_type"] = sub_type
        _xds_to_casa_image(xds, str(path))
        assert imageinfo_of(path)["imagetype"] == imagetype

    def test_single_beam_written_as_restoring_beam(self, tmp_path):
        path = tmp_path / "single_beam.im"
        xds = make_sky_xds()
        beam = np.array([2e-5, 1e-5, 0.3])
        beams = np.broadcast_to(beam, (1, 3, 2, 3)).copy()
        _xds_to_casa_image(add_beams(xds, beams), str(path))
        info = imageinfo_of(path)
        assert "perplanebeams" not in info
        restoring = info["restoringbeam"]
        got = [
            (restoring[k]["value"] * u.Unit(restoring[k]["unit"])).to_value(u.rad)
            for k in ("major", "minor", "positionangle")
        ]
        np.testing.assert_allclose(got, beam)

    def test_float32_equal_beams_are_one_beam(self, tmp_path):
        path = tmp_path / "f32_beams.im"
        xds = make_sky_xds()
        beam = np.array([2e-5, 1e-5, 0.3])
        beams = np.broadcast_to(beam, (1, 3, 2, 3)).copy()
        beams[0, 1, 1] = beams[0, 1, 1].astype(np.float32)
        _xds_to_casa_image(add_beams(xds, beams), str(path))
        assert "restoringbeam" in imageinfo_of(path)

    def test_distinct_beams_written_per_plane(self, tmp_path):
        """Beams differing by much less than 1e-8 rad are still per plane
        (no absolute tolerance on the beam axes, see F9)."""
        path = tmp_path / "per_plane.im"
        xds = make_sky_xds()
        beams = np.broadcast_to(np.array([2e-8, 1e-8, 0.3]), (1, 3, 2, 3)).copy()
        beams[0, 2, 1, 0] *= 1.01
        _xds_to_casa_image(add_beams(xds, beams), str(path))
        info = imageinfo_of(path)
        assert "restoringbeam" not in info
        pp = info["perplanebeams"]
        assert pp["nChannels"] == 3 and pp["nStokes"] == 2
        for chan in range(3):
            for pol in range(2):
                major = pp[f"*{3 * pol + chan}"]["major"]
                np.testing.assert_allclose(
                    (major["value"] * u.Unit(major["unit"])).to_value(u.rad),
                    beams[0, chan, pol, 0],
                )

    def test_beam_params_by_label(self, tmp_path):
        """R14(b): the beam parameters are picked by label, not position."""
        path = tmp_path / "labels.im"
        xds = make_sky_xds()
        beams = np.broadcast_to(np.array([1e-5, 2e-5, 0.3]), (1, 3, 2, 3)).copy()
        _xds_to_casa_image(
            add_beams(xds, beams, labels=("minor", "major", "pa")), str(path)
        )
        restoring = imageinfo_of(path)["restoringbeam"]
        assert restoring["major"]["value"] == pytest.approx(2e-5)
        assert restoring["minor"]["value"] == pytest.approx(1e-5)
        assert restoring["positionangle"]["value"] == pytest.approx(0.3)

    def test_beam_missing_label_raises_before_writing(self, tmp_path):
        path = tmp_path / "two_labels.im"
        xds = make_sky_xds().drop_vars("beam_params_label")
        xds = xds.assign_coords(beam_params_label=["major", "minor"])
        xds["BEAM_FIT_PARAMS"] = (
            ("time", "frequency", "polarization", "beam_params_label"),
            np.ones((1, 3, 2, 2)) * 1e-5,
            {"type": "beam_fit_params_sky", "units": "rad"},
        )
        with pytest.raises(ValueError, match="beam_params_label"):
            _xds_to_casa_image(xds, str(path))
        assert not path.exists()

    def test_beam_round_trip(self, tmp_path):
        path = tmp_path / "beam_rt.im"
        xds = make_sky_xds()
        beams = np.broadcast_to(np.array([2e-5, 1e-5, 0.3]), (1, 3, 2, 3)).copy()
        _xds_to_casa_image(add_beams(xds, beams), str(path))
        rt = open_image(str(path))
        np.testing.assert_allclose(rt.BEAM_FIT_PARAMS_SKY.values, beams)


class TestPixelsAndMasks:
    """Pixel types, byte order, masks and the single pixel pass."""

    def test_complex64_written_as_complex(self, tmp_path):
        """C9: single precision complex stays single precision."""
        path = tmp_path / "complex64.im"
        xds = make_sky_xds(dtype=np.complex64)
        _xds_to_casa_image(xds, str(path))
        with open_image_ro(str(path)) as im:
            assert im.datatype() == "Complex"
            data = np.transpose(im.getdata(), (0, 1, 3, 2))
        np.testing.assert_array_equal(data, xds.SKY.values[0])

    @pytest.mark.parametrize(
        "dtype, casa_type",
        [(np.float32, "float"), (np.float64, "double"), (np.complex128, "DComplex")],
    )
    def test_pixel_types(self, tmp_path, dtype, casa_type):
        path = tmp_path / "ptype.im"
        xds = make_sky_xds(dtype=dtype)
        _xds_to_casa_image(xds, str(path))
        with open_image_ro(str(path)) as im:
            assert im.datatype() == casa_type

    def test_bool_image_written_as_float32(self, tmp_path):
        """C4: a bool mask image is written as float32 ones and zeros."""
        path = tmp_path / "bool.mask"
        xds = make_sky_xds()
        mask = xds.SKY.values > 0
        xds["SKY"] = (_DIMS, mask, {"type": "mask"})
        _xds_to_casa_image(xds, str(path))
        with open_image_ro(str(path)) as im:
            assert im.datatype() == "float"
        np.testing.assert_array_equal(pixels_of(path), mask[0].astype(np.float32))

    def test_mask_image_does_not_mask_itself(self, tmp_path):
        """C3: the variable written as the image is never one of its masks."""
        path = tmp_path / "deconvolve.mask"
        xds = make_sky_xds()
        mask = (xds.SKY.values > 0).astype(np.float32)
        xds["SKY"] = (_DIMS, mask, {"type": "mask"})
        _xds_to_casa_image(xds, str(path))
        keywords = keywords_of(path)
        assert "SKY" not in keywords.get("masks", {})
        assert not os.path.exists(os.path.join(str(path), "SKY"))
        np.testing.assert_array_equal(pixels_of(path), mask[0])
        loaded = load_image(str(path))
        image_vars = [v for v in loaded.data_vars if loaded[v].ndim == 5]
        assert image_vars, "load_image returned no image"
        np.testing.assert_array_equal(loaded[image_vars[0]].values, mask)

    def test_flag_and_unflagged_nans(self, tmp_path):
        """C1, C6: each mask entry points at its own table; the nan masks are
        correct and the nans-or-flag mask is the default mask."""
        path = tmp_path / "nans.im"
        xds = make_sky_xds()
        sky = xds.SKY.values
        sky[0, 0, 0, 0, 0] = np.nan  # not flagged
        sky[0, 2, 1, 5, 4] = np.nan  # not flagged, another chunk
        sky[0, 1, 1, 3, 3] = np.nan  # also flagged
        flag = np.zeros(sky.shape, dtype=bool)
        flag[0, 1, 1, 3, 3] = True
        flag[0, 1, 0, 2, 2] = True
        xds = add_flag(xds, flag)
        xds["SKY"] = xds.SKY.chunk({"frequency": 1, "l": 3})
        xds["MASK_0"] = xds.MASK_0.chunk({"frequency": 1, "l": 3})
        with dask.config.set(scheduler="synchronous"):
            _xds_to_casa_image(xds, str(path))
        keywords = keywords_of(path)
        masks = keywords["masks"]
        names = ["MASK_0", "mask_xds_nans", "mask_xds_nans_or_MASK_0"]
        assert sorted(masks) == sorted(names)
        for name in names:
            assert masks[name]["mask"].endswith(os.sep + name), masks[name]["mask"]
        assert keywords["Image_defaultmask"] == "mask_xds_nans_or_MASK_0"
        nan = np.isnan(sky[0])
        np.testing.assert_array_equal(mask_table_of(path, "MASK_0"), ~flag[0])
        np.testing.assert_array_equal(mask_table_of(path, "mask_xds_nans"), ~nan)
        np.testing.assert_array_equal(
            mask_table_of(path, "mask_xds_nans_or_MASK_0"), ~nan & ~flag[0]
        )
        with open_image_ro(str(path)) as im:
            casa_mask = np.transpose(im.getmask(), (0, 1, 3, 2))
        np.testing.assert_array_equal(casa_mask, nan | flag[0])

    @pytest.mark.parametrize("with_flag", [True, False])
    def test_first_nan_in_a_later_block(self, tmp_path, with_flag):
        """C6: the nan masks are filled in for the blocks written before the
        first nan (the blocks are written in one pass)."""
        path = tmp_path / "late_nan.im"
        xds = make_sky_xds()
        sky = xds.SKY.values
        sky[0, 2, 1, 4, 3] = np.nan  # in the last frequency block
        flag = np.zeros(sky.shape, dtype=bool)
        flag[0, 0, 0, 1, 1] = True  # in the first block
        flag[0, 2, 0, 5, 0] = True
        if with_flag:
            xds = add_flag(xds, flag)
            xds["MASK_0"] = xds.MASK_0.chunk({"frequency": 1})
        else:
            flag[:] = False
        xds["SKY"] = xds.SKY.chunk({"frequency": 1})
        with dask.config.set(scheduler="synchronous"):
            _xds_to_casa_image(xds, str(path))
        nan = np.isnan(sky[0])
        np.testing.assert_array_equal(mask_table_of(path, "mask_xds_nans"), ~nan)
        if with_flag:
            np.testing.assert_array_equal(mask_table_of(path, "MASK_0"), ~flag[0])
            np.testing.assert_array_equal(
                mask_table_of(path, "mask_xds_nans_or_MASK_0"), ~nan & ~flag[0]
            )
        with open_image_ro(str(path)) as im:
            casa_mask = np.transpose(im.getmask(), (0, 1, 3, 2))
        np.testing.assert_array_equal(casa_mask, nan | flag[0])

    def test_flagged_nans_need_no_nan_masks(self, tmp_path):
        path = tmp_path / "flagged_nans.im"
        xds = make_sky_xds()
        sky = xds.SKY.values
        sky[0, 1, 1, 3, 3] = np.nan
        flag = np.zeros(sky.shape, dtype=bool)
        flag[0, 1, 1, 3, 3] = True
        _xds_to_casa_image(add_flag(xds, flag), str(path))
        keywords = keywords_of(path)
        assert list(keywords["masks"]) == ["MASK_0"]
        assert keywords["Image_defaultmask"] == "MASK_0"
        assert sorted(
            d for d in os.listdir(str(path)) if os.path.isdir(os.path.join(path, d))
        ) == ["MASK_0", "logtable"]

    def test_nans_without_flag(self, tmp_path):
        path = tmp_path / "nans_no_flag.im"
        xds = make_sky_xds()
        xds.SKY.values[0, 2, 0, 1, 1] = np.nan
        _xds_to_casa_image(xds, str(path))
        keywords = keywords_of(path)
        assert list(keywords["masks"]) == ["mask_xds_nans"]
        assert keywords["Image_defaultmask"] == "mask_xds_nans"
        np.testing.assert_array_equal(
            mask_table_of(path, "mask_xds_nans"), ~np.isnan(xds.SKY.values[0])
        )

    def test_no_flag_no_nans_has_no_mask(self, tmp_path):
        path = tmp_path / "maskless.im"
        _xds_to_casa_image(make_sky_xds(), str(path))
        keywords = keywords_of(path)
        assert "masks" not in keywords
        assert "Image_defaultmask" not in keywords
        assert not os.path.exists(os.path.join(str(path), "mask_xds_nans"))

    def test_masks_survive_moving_the_image(self, tmp_path):
        """Images are written next to their target and moved into place."""
        written = tmp_path / "tmp_write" / "moved.im"
        written.parent.mkdir()
        xds = make_sky_xds()
        xds.SKY.values[0, 0, 0, 0, 0] = np.nan
        flag = np.zeros(xds.SKY.shape, dtype=bool)
        flag[0, 1, 1, 1, 1] = True
        _xds_to_casa_image(add_flag(xds, flag), str(written))
        target = tmp_path / "moved.im"
        os.replace(written, target)
        shutil.rmtree(written.parent)
        with open_image_ro(str(target)) as im:
            casa_mask = np.transpose(im.getmask(), (0, 1, 3, 2))
        expected = np.isnan(xds.SKY.values[0]) | flag[0]
        np.testing.assert_array_equal(casa_mask, expected)
        np.testing.assert_array_equal(mask_table_of(target, "MASK_0"), ~flag[0])
        assert "SKY" in open_image(str(target)).data_vars

    def test_dangling_flag_reference_raises_before_writing(self, tmp_path):
        """C8: a flag attribute naming a missing variable is a clear error."""
        path = tmp_path / "dangling.im"
        xds = make_sky_xds()
        xds.SKY.attrs["flag"] = "FLAG_SKY"
        with pytest.raises(ValueError, match="FLAG_SKY"):
            _write_casa_data(xds, str(path))
        assert not path.exists()

    @pytest.mark.parametrize("chunked", [False, True])
    def test_big_endian_pixels(self, tmp_path, chunked):
        """R2: big endian data (as read from FITS) is written correctly."""
        path = tmp_path / "big_endian.im"
        xds = make_sky_xds()
        expected = xds.SKY.values.copy()
        big = expected.astype(">f4")
        xds["SKY"] = (_DIMS, big, xds.SKY.attrs)
        if chunked:
            xds["SKY"] = xds.SKY.chunk({"frequency": 1})
        _xds_to_casa_image(xds, str(path))
        np.testing.assert_array_equal(pixels_of(path), expected[0])

    def test_non_contiguous_flag(self, tmp_path):
        path = tmp_path / "strided.im"
        xds = make_sky_xds()
        flag = np.zeros((1, 3, 2, 5, 6), dtype=bool)
        flag[0, :, :, 1, 2] = True
        xds["MASK_0"] = (_DIMS, flag.transpose(0, 1, 2, 4, 3), {"type": "flag"})
        xds["SKY"].attrs["flag"] = "MASK_0"
        _xds_to_casa_image(xds, str(path))
        np.testing.assert_array_equal(
            mask_table_of(path, "MASK_0"), ~flag.transpose(0, 1, 2, 4, 3)[0]
        )

    @pytest.mark.parametrize(
        "sky_chunks, flag_chunks",
        [
            # flag chunks twice the image chunks (zarr default chunks by dtype)
            ((1, 1, 1, 2, 5), (1, 2, 1, 4, 5)),
            # misaligned chunk grids (blocks of 6 pixels along l)
            ((1, 1, 2, 3, 5), (1, 3, 1, 2, 5)),
            # flag chunks finer than the image chunks
            ((1, 3, 2, 6, 5), (1, 1, 1, 2, 5)),
        ],
    )
    def test_each_chunk_computed_once(self, tmp_path, sky_chunks, flag_chunks):
        """C7: also when the flag's chunks do not lie within image chunks."""
        path = tmp_path / "once.im"
        xds = make_sky_xds()
        sky = xds.SKY.values
        sky[0, 0, 0, 0, 0] = np.nan
        flag = np.zeros(sky.shape, dtype=bool)
        flag[0, 2, 1, 4, 4] = True
        xds = add_flag(xds, flag)
        xds["SKY"], sky_count = count_chunks(xds.SKY, sky_chunks)
        xds["MASK_0"], flag_count = count_chunks(xds.MASK_0, flag_chunks)
        with dask.config.set(scheduler="synchronous"):
            _xds_to_casa_image(xds, str(path))
        for name, count in (("SKY", sky_count), ("MASK_0", flag_count)):
            assert count["counter"]["n"] == count["nchunks"], (
                f"{name}: {count['counter']['n']} computations for "
                f"{count['nchunks']} chunks"
            )
        np.testing.assert_array_equal(
            np.nan_to_num(pixels_of(path), nan=-99), np.nan_to_num(sky[0], nan=-99)
        )
        np.testing.assert_array_equal(mask_table_of(path, "MASK_0"), ~flag[0])
        np.testing.assert_array_equal(
            mask_table_of(path, "mask_xds_nans_or_MASK_0"),
            ~np.isnan(sky[0]) & ~flag[0],
        )

    @pytest.mark.parametrize("one_block_per_batch", [False, True])
    def test_misaligned_chunks_bound_the_block_size(
        self, tmp_path, monkeypatch, one_block_per_batch
    ):
        """A flag in one chunk and an image in many small chunks would need a
        block of the whole cube; the pass keeps the image's chunk grid instead
        (bounded memory). The flag chunk is computed once per batch of blocks:
        once when the cube fits in one batch, per image chunk when every block
        is a batch of its own."""
        if one_block_per_batch:
            monkeypatch.setattr(_blocks, "BATCH_BYTES", 1)
        path = tmp_path / "bounded.im"
        xds = make_sky_xds()
        flag = np.zeros(xds.SKY.shape, dtype=bool)
        flag[0, 2, 1, 4, 4] = True
        xds = add_flag(xds, flag)
        xds["SKY"], sky_count = count_chunks(xds.SKY, (1, 1, 1, 1, 5))
        xds["MASK_0"], flag_count = count_chunks(xds.MASK_0, (1, 3, 2, 6, 5))
        image_t = xds.SKY.isel(time=0).transpose("frequency", "polarization", "m", "l")
        flag_t = xds.MASK_0.isel(time=0).transpose(
            "frequency", "polarization", "m", "l"
        )
        assert _block_bounds(image_t, flag_t) == image_t.chunks
        with dask.config.set(scheduler="synchronous"):
            _xds_to_casa_image(xds, str(path))
        assert sky_count["counter"]["n"] == sky_count["nchunks"] == 36
        assert flag_count["counter"]["n"] == (36 if one_block_per_batch else 1)
        np.testing.assert_array_equal(mask_table_of(path, "MASK_0"), ~flag[0])

    def test_caller_dataset_unchanged(self, tmp_path):
        path = tmp_path / "unchanged.im"
        xds = add_flag(make_sky_xds(), np.zeros((1, 3, 2, 6, 5), dtype=bool))
        before = xds.copy(deep=True)
        _xds_to_casa_image(xds, str(path))
        xr.testing.assert_identical(xds, before)


class TestLinearAxes:
    """F5, R13: u/v axes of aperture images."""

    @staticmethod
    def _aperture_xds(u_values, v_values, u_attrs=None, v_attrs=None):
        xds = make_sky_xds(nl=len(u_values), nm=len(v_values))
        xds = xds.rename({"l": "u", "m": "v", "SKY": "APERTURE"})
        xds = xds.assign_coords(
            u=("u", np.asarray(u_values, dtype=float), u_attrs or {}),
            v=("v", np.asarray(v_values, dtype=float), v_attrs or {}),
        )
        xds["APERTURE"] = xds.APERTURE.astype(np.complex64)
        xds["APERTURE"].attrs = {"type": "aperture", "units": "Jy"}
        del xds.attrs["coordinate_system_info"]
        return xds

    def test_reference_pixel_outside_grid(self, tmp_path):
        path = tmp_path / "uv_cutout.im"
        step = 85.0
        xds = self._aperture_xds(
            (np.arange(6) + 4) * step,
            (np.arange(5) - 9) * -step,
            {"crval": 0.0, "cdelt": step, "units": "lambda"},
            {"crval": 0.0, "cdelt": -step, "units": "lambda"},
        )
        _xds_to_casa_image(xds, str(path))
        linear = coords_of(path)["linear0"]
        np.testing.assert_allclose(linear["crpix"], [-4.0, 9.0])
        np.testing.assert_allclose(linear["cdelt"], [step, -step])
        np.testing.assert_allclose(linear["crval"], [0.0, 0.0])

    def test_nonzero_reference_value(self, tmp_path):
        path = tmp_path / "uv_crval.im"
        xds = self._aperture_xds(
            np.arange(4) * 10.0 + 100.0,
            np.arange(3) * 10.0,
            {"crval": 120.0, "cdelt": 10.0, "units": "lambda"},
            {"crval": 0.0, "cdelt": 10.0, "units": "lambda"},
        )
        _xds_to_casa_image(xds, str(path))
        linear = coords_of(path)["linear0"]
        np.testing.assert_allclose(linear["crpix"], [2.0, 0.0])
        np.testing.assert_allclose(linear["crval"], [120.0, 0.0])

    def test_attrs_missing(self, tmp_path):
        path = tmp_path / "uv_no_attrs.im"
        xds = self._aperture_xds((np.arange(4) - 2) * -5.0, (np.arange(3) - 1) * 7.0)
        _xds_to_casa_image(xds, str(path))
        linear = coords_of(path)["linear0"]
        np.testing.assert_allclose(linear["cdelt"], [-5.0, 7.0])
        np.testing.assert_allclose(linear["crpix"], [2.0, 1.0])
        np.testing.assert_allclose(linear["crval"], [0.0, 0.0])
        assert list(linear["units"]) == ["lambda", "lambda"]

    def test_uv_image_written_as_sky(self, tmp_path):
        """The image axes follow the dims, not the variable name."""
        path = tmp_path / "visibility.im"
        xds = self._aperture_xds((np.arange(4) - 2) * -5.0, (np.arange(3) - 1) * 7.0)
        xds = xds.rename({"APERTURE": "SKY"})
        _xds_to_casa_image(xds, str(path))
        coords = coords_of(path)
        assert "linear0" in coords and "direction0" not in coords
        np.testing.assert_array_equal(pixels_of(path), xds.SKY.values[0])

    def test_image_without_spatial_dims_raises(self, tmp_path):
        path = tmp_path / "normalization.im"
        xds = make_sky_xds().isel(l=0, m=0, drop=True)
        with pytest.raises(ValueError, match="dims"):
            _write_casa_data(xds, str(path))
        assert not path.exists()

    def test_non_uniform_u_raises(self, tmp_path):
        path = tmp_path / "uv_bad.im"
        xds = self._aperture_xds([0.0, 1.0, 3.0, 4.0], [0.0, 1.0, 2.0])
        with pytest.raises(ValueError, match="not uniformly spaced"):
            _xds_to_casa_image(xds, str(path))
        assert not path.exists()

    @pytest.mark.parametrize("factory", ["aperture", "lmuv"])
    def test_factory_aperture_images(self, tmp_path, factory):
        """The factories nest u/v units in a quantity dict (R13)."""
        path = tmp_path / f"{factory}.im"
        args = dict(
            phase_center=[1.0, -0.5],
            image_size=[8, 6],
            sky_image_cell_size=[_CELL, _CELL],
            frequency_coords=[1.4e9, 1.401e9],
            pol_coords=["I"],
            time_coords=[59000.5],
        )
        if factory == "aperture":
            xds = make_empty_aperture_image(**args)
        else:
            xds = make_empty_lmuv_image(**args)
        shape = (1, 2, 1, 8, 6)
        data = (np.arange(np.prod(shape)) * (1 + 1j)).reshape(shape)
        aperture = xr.DataArray(
            data.astype(np.complex64),
            dims=("time", "frequency", "polarization", "u", "v"),
            coords={
                c: xds.coords[c]
                for c in ("time", "frequency", "polarization", "u", "v")
            },
            attrs={"type": "aperture", "units": "Jy"},
        )
        image_xds = xr.Dataset({"APERTURE": aperture}, attrs=xds.attrs)
        _xds_to_casa_image(image_xds, str(path))
        linear = coords_of(path)["linear0"]
        u_values = xds.u.values
        v_values = xds.v.values
        np.testing.assert_allclose(
            linear["cdelt"], [u_values[1] - u_values[0], v_values[1] - v_values[0]]
        )
        np.testing.assert_allclose(linear["crpix"], [4.0, 3.0])
        assert list(linear["units"]) == ["lambda", "lambda"]
        with open_image_ro(str(path)) as im:
            assert im.datatype() == "Complex"
            np.testing.assert_array_equal(
                np.transpose(im.getdata(), (0, 1, 3, 2)), data[0]
            )


class TestWriteImage:
    """write_image end to end (C3): every written image reopens."""

    def test_single_image_keeps_the_name(self, tmp_path):
        path = tmp_path / "single.im"
        xds = make_sky_xds(frame="icrs", equinox=None)
        write_image(xds, str(path), out_format="casa")
        assert coords_of(path)["direction0"]["system"] == "ICRS"
        np.testing.assert_array_equal(pixels_of(path), xds.SKY.values[0])

    def test_every_written_image_reopens(self, tmp_path):
        xds = make_sky_xds()
        sky = xds.SKY.values
        sky[0, 0, 0, 0, 0] = np.nan
        flag = np.zeros(sky.shape, dtype=bool)
        flag[0, 1, 1, 2, 2] = True
        xds["FLAG_SKY"] = (_DIMS, flag, {"type": "flag"})
        xds["SKY"].attrs["flag"] = "FLAG_SKY"
        xds["MASK"] = (_DIMS, np.nan_to_num(sky) > 0, {"type": "mask"})
        beams = np.broadcast_to(np.array([2e-5, 1e-5, 0.3]), (1, 3, 2, 3)).copy()
        xds = add_beams(xds, beams)
        xds = xds.rename({"BEAM_FIT_PARAMS": "BEAM_FIT_PARAMS_SKY"})
        xds.attrs["data_groups"] = {
            "base": {
                "sky": "SKY",
                "flag": "FLAG_SKY",
                "mask": "MASK",
                "beam_fit_params_sky": "BEAM_FIT_PARAMS_SKY",
            }
        }
        name = str(tmp_path / "multi")
        write_image(xds, name, out_format="casa")
        outputs = sorted(glob(name + ".*"))
        assert len(outputs) == 2, outputs
        for output in outputs:
            masks = keywords_of(output).get("masks", {})
            for mask in masks:
                # every registered mask has its table (no dangling 'SKY' mask)
                assert os.path.isdir(os.path.join(output, mask)), (output, mask)
            with open_image_ro(output) as im:
                im.getdata()
                im.getmask()
            assert load_image(output).data_vars
            assert open_image(output).data_vars
        sky_output = [o for o in outputs if o.endswith(".sky")][0]
        mask_output = [o for o in outputs if o.endswith(".mask")][0]
        with open_image_ro(sky_output) as im:
            casa_mask = np.transpose(im.getmask(), (0, 1, 3, 2))
            assert "restoringbeam" in im.imageinfo()
        np.testing.assert_array_equal(casa_mask, np.isnan(sky[0]) | flag[0])
        np.testing.assert_array_equal(
            pixels_of(mask_output), xds.MASK.values[0].astype(np.float32)
        )
        assert "masks" not in keywords_of(mask_output)


class TestNullAndSwappedBeams:
    """casacore cannot open images with null (zero axis) or swapped beams."""

    def test_all_zero_beams_write_no_beam(self, tmp_path):
        path = tmp_path / "no_beam.im"
        xds = add_beams(make_sky_xds(), np.zeros((1, 3, 2, 3)))
        _xds_to_casa_image(xds, str(path))
        info = imageinfo_of(path)
        assert "restoringbeam" not in info and "perplanebeams" not in info
        assert "BEAM_FIT_PARAMS_SKY" not in open_image(str(path)).data_vars

    @pytest.mark.parametrize(
        "bad, match",
        [((0.0, 0.0, 0.0), "zero axis"), ((1e-5, 2e-5, 0.3), "shorter than")],
    )
    def test_invalid_beams_raise_before_writing(self, tmp_path, bad, match):
        path = tmp_path / "bad_beam.im"
        beams = np.broadcast_to(np.array([2e-5, 1e-5, 0.3]), (1, 3, 2, 3)).copy()
        beams[0, 1, 0] = bad
        xds = add_beams(make_sky_xds(), beams)
        with pytest.raises(ValueError, match=match):
            _xds_to_casa_image(xds, str(path))
        assert not path.exists()


class TestKeywordValues:
    """miscinfo values casacore cannot store are dropped with a warning, not
    left to fail in putkeyword after the pixels are written."""

    @pytest.mark.parametrize(
        "value, ok",
        [
            (np.datetime64("2020-01-01T00:00:00"), False),
            (np.timedelta64(5, "s"), False),
            (np.array(["2020-01-01"], dtype="datetime64[D]"), False),
            (np.array([1, 2], dtype="timedelta64[s]"), False),
            (2**70, False),
            ([1, 2**70], False),
            (np.array([2**64 - 1], dtype=np.uint64), False),
            (2**62, True),
            (-(2**63), True),
            (np.int64(5), True),
            (np.array([1.5, 2.5]), True),
            ("text", True),
            (True, True),
            ({"a": 1, "b": [1, 2]}, True),
        ],
    )
    def test_keyword_value_check(self, value, ok):
        from xradio.image._util.casacore import _casa_keyword_value_ok

        assert _casa_keyword_value_ok(value) is ok

    def test_unstorable_user_values_are_dropped(self, tmp_path):
        path = tmp_path / "user.im"
        xds = make_sky_xds()
        xds.SKY.attrs["user"] = {
            "date": np.datetime64("2020-01-01T00:00:00"),
            "big": 2**70,
            "kept": 7,
        }
        _xds_to_casa_image(xds, str(path))
        miscinfo = keywords_of(path)["miscinfo"]
        assert miscinfo == {"kept": 7}


class TestDatetimeTime:
    def test_datetime64_time_is_converted(self, tmp_path):
        """A datetime64 time coordinate gives the right observation date,
        not its nanoseconds since 1970 taken as days."""
        path = tmp_path / "datetime.im"
        xds = make_sky_xds()
        xds = xds.assign_coords(
            time=(
                "time",
                np.array(["2000-01-01T12:00:00"], dtype="datetime64[ns]"),
                {"type": "time", "scale": "utc"},
            )
        )
        _xds_to_casa_image(xds, str(path))
        obsdate = coords_of(path)["obsdate"]
        assert obsdate["refer"] == "UTC"
        np.testing.assert_allclose(obsdate["m0"]["value"], 51544.5, atol=1e-9)


class TestDriverErrors:
    """Errors of write_image(casa) name the variable of the dataset, not the
    SKY or APERTURE the driver writes every image as."""

    def _with(self, variable: str, data_array: xr.DataArray) -> xr.Dataset:
        xds = make_sky_xds()
        xds[variable] = data_array
        xds.attrs["data_groups"]["base"][variable.lower()] = variable
        return xds

    def test_image_without_polarization(self, tmp_path):
        xds = make_sky_xds()
        psf = xds.SKY.isel(polarization=0, drop=True).copy()
        psf.attrs = {"type": "point_spread_function"}
        xds = self._with("POINT_SPREAD_FUNCTION", psf)
        with pytest.raises(ValueError) as error:
            write_image(xds, str(tmp_path / "out"), out_format="casa")
        message = str(error.value)
        assert message.startswith("Cannot write POINT_SPREAD_FUNCTION with dims")
        assert "SKY" not in message.split(":")[0]
        assert not list(tmp_path.iterdir())

    def test_image_of_strings(self, tmp_path):
        xds = make_sky_xds()
        pb = xr.full_like(xds.SKY, "abc", dtype="<U3")
        pb.attrs = {"type": "primary_beam"}
        xds = self._with("PRIMARY_BEAM", pb)
        with pytest.raises(TypeError, match="^Cannot write PRIMARY_BEAM of data type"):
            write_image(xds, str(tmp_path / "out"), out_format="casa")

    def test_metadata_error_names_the_variable(self, tmp_path):
        xds = make_sky_xds(spectral_frame="LSRK")
        xds.frequency.attrs["frame"] = "icrs"
        xds.frequency.attrs["reference_frequency"]["attrs"]["observer"] = "icrs"
        with pytest.raises(ValueError, match="^Cannot write SKY to CASA: "):
            write_image(xds, str(tmp_path / "out"), out_format="casa")


class TestDirectFitsToCasa:
    """R2: a FITS image (big endian pixels, native chunks) written directly to
    CASA, without a zarr store in between, and a dangling SKY flag attribute
    (C8) that write_image ignores in favour of the data group."""

    def test_fits_image_written_to_casa(self, tmp_path):
        from astropy.io import fits as afits

        data = np.arange(2 * 3 * 4 * 5, dtype=np.float32).reshape(2, 3, 4, 5)
        data[1, 0, 0, 0] = np.nan
        header = afits.Header()
        for key, value in {
            "CTYPE1": "RA---SIN",
            "CRVAL1": 10.0,
            "CDELT1": -1 / 3600,
            "CRPIX1": 3.0,
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
            "SPECSYS": "BARYCENT",
            "RADESYS": "FK5",
            "EQUINOX": 2000.0,
            "DATE-OBS": "2020-01-01T00:00:00",
            "TIMESYS": "UTC",
            "TELESCOP": "ALMA",
            "BUNIT": "Jy/beam",
            "BMAJ": 1 / 3600,
            "BMIN": 0.5 / 3600,
            "BPA": 10.0,
        }.items():
            header[key] = value
        fits_path = str(tmp_path / "in.fits")
        afits.PrimaryHDU(data=data, header=header).writeto(fits_path)
        for chunks in ({}, {"frequency": 1}):
            xds = open_image(fits_path, chunks=chunks)
            out = tmp_path / f"out{len(chunks)}.im"
            write_image(xds, str(out), out_format="casa")
            back = open_image(str(out))
            np.testing.assert_array_equal(back.SKY.values, xds.SKY.values)
            np.testing.assert_array_equal(back.FLAG_SKY.values, xds.FLAG_SKY.values)
            assert back.frequency.attrs["frame"] == "BARY"
            np.testing.assert_allclose(
                back.BEAM_FIT_PARAMS_SKY.values, xds.BEAM_FIT_PARAMS_SKY.values
            )

    def test_dangling_flag_attribute_is_ignored(self, tmp_path):
        xds = make_sky_xds()
        xds.SKY.attrs["flag"] = "FLAG_SKY_DOES_NOT_EXIST"
        path = tmp_path / "dangling.im"
        write_image(xds, str(path), out_format="casa")
        assert "masks" not in keywords_of(path)
        np.testing.assert_array_equal(pixels_of(path), xds.SKY.values[0])


class TestUndefinedSpectralFrame:
    """casacore's 'Undefined' spectral frame, which the CASA reader keeps, is
    written back to CASA; FITS gets no SPECSYS."""

    def test_round_trip(self, tmp_path):
        from astropy.io import fits as afits

        path = tmp_path / "undefined.im"
        xds = make_sky_xds()
        xds.frequency.attrs["frame"] = "Undefined"
        xds.frequency.attrs["reference_frequency"]["attrs"]["observer"] = "undefined"
        _xds_to_casa_image(xds, str(path))
        assert coords_of(path)["spectral2"]["system"] == "Undefined"
        back = open_image(str(path))
        assert back.frequency.attrs["frame"] == "Undefined"
        again = tmp_path / "again.im"
        write_image(back, str(again), out_format="casa")
        assert coords_of(again)["spectral2"]["system"] == "Undefined"
        fits_path = tmp_path / "undefined.fits"
        write_image(back, str(fits_path), out_format="fits")
        header = afits.getheader(fits_path)
        assert "SPECSYS" not in header and "VELREF" not in header


class TestApertureOnlyDataset:
    def test_lmuv_dataset_with_only_an_aperture(self, tmp_path):
        """The keywords and miscinfo of a single image dataset come from its
        image variable, also for an aperture image of a dataset with l and m
        coordinates."""
        xds = make_empty_lmuv_image(
            [0.2, -0.5], [8, 6], [_CELL, _CELL], [1.4e9, 1.401e9], ["I"], [59000.5]
        )
        shape = tuple(
            xds.sizes[d] for d in ("time", "frequency", "polarization", "u", "v")
        )
        xds["APERTURE"] = (
            ("time", "frequency", "polarization", "u", "v"),
            np.ones(shape, dtype=np.complex64),
            {"type": "aperture", "user": {"KEY": 1}},
        )
        path = tmp_path / "aperture.im"
        _xds_to_casa_image(xds, str(path))
        assert keywords_of(path)["miscinfo"] == {"KEY": 1}
        with open_image_ro(str(path)) as im:
            assert im.datatype() == "Complex"
