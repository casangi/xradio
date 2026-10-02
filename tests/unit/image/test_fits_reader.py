"""Unit tests for the FITS image reader
(``xradio.image._util._fits.xds_from_fits``).

The FITS files are synthesized with astropy in ``tmp_path``, with the cards
casacore's FITS export writes, so the tests need no downloads and, apart from
the optional casacore export test, neither python-casacore nor casatools.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
import warnings

import dask
import numpy as np
import pytest
import xarray as xr
from astropy.io import fits
from astropy.time import Time

from xradio.image import check_image, open_image
from xradio.image._util._fits import xds_from_fits
from xradio.image._util._fits.xds_from_fits import (
    _fits_image_to_xds,
    _read_image_chunk,
)

C = 299792458.0
REST_FREQUENCY = 1.420405751786e9
NX, NY, NCHAN = 6, 5, 3
DEG = np.pi / 180.0

# FITS axis descriptions: (CTYPE, NAXIS, CRVAL, CDELT, CRPIX, CUNIT)
RA = ("RA---SIN", NX, 105.0, -1.0 / 60.0, 4.0, "deg")
DEC = ("DEC--SIN", NY, -40.0, 1.0 / 60.0, 3.0, "deg")
FREQ = ("FREQ", NCHAN, 1.415e9, 1.0e3, 2.0, "Hz")


def stokes_axis(crval=1.0, cdelt=1.0, n=4, crpix=1.0):
    return ("STOKES", n, crval, cdelt, crpix, "")


IQUV = stokes_axis()


# The dataset dimension of each FITS axis type
_DIMS = {"RA": "l", "DEC": "m", "STOKES": "polarization"}


def _dim(ctype: str) -> str:
    return _DIMS.get(ctype.split("-")[0], "frequency")


@pytest.fixture(autouse=True)
def synchronous_dask():
    with dask.config.set(scheduler="synchronous"):
        yield


@pytest.fixture
def logged_warnings(monkeypatch):
    """Warnings the reader logs, as a list of messages."""

    class _Logger:
        def __init__(self):
            self.warnings = []

        def warning(self, message, *args, **kwargs):
            self.warnings.append(str(message))

        def __getattr__(self, name):
            return lambda *args, **kwargs: None

    logger = _Logger()
    monkeypatch.setattr(xds_from_fits, "xradio_logger", lambda: logger)
    return logger.warnings


def write_fits(
    path,
    axes=(RA, DEC, FREQ, IQUV),
    cards: dict | None = None,
    remove=(),
    data=None,
    dtype=np.float32,
    beams=None,
    extra_hdus=(),
):
    """Write a FITS image with the cards of a casacore FITS export.

    Parameters
    ----------
    path : path-like
        Output file.
    axes : sequence of tuple
        (CTYPE, NAXIS, CRVAL, CDELT, CRPIX, CUNIT) of each FITS axis.
    cards : dict, optional
        Header cards to add or replace.
    remove : sequence of str
        Header cards to remove.
    data : np.ndarray, optional
        Pixels in numpy (reversed FITS axis) order; by default every pixel
        holds its own flat index, so that reordering is detectable.
    dtype : numpy dtype
        Pixel type of the default data.
    beams : np.ndarray, optional
        Per-plane beams (nchan, npol, 3) in arcsec, arcsec, deg, indexed by
        FITS plane, written as a CASA style BEAMS table.
    extra_hdus : sequence of HDU
        Further extension HDUs.

    Returns
    -------
    tuple
        (str path, pixel array).
    """
    shape = tuple(axis[1] for axis in axes)[::-1]
    if data is None:
        data = np.arange(np.prod(shape)).reshape(shape).astype(dtype)
    header = fits.Header()
    header["BTYPE"] = "Intensity"
    header["OBJECT"] = "TEST"
    header["BUNIT"] = "Jy/beam"
    header["EQUINOX"] = 2000.0
    header["RADESYS"] = "FK5"
    header["LONPOLE"] = 180.0
    header["LATPOLE"] = -40.0
    for i in range(1, len(axes) + 1):
        for j in range(1, len(axes) + 1):
            header[f"PC{i}_{j}"] = 1.0 if i == j else 0.0
    for i, (ctype, _n, crval, cdelt, crpix, cunit) in enumerate(axes, start=1):
        header[f"CTYPE{i}"] = ctype
        header[f"CRVAL{i}"] = crval
        header[f"CDELT{i}"] = cdelt
        header[f"CRPIX{i}"] = crpix
        header[f"CUNIT{i}"] = cunit
    header["PV2_1"] = 0.0
    header["PV2_2"] = 0.0
    header["RESTFRQ"] = REST_FREQUENCY
    header["SPECSYS"] = "LSRK"
    header["VELREF"] = 257
    header["TELESCOP"] = "ALMA"
    header["OBSERVER"] = "Karl Jansky"
    header["DATE-OBS"] = "2000-01-01T00:00:00.000"
    header["TIMESYS"] = "UTC"
    header["OBSGEO-X"] = 2.225142180269e06
    header["OBSGEO-Y"] = -5.440307370349e06
    header["OBSGEO-Z"] = -2.481029851874e06
    if beams is None:
        header["BMAJ"] = 1.0 / 3600.0
        header["BMIN"] = 0.5 / 3600.0
        header["BPA"] = 30.0
    else:
        header["CASAMBM"] = True
    for key, value in (cards or {}).items():
        header[key] = value
    for key in remove:
        if key in header:
            del header[key]
    hdus = [fits.PrimaryHDU(data=data, header=header)]
    if beams is not None:
        nchan, npol = beams.shape[:2]
        chans = np.repeat(np.arange(nchan), npol)
        pols = np.tile(np.arange(npol), nchan)
        rows = beams[chans, pols]
        table = fits.BinTableHDU.from_columns(
            [
                fits.Column(name="BMAJ", format="E", unit="arcsec", array=rows[:, 0]),
                fits.Column(name="BMIN", format="E", unit="arcsec", array=rows[:, 1]),
                fits.Column(name="BPA", format="E", unit="deg", array=rows[:, 2]),
                fits.Column(name="CHAN", format="J", array=chans),
                fits.Column(name="POL", format="J", array=pols),
            ],
            name="BEAMS",
        )
        table.header["NCHAN"] = nchan
        table.header["NPOL"] = npol
        hdus.append(table)
    hdus.extend(extra_hdus)
    fits.HDUList(hdus).writeto(path, overwrite=True)
    return str(path), data


def read(path, chunks=None, **kwargs) -> xr.Dataset:
    return _fits_image_to_xds(
        str(path),
        {} if chunks is None else chunks,
        False,
        kwargs.pop("do_sky_coords", True),
        kwargs.pop("compute_mask", True),
        **kwargs,
    )


def expected_sky(data, axes, order=None) -> np.ndarray:
    """The SKY array (time, frequency, polarization, l, m) that the reader
    should return for ``data``, with the polarization axis permuted by
    ``order`` (FITS plane indices in output order)."""
    dims = [_dim(axis[0]) for axis in axes][::-1]
    xda = xr.DataArray(data, dims=dims)
    for dim in ("time", "frequency", "polarization"):
        if dim not in xda.dims:
            xda = xda.expand_dims(dim)
    xda = xda.transpose("time", "frequency", "polarization", "l", "m")
    if order is not None:
        xda = xda.isel(polarization=order)
    return xda.values


# --------------------------------------------------------------------------- #
# Polarization axis (F2)                                                       #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "crval,cdelt,n,fits_labels,expected",
    [
        (1, 1, 4, ["I", "Q", "U", "V"], ["I", "Q", "U", "V"]),
        (-1, -1, 4, ["RR", "LL", "RL", "LR"], ["RR", "RL", "LR", "LL"]),
        (-5, -1, 4, ["XX", "YY", "XY", "YX"], ["XX", "XY", "YX", "YY"]),
        (-5, -1, 2, ["XX", "YY"], ["XX", "YY"]),
        (-1, -1, 2, ["RR", "LL"], ["RR", "LL"]),
        (-6, 1, 2, ["YY", "XX"], ["XX", "YY"]),
        (1, 3, 2, ["I", "V"], ["I", "V"]),
        (-1, 1, 1, ["RR"], ["RR"]),
    ],
)
def test_stokes_axis_decoded_and_put_in_canonical_order(
    tmp_path, crval, cdelt, n, fits_labels, expected
):
    """FITS STOKES codes are decoded and the planes, with their flags and
    beams, are put in canonical (Jones) order."""
    axes = (RA, DEC, FREQ, stokes_axis(crval, cdelt, n))
    shape = (n, NCHAN, NY, NX)
    data = np.arange(np.prod(shape)).reshape(shape).astype(np.float32)
    # flag one pixel of the last FITS plane
    data[n - 1, 1, 2, 3] = np.nan
    # beam major axis 1 + chan + 0.1 * FITS plane arcsec, pa 10 * plane deg
    beams = np.zeros((NCHAN, n, 3))
    for chan in range(NCHAN):
        for plane in range(n):
            beams[chan, plane] = (1.0 + chan + 0.1 * plane, 0.5, 10.0 * plane)
    path, data = write_fits(tmp_path / "pol.fits", axes, data=data, beams=beams)

    xds = read(path)

    assert list(xds.polarization.values) == expected
    order = [fits_labels.index(label) for label in expected]
    np.testing.assert_array_equal(
        xds.SKY.values, expected_sky(data, axes, order), strict=True
    )
    flagged = xds.FLAG_SKY.sel(polarization=fits_labels[-1]).values
    assert flagged[0, 1, 3, 2]
    assert int(xds.FLAG_SKY.values.sum()) == 1
    beam = xds.BEAM_FIT_PARAMS_SKY.isel(time=0)
    np.testing.assert_allclose(
        beam.sel(beam_params_label="major").values,
        beams[:, order, 0] / 3600.0 * DEG,
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        beam.sel(beam_params_label="pa").values, beams[:, order, 2] * DEG, rtol=1e-6
    )


def test_casa_export_layout_rr_ll_rl_lr(tmp_path):
    """A CASA export of a correlation image (STOKES last, CRVAL4 = CDELT4 = -1,
    i.e. RR, LL, RL, LR, with a BEAMS table indexed by FITS plane) opens as
    RR, RL, LR, LL."""
    axes = (RA, DEC, FREQ, stokes_axis(-1.0, -1.0, 4))
    shape = (4, NCHAN, NY, NX)
    data = np.zeros(shape, dtype=np.float32)
    for plane in range(4):
        for chan in range(NCHAN):
            data[plane, chan] = 10 * chan + plane
    beams = np.zeros((NCHAN, 4, 3))
    for chan in range(NCHAN):
        for plane in range(4):
            beams[chan, plane] = (1.0 + chan + 0.1 * plane, 0.5, 10.0 * plane)
    path, _ = write_fits(tmp_path / "rrll.fits", axes, data=data, beams=beams)

    xds = read(path, chunks={"polarization": 1, "frequency": 2})

    assert list(xds.polarization.values) == ["RR", "RL", "LR", "LL"]
    np.testing.assert_array_equal(
        xds.SKY.isel(time=0, l=0, m=0).values,
        [[0, 2, 3, 1], [10, 12, 13, 11], [20, 22, 23, 21]],
    )
    major = (
        xds.BEAM_FIT_PARAMS_SKY.isel(time=0).sel(beam_params_label="major").values
        / DEG
        * 3600.0
    )
    np.testing.assert_allclose(
        major, [[1.0, 1.2, 1.3, 1.1], [2.0, 2.2, 2.3, 2.1], [3.0, 3.2, 3.3, 3.1]]
    )


def test_casacore_export_opens_in_canonical_order(tmp_path):
    """A FITS file written by casacore from a CASA image with polarizations
    RR, LL, RL, LR opens as RR, RL, LR, LL with the matching pixels."""
    pytest.importorskip("casacore.images")
    from casacore.images import coordinates
    from casacore.images import image as casa_image

    template = casa_image(str(tmp_path / "template.im"), shape=[3, 4, 5, 6])
    csys = template.coordinates().dict()
    del template
    csys["stokes1"]["stokes"] = ["RR", "LL", "RL", "LR"]
    image = casa_image(
        str(tmp_path / "rrll.im"),
        shape=[3, 4, 5, 6],
        coordsys=coordinates.coordinatesystem(csys),
    )
    data = np.zeros((3, 4, 5, 6), dtype=np.float32)
    for chan in range(3):
        for pol in range(4):
            data[chan, pol] = 10 * chan + pol
    image.putdata(data)
    fits_path = str(tmp_path / "rrll.fits")
    image.tofits(fits_path)
    del image

    xds = read(fits_path)

    assert list(xds.polarization.values) == ["RR", "RL", "LR", "LL"]
    np.testing.assert_array_equal(
        xds.SKY.isel(time=0, l=0, m=0).values,
        [[0, 2, 3, 1], [10, 12, 13, 11], [20, 22, 23, 21]],
    )


def test_unsupported_stokes_code_raises(tmp_path):
    """AIPS polarized intensity codes (5 to 8) are not polarizations."""
    path, _ = write_fits(tmp_path / "pi.fits", (RA, DEC, FREQ, stokes_axis(5.0)))
    with pytest.raises(ValueError, match="not a supported Stokes or correlation"):
        read(path)


def test_fits_round_trip_keeps_canonical_order(tmp_path):
    """write_image(out_format="fits") of RR, RL, LR, LL images and reopening
    gives RR, RL, LR, LL with the same pixels, flags and beams."""
    from xradio.image import write_image

    axes = (RA, DEC, FREQ, stokes_axis(-1.0, -1.0, 4))
    shape = (4, NCHAN, NY, NX)
    data = np.arange(np.prod(shape)).reshape(shape).astype(np.float32)
    data[1, 0, 0, 0] = np.nan
    beams = np.zeros((NCHAN, 4, 3))
    for chan in range(NCHAN):
        for plane in range(4):
            beams[chan, plane] = (1.0 + chan + 0.1 * plane, 0.5, 10.0 * plane)
    path, _ = write_fits(tmp_path / "in.fits", axes, data=data, beams=beams)
    xds = open_image(path)
    assert list(xds.polarization.values) == ["RR", "RL", "LR", "LL"]

    out = str(tmp_path / "out.fits")
    write_image(xds, out, out_format="fits")
    back = open_image(out)

    assert list(back.polarization.values) == ["RR", "RL", "LR", "LL"]
    np.testing.assert_array_equal(back.SKY.values, xds.SKY.values)
    np.testing.assert_array_equal(back.FLAG_SKY.values, xds.FLAG_SKY.values)
    np.testing.assert_allclose(
        back.BEAM_FIT_PARAMS_SKY.values, xds.BEAM_FIT_PARAMS_SKY.values, rtol=1e-6
    )


# --------------------------------------------------------------------------- #
# Spectral reference frame (F3, R9) and doppler type (F10)                     #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "specsys,frame,observer",
    [
        ("LSRK", "LSRK", "lsrk"),
        ("LSRD", "LSRD", "lsrd"),
        ("BARYCENT", "BARY", "BARY"),
        ("HELIOCEN", "BARY", "BARY"),
        ("TOPOCENT", "TOPO", "TOPO"),
        ("GEOCENTR", "GEO", "gcrs"),
        ("SOURCE", "REST", "REST"),
        ("GALACTOC", "GALACTO", "GALACTO"),
        ("LOCALGRP", "LGROUP", "LGROUP"),
        ("CMBDIPOL", "CMB", "CMB"),
        # casacore frame name, as written by earlier xradio FITS writers
        ("BARY", "BARY", "BARY"),
    ],
)
def test_specsys_translated_to_casacore_frame(tmp_path, specsys, frame, observer):
    path, _ = write_fits(tmp_path / "frame.fits", cards={"SPECSYS": specsys})
    xds = read(path)
    assert xds.frequency.attrs["frame"] == frame
    reference = xds.frequency.attrs["reference_frequency"]
    assert reference["attrs"]["observer"] == observer
    assert reference["attrs"]["units"] == "Hz"
    assert not check_image(open_image(path))


@pytest.mark.parametrize(
    "velref,frame",
    [(257, "LSRK"), (258, "BARY"), (259, "TOPO"), (260, "LSRD"), (2, "BARY")],
)
def test_missing_specsys_uses_velref(tmp_path, logged_warnings, velref, frame):
    path, _ = write_fits(
        tmp_path / "velref.fits", cards={"VELREF": velref}, remove=("SPECSYS",)
    )
    assert read(path).frequency.attrs["frame"] == frame
    assert not logged_warnings


def test_missing_specsys_uses_ctype_frame_suffix(tmp_path, logged_warnings):
    axes = (RA, DEC, ("FREQ-HEL",) + FREQ[1:], stokes_axis())
    path, _ = write_fits(tmp_path / "hel.fits", axes, remove=("SPECSYS", "VELREF"))
    xds = read(path)
    assert xds.frequency.attrs["frame"] == "BARY"
    assert xds.sizes["frequency"] == NCHAN
    assert not logged_warnings


def test_missing_specsys_and_velref_assume_lsrk(tmp_path, logged_warnings):
    path, _ = write_fits(tmp_path / "none.fits", remove=("SPECSYS", "VELREF"))
    xds = read(path)
    assert xds.frequency.attrs["frame"] == "LSRK"
    assert any("assuming LSRK" in message for message in logged_warnings)


def test_unknown_specsys_is_ignored_with_warning(tmp_path, logged_warnings):
    path, _ = write_fits(
        tmp_path / "unknown.fits", cards={"SPECSYS": "NONSENSE", "VELREF": 259}
    )
    assert read(path).frequency.attrs["frame"] == "TOPO"
    assert any("NONSENSE" in message for message in logged_warnings)


@pytest.mark.parametrize(
    "velref,doppler_type", [(257, "radio"), (1, "z"), (3, "z"), (None, "radio")]
)
def test_frequency_axis_doppler_type_from_velref(tmp_path, velref, doppler_type):
    """VELREF above 256 means radio velocities, else optical (AIPS); without
    VELREF a frequency axis gets radio velocities."""
    cards = {} if velref is None else {"VELREF": velref}
    remove = ("VELREF",) if velref is None else ()
    path, _ = write_fits(tmp_path / "doppler.fits", cards=cards, remove=remove)
    xds = read(path)
    assert xds.velocity.attrs["doppler_type"] == doppler_type
    freq = xds.frequency.values
    if doppler_type == "radio":
        expected = C * (1.0 - freq / REST_FREQUENCY)
    else:
        expected = C * (REST_FREQUENCY / freq - 1.0)
    np.testing.assert_allclose(xds.velocity.values, expected, rtol=1e-12)


# --------------------------------------------------------------------------- #
# Spectral axis types, units and the rest frequency (R8, R11, R13)             #
# --------------------------------------------------------------------------- #


def test_vopt_axis(tmp_path):
    """A VOPT axis is linear in optical velocity; its channel width is the
    frequency step from the reference channel to the next (the step casacore
    exports as CDELT)."""
    f_ref, df = 1.415e9, 1.0e3
    v_ref = C * (REST_FREQUENCY / f_ref - 1.0)
    dv = C * (REST_FREQUENCY / (f_ref + df) - 1.0) - v_ref
    axes = (RA, DEC, stokes_axis(), ("VOPT", NCHAN, v_ref, dv, 2.0, "m/s"))
    path, _ = write_fits(tmp_path / "vopt.fits", axes, cards={"VELREF": 1})
    xds = read(path)
    velocity = v_ref + (np.arange(NCHAN) - 1.0) * dv
    np.testing.assert_allclose(xds.velocity.values, velocity, rtol=1e-12)
    np.testing.assert_allclose(
        xds.frequency.values, REST_FREQUENCY / (1.0 + velocity / C), rtol=1e-12
    )
    assert xds.velocity.attrs["doppler_type"] == "z"
    attrs = xds.frequency.attrs
    assert attrs["reference_frequency"]["data"] == pytest.approx(f_ref, rel=1e-12)
    assert attrs["channel_width"]["data"] == pytest.approx(df, rel=1e-9)
    assert attrs["channel_width"]["attrs"]["units"] == "Hz"


def test_vopt_axis_in_km_per_s(tmp_path):
    axes_ms = (RA, DEC, stokes_axis(), ("VOPT", NCHAN, 1.2e6, -200.0, 2.0, "m/s"))
    axes_kms = (RA, DEC, stokes_axis(), ("VOPT", NCHAN, 1.2e3, -0.2, 2.0, "km/s"))
    ms = read(write_fits(tmp_path / "ms.fits", axes_ms)[0])
    kms = read(write_fits(tmp_path / "kms.fits", axes_kms)[0])
    np.testing.assert_allclose(kms.frequency.values, ms.frequency.values, rtol=1e-12)
    np.testing.assert_allclose(kms.velocity.values, ms.velocity.values, rtol=1e-12)


def test_vrad_axis(tmp_path):
    axes = (RA, DEC, stokes_axis(), ("VRAD", NCHAN, 1.0e6, -200.0, 2.0, "m/s"))
    path, _ = write_fits(tmp_path / "vrad.fits", axes, cards={"VELREF": 1})
    xds = read(path)
    velocity = 1.0e6 + (np.arange(NCHAN) - 1.0) * -200.0
    np.testing.assert_allclose(xds.velocity.values, velocity, rtol=1e-12)
    np.testing.assert_allclose(
        xds.frequency.values, REST_FREQUENCY * (1.0 - velocity / C), rtol=1e-12
    )
    assert xds.velocity.attrs["doppler_type"] == "radio"
    attrs = xds.frequency.attrs
    assert attrs["channel_width"]["data"] == pytest.approx(
        REST_FREQUENCY * 200.0 / C, rel=1e-12
    )
    assert attrs["reference_frequency"]["data"] == pytest.approx(
        REST_FREQUENCY * (1.0 - 1.0e6 / C), rel=1e-12
    )


def test_aips_felo_axis(tmp_path, logged_warnings):
    """An AIPS FELO axis is an optical velocity axis sampled linearly in
    frequency, its CDELT the velocity increment at the reference pixel; the
    frame comes from the CTYPE suffix."""
    v_ref, dv = 1.2e6, -200.0
    axes = (RA, DEC, stokes_axis(), ("FELO-HEL", NCHAN, v_ref, dv, 2.0, "m/s"))
    path, _ = write_fits(tmp_path / "felo.fits", axes, remove=("SPECSYS", "VELREF"))
    xds = read(path)
    f_ref = REST_FREQUENCY / (1.0 + v_ref / C)
    df = -dv * f_ref / (C + v_ref)
    np.testing.assert_allclose(
        xds.frequency.values, f_ref + (np.arange(NCHAN) - 1.0) * df, rtol=1e-12
    )
    assert xds.frequency.attrs["frame"] == "BARY"
    assert xds.velocity.attrs["doppler_type"] == "z"
    assert xds.frequency.attrs["channel_width"]["data"] == pytest.approx(abs(df))
    assert not logged_warnings


def test_velocity_axis_without_rest_frequency_raises(tmp_path):
    axes = (RA, DEC, stokes_axis(), ("VOPT", NCHAN, 1.2e6, -200.0, 2.0, "m/s"))
    path, _ = write_fits(tmp_path / "norest.fits", axes, remove=("RESTFRQ",))
    with pytest.raises(RuntimeError, match="no rest frequency"):
        read(path)


@pytest.mark.parametrize("ctype", ["VOPT-F2W", "WAVE", "VELO-LSR"])
def test_unsupported_spectral_axis_raises(tmp_path, ctype):
    axes = (RA, DEC, stokes_axis(), (ctype, NCHAN, 1.0, 1.0, 1.0, "m"))
    path, _ = write_fits(tmp_path / "wave.fits", axes)
    with pytest.raises(RuntimeError, match="unsupported spectral axis"):
        read(path)


@pytest.mark.parametrize(
    "cunit,scale", [("GHz", 1e9), ("MHz", 1e6), ("kHz", 1e3), ("HZ", 1.0)]
)
def test_frequency_axis_converted_to_hz(tmp_path, cunit, scale):
    axes = (RA, DEC, ("FREQ", NCHAN, 1.415e9 / scale, 1.0e3 / scale, 2.0, cunit))
    path, _ = write_fits(tmp_path / "ghz.fits", axes + (stokes_axis(),))
    xds = read(path)
    expected = 1.415e9 + (np.arange(NCHAN) - 1.0) * 1.0e3
    np.testing.assert_allclose(xds.frequency.values, expected, rtol=1e-12)
    attrs = xds.frequency.attrs
    assert attrs["units"] == "Hz"
    assert attrs["reference_frequency"]["data"] == pytest.approx(1.415e9, rel=1e-12)
    assert attrs["channel_width"]["data"] == pytest.approx(1.0e3, rel=1e-9)
    np.testing.assert_allclose(
        xds.velocity.values, C * (1.0 - expected / REST_FREQUENCY), rtol=1e-9
    )


def test_restfreq_alias_and_rest_wavelength(tmp_path):
    alias, _ = write_fits(
        tmp_path / "alias.fits",
        cards={"RESTFREQ": 1.0e9},
        remove=("RESTFRQ",),
    )
    assert read(alias).frequency.attrs["rest_frequency"]["data"] == 1.0e9
    wavelength, _ = write_fits(
        tmp_path / "restwav.fits",
        cards={"RESTWAV": 0.21},
        remove=("RESTFRQ",),
    )
    assert read(wavelength).frequency.attrs["rest_frequency"]["data"] == (
        pytest.approx(C / 0.21, rel=1e-12)
    )


def test_frequency_axis_without_rest_frequency(tmp_path):
    """Without a rest frequency the image has no velocity coordinate and the
    rest frequency is 0 Hz (casacore's value for an unknown one)."""
    path, _ = write_fits(tmp_path / "norest.fits", remove=("RESTFRQ",))
    xds = read(path)
    assert "velocity" not in xds.coords
    assert xds.frequency.attrs["rest_frequency"]["data"] == 0.0
    assert not check_image(open_image(path))


def test_channel_width_of_single_channel(tmp_path):
    axes = (RA, DEC, ("FREQ", 1, 1.0e11, -2.0e9, 1.0, "Hz"), stokes_axis())
    path, _ = write_fits(tmp_path / "one.fits", axes)
    attrs = read(path).frequency.attrs
    assert attrs["channel_width"]["data"] == 2.0e9
    assert attrs["channel_width"]["attrs"] == {"units": "Hz", "type": "quantity"}


@pytest.mark.parametrize(
    "axes,polarization,nchan",
    [
        pytest.param((RA, DEC), ["I"], 1, id="2d"),
        pytest.param((RA, DEC, FREQ), ["I"], NCHAN, id="no_stokes"),
        pytest.param((RA, DEC, stokes_axis()), ["I", "Q", "U", "V"], 1, id="no_freq"),
        pytest.param((DEC, RA, stokes_axis(-5, -1, 2)), ["XX", "YY"], 1, id="dec_ra"),
    ],
)
def test_images_without_spectral_or_stokes_axis(tmp_path, axes, polarization, nchan):
    path, data = write_fits(tmp_path / "axes.fits", axes)
    xds = read(path)
    assert xds.SKY.dims == ("time", "frequency", "polarization", "l", "m")
    assert list(xds.polarization.values) == polarization
    assert xds.sizes["frequency"] == nchan
    np.testing.assert_array_equal(xds.SKY.values, expected_sky(data, axes))
    attrs = xds.frequency.attrs
    assert attrs["units"] == "Hz"
    assert attrs["frame"] == "LSRK"
    assert attrs["reference_frequency"]["attrs"]["observer"] == "lsrk"
    assert attrs["channel_width"]["data"] == 1.0e3
    assert not check_image(open_image(path))


def test_image_without_spectral_axis_gets_casacore_default(tmp_path):
    """Like a CASA image without a spectral axis: one channel of casacore's
    default spectral coordinate, whatever SPECSYS and RESTFRQ say."""
    path, _ = write_fits(
        tmp_path / "nofreq.fits",
        (RA, DEC, stokes_axis()),
        cards={"SPECSYS": "BARYCENT", "RESTFRQ": 1.0e11},
    )
    xds = read(path)
    np.testing.assert_array_equal(xds.frequency.values, [1.415e9])
    attrs = xds.frequency.attrs
    assert attrs["frame"] == "LSRK"
    assert attrs["rest_frequency"]["data"] == pytest.approx(REST_FREQUENCY)
    assert attrs["reference_frequency"]["data"] == 1.415e9
    assert xds.velocity.attrs["doppler_type"] == "radio"
    np.testing.assert_allclose(
        xds.velocity.values, C * (1.0 - 1.415e9 / REST_FREQUENCY), rtol=1e-9
    )


# --------------------------------------------------------------------------- #
# Direction axes (F12, F13, F14)                                               #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "radesys,equinox,frame,expected_equinox",
    [
        ("FK5", 2000.0, "fk5", "j2000.0"),
        ("FK5", None, "fk5", "j2000.0"),
        ("FK5", "J2000", "fk5", "j2000.0"),
        ("FK4", 1950.0, "fk4", "b1950.0"),
        ("FK4", None, "fk4", "b1950.0"),
        ("ICRS", None, "icrs", None),
        (None, 1950.0, "fk4", "b1950.0"),
        (None, 2000.0, "fk5", "j2000.0"),
        (None, None, "icrs", None),
    ],
)
def test_direction_frame_and_equinox(
    tmp_path, radesys, equinox, frame, expected_equinox
):
    cards = {}
    remove = []
    for key, value in (("RADESYS", radesys), ("EQUINOX", equinox)):
        if value is None:
            remove.append(key)
        else:
            cards[key] = value
    path, _ = write_fits(tmp_path / "frame.fits", cards=cards, remove=remove)
    xds = read(path)
    reference = xds.attrs["coordinate_system_info"]["reference_direction"]
    assert reference["attrs"]["frame"] == frame
    assert reference["attrs"].get("equinox") == expected_equinox
    assert xds.SKY.attrs["pointing_center"]["attrs"]["frame"] == frame
    assert not check_image(open_image(path))


def test_unsupported_radesys_raises(tmp_path):
    path, _ = write_fits(tmp_path / "gappt.fits", cards={"RADESYS": "GAPPT"})
    with pytest.raises(RuntimeError, match="Unsupported FITS RADESYS"):
        read(path)


@pytest.mark.parametrize(
    "cards,remove,expected",
    [
        ({"PV2_1": 0.1, "PV2_2": -0.2}, (), [0.1, -0.2]),
        ({}, ("PV2_1", "PV2_2"), [0.0, 0.0]),
        ({"PV2_2": 0.3}, ("PV2_1",), [0.0, 0.3]),
        # PV cards of other axes are not projection parameters
        ({"PV1_1": 5.0}, ("PV2_1", "PV2_2"), [0.0, 0.0]),
    ],
)
def test_projection_parameters(tmp_path, cards, remove, expected):
    path, _ = write_fits(tmp_path / "pv.fits", cards=cards, remove=remove)
    csys = read(path).attrs["coordinate_system_info"]
    assert csys["projection_parameters"] == pytest.approx(expected)


def test_missing_pole_and_pc_cards_take_fits_defaults(tmp_path):
    remove = ["LONPOLE", "LATPOLE"] + [f"PC{i}_{j}" for i in (1, 2) for j in (1, 2)]
    path, _ = write_fits(tmp_path / "defaults.fits", remove=remove)
    csys = read(path).attrs["coordinate_system_info"]
    np.testing.assert_allclose(
        csys["native_pole_direction"]["data"], [np.pi, -40.0 * DEG], rtol=1e-12
    )
    assert csys["pixel_coordinate_transformation_matrix"] == [[1.0, 0.0], [0.0, 1.0]]


_ALL_PC_CARDS = [f"PC{i}_{j}" for i in range(1, 5) for j in range(1, 5)]


def test_old_style_crota_rotation_becomes_pc_matrix(tmp_path):
    """Without PC cards, an AIPS CROTA2 rotation is converted to the PC matrix
    (FITS WCS Paper II) instead of being passed through as a user keyword."""
    path, _ = write_fits(
        tmp_path / "crota.fits", cards={"CROTA2": 30.0}, remove=_ALL_PC_CARDS
    )
    xds = read(path)
    rho = 30.0 * DEG
    ratio = DEC[3] / RA[3]
    np.testing.assert_allclose(
        xds.attrs["coordinate_system_info"]["pixel_coordinate_transformation_matrix"],
        [[np.cos(rho), -np.sin(rho) * ratio], [np.sin(rho) / ratio, np.cos(rho)]],
        rtol=1e-12,
    )
    assert "crota2" not in xds.SKY.attrs["user"]


@pytest.mark.parametrize(
    "cards",
    [
        # slant orthographic (NCP-like) SIN, as importfits writes NCP images
        {"PV2_2": 1.0 / np.tan(-40.0 * DEG)},
        # rotated pixel grid
        {
            "PC1_1": np.cos(20 * DEG),
            "PC1_2": -np.sin(20 * DEG),
            "PC2_1": np.sin(20 * DEG),
            "PC2_2": np.cos(20 * DEG),
        },
        # old style rotation
        {"CROTA2": 15.0},
    ],
    ids=["slant", "pc_rotation", "crota"],
)
def test_sky_coordinates_follow_the_full_wcs(tmp_path, cards):
    """right_ascension and declination use the projection parameters, the
    rotation and the native pole, as astropy (wcslib) and casacore do."""
    from astropy.wcs import WCS

    remove = _ALL_PC_CARDS if "CROTA2" in cards else ()
    path, _ = write_fits(tmp_path / "wcs.fits", cards=cards, remove=remove)
    xds = read(path)
    with warnings.catch_warnings():
        # wcslib's header fixes (MJD-OBS, OBSGEO) do not matter here
        warnings.simplefilter("ignore")
        wcs = WCS(fits.getheader(path)).celestial
    x, y = np.indices((NX, NY))
    ra, dec = wcs.pixel_to_world_values(x, y)
    tolerance = 1e-9 / 3600  # degrees
    ra_difference = (np.degrees(xds.right_ascension.values) - ra + 180) % 360 - 180
    assert np.abs(ra_difference).max() < tolerance
    assert np.abs(np.degrees(xds.declination.values) - dec).max() < tolerance


def test_legacy_pc_cards(tmp_path):
    cards = {"PC001001": 0.9, "PC001002": 0.1, "PC002001": -0.1, "PC002002": 0.9}
    path, _ = write_fits(tmp_path / "pc.fits", cards=cards, remove=_ALL_PC_CARDS)
    xds = read(path)
    assert xds.attrs["coordinate_system_info"][
        "pixel_coordinate_transformation_matrix"
    ] == [[0.9, 0.1], [-0.1, 0.9]]
    assert not any(key.startswith("pc") for key in xds.SKY.attrs["user"])


def test_layout_cards_are_not_user_keywords(tmp_path):
    """Cards that describe the file (checksums) or that the reader interprets
    are not carried as user keywords, so writers cannot copy stale values."""
    cards = {
        "CHECKSUM": "9cTEAZQD9bQDAZQD",
        "DATASUM": "123",
        "RESTFREQ": 1.0e9,
        "MJD-OBS": 51544.0,
        "INSTRUME": "BAND6",
    }
    path, _ = write_fits(tmp_path / "user.fits", cards=cards)
    user = read(path).SKY.attrs["user"]
    for key in ("checksum", "datasum", "restfreq", "restfrq", "mjd-obs", "specsys"):
        assert key not in user
    assert user["instrume"] == "BAND6"


@pytest.mark.parametrize(
    "cdelt1_sign,cdelt2_sign", [(-1, 1), (1, 1), (-1, -1), (1, -1)]
)
def test_l_and_m_keep_the_sign_of_cdelt(tmp_path, cdelt1_sign, cdelt2_sign):
    """l follows the RA axis and m the Dec axis whichever way the pixels run,
    so that l, m and the sky coordinates of each pixel agree."""
    ra = RA[:3] + (cdelt1_sign / 60.0,) + RA[4:]
    dec = DEC[:3] + (cdelt2_sign / 60.0,) + DEC[4:]
    path, _ = write_fits(tmp_path / "lm.fits", (ra, dec, FREQ, stokes_axis()))
    xds = read(path)
    cdelt = DEG / 60.0
    np.testing.assert_allclose(
        xds.l.values, (np.arange(NX) - 3.0) * cdelt1_sign * cdelt, rtol=1e-12
    )
    np.testing.assert_allclose(
        xds.m.values, (np.arange(NY) - 2.0) * cdelt2_sign * cdelt, rtol=1e-12
    )
    # east (larger RA) where l > 0 and north where m > 0, off the reference
    # pixel
    ra_offset = xds.right_ascension.isel(m=2).values - 105.0 * DEG
    dec_offset = xds.declination.isel(l=3).values + 40.0 * DEG
    l_values = xds.l.values
    m_values = xds.m.values
    assert (np.sign(ra_offset[l_values != 0]) == np.sign(l_values[l_values != 0])).all()
    assert (
        np.sign(dec_offset[m_values != 0]) == np.sign(m_values[m_values != 0])
    ).all()


def test_pointing_center_from_obsra_obsdec(tmp_path):
    path, _ = write_fits(
        tmp_path / "obs.fits", cards={"OBSRA": 105.1, "OBSDEC": -40.05}
    )
    sky = read(path).SKY
    np.testing.assert_allclose(
        sky.attrs["pointing_center"]["data"], [105.1 * DEG, -40.05 * DEG], rtol=1e-12
    )
    assert "obsra" not in sky.attrs["user"]
    assert "obsdec" not in sky.attrs["user"]
    plain, _ = write_fits(tmp_path / "crval.fits")
    np.testing.assert_allclose(
        read(plain).SKY.attrs["pointing_center"]["data"],
        [105.0 * DEG, -40.0 * DEG],
        rtol=1e-12,
    )


# --------------------------------------------------------------------------- #
# Pixels (R2)                                                                  #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int16, np.int32])
def test_pixels_in_native_byte_order_with_declared_dtype(tmp_path, dtype):
    path, data = write_fits(tmp_path / "dtype.fits", dtype=dtype)
    xds = read(path, chunks={"frequency": 2})
    assert xds.SKY.dtype == np.dtype(dtype)
    values = xds.SKY.values
    assert values.dtype == np.dtype(dtype)
    assert values.dtype.isnative
    np.testing.assert_array_equal(
        values, expected_sky(data, (RA, DEC, FREQ, stokes_axis()))
    )
    # the FITS file is big-endian: the chunk reader returns a native copy
    chunk = _read_image_chunk(path, (2, 2, 3, 4), (1, 0, 1, 2), np.dtype(dtype).name)
    assert chunk.dtype == np.dtype(dtype) and chunk.dtype.isnative
    assert chunk.flags["C_CONTIGUOUS"]
    np.testing.assert_array_equal(chunk, data[1:3, 0:2, 1:4, 2:6])
    assert "FLAG_SKY" not in xds


# --------------------------------------------------------------------------- #
# Telescope and observation date (R11)                                         #
# --------------------------------------------------------------------------- #


def test_missing_telescope_is_unknown(tmp_path):
    path, _ = write_fits(tmp_path / "notel.fits", remove=("TELESCOP",))
    assert read(path).SKY.attrs["telescope"]["name"] == "UNKNOWN"


@pytest.mark.parametrize(
    "cards,remove,mjd,scale",
    [
        ({}, (), 51544.0, "utc"),
        ({}, ("TIMESYS",), 51544.0, "utc"),
        ({"TIMESYS": "TAI"}, (), 51544.0, "tai"),
        ({"MJD-OBS": 51000.5}, ("DATE-OBS",), 51000.5, "utc"),
        ({"DATE-OBS": "31/12/98"}, (), Time("1998-12-31").mjd, "utc"),
        ({"DATE-OBS": "2000-01-01"}, (), 51544.0, "utc"),
    ],
)
def test_observation_date(tmp_path, cards, remove, mjd, scale):
    path, _ = write_fits(tmp_path / "date.fits", cards=cards, remove=remove)
    xds = read(path)
    assert xds.time.values[0] == pytest.approx(mjd, abs=1e-9)
    assert xds.time.attrs == {
        "units": "d",
        "scale": scale,
        "format": "mjd",
        "type": "time",
    }
    assert xds.SKY.attrs["obsdate"]["data"] == pytest.approx(mjd, abs=1e-9)
    assert "mjd-obs" not in xds.SKY.attrs["user"]


@pytest.mark.parametrize(
    "cards,remove",
    [
        ({}, ("DATE-OBS",)),
        ({"DATE-OBS": "not a date"}, ()),
        # MJD 0, casacore's unset date, as a writer may write it
        ({"DATE-OBS": "1858-11-17T00:00:00.000"}, ()),
        ({"MJD-OBS": 0.0}, ("DATE-OBS",)),
    ],
)
def test_missing_observation_date_uses_placeholder(
    tmp_path, logged_warnings, cards, remove
):
    path, _ = write_fits(tmp_path / "nodate.fits", cards=cards, remove=remove)
    xds = read(path)
    assert xds.time.values[0] == 0.0
    assert xds.time.attrs["format"] == "mjd"
    # like a CASA image with an unset date, no obsdate attribute
    assert "obsdate" not in xds.SKY.attrs
    assert any("observation date" in message for message in logged_warnings)
    assert not check_image(open_image(path))


# --------------------------------------------------------------------------- #
# Sum of weights images                                                        #
# --------------------------------------------------------------------------- #


def test_sumwt_fits_opens_by_name(tmp_path):
    """A tclean sum of weights exported to FITS (1 x 1 pixel direction axes)
    is opened as VISIBILITY_NORMALIZATION without the direction axes, as the
    CASA reader opens a .sumwt image, alone, in a list of tclean products and
    through the xarray engine."""
    one_pixel = (
        ("RA---SIN", 1, 105.0, -1.0 / 60.0, 1.0, "deg"),
        ("DEC--SIN", 1, -40.0, 1.0 / 60.0, 1.0, "deg"),
        FREQ,
        IQUV,
    )
    sumwt, data = write_fits(tmp_path / "target.sumwt.fits", one_pixel)
    sky, _ = write_fits(tmp_path / "target.image.fits")

    xds = open_image(sumwt)
    assert list(xds.data_vars) == ["VISIBILITY_NORMALIZATION"]
    assert xds.VISIBILITY_NORMALIZATION.dims == ("time", "frequency", "polarization")
    assert "l" not in xds.coords and "right_ascension" not in xds.coords
    np.testing.assert_array_equal(
        xds.VISIBILITY_NORMALIZATION.values[0], data[:, :, 0, 0].T
    )

    both = open_image([sky, sumwt])
    assert both.VISIBILITY_NORMALIZATION.dims == ("time", "frequency", "polarization")
    assert both.attrs["data_groups"]["base"]["visibility_normalization"] == (
        "VISIBILITY_NORMALIZATION"
    )
    assert not check_image(both)

    engine = xr.open_dataset(sumwt, engine="xradio_fits_image")
    np.testing.assert_array_equal(
        engine.VISIBILITY_NORMALIZATION.values, xds.VISIBILITY_NORMALIZATION.values
    )


def test_sumwt_with_direction_axes_raises(tmp_path):
    path, _ = write_fits(tmp_path / "big.fits")
    with pytest.raises(ValueError, match="must have direction axes of one pixel"):
        open_image({"visibility_normalization": path})


# --------------------------------------------------------------------------- #
# Other HDUs and packaging (X4)                                                #
# --------------------------------------------------------------------------- #


def test_other_extension_hdus_are_ignored_with_a_warning(tmp_path, logged_warnings):
    table = fits.BinTableHDU.from_columns(
        [fits.Column(name="FLUX", format="E", array=np.ones(3))], name="AIPS CC"
    )
    path, data = write_fits(tmp_path / "aips.fits", extra_hdus=[table])
    xds = read(path)
    np.testing.assert_array_equal(
        xds.SKY.values, expected_sky(data, (RA, DEC, FREQ, stokes_axis()))
    )
    assert any("AIPS CC" in message for message in logged_warnings)


def test_files_without_a_primary_image_raise(tmp_path):
    """FITS tables and UVFITS (random groups) files are rejected clearly."""
    table = str(tmp_path / "table.fits")
    fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.BinTableHDU.from_columns(
                [fits.Column(name="A", format="E", array=np.ones(3))]
            ),
        ]
    ).writeto(table)
    groups = fits.GroupData(
        np.zeros((2, 1, 1, 1, 1, 3), dtype=np.float32),
        parnames=["UU", "VV"],
        pardata=[np.zeros(2), np.zeros(2)],
        bitpix=-32,
    )
    uvfits = str(tmp_path / "uv.fits")
    fits.GroupsHDU(groups).writeto(uvfits)
    for path in (table, uvfits):
        with pytest.raises(RuntimeError, match="primary HDU of the FITS file holds no"):
            read(path)


_WITHOUT_CASACORE = """
import sys
for name in ("casacore", "casatools", "casaconfig"):
    sys.modules[name] = None
import dask
dask.config.set(scheduler="synchronous")
"""


def _run_without_casacore(code: str, path: str) -> list[str]:
    """Run ``code`` in a python process in which python-casacore and
    casatools cannot be imported; return the lines it prints with
    ``print("RESULT", ...)``, without the prefix."""
    result = subprocess.run(
        [sys.executable, "-c", _WITHOUT_CASACORE + textwrap.dedent(code), path],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr[-3000:]
    return [
        line.removeprefix("RESULT ")
        for line in result.stdout.splitlines()
        if line.startswith("RESULT ")
    ]


def test_reader_does_not_import_casacore(tmp_path):
    axes = (RA, DEC, FREQ, stokes_axis(-1, -1))
    path, _ = write_fits(tmp_path / "plain.fits", axes)
    out = _run_without_casacore(
        """
        from xradio.image._util._fits.xds_from_fits import _fits_image_to_xds
        xds = _fits_image_to_xds(sys.argv[1], {}, False, True, True)
        xds.SKY.values
        print("RESULT", [str(p) for p in xds.polarization.values])
        casa = ("casacore", "casatools")
        loaded = [m for m in sys.modules if m.split(".")[0] in casa and sys.modules[m]]
        print("RESULT", loaded)
        """,
        path,
    )
    assert out == ["['RR', 'RL', 'LR', 'LL']", "[]"]


def test_open_image_and_backend_without_casacore(tmp_path):
    path, _ = write_fits(tmp_path / "plain.fits")
    out = _run_without_casacore(
        """
        import xarray as xr
        from xradio.image import open_image
        print("RESULT", sorted(open_image(sys.argv[1]).data_vars))
        xds = xr.open_dataset(sys.argv[1], engine="xradio_fits_image")
        print("RESULT", sorted(xds.data_vars))
        """,
        path,
    )
    assert out == ["['BEAM_FIT_PARAMS_SKY', 'SKY']"] * 2
