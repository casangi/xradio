"""
Writer for a single FITS image.

The complete FITS header (and the BEAMS extension of per plane beams) is
built and validated before the output file is created. The pixels are then
written region by region into the data area of the primary HDU, so that each
chunk of dask backed data is computed exactly once and the cube is never
held in memory as a whole (unless the input itself is a single chunk). The
regions are computed in batches with the active dask scheduler (see
:mod:`xradio.image._util._blocks`).

FITS only needs astropy: this module must not import python-casacore or
casatools.
"""

import contextlib
import itertools
import os
import re

import numpy as np
import xarray as xr
from astropy import units as u
from astropy.io import fits
from astropy.time import ScaleValueError

from xradio._utils.logging import xradio_logger
from xradio.image._util import conventions
from xradio.image._util._blocks import RegionReader
from xradio.image._util.common import (
    _compute_sky_reference_pixel,
    _time_coord_to_astropy,
)

# Dimensions of an image data variable written to FITS
_IMAGE_DIMS = ("time", "frequency", "polarization", "l", "m")
# numpy axis order of the FITS data cube (NAXIS4, NAXIS3, NAXIS2, NAXIS1)
_FITS_DIMS = ("frequency", "polarization", "m", "l")
# FITS headers and data areas are padded to whole blocks of this size
_FITS_BLOCK_BYTES = 2880

# Frames the FITS reader can round trip (it only supports RA-/DEC- CTYPEs)
_EQUATORIAL_FRAMES = ("fk4", "fk5", "icrs")
# FITS EQUINOX written when the reference direction has no equinox attribute
_DEFAULT_EQUINOX = {"fk5": 2000.0, "fk4": 1950.0}

# VELREF frame codes (AIPS convention: 1 LSR, 2 HEL, 3 OBS; 4 to 7 are
# casacore's non-standard extension), keyed by casacore spectral frame. The
# Local Group and CMB frames have no code, so no VELREF is written for them.
_VELREF_FRAME_CODES = {
    "LSRK": 1,
    "BARY": 2,
    "TOPO": 3,
    "LSRD": 4,
    "GEO": 5,
    "REST": 6,
    "GALACTO": 7,
}
_VELREF_RADIO = 256
_VELREF_COMMENT = "1 LSR, 2 HEL, 3 OBS, +256 Radio"

_BEAM_LABELS = ("major", "minor", "pa")
# Per plane beams that agree to this relative tolerance are written as a
# single beam (no absolute tolerance: beam sizes are tiny numbers in radians)
_BEAM_RTOL = 1e-5

# Header keywords that are never copied from the 'user' attributes: the
# structural and WCS keywords this writer manages (whether or not it writes
# them for a given image), and keywords that would be stale for new pixels.
_RESERVED_USER_KEYWORDS = frozenset(
    """
    SIMPLE BITPIX NAXIS EXTEND END XTENSION PCOUNT GCOUNT GROUPS EXTNAME EXTVER
    EXTLEVEL BSCALE BZERO BLANK BUNIT BTYPE DATAMIN DATAMAX CHECKSUM DATASUM
    WCSAXES WCSNAME LONPOLE LATPOLE RADESYS RADECSYS EQUINOX EPOCH SPECSYS
    SSYSOBS SSYSSRC VELREF VELOSYS ZSOURCE RESTFRQ RESTFREQ RESTWAV ALTRVAL
    ALTRPIX DATE-OBS MJD-OBS TIMESYS OBSRA OBSDEC BMAJ BMIN BPA CASAMBM OBJECT
    OBSERVER TELESCOP
    """.split()
)
_RESERVED_USER_KEYWORD_PATTERN = re.compile(
    r"^(NAXIS|CTYPE|CRVAL|CDELT|CRPIX|CUNIT|CROTA|CRDER|CSYER)\d+$"
    r"|^(PC|CD|PV|PS)\d+_\d+$"
    r"|^PC\d{6}$"
    r"|^OBSGEO-[XYZ]$"
)

# FITS header text must be printable ASCII
_NON_PRINTABLE_ASCII = re.compile(r"[^\x20-\x7e]")


class _Header(fits.Header):
    """FITS header whose errors name the card: astropy rejects for example
    non-finite values without saying which keyword they were meant for."""

    def __setitem__(self, key, value):
        try:
            super().__setitem__(key, value)
        except ValueError as exc:
            raise ValueError(
                f"Cannot write the header card {key} to FITS: {exc}"
            ) from None


def _fits_text(value, keyword: str, replaced: list) -> str:
    """Return ``value`` as FITS header text: characters that are not
    printable ASCII are replaced with '?', and ``keyword`` is recorded in
    ``replaced`` when that happens."""
    text = str(value)
    clean = _NON_PRINTABLE_ASCII.sub("?", text)
    if clean != text and keyword not in replaced:
        replaced.append(keyword)
    return clean


def _first(value):
    """First element of a list valued attribute (units are sometimes stored
    as one element lists), else the value itself."""
    if isinstance(value, list | tuple):
        return value[0] if value else None
    return value


def _unit_factor(units, target: str, what: str) -> float:
    """Factor converting values in ``units`` to ``target``."""
    try:
        return float(u.Unit(_first(units)).to(target))
    except (ValueError, TypeError, u.UnitsError) as exc:
        raise ValueError(
            f"Cannot convert {what} with units {units!r} to {target}: {exc}"
        ) from None


def _measure_values(measure, target: str, what: str, default_units=None) -> np.ndarray:
    """Values of a measure or quantity in ``target`` units.

    Parameters
    ----------
    measure : dict or float
        A measure dict (``{"data": ..., "attrs": {"units": ...}}``) or a bare
        number.
    target : str
        Units to convert to.
    what : str
        Description of the measure for error messages.
    default_units : str, optional
        Units of a bare number or of a measure without ``units`` (default
        ``target``).

    Returns
    -------
    np.ndarray
        Flat float array of the values in ``target`` units.
    """
    units = default_units or target
    data = measure
    if isinstance(measure, dict):
        if "data" not in measure:
            raise ValueError(f"{what} has no 'data'")
        data = measure["data"]
        units = (measure.get("attrs") or {}).get("units") or units
    try:
        values = np.asarray(data, dtype=float).reshape(-1)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{what} is not numeric: {exc}") from None
    if isinstance(units, list | tuple) and len(units) == values.size > 1:
        factors = np.array([_unit_factor(x, target, what) for x in units])
    else:
        factors = _unit_factor(units, target, what)
    return values * factors


def _scalar(measure, target: str, what: str, default_units=None) -> float:
    """Single value of a measure or quantity in ``target`` units."""
    values = _measure_values(measure, target, what, default_units)
    if values.size == 0:
        raise ValueError(f"{what} is empty")
    return float(values[0])


def _fits_data_type(dtype: np.dtype, name: str) -> tuple:
    """FITS data type for image pixels of numpy type ``dtype``.

    Floating point data keeps its precision (byte order does not matter);
    booleans and integers are converted exactly (16 bit and smaller to
    float32, larger to float64). FITS has no complex image type.

    Returns
    -------
    tuple of (np.dtype, int)
        Big endian output dtype and the FITS BITPIX value.
    """
    dtype = np.dtype(dtype)
    if dtype.kind == "f" and dtype.itemsize <= 4:
        out = np.dtype(">f4")
    elif dtype.kind == "f":
        out = np.dtype(">f8")
        if dtype.itemsize > 8:
            xradio_logger().warning(
                f"Writing {name} of type {dtype} to FITS as float64, the "
                "largest FITS floating point type"
            )
    elif dtype.kind == "b" or (dtype.kind in "iu" and dtype.itemsize <= 2):
        out = np.dtype(">f4")
    elif dtype.kind in "iu":
        out = np.dtype(">f8")
    elif dtype.kind == "c":
        raise ValueError(
            f"Cannot write {name} to FITS: its data are complex ({dtype}) and "
            "FITS images can only hold real values. Write the real and "
            "imaginary parts (or amplitude and phase) as separate images."
        )
    else:
        raise ValueError(f"Cannot write {name} to FITS: unsupported data type {dtype}")
    return out, -8 * out.itemsize


def _check_image_layout(xds: xr.Dataset) -> None:
    """Check that the dataset holds an image FITS can represent."""
    if "SKY" not in xds.data_vars:
        raise RuntimeError("The dataset to write to FITS has no SKY variable")
    image = xds["SKY"]
    if "l" not in image.dims or "m" not in image.dims:
        raise RuntimeError(
            "Writing to FITS is only supported for sky plane images (with l "
            "and m dimensions)"
        )
    if set(image.dims) != set(_IMAGE_DIMS):
        raise RuntimeError(
            f"Writing to FITS requires an image with dimensions {_IMAGE_DIMS}, "
            f"got {image.dims}"
        )
    if image.sizes["time"] != 1:
        raise RuntimeError(
            "XDS can only be converted to FITS if it has exactly one time plane"
        )
    empty = [dim for dim in _IMAGE_DIMS if image.sizes[dim] == 0]
    if empty:
        raise RuntimeError(
            f"Cannot write an empty image to FITS: {empty} have no pixels"
        )
    if "FLAG" in xds.data_vars and set(xds["FLAG"].dims) != set(_IMAGE_DIMS):
        raise RuntimeError(
            f"The flags of an image written to FITS must have dimensions "
            f"{_IMAGE_DIMS}, got {xds['FLAG'].dims}"
        )
    for name in ("l", "m", "frequency", "polarization"):
        if name not in xds.coords:
            raise RuntimeError(
                f"Writing to FITS requires a {name} coordinate, but the "
                "dataset has none"
            )


def _direction_axis_increments(xds: xr.Dataset) -> tuple:
    """Pixel increments of the l and m axes, in radians.

    A single value does not define the increment of a single pixel axis: it
    comes from the coordinate's 'cdelt' attribute, else (as in the CASA
    writer) it is the other axis' pixel size, with the usual signs (l
    decreases and m increases with the pixel index). Any value keeps the
    world coordinate of the single pixel; it only sets the pixel's extent.
    """
    increments = {}
    for name in ("l", "m"):
        coord = xds[name]
        values = np.asarray(coord.values, dtype=float)
        if values.size > 1:
            increments[name] = conventions.linear_axis_increment(values, name)
            continue
        cdelt = coord.attrs.get("cdelt")
        if cdelt is None:
            increments[name] = None
            continue
        increment = _scalar(cdelt, "rad", f"The {name} coordinate's cdelt")
        if not np.isfinite(increment) or increment == 0:
            raise ValueError(
                f"The {name} coordinate's cdelt attribute must be a non-zero "
                f"finite increment, got {increment}"
            )
        increments[name] = increment
    for name, other, sign in (("l", "m", -1.0), ("m", "l", 1.0)):
        if increments[name] is None:
            if increments[other] is None:
                raise ValueError(
                    "Cannot write a 1 x 1 pixel image to FITS: the pixel "
                    "increments cannot be derived from single l and m values. "
                    "Set the l and m coordinates' 'cdelt' attributes (the "
                    "increments in radians, or quantities with units)."
                )
            increments[name] = sign * abs(increments[other])
    return increments["l"], increments["m"]


def _sky_reference_pixel(xds: xr.Dataset, increments: tuple) -> np.ndarray:
    """0-based (l, m) pixel position where l and m are zero (outside the
    image for a cutout that does not contain the reference direction)."""
    if xds.sizes["l"] > 1 and xds.sizes["m"] > 1:
        return np.asarray(_compute_sky_reference_pixel(xds), dtype=float)
    # one value does not define the increment of a single pixel axis
    return np.asarray(
        _compute_sky_reference_pixel(xds, cdelt=list(increments)), dtype=float
    )


def _equinox_to_fits(equinox) -> float:
    """Convert an equinox such as 'j2000.0', 'B1950' or 2000 to the FITS
    EQUINOX value (a year number)."""
    try:
        return float(str(equinox).strip().lower().lstrip("jb"))
    except ValueError:
        raise ValueError(f"Cannot interpret equinox {equinox!r}") from None


def _direction_cards(xds: xr.Dataset, header: fits.Header) -> dict:
    """Add the celestial axis cards (axes 1 and 2).

    Returns
    -------
    dict
        The direction reference frame (``frame`` and ``equinox``), used to
        express the pointing center in the same frame.
    """
    csys = xds.attrs.get("coordinate_system_info")
    if not csys:
        raise RuntimeError(
            "Writing to FITS requires the coordinate_system_info dataset "
            "attribute (only sky plane images are supported)"
        )
    ref_dir = csys["reference_direction"]
    frame = str(ref_dir["attrs"]["frame"]).lower()
    if frame not in _EQUATORIAL_FRAMES:
        raise RuntimeError(
            f"Writing to FITS is only supported for equatorial direction "
            f"reference frames {_EQUATORIAL_FRAMES}, got '{frame}'"
        )
    projection = str(csys["projection"]).strip().upper()
    if not re.fullmatch(r"[A-Z0-9]{3}", projection):
        raise ValueError(
            f"Cannot write projection {csys['projection']!r} to FITS: FITS "
            "projection codes have three characters"
        )
    crval = _measure_values(ref_dir, "deg", "The reference direction")
    pole = _measure_values(
        csys["native_pole_direction"], "deg", "The native pole direction", "rad"
    )
    pc = np.asarray(csys["pixel_coordinate_transformation_matrix"], dtype=float)
    if pc.shape != (2, 2):
        raise ValueError(
            "pixel_coordinate_transformation_matrix must be a 2x2 matrix, got "
            f"shape {pc.shape}"
        )
    projection_parameters = csys.get("projection_parameters")
    projection_parameters = np.asarray(
        [] if projection_parameters is None else projection_parameters, dtype=float
    ).reshape(-1)
    increments = _direction_axis_increments(xds)
    crpix = _sky_reference_pixel(xds, increments)

    equinox = ref_dir["attrs"].get("equinox")
    if frame == "icrs":
        equinox = None
    elif equinox is not None:
        header["EQUINOX"] = _equinox_to_fits(equinox)
    else:
        header["EQUINOX"] = _DEFAULT_EQUINOX[frame]
    header["RADESYS"] = frame.upper()
    header["LONPOLE"] = float(pole[0])
    header["LATPOLE"] = float(pole[1])
    for i in (0, 1):
        for j in (0, 1):
            header[f"PC{i + 1}_{j + 1}"] = float(pc[i][j])
    header["PC3_3"] = 1.0
    header["PC4_4"] = 1.0
    for axis, (ctype, value, increment, pixel) in enumerate(
        zip(("RA---", "DEC--"), crval, increments, crpix, strict=True), start=1
    ):
        header[f"CTYPE{axis}"] = ctype + projection
        header[f"CRVAL{axis}"] = float(value)
        header[f"CDELT{axis}"] = float(np.degrees(increment))
        header[f"CRPIX{axis}"] = float(pixel) + 1.0
        header[f"CUNIT{axis}"] = "deg"
    # Projection parameters belong to the latitude axis (FITS WCS paper II),
    # numbered from 1 as casacore writes them (for example SIN xi, eta), but
    # from 0 for ZPN, whose polynomial starts with the constant term PV2_0 (as
    # the reader numbers them)
    if np.any(projection_parameters != 0):
        first = 0 if projection == "ZPN" else 1
        for k, value in enumerate(projection_parameters, start=first):
            header[f"PV2_{k}"] = float(value)
    return {"frame": frame, "equinox": equinox}


def _stokes_cards(header: fits.Header, crval: int, cdelt: int) -> None:
    """Add the STOKES axis cards (axis 3)."""
    header["CTYPE3"] = "STOKES"
    header["CRVAL3"] = float(crval)
    header["CDELT3"] = float(cdelt)
    header["CRPIX3"] = 1.0
    header["CUNIT3"] = ""


def _spectral_frame(freq_attrs: dict) -> str | None:
    """casacore spectral reference frame of the frequency axis.

    The frequency coordinate's ``frame`` attribute holds it; stores written by
    xradio 1.2.3 and earlier lack that attribute, so the reference
    frequency's observer is used then. None for casacore's ``Undefined``
    frame (of CASA images without a known frame), which has no ``SPECSYS``.
    """
    frame = freq_attrs.get("frame")
    observer = ((freq_attrs.get("reference_frequency") or {}).get("attrs") or {}).get(
        "observer"
    )
    if frame:
        source, name = "frame", frame
    elif observer:
        source, name = "reference frequency observer", observer
    else:
        raise ValueError(
            "Cannot write the frequency axis to FITS: the frequency coordinate "
            "has neither a 'frame' attribute nor a reference_frequency "
            "observer, so its spectral reference frame (FITS SPECSYS) is "
            "unknown"
        )
    if str(name).strip().lower() == "undefined":
        xradio_logger().warning(
            "The spectral reference frame of the image is undefined (casacore's "
            "'Undefined' frame): writing the FITS image without SPECSYS, which "
            "FITS readers take as their default frame"
        )
        return None
    try:
        casacore_frame = conventions.normalize_spectral_frame(name)
    except ValueError as exc:
        raise ValueError(
            f"Cannot write the frequency axis to FITS: its {source} {name!r} "
            f"has no FITS SPECSYS equivalent. {exc}"
        ) from None
    if frame and observer:
        try:
            observer_frame = conventions.normalize_spectral_frame(observer)
        except ValueError:
            observer_frame = None
        if observer_frame != casacore_frame:
            xradio_logger().warning(
                f"The frequency coordinate's frame {frame!r} and its reference "
                f"frequency observer {observer!r} disagree; writing SPECSYS for "
                f"frame {frame!r}"
            )
    return casacore_frame


def _velref(casacore_frame: str, xds: xr.Dataset) -> int | None:
    """AIPS/casacore VELREF value: the frame code plus 256 for radio
    velocities, or None for frames without a code."""
    code = _VELREF_FRAME_CODES.get(casacore_frame)
    if code is None:
        return None
    doppler = "radio"
    if "velocity" in xds.coords:
        doppler = str(xds["velocity"].attrs.get("doppler_type") or "radio").lower()
    if doppler in ("optical", "z"):
        return code
    if doppler != "radio":
        xradio_logger().warning(
            f"FITS VELREF can only describe radio or optical velocities; "
            f"writing the {doppler!r} velocity convention as radio"
        )
    return code + _VELREF_RADIO


def _frequency_cards(xds: xr.Dataset, header: fits.Header) -> None:
    """Add the FREQ axis cards (axis 4) and the spectral keywords."""
    freq = xds["frequency"]
    attrs = freq.attrs
    freq_units = attrs.get("units") or "Hz"
    values = np.asarray(freq.values, dtype=float) * _unit_factor(
        freq_units, "Hz", "the frequency coordinate"
    )
    if "reference_frequency" not in attrs:
        raise ValueError(
            "Cannot write the frequency axis to FITS: the frequency coordinate "
            "has no reference_frequency attribute"
        )
    crval = _scalar(
        attrs["reference_frequency"], "Hz", "The reference frequency", freq_units
    )
    # raises for a non-uniform axis (FITS FREQ axes are linear)
    cdelt = conventions.linear_axis_increment(values, "frequency")
    if cdelt is None:
        # a single channel: its width, else the conventional fallback width
        # (also for an unusable width, as in the CASA writer)
        cdelt = conventions.SINGLE_CHANNEL_WIDTH_FALLBACK_HZ
        width = attrs.get("channel_width")
        if width is not None:
            value = _scalar(width, "Hz", "The channel_width attribute", freq_units)
            if np.isfinite(value) and value != 0:
                cdelt = value
            else:
                xradio_logger().warning(
                    f"The frequency coordinate's channel_width is {value} Hz; "
                    f"writing the single channel with a width of {cdelt} Hz"
                )
    casacore_frame = _spectral_frame(attrs)

    header["CTYPE4"] = "FREQ"
    header["CRVAL4"] = crval
    header["CDELT4"] = float(cdelt)
    header["CRPIX4"] = (crval - float(values[0])) / cdelt + 1.0
    header["CUNIT4"] = "Hz"
    if attrs.get("rest_frequency") is not None:
        header["RESTFRQ"] = (
            _scalar(attrs["rest_frequency"], "Hz", "The rest frequency"),
            "Rest Frequency (Hz)",
        )
    if casacore_frame is None:
        return
    header["SPECSYS"] = (
        conventions.spectral_frame_to_fits_specsys(casacore_frame),
        "Spectral reference frame",
    )
    velref = _velref(casacore_frame, xds)
    if velref is not None:
        header["VELREF"] = (velref, _VELREF_COMMENT)


def _time_cards(xds: xr.Dataset, header: fits.Header) -> None:
    """Add DATE-OBS (full precision), MJD-OBS and TIMESYS from the time
    coordinate, honouring its units, format and scale attributes. MJD 0, the
    placeholder the readers use for an unknown date (casacore's unset date),
    is not written, as in casacore's FITS export."""
    if "time" not in xds.coords:
        xradio_logger().warning(
            "Not writing DATE-OBS to FITS: the dataset has no time coordinate"
        )
        return
    time = xds["time"]
    attrs = dict(time.attrs)
    try:
        # datetime64 values are converted, not taken as numbers
        obstime = _time_coord_to_astropy(np.ravel(time.values)[0], attrs)
        # nanoseconds: more than the precision of a float64 MJD
        obstime.precision = 9
        date_obs = obstime.isot
        mjd_obs = float(obstime.mjd)
    except (ValueError, TypeError, ScaleValueError, u.UnitsError) as exc:
        xradio_logger().warning(
            f"Not writing DATE-OBS to FITS: the time {time.values.tolist()} "
            f"with units {attrs.get('units')!r}, format {attrs.get('format')!r} "
            f"and scale {attrs.get('scale')!r} is not a valid date ({exc})"
        )
        return
    if mjd_obs == 0.0:
        xradio_logger().info(
            "Not writing DATE-OBS to FITS: the time is MJD 0, the placeholder "
            "of an unknown observation date"
        )
        return
    header["DATE-OBS"] = date_obs
    header["MJD-OBS"] = mjd_obs
    header["TIMESYS"] = obstime.scale.upper()


def _pointing_center_cards(
    image: xr.DataArray, direction_frame: dict, header: fits.Header
) -> None:
    """Add OBSRA/OBSDEC (degrees, as casacore writes them) from the pointing
    center, expressed in the frame of the image's direction axes."""
    pointing = image.attrs.get("pointing_center")
    if not pointing:
        return
    if not isinstance(pointing, dict):
        xradio_logger().warning(
            "Not writing OBSRA/OBSDEC to FITS: the pointing center is not a "
            "sky coordinate measure"
        )
        return
    try:
        lon, lat = _measure_values(pointing, "rad", "The pointing center", "rad")[:2]
        attrs = pointing.get("attrs") or {}
        # without a frame the pointing center is in the image's frame, as in
        # casacore images
        frame = str(attrs.get("frame") or direction_frame["frame"]).lower()
        equinox = attrs.get("equinox")
        target_frame = direction_frame["frame"]
        target_equinox = direction_frame["equinox"]
        other_equinox = (
            equinox is not None
            and target_equinox is not None
            and _equinox_to_fits(equinox) != _equinox_to_fits(target_equinox)
        )
        if frame != target_frame or other_equinox:
            lon, lat = _convert_direction(
                lon, lat, frame, equinox, target_frame, target_equinox
            )
    except Exception as exc:
        # optional cards; astropy frame conversions raise several exception
        # types (ValueError, UnitsError, ConvertError, ...)
        xradio_logger().warning(
            f"Not writing OBSRA/OBSDEC to FITS: cannot interpret the pointing "
            f"center ({exc})"
        )
        return
    header["OBSRA"] = float(np.degrees(lon) % 360.0)
    header["OBSDEC"] = float(np.degrees(lat))


def _astropy_frame(frame: str, equinox):
    """astropy frame instance for a direction frame name and equinox."""
    from astropy.coordinates import frame_transform_graph
    from astropy.time import Time

    frame_class = frame_transform_graph.lookup_name(frame)
    if frame_class is None:
        raise ValueError(f"Unknown direction frame {frame!r}")
    if frame in ("fk5", "fk4", "fk4noterms"):
        prefix = "B" if frame.startswith("fk4") else "J"
        year = _equinox_to_fits(equinox) if equinox is not None else None
        if year is not None:
            return frame_class(equinox=Time(f"{prefix}{year}"))
    return frame_class()


def _convert_direction(lon, lat, frame, equinox, target_frame, target_equinox):
    """Convert a direction (radians) from one sky frame to another."""
    from astropy.coordinates import SkyCoord

    if target_equinox is None and target_frame in _DEFAULT_EQUINOX:
        target_equinox = _DEFAULT_EQUINOX[target_frame]
    coord = SkyCoord(
        lon * u.rad, lat * u.rad, frame=_astropy_frame(frame, equinox)
    ).transform_to(_astropy_frame(target_frame, target_equinox))
    spherical = coord.spherical
    return float(spherical.lon.to_value(u.rad)), float(spherical.lat.to_value(u.rad))


def _telescope_cards(image: xr.DataArray, header: fits.Header, replaced: list) -> None:
    """Add TELESCOP and the geocentric OBSGEO-X/Y/Z telescope position."""
    telescope = image.attrs.get("telescope")
    if not isinstance(telescope, dict):
        telescope = {}
    header["TELESCOP"] = _fits_text(
        telescope.get("name") or "UNKNOWN", "TELESCOP", replaced
    )
    direction = telescope.get("direction")
    distance = telescope.get("distance")
    if direction is None or distance is None:
        return
    try:
        # both readers store geocentric spherical ITRF coordinates
        lon, lat = _measure_values(direction, "rad", "The telescope direction")[:2]
        r = _scalar(distance, "m", "The telescope distance")
    except (ValueError, TypeError, KeyError, IndexError) as exc:
        xradio_logger().warning(
            f"Not writing OBSGEO-X/Y/Z to FITS: cannot interpret the telescope "
            f"position ({exc})"
        )
        return
    header["OBSGEO-X"] = r * np.cos(lat) * np.cos(lon)
    header["OBSGEO-Y"] = r * np.cos(lat) * np.sin(lon)
    header["OBSGEO-Z"] = r * np.sin(lat)


def _image_info_cards(image: xr.DataArray, header: fits.Header, replaced: list) -> None:
    """Add BTYPE, OBJECT and BUNIT."""
    attrs = image.attrs
    sub_type = attrs.get("sub_type")
    if sub_type:
        # casacore only recognizes its own spelling ('Spectral Index')
        btype = conventions.sub_type_to_casacore(sub_type) or sub_type
        header["BTYPE"] = _fits_text(btype, "BTYPE", replaced)
    if attrs.get("object_name"):
        header["OBJECT"] = _fits_text(attrs["object_name"], "OBJECT", replaced)
    units = _first(attrs.get("units"))
    if units:
        header["BUNIT"] = (
            _fits_text(units, "BUNIT", replaced),
            "Brightness (pixel) unit",
        )


def _beam_cards(
    xds: xr.Dataset, header: fits.Header, pol_order: list
) -> fits.BinTableHDU | None:
    """Add a single beam as BMAJ/BMIN/BPA cards, or return per plane beams as
    a CASA style BEAMS binary table (rows in the FITS polarization order)."""
    if "BEAM_FIT_PARAMS" not in xds.data_vars:
        return None
    bfp = xds["BEAM_FIT_PARAMS"]
    if "time" in bfp.dims:
        bfp = bfp.isel(time=0)
    if set(bfp.dims) != {"frequency", "polarization", "beam_params_label"}:
        raise ValueError(
            "Beam fit parameters written to FITS must have dimensions (time, "
            f"frequency, polarization, beam_params_label), got {xds['BEAM_FIT_PARAMS'].dims}"
        )
    if "beam_params_label" in bfp.coords:
        labels = [str(label) for label in bfp["beam_params_label"].values]
        missing = [label for label in _BEAM_LABELS if label not in labels]
        if missing:
            raise ValueError(
                f"Cannot write beams to FITS: beam_params_label {labels} lacks "
                f"{missing} (expected {list(_BEAM_LABELS)})"
            )
        bfp = bfp.sel(beam_params_label=list(_BEAM_LABELS))
    elif bfp.sizes["beam_params_label"] != len(_BEAM_LABELS):
        raise ValueError(
            "Cannot write beams to FITS: beam fit parameters need the three "
            f"values {list(_BEAM_LABELS)}, got {bfp.sizes['beam_params_label']}"
        )
    bfp = bfp.transpose("frequency", "polarization", "beam_params_label")
    to_rad = _unit_factor(bfp.attrs.get("units") or "rad", "rad", "The beams")
    beams = np.asarray(bfp.values, dtype=float) * to_rad  # (chan, pol, 3)
    if np.allclose(beams, beams[0, 0], rtol=_BEAM_RTOL, atol=0.0):
        bmaj, bmin, bpa = np.degrees(beams[0, 0])
        header["BMAJ"] = float(bmaj)
        header["BMIN"] = float(bmin)
        header["BPA"] = float(bpa)
        return None
    header["CASAMBM"] = True
    nchan, npol = beams.shape[:2]
    # BEAMS rows index the FITS axes: polarization pixel q holds the original
    # polarization pol_order[q]
    chans = np.repeat(np.arange(nchan), npol)
    pols = np.tile(np.arange(npol), nchan)
    beam_rows = beams[chans, np.asarray(pol_order)[pols]]
    columns = fits.ColDefs(
        [
            fits.Column(
                name="BMAJ",
                format="E",
                unit="arcsec",
                array=np.degrees(beam_rows[:, 0]) * 3600.0,
            ),
            fits.Column(
                name="BMIN",
                format="E",
                unit="arcsec",
                array=np.degrees(beam_rows[:, 1]) * 3600.0,
            ),
            fits.Column(
                name="BPA", format="E", unit="deg", array=np.degrees(beam_rows[:, 2])
            ),
            fits.Column(name="CHAN", format="J", array=chans),
            fits.Column(name="POL", format="J", array=pols),
        ]
    )
    beams_hdu = fits.BinTableHDU.from_columns(columns, name="BEAMS")
    beams_hdu.header["NCHAN"] = nchan
    beams_hdu.header["NPOL"] = npol
    return beams_hdu


def _is_reserved_keyword(keyword: str) -> bool:
    return keyword in _RESERVED_USER_KEYWORDS or bool(
        _RESERVED_USER_KEYWORD_PATTERN.match(keyword)
    )


def _user_cards(
    image: xr.DataArray, xds: xr.Dataset, header: fits.Header, replaced: list
) -> None:
    """Copy the 'user' keywords (dataset level, overridden by the image's
    own) that do not clash with keywords this writer manages."""
    user = {**(xds.attrs.get("user") or {}), **(image.attrs.get("user") or {})}
    for key, value in user.items():
        keyword = str(key).strip().upper()
        if keyword in ("COMMENT", "HISTORY"):
            values = value if isinstance(value, list | tuple) else [value]
            for text in values:
                if keyword == "COMMENT":
                    header.add_comment(_fits_text(text, keyword, replaced))
                else:
                    header.add_history(_fits_text(text, keyword, replaced))
            continue
        if not keyword or len(keyword) > 8 or keyword in header:
            continue
        if _is_reserved_keyword(keyword):
            xradio_logger().debug(
                f"Not copying user keyword {keyword} to the FITS header: the "
                "writer manages it"
            )
            continue
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, str):
            value = _fits_text(value, keyword, replaced)
        try:
            header[keyword] = value
        except (ValueError, TypeError):
            xradio_logger().warning(
                f"Could not write user keyword {keyword} to the FITS header"
            )


def _image_label(image: xr.DataArray) -> str:
    """How messages name an image: the writers put every image under SKY, so
    it is named by its role (its ``type`` attribute) when it has one."""
    image_type = image.attrs.get("type")
    return f"the {image_type} image" if image_type else "SKY"


def _fits_image_header(xds: xr.Dataset) -> tuple:
    """Build and validate the complete FITS header of an image.

    Nothing is computed from the pixels and nothing is written: callers can
    validate every output of a multi-image write before writing any.

    Parameters
    ----------
    xds : xr.Dataset
        Single image dataset: data variable SKY (dimensions time, frequency,
        polarization, l, m) with optional FLAG and BEAM_FIT_PARAMS variables.

    Returns
    -------
    header : astropy.io.fits.Header
        Primary HDU header.
    beams_hdu : astropy.io.fits.BinTableHDU or None
        BEAMS extension holding per plane beams, if any.
    pol_order : list of int
        FITS polarization axis order: FITS polarization pixel ``q`` holds the
        dataset's polarization ``pol_order[q]``.

    Raises
    ------
    RuntimeError, ValueError
        If the image cannot be represented in FITS.
    """
    _check_image_layout(xds)
    image = xds["SKY"]
    _, bitpix = _fits_data_type(image.dtype, _image_label(image))
    labels = [str(label) for label in xds["polarization"].values]
    # FITS STOKES codes form a linear axis, which may need another plane order
    pol_order, stokes_crval, stokes_cdelt = conventions.fits_stokes_axis(labels)

    header = _Header()
    header["SIMPLE"] = (True, "Standard FITS")
    header["BITPIX"] = bitpix
    header["NAXIS"] = 4
    for axis, dim in enumerate(reversed(_FITS_DIMS), start=1):
        header[f"NAXIS{axis}"] = xds.sizes[dim]
    header["EXTEND"] = True
    replaced = []
    _image_info_cards(image, header, replaced)
    beams_hdu = _beam_cards(xds, header, pol_order)
    direction_frame = _direction_cards(xds, header)
    _stokes_cards(header, stokes_crval, stokes_cdelt)
    _frequency_cards(xds, header)
    _telescope_cards(image, header, replaced)
    if image.attrs.get("observer"):
        header["OBSERVER"] = _fits_text(image.attrs["observer"], "OBSERVER", replaced)
    _time_cards(xds, header)
    _pointing_center_cards(image, direction_frame, header)
    _user_cards(image, xds, header, replaced)
    if replaced:
        xradio_logger().warning(
            "Replaced characters that are not printable ASCII with '?' in the "
            f"FITS header keywords {', '.join(replaced)}"
        )
    return header, beams_hdu, list(pol_order)


def _block_slices(sky: xr.DataArray, flag: xr.DataArray | None):
    """Regions of the (frequency, polarization, m, l) cube to load at once.

    For dask backed data the regions follow the chunk boundaries shared by
    the image and its flags, so each chunk of either is computed exactly once;
    other data is loaded one frequency plane at a time.
    """
    chunked = [x.data for x in (sky, flag) if x is not None and x.chunks is not None]
    edges = []
    for axis, size in enumerate(sky.shape):
        if chunked:
            common = set.intersection(
                *({0, *np.cumsum(x.chunks[axis]).tolist()} for x in chunked)
            )
        elif axis == 0:
            common = set(range(size + 1))
        else:
            common = {0, size}
        edges.append(sorted(common))
    for region in itertools.product(
        *(list(zip(e[:-1], e[1:], strict=False)) for e in edges)
    ):
        yield tuple(slice(start, stop) for start, stop in region)


def _put_block(
    fobj,
    data_offset: int,
    dtype: np.dtype,
    cube_shape: tuple,
    slices: tuple,
    sky_block: np.ndarray,
    flag_block: np.ndarray | None,
    fits_position: np.ndarray,
) -> None:
    """Write one region into the data area of the primary HDU; flagged pixels
    become NaN (the FITS blanking convention for floating point data).

    A region of whole planes is written plane by plane (each plane is
    contiguous in the file). Part of a plane is written through a memory map
    of that plane only, so that the written pages leave the process' resident
    memory when the plane is unmapped.
    """
    chan_slice, pol_slice, m_slice, l_slice = slices
    npol, nm, nl = cube_shape[1:]
    plane_bytes = nm * nl * dtype.itemsize
    whole_planes = m_slice == slice(0, nm) and l_slice == slice(0, nl)
    for i, chan in enumerate(range(chan_slice.start, chan_slice.stop)):
        for j, pol in enumerate(range(pol_slice.start, pol_slice.stop)):
            offset = data_offset + (chan * npol + int(fits_position[pol])) * plane_bytes
            if whole_planes:
                # big endian copy of one plane
                plane = np.array(sky_block[i, j], dtype=dtype, order="C")
                if flag_block is not None:
                    plane[flag_block[i, j]] = np.nan
                fobj.seek(offset)
                fobj.write(plane.data)
                continue
            plane = np.memmap(
                fobj, dtype=dtype, mode="r+", offset=offset, shape=(nm, nl)
            )
            target = plane[m_slice, l_slice]
            target[...] = sky_block[i, j]
            if flag_block is not None:
                target[flag_block[i, j]] = np.nan
            del target, plane


def _xds_to_fits_image(
    xds: xr.Dataset, image_store_name: str, *, prepared: tuple | None = None
) -> None:
    """Write a single image dataset to a FITS file.

    The dataset holds the data variable SKY (dimensions time, frequency,
    polarization, l, m, with a single time plane) and optionally FLAG and
    BEAM_FIT_PARAMS. The file follows CASA's exportfits conventions, so CASA
    and the xradio FITS reader can read it:

    * the complete header is built and validated before the file is created,
    * flagged pixels are written as NaN, following FITS convention,
    * polarizations are written as FITS STOKES codes on a linear axis; when
      the dataset's order is not linear in those codes (the Jones order of
      correlations, for example RR, RL, LR, LL) the planes, their flags and
      per plane beams are written in an order that is (RR, LL, RL, LR), and
      the FITS reader restores the canonical (Jones) order,
    * one beam for all planes is written as BMAJ/BMIN/BPA cards, per plane
      beams as a CASA style BEAMS binary table extension,
    * pixels are written region by region, each region made of the dask
      chunks of SKY and FLAG that share boundaries: every chunk is computed
      once, by the active dask scheduler in batches of regions of up to
      :data:`xradio.image._util._blocks.BATCH_BYTES`, and memory holds one
      batch plus one plane (data that is not dask backed is written one
      frequency plane at a time).

    Parameters
    ----------
    xds : xr.Dataset
        Single image dataset. It is not modified.
    image_store_name : str
        Path of the FITS file to create. It must not exist.
    prepared : tuple, optional
        The result of :func:`_fits_image_header` for ``xds``, for callers
        that validated the header before (it is built here otherwise).

    Raises
    ------
    FileExistsError
        If ``image_store_name`` exists.
    RuntimeError, ValueError
        If the image cannot be represented in FITS (raised before the file
        is created).
    """
    header, beams_hdu, pol_order = (
        _fits_image_header(xds) if prepared is None else prepared
    )
    path = os.path.expanduser(os.fspath(image_store_name))
    dtype = np.dtype(">f4" if header["BITPIX"] == -32 else ">f8")
    sky = xds["SKY"].isel(time=0).transpose(*_FITS_DIMS)
    flag = (
        xds["FLAG"].isel(time=0).transpose(*_FITS_DIMS)
        if "FLAG" in xds.data_vars
        else None
    )
    header_bytes = header.tostring().encode("ascii")
    data_bytes = int(np.prod(sky.shape)) * dtype.itemsize
    padded_bytes = -(-data_bytes // _FITS_BLOCK_BYTES) * _FITS_BLOCK_BYTES
    # dataset polarization index -> FITS polarization pixel
    fits_position = np.argsort(pol_order)

    # "x" never replaces an existing file (FileExistsError, nothing touched)
    fobj = open(path, "xb")
    try:
        with fobj:
            fobj.write(header_bytes)
            # the data area (zero padded to a whole FITS block) is filled below
            fobj.truncate(len(header_bytes) + padded_bytes)
        with open(path, "r+b") as data_file:
            reader = RegionReader([sky, flag])
            for region, (sky_block, flag_block) in reader.regions(
                _block_slices(sky, flag)
            ):
                _put_block(
                    data_file,
                    len(header_bytes),
                    dtype,
                    sky.shape,
                    region,
                    sky_block,
                    None if flag_block is None else np.asarray(flag_block, dtype=bool),
                    fits_position,
                )
        if beams_hdu is not None:
            fits.append(path, beams_hdu.data, header=beams_hdu.header, verify=False)
    except BaseException:
        with contextlib.suppress(OSError):
            os.remove(path)
        raise
