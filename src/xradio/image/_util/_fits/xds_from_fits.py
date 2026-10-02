"""
Read FITS images into xradio image datasets.

The reader interprets the WCS cards of the primary HDU (FITS WCS papers I to
III, plus the AIPS conventions that casacore also reads) and a CASA style
``BEAMS`` binary table of per-plane restoring beams. Pixels are read lazily, in
dask chunks, from the memory mapped primary HDU. Only astropy is used for FITS
access: reading FITS images needs neither python-casacore nor casatools.
"""

import copy
import re
import warnings

import dask
import dask.array as da
import numpy as np
import xarray as xr
from astropy import units as u
from astropy.io import fits
from astropy.time import Time
from erfa import ErfaWarning

from xradio._utils.coord_math import _deg_to_rad
from xradio._utils.dict_helpers import (
    make_direction_location_dict,
    make_quantity,
    make_skycoord_dict,
    make_spectral_coord_reference_dict,
    make_time_measure_dict,
)
from xradio._utils.list_and_array import to_python_type
from xradio._utils.logging import xradio_logger
from xradio.image._util import conventions
from xradio.image._util.common import (
    _DEFAULT_CHANNEL_WIDTH_HZ,
    _DEFAULT_FREQUENCY_HZ,
    _DEFAULT_REST_FREQUENCY_HZ,
    _c,
    _compute_linear_world_values,
    _compute_velocity_values,
    _compute_world_sph_dims,
    _convert_beam_to_rad,
    _freq_from_vel,
    _get_unit,
    _get_xds_dim_order,
    _image_type,
    _l_m_attr_notes,
    _to_canonical_polarization_order,
)

# from xradio.image._util._casacore.common import (
#     _image_flag,
#     _beam_fit_params,
# )
_image_flag = "flag"
_beam_fit_params = "beam_fit_params"

#: FITS ``BITPIX`` -> numpy dtype of the pixel values (FITS standard 4.0,
#: Table 8). The file stores them big-endian; the reader returns them in
#: native byte order.
_BITPIX_DTYPES = {
    8: "uint8",
    16: "int16",
    32: "int32",
    64: "int64",
    -32: "float32",
    -64: "float64",
}

#: Spectral axis types (the first four characters of ``CTYPEi``) that the
#: reader converts to frequency: FITS WCS Paper III frequency, optical and
#: radio velocity, and the AIPS optical velocity axis ``FELO`` (sampled
#: linearly in frequency, as casacore reads it).
_SPECTRAL_AXIS_TYPES = ("FREQ", "VOPT", "VRAD", "FELO")

#: Spectral axis types of FITS WCS Paper III and AIPS that the reader does not
#: support, named in its error message.
_OTHER_SPECTRAL_AXIS_TYPES = ("VELO", "WAVE", "AWAV", "WAVN", "ZOPT", "BETA", "ENER")

#: AIPS spectral frame suffixes of ``CTYPEi`` (``"FREQ-LSR"``, ``"FELO-HEL"``)
#: -> casacore frame, as casacore reads them (``FITSSpectralUtil::frameFromTag``).
_CTYPE_FRAME_TAGS = {
    "LSR": "LSRK",
    "LSRK": "LSRK",
    "HEL": "BARY",
    "OBS": "TOPO",
    "LSD": "LSRD",
    "GEO": "GEO",
    "SOU": "REST",
    "REST": "REST",
    "GAL": "GALACTO",
}

#: AIPS ``VELREF`` frame codes (``VELREF`` modulo 256; casacore writes and
#: reads 4 to 7 as a non-standard extension) -> casacore frame.
_VELREF_FRAMES = {
    1: "LSRK",
    2: "BARY",
    3: "TOPO",
    4: "LSRD",
    5: "GEO",
    6: "REST",
    7: "GALACTO",
}

#: FITS ``TIMESYS`` -> astropy time scale.
_TIMESYS_SCALES = {**conventions.CASACORE_EPOCH_REF_TO_SCALE, "GMT": "utc"}

#: FITS ``RADESYS`` -> astropy sky frame.
_RADESYS_FRAMES = {
    "ICRS": "icrs",
    "FK5": "fk5",
    "FK4": "fk4",
    "FK4-NO-E": "fk4noterms",
}

#: Spellings of spectral axis units that astropy does not parse, as written by
#: some FITS writers.
_FITS_UNIT_ALIASES = {
    "HZ": "Hz",
    "KHZ": "kHz",
    "MHZ": "MHz",
    "GHZ": "GHz",
    "THZ": "THz",
    "M/S": "m/s",
    "KM/S": "km/s",
}

#: Observation date (MJD, UTC) of images whose header has neither a usable
#: ``DATE-OBS`` nor ``MJD-OBS``. MJD 0 (1858-11-17) is the value casacore uses
#: for an unset observation date (casacore does not write it to FITS).
_UNKNOWN_OBSDATE_MJD = 0.0

#: Rest frequency (Hz) recorded when the header of an image with a frequency
#: axis has none: casacore's value for an unknown rest frequency. Such images
#: have no velocity coordinate.
_UNKNOWN_REST_FREQUENCY_HZ = 0.0


def _fits_image_to_xds(
    img_full_path: str,
    chunks: dict,
    verbose: bool,
    do_sky_coords: bool,
    compute_mask: bool,
    image_type: str = "SKY",
) -> xr.Dataset:
    """
    Read a FITS image into an image dataset whose pixels are read lazily.

    Parameters
    ----------
    img_full_path : str
        Path to the FITS file.
    chunks : dict
        Dask chunk lengths by dimension ('l', 'm', 'frequency',
        'polarization'); dimensions without a key are read in one chunk.
    verbose : bool
        Emit debugging messages.
    do_sky_coords : bool
        Add the sky coordinates of each pixel (``right_ascension`` and
        ``declination``) as non-dimensional coordinates.
    compute_mask : bool
        If True, scan the pixels for NaNs and, if there are any, add a
        ``FLAG_<image_type>`` variable that flags them. If False, skip the scan
        and the flag for performance. It is then solely the responsibility of
        the user to ensure downstream apps can handle NaN values; do not ask
        package developers to add this non-standard behavior.
    image_type : str, optional
        Name of the image data variable, by default ``"SKY"``.

    Returns
    -------
    xr.Dataset
        The image dataset. Its polarization axis is in canonical (Jones
        matrix) order, see
        :func:`xradio.image._util.conventions.canonical_polarization_order`.
        A FITS ``STOKES`` axis must be a linear sequence of FITS codes, so FITS
        files store four correlations as RR, LL, RL, LR (or XX, YY, XY, YX);
        the reader reorders those planes, with their flags and per-plane
        beams, into RR, RL, LR, LL (XX, XY, YX, YY).
    """
    # may also need to pass mode='denywrite'
    # https://stackoverflow.com/questions/35759713/astropy-io-fits-read-row-from-large-fits-file-with-mutliple-hdus
    with fits.open(img_full_path, memmap=True) as hdulist:
        attrs, helpers, header = _fits_header_to_xds_attrs(hdulist, compute_mask)
    xds = _create_coords(helpers, header, do_sky_coords)
    sphr_dims = helpers["sphr_dims"]
    ary = _read_image_array(img_full_path, chunks, helpers, verbose)
    dim_order = _get_xds_dim_order(sphr_dims, image_type)
    if len(dim_order) < ary.ndim:
        # an image type without l and m (the sum of weights, which tclean and
        # casacore's FITS export store with 1 x 1 pixel direction axes)
        ary, xds = _drop_direction_axes(ary, xds, img_full_path, image_type)
    xds = _add_sky_or_aperture(
        xds, ary, dim_order, header, helpers, sphr_dims, image_type
    )
    xds.attrs = attrs
    xds = _add_coord_attrs(xds, helpers)
    if helpers["has_multibeam"]:
        xds = _do_multibeam(xds, img_full_path, image_type=image_type)
        xds[image_type.upper()].attrs[_beam_fit_params] = (
            "BEAM_FIT_PARAMS_" + image_type.upper()
        )
        xds["BEAM_FIT_PARAMS_" + image_type.upper()].attrs["type"] = (
            "beam_fit_params_" + image_type.lower()
        )
    elif "beam" in helpers and helpers["beam"] is not None:
        xds = _add_beam(xds, helpers, image_type)
        xds[image_type.upper()].attrs[_beam_fit_params] = (
            "BEAM_FIT_PARAMS_" + image_type.upper()
        )
        xds["BEAM_FIT_PARAMS_" + image_type.upper()].attrs["type"] = (
            "beam_fit_params_" + image_type.lower()
        )
    return _to_canonical_polarization_order(xds)


def _drop_direction_axes(
    ary: da.Array, xds: xr.Dataset, img_full_path: str, image_type: str
) -> tuple[da.Array, xr.Dataset]:
    """
    Remove the direction axes of an image type that has none.

    The sum of weights (``VISIBILITY_NORMALIZATION``) has dimensions time,
    frequency and polarization only, but FITS images of it (for example a
    tclean ``.sumwt`` exported by CASA) have direction axes of one pixel. As
    the CASA reader does for such images, those axes are squeezed out and the
    coordinates along them dropped.

    Parameters
    ----------
    ary : dask.array.Array
        Pixels with dimensions (time, frequency, polarization, l, m).
    xds : xr.Dataset
        Coordinates of the image.
    img_full_path : str
        Path of the FITS file, for the error message.
    image_type : str
        Image type, for the error message.

    Returns
    -------
    tuple of (dask.array.Array, xr.Dataset)
        The pixels with dimensions (time, frequency, polarization) and the
        coordinates without those along l and m.

    Raises
    ------
    ValueError
        If the direction axes are longer than one pixel.
    """
    if ary.shape[3:] != (1, 1):
        raise ValueError(
            f"The {image_type} image {img_full_path} must have direction axes of "
            f"one pixel, found {ary.shape[3:]} pixels"
        )
    along_direction = [
        name for name, coord in xds.coords.items() if {"l", "m"} & set(coord.dims)
    ]
    return ary[:, :, :, 0, 0], xds.drop_vars(along_direction)


def _add_coord_attrs(xds: xr.Dataset, helpers: dict) -> xr.Dataset:
    xds = _add_time_attrs(xds, helpers)
    xds = _add_freq_attrs(xds, helpers)
    xds = _add_vel_attrs(xds, helpers)
    xds = _add_l_m_attrs(xds, helpers)
    xds = _add_lin_attrs(xds, helpers)
    return xds


def _add_time_attrs(xds: xr.Dataset, helpers: dict) -> xr.Dataset:
    time_coord = xds.coords["time"]
    time_coord.attrs = copy.deepcopy(helpers["obsdate"]["attrs"])
    xds.assign_coords(time=time_coord)
    return xds


def _add_freq_attrs(xds: xr.Dataset, helpers: dict) -> xr.Dataset:
    freq_coord = xds.coords["frequency"]
    spectral = helpers["spectral"]
    frame = spectral["frame"]
    meta = {}
    meta["rest_frequency"] = make_quantity(spectral["rest_frequency"], "Hz")
    meta["type"] = "spectral_coord"
    meta["units"] = "Hz"
    # the casacore name of the frame, e.g. "BARY" for SPECSYS = 'BARYCENT'
    meta["frame"] = frame
    meta["wave_units"] = "mm"
    reference_frequency = make_spectral_coord_reference_dict(
        spectral["reference_frequency"], "Hz", frame
    )
    reference_frequency["attrs"]["observer"] = conventions.spectral_frame_to_observer(
        frame
    )
    meta["reference_frequency"] = reference_frequency
    meta["channel_width"] = make_quantity(spectral["channel_width"], "Hz")
    freq_coord.attrs = copy.deepcopy(meta)
    xds["frequency"] = freq_coord
    return xds


def _add_vel_attrs(xds: xr.Dataset, helpers: dict) -> xr.Dataset:
    if "velocity" not in xds.coords:
        # no rest frequency, so no velocities
        return xds
    vel_coord = xds.coords["velocity"]
    meta = {"units": "m/s"}
    meta["doppler_type"] = helpers["spectral"]["doppler_type"]
    meta["type"] = "doppler"
    vel_coord.attrs = copy.deepcopy(meta)
    xds.coords["velocity"] = vel_coord
    return xds


def _add_l_m_attrs(xds: xr.Dataset, helpers: dict) -> xr.Dataset:
    attr_note = _l_m_attr_notes()
    for c in ["l", "m"]:
        if c in xds.coords:
            xds[c].attrs = {
                "note": attr_note[c],
            }
    return xds


def _add_lin_attrs(xds: xr.Dataset, helpers: dict) -> xr.Dataset:
    if not helpers["sphr_dims"]:
        for i, j in zip(helpers["dir_axes"], ("u", "v"), strict=False):
            meta = {
                "units": "wavelengths",
                "crval": helpers["crval"][i],
                "cdelt": helpers["cdelt"][i],
            }
            xds.coords[j].attrs = meta
    return xds


def _spectral_ctype_tag(ctype: str) -> str:
    """Return the AIPS frame suffix of a spectral ``CTYPE`` (``"LSR"`` for
    ``"FREQ-LSR"``), or ``""`` if it has none."""
    return ctype[4:].replace("-", "").strip().upper()


def _is_freq_like(v: str) -> bool:
    """Return whether a ``CTYPE`` is a spectral axis the reader supports: one
    of :data:`_SPECTRAL_AXIS_TYPES`, alone or with an AIPS frame suffix (but
    not with a FITS WCS Paper III algorithm code such as ``"VOPT-F2W"``)."""
    tag = _spectral_ctype_tag(v)
    return v[:4].upper() in _SPECTRAL_AXIS_TYPES and (
        tag == "" or tag in _CTYPE_FRAME_TAGS
    )


def _equinox_year(value) -> float:
    """Return the year of a FITS ``EQUINOX`` (or ``EPOCH``) value, a number
    such as 2000.0 or a string such as ``"J2000"``."""
    if isinstance(value, str):
        match = re.fullmatch(r"\s*[JjBb]?\s*(\d+(?:\.\d*)?)\s*", value)
        if match is None:
            raise RuntimeError(f"Cannot interpret the FITS EQUINOX {value!r}")
        return float(match.group(1))
    return float(value)


def _direction_reference_frame(header) -> tuple[str, str | None]:
    """
    Return the sky frame and equinox of the image's celestial axes.

    Missing values take their FITS WCS Paper II defaults: ``RADESYS`` (or the
    deprecated ``RADECSYS``) is ICRS without an equinox, FK4 for an equinox
    before 1984 and FK5 otherwise; ``EQUINOX`` (or the deprecated ``EPOCH``)
    is 2000.0 for FK5 and 1950.0 for FK4.

    Parameters
    ----------
    header : astropy.io.fits.Header
        Primary HDU header.

    Returns
    -------
    frame : str
        Astropy frame name, for example ``"fk5"``.
    equinox : str or None
        Equinox as the CASA reader spells it, ``"j2000.0"`` for FK5 (Julian)
        and ``"b1950.0"`` for FK4 (Besselian), or None for ICRS.

    Raises
    ------
    RuntimeError
        For a reference system without an astropy frame (e.g. GAPPT).
    """
    equinox = header.get("EQUINOX", header.get("EPOCH"))
    year = None if equinox is None else _equinox_year(equinox)
    radesys = str(header.get("RADESYS", header.get("RADECSYS", ""))).strip().upper()
    if not radesys:
        if year is None:
            radesys = "ICRS"
        else:
            radesys = "FK4" if year < 1984.0 else "FK5"
    frame = _RADESYS_FRAMES.get(radesys)
    if frame is None:
        raise RuntimeError(
            f"Unsupported FITS RADESYS {radesys!r}; supported reference systems "
            f"are {', '.join(_RADESYS_FRAMES)}"
        )
    if frame == "icrs":
        return frame, None
    if frame == "fk5":
        return frame, f"j{2000.0 if year is None else year:.1f}"
    return frame, f"b{1950.0 if year is None else year:.1f}"


def _projection_parameters(header, lat_axis: int, projection: str) -> list[float]:
    """
    Return the projection parameters ``PVi_m`` of the latitude axis.

    Parameters
    ----------
    header : astropy.io.fits.Header
        Primary HDU header.
    lat_axis : int
        FITS (1-based) number of the latitude axis.
    projection : str
        Projection code, for example ``"SIN"``.

    Returns
    -------
    list of float
        ``PVi_1, PVi_2, ...`` (``PVi_0, PVi_1, ...`` for ZPN), the order
        casacore uses, with 0.0 for parameters missing in between; ``[0.0,
        0.0]`` when the header has none.
    """
    values = {}
    for key in header.keys():
        match = re.fullmatch(r"PV0?(\d+)_0?(\d+)", key)
        if match is not None and int(match.group(1)) == lat_axis:
            values[int(match.group(2))] = float(header[key])
    if not values:
        return [0.0, 0.0]
    first = 0 if projection == "ZPN" else 1
    return [values.get(m, 0.0) for m in range(first, max(values) + 1)]


def _pc_keys(i: int, j: int) -> tuple[str, str, str]:
    """``PCi_j`` and its legacy spellings ``PC0i_0j`` and ``PC00i00j``."""
    return (f"PC{i}_{j}", f"PC0{i}_0{j}", f"PC{i:03d}{j:03d}")


def _pc_matrix(header, lon_axis: int, lat_axis: int, cdelt: list) -> list:
    """
    Return the PC matrix of the celestial axes.

    Missing elements default to the identity matrix (FITS WCS Paper I). A
    header without any PC card of the celestial axes may give the rotation of
    the old AIPS convention, ``CROTAi`` on the latitude axis, which is
    converted to a PC matrix as FITS WCS Paper II (section 6.1) prescribes.

    Parameters
    ----------
    header : astropy.io.fits.Header
        Primary HDU header.
    lon_axis, lat_axis : int
        FITS (1-based) numbers of the longitude and latitude axes.
    cdelt : list of float
        ``CDELTi`` of every axis (0-based list).

    Returns
    -------
    list of list of float
        ``[[PC_lon_lon, PC_lon_lat], [PC_lat_lon, PC_lat_lat]]``.
    """
    axes = (lon_axis, lat_axis)
    pc = np.eye(2)
    has_pc = False
    for i in (0, 1):
        for j in (0, 1):
            for key in _pc_keys(axes[i], axes[j]):
                if key in header:
                    pc[i, j] = float(header[key])
                    has_pc = True
                    break
    crota = float(header.get(f"CROTA{lat_axis}", 0.0))
    if not has_pc and crota != 0.0:
        rho = crota * _deg_to_rad
        ratio = cdelt[lat_axis - 1] / cdelt[lon_axis - 1]
        pc = np.array(
            [
                [np.cos(rho), -np.sin(rho) * ratio],
                [np.sin(rho) / ratio, np.cos(rho)],
            ]
        )
    return to_python_type(pc)


def _native_pole(header, t_axes) -> list[float]:
    """
    Return ``LONPOLE`` and ``LATPOLE`` in degrees.

    When the header lacks them, wcslib (through astropy) computes their FITS
    WCS Paper II defaults, as casacore does when it reads a FITS image.

    Parameters
    ----------
    header : astropy.io.fits.Header
        Primary HDU header.
    t_axes : array_like of int
        FITS (1-based) numbers of the longitude and latitude axes.

    Returns
    -------
    list of float
        ``[LONPOLE, LATPOLE]``.
    """
    if "LONPOLE" in header and "LATPOLE" in header:
        return [float(header["LONPOLE"]), float(header["LATPOLE"])]
    from astropy.wcs import WCS

    celestial = fits.Header()
    for new, old in ((1, int(t_axes[0])), (2, int(t_axes[1]))):
        for key in ("CTYPE", "CRVAL", "CDELT", "CRPIX", "CUNIT"):
            if f"{key}{old}" in header:
                celestial[f"{key}{new}"] = header[f"{key}{old}"]
    for key in ("LONPOLE", "LATPOLE"):
        if key in header:
            celestial[key] = header[key]
    for key in header.keys():
        match = re.fullmatch(r"PV0?(\d+)_0?(\d+)", key)
        if match is not None and int(match.group(1)) == int(t_axes[1]):
            celestial[f"PV2_{int(match.group(2))}"] = header[key]
    with warnings.catch_warnings():
        # wcslib "fixes" are irrelevant for the native pole
        warnings.simplefilter("ignore")
        wcs = WCS(celestial)
        wcs.wcs.set()
    return [float(wcs.wcs.lonpole), float(wcs.wcs.latpole)]


def _xds_coordinate_system_info_attrs_from_header(helpers: dict, header) -> dict:
    # helpers is modified in place, headers is not modified
    t_axes = helpers["t_axes"]
    p0 = header[f"CTYPE{t_axes[0]}"][-3:]
    p1 = header[f"CTYPE{t_axes[1]}"][-3:]
    if p0 != p1:
        raise RuntimeError(
            f"Projections for direction axes ({p0}, {p1}) differ, but they "
            "must be the same"
        )
    coordinate_system_info = {}
    coordinate_system_info["projection"] = p0
    helpers["projection"] = p0
    ref_sys, ref_eqx = _direction_reference_frame(header)
    helpers["ref_sys"] = ref_sys
    helpers["ref_eqx"] = ref_eqx
    dir_axes = helpers["dir_axes"]
    ddata = []
    dunits = []
    for i in dir_axes:
        x = helpers["crval"][i] * u.Unit(_get_unit(helpers["cunit"][i]))
        x = x.to("rad")
        ddata.append(x.value)
        dunits.append("rad")
    # fits does not support conversion frames
    coordinate_system_info["reference_direction"] = make_skycoord_dict(
        ddata, units=dunits, frame=ref_sys
    )
    if ref_eqx is not None:
        coordinate_system_info["reference_direction"]["attrs"]["equinox"] = ref_eqx

    native_pole = _native_pole(header, t_axes)
    coordinate_system_info["native_pole_direction"] = make_direction_location_dict(
        [x * _deg_to_rad for x in native_pole],
        units="rad",
        frame="native_projection",
    )

    # dir_axes are now 0-based, but fits needs 1-based
    coordinate_system_info["pixel_coordinate_transformation_matrix"] = _pc_matrix(
        header, int(dir_axes[0]) + 1, int(dir_axes[1]) + 1, helpers["cdelt"]
    )
    coordinate_system_info["projection_parameters"] = _projection_parameters(
        header, int(t_axes[1]), p0
    )
    # the sky coordinates of the pixels (_create_coords) need all of these
    helpers["native_pole"] = native_pole
    helpers["pc"] = coordinate_system_info["pixel_coordinate_transformation_matrix"]
    helpers["projection_parameters"] = coordinate_system_info["projection_parameters"]
    return coordinate_system_info


def _default_axis_unit(ctype: str) -> str:
    """Unit of an axis without ``CUNITi``: degrees for celestial axes (FITS
    WCS Paper II), Hz for frequency and m/s for velocity axes (Paper III)."""
    if ctype[:4].upper() == "FREQ":
        return "Hz"
    if ctype[:4].upper() in _SPECTRAL_AXIS_TYPES:
        return "m/s"
    if ctype.upper() == "STOKES":
        return ""
    return "deg"


def _fits_header_c_values_to_metadata(helpers: dict, header) -> None:
    # The helpers dict is modified in place. header is not modified
    ctypes = []
    shape = []
    crval = []
    cdelt = []
    crpix = []
    cunit = []
    for i in range(1, helpers["naxes"] + 1):
        ax_type = header[f"CTYPE{i}"]
        ctypes.append(ax_type)
        shape.append(header[f"NAXIS{i}"])
        crval.append(header[f"CRVAL{i}"])
        if f"CDELT{i}" not in header and any(
            re.fullmatch(r"CD\d+_\d+", key) for key in header.keys()
        ):
            raise RuntimeError(
                "FITS images whose coordinates are given by a CDi_j matrix "
                "instead of CDELTi (and PCi_j) are not supported"
            )
        cdelt.append(header[f"CDELT{i}"])
        # FITS 1-based to python 0-based
        crpix.append(header[f"CRPIX{i}"] - 1)
        unit = str(header.get(f"CUNIT{i}", "")).strip()
        cunit.append(unit if unit else _default_axis_unit(ax_type))
    helpers["shape"] = shape
    helpers["ctype"] = ctypes
    helpers["crval"] = crval
    helpers["cdelt"] = cdelt
    helpers["crpix"] = crpix
    helpers["cunit"] = cunit


def _get_telescope_metadata(helpers: dict, header) -> dict:
    # The helpers dict is modified in place. header is not modified
    tel = {}
    # casacore's name for an unknown telescope
    tel["name"] = str(header.get("TELESCOP", "")).strip() or "UNKNOWN"
    if all(k in header for k in ("OBSGEO-X", "OBSGEO-Y", "OBSGEO-Z")):
        x = header["OBSGEO-X"]
        y = header["OBSGEO-Y"]
        z = header["OBSGEO-Z"]
        xyz = np.array([x, y, z])
        r = np.sqrt(np.sum(xyz * xyz))
        lat = np.arcsin(z / r)
        long = np.arctan2(y, x)
        tel["direction"] = {
            "attrs": {
                "coordinate_system": "geocentric",
                # I haven't seen a FITS keyword for reference frame of telescope posiiton
                "frame": "ITRF",
                "origin_object_name": "earth",
                "type": "location",
                "units": "rad",
            },
            "data": [long, lat],
            "dims": ["ellipsoid_dir_label"],
            "coords": {
                "ellipsoid_dir_label": {
                    "dims": ["ellipsoid_dir_label"],
                    "data": ["lon", "lat"],
                }
            },
        }
        tel["distance"] = {
            "attrs": {
                "coordinate_system": "geocentric",
                # I haven't seen a FITS keyword for reference frame of telescope posiiton
                "frame": "ITRF",
                "origin_object_name": "earth",
                "type": "location",
                "units": "m",
            },
            "data": [r],
            "dims": ["ellipsoid_dis_label"],
            "coords": {
                "ellipsoid_dis_label": {
                    "dims": ["ellipsoid_dis_label"],
                    "data": [
                        "dist",
                    ],
                }
            },
        }
    return tel


def _compute_pointing_center(helpers: dict, header) -> dict:
    # Neither helpers or header is modified
    if "OBSRA" in header and "OBSDEC" in header:
        # The pointing center, in degrees, as casacore writes it
        pc_long = float(header["OBSRA"]) * _deg_to_rad
        pc_lat = float(header["OBSDEC"]) * _deg_to_rad
    else:
        t_axes = helpers["t_axes"]
        unit = [u.Unit(_get_unit(helpers["cunit"][i - 1])) for i in t_axes]
        pc_long = float(header[f"CRVAL{t_axes[0]}"]) * unit[0]
        pc_lat = float(header[f"CRVAL{t_axes[1]}"]) * unit[1]
        pc_long = pc_long.to(u.rad).value
        pc_lat = pc_lat.to(u.rad).value
    return make_skycoord_dict([pc_long, pc_lat], units="rad", frame=helpers["ref_sys"])


def _user_attrs_from_header(header) -> dict:
    # header is not modified
    # Cards the reader interprets (or that describe the file layout) are
    # excluded, so that writers do not copy stale values of them
    exclude = [
        "ALTRPIX",
        "ALTRVAL",
        "BITPIX",
        "BLANK",
        "BMAJ",
        "BMIN",
        "BPA",
        "BSCALE",
        "BTYPE",
        "BUNIT",
        "BZERO",
        "CASAMBM",
        "CHECKSUM",
        "DATASUM",
        "DATE",
        "DATE-OBS",
        "EPOCH",
        "EQUINOX",
        "EXTEND",
        "HISTORY",
        "LATPOLE",
        "LONPOLE",
        "MJD-OBS",
        "OBSDEC",
        "OBSERVER",
        "OBSRA",
        "ORIGIN",
        "TELESCOP",
        "OBJECT",
        "RADECSYS",
        "RADESYS",
        "RESTFREQ",
        "RESTFRQ",
        "RESTWAV",
        "SIMPLE",
        "SPECSYS",
        "TIMESYS",
        "VELREF",
    ]
    regex = r"|".join(
        [
            r"^NAXIS\d?$",
            r"^CRVAL\d$",
            r"^CRPIX\d$",
            r"^CTYPE\d$",
            r"^CDELT\d$",
            r"^CUNIT\d$",
            r"^OBSGEO-(X|Y|Z)$",
            r"^P(C|V)0?\d_0?\d",
            r"^PC\d{6}$",
            r"^CD\d_\d$",
            r"^CROTA\d$",
        ]
    )
    user = {}
    for k, v in header.items():
        if not (re.search(regex, k) or k in exclude):
            user[k.lower()] = v
    return user


def _beam_attr_from_header(helpers: dict, header) -> dict | str | None:
    # The helpers dict is modified in place. header is not modified
    helpers["has_multibeam"] = False
    if "BMAJ" in header:
        # single global beam
        beam = {
            "bmaj": make_quantity(header["BMAJ"], "deg"),
            "bmin": make_quantity(header["BMIN"], "deg"),
            "pa": make_quantity(header["BPA"], "deg"),
        }
        return _convert_beam_to_rad(beam)
    elif "CASAMBM" in header and header["CASAMBM"]:
        # multi-beam
        helpers["has_multibeam"] = True
        return "mb"
    else:
        # no beam
        return None


def _create_dim_map(helpers: dict, header) -> dict:
    # The helpers dict is modified in place. header is not modified
    t_axes = np.array([0, 0])
    dim_map = {}
    helpers["has_freq"] = False
    # fits indexing starts at 1, not 0
    for i in range(1, helpers["naxes"] + 1):
        ax_type = header[f"CTYPE{i}"]
        if ax_type.startswith("RA-"):
            t_axes[0] = i
        elif ax_type.startswith("DEC-"):
            t_axes[1] = i
        elif ax_type == "STOKES":
            dim_map["polarization"] = i - 1
        elif _is_freq_like(ax_type):
            dim_map["freq"] = i - 1
            helpers["has_freq"] = True
        elif ax_type[:4].upper() in _SPECTRAL_AXIS_TYPES + _OTHER_SPECTRAL_AXIS_TYPES:
            raise RuntimeError(
                f"{ax_type} is an unsupported spectral axis; supported spectral "
                f"axes are {', '.join(_SPECTRAL_AXIS_TYPES)}, optionally with an "
                f"AIPS frame suffix such as '-LSR'"
            )
        else:
            raise RuntimeError(f"{ax_type} is an unsupported axis")
    helpers["t_axes"] = t_axes
    helpers["dim_map"] = dim_map
    return dim_map


def _dtype_from_bitpix(header) -> str:
    """Return the numpy dtype (native byte order) of the primary HDU pixels."""
    bitpix = header["BITPIX"]
    if bitpix not in _BITPIX_DTYPES:
        raise RuntimeError(f"Unhandled data type {bitpix}")
    return _BITPIX_DTYPES[bitpix]


def _fits_header_to_xds_attrs(
    hdulist: fits.hdu.hdulist.HDUList, compute_mask: bool
) -> tuple:
    # First: Guard for unsupported compressed images
    for i, hdu in enumerate(hdulist):
        if isinstance(hdu, fits.CompImageHDU):
            raise RuntimeError(
                f"HDU {i}, name={hdu.name} is a CompImageHDU, which is not supported "
                "for memory-mapping. "
                "Cannot memory-map compressed FITS image (CompImageHDU). "
                "Workaround: decompress the FITS using tools like `funpack`, `cfitsio`, "
                "or Astropy's `.scale()`/`.copy()` workflows"
            )
    primary = None
    ignored = []
    for hdu in hdulist:
        if hdu.name == "PRIMARY":
            primary = hdu
            # Memory map support check
            # avoid possibly non-existent hdu.scale_type attribute check and check header instead
            header = hdu.header
            scale = hdu.header.get("BSCALE", 1.0)
            zero = hdu.header.get("BZERO", 0.0)
            if not (scale == 1.0 and zero == 0.0):
                raise RuntimeError(
                    "Cannot memory-map scaled FITS data (BSCALE/BZERO set). "
                    f"BZERO={zero}, BSCALE={scale}. "
                    "Workaround: remove scaling with Astropy's"
                    "  `HDU.data = HDU.data * BSCALE + BZERO` and save a new file"
                )
            # NOTE: check for primary.data size being too large removed, since
            # data is read in chunks, so no danger of exhausting memory
            # NOTE: sanity-check for ndarray type has been removed to avoid
            # forcing eager memory load of possibly very large data array.
        elif hdu.name == "BEAMS":
            # per-plane beams, read by _do_multibeam if CASAMBM is set
            pass
        else:
            ignored.append(hdu.name)
    if primary is None:
        raise RuntimeError("No PRIMARY HDU found in fits file")
    header = primary.header
    if not primary.is_image or header.get("NAXIS", 0) < 2:
        raise RuntimeError(
            "The primary HDU of the FITS file holds no image (it has fewer than "
            "two axes, or holds random groups data such as UVFITS); only images "
            "in the primary HDU are supported"
        )
    for name in ignored:
        xradio_logger().warning(
            f"Ignoring the FITS extension HDU {name!r}: only the image in the "
            "primary HDU and a BEAMS table of per-plane beams are read"
        )
    helpers = {}
    attrs = {}
    naxes = header["NAXIS"]
    helpers["naxes"] = naxes
    dim_map = _create_dim_map(helpers, header)
    _fits_header_c_values_to_metadata(helpers, header)
    t_axes = helpers["t_axes"]
    if (t_axes > 0).all():
        dir_axes = t_axes[:]
        dir_axes = dir_axes - 1
        helpers["dir_axes"] = dir_axes
        dim_map["l"] = dir_axes[0]
        dim_map["m"] = dir_axes[1]
        helpers["dim_map"] = dim_map
    else:
        raise RuntimeError("Could not find both direction axes")
    attrs["coordinate_system_info"] = _xds_coordinate_system_info_attrs_from_header(
        helpers, header
    )
    helpers["polarization"] = _get_pol_values(helpers)
    helpers["spectral"] = _spectral_metadata(helpers, header)
    helpers["dtype"] = _dtype_from_bitpix(header)
    helpers["has_mask"] = False
    if compute_mask and np.dtype(helpers["dtype"]).kind == "f":
        # primary.data is a memory-mapped numpy array (fits.open uses memmap=True
        # upstream). Scan it in fixed-size flat blocks: np.isnan on the whole
        # memmap would page in the entire file AND allocate a full-size bool
        # temporary, while a block-wise scan bounds memory to one block and
        # short-circuits on the first NaN found (typically in the first,
        # NaN-blanked corner block).
        # Using dask here caused a "large graph" warning when a distributed client was
        # active, because da.from_array(memmap) embeds array slices as task arguments
        # which get serialised and shipped to the scheduler.
        # Integer images have no NaNs, so they are not scanned.
        flat = primary.data.reshape(-1)
        block_size = 2**23
        for start in range(0, flat.size, block_size):
            if np.isnan(flat[start : start + block_size]).any():
                helpers["has_mask"] = True
                break
    beam = _beam_attr_from_header(helpers, header)
    if beam != "mb":
        helpers["beam"] = beam
    helpers["obsdate"], helpers["obsdate_known"] = _obsdate_from_header(header)

    # TODO complete _make_history_xds when spec has been finalized
    # attrs['history'] = _make_history_xds(header)
    return attrs, helpers, header


def _time_scale_from_header(header) -> str:
    """Return the astropy time scale of the header's dates (``TIMESYS``,
    which defaults to UTC)."""
    timesys = str(header.get("TIMESYS", "")).strip().upper() or "UTC"
    scale = _TIMESYS_SCALES.get(timesys)
    if scale is None:
        xradio_logger().warning(
            f"The FITS TIMESYS {timesys!r} is not supported; assuming UTC"
        )
        scale = "utc"
    return scale


def _mjd_from_date_obs(date_obs, scale: str) -> float | None:
    """Return the MJD of a FITS ``DATE-OBS`` value, or None (with a warning)
    if it cannot be interpreted."""
    text = str(date_obs).strip()
    # DD/MM/YY, the FITS date format before 1999, for years 1900 to 1999
    old_style = re.fullmatch(r"(\d{2})/(\d{2})/(\d{2})", text)
    if old_style is not None:
        text = f"19{old_style.group(3)}-{old_style.group(2)}-{old_style.group(1)}"
    try:
        with warnings.catch_warnings():
            # ERFA warns that UTC before 1960 is "dubious"; the date is only
            # parsed here, never converted to another time scale
            warnings.simplefilter("ignore", ErfaWarning)
            return float(Time(text, format="fits", scale=scale).mjd)
    except ValueError as exc:
        xradio_logger().warning(
            f"Cannot interpret the FITS DATE-OBS {date_obs!r}: {exc}"
        )
        return None


def _obsdate_from_header(header) -> tuple[dict, bool]:
    """
    Return the observation date measure of a FITS header.

    The date is ``DATE-OBS``, else ``MJD-OBS``, in the ``TIMESYS`` time scale
    (UTC if absent). Without a usable date the reader logs a warning and uses
    :data:`_UNKNOWN_OBSDATE_MJD`, MJD 0, which casacore treats as an unset
    observation date, for the time coordinate; a date of MJD 0 in the header
    is likewise taken as unset.

    Parameters
    ----------
    header : astropy.io.fits.Header
        Primary HDU header.

    Returns
    -------
    obsdate : dict
        Time measure dictionary (MJD days).
    known : bool
        False when ``obsdate`` is the placeholder; the image then has no
        ``obsdate`` attribute, like a CASA image with an unset date.
    """
    scale = _time_scale_from_header(header)
    mjd = None
    if "DATE-OBS" in header:
        mjd = _mjd_from_date_obs(header["DATE-OBS"], scale)
    if mjd is None and "MJD-OBS" in header:
        mjd = float(header["MJD-OBS"])
    known = mjd is not None and mjd != _UNKNOWN_OBSDATE_MJD
    if not known:
        xradio_logger().warning(
            "The FITS header has no usable observation date (DATE-OBS or "
            f"MJD-OBS other than MJD {_UNKNOWN_OBSDATE_MJD}); its time "
            f"coordinate is MJD {_UNKNOWN_OBSDATE_MJD} (1858-11-17), the value "
            "casacore uses for an unset observation date, and the image has no "
            "obsdate attribute"
        )
        mjd = _UNKNOWN_OBSDATE_MJD
    obsdate = make_time_measure_dict(
        data=mjd,
        units=["d"],
        scale=scale,
        time_format="mjd",
    )
    return obsdate, known


def _make_history_xds(header):
    # TODO complete writing history when we actually have a spec for what
    # the image history is supposed to be, since doing this now may
    # be a waste of time if the final spec turns out to be significantly
    # different from our current ad hoc history xds
    # in astropy, 3506803168 seconds corresponds to 1970-01-01T00:00:00
    history_list = list(header.get("HISTORY"))
    for i in range(len(history_list) - 1, -1, -1):
        if (i == len(history_list) - 1 and history_list[i] == "CASA END LOGTABLE") or (
            i == 0 and history_list[i] == "CASA START LOGTABLE"
        ):
            history_list.pop(i)
        elif history_list[i].startswith(">"):
            # entry continuation line
            history_list[i - 1] = history_list[i - 1] + history_list[i][1:]
            history_list.pop(i)


def _create_coords(
    helpers: dict, header: fits.header, do_sky_coords: bool
) -> xr.Dataset:
    dim_map = helpers["dim_map"]
    sphr_dims = (
        [dim_map["l"], dim_map["m"]] if ("l" in dim_map) and ("m" in dim_map) else []
    )
    helpers["sphr_dims"] = sphr_dims
    spectral = helpers["spectral"]
    coords = {}
    coords["time"] = _get_time_values(helpers)
    coords["frequency"] = spectral["frequency"]
    # in FITS axis order; _fits_image_to_xds reorders it at the end
    coords["polarization"] = helpers["polarization"]
    if spectral["velocity"] is not None:
        coords["velocity"] = (["frequency"], spectral["velocity"])
    if len(sphr_dims) > 0:
        for i, c in enumerate(["l", "m"]):
            idx = sphr_dims[i]
            cdelt_rad = helpers["cdelt"][idx] * u.Unit(_get_unit(helpers["cunit"][idx]))
            # Keep the sign of CDELT: l increases to the east and m to the
            # north (AIPS Memo 27), so l follows the RA axis and m the Dec
            # axis whichever way the pixels run (CDELT1 < 0, the usual sky
            # orientation, gives a decreasing l)
            cdelt_rad = cdelt_rad.to("rad").value
            helpers[c] = {}
            helpers[c]["cunit"] = "rad"
            helpers[c]["cdelt"] = cdelt_rad
            coords[c] = _compute_linear_world_values(
                naxis=helpers["shape"][idx],
                crpix=helpers["crpix"][idx],
                crval=0.0,
                cdelt=cdelt_rad,
            )
        if do_sky_coords:

            def pick(mylist):
                return [mylist[i] for i in sphr_dims]

            my_ret = _compute_world_sph_dims(
                projection=helpers["projection"],
                shape=pick(helpers["shape"]),
                ctype=pick(helpers["ctype"]),
                crpix=pick(helpers["crpix"]),
                crval=pick(helpers["crval"]),
                cdelt=pick(helpers["cdelt"]),
                cunit=pick(helpers["cunit"]),
                projection_parameters=helpers["projection_parameters"],
                pc=helpers["pc"],
                lonpole=helpers["native_pole"][0],
                latpole=helpers["native_pole"][1],
            )
            coords[my_ret["axis_name"][0]] = (["l", "m"], my_ret["value"][0])
            coords[my_ret["axis_name"][1]] = (["l", "m"], my_ret["value"][1])
            helpers["sphr_axis_names"] = tuple(my_ret["axis_name"])
    else:
        # Fourier image
        coords["u"], coords["v"] = _get_uv_values(helpers)
    coords["beam_params_label"] = ["major", "minor", "pa"]

    xds = xr.Dataset(coords=coords)
    return xds


def _get_time_values(helpers):
    return [helpers["obsdate"]["data"]]


def _get_pol_values(helpers: dict) -> list[str]:
    """
    Return the polarization labels of the image, in FITS axis order.

    Parameters
    ----------
    helpers : dict
        Header metadata.

    Returns
    -------
    list of str
        The labels of the FITS ``STOKES`` axis codes (FITS standard, Table 7:
        I, Q, U, V = 1 to 4, RR, LL, RL, LR = -1 to -4, XX, YY, XY, YX = -5 to
        -8), or ``["I"]`` for an image without a ``STOKES`` axis.

    Raises
    ------
    ValueError
        If a plane's code is not a Stokes or correlation code.
    """
    idx = helpers["dim_map"].get("polarization")
    if idx is None:
        return ["I"]
    return conventions.fits_stokes_labels(
        helpers["crval"][idx],
        helpers["cdelt"][idx],
        # fits_stokes_labels takes the 1-based FITS CRPIX
        helpers["crpix"][idx] + 1,
        helpers["shape"][idx],
    )


def _normalize_unit(unit: str, ctype: str) -> u.UnitBase:
    """Parse an axis unit, accepting the upper case spellings in
    :data:`_FITS_UNIT_ALIASES` that astropy does not parse."""
    try:
        return u.Unit(unit)
    except ValueError:
        if unit.upper() in _FITS_UNIT_ALIASES:
            return u.Unit(_FITS_UNIT_ALIASES[unit.upper()])
        raise RuntimeError(
            f"Cannot interpret the unit {unit!r} of the {ctype} axis"
        ) from None


def _unit_scale(unit: str, target: u.UnitBase, ctype: str) -> float:
    """Return the factor that converts values of an axis to ``target``."""
    try:
        return float((1.0 * _normalize_unit(unit, ctype)).to(target).value)
    except u.UnitConversionError as exc:
        raise RuntimeError(
            f"The {ctype} axis has unit {unit!r}, which cannot be converted to {target}"
        ) from exc


def _rest_frequency_from_header(header) -> float | None:
    """Return the rest frequency in Hz: ``RESTFRQ``, its AIPS alias
    ``RESTFREQ``, or the frequency of the rest wavelength ``RESTWAV`` (in m);
    None if the header has no positive value."""
    for key in ("RESTFRQ", "RESTFREQ"):
        if key in header and float(header[key]) > 0:
            return float(header[key])
    if "RESTWAV" in header and float(header["RESTWAV"]) > 0:
        return float((_c / (float(header["RESTWAV"]) * u.m)).to(u.Hz).value)
    return None


def _velref(header) -> int | None:
    """Return the AIPS ``VELREF`` value, or None if absent or not an integer."""
    try:
        return int(header["VELREF"])
    except (KeyError, TypeError, ValueError):
        return None


def _spectral_frame_from_header(header, ctype: str | None) -> str:
    """
    Return the casacore spectral reference frame of the image.

    The frame is taken, as casacore does, from ``SPECSYS``, else from the AIPS
    frame suffix of the spectral ``CTYPE`` (``"FREQ-LSR"``), else from
    ``VELREF``. Without any of them the reader assumes LSRK, with a warning if
    the image has a spectral axis.

    Parameters
    ----------
    header : astropy.io.fits.Header
        Primary HDU header.
    ctype : str or None
        ``CTYPE`` of the spectral axis, None if the image has none.

    Returns
    -------
    str
        A casacore frame name (see
        :data:`xradio.image._util.conventions.CASACORE_SPECTRAL_FRAMES`).
    """
    specsys = str(header.get("SPECSYS", "")).strip()
    if specsys:
        try:
            return conventions.normalize_spectral_frame(specsys)
        except ValueError:
            xradio_logger().warning(
                f"The FITS SPECSYS {specsys!r} is not a known spectral "
                "reference frame; ignoring it"
            )
    if ctype is not None and _spectral_ctype_tag(ctype) in _CTYPE_FRAME_TAGS:
        return _CTYPE_FRAME_TAGS[_spectral_ctype_tag(ctype)]
    velref = _velref(header)
    if velref is not None and velref >= 0 and velref % 256 in _VELREF_FRAMES:
        return _VELREF_FRAMES[velref % 256]
    if ctype is not None:
        xradio_logger().warning(
            "The FITS header gives no spectral reference frame (SPECSYS, a "
            "frame suffix of the spectral CTYPE or VELREF); assuming LSRK"
        )
    return "LSRK"


def _doppler_type_from_header(header, axis_type: str) -> str:
    """Return the doppler convention of the velocity coordinate: optical
    (``"z"``) for VOPT and FELO axes, radio for VRAD axes, and for frequency
    axes the AIPS ``VELREF`` convention (radio above 256, else optical),
    radio without ``VELREF``."""
    if axis_type in ("VOPT", "FELO"):
        return "z"
    if axis_type == "VRAD":
        return "radio"
    velref = _velref(header)
    if velref is None:
        return "radio"
    return "radio" if velref > 256 else "z"


def _spectral_metadata(helpers: dict, header) -> dict:
    """
    Return the frequency axis of the image and its metadata.

    Frequencies, the reference frequency and the channel width are in Hz,
    whatever the unit of the FITS axis. FREQ axes are linear; VOPT and VRAD
    axes are linear in velocity and converted to frequency exactly; FELO axes
    are linear in frequency (the AIPS convention casacore reads). An image
    without a spectral axis gets, like a CASA image without one, a single
    channel described by casacore's default spectral coordinate (LSRK,
    1.415 GHz, 1 kHz wide, the HI rest frequency and radio velocities), and
    its SPECSYS and rest frequency cards are ignored, as casacore does.

    Parameters
    ----------
    helpers : dict
        Header metadata.
    header : astropy.io.fits.Header
        Primary HDU header.

    Returns
    -------
    dict
        ``frequency`` (array, Hz), ``velocity`` (array, m/s, or None when the
        rest frequency is unknown), ``doppler_type``, ``rest_frequency``
        (Hz, :data:`_UNKNOWN_REST_FREQUENCY_HZ` when unknown),
        ``reference_frequency`` (Hz), ``channel_width`` (Hz, the absolute
        frequency step from the reference pixel to the next one) and
        ``frame`` (casacore name).

    Raises
    ------
    RuntimeError
        For a velocity axis without a rest frequency, or an axis unit that
        cannot be converted.
    """
    idx = helpers["dim_map"].get("freq")
    if idx is None:
        rest_frequency = _DEFAULT_REST_FREQUENCY_HZ
        frequency = np.array([_DEFAULT_FREQUENCY_HZ])
        spectral = {
            "frame": "LSRK",
            "rest_frequency": rest_frequency,
            "reference_frequency": _DEFAULT_FREQUENCY_HZ,
            "channel_width": _DEFAULT_CHANNEL_WIDTH_HZ,
            "doppler_type": "radio",
            "velocity": None,
        }
    else:
        rest_frequency = _rest_frequency_from_header(header)
        ctype = helpers["ctype"][idx]
        spectral = {
            "frame": _spectral_frame_from_header(header, ctype),
            "rest_frequency": (
                _UNKNOWN_REST_FREQUENCY_HZ if rest_frequency is None else rest_frequency
            ),
            "velocity": None,
        }
        axis_type = ctype[:4].upper()
        nchan = helpers["shape"][idx]
        crval = helpers["crval"][idx]
        cdelt = helpers["cdelt"][idx]
        crpix = helpers["crpix"][idx]
        cunit = helpers["cunit"][idx]
        spectral["doppler_type"] = _doppler_type_from_header(header, axis_type)
        if axis_type == "FREQ":
            to_hz = _unit_scale(cunit, u.Hz, ctype)
            frequency = _compute_linear_world_values(
                naxis=nchan, crval=crval, crpix=crpix, cdelt=cdelt
            )
            if to_hz != 1.0:
                frequency = frequency * to_hz
            spectral["reference_frequency"] = crval * to_hz
            spectral["channel_width"] = abs(cdelt * to_hz)
        else:
            if rest_frequency is None:
                raise RuntimeError(
                    f"Spectral axis {ctype} in FITS header is velocity, but there "
                    "is no rest frequency (RESTFRQ, RESTFREQ or RESTWAV) so "
                    "converting to frequency is not possible"
                )
            to_ms = _unit_scale(cunit, u.m / u.s, ctype)
            c = _c.to(u.m / u.s).value
            if axis_type == "VOPT":
                freq, vel = _freq_from_vel(
                    crval,
                    cdelt,
                    crpix,
                    str(_normalize_unit(cunit, ctype)),
                    "Z",
                    nchan,
                    rest_frequency * u.Hz,
                )
                frequency = np.asarray(freq["value"])
                spectral["velocity"] = (
                    (vel["value"] * u.Unit(vel["units"])).to(u.m / u.s).value
                )
                spectral["reference_frequency"] = (
                    (freq["crval"] * u.Unit(freq["units"])).to(u.Hz).value
                )
                # The frequency step from the reference channel to the next:
                # casacore exports a frequency axis as VOPT with CDELT the
                # velocity step over that channel (not the derivative), so
                # this recovers the image's channel width exactly
                ref_velocity = crval * to_ms
                spectral["channel_width"] = abs(
                    rest_frequency / (1.0 + (ref_velocity + cdelt * to_ms) / c)
                    - rest_frequency / (1.0 + ref_velocity / c)
                )
            elif axis_type == "FELO":
                # optical velocity at the reference pixel, frequency linear
                ref_velocity = crval * to_ms
                reference = rest_frequency / (1.0 + ref_velocity / c)
                increment = -cdelt * to_ms * reference / (c + ref_velocity)
                frequency = _compute_linear_world_values(
                    naxis=nchan, crval=reference, crpix=crpix, cdelt=increment
                )
                spectral["reference_frequency"] = reference
                spectral["channel_width"] = abs(increment)
            else:
                # VRAD: radio velocity, linear in velocity and frequency
                velocity = (
                    _compute_linear_world_values(
                        naxis=nchan, crval=crval, crpix=crpix, cdelt=cdelt
                    )
                    * to_ms
                )
                frequency = rest_frequency * (1.0 - velocity / c)
                spectral["velocity"] = velocity
                spectral["reference_frequency"] = rest_frequency * (
                    1.0 - crval * to_ms / c
                )
                spectral["channel_width"] = abs(rest_frequency * cdelt * to_ms / c)
    spectral["frequency"] = frequency
    if spectral["velocity"] is None and rest_frequency is not None:
        spectral["velocity"] = np.asarray(
            _compute_velocity_values(
                restfreq=rest_frequency,
                freq_values=frequency,
                doppler=spectral["doppler_type"],
            )
        )
    return spectral


# FIXME change namee, even if there is only a single beam, we make a
# multi beam array using it. If we have a beam, it will always be
# "mutltibeam" is name is redundant and confusing
def _do_multibeam(xds: xr.Dataset, imname: str, image_type: str = "SKY") -> xr.Dataset:
    """
    Add the per-plane beams of a CASA style BEAMS table.

    Only run if we are sure there are multiple beams. The table's CHAN and POL
    columns index the image planes in FITS axis order, the order of ``xds``
    at this point (``_fits_image_to_xds`` reorders the polarization axis of
    the pixels, flags and beams together afterwards). A table with a single
    channel or polarization is broadcast over the image's planes.

    Parameters
    ----------
    xds : xr.Dataset
        Image dataset in FITS axis order.
    imname : str
        Path to the FITS file.
    image_type : str, optional
        Name of the image data variable, by default ``"SKY"``.

    Returns
    -------
    xr.Dataset
        ``xds`` with the ``BEAM_FIT_PARAMS_<image_type>`` variable (radians).
    """
    with fits.open(imname) as hdulist:
        beams_hdu = None
        for hdu in hdulist:
            if hdu.header.get("EXTNAME") == "BEAMS":
                beams_hdu = hdu
                break
        if beams_hdu is None:
            raise RuntimeError(
                "It looks like there should be a BEAMS table but no "
                "such table found in FITS file"
            )
        nchan = int(beams_hdu.header["NCHAN"])
        npol = int(beams_hdu.header["NPOL"])
        params = []
        for name, default_unit in (
            ("BMAJ", "arcsec"),
            ("BMIN", "arcsec"),
            ("BPA", "deg"),
        ):
            unit = u.Unit(beams_hdu.columns[name].unit or default_unit)
            values = np.asarray(beams_hdu.data[name], dtype=float)
            params.append((values * unit).to(u.rad).value)
        chans = np.asarray(beams_hdu.data["CHAN"], dtype=int)
        pols = np.asarray(beams_hdu.data["POL"], dtype=int)
    beams = np.zeros([nchan, npol, 3])
    beams[chans, pols] = np.stack(params, axis=-1)
    planes = (xds.sizes["frequency"], xds.sizes["polarization"])
    if beams.shape[:2] != planes:
        try:
            beams = np.broadcast_to(beams, planes + (3,))
        except ValueError:
            raise RuntimeError(
                f"The BEAMS table describes {nchan} channels and {npol} "
                f"polarizations, but the image has {planes[0]} channels and "
                f"{planes[1]} polarizations"
            ) from None
    return _create_beam_data_var(xds, beams[np.newaxis].copy(), image_type)


def _add_beam(xds: xr.Dataset, helpers: dict, image_type: str = "SKY") -> xr.Dataset:
    nchan = xds.sizes["frequency"]
    npol = xds.sizes["polarization"]
    beam_array = np.zeros([1, nchan, npol, 3])
    beam_array[0, :, :, 0] = helpers["beam"]["bmaj"]["data"]
    beam_array[0, :, :, 1] = helpers["beam"]["bmin"]["data"]
    beam_array[0, :, :, 2] = helpers["beam"]["pa"]["data"]
    return _create_beam_data_var(xds, beam_array, image_type)


def _create_beam_data_var(
    xds: xr.Dataset, beam_array: np.array, image_type: str = "SKY"
) -> xr.Dataset:
    xdb = xr.DataArray(
        beam_array, dims=["time", "frequency", "polarization", "beam_params_label"]
    )
    xdb = xdb.rename("BEAM_FIT_PARAMS_" + image_type.upper())
    xdb = xdb.assign_coords(beam_params_label=["major", "minor", "pa"])
    xdb.attrs["units"] = "rad"
    xds["BEAM_FIT_PARAMS_" + image_type.upper()] = xdb
    return xds


def _get_uv_values(helpers: dict) -> tuple:
    shape = helpers["shape"]
    ctype = helpers["ctype"]
    delt = helpers["cdelt"]
    ref_pix = helpers["crpix"]
    ref_val = helpers["crval"]
    for i, axis in enumerate(["UU", "VV"]):
        idx = ctype.index(axis)
        if idx >= 0:
            z = []
            crpix = ref_pix[i]
            crval = ref_val[i]
            cdelt = delt[i]
            for i in range(shape[idx]):
                f = (i - crpix) * cdelt + crval
                z.append(f)
            if axis == "UU":
                u = z
            else:
                v = z
    return u, v


def _add_sky_or_aperture(
    xds: xr.Dataset,
    ary: np.ndarray | da.Array,
    dim_order: list,
    header,
    helpers: dict,
    has_sph_dims: bool,
    image_type: str = "SKY",
) -> xr.Dataset:
    # TODO add code to recognize aperture images and set image_type accordingly
    xda = xr.DataArray(ary, dims=dim_order)
    for h, a in zip(
        ["BUNIT", "BTYPE", "OBJECT", "OBSERVER"],
        ["units", _image_type, "object_name", "observer"],
        strict=False,
    ):
        if h in header:
            xda.attrs[a] = header[h]
    if helpers["obsdate_known"]:
        xda.attrs["obsdate"] = helpers["obsdate"].copy()
    xda.attrs["pointing_center"] = _compute_pointing_center(helpers, header)
    xda.attrs["telescope"] = _get_telescope_metadata(helpers, header)
    xda.attrs["description"] = None
    xda.attrs["user"] = _user_attrs_from_header(header)
    # name = "SKY" if has_sph_dims else "APERTURE"
    name = image_type
    xda = xda.rename(name)
    xds[xda.name] = xda
    if helpers["has_mask"]:
        pp = da if type(xda[0].data) is dask.array.core.Array else np
        mask = pp.isnan(xda)
        mask.attrs = {}
        mask = mask.rename("FLAG_" + name.upper())
        xds["FLAG_" + name.upper()] = mask
        xds[name.upper()].attrs[_image_flag] = "FLAG_" + name.upper()
    xda = xda.rename(name)
    xds[xda.name] = xda
    return xds


def _read_image_array(
    img_full_path: str, chunks: dict, helpers: dict, verbose: bool
) -> da.array:
    # memmap = True allows only part of data to be loaded into memory
    # may also need to pass mode='denywrite'
    # https://stackoverflow.com/questions/35759713/astropy-io-fits-read-row-from-large-fits-file-with-mutliple-hdus
    if isinstance(chunks, dict):
        mychunks = _get_chunk_list(chunks, helpers)
    else:
        raise ValueError(
            f"incorrect type {type(chunks)} for parameter chunks. Must be dict"
        )
    transpose_list, new_axes = _get_transpose_list(helpers)
    data_type = helpers["dtype"]
    rshape = helpers["shape"][::-1]
    full_chunks = mychunks + tuple([1 for rr in range(5) if rr >= len(mychunks)])
    d0slices = []
    blc = tuple(5 * [0])
    trc = tuple(rshape) + tuple([1 for rr in range(5) if rr >= len(mychunks)])
    for d0 in range(blc[0], trc[0], full_chunks[0]):
        d0len = min(full_chunks[0], trc[0] - d0)
        d1slices = []
        for d1 in range(blc[1], trc[1], full_chunks[1]):
            d1len = min(full_chunks[1], trc[1] - d1)
            d2slices = []
            for d2 in range(blc[2], trc[2], full_chunks[2]):
                d2len = min(full_chunks[2], trc[2] - d2)
                d3slices = []
                for d3 in range(blc[3], trc[3], full_chunks[3]):
                    d3len = min(full_chunks[3], trc[3] - d3)
                    d4slices = []
                    for d4 in range(blc[4], trc[4], full_chunks[4]):
                        d4len = min(full_chunks[4], trc[4] - d4)
                        shapes = tuple(
                            [d0len, d1len, d2len, d3len, d4len][: len(rshape)]
                        )
                        starts = tuple([d0, d1, d2, d3, d4][: len(rshape)])
                        delayed_array = dask.delayed(_read_image_chunk)(
                            img_full_path, shapes, starts, data_type
                        )
                        d4slices += [da.from_delayed(delayed_array, shapes, data_type)]
                    d3slices += (
                        [da.concatenate(d4slices, axis=4)]
                        if len(rshape) > 4
                        else d4slices
                    )
                d2slices += (
                    [da.concatenate(d3slices, axis=3)] if len(rshape) > 3 else d3slices
                )
            d1slices += (
                [da.concatenate(d2slices, axis=2)] if len(rshape) > 2 else d2slices
            )
        d0slices += [da.concatenate(d1slices, axis=1)] if len(rshape) > 1 else d1slices
    ary = da.concatenate(d0slices, axis=0)
    ary = da.expand_dims(ary, new_axes)
    return ary.transpose(transpose_list)


def _get_chunk_list(chunks: dict, helpers: dict) -> tuple:
    ret_list = list(helpers["shape"])[::-1]
    axis = 0
    ctype = helpers["ctype"]
    for c in ctype[::-1]:
        if c.startswith("RA"):
            if "l" in chunks:
                ret_list[axis] = chunks["l"]
        elif c.startswith("DEC"):
            if "m" in chunks:
                ret_list[axis] = chunks["m"]
        elif _is_freq_like(c):
            if "frequency" in chunks:
                ret_list[axis] = chunks["frequency"]
        elif c.startswith("STOKES"):
            if "polarization" in chunks:
                ret_list[axis] = chunks["polarization"]
        else:
            raise RuntimeError(f"Unhandled coordinate type {c}")
        axis += 1
    return tuple(ret_list)


def _get_transpose_list(helpers: dict) -> tuple:
    ctype = helpers["ctype"]
    transpose_list = 5 * [-1]
    # time axis
    transpose_list[0] = 4
    new_axes = [4]
    last_axis = 3
    not_covered = ["l", "m", "u", "v", "s", "f"]
    for i, c in enumerate(ctype[::-1]):
        b = c.lower()
        if b.startswith("ra") or b.startswith("uu"):
            transpose_list[3] = i
            not_covered.remove("l")
            not_covered.remove("u")
        elif b.startswith("dec") or b.startswith("vv"):
            transpose_list[4] = i
            not_covered.remove("m")
            not_covered.remove("v")
        elif _is_freq_like(c):
            transpose_list[1] = i
            not_covered.remove("f")
        elif b.startswith("stok"):
            transpose_list[2] = i
            not_covered.remove("s")
        else:
            raise RuntimeError(f"Unhandled axis name {c}")
    # output position of each dimension: (time, frequency, polarization, l/u, m/v)
    h = {"l": 3, "m": 4, "u": 3, "v": 4, "f": 1, "s": 2}
    for p in not_covered:
        transpose_list[h[p]] = last_axis
        new_axes.append(last_axis)
        last_axis -= 1
    new_axes.sort()
    if transpose_list.count(-1) > 0:
        raise RuntimeError(
            f"Logic error: axes {ctype}, transpose_list {transpose_list}"
        )
    return transpose_list, new_axes


def _read_image_chunk(
    img_full_path, shapes: tuple, starts: tuple, dtype: str | None = None
) -> np.ndarray:
    """
    Read a block of pixels from the primary HDU of a FITS file.

    Parameters
    ----------
    img_full_path : str
        Path to the FITS file.
    shapes : tuple of int
        Shape of the block, in FITS (reversed) axis order.
    starts : tuple of int
        Index of the first pixel of the block on each axis.
    dtype : str, optional
        Declared dtype of the pixels (by default the file's).

    Returns
    -------
    np.ndarray
        A C-contiguous copy of the block in native byte order. FITS stores
        values big-endian, and returning the memory map's big-endian view
        would contradict the dtype declared for the dask array (casacore's
        putcellslice rejects byte-swapped arrays, and casatools writes them
        without swapping).
    """
    # Chunk slice
    slices = tuple(
        slice(start, start + length)
        for start, length in zip(starts, shapes, strict=False)
    )
    with fits.open(img_full_path, memmap=True) as hdulist:
        chunk = hdulist[0].data[slices]
        native = np.dtype(chunk.dtype if dtype is None else dtype).newbyteorder("=")
        chunk = np.ascontiguousarray(chunk, dtype=native)
    return chunk
