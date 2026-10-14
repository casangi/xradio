import astropy as ap
import astropy.units as u
import dask
import dask.array as da
import numpy as np
import xarray as xr
from astropy.time import Time

from xradio._utils.coord_math import _deg_to_rad
from xradio._utils.dict_helpers import (
    make_quantity,
    make_spectral_coord_reference_dict,
)
from xradio.image._util.conventions import (
    canonical_polarization_order,
    time_values_to_astropy,
)

_c = 2.99792458e08 * u.m / u.s
# Factors of the common frequency units to Hz, looked up before falling back
# to parsing the unit with astropy (slower, and composite unit strings leave
# cyclic garbage, see _make_coords in image_factory.py).
_HZ_PER_UNIT = {"Hz": 1.0, "kHz": 1e3, "MHz": 1e6, "GHz": 1e9, "THz": 1e12}
# A reference pixel computed by extrapolation that is this close to an integer
# (in pixels) is snapped to it, removing the round-off of the extrapolation.
_REFERENCE_PIXEL_SNAP_TOLERANCE = 1e-9
# OPTICAL = Z
_doppler_types = [
    "radio",
    "z",
    "ratio",
    "beta",
    "gamma",
]
_image_type = "type"


def _aperture_or_sky(xds: xr.Dataset) -> str:
    """
    Classify an image dataset as sky-domain or aperture-domain.

    Parameters
    ----------
    xds : xr.Dataset
        Input image dataset.

    Returns
    -------
    str
        ``"SKY"`` when sky coordinates/data variables are present, otherwise
        ``"APERTURE"``.
    """
    return "SKY" if "SKY" in xds.data_vars or "l" in xds.coords else "APERTURE"


def _get_xds_dim_order(has_sph: bool, image_type: str) -> list:
    """
    Compute canonical dimension order for an image dataset.

    Parameters
    ----------
    has_sph : bool
        Whether spherical sky coordinates are present.
    image_type : str
        Image type label.

    Returns
    -------
    list
        Ordered list of dimension names.
    """
    dimorder = ["time", "frequency", "polarization"]
    if image_type.upper() != "VISIBILITY_NORMALIZATION":
        dir_lin = ["l", "m"] if has_sph else ["u", "v"]
        dimorder.extend(dir_lin)
    return dimorder


def _convert_beam_to_rad(beam: dict) -> dict:
    """
    Convert a CASA-like beam dictionary to xradio beam quantities in radians.

    Parameters
    ----------
    beam : dict
        Beam dictionary keyed by beam parameter names with nested ``data`` and
        ``attrs`` (including units).

    Returns
    -------
    dict
        Beam dictionary keyed by ``major``, ``minor``, and ``pa`` with values
        expressed as xradio quantity dictionaries in radians.
    """
    mybeam = {}
    for k in beam:
        myu = beam[k]["attrs"]["units"]
        myu = myu[0] if isinstance(myu, list) else myu
        units = _get_unit(myu)
        q = u.quantity.Quantity(f"{beam[k]['data']}{units}")
        q = q.to("rad")
        j = "pa" if k == "positionangle" else k
        mybeam[j] = make_quantity(q.value, "rad")
    return mybeam


def _get_unit(u: str) -> str:
    """
    Normalize shorthand angular units to astropy-compatible names.

    Parameters
    ----------
    u : str
        Unit string.

    Returns
    -------
    str
        Normalized unit string.
    """
    if u == "'":
        return "arcmin"
    elif u == '"':
        return "arcsec"
    else:
        return u


def _coords_to_numpy(xds):
    """
    Materialize dask-backed coordinates as NumPy arrays.

    Parameters
    ----------
    xds : xr.Dataset
        Dataset whose coordinates may be backed by dask arrays.

    Returns
    -------
    xr.Dataset
        Dataset with dask-backed coordinate values converted to NumPy arrays.
    """
    for k, v in xds.coords.items():
        if dask.is_dask_collection(v):
            attrs = xds[k].attrs
            xds = xds.assign_coords({k: (v.sizes, v.to_numpy())})
            xds[k].attrs = attrs
    return xds


def _dask_arrayize_dv(xds: xr.Dataset) -> xr.Dataset:
    """
    Convert NumPy-backed data variables to dask arrays when needed.

    Parameters
    ----------
    xds : xr.Dataset
        Dataset whose data variables may be NumPy-backed.

    Returns
    -------
    xr.Dataset
        Dataset with all data variables backed by dask arrays.
    """
    for k, v in xds.data_vars.items():
        if not dask.is_dask_collection(v):
            dv_attrs = xds[k].attrs
            xds = xds.drop_vars([k])
            # may need to add sizes to this call as in numpy method analogs in this file
            xds = xds.assign({k: da.array(v)})
            xds[k].attrs = dv_attrs
    # only do the upper level data variables for now,
    # we don't have any data variables at sublevels so don't worry about them (yet)
    """
    if not is_copy:
        for k, v in xds.attrs.items():
            if isinstance(v, xr.Dataset):
                xds.attrs[k], is_copy = _dask_arrayize(v, is_copy)
    """
    return xds


def _numpy_arrayize_dv(xds: xr.Dataset) -> xr.Dataset:
    """
    Convert dask-backed data variables to NumPy arrays.

    Parameters
    ----------
    xds : xr.Dataset
        Dataset whose data variables may be backed by dask arrays.

    Returns
    -------
    xr.Dataset
        Dataset with all data variables backed by NumPy arrays.
    """
    # just data variables right now
    # xds, is_copy = _coords_to_numpy(xds, is_copy)
    for k, v in xds.data_vars.items():
        if dask.is_dask_collection(v):
            attrs_dv = xds[k].attrs
            xds = xds.drop_vars([k])
            xds = xds.assign({k: (v.sizes, v.to_numpy())})
            xds[k].attrs = attrs_dv
    """
    for k, v in xds.attrs.items():
        if isinstance(v, xr.Dataset):
            xds.attrs[k], is_copy = _dask_arrayize(v, is_copy)
    """
    return xds


#: Frequency (Hz) of the single channel of an image without a spectral axis:
#: the reference value of casacore's default spectral coordinate.
_DEFAULT_FREQUENCY_HZ = 1415000000.0
#: Increment (Hz) of casacore's default spectral coordinate.
_DEFAULT_CHANNEL_WIDTH_HZ = 1000.0
#: Rest frequency (Hz) of casacore's default spectral coordinate (HI).
_DEFAULT_REST_FREQUENCY_HZ = 1420405751.7860003


def _default_freq_info() -> dict:
    """
    Build the frequency coordinate attributes of an image without a spectral axis.

    The values are those of the default spectral coordinate casacore creates
    (LSRK, reference value 1.415 GHz, increment 1 kHz, HI rest frequency); the
    image is given a single channel at :data:`_DEFAULT_FREQUENCY_HZ`.

    Parameters
    ----------
    None

    Returns
    -------
    dict
        Frequency coordinate attributes conforming to the image schema.
    """
    return {
        "rest_frequency": make_quantity(_DEFAULT_REST_FREQUENCY_HZ, "Hz"),
        "type": "spectral_coord",
        "frame": "LSRK",
        "units": "Hz",
        "wave_units": "mm",
        "reference_frequency": make_spectral_coord_reference_dict(
            value=_DEFAULT_FREQUENCY_HZ, units="Hz", observer="lsrk"
        ),
        "channel_width": make_quantity(_DEFAULT_CHANNEL_WIDTH_HZ, "Hz"),
    }


def _hz_per_unit(unit: str) -> float:
    """
    Return the factor that converts values in a frequency unit to Hz.

    Parameters
    ----------
    unit : str
        Frequency unit, for example ``"Hz"`` or ``"GHz"``.

    Returns
    -------
    float
        Number of Hz in one ``unit``.

    Raises
    ------
    ValueError
        If ``unit`` is not a frequency unit.
    """
    if unit in _HZ_PER_UNIT:
        return _HZ_PER_UNIT[unit]
    try:
        return float(u.Unit(unit).to(u.Hz))
    except (ValueError, TypeError, u.UnitConversionError) as exc:
        raise ValueError(f"{unit!r} is not a frequency unit") from exc


def _freq_from_vel(
    crval: float,
    cdelt: float,
    crpix: float,
    cunit: str,
    ctype: str,
    nchan: float,
    restfreq: float,
) -> tuple:
    """
    Convert optical velocity-axis WCS parameters to frequency-axis values.

    Parameters
    ----------
    crval : float
        Velocity reference value.
    cdelt : float
        Velocity increment per channel.
    crpix : float
        Velocity reference pixel index (0-based).
    cunit : str
        Velocity unit string.
    ctype : str
        Doppler axis type; currently optical/z is supported.
    nchan : float
        Number of channels.
    restfreq : float
        Rest frequency in Hz.

    Returns
    -------
    tuple
        Two dictionaries ``(frequency_dict, velocity_dict)`` containing
        ``value``, ``units``, ``crval``, ``cdelt``, and ``crpix``.
    """
    v0 = crval - cdelt * crpix
    vel = [v0 + i * cdelt for i in range(nchan)]
    vel = vel * u.Unit(cunit)
    v_dict = {
        "value": vel.value,
        "units": cunit,
        "crval": crval,
        "cdelt": cdelt,
        "crpix": crpix,
    }
    uctype = ctype.lower()
    if uctype in ["z", "optical"]:
        freq = restfreq / (np.array(vel.value) * vel.unit / _c + 1)
        freq = freq.to(u.Hz)
        fcrval = restfreq / (crval * vel.unit / _c + 1)
        fcdelt = -restfreq / _c / (crval * vel.unit / _c + 1) ** 2 * cdelt * vel.unit
        f_dict = {
            "value": freq.value,
            "units": "Hz",
            "crval": fcrval.to(u.Hz).value,
            "cdelt": fcdelt.to(u.Hz).value,
            "crpix": crpix,
        }
    else:
        raise RuntimeError(f"Unhandled doppler type {ctype}")
    return f_dict, v_dict


def _compute_world_sph_dims(
    projection: str,
    shape: list[int],  # two element list of long-lat shape
    ctype: list[str],  # two element list of long-lat axis names
    crpix: list[float],  # two element list of long-lat crpix (zero-based)
    crval: list[float],  # two element list of long-lat crval
    cdelt: list[float],  # two element list of long-lat increments
    cunit: list[str],  # two element list of long-lat units
    projection_parameters: list[float] | None = None,
    pc: list[list[float]] | None = None,
    lonpole: float | None = None,
    latpole: float | None = None,
) -> dict:
    """
    Compute spherical world-coordinate grids from two-axis WCS inputs.

    Parameters
    ----------
    projection : str
        Spherical projection code (for example ``"SIN"``).
    shape : list[int]
        Two-element output grid shape.
    ctype : list[str]
        Two-element axis type names: right ascension and declination
        (``"Right Ascension"``, ``"RA"``, ``"Declination"``, ``"DEC"``) or
        galactic longitude and latitude (``"GLON"``, ``"GLAT"``, or casacore's
        ``"Longitude"`` and ``"Latitude"``).
    crpix : list[float]
        Two-element reference pixel indices (0-based).
    crval : list[float]
        Two-element reference coordinate values.
    cdelt : list[float]
        Two-element coordinate increments.
    cunit : list[str]
        Two-element coordinate units.
    projection_parameters : list[float], optional
        Projection parameters of the latitude axis in casacore's order
        (``PV2_1, PV2_2, ...``; ``PV2_0, PV2_1, ...`` for ZPN), for example
        ``[0, cot(dec0)]`` for the slant orthographic (NCP-like) SIN
        projection. Zero values are left at their FITS defaults.
    pc : list[list[float]], optional
        2x2 linear transformation matrix in (longitude, latitude) order, for
        example a rotation. The identity is used when not given.
    lonpole, latpole : float, optional
        Native longitude and latitude of the celestial pole in degrees
        (FITS ``LONPOLE`` and ``LATPOLE``). The FITS defaults are used when
        not given.

    Returns
    -------
    dict
        Dictionary containing axis names, reference values, increments, and
        world-coordinate value grids in radians.
    """
    # Note that if doesn't matter if the inputs are in long, lat or lat, long order,
    # as long as all inputs have consistent ordering
    wcs_dict = {}
    ret = {
        "axis_name": [None, None],
        "ref_val": [None, None],
        "inc": [None, None],
        "units": "rad",
        "value": [None, None],
    }
    for i in range(2):
        axis_name = ctype[i].lower()
        if axis_name.startswith("right") or axis_name.startswith("ra"):
            fi = 1
            wcs_dict["CTYPE1"] = f"RA---{projection}"
            new_name = "right_ascension"
        elif axis_name.startswith("dec"):
            fi = 2
            wcs_dict["CTYPE2"] = f"DEC--{projection}"
            new_name = "declination"
        elif axis_name.startswith(("galactic_longitude", "glon", "longitude")):
            # casacore names the axes of galactic images Longitude, Latitude
            fi = 1
            wcs_dict["CTYPE1"] = f"GLON-{projection}"
            new_name = "galactic_longitude"
        elif axis_name.startswith(("galactic_latitude", "glat", "latitude")):
            fi = 2
            wcs_dict["CTYPE2"] = f"GLAT-{projection}"
            new_name = "galactic_latitude"
        else:
            raise RuntimeError(f"Unhandled sky axis name {ctype[i]}")
        wcs_dict[f"NAXIS{fi}"] = shape[i]
        j = fi - 1
        x_unit = _get_unit(cunit[i])
        wcs_dict[f"CUNIT{fi}"] = x_unit
        wcs_dict[f"CDELT{fi}"] = cdelt[i]
        # FITS arrays are 1-based
        wcs_dict[f"CRPIX{fi}"] = crpix[i] + 1
        wcs_dict[f"CRVAL{fi}"] = crval[i]
        ret["axis_name"][j] = new_name
        ret["ref_val"][j] = u.quantity.Quantity(f"{crval[i]}{x_unit}").to("rad").value
        ret["inc"][j] = u.quantity.Quantity(f"{cdelt[i]}{x_unit}").to("rad").value
    if projection_parameters is not None:
        first = 0 if projection.upper() == "ZPN" else 1
        for m, value in enumerate(projection_parameters, start=first):
            if value != 0:
                wcs_dict[f"PV2_{m}"] = float(value)
    if pc is not None and not np.array_equal(np.asarray(pc, dtype=float), np.eye(2)):
        for i in range(2):
            for j in range(2):
                wcs_dict[f"PC{i + 1}_{j + 1}"] = float(pc[i][j])
    if lonpole is not None:
        wcs_dict["LONPOLE"] = float(lonpole)
    if latpole is not None:
        wcs_dict["LATPOLE"] = float(latpole)
    w = ap.wcs.WCS(wcs_dict)
    x, y = np.indices(w.pixel_shape)
    long, lat = w.pixel_to_world_values(x, y)
    # long, lat from above eqn will always be in degrees, so convert to rad
    ret["value"][0] = long * _deg_to_rad
    ret["value"][1] = lat * _deg_to_rad
    return ret


def _compute_velocity_values(
    restfreq: float,  # in Hz
    freq_values: list[float],  # in Hz
    doppler: str,  # doppler definition
) -> list[float]:
    """
    Convert frequency values to velocity values for a doppler definition.

    Parameters
    ----------
    restfreq : float
        Rest frequency in Hz.
    freq_values : list[float]
        Frequency values in Hz.
    doppler : str
        Doppler definition name.

    Returns
    -------
    list[float]
        Velocity values in m/s.
    """
    dop = doppler.lower()
    if dop == "radio":
        return [((1 - f / restfreq) * _c).value for f in freq_values]
    elif dop in ["z", "optical"]:
        return [((restfreq / f - 1) * _c).value for f in freq_values]
    else:
        raise RuntimeError(f"Doppler definition {doppler} not supported")


def _compute_linear_world_values(
    naxis: int, crval: float, crpix: float, cdelt: float
) -> np.ndarray:
    """
    Compute linearly sampled world-coordinate values.

    Parameters
    ----------
    naxis : int
        Number of points to compute.
    crval : float
        Reference coordinate value.
    crpix : float
        Reference pixel index (0-based).
    cdelt : float
        Increment per pixel.

    Returns
    -------
    np.ndarray
        Array of world-coordinate values.
    """
    return np.array([crval + (i - crpix) * cdelt for i in range(naxis)])


def _linear_axis_reference_pixel(
    values, reference_value: float = 0.0, increment: float | None = None
) -> float:
    """
    Compute the pixel at which a linear coordinate axis takes a given value.

    The reference pixel of a CASA or FITS axis does not have to lie on the
    grid: a cutout that does not contain the reference direction has its
    reference pixel outside ``[0, n - 1]``. Inside the axis the pixel is
    interpolated between the two neighbouring values (so a value on the grid
    gives its exact index); outside it is extrapolated with the increment
    between the first two pixels (which equals the increment the CASA and
    FITS writers write, ``(x[-1] - x[0]) / (n - 1)`` for a linear axis, to
    round-off), and snapped to an integer within
    :data:`_REFERENCE_PIXEL_SNAP_TOLERANCE` pixels.

    Parameters
    ----------
    values : array_like
        Coordinate values of the axis, one per pixel, monotonic.
    reference_value : float, default 0.0
        The value whose pixel is computed (0 for ``l`` and ``m``).
    increment : float or None, default None
        Increment per pixel. Only used for an axis with a single pixel, whose
        values do not define an increment.

    Returns
    -------
    float
        The 0-based (fractional) pixel at which the axis takes
        ``reference_value``.

    Raises
    ------
    ValueError
        If the axis is empty or constant, or if it has a single pixel whose
        value differs from ``reference_value`` and ``increment`` is not given.
    """
    x = np.asarray(values, dtype=np.float64)
    if x.size == 0:
        raise ValueError("Cannot compute the reference pixel of an empty axis")
    if x.size == 1:
        if x[0] == reference_value:
            return 0.0
        if not increment:
            raise ValueError(
                f"Cannot compute the reference pixel of an axis with a single "
                f"pixel at {x[0]} (not at {reference_value}) without its increment"
            )
        return float((reference_value - x[0]) / increment)
    step = x[1] - x[0]
    if step == 0:
        raise ValueError(
            "Cannot compute the reference pixel of an axis whose first two "
            "values are equal"
        )
    if min(x[0], x[-1]) <= reference_value <= max(x[0], x[-1]):
        pixels = np.arange(x.size, dtype=np.float64)
        # np.interp requires increasing sample points
        if x[-1] < x[0]:
            return float(np.interp(reference_value, x[::-1], pixels[::-1]))
        return float(np.interp(reference_value, x, pixels))
    pixel = (reference_value - x[0]) / step
    nearest = np.round(pixel)
    if abs(pixel - nearest) <= _REFERENCE_PIXEL_SNAP_TOLERANCE:
        pixel = nearest
    return float(pixel)


def _compute_sky_reference_pixel(
    xds: xr.Dataset, cdelt: list[float] | None = None
) -> np.ndarray:
    """
    Compute the reference pixel, where ``l`` and ``m`` are zero.

    The reference pixel is extrapolated when it lies outside the image, for
    example for a cutout that does not contain the reference direction (see
    :func:`_linear_axis_reference_pixel`).

    Parameters
    ----------
    xds : xr.Dataset
        Dataset containing ``l`` and ``m`` coordinates.
    cdelt : list of float or None, default None
        Increments of ``l`` and ``m`` (radians), only used for an axis with a
        single pixel. Without it, a single pixel axis must hold the reference
        direction (``l = 0`` or ``m = 0``).

    Returns
    -------
    np.ndarray
        Two-element array with the 0-based reference pixel along ``l`` and
        ``m``.

    Raises
    ------
    ValueError
        If the reference pixel of an axis cannot be determined (see
        :func:`_linear_axis_reference_pixel`).
    """
    crpix = []
    for i, c in enumerate(["l", "m"]):
        increment = None if cdelt is None else cdelt[i]
        try:
            crpix.append(_linear_axis_reference_pixel(xds[c].values, 0.0, increment))
        except ValueError as exc:
            raise ValueError(f"Reference pixel of the {c} axis: {exc}") from exc
    return np.array(crpix)


def _time_coord_to_astropy(values, attrs: dict):
    """
    Interpret the values of a time coordinate as astropy times.

    Parameters
    ----------
    values : array_like
        Time values: numbers, interpreted through the ``units``, ``format``
        and ``scale`` attributes (see
        :func:`xradio.image._util.conventions.time_values_to_astropy`), or
        ``datetime64`` values, which are times in the ``scale`` attribute's
        time scale (UTC by default).
    attrs : dict
        Attributes of the time coordinate.

    Returns
    -------
    astropy.time.Time
        The times.

    Raises
    ------
    ValueError
        If the values cannot be interpreted with the attributes.
    """
    values = np.asarray(values)
    if values.dtype.kind == "M":
        scale = str(attrs.get("scale") or "utc").lower()
        return Time(values, format="datetime64", scale=scale)
    return time_values_to_astropy(values, attrs)


def _to_canonical_polarization_order(xds: xr.Dataset) -> xr.Dataset:
    """
    Put the polarization axis of an image dataset in canonical order.

    Image datasets keep their polarization axis in the casacore ``Stokes``
    order (``I, Q, U, V``; ``RR, RL, LR, LL``; ``XX, XY, YX, YY``), so that
    the correlations of a pair of feeds map onto 2x2 Jones matrices. Images on
    disk can hold another order: FITS files must (a FITS ``STOKES`` axis is a
    linear sequence of codes, so four correlations are stored as RR, LL, RL,
    LR), CASA images converted from FITS keep the FITS order, and zarr stores
    hold whatever dataset was written. The readers restore the canonical
    order with this function.

    Parameters
    ----------
    xds : xr.Dataset
        Image dataset.

    Returns
    -------
    xr.Dataset
        The dataset itself when it has no polarization coordinate or its
        polarization axis already is in canonical order, else a copy in which
        every variable with a polarization dimension (pixels, flags, masks,
        per-plane beams) is reordered the same way (lazily for dask backed
        variables).
    """
    if "polarization" not in xds.coords:
        return xds
    order = canonical_polarization_order(xds.polarization.values)
    if order == list(range(len(order))):
        return xds
    return xds.isel(polarization=order)


def _l_m_attr_notes() -> dict[str, str]:
    """
    Provide explanatory note text for ``l`` and ``m`` coordinates.

    Parameters
    ----------
    None

    Returns
    -------
    dict[str, str]
        Mapping from coordinate name to explanatory note.
    """
    return {
        "l": "l is the projection plane coordinate towards the east, measured from "
        "the reference direction: l = x*cdelt, where x is the pixel offset from the "
        "reference pixel. For the SIN projection without projection parameters it "
        "is the direction cosine l of AIPS Memo #27, Section III.",
        "m": "m is the projection plane coordinate towards the north, measured from "
        "the reference direction: m = y*cdelt, where y is the pixel offset from the "
        "reference pixel. For the SIN projection without projection parameters it "
        "is the direction cosine m of AIPS Memo #27, Section III.",
    }
