import astropy as ap
import astropy.units as u
import dask
import dask.array as da
import numpy as np
from typing import Dict, List
import xarray as xr
from xradio._utils.coord_math import _deg_to_rad
from xradio._utils.dict_helpers import make_quantity

_c = 2.99792458e08 * u.m / u.s
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


def _default_freq_info() -> dict:
    """
    Build default spectral-coordinate metadata values.

    Parameters
    ----------
    None

    Returns
    -------
    dict
        Dictionary containing default spectral coordinate metadata.
    """
    return {
        "rest_frequency": make_quantity(1420405751.7860003, "Hz"),
        "type": "spectral_coord",
        "frame": "lsrk",
        "units": "Hz",
        "waveUnit": "mm",
        "cdelt": 1000.0,
        "crval": 1415000000.0,
    }


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
    shape: List[int],  # two element list of long-lat shape
    ctype: List[str],  # two element list of long-lat axis names
    crpix: List[float],  # two element list of long-lat crpix (zero-based)
    crval: List[float],  # two element list of long-lat crval
    cdelt: List[float],  # two element list of long-lat increments
    cunit: List[str],  # two element list of long-lat units
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
        Two-element axis type names.
    crpix : list[float]
        Two-element reference pixel indices (0-based).
    crval : list[float]
        Two-element reference coordinate values.
    cdelt : list[float]
        Two-element coordinate increments.
    cunit : list[str]
        Two-element coordinate units.

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
            wcs_dict[f"CTYPE1"] = f"RA---{projection}"
            new_name = "right_ascension"
        elif axis_name.startswith("dec"):
            fi = 2
            wcs_dict["CTYPE2"] = f"DEC--{projection}"
            new_name = "declination"
        elif axis_name.startswith("galactic_longitude") or axis_name.startswith("glon"):
            fi = 1
            wcs_dict[f"CTYPE1"] = f"GLON-{projection}"
            new_name = "galactic_longitude"
        elif axis_name.startswith("galactic_latitude") or axis_name.startswith("glat"):
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
    w = ap.wcs.WCS(wcs_dict)
    x, y = np.indices(w.pixel_shape)
    long, lat = w.pixel_to_world_values(x, y)
    # long, lat from above eqn will always be in degrees, so convert to rad
    ret["value"][0] = long * _deg_to_rad
    ret["value"][1] = lat * _deg_to_rad
    return ret


def _compute_velocity_values(
    restfreq: float,  # in Hz
    freq_values: List[float],  # in Hz
    doppler: str,  # doppler definition
) -> List[float]:
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


# def _compute_ref_pix(xds: xr.Dataset, direction: dict) -> np.ndarray:
def _compute_sky_reference_pixel(xds: xr.Dataset) -> np.ndarray:
    """
    Estimate reference pixel indices where ``l`` and ``m`` are zero.

    Parameters
    ----------
    xds : xr.Dataset
        Dataset containing ``l`` and ``m`` coordinates.

    Returns
    -------
    np.ndarray
        Two-element array with interpolated reference pixel indices for
        ``l`` and ``m``.
    """
    crpix = [None, None]
    for i, c in enumerate(["l", "m"]):
        x = xds[c].values
        y = np.array(range(len(x)))
        if x[1] < x[0]:
            x = x[::-1]
            y = y[::-1]
        crpix[i] = np.interp(0.0, x, y)
    return np.array(crpix)


def _l_m_attr_notes() -> Dict[str, str]:
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
        "l": "l is the angle measured from the reference direction to the east. "
        "So l = x*cdelt, where x is the number of pixels from the reference direction. "
        "See AIPS Memo #27, Section III.",
        "m": "m is the angle measured from the reference direction to the north. "
        "So m = y*cdelt, where y is the number of pixels from the reference direction. "
        "See AIPS Memo #27, Section III.",
    }
