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
_image_type = "image_type"


def _aperture_or_sky(xds: xr.Dataset) -> str:
    return "SKY" if "SKY" in xds.data_vars or "l" in xds.coords else "APERTURE"


def _get_xds_dim_order(has_sph: bool) -> list:
    dimorder = ["time", "frequency", "polarization"]
    dir_lin = ["l", "m"] if has_sph else ["u", "v"]
    dimorder.extend(dir_lin)
    return dimorder


def _convert_beam_to_rad(beam: dict) -> dict:
    """
    Convert something that looks like a CASA beam dictionary or close to
    to it to an xradio beam dict, with xradio quantities for major, minor,
    and pa, with the quantities converted to radians. Conversions are
    done using astropy. The input beam is not modified.
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
    if u == "'":
        return "arcmin"
    elif u == '"':
        return "arcsec"
    else:
        return u


def _coords_to_numpy(xds):
    for k, v in xds.coords.items():
        if dask.is_dask_collection(v):
            attrs = xds[k].attrs
            xds = xds.assign_coords({k: (v.sizes, v.to_numpy())})
            xds[k].attrs = attrs
    return xds


def _dask_arrayize_dv(xds: xr.Dataset) -> xr.Dataset:
    """
    If necessary, change data variables to dask arrays
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
    return {
        "rest_frequency": make_quantity(1420405751.7860003, "Hz"),
        "type": "frequency",
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
    inputs are velocity info
    restfreq is an astropy quantity with frequency units
    a tuple of two dicts is returned. The first dict represents
    the frequencies, the second the velocities. Both dicts have
    keys 'value', 'unit', 'crval', 'cdelt', 'crpix'. 'value'
    is a list.
    """
    v0 = crval - cdelt * crpix
    vel = [v0 + i * cdelt for i in range(nchan)]
    vel = vel * u.Unit(cunit)
    v_dict = {
        "value": vel.value,
        "unit": cunit,
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
            "unit": "Hz",
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
    # Note that if doesn't matter if the inputs are in long, lat or lat, long order,
    # as long as all inputs have consistent ordering
    wcs_dict = {}
    ret = {
        "axis_name": [None, None],
        "ref_val": [None, None],
        "inc": [None, None],
        "unit": ["rad", "rad"],
        "value": [None, None],
    }
    for i in range(2):
        axis_name = ctype[i].lower()
        if axis_name.startswith("right") or axis_name.startswith("ra"):
            fi = 1
            wcs_dict[f"CTYPE1"] = f"RA---{projection}"
            new_name = "right_ascension"
        if axis_name.startswith("dec"):
            fi = 2
            wcs_dict["CTYPE2"] = f"DEC--{projection}"
            new_name = "declination"
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
    Simple linear transformation to get world values which can be used for certain
    coordinates like (often) frequency, l, m, u, v
    """
    return np.array([crval + (i - crpix) * cdelt for i in range(naxis)])


# def _compute_ref_pix(xds: xr.Dataset, direction: dict) -> np.ndarray:
def _compute_sky_reference_pixel(xds: xr.Dataset) -> np.ndarray:
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
    return {
        "l": "l is the angle measured from the phase center to the east. "
        "So l = x*cdelt, where x is the number of pixels from the phase center. "
        "See AIPS Memo #27, Section III.",
        "m": "m is the angle measured from the phase center to the north. "
        "So m = y*cdelt, where y is the number of pixels from the phase center. "
        "See AIPS Memo #27, Section III.",
    }
