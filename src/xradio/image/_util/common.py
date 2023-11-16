import astropy as ap
import astropy.units as u
import dask
import dask.array as da
import numpy as np
import xarray as xr


_c = 2.99792458e08 * u.m / u.s
# OPTICAL = Z
_doppler_types = ["RADIO", "Z", "RATIO", "BETA", "GAMMA"]
_image_type = "image_type"


def _get_xds_dim_order(has_sph: bool) -> list:
    dimorder = ["time", "polarization", "frequency"]
    dir_lin = ["l", "m"] if has_sph else ["u", "v"]
    dimorder.extend(dir_lin)
    return dimorder


def _convert_beam_to_rad(beam: dict) -> dict:
    """Convert a beam dictionary to radians"""
    mybeam = {}
    for k in beam:
        q = u.quantity.Quantity(f"{beam[k]['value']}{beam[k]['unit']}")
        q = q.to("rad")
        # q = quanta.quantity(beam[k])
        # q.convert(quanta.quantity("1rad"))
        j = "pa" if k == "positionangle" else k
        mybeam[j] = {"type": "quantity", "value": q.value, "units": str(q.unit)}
        # mybeam[j]['type'] = 'quantity'
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
            xds = xds.assign_coords({k: (v.dims, v.to_numpy())})
            xds[k].attrs = attrs
    return xds


def _dask_arrayize(xds):
    """
    If necessary, change coordinates to numpy arrays and data
    variables to dask arrays
    """
    xds = _coords_to_numpy(xds)
    for k, v in xds.data_vars.items():
        if not dask.is_dask_collection(v):
            attrs = xds[k].attrs
            xds = xds.drop_vars([k])
            # may need to add dims to this call as in numpy method analogs in this file
            xds = xds.assign({k: da.array(v)})
            xds[k].attrs = attrs
    for k, v in xds.attrs.items():
        if isinstance(v, xr.Dataset):
            xds.attrs[k] = _dask_arrayize(v)
    return xds


def _numpy_arrayize(xds):
    xds = _coords_to_numpy(xds)
    for k, v in xds.data_vars.items():
        if dask.is_dask_collection(v):
            attrs = xds[k].attrs
            xds = xds.drop_vars([k])
            xds = xds.assign({k: (v.dims, v.to_numpy())})
            xds[k].attrs = attrs
    for k, v in xds.attrs.items():
        if isinstance(v, xr.Dataset):
            xds.attrs[k] = _dask_arrayize(v)
    return xds


def _default_freq_info() -> dict:
    return {
        """
        "conversion": {
            "direction": {
                "m0": {"unit": "rad", "value": 0.0},
                "m1": {"unit": "rad", "value": 1.5707963267948966},
                "system": "FK5",
                "equinox": "J2000",
            },
            "epoch": {
                "m0": {"unit": "d", "value": 0.0},
                "refer": "LAST",
            },
            "position": {
                "m0": {"unit": "rad", "value": 0.0},
                "m1": {"unit": "rad", "value": 0.0},
                "m2": {"unit": "m", "value": 0.0},
                "refer": "ITRF",
            },
            "system": "LSRK",
        },
        """
        # "nativeType": "FREQ",
        "rest_frequency": 1420405751.7860003,
        # "restfreqs": [1420405751.7860003],
        "system": "LSRK",
        "unit": "Hz",
        "waveUnit": "mm",
        "wcs": {
            "cdelt": 1000.0,
            "crval": 1415000000.0,
        },
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
    uctype = ctype.upper()
    if uctype == "Z" or uctype == "OPTICAL":
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
