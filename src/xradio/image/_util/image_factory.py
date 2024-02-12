from astropy.wcs import WCS
import numpy as np
import xarray as xr
from typing import List, Union
from .common import _c, _compute_world_sph_dims, _l_m_attr_notes
from ..._utils.common import _deg_to_rad


def _input_checks(
    phase_center: Union[list, np.ndarray],
    image_size: Union[list, np.ndarray],
    cell_size: Union[list, np.ndarray],
) -> None:
    if len(image_size) != 2:
        raise ValueError("image_size must have exactly two elements")
    if len(phase_center) != 2:
        raise ValueError("phase_center must have exactly two elements")
    if len(cell_size) != 2:
        raise ValueError("cell_size must have exactly two elements")


def _make_coords(
    chan_coords: Union[list, np.ndarray],
    time_coords: Union[list, np.ndarray],
) -> dict:
    if not isinstance(chan_coords, list) and not isinstance(chan_coords, np.ndarray):
        chan_coords = [chan_coords]
    chan_coords = np.array(chan_coords, dtype=np.float64)
    restfreq = chan_coords[len(chan_coords) // 2]
    vel_coords = (1 - chan_coords / restfreq) * _c
    if not isinstance(time_coords, list) and not isinstance(time_coords, np.ndarray):
        time_coords = [time_coords]
    time_coords = np.array(time_coords, dtype=np.float64)
    return dict(chan=chan_coords, vel=vel_coords, time=time_coords, restfreq=restfreq)


def _add_common_attrs(
    xds: xr.Dataset,
    restfreq: float,
    spectral_reference: str,
    direction_reference: str,
    phase_center: Union[List[float], np.ndarray],
    cell_size: Union[List[float], np.ndarray],
    projection: str,
) -> xr.Dataset:
    xds.time.attrs = {"format": "MJD", "scale": "UTC", "units": "d"}
    freq_vals = np.array(xds.frequency)
    xds.frequency.attrs = {
        "rest_frequency": {
            "type": "quantity",
            "units": "Hz",
            "value": restfreq,
        },
        "frame": spectral_reference.upper(),
        "units": "Hz",
        "wave_unit": "mm",
        # "crval": chan_coords[len(chan_coords) // 2],
        "crval": freq_vals[len(freq_vals) // 2].item(),
        "cdelt": (freq_vals[1] - freq_vals[0] if len(freq_vals) > 1 else 1000.0),
        "pc": 1.0,
    }
    xds.velocity.attrs = {"doppler_type": "RADIO", "units": "m/s"}
    xds.attrs = {
        "direction": {
            "reference": {
                "type": "sky_coord",
                "frame": direction_reference,
                "equinox": "J2000",
                "value": list(phase_center),
                "units": ["rad", "rad"],
                "cdelt": [-abs(cell_size[0]), abs(cell_size[1])],
            },
            "longpole": {"type": "quantity", "value": np.pi, "units": "rad"},
            "latpole": {"type": "quantity", "value": 0.0, "units": "rad"},
            "pc": np.array([[1.0, 0.0], [0.0, 1.0]]),
            "projection": projection,
            "projection_parameters": [0.0, 0.0],
        },
        "active_mask": "",
        "beam": None,
        "object_name": "",
        "obsdate": {
            "type": "time",
            "scale": "UTC",
            "format": "MJD",
            "value": np.array(xds.time)[0],
            "units": "d",
        },
        "observer": "Karl Jansky",
        "pointing_center": {"value": list(phase_center), "initial": True},
        "description": "",
        "telescope": {
            "name": "ALMA",
            "position": {
                "type": "position",
                "ellipsoid": "GRS80",
                "units": ["rad", "rad", "m"],
                "value": [-1.1825465955049892, -0.3994149869262738, 6379946.01326443],
            },
        },
        "history": None,
    }
    return xds


def _make_empty_sky_image(
    phase_center: Union[list, np.ndarray],
    image_size: Union[list, np.ndarray],
    cell_size: Union[list, np.ndarray],
    chan_coords: Union[list, np.ndarray],
    pol_coords: Union[list, np.ndarray],
    time_coords: Union[list, np.ndarray],
    direction_reference: str,
    projection: str,
    spectral_reference: str,
    do_sky_coords: bool,
) -> xr.Dataset:
    _input_checks(phase_center, image_size, cell_size)
    some_coords = _make_coords(chan_coords, time_coords)
    if do_sky_coords:
        long, lat = _compute_world_sph_dims(
            projection=projection,
            shape=image_size,
            ctype=["RA", "Dec"],
            crpix=[image_size[0] // 2, image_size[1] // 2],
            crval=phase_center,
            cdelt=[-abs(cell_size[0]), abs(cell_size[1])],
            cunit=["rad", "rad"],
        )["value"]
    # l follows RA as far as increasing/decreasing, see AIPS Meme 27, change in alpha
    # definition three lines below Figure 2 and the first of the pair of equations 10.
    l_coords = [
        (i - image_size[0] // 2) * (-1) * abs(cell_size[0])
        for i in range(image_size[0])
    ]
    m_coords = [
        (i - image_size[1] // 2) * abs(cell_size[1]) for i in range(image_size[1])
    ]
    coords = {
        "time": some_coords["time"],
        "polarization": pol_coords,
        "frequency": some_coords["chan"],
        "velocity": ("frequency", some_coords["vel"]),
        "l": l_coords,
        "m": m_coords,
    }
    if do_sky_coords:
        coords["right_ascension"] = (("l", "m"), long)
        coords["declination"] = (("l", "m"), lat)
    xds = xr.Dataset(coords=coords)
    attr_note = _l_m_attr_notes()
    xds.l.attrs = {
        "type": "quantity",
        "crval": 0.0,
        "cdelt": -abs(cell_size[0]),
        "units": "rad",
        "type": "quantity",
        "note": attr_note["l"],
    }
    xds.m.attrs = {
        "type": "quantity",
        "crval": 0.0,
        "cdelt": abs(cell_size[1]),
        "units": "rad",
        "type": "quantity",
        "note": attr_note["m"],
    }
    _add_common_attrs(
        xds,
        some_coords["restfreq"],
        spectral_reference,
        direction_reference,
        phase_center,
        cell_size,
        projection,
    )
    return xds


def _make_empty_aperture_image(
    phase_center: Union[list, np.ndarray],
    image_size: Union[list, np.ndarray],
    sky_image_cell_size: Union[list, np.ndarray],
    chan_coords: Union[list, np.ndarray],
    pol_coords: Union[list, np.ndarray],
    time_coords: Union[list, np.ndarray],
    direction_reference: str,
    projection: str,
    spectral_reference: str,
) -> xr.Dataset:
    _input_checks(phase_center, image_size, sky_image_cell_size)
    some_coords = _make_coords(chan_coords, time_coords)
    im_size_wave = 1 / np.array(sky_image_cell_size)
    uv_cell_size = im_size_wave / np.array(image_size)
    u_coords = [
        (i - image_size[0] // 2) * abs(uv_cell_size[0]) for i in range(image_size[0])
    ]
    v_coords = [
        (i - image_size[1] // 2) * abs(uv_cell_size[1]) for i in range(image_size[1])
    ]
    coords = {
        "time": some_coords["time"],
        "polarization": pol_coords,
        "frequency": some_coords["chan"],
        "velocity": ("frequency", some_coords["vel"]),
        "u": u_coords,
        "v": v_coords,
    }
    xds = xr.Dataset(coords=coords)
    xds.u.attrs = {
        "crval": 0.0,
        "cdelt": abs(uv_cell_size[0]),
    }
    xds.v.attrs = {
        "crval": 0.0,
        "cdelt": abs(uv_cell_size[1]),
    }
    _add_common_attrs(
        xds,
        some_coords["restfreq"],
        spectral_reference,
        direction_reference,
        phase_center,
        sky_image_cell_size,
        projection,
    )
    return xds


def _make_empty_lmuv_image(
    phase_center: Union[list, np.ndarray],
    image_size: Union[list, np.ndarray],
    sky_image_cell_size: Union[list, np.ndarray],
    chan_coords: Union[list, np.ndarray],
    pol_coords: Union[list, np.ndarray],
    time_coords: Union[list, np.ndarray],
    direction_reference: str,
    projection: str,
    spectral_reference: str,
    do_sky_coords: bool,
) -> xr.Dataset:
    _input_checks(phase_center, image_size, sky_image_cell_size)
    some_coords = _make_coords(chan_coords, time_coords)
    if do_sky_coords:
        long, lat = _compute_world_sph_dims(
            projection=projection,
            shape=image_size,
            ctype=["RA", "Dec"],
            crpix=[image_size[0] // 2, image_size[1] // 2],
            crval=phase_center,
            cdelt=[-abs(sky_image_cell_size[0]), abs(sky_image_cell_size[1])],
            cunit=["rad", "rad"],
        )["value"]
    # L follows RA as far as increasing/decreasing, see AIPS Meme 27, change in alpha
    # definition three lines below Figure 2 and the first of the pair of equations 10.
    l_coords = [
        (i - image_size[0] // 2) * (-1) * abs(sky_image_cell_size[0])
        for i in range(image_size[0])
    ]
    m_coords = [
        (i - image_size[1] // 2) * abs(sky_image_cell_size[1])
        for i in range(image_size[1])
    ]
    # im_size_wave = 1 / np.array(sky_image_cell_size)
    # uv_cell_size = im_size_wave / np.array(image_size)
    uv_cell_size = 1 / (np.array(image_size) * np.array(sky_image_cell_size))
    u_coords = [
        (i - image_size[0] // 2) * abs(uv_cell_size[0]) for i in range(image_size[0])
    ]
    v_coords = [
        (i - image_size[1] // 2) * abs(uv_cell_size[1]) for i in range(image_size[1])
    ]
    coords = {
        "time": some_coords["time"],
        "polarization": pol_coords,
        "frequency": some_coords["chan"],
        "velocity": ("frequency", some_coords["vel"]),
        "l": l_coords,
        "m": m_coords,
        "u": u_coords,
        "v": v_coords,
    }
    if do_sky_coords:
        coords["right_ascension"] = (("l", "m"), long)
        coords["declination"] = (("l", "m"), lat)
    xds = xr.Dataset(coords=coords)
    attr_note = _l_m_attr_notes()
    xds.l.attrs = {
        "type": "quantity",
        "crval": 0.0,
        "cdelt": -abs(sky_image_cell_size[0]),
        "units": "rad",
        "type": "quantity",
        "note": attr_note["l"],
    }
    xds.m.attrs = {
        "type": "quantity",
        "crval": 0.0,
        "cdelt": abs(sky_image_cell_size[1]),
        "units": "rad",
        "type": "quantity",
        "note": attr_note["m"],
    }
    xds.u.attrs = {
        "crval": 0.0,
        "cdelt": abs(uv_cell_size[0]),
    }
    xds.v.attrs = {
        "crval": 0.0,
        "cdelt": abs(uv_cell_size[1]),
    }
    _add_common_attrs(
        xds,
        some_coords["restfreq"],
        spectral_reference,
        direction_reference,
        phase_center,
        sky_image_cell_size,
        projection,
    )
    return xds
