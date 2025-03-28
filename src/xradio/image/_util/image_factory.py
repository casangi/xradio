from astropy.wcs import WCS
from collections import OrderedDict
import numpy as np
import xarray as xr
from typing import List, Union
from .common import _c, _compute_world_sph_dims, _l_m_attr_notes
from xradio._utils.coord_math import _deg_to_rad
from xradio._utils.dict_helpers import (
    make_frequency_reference_dict,
    make_quantity,
    make_skycoord_dict,
    make_time_coord_attrs,
)


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
    vel_coords = (1 - chan_coords / restfreq) * _c.to("m/s").value
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
    xds.time.attrs = make_time_coord_attrs(units=["d"], scale="utc", time_format="mjd")
    freq_vals = np.array(xds.frequency)
    xds.frequency.attrs = {
        "observer": spectral_reference.lower(),
        "reference_value": make_frequency_reference_dict(
            value=freq_vals[len(freq_vals) // 2].item(),
            units=["Hz"],
            observer=spectral_reference.lower(),
        ),
        "rest_frequencies": make_quantity(restfreq, "Hz"),
        "rest_frequency": make_quantity(restfreq, "Hz"),
        "type": "frequency",
        "units": ["Hz"],
        "wave_unit": ["mm"],
    }
    xds.velocity.attrs = {"doppler_type": "radio", "type": "doppler", "units": "m/s"}
    reference = make_skycoord_dict(
        data=phase_center, units=["rad", "rad"], frame=direction_reference
    )
    reference["attrs"].update({"equinox": "j2000.0"})
    xds.attrs = {
        "data_groups": {"base": {}},
        "direction": {
            "reference": reference,
            "lonpole": make_quantity(np.pi, "rad", ["l", "m"]),
            "latpole": make_quantity(0.0, "rad", ["l", "m"]),
            "pc": [[1.0, 0.0], [0.0, 1.0]],
            "projection": projection,
            "projection_parameters": [0.0, 0.0],
        },
    }
    return xds


def _make_common_coords(
    pol_coords: Union[list, np.ndarray],
    chan_coords: Union[list, np.ndarray],
    time_coords: Union[list, np.ndarray],
) -> dict:
    some_coords = _make_coords(chan_coords, time_coords)
    return {
        "coords": {
            "time": some_coords["time"],
            "frequency": some_coords["chan"],
            "velocity": ("frequency", some_coords["vel"]),
            "polarization": pol_coords,
        },
        "restfreq": some_coords["restfreq"],
    }


def _make_lm_values(
    image_size: Union[list, np.ndarray],
    cell_size: Union[list, np.ndarray],
) -> dict:
    # l follows RA as far as increasing/decreasing, see AIPS Meme 27, change in alpha
    # definition three lines below Figure 2 and the first of the pair of equations 10.
    l = [
        (i - image_size[0] // 2) * (-1) * abs(cell_size[0])
        for i in range(image_size[0])
    ]
    m = [(i - image_size[1] // 2) * abs(cell_size[1]) for i in range(image_size[1])]
    return {"l": l, "m": m}


def _make_sky_coords(
    projection: str,
    image_size: Union[list, np.ndarray],
    cell_size: Union[list, np.ndarray],
    phase_center: Union[list, np.ndarray],
) -> dict:
    long, lat = _compute_world_sph_dims(
        projection=projection,
        shape=image_size,
        ctype=["RA", "Dec"],
        crpix=[image_size[0] // 2, image_size[1] // 2],
        crval=phase_center,
        cdelt=[-abs(cell_size[0]), abs(cell_size[1])],
        cunit=["rad", "rad"],
    )["value"]
    return {"right_ascension": (("l", "m"), long), "declination": (("l", "m"), lat)}


def _add_lm_coord_attrs(xds: xr.Dataset) -> xr.Dataset:
    attr_note = _l_m_attr_notes()
    xds.l.attrs = {
        "note": attr_note["l"],
    }
    xds.m.attrs = {
        "note": attr_note["m"],
    }


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
    cc = _make_common_coords(pol_coords, chan_coords, time_coords)
    coords = cc["coords"]
    lm_values = _make_lm_values(image_size, cell_size)
    coords.update(lm_values)
    if do_sky_coords:
        coords.update(_make_sky_coords(projection, image_size, cell_size, phase_center))
    xds = xr.Dataset(coords=coords)
    xds = _move_beam_param_dim_coord(xds)
    _add_lm_coord_attrs(xds)
    _add_common_attrs(
        xds,
        cc["restfreq"],
        spectral_reference,
        direction_reference,
        phase_center,
        cell_size,
        projection,
    )
    return xds


def _make_uv_coords(
    xds: xr.Dataset,
    image_size: Union[list, np.ndarray],
    sky_image_cell_size: Union[list, np.ndarray],
) -> dict:
    uv_values = _make_uv_values(image_size, sky_image_cell_size)
    xds = xds.assign_coords(uv_values)
    attr = make_quantity(0.0, "lambda")
    xds.u.attrs = attr.copy()
    xds.v.attrs = attr.copy()
    return xds


def _make_uv_values(
    image_size: Union[list, np.ndarray],
    sky_image_cell_size: Union[list, np.ndarray],
) -> dict:
    im_size_wave = 1 / np.array(sky_image_cell_size)
    uv_cell_size = im_size_wave / np.array(image_size)
    u_vals = [
        (i - image_size[0] // 2) * abs(uv_cell_size[0]) for i in range(image_size[0])
    ]
    v_vals = [
        (i - image_size[1] // 2) * abs(uv_cell_size[1]) for i in range(image_size[1])
    ]
    return {"u": u_vals, "v": v_vals}


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
    cc = _make_common_coords(pol_coords, chan_coords, time_coords)
    coords = cc["coords"]
    xds = xr.Dataset(coords=coords)
    xds = _make_uv_coords(xds, image_size, sky_image_cell_size)
    _add_common_attrs(
        xds,
        cc["restfreq"],
        spectral_reference,
        direction_reference,
        phase_center,
        sky_image_cell_size,
        projection,
    )
    xds = _move_beam_param_dim_coord(xds)
    return xds


def _move_beam_param_dim_coord(xds: xr.Dataset) -> xr.Dataset:
    return xds.assign_coords(beam_param=("beam_param", ["major", "minor", "pa"]))


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
    xds = _make_empty_sky_image(
        phase_center,
        image_size,
        sky_image_cell_size,
        chan_coords,
        pol_coords,
        time_coords,
        direction_reference,
        projection,
        spectral_reference,
        do_sky_coords,
    )
    xds = _make_uv_coords(xds, image_size, sky_image_cell_size)
    xds = _move_beam_param_dim_coord(xds)
    return xds
