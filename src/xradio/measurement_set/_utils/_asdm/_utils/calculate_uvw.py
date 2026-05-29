import numpy as np

import xarray as xr

import astropy.coordinates as coord
from astropy.time import Time
import astropy.units as u


def calculate_uvw(
    key: tuple[slice] | None,
    time: xr.DataArray,
    baseline_antenna1_name: xr.DataArray,
    baseline_antenna2_name: xr.DataArray,
    antenna_position: xr.DataArray,
    field_phase_center_direction: xr.DataArray,
) -> np.ndarray:
    """
    Produces an array of UVW coordinates of shape (num_times, num_antennas, 3),
    where num_times is the length of the time input, and num_antenna
    is the length of the antenna_position input.
    """
    site_position = {
        "m0": {
            "unit": "m",
            "value": 2225142.180268967,
        },
        "m1": {"unit": "m", "value": -5440307.370348562},
        "m2": {"unit": "m", "value": -2481029.851873547},
        "refer": "ITRF",
        "type": "position",
    }

    antenna1_idx = _antenna_names_to_indices(
        antenna_position.coords["antenna_name"], baseline_antenna1_name
    )
    antenna2_idx = _antenna_names_to_indices(
        antenna_position.coords["antenna_name"], baseline_antenna2_name
    )

    if key:
        time_index = key[0]
        time = time[time_index]

        baseline_index = key[1]
        antenna1_idx = antenna1_idx[baseline_index]
        antenna2_idx = antenna2_idx[baseline_index]

    uvw = _calculate_uvw_astropy(
        time,
        field_phase_center_direction,
        site_position,
        antenna_position,
        antenna1_idx,
        antenna2_idx,
    )

    if key:
        uvw_index = key[2]
        uvw = uvw[uvw_index]

    return uvw


def _antenna_names_to_indices(antenna_name, baseline_antenna_name) -> np.ndarray:
    """From the antenna_name coordinate to the (original) antenna ids"""
    sorted_name_indices = np.argsort(antenna_name)  # .values
    sorted_antenna_name = antenna_name[sorted_name_indices]

    sorted_antenna_indices = np.searchsorted(sorted_antenna_name, baseline_antenna_name)
    baseline_antenna_indices = sorted_name_indices[sorted_antenna_indices]
    return baseline_antenna_indices


def _calculate_uvw_astropy(
    time: np.ndarray,
    phase_center_ra_dec: xr.DataArray,
    site_position: dict,
    antenna_position: xr.DataArray,
    antenna1_idx: np.ndarray,
    antenna2_idx: np.ndarray,
) -> np.ndarray:
    """
    Produces an array of UVWs per time, baseline

    Borrowed and adapted from sirius/_sirius_utils/_uvw_utils.py
    """

    # Time of observation:
    num_ant = antenna_position.sizes["antenna_name"]
    time_x_antenna = np.tile(time.values[:, np.newaxis], (1, num_ant))
    time_observation = Time(
        Time(
            time_x_antenna * u.Unit(time.attrs["units"]),
            format="mjd",
            scale=time.attrs["scale"],
        ),
        format="mjd",
        scale="utc",
    )

    # Format antenna positions and array center as EarthLocation.
    num_time = len(time.values)
    times_x_antenna_position = np.tile(
        antenna_position.values[np.newaxis, :, :] * u.m, (num_time, 1, 1)
    )
    antpos_earth_location = coord.EarthLocation(
        x=times_x_antenna_position[:, :, 0],
        y=times_x_antenna_position[:, :, 1],
        z=times_x_antenna_position[:, :, 2],
    )
    telescope_site = coord.EarthLocation(
        x=site_position["m0"]["value"] * u.m,
        y=site_position["m1"]["value"] * u.m,
        z=site_position["m2"]["value"] * u.m,
    )

    # Convert antenna pos terrestrial to celestial.  For astropy use
    # get_gcrs_posvel(t)[0] rather than get_gcrs(t) because if a velocity
    # is attached to the coordinate astropy will not allow us to do additional
    # transformations with it (https://github.com/astropy/astropy/issues/6280)
    telescope_site_p, telescope_site_v = telescope_site.get_gcrs_posvel(
        time_observation
    )
    antpos_c_ap = coord.GCRS(
        antpos_earth_location.get_gcrs_posvel(time_observation)[0],
        obstime=time_observation,
        obsgeoloc=telescope_site_p,
        obsgeovel=telescope_site_v,
    )

    phase_center_sky_coord = coord.SkyCoord(
        phase_center_ra_dec.sel(sky_dir_label="ra").values * u.rad,
        phase_center_ra_dec.sel(sky_dir_label="dec").values * u.rad,
        frame="icrs",
    )
    # For ICRS:
    # frame_uvw = phase_center_sky_coord.skyoffset_frame()
    # For GCRS:
    frame_uvw = phase_center_sky_coord.transform_to(antpos_c_ap).skyoffset_frame()

    # Rotate antenna positions into UVW frame.
    antpos_uvw_cartesian = antpos_c_ap.transform_to(frame_uvw).cartesian

    antenna_uvw = np.array(
        [antpos_uvw_cartesian.y, antpos_uvw_cartesian.z, antpos_uvw_cartesian.x]
    )
    antenna_uvw = np.moveaxis(antenna_uvw, 0, -1)

    uvw = np.ascontiguousarray(
        antenna_uvw[:, antenna1_idx, :] - antenna_uvw[:, antenna2_idx, :]
    )

    return uvw
