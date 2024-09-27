import numpy as np

_deg_to_rad = np.pi / 180


def haversine(ra1, dec1, ra2, dec2):
    """
    Calculate the great circle distance between two points
    on the celestial sphere specified in radians.
    """
    # Haversine formula
    d_ra = ra2 - ra1
    d_dec = dec2 - dec1
    a = np.sin(d_dec / 2) ** 2 + np.cos(dec1) * np.cos(dec2) * np.sin(d_ra / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return c


def add_position_offsets(dv_1, dv_2):
    # Fun with angles: We are adding angles together. We need to make sure that the results are between -pi and pi.
    new_pos = dv_1 + dv_2

    new_pos = wrap_to_pi(new_pos)

    # while np.any(new_pos[:, 0] > np.pi) or np.any(new_pos[:, 0] < -np.pi):
    #     new_pos[:, 0] = np.where(
    #         new_pos[:, 0] > np.pi, new_pos[:, 0] - 2 * np.pi, new_pos[:, 0]
    #     )
    #     new_pos[:, 0] = np.where(
    #         new_pos[:, 0] < -np.pi, new_pos[:, 0] + 2 * np.pi, new_pos[:, 0]
    #     )

    # while np.any(new_pos[:, 1] > np.pi / 2) or np.any(new_pos[:, 1] < -np.pi / 2):
    #     new_pos[:, 1] = np.where(
    #         new_pos[:, 1] > np.pi / 2, new_pos[:, 1] - np.pi, new_pos[:, 1]
    #     )
    #     new_pos[:, 1] = np.where(
    #         new_pos[:, 1] < -np.pi / 2, new_pos[:, 1] + np.pi, new_pos[:, 1]
    #     )

    return new_pos


def wrap_to_pi(angles):
    """
    Wraps an array of angles in radians to the range (-π, π].

    Parameters:
    -----------
    angles : array_like
        Input array of angles in radians.

    Returns:
    --------
    array_like
        The input angles wrapped to the range (-π, π].

    Examples:
    ---------
    >>> wrap_to_pi(np.array([0, np.pi, -np.pi, 3*np.pi, -3*np.pi]))
    array([ 0.        ,  3.14159265, -3.14159265,  3.14159265, -3.14159265])
    """
    return (angles + np.pi) % (2 * np.pi) - np.pi


def wrap_to_pi(angles):
    return (angles + np.pi) % (2 * np.pi) - np.pi


def convert_to_si_units(xds):
    for data_var in xds.data_vars:
        if "units" in xds[data_var].attrs:
            for u_i, u in enumerate(xds[data_var].attrs["units"]):
                if u == "km":
                    xds[data_var][..., u_i] = xds[data_var][..., u_i] * 1e3
                    xds[data_var].attrs["units"][u_i] = "m"
                if u == "km/s":
                    xds[data_var][..., u_i] = xds[data_var][..., u_i] * 1e3
                    xds[data_var].attrs["units"][u_i] = "m/s"
                if u == "deg":
                    xds[data_var][..., u_i] = xds[data_var][..., u_i] * np.pi / 180
                    xds[data_var].attrs["units"][u_i] = "rad"
                if u == "Au" or u == "AU":
                    xds[data_var][..., u_i] = xds[data_var][..., u_i] * 149597870700
                    xds[data_var].attrs["units"][u_i] = "m"
                if u == "Au/d" or u == "AU/d":
                    xds[data_var][..., u_i] = (
                        xds[data_var][..., u_i] * 149597870700 / 86400
                    )
                    xds[data_var].attrs["units"][u_i] = "m/s"
                if u == "arcsec":
                    xds[data_var][..., u_i] = xds[data_var][..., u_i] * np.pi / 648000
                    xds[data_var].attrs["units"][u_i] = "rad"
                if u == "hPa":
                    xds[data_var][..., u_i] = xds[data_var][..., u_i] * 100.0
                    xds[data_var].attrs["units"][u_i] = "Pa"
                if u == "m-2":
                    # IONOS_ELECTRON sometimes has "m-2" instead of "/m^2"
                    xds[data_var].attrs["units"][u_i] = "/m^2"
    return xds
