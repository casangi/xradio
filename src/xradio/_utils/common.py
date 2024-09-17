import numpy as np

_deg_to_rad = np.pi / 180


def cast_to_str(x):
    if isinstance(x, list):
        return x[0]
    else:
        return x


def get_pad_value(col_dtype: np.dtype) -> object:
    """
    Produce a padding/missing/nan value appropriate for a casacore data column
    (for when we need to pad data vars coming from columns with rows of
    variable size array values)

    Parameters
    ----------
    col_dtype : dtype
        dtype of data being loaded from a table column

    Returns
    -------
    object
        pad value ("missing" / "fill") for the type given
    """
    # Fill values for missing/NaN data in integer variables, based on usual
    # numpy fill values. See https://github.com/numpy/numpy/issues/21166,
    # https://github.com/casangi/xradio/issues/219, https://github.com/casangi/xradio/pull/177
    fill_value_int32 = np.int32(-2147483648)
    fill_value_int64 = np.int64(-9223372036854775808)

    if col_dtype == np.int32:
        return fill_value_int32
    elif col_dtype == np.int64 or col_dtype == "int":
        return fill_value_int64
    elif np.issubdtype(col_dtype, np.floating):
        return np.nan
    elif np.issubdtype(col_dtype, np.complexfloating):
        return complex(np.nan, np.nan)
    elif np.issubdtype(col_dtype, np.bool_):
        return False
    elif np.issubdtype(col_dtype, str):
        return ""
    else:
        raise RuntimeError(
            "Padding / missing value not defined for the type requested: "
            f"{col_dtype} (of type: {type(col_dtype)})"
        )


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
    return xds


def add_position_offsets(dv_1, dv_2):
    # Fun with angles: We are adding angles together. We need to make sure that the results are between -pi and pi.
    new_pos = dv_1 + dv_2

    while np.any(new_pos[:, 0] > np.pi) or np.any(new_pos[:, 0] < -np.pi):
        new_pos[:, 0] = np.where(
            new_pos[:, 0] > np.pi, new_pos[:, 0] - 2 * np.pi, new_pos[:, 0]
        )
        new_pos[:, 0] = np.where(
            new_pos[:, 0] < -np.pi, new_pos[:, 0] + 2 * np.pi, new_pos[:, 0]
        )

    while np.any(new_pos[:, 1] > np.pi / 2) or np.any(new_pos[:, 1] < -np.pi / 2):
        new_pos[:, 1] = np.where(
            new_pos[:, 1] > np.pi / 2, new_pos[:, 1] - np.pi, new_pos[:, 1]
        )
        new_pos[:, 1] = np.where(
            new_pos[:, 1] < -np.pi / 2, new_pos[:, 1] + np.pi, new_pos[:, 1]
        )

    return new_pos
