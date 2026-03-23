from typing import Union

import xarray as xr

from xradio._utils.logging import xradio_logger


def interpolate_to_time(
    xds: xr.Dataset,
    interp_time: Union[xr.DataArray, None],
    message_prefix: str,
    time_name: str = "time",
) -> xr.Dataset:
    """
    Interpolate the time coordinate of the input xarray dataset to the
    a data array. This can be used for example to interpolate a pointing_xds
    to the time coord of the (main) MSv4, or similarly the ephemeris
    data variables of a field_and_source_xds.

    Uses interpolation method "linear", unless the source number of points is
    1 in which case "nearest" is used, to avoid divide-by-zero issues.

    Parameters:
    ----------
    xds : xr.Dataset
        Xarray dataset to interpolate (presumably a pointing_xds or an xds of
        ephemeris variables)
    interp_time : Union[xr.DataArray, None]
        Time axis to interpolate the dataset to (usually main MSv4 time)
    message_prefix: str
        A prefix for info/debug/etc. messages

    Returns:
    -------
    interpolated_xds : xr.Dataset
        xarray dataset with time axis interpolated to interp_time.
    """
    if interp_time is not None:
        points_before = xds[time_name].size
        if points_before > 1:
            method = "linear"
        else:
            method = "nearest"

        # print("xds before interp:",xds.NORTH_POLE_ANGULAR_DISTANCE.values, xds[time_name].values)
        # print("interp_time data:",interp_time,interp_time.data)
        # print("method:",method)
        xds = xds.interp(
            {time_name: interp_time.data}, method=method, assume_sorted=True
        )
        # scan_name sneaks in as a coordinate of the main time axis, drop it
        if (
            "type" in xds.attrs
            and xds.attrs["type"] not in ["visibility", "spectrum", "wvr"]
            and "scan_name" in xds.coords
        ):
            xds = xds.drop_vars("scan_name")
        points_after = xds[time_name].size
        xradio_logger().debug(
            f"{message_prefix}: interpolating the time coordinate "
            f"from {points_before} to {points_after} points"
        )
        # print("xds after interp:",xds.NORTH_POLE_ANGULAR_DISTANCE.values, xds[time_name].values)

    return xds
