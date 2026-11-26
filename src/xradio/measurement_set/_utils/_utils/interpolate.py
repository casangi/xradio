import xarray as xr

from xradio._utils.dict_helpers import make_time_measure_attrs
from xradio._utils.logging import xradio_logger

standard_time_coord_attrs = make_time_measure_attrs(time_format="unix")


def rename_and_interpolate_to_time(
    xds: xr.Dataset,
    time_initial_name: str,
    interp_time: xr.DataArray | None,
    message_prefix: str,
) -> xr.Dataset:
    """
    This function interpolates the time dimension and renames it:

    - interpolates a time_* dimension to values given in interp_time (presumably the time
    axis of the main xds)
    - rename/replace that time_* dimension to "time", where time_* is a (sub)xds specific
    time axis
    (for example "time_pointing", "time_ephemeris", "time_syscal", "time_phase_cal").

    If interp_time is None this will simply return the input xds without modificaitons.
    Uses interpolate_to_time() for interpolation.
    ...

    Parameters:
    ----------
    xds : xr.Dataset
        Xarray dataset to interpolate (presumably a pointing_xds or an xds of
        ephemeris variables)
    time_initial_name: str = None
        Name of time to be renamed+interpolated. Expected an existing time_* coordinate in the
        dataset
    interp_time:
        Time axis to interpolate the dataset to (usually main MSv4 time)
    message_prefix:
        A prefix for info/debug/etc. messages about the specific xds being interpolated/
        time-renamed

    Returns:
    -------
    renamed_interpolated_xds : xr.Dataset
        xarray dataset with time axis renamed to "time" (from time_name, for example
        "time_ephemeris") and interpolated to interp_time.
    """
    if interp_time is None:
        return xds

    interpolated_xds = interpolate_to_time(
        xds,
        interp_time,
        message_prefix,
        time_name=time_initial_name,
    )

    # rename the time_* axis to time.

    time_coord = {"time": (time_initial_name, interp_time.data)}
    renamed_time_xds = interpolated_xds.assign_coords(time_coord)
    renamed_time_xds.coords["time"].attrs.update(standard_time_coord_attrs)
    renamed_time_xds = renamed_time_xds.swap_dims({time_initial_name: "time"})
    if time_initial_name != "time":
        renamed_time_xds = renamed_time_xds.drop_vars(time_initial_name)

    return renamed_time_xds


def interpolate_to_time(
    xds: xr.Dataset,
    interp_time: xr.DataArray | None,
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
