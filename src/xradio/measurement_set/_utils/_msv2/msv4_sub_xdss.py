import toolviper.utils.logger as logger
import os
import time
from typing import Optional, Tuple, Union

import numpy as np
import xarray as xr
from numpy.typing import ArrayLike

from xradio._utils.coord_math import convert_to_si_units
from xradio._utils.schema import (
    column_description_casacore_to_msv4_measure,
    convert_generic_xds_to_xradio_schema,
)
from .subtables import subt_rename_ids
from ._tables.read import (
    load_generic_table,
    make_taql_where_between_min_max,
    table_exists,
    table_has_column,
)


standard_time_coord_attrs = {
    "type": "time",
    "units": ["s"],
    "scale": "utc",
    "format": "unix",
}


def rename_and_interpolate_to_time(
    xds: xr.Dataset,
    time_initial_name: str,
    interp_time: Union[xr.DataArray, None],
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
        xds = xds.interp(
            {time_name: interp_time.data}, method=method, assume_sorted=True
        )
        # scan_name sneaks in as a coordinate of the main time axis, drop it
        if "scan_name" in xds.coords:
            xds = xds.drop_vars("scan_name")
        points_after = xds[time_name].size
        logger.debug(
            f"{message_prefix}: interpolating the time coordinate "
            f"from {points_before} to {points_after} points"
        )

    return xds


def make_taql_where_weather(
    in_file: str, ant_xds_station_name_ids: xr.DataArray
) -> str:
    """
    The use of taql_where with WEATHER is complicated because of ALMA and its
    (ab)use of NS_WX_STATION_ID vs. ANTENNA_ID (all -1) (see read.py). We
    cannot simply use a 'WHERE ANTENNA_ID=...'. This function produces a TaQL
    where string that uses ANTENNA_ID/NS_WX_STATION_ID depending on waht is
    found in the WEATHER subtable.

    Parameters
    ----------
    in_file : str
        Input MS name.
    ant_xds_station_name_ids : xr.DataArray
        station name data array from antenna_xds, with name/id information

    Returns
    -------
    taql_where: str
        WHERE substring safe to use in taql_where when loading WEATHER subtable
    """
    weather_path = os.path.join(in_file, "WEATHER")
    if table_exists(weather_path) and table_has_column(
        weather_path, "NS_WX_STATION_ID"
    ):
        taql_where = f"WHERE (NS_WX_STATION_ID IN [{','.join(map(str, ant_xds_station_name_ids.antenna_id.values))}])"
    else:
        taql_where = f"WHERE (ANTENNA_ID IN [{','.join(map(str, ant_xds_station_name_ids.antenna_id.values))}])"

    return taql_where


def prepare_generic_weather_xds_and_station_name(
    generic_weather_xds: xr.Dataset,
    in_file: str,
    ant_position_with_ids: xr.DataArray,
    has_asdm_station_position: bool,
) -> tuple[xr.Dataset, np.ndarray]:
    """
    A generic_weather_xds loaded with load_generic_table() might still need to be reloaded
    with an additional WHERE condition to constrain the indices of antennas. But this depends on whether
    ASDM/importasdm extension columns are present or not.

    This also prepares the station_name values:
    - if has_asdm_station_ids:
       - tries to find from ASDM_STATION the station names,
       - otherwise, takes ids (antenna_ids in generic_weather were actually the ASDM_STATION_IDs
    - else: get the values from antenna_xds (the stations present)


    Parameters
    ----------
    generic_weather_xds : xr.Dataset
        generic dataset read from an MSv2 WEATHER subtable
    in_file : str
        Input MS name.
    ant_position_with_ids : xr.DataArray
        antenna_position data var from the antenna_xds (expected to still include the initial ANTENNA_ID
        coordinate as well as other coordinates from the antenna_xds)
    has_asdm_station_position : bool
        wHether this generic weatehr_xds should be treated as including the nonstandard extensions
        NS_WX_STATION_ID and NS_WX_STATION_POSITION as created by CASA/importasdm (ALMA and VLA).

    Returns
    -------
    (generic_weather_xds, station_name): tuple[[xarray.Dataset, numpy.ndarray]
        Weather Xarray Dataset prepared for generic conversion to MSv4, values for the station_name coordinate
    """

    if has_asdm_station_position:
        asdm_station_path = os.path.join(in_file, "ASDM_STATION")
        if table_exists(asdm_station_path):
            asdm_station_xds = load_generic_table(in_file, "ASDM_STATION")
            station_name = asdm_station_xds.name.values[
                generic_weather_xds["ANTENNA_ID"].values
            ]
        else:
            # if no info from ASDM_STATION, use the indices from antenna_id which was actually the NS_WX_STATION_ID
            len_antenna_id = generic_weather_xds.sizes["ANTENNA_ID"]
            station_name = list(
                map(
                    lambda x, y: x + "_" + y,
                    ["Station"] * len_antenna_id,
                    generic_weather_xds["ANTENNA_ID"].values.astype(str),
                )
            )

    else:
        taql_where = make_taql_where_weather(in_file, ant_position_with_ids)
        generic_weather_xds = load_generic_table(
            in_file,
            "WEATHER",
            rename_ids=subt_rename_ids["WEATHER"],
            taql_where=taql_where,
        )

        if not generic_weather_xds.data_vars:
            # for example when the weather subtable only has info for antennas/stations
            # not present in the MSv4 (no overlap between antennas loaded in ant_xds and weather)
            return None, None

        stations_present = ant_position_with_ids.sel(
            antenna_id=generic_weather_xds["ANTENNA_ID"]
        ).station_name
        station_name = stations_present.values

    return generic_weather_xds, station_name


def finalize_station_position(
    weather_xds: xr.Dataset, ant_position_with_ids, has_asdm_station_position: bool
) -> xr.Dataset:
    """
    For a STATION_POSITION data var being added to a weather_xds, make sure coordinates and dimensions
    are conforming to the schema.

    Parameters
    ----------
    weather_xds : xr.Dataset
        weather_xds where we still need to ensure the right coordinates and attributes
    ant_position_with_ids : xr.DataArray
        antenna_position data var from the antenna_xds (expected to still include the initial ANTENNA_ID
        coordinate as well as other coordinates from the antenna_xds)
    has_asdm_station_position : bool
        Whether this generic weatehr_xds should be treated as including the nonstandard extensions
        NS_WX_STATION_ID and NS_WX_STATION_POSITION as created by CASA/importasdm (ALMA and VLA).

    Returns
    -------
    weather_xds: xarray.Dataset
        Weather Xarray Dataset with all coordinates and attributes in STATION_POSITION
    """
    if has_asdm_station_position:
        # STATION_POSITION has been created but needs prooper dimensions and attrs
        # Drop the time dim
        weather_xds["STATION_POSITION"] = weather_xds["STATION_POSITION"].sel(
            time_weather=0, drop=True, method="nearest"
        )
        # borrow location frame attributes from antenna position
        weather_xds["STATION_POSITION"].attrs = ant_position_with_ids.attrs
    else:
        # borrow from ant_posision_with_ids but without carrying over other coords
        weather_xds = weather_xds.assign(
            {
                "STATION_POSITION": (
                    ["station_name", "cartesian_pos_label"],
                    ant_position_with_ids.values,
                    ant_position_with_ids.attrs,
                )
            }
        )

    return weather_xds


def create_weather_xds(in_file: str, ant_position_with_ids: xr.DataArray):
    """
    Creates a Weather Xarray Dataset from a MS v2 WEATHER table.

    Parameters
    ----------
    in_file : str
        Input MS name.
    ant_position_with_ids : xr.DataArray
        antenna_position data var from the antenna_xds (expected to still including the initial ANTENNA_ID coordinate
        as wellas other coordinates from the antenna_xds)

    Returns
    -------
    xr.Dataset
        Weather Xarray Dataset.
    """

    try:
        generic_weather_xds = load_generic_table(
            in_file,
            "WEATHER",
            rename_ids=subt_rename_ids["WEATHER"],
        )
    except ValueError as _exc:
        return None

    has_asdm_station_position = (
        "NS_WX_STATION_POSITION" in generic_weather_xds.data_vars
    )
    generic_weather_xds, station_name = prepare_generic_weather_xds_and_station_name(
        generic_weather_xds, in_file, ant_position_with_ids, has_asdm_station_position
    )
    if not generic_weather_xds:
        return None

    weather_xds = xr.Dataset(attrs={"type": "weather"})
    coords = {
        "station_name": station_name,
        "cartesian_pos_label": ["x", "y", "z"],
    }
    weather_xds = weather_xds.assign_coords(coords)

    dims_station_time = ["station_name", "time_weather"]
    dims_station_time_position = dims_station_time + ["cartesian_pos_label"]
    to_new_data_variables = {
        "H20": ["H2O", dims_station_time],
        "IONOS_ELECTRON": ["IONOS_ELECTRON", dims_station_time],
        "PRESSURE": ["PRESSURE", dims_station_time],
        "REL_HUMIDITY": ["REL_HUMIDITY", dims_station_time],
        "TEMPERATURE": ["TEMPERATURE", dims_station_time],
        "DEW_POINT": ["DEW_POINT", dims_station_time],
        "WIND_DIRECTION": ["WIND_DIRECTION", dims_station_time],
        "WIND_SPEED": ["WIND_SPEED", dims_station_time],
    }
    if has_asdm_station_position:
        to_new_data_variables.update(
            {
                "NS_WX_STATION_POSITION": [
                    "STATION_POSITION",
                    dims_station_time_position,
                ],
            }
        )

    to_new_coords = {
        "TIME": ["time_weather", ["time_weather"]],
    }

    weather_xds = convert_generic_xds_to_xradio_schema(
        generic_weather_xds, weather_xds, to_new_data_variables, to_new_coords
    )
    weather_xds = finalize_station_position(
        weather_xds, ant_position_with_ids, has_asdm_station_position
    )

    # TODO: option to interpolate to main time

    # PRESSURE: hPa in MSv2 specs and some MSs => Pa
    weather_xds = convert_to_si_units(weather_xds)

    # correct expected types (for example "IONOS_ELECTRON", "PRESSURE" can be float32)
    for data_var in weather_xds:
        if weather_xds.data_vars[data_var].dtype != np.float64:
            weather_xds[data_var] = weather_xds[data_var].astype(np.float64)

    return weather_xds


def correct_generic_pointing_xds(
    generic_pointing_xds: xr.Dataset, to_new_data_variables: dict[str, list]
) -> xr.Dataset:
    """
    Takes a (generic) pointing_xds as read from a POINTING subtable of an MSv2
    and tries to correct several deviations from the MSv2 specs seen in
    common test data.
    The problems fixed here include wrong dimensions:

    - for example transposed dimensions with respect to the MSv2 specs (output
    from CASA simulator),
    - missing/additional unexpected dimensions when some of the columns are
    empty (in the sense of "empty casacore cells").

    This function modifies the data arrays of the data vars affected by such
    issues.

    Parameters
    ----------
    generic_pointing_xds: xr.Dataset
        The generic pointing dataset (loaded from MSv2) to be fixed
    to_new_data_variables: dict
        The dict used for convert_generic_xds_to_xradio_schema, which gives all
        the data variables relevant for the final MSv4 dataset.

    Returns:
    --------
    xr.Dataset
        Corrected dataset with dimensions conforming to MSv2 specs.
    """

    correct_pointing_xds = generic_pointing_xds.copy()

    for data_var_name in generic_pointing_xds:
        if data_var_name in to_new_data_variables:
            # Corrects dim sizes of "empty cell" variables, such as empty DIRECTION, TARGET, etc.
            if (
                "dim_2" in generic_pointing_xds.sizes
                and generic_pointing_xds.sizes["dim_2"] == 0
            ):
                # When all direction variables are "empty"
                data_var_data = xr.DataArray(
                    [[[[np.nan, np.nan]]]],
                    dims=generic_pointing_xds.dims,
                ).isel(n_polynomial=0, drop=True)
                correct_pointing_xds[data_var_name].data = data_var_data

            elif (
                "dir" in generic_pointing_xds.sizes
                and generic_pointing_xds.sizes["dir"] == 0
            ):
                # When some direction variables are "empty" but some are populated properly
                if "dim_2" in generic_pointing_xds[key].sizes:
                    data_var_data = xr.DataArray(
                        generic_pointing_xds[key].values,
                        dims=generic_pointing_xds[key].dims,
                    )
                else:
                    shape = tuple(
                        generic_pointing_xds.sizes[dim]
                        for dim in ["TIME", "ANTENNA_ID"]
                    ) + (2,)
                    data_var_data = xr.DataArray(
                        np.full(shape, np.nan),
                        dims=generic_pointing_xds[key].dims,
                    )
                correct_pointing_xds[data_var_name].data = data_var_data

    return correct_pointing_xds


def create_pointing_xds(
    in_file: str,
    ant_xds_name_ids: xr.DataArray,
    time_min_max: Union[Tuple[np.float64, np.float64], None],
    interp_time: Union[xr.DataArray, None] = None,
) -> xr.Dataset:
    """
    Creates a Pointing Xarray Dataset from an MS v2 POINTING (sub)table.

    WIP: details of a few direction variables (and possibly moving some to attributes) to be
    settled (see MSv4 spreadsheet).

    Parameters
    ----------
    in_file : str
        Input MS name.
    ant_xds_name_ids : xr.DataArray
        antenna_name data array from antenna_xds, with name/id information
    time_min_max : tuple
        min / max times values to constrain loading (from the TIME column)
    interp_time : Union[xr.DataArray, None] (Default value = None)
        interpolate time to this (presumably main dataset time)

    Returns
    -------
    xr.Dataset
         Pointing Xarray dataset
    """
    start = time.time()

    taql_time_range = make_taql_where_between_min_max(
        time_min_max, in_file, "POINTING", "TIME"
    )

    if taql_time_range is None:
        taql_where = f"WHERE (ANTENNA_ID IN [{','.join(map(str, ant_xds_name_ids.antenna_id.values))}])"
    else:
        taql_where = (
            taql_time_range
            + f" AND (ANTENNA_ID IN [{','.join(map(str, ant_xds_name_ids.antenna_id.values))}])"
        )
    # Read POINTING table into a Xarray Dataset.
    generic_pointing_xds = load_generic_table(
        in_file,
        "POINTING",
        rename_ids=subt_rename_ids["POINTING"],
        taql_where=taql_where,
    )

    if not generic_pointing_xds.data_vars:
        # apparently empty MS/POINTING table => produce empty xds
        return xr.Dataset()

    # Checking a simple way of using only the one single coefficient of the polynomials
    if "n_polynomial" in generic_pointing_xds.sizes:
        size = generic_pointing_xds.sizes["n_polynomial"]
        if size == 1:
            generic_pointing_xds = generic_pointing_xds.sel({"n_polynomial": 0})
        elif size == 0:
            generic_pointing_xds = generic_pointing_xds.drop_dims("n_polynomial")

    time_ant_dims = ["time_pointing", "antenna_name"]
    time_ant_dir_dims = time_ant_dims + ["local_sky_dir_label"]
    to_new_data_variables = {
        "DIRECTION": ["POINTING_BEAM", time_ant_dir_dims],
        "ENCODER": ["POINTING_DISH_MEASURED", time_ant_dir_dims],
        "OVER_THE_TOP": ["POINTING_OVER_THE_TOP", time_ant_dims],
    }

    to_new_coords = {
        "TIME": ["time_pointing", ["time_pointing"]],
        "dim_2": ["local_sky_dir_label", ["local_sky_dir_label"]],
    }

    generic_pointing_xds = correct_generic_pointing_xds(
        generic_pointing_xds, to_new_data_variables
    )
    pointing_xds = xr.Dataset(attrs={"type": "pointing"})
    coords = {
        "antenna_name": ant_xds_name_ids.sel(
            antenna_id=generic_pointing_xds["ANTENNA_ID"]
        ).data,
        "local_sky_dir_label": ["az", "alt"],
    }
    pointing_xds = pointing_xds.assign_coords(coords)
    pointing_xds = convert_generic_xds_to_xradio_schema(
        generic_pointing_xds, pointing_xds, to_new_data_variables, to_new_coords
    )

    pointing_xds = rename_and_interpolate_to_time(
        pointing_xds, "time_pointing", interp_time, "pointing_xds"
    )

    logger.debug(f"create_pointing_xds() execution time {time.time() - start:0.2f} s")

    return pointing_xds


def prepare_generic_sys_cal_xds(generic_sys_cal_xds: xr.Dataset) -> xr.Dataset:
    """
    A generic_sys_cal_xds loaded with load_generic_table() cannot be easily
    used in convert_generic_xds_to_xradio_schema() to produce an MSv4
    sys_cal_xds dataset, as their structure differs in dimensions and order
    of dimensions.
    This function performs various prepareation steps, such as:

    - filter out dimensions not neeed for an individual MSv4 (SPW, FEED),
    - drop variables loaded from columns with all items set to empty array,
    - transpose the dimensions frequency,receptor
    - fix dimension names (and order) when needed.

    Parameters
    ----------
    generic_sys_cal_xds : xr.Dataset
        generic dataset read from an MSv2 SYSCAL subtable

    Returns
    -------
    generic_sys_cal_xds: xr.Dataset
        System calibration Xarray Dataset prepared for generic conversion
        to MSv4.
    """

    # drop SPW and feed dims
    generic_sys_cal_xds = generic_sys_cal_xds.isel(SPECTRAL_WINDOW_ID=0, drop=True)
    generic_sys_cal_xds = generic_sys_cal_xds.isel(FEED_ID=0, drop=True)

    # Often some of the T*_SPECTRUM are present but all the cells are populated
    # with empty arrays
    empty_arrays_vars = []
    for data_var in generic_sys_cal_xds.data_vars:
        if generic_sys_cal_xds[data_var].size == 0:
            empty_arrays_vars.append(data_var)
    if empty_arrays_vars:
        generic_sys_cal_xds = generic_sys_cal_xds.drop_vars(empty_arrays_vars)

    # Re-arrange receptor and frequency dims depending on input structure
    if (
        "receptor" in generic_sys_cal_xds.sizes
        and "frequency" in generic_sys_cal_xds.sizes
    ):
        # dim_3 can be created for example when the T*_SPECTRUM have varying # channels!
        # more generaly, could transpose with ... to avoid errors with additional spurious dimensions
        if "dim_3" in generic_sys_cal_xds.dims:
            generic_sys_cal_xds = generic_sys_cal_xds.drop_dims("dim_3")
        # From MSv2 tables we get (...,frequency, receptor)
        #  -> transpose to (...,receptor,frequency) ready for MSv4 sys_cal_xds
        generic_sys_cal_xds = generic_sys_cal_xds.transpose(
            "ANTENNA_ID", "TIME", "receptor", "frequency"
        )
    elif (
        "frequency" in generic_sys_cal_xds.sizes
        and not "dim_3" in generic_sys_cal_xds.sizes
    ):
        # because order is (...,frequency,receptor), when frequency is missing
        # receptor can get wrongly labeled as frequency
        generic_sys_cal_xds = generic_sys_cal_xds.rename_dims({"frequency": "receptor"})
    elif (
        "frequency" not in generic_sys_cal_xds.sizes
        and "receptor" in generic_sys_cal_xds.sizes
        and "dim_3" in generic_sys_cal_xds.sizes
    ):
        # different *_SPECTRUM array sizes + some empty arrays can create an additional spurious
        # generic dimension, which should have been "receptor"
        generic_sys_cal_xds = generic_sys_cal_xds.rename_dims({"receptor": "frequency"})
        generic_sys_cal_xds = generic_sys_cal_xds.rename_dims({"dim_3": "receptor"})
        generic_sys_cal_xds = generic_sys_cal_xds.transpose(
            "ANTENNA_ID", "TIME", "receptor", "frequency"
        )
    else:
        raise RuntimeError(
            "Cannot understand the arrangement of dimensions of {generic_sys_cal_xds=}"
        )

    return generic_sys_cal_xds


def create_system_calibration_xds(
    in_file: str,
    main_xds_frequency: xr.DataArray,
    ant_xds: xr.DataArray,
    sys_cal_interp_time: Union[xr.DataArray, None] = None,
):
    """
    Creates a system calibration Xarray Dataset from a MSv2 SYSCAL table.

    Parameters
    ----------
    in_file: str
        Input MS name.
    main_xds_frequency: xr.DataArray
        frequency array of the main xds (MSv4), containing among other things
        spectral_window_id and measures metadata
    ant_xds : xr.Dataset
        The antenna_xds that has information such as names, stations, etc., for coordinates
    sys_cal_interp_time: Union[xr.DataArray, None] = None,
        Time axis to interpolate the data vars to (usually main MSv4 time)

    Returns
    -------
    sys_cal_xds: xr.Dataset
        System calibration Xarray Dataset.
    """

    spectral_window_id = main_xds_frequency.attrs["spectral_window_id"]
    try:
        generic_sys_cal_xds = load_generic_table(
            in_file,
            "SYSCAL",
            rename_ids=subt_rename_ids["SYSCAL"],
            taql_where=(
                f" where (SPECTRAL_WINDOW_ID = {spectral_window_id})"
                f" AND (ANTENNA_ID IN [{','.join(map(str, ant_xds.antenna_id.values))}])"
            ),
        )
    except ValueError as _exc:
        return None

    if not generic_sys_cal_xds.data_vars:
        # even though SYSCAL is an optional subtable, some write it empty
        return None

    generic_sys_cal_xds = prepare_generic_sys_cal_xds(generic_sys_cal_xds)

    mandatory_dimensions = ["antenna_name", "time_system_cal", "receptor_label"]
    if "frequency" not in generic_sys_cal_xds.sizes:
        dims_all = mandatory_dimensions
    else:
        dims_all = mandatory_dimensions + ["frequency_system_cal"]

    to_new_data_variables = {
        "PHASE_DIFF": ["PHASE_DIFFERENCE", ["antenna_name", "time_system_cal"]],
        "TCAL": ["TCAL", dims_all],
        "TCAL_SPECTRUM": ["TCAL", dims_all],
        "TRX": ["TRX", dims_all],
        "TRX_SPECTRUM": ["TRX", dims_all],
        "TSKY": ["TSKY", dims_all],
        "TSKY_SPECTRUM": ["TSKY", dims_all],
        "TSYS": ["TSYS", dims_all],
        "TSYS_SPECTRUM": ["TSYS", dims_all],
        "TANT": ["TANT", dims_all],
        "TANT_SPECTRUM": ["TANT", dims_all],
        "TAN_TSYS": ["TAN_TSYS", dims_all],
        "TANT_SYS_SPECTRUM": ["TANT_TSYS", dims_all],
    }

    to_new_coords = {
        "TIME": ["time_system_cal", ["time_system_cal"]],
        "receptor": ["receptor_label", ["receptor_label"]],
        "frequency": ["frequency_system_cal", ["frequency_system_cal"]],
    }

    sys_cal_xds = xr.Dataset(attrs={"type": "system_calibration"})
    ant_borrowed_coords = {
        "antenna_name": ant_xds.coords["antenna_name"],
        "receptor_label": ant_xds.coords["receptor_label"],
        "polarization_type": ant_xds.coords["polarization_type"],
    }
    sys_cal_xds = sys_cal_xds.assign_coords(ant_borrowed_coords)
    sys_cal_xds = convert_generic_xds_to_xradio_schema(
        generic_sys_cal_xds, sys_cal_xds, to_new_data_variables, to_new_coords
    )

    # Add frequency coord and its measures data, if present
    if "frequency_system_cal" in dims_all:
        frequency_coord = {
            "frequency_system_cal": generic_sys_cal_xds.coords["frequency"].data
        }
        sys_cal_xds = sys_cal_xds.assign_coords(frequency_coord)
        frequency_measure = {
            "type": main_xds_frequency.attrs["type"],
            "units": main_xds_frequency.attrs["units"],
            "observer": main_xds_frequency.attrs["observer"],
        }
        sys_cal_xds.coords["frequency_system_cal"].attrs.update(frequency_measure)

    sys_cal_xds = rename_and_interpolate_to_time(
        sys_cal_xds, "time_system_cal", sys_cal_interp_time, "system_calibration_xds"
    )

    # correct expected types
    for data_var in sys_cal_xds:
        if sys_cal_xds.data_vars[data_var].dtype != np.float64:
            sys_cal_xds[data_var] = sys_cal_xds[data_var].astype(np.float64)

    return sys_cal_xds


def create_phased_array_xds(
    in_file: str,
    antenna_names: list[str],
    receptor_label: list[str],
    polarization_type: ArrayLike,
) -> Optional[xr.Dataset]:
    """
    Create an Xarray Dataset containing phased array information.

    Parameters
    ----------
    in_file : str
        Path to the input MSv2.
    antenna_names: DataArray or Sequence[str]
        Content of the antenna_name coordinate of the antenna_xds.
    receptor_label: DataArray or Sequence[str]
        Content of the receptor_label coordinate of the antenna_xds. Used to
        label the corresponding axis of ELEMENT_FLAG.
    polarization_type: DataArray or ArrayLike
        Contents of the polarization_type coordinate of the antenna_xds.
        Array-like of shape (num_antennas, 2) containing the polarization
        hands for each antenna.

    Returns
    ----------
        xr.Dataset or None: If the input MS contains a PHASED_ARRAY table,
           returns the Xarray Dataset containing the phased array information.
           Otherwise, return None.
    """

    def extract_data(dataarray_or_sequence):
        if hasattr(dataarray_or_sequence, "data"):
            return dataarray_or_sequence.data.tolist()
        return dataarray_or_sequence

    antenna_names = extract_data(antenna_names)
    receptor_label = extract_data(receptor_label)
    polarization_type = extract_data(polarization_type)

    # NOTE: We cannot use the dimension renaming option of `load_generic_table`
    # here, because it leads to a dimension name collision. This is caused by
    # the presence of two dimensions of size 3 in multiple arrays.
    # Instead, we do the renaming manually below.
    try:
        raw_xds = load_generic_table(
            in_file,
            "PHASED_ARRAY",
            # Some MSes carry COORDINATE_SYSTEM as a copy of COORDINATE_AXES
            # due to a past ambiguity on the PHASED_ARRAY schema
            ignore=["COORDINATE_SYSTEM", "ANTENNA_ID"],
        )
    except ValueError:
        return None

    # Defend against empty PHASED_ARRAY table.
    # The test MS "AA2-Mid-sim_00000.ms" has that problem.
    required_keys = {"COORDINATE_AXES", "ELEMENT_OFFSET", "ELEMENT_FLAG"}
    if not all(k in raw_xds for k in required_keys):
        return None

    def msv4_measure(raw_name: str) -> dict:
        coldesc = raw_xds.attrs["other"]["msv2"]["ctds_attrs"]["column_descriptions"]
        return column_description_casacore_to_msv4_measure(coldesc[raw_name])

    def make_data_variable(raw_name: str, dim_names: list[str]) -> xr.DataArray:
        da = raw_xds[raw_name]
        da = xr.DataArray(da.data, dims=tuple(dim_names))
        return da.assign_attrs(msv4_measure(raw_name))

    raw_datavar_names_and_dims = [
        (
            "COORDINATE_AXES",
            ("antenna_name", "cartesian_pos_label_local", "cartesian_pos_label"),
        ),
        ("ELEMENT_OFFSET", ("antenna_name", "cartesian_pos_label_local", "element_id")),
        ("ELEMENT_FLAG", ("antenna_name", "receptor_label", "element_id")),
    ]

    data_vars = {
        name: make_data_variable(name, dims)
        for name, dims in raw_datavar_names_and_dims
    }
    data_vars["COORDINATE_AXES"].attrs = {
        "type": "rotation_matrix",
        "units": ["undimensioned", "undimensioned", "undimensioned"],
    }
    # Remove the "frame" attribute if it exists, because ELEMENT_OFFSET is
    # defined in a station-local frame for which no standard name exists
    data_vars["ELEMENT_OFFSET"].attrs.pop("frame", None)
    data_vars["ELEMENT_OFFSET"].attrs.update(
        {
            "coordinate_system": "topocentric",
            "origin": "ANTENNA_POSITION",
        }
    )

    num_elements = data_vars["ELEMENT_OFFSET"].sizes["element_id"]

    data_vars = {"PHASED_ARRAY_" + key: val for key, val in data_vars.items()}
    coords = {
        "antenna_name": antenna_names,
        "element_id": np.arange(num_elements),
        "receptor_label": receptor_label,
        "polarization_type": (
            ("antenna_name", "receptor_label"),
            polarization_type,
        ),
        "cartesian_pos_label": ["x", "y", "z"],
        "cartesian_pos_label_local": ["p", "q", "r"],
    }
    attrs = {"type": "phased_array"}
    return xr.Dataset(data_vars, coords, attrs)
