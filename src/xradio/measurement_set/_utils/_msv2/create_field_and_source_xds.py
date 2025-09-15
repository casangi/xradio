import os
import time
from typing import Tuple, Union

import numpy as np
import xarray as xr

import toolviper.utils.logger as logger
from xradio.measurement_set._utils._msv2.msv4_sub_xdss import (
    interpolate_to_time,
    rename_and_interpolate_to_time,
    standard_time_coord_attrs,
)
from xradio.measurement_set._utils._msv2.subtables import subt_rename_ids
from xradio.measurement_set._utils._msv2._tables.read import (
    convert_casacore_time_to_mjd,
    make_taql_where_between_min_max,
    load_generic_table,
)
from xradio._utils.list_and_array import cast_to_str, get_pad_value
from xradio._utils.dict_helpers import make_quantity_attrs
from xradio._utils.coord_math import (
    convert_to_si_units,
    wrap_to_pi,
)

from xradio._utils.list_and_array import (
    check_if_consistent,
    unique_1d,
    to_np_array,
)
from xradio._utils.schema import (
    casacore_to_msv4_measure_type,
    column_description_casacore_to_msv4_measure,
    convert_generic_xds_to_xradio_schema,
)


def create_field_and_source_xds(
    in_file: str,
    field_id: list,
    spectral_window_id: int,
    field_times: list,
    is_single_dish: bool,
    time_min_max: Tuple[np.float64, np.float64],
    ephemeris_interpolate: bool = True,
) -> tuple[xr.Dataset, np.ndarray, int]:
    """
    Create a field and source xarray dataset (xds) from the given input file, field ID, and spectral window ID.
    Data is extracted from the FIELD and SOURCE tables and if there is ephemeris data, it is also extracted.
    field_id will only be guaranteed to be a list of length 1 when partition_scheme does not include "FIELD_ID".

    Parameters:
    ----------
    in_file : str
        The path to the input file.
    field_id : list of int
        The field ids to select.
    spectral_window_id : int
        The ID of the spectral window.
    field_times: list
        Time data for field. It is the same as the time axis in the main MSv4 dataset.
    is_single_dish: bool
        whether the main xds has single-dish (SPECTRUM) data
    time_min_max : Tuple[np.float64, np.float46]
        Min / max times to constrain loading (usually to the time range relevant to an MSv4)
    ephemeris_interpolate : bool
        If true ephemeris data is interpolated to the main MSv4 time axis given in field_times.

    Returns:
    -------
    field_and_source_xds : xr.Dataset
        The xarray dataset containing the field and source information.
    source_id : np.ndarray[int]
        Source ID(s) corresponding to the field(s)
    num_lines : int
        Sum of num_lines for all unique sources.
    """

    start_time = time.time()

    field_and_source_xds = xr.Dataset(attrs={"type": "field_and_source"})

    (
        field_and_source_xds,
        ephemeris_path,
        ephemeris_table_name,
        source_id,
        field_names,
    ) = extract_field_info_and_check_ephemeris(
        field_and_source_xds, in_file, field_id, field_times, is_single_dish
    )

    field_and_source_xds, num_lines = extract_source_info(
        field_and_source_xds, in_file, source_id, spectral_window_id
    )

    if field_and_source_xds.attrs["type"] == "field_and_source_ephemeris":
        field_and_source_xds = extract_ephemeris_info(
            field_and_source_xds,
            ephemeris_path,
            ephemeris_table_name,
            is_single_dish,
            time_min_max,
            field_times,
            field_names,
            ephemeris_interpolate,
        )

    logger.debug(
        f"create_field_and_source_xds() execution time {time.time() - start_time:0.2f} s"
    )

    # # Check if we can drop time axis. The phase centers are repeated.
    # if field_times is not None:
    #     if is_single_dish:
    #         center_dv = "FIELD_REFERENCE_CENTER"
    #     else:
    #         center_dv = "FIELD_PHASE_CENTER"

    #     if np.unique(field_and_source_xds[center_dv], axis=0).shape[0] == 1:
    #         field_and_source_xds = field_and_source_xds.isel(time=0).drop_vars("time")

    return field_and_source_xds, source_id, num_lines, field_names


def extract_ephemeris_info(
    xds,
    path,
    table_name,
    is_single_dish,
    time_min_max: Tuple[np.float64, np.float64],
    interp_time: Union[xr.DataArray, None],
    field_names: list,
    ephemeris_interpolate: bool = True,
):
    """
    Extracts ephemeris information from the given path and table name and adds it to the xarray dataset.

    Parameters:
    ----------
    xds : xr.Dataset
        The xarray dataset to which the ephemeris information will be added.
    path : str
        The path to the input file.
    table_name : str
        The name of the ephemeris table.
    time_min_max : Tuple[np.float46, np.float64]
        Min / max times to constrain loading (usually to the time range relevant to an MSv4)
    interp_time : Union[xr.DataArray, None]
        Time axis to interpolate the data vars to (usually main MSv4 time)

    Returns:
    -------
    xds : xr.Dataset
        The xarray dataset with the added ephemeris information.
    """
    # The JPL-Horizons ephemeris table implementation in CASA does not follow the standard way of defining measures.
    # Consequently a lot of hardcoding is needed to extract the information.
    # https://casadocs.readthedocs.io/en/latest/notebooks/external-data.html

    xds = xds.assign_coords(
        {"sky_dis_label": ["dist"], "cartesian_pos_label": ["x", "y", "z"]}
    )

    # Only read data between the min and max times of the visibility data in the MSv4.
    min_max_mjd = (
        convert_casacore_time_to_mjd(time_min_max[0]),
        convert_casacore_time_to_mjd(time_min_max[1]),
    )
    taql_time_range = make_taql_where_between_min_max(
        min_max_mjd, path, table_name, "MJD"
    )
    ephemeris_xds = load_generic_table(
        path, table_name, timecols=["MJD"], taql_where=taql_time_range
    )

    assert (
        len(ephemeris_xds.ephemeris_id) == 1
    ), "Non standard ephemeris table. Only a single ephemeris is allowed per MSv4."
    ephemeris_xds = ephemeris_xds.isel(
        ephemeris_id=0
    )  # Collapse the ephemeris_id dimension.
    # Data variables  ['time', 'RA', 'DEC', 'Rho', 'RadVel', 'NP_ang', 'NP_dist', 'DiskLong', 'DiskLat', 'Sl_lon', 'Sl_lat', 'r', 'rdot', 'phang']

    # Get meta data.
    ephemeris_meta = ephemeris_xds.attrs["other"]["msv2"]["ctds_attrs"]
    ephemeris_column_description = ephemeris_xds.attrs["other"]["msv2"]["ctds_attrs"][
        "column_descriptions"
    ]

    assert (
        ephemeris_meta["obsloc"] == "GEOCENTRIC"
    ), "Only geocentric observer ephemeris are supported."

    if "posrefsys" in ephemeris_meta:
        # Note the phase center can be given as "J2000" or "J2000.0"
        ref_frame = (
            ephemeris_meta["posrefsys"]
            .replace("ICRF/", "", 1)
            .replace("J2000.0", "J2000", 1)
        )
        if ref_frame in casacore_to_msv4_measure_type["direction"].get("Ref_map", {}):
            ref_frame = casacore_to_msv4_measure_type["direction"]["Ref_map"][ref_frame]
        else:
            logger.debug(
                f"Unrecognized casacore direction reference frame found in posrefsys: {ref_frame}"
            )
        sky_coord_frame = ref_frame.lower()
    else:
        sky_coord_frame = "icrs"  # We will have to just assume this.

    # Find out witch keyword is used for units (UNIT/QuantumUnits)
    if "UNIT" in ephemeris_column_description["RA"]["keywords"]:
        unit_keyword = "UNIT"
    else:
        unit_keyword = "QuantumUnits"

    temp_xds = xr.Dataset()

    # Add mandatory data: OBSERVER_POSITION
    # Convert Observer position to geodetic coordinates
    from astropy import units as u
    from astropy.coordinates import EarthLocation
    from astropy.time import Time

    # Create an EarthLocation object for the geodetic coordinates
    # Assume that geodetic coordinates are given in degrees and meters
    location = EarthLocation.from_geodetic(
        lon=ephemeris_meta["GeoLong"] * u.deg,
        lat=ephemeris_meta["GeoLat"] * u.deg,
        height=ephemeris_meta["GeoDist"] * u.m,
        ellipsoid="WGS84",  # Explicitly specify WGS84
    )

    # Get the ITRS Cartesian coordinates (x, y, z)
    observer_position = location.itrs.cartesian.xyz

    temp_xds["OBSERVER_POSITION"] = xr.DataArray(
        observer_position, dims=["cartesian_pos_label"]
    )
    temp_xds["OBSERVER_POSITION"].attrs.update(
        {
            "type": "location",
            "units": "m",
            "frame": "ITRS",
            "origin_object_name": "Earth",
            "coordinate_system": ephemeris_meta["obsloc"].lower(),
        }
    )

    temp_xds["SOURCE_DIRECTION"] = xr.DataArray(
        np.column_stack(
            (
                ephemeris_xds["RA"].data,
                ephemeris_xds["DEC"].data,
            )
        ),
        dims=["time_ephemeris", "sky_dir_label"],
    )

    temp_xds["SOURCE_DISTANCE"] = xr.DataArray(
        np.column_stack((ephemeris_xds["Rho"].data,)),
        dims=["time_ephemeris", "sky_dis_label"],
    )

    # Have to use cast_to_str because the ephemeris table units are not consistently in a list or a string.
    sky_coord_units = cast_to_str(
        ephemeris_column_description["RA"]["keywords"][unit_keyword]
    )

    temp_xds["SOURCE_DIRECTION"].attrs.update(
        {"type": "sky_coord", "frame": sky_coord_frame, "units": sky_coord_units}
    )

    sky_coord_units = cast_to_str(
        ephemeris_column_description["Rho"]["keywords"][unit_keyword]
    )
    temp_xds["SOURCE_DISTANCE"].attrs.update(
        {"type": "sky_coord", "frame": sky_coord_frame, "units": sky_coord_units}
    )

    # Convert a few columns/variables that can be converted with standard
    # convert_generic_xds_to_xradio_schema().
    # Metadata has to be fixed manually. Alternatively, issues like
    # UNIT/QuantumUnits issue could be handled in convert_generic_xds_to_xradio_schema,
    # but for now preferring not to pollute that function.
    time_ephemeris_dim = ["time_ephemeris"]
    to_new_data_variables = {
        # mandatory: SOURCE_RADIAL_VELOCITY
        "RadVel": ["SOURCE_RADIAL_VELOCITY", time_ephemeris_dim],
        # optional: data NORTH_POLE_POSITION_ANGLE and NORTH_POLE_ANGULAR_DISTANCE
        "NP_ang": ["NORTH_POLE_POSITION_ANGLE", time_ephemeris_dim],
        "NP_dist": ["NORTH_POLE_ANGULAR_DISTANCE", time_ephemeris_dim],
        # optional: HELIOCENTRIC_RADIAL_VELOCITY
        "rdot": ["HELIOCENTRIC_RADIAL_VELOCITY", time_ephemeris_dim],
        # optional: OBSERVER_PHASE_ANGLE
        "phang": ["OBSERVER_PHASE_ANGLE", time_ephemeris_dim],
    }
    convert_generic_xds_to_xradio_schema(
        ephemeris_xds, temp_xds, to_new_data_variables, {}
    )

    # Adjust metadata:
    for generic_var_name, msv4_variable_def in to_new_data_variables.items():
        msv4_var_name = msv4_variable_def[0]
        if msv4_var_name in temp_xds:
            temp_xds[msv4_var_name].attrs.update(
                make_quantity_attrs(
                    cast_to_str(
                        ephemeris_column_description[generic_var_name]["keywords"][
                            unit_keyword
                        ]
                    )
                )
            )

    # Add optional data: SUB_OBSERVER_DIRECTION and SUB_SOLAR_POSITION
    if "DiskLong" in ephemeris_column_description:
        key_lon = "DiskLong"
        key_lat = "DiskLat"
    else:
        key_lon = "diskLong"
        key_lat = "diskLat"

    if key_lon in ephemeris_xds.data_vars:
        temp_xds["SUB_OBSERVER_DIRECTION"] = xr.DataArray(
            np.column_stack(
                (
                    ephemeris_xds[key_lon].data,
                    ephemeris_xds[key_lat].data,
                )
            ),
            dims=["time_ephemeris", "ellipsoid_dir_label"],
        )

        temp_xds["SUB_OBSERVER_DIRECTION"].attrs.update(
            {
                "type": "location",
                "frame": "Undefined",
                "origin_object_name": ephemeris_meta["NAME"],
                "coordinate_system": "planetodetic",
                "units": cast_to_str(
                    ephemeris_column_description[key_lon]["keywords"][unit_keyword]
                ),
            }
        )

    if "SI_lon" in ephemeris_xds.data_vars:
        temp_xds["SUB_SOLAR_DIRECTION"] = xr.DataArray(
            np.column_stack(
                (
                    ephemeris_xds["SI_lon"].data,
                    ephemeris_xds["SI_lat"].data,
                )
            ),
            dims=["time_ephemeris", "ellipsoid_dir_label"],
        )
        temp_xds["SUB_SOLAR_DIRECTION"].attrs.update(
            {
                "type": "location",
                "frame": "Undefined",
                "origin_object_name": "Sun",
                "coordinate_system": "planetodetic",
                "units": cast_to_str(
                    ephemeris_column_description["SI_lon"]["keywords"][unit_keyword]
                ),
            }
        )

        temp_xds["SUB_SOLAR_DISTANCE"] = xr.DataArray(
            [ephemeris_xds["r"].data],
            dims=["time_ephemeris", "ellipsoid_dis_label"],
        )
        temp_xds["SUB_SOLAR_DISTANCE"].attrs.update(
            {
                "type": "location",
                "frame": "Undefined",
                "origin_object_name": "Sun",
                "coordinate_system": "planetodetic",
                "units": cast_to_str(
                    ephemeris_column_description["r"]["keywords"][unit_keyword]
                ),
            }
        )

    # We are using the "time_ephemeris" label because it might not match the optional time axis of the source and field info. If ephemeris_interpolate=True then rename it to time.
    coords = {
        "ellipsoid_dir_label": ["lon", "lat"],
        "ellipsoid_dis_label": ["dist"],
        "time_ephemeris": ephemeris_xds["time"].data,
    }
    temp_xds = temp_xds.assign_coords(coords)
    temp_xds["time_ephemeris"].attrs.update(standard_time_coord_attrs)

    # Convert to si units
    temp_xds = convert_to_si_units(temp_xds)

    # interpolate if ephemeris_interpolate/interp_time=True, and rename time_ephemeris=>time
    if ephemeris_interpolate:
        temp_xds = rename_and_interpolate_to_time(
            temp_xds, "time_ephemeris", interp_time, "field_and_source_xds"
        )
        source_direction_interp = temp_xds["SOURCE_DIRECTION"]
        source_distance_interp = temp_xds["SOURCE_DISTANCE"]

    else:
        source_direction_interp = interpolate_to_time(
            temp_xds["SOURCE_DIRECTION"],
            interp_time,
            "field_and_source_xds",
            "time_ephemeris",
        )
        source_distance_interp = interpolate_to_time(
            temp_xds["SOURCE_DISTANCE"],
            interp_time,
            "field_and_source_xds",
            "time_ephemeris",
        )

    xds = xr.merge([xds, temp_xds])

    # Add the SOURCE_DIRECTION to the FIELD_PHASE_CENTER or FIELD_REFERENCE_CENTER. Ephemeris obs: When loaded from the MSv2 field table the FIELD_REFERENCE_CENTER or FIELD_PHASE_CENTER only contain an offset from the SOURCE_DIRECTION.
    # We also need to add a distance dimension to the FIELD_PHASE_CENTER or FIELD_REFERENCE_CENTER to match the SOURCE_DIRECTION.
    # FIELD_PHASE_CENTER_DIRECTION is used for interferometer data and FIELD_REFERENCE_CENTER_DIRECTION is used for single dish data.
    if is_single_dish:
        center_dir_dv = "FIELD_REFERENCE_CENTER_DIRECTION"
        center_dis_dv = "FIELD_REFERENCE_CENTER_DISTANCE"
    else:
        center_dir_dv = "FIELD_PHASE_CENTER_DIRECTION"
        center_dis_dv = "FIELD_PHASE_CENTER_DISTANCE"

    xds = xds.sel(field_name=field_names)  # Expand for all times in ms
    xds = xds.assign_coords({"time": ("field_name", interp_time)})
    xds["time"].attrs.update(standard_time_coord_attrs)
    xds = xds.swap_dims({"field_name": "time"})

    field_phase_center_direction = wrap_to_pi(
        xds[center_dir_dv].values + source_direction_interp.values
    )

    xds[center_dir_dv] = xr.DataArray(
        field_phase_center_direction,
        dims=["time", "sky_dir_label"],
    )

    xds[center_dis_dv] = xr.DataArray(
        source_distance_interp.values,
        dims=["time", "sky_dis_label"],
    )

    xds[center_dir_dv].attrs.update(xds["SOURCE_DIRECTION"].attrs)

    xds[center_dis_dv].attrs.update(xds["SOURCE_DISTANCE"].attrs)

    return xds


def make_line_dims_and_coords(
    source_xds: xr.Dataset, source_id: Union[int, np.ndarray], num_lines: int
) -> tuple[list, dict]:
    """
    Produces the dimensions and coordinates used in data variables related
    to line information (LINE_REST_FREQUENCY, LINE_SYSTEMIC_VELOCITY).

    In the dimensions, "time" is optional. To produce the points of the
    coordinates we need to look into the (optional) TRANSITION column or
    alternatively other columns (DIRECTION) to produce coordinates points of
    appropriate shape, given the "num_lines" "and source_id".

    Parameters:
    ----------
    source_xds: xr.Dataset
        generic source xarray dataset
    source_id: Union[int, np.ndarray]
        source_id of the dataset, when it is an array that indicates the
        presence of the "time" dimension
    num_line: int
        number of lines in the source dataset

    Returns:
    -------
    tuple : tuple[list, dict]
        The dimensions and coordinates to use with line data variables. The
        dimensions are produced as a list of dimension names, and the
        coordinates as a dict for xarray coords.
    """

    # Transition is an optional column and occasionally not populated
    if "TRANSITION" in source_xds.data_vars:
        transition_var_data = source_xds["TRANSITION"].data
    else:
        transition_var_data = np.zeros(source_xds["DIRECTION"].shape, dtype="str")

    # if TRANSITION is left empty (or otherwise incomplete), and num_lines > 1,
    # the data_vars expect a "num_lines" size in the last dimension
    vars_shape = transition_var_data.shape[:-1] + (np.max(num_lines),)
    if transition_var_data.shape == vars_shape:
        coords_lines_data = transition_var_data
    else:
        coords_lines_data = np.broadcast_to(
            transition_var_data, max(transition_var_data.shape, vars_shape)
        )

    line_label_data = np.arange(coords_lines_data.shape[-1]).astype(str)

    line_coords = {
        "line_name": (("field_name", "line_label"), coords_lines_data),
        "line_label": line_label_data,
    }
    line_dims = ["field_name", "line_label"]

    return line_dims, line_coords


def pad_missing_sources(
    source_xds: xr.Dataset, unique_source_ids: np.array
) -> xr.Dataset:
    """
    In some MSs there can be source IDs referenced from the field subtable which do not exist in
    the source table: https://github.com/casangi/xradio/issues/266

    This addresses the issue by padding/filling those IDs with "Unknown"/nan values. Produces a
    source_xds that, in addition to the information loaded for the non-missing source IDs, has
    padding for the IDs that are missing from the input MSv2 source table.
    This function does not need to do anything when unique_source_ids is a single value
    (partitioning by "FIELD_ID" or othwerwise single field/source)

    Parameters:
    ----------
    xds: xr.Dataset
        source dataset to fix/pad missing sources
    unique_source_ids: np.array
        IDs of the sources included in this partition

    Returns:
    -------
    filled_source_xds : xr.Dataset
        source dataset with padding in the originally missing sources
    """

    # Only fill gaps in multi-source xdss. If single source_id, no need to
    if len(unique_source_ids) <= 1:
        return source_xds

    missing_source_ids = [
        source_id
        for source_id in unique_source_ids
        if source_id not in source_xds.coords["SOURCE_ID"]
    ]
    if len(missing_source_ids) < 1:
        return source_xds

    # would like to use the new-ish xr.pad, but it creates issues with indices/coords and is
    # also not free of overheads, as it for example changes all numeric types to float64
    fill_value = {
        var_name: get_pad_value(var.dtype)
        for var_name, var in source_xds.data_vars.items()
    }
    missing_source_xds = xr.full_like(source_xds.isel(SOURCE_ID=0), fill_value)
    pad_str = "Unknown"
    pad_str_type = "<U9"
    for var in missing_source_xds.data_vars:
        if np.issubdtype(missing_source_xds.data_vars[var].dtype, np.str_):
            # Avoid truncation to length of previously loaded strings
            missing_source_xds[var] = missing_source_xds[var].astype(
                np.dtype(pad_str_type)
            )
            missing_source_xds[var] = pad_str

    concat_dim = "SOURCE_ID"
    xdss_to_concat = [source_xds]
    for missing_id in missing_source_ids:
        missing_source_xds[concat_dim] = missing_id
        xdss_to_concat.append(missing_source_xds)
    filled_source_xds = xr.concat(xdss_to_concat, concat_dim, join="outer").sortby(
        concat_dim
    )

    return filled_source_xds


def extract_source_info(
    xds: xr.Dataset,
    path: str,
    source_id: Union[int, np.ndarray],
    spectral_window_id: int,
) -> tuple[xr.Dataset, int]:
    """
    Extracts source information from the given path and adds it to the xarray dataset.

    Parameters:
    ----------
    xds : xr.Dataset
        The xarray dataset to which the source information will be added.
    path : str
        The path to the input file.
    source_id : Union[int, np.ndarray]
        The ID of the source.
    spectral_window_id : int
        The ID of the spectral window.

    Returns:
    -------
    xds : xr.Dataset
        The xarray dataset with the added source information.
    num_lines : int
        Sum of num_lines for all unique sources extracted.
    """
    unknown = to_np_array(["Unknown"] * len(source_id))

    coords = {}

    if all(source_id == -1):
        logger.warning(
            "Source_id is -1. No source information will be included in the field_and_source_xds."
        )
        xds = xds.assign_coords(
            {"source_name": ("field_name", unknown)}
        )  # Need to add this for ps.summary() to work.
        return xds, 0

    if not os.path.isdir(os.path.join(path, "SOURCE")):
        logger.warning(
            f"Could not find SOURCE table for source_id {source_id}. Source information will not be included in the field_and_source_xds."
        )
        xds = xds.assign_coords({"source_name": ("field_name", unknown)})
        return xds, 0

    unique_source_id = unique_1d(source_id)
    taql_where = f"where (SOURCE_ID IN [{','.join(map(str, unique_source_id))}]) AND (SPECTRAL_WINDOW_ID = {spectral_window_id})"

    source_xds = load_generic_table(
        path,
        "SOURCE",
        ignore=["SOURCE_MODEL"],  # Trying to read SOURCE_MODEL causes an error.
        taql_where=taql_where,
    )

    if len(source_xds.data_vars) == 0:  # The source xds is empty.
        logger.warning(
            f"SOURCE table empty for (unique) source_id {unique_source_id} and spectral_window_id {spectral_window_id}."
        )
        xds = xds.assign_coords(
            {"source_name": ("field_name", unknown)}
        )  # Need to add this for ps.summary() to work.
        return xds, 0

    assert (
        len(source_xds.SPECTRAL_WINDOW_ID) == 1
    ), "Can only process source table with a single spectral_window_id for a given MSv4 partition."

    # This source table time is not the same as the time in the field_and_source_xds that is derived from the main MSv4 time axis.
    # The source_id maps to the time axis in the field_and_source_xds. That is why "if len(source_id) == 1" is used to check if there should be a time axis.
    # assert len(source_xds.TIME) <= len(
    #     unique_source_id
    # ), "Can only process source table with a single time entry for a source_id and spectral_window_id."
    if len(source_xds.TIME) > len(unique_source_id):
        logger.warning(
            "Source table has more than one time entry for a source_id and spectral_window_id. This is not currently supported. Only the first time entry will be used."
        )
        source_xds = source_xds.drop_duplicates("SOURCE_ID", keep="first")

    source_xds = source_xds.isel(TIME=0, SPECTRAL_WINDOW_ID=0, drop=True)
    source_column_description = source_xds.attrs["other"]["msv2"]["ctds_attrs"][
        "column_descriptions"
    ]

    source_xds = pad_missing_sources(source_xds, unique_source_id)

    # Get source name (the time axis is optional and will probably be required if the partition scheme does not include 'FIELD_ID' or 'SOURCE_ID'.).
    # Note again that this optional time axis has nothing to do with the original time axis in the source table that we drop.
    # if len(source_id) == 1:
    #     source_xds = source_xds.sel(SOURCE_ID=source_id[0])
    #     coords["source_name"] = (
    #         source_xds["NAME"].values.item() + "_" + str(source_id[0])
    #     )
    #     direction_dims = ["sky_dir_label"]
    #     # coords["source_id"] = source_id[0]
    # else:

    source_xds = source_xds.sel(SOURCE_ID=source_id)
    coords["source_name"] = (
        "field_name",
        np.char.add(source_xds["NAME"].data, np.char.add("_", source_id.astype(str))),
    )
    direction_dims = ["field_name", "sky_dir_label"]
    # coords["source_id"] = ("time", source_id)

    is_ephemeris = xds.attrs["type"] == "field_and_source_ephemeris"
    # If ephemeris data is present we ignore the SOURCE_DIRECTION in the source table.
    if not is_ephemeris:
        direction_msv2_col = "DIRECTION"

        msv2_direction_dims = source_xds[direction_msv2_col].dims
        if (
            len(msv2_direction_dims) == len(direction_dims) + 1
            and "dim_1" in msv2_direction_dims
            and "dim_2" in msv2_direction_dims
        ):
            # CASA simulator produces transposed direction values, adding an
            # unexpected dimension. Drop it (https://github.com/casangi/xradio/issues/#196)
            direction_var = source_xds[direction_msv2_col].isel(dim_1=0, drop=True)
        else:
            direction_var = source_xds[direction_msv2_col]

        # SOURCE_LOCATION (DIRECTION / sky_dir_label)
        location_msv4_measure = column_description_casacore_to_msv4_measure(
            source_column_description[direction_msv2_col]
        )
        xds["SOURCE_DIRECTION"] = xr.DataArray(
            direction_var.data, dims=direction_dims, attrs=location_msv4_measure
        )

    # Do we have line data:
    if source_xds["NUM_LINES"].data.ndim == 0:
        num_lines = np.array([source_xds["NUM_LINES"].data.item()])
    else:
        num_lines = source_xds["NUM_LINES"].data

    if any(num_lines > 0):
        line_dims, line_coords = make_line_dims_and_coords(
            source_xds, source_id, num_lines
        )

        xds = xds.assign_coords(line_coords)

        to_new_data_variables = {
            "REST_FREQUENCY": ["LINE_REST_FREQUENCY", line_dims],
            "SYSVEL": ["LINE_SYSTEMIC_VELOCITY", line_dims],
        }
        to_new_coords = {
            "TIME": ["field_name", ["field_name"]],
        }
        convert_generic_xds_to_xradio_schema(
            source_xds, xds, to_new_data_variables, to_new_coords
        )

    # Need to add doppler info if present. Add check.
    try:
        load_generic_table(
            path,
            "DOPPLER",
        )
        logger.warning(
            "Doppler table present. Please open an issue on "
            "https://github.com/casangi/xradio/issues so that we can add support for this."
        )
    except ValueError:
        pass

    xds = xds.assign_coords(coords)

    _, unique_source_ids_indices = np.unique(source_xds.SOURCE_ID, return_index=True)

    return xds, np.sum(num_lines[unique_source_ids_indices])


def extract_field_info_and_check_ephemeris(
    field_and_source_xds: xr.Dataset,
    in_file: str,
    field_id: Union[int, np.ndarray],
    field_times: list,
    is_single_dish: bool,
):
    """
    Create field information and check for ephemeris in the FIELD table folder.

    Parameters:
    ----------
    field_and_source_xds : xr.Dataset
        The xarray dataset to which the field and source information will be added.
    in_file : str
        The path to the input file.
    field_id : Union[int, np.ndarray]
        The ID of the field.
    field_times: list
        Time of the MSv4
    is_single_dish: bool
        Whether to extract single dish (FIELD_REFERENCE_CENTER) info

    Returns:
    -------
    field_and_source_xds : xr.Dataset
        The xarray dataset with the added field and source information.
    ephemeris_path : str
        The path to the ephemeris table.
    ephemeris_table_name : str
        The name of the ephemeris table.
    """

    unique_field_id = unique_1d(
        field_id
    )  # field_ids can be repeated so that the time mapping is correct if there are multiple fields. The load_generic_table required unique field_ids.
    taql_where = f"where (ROWID() IN [{','.join(map(str, unique_field_id))}])"
    field_xds = load_generic_table(
        in_file,
        "FIELD",
        rename_ids=subt_rename_ids["FIELD"],
        taql_where=taql_where,
    )

    assert (
        len(field_xds.poly_id) == 1
    ), "Polynomial field positions not supported. Please open an issue on https://github.com/casangi/xradio/issues so that we can add support for this."

    field_xds = field_xds.isel(poly_id=0, drop=True)

    # field_xds = field_xds.assign_coords({'field_id':field_xds['field_id'].data})
    field_xds = field_xds.assign_coords({"field_id": unique_field_id})
    # field_xds = field_xds.sel(
    #     field_id=to_np_array(field_id), drop=False
    # )  # Make sure field_id match up with time axis (duplicate fields are allowed).
    source_id = to_np_array(field_xds.SOURCE_ID.values)

    ephemeris_table_name = None
    ephemeris_path = None

    # Need to check if ephemeris_id is present and if ephemeris table is present.
    if "EPHEMERIS_ID" in field_xds:
        # Note: this assumes partition_scheme includes "FIELD_ID"
        ephemeris_id = check_if_consistent(field_xds.EPHEMERIS_ID, "EPHEMERIS_ID")

        if ephemeris_id > -1:
            files = os.listdir(os.path.join(in_file, "FIELD"))
            ephemeris_table_name_start = "EPHEM" + str(ephemeris_id)

            ephemeris_name_table_index = [
                i for i in range(len(files)) if ephemeris_table_name_start in files[i]
            ]

            assert len(ephemeris_name_table_index) == 1, (
                "More than one ephemeris table which starts with "
                + ephemeris_table_name_start
            )

            if len(ephemeris_name_table_index) > 0:  # Are there any ephemeris tables.
                e_index = ephemeris_name_table_index[0]
                ephemeris_path = os.path.join(in_file, "FIELD")
                ephemeris_table_name = files[e_index]
                field_and_source_xds.attrs["type"] = "field_and_source_ephemeris"
            else:
                logger.warning(
                    f"Could not find ephemeris table for field_id {field_id}. Ephemeris information will not be included in the field_and_source_xds."
                )
    from xradio._utils.schema import convert_generic_xds_to_xradio_schema

    if is_single_dish:
        to_new_data_variables = {
            "REFERENCE_DIR": [
                "FIELD_REFERENCE_CENTER_DIRECTION",
                ["field_name", "sky_dir_label"],
            ],
            "FIELD_ID": ["FIELD_ID", ["field_name"]],
        }
    else:
        to_new_data_variables = {
            "PHASE_DIR": [
                "FIELD_PHASE_CENTER_DIRECTION",
                ["field_name", "sky_dir_label"],
            ],
            # "DELAY_DIR": ["FIELD_DELAY_CENTER",["field_name", "sky_dir_label"]],
            # "REFERENCE_DIR": ["FIELD_REFERENCE_CENTER",["field_name", "sky_dir_label"]],
        }

    to_new_coords = {
        "NAME": ["field_name", ["field_name"]],
        "field_id": ["field_id", ["field_name"]],
    }

    delay_dir_ref_col = "DelayDir_Ref"
    if field_xds.get(delay_dir_ref_col) is None:
        ref_code = None
    else:
        ref_code = check_if_consistent(
            field_xds.get(delay_dir_ref_col), delay_dir_ref_col
        )

    field_and_source_xds = convert_generic_xds_to_xradio_schema(
        field_xds, field_and_source_xds, to_new_data_variables, to_new_coords, ref_code
    )

    # Some field names are not unique. We need to add the field_id to the field_name to make it unique.
    field_and_source_xds = field_and_source_xds.assign_coords(
        {
            "field_name": np.char.add(
                field_and_source_xds["field_name"].data,
                np.char.add("_", field_and_source_xds["field_id"].astype(str)),
            ),
            "sky_dir_label": ["ra", "dec"],
        }
    )

    temp = field_and_source_xds.set_xindex("field_id")
    field_names = temp.sel(field_id=field_id).field_name.data
    # field_id shouldn ot be in final xds, and no longer needed past this point
    field_and_source_xds = field_and_source_xds.drop_vars("field_id")

    return (
        field_and_source_xds,
        ephemeris_path,
        ephemeris_table_name,
        source_id,
        field_names,
    )
