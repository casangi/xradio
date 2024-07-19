import os
import time
from typing import Tuple, Union

import numpy as np
import xarray as xr

from xradio.vis._vis_utils._ms.msv2_to_msv4_meta import (
    column_description_casacore_to_msv4_measure,
)
from xradio.vis._vis_utils._ms.msv4_sub_xdss import interpolate_to_time
from xradio.vis._vis_utils._ms.subtables import subt_rename_ids
from xradio.vis._vis_utils._ms._tables.read import (
    convert_casacore_time_to_mjd,
    make_taql_where_between_min_max,
    read_generic_table,
)
from xradio.vis._vis_utils._ms._tables.table_query import open_table_ro
import graphviper.utils.logger as logger
from xradio._utils.list_and_array import (
    check_if_consistent,
    unique_1d,
    to_list,
    to_np_array,
)
from xradio._utils.common import cast_to_str, convert_to_si_units, add_position_offsets


def create_field_and_source_xds(
    in_file: str,
    field_id: list,
    spectral_window_id: int,
    field_times: list,
    is_single_dish: bool,
    time_min_max: Tuple[np.float64, np.float64],
    ephemeris_interp_time: Union[xr.DataArray, None] = None,
):
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
    time_min_max : Tuple[np.float64, np.float46]
        Min / max times to constrain loading (usually to the time range relevant to an MSv4)
    ephemeris_interp_time : Union[xr.DataArray, None]
        Time axis to interpolate the ephemeris data vars to (usually main MSv4 time)

    Returns:
    -------
    field_and_source_xds : xr.Dataset
        The xarray dataset containing the field and source information.
    """

    start_time = time.time()

    field_and_source_xds = xr.Dataset()

    field_and_source_xds, ephemeris_path, ephemeris_table_name, source_id = (
        create_field_info_and_check_ephemeris(
            field_and_source_xds, in_file, field_id, field_times, is_single_dish
        )
    )

    if field_and_source_xds.attrs["is_ephemeris"]:
        field_and_source_xds = extract_ephemeris_info(
            field_and_source_xds,
            ephemeris_path,
            ephemeris_table_name,
            is_single_dish,
            time_min_max,
            ephemeris_interp_time,
        )

    field_and_source_xds = extract_source_info(
        field_and_source_xds, in_file, source_id, spectral_window_id
    )

    logger.debug(
        f"create_field_and_source_xds() execution time {time.time() - start_time:0.2f} s"
    )

    return field_and_source_xds, source_id


def extract_ephemeris_info(
    xds,
    path,
    table_name,
    is_single_dish,
    time_min_max: Tuple[np.float64, np.float64],
    interp_time: Union[xr.DataArray, None],
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
    ephemeris_interp_time : Union[xr.DataArray, None]
        Time axis to interpolate the data vars to (usually main MSv4 time)

    Returns:
    -------
    xds : xr.Dataset
        The xarray dataset with the added ephemeris information.
    """
    # The JPL-Horizons ephemeris table implementation in CASA does not follow the standard way of defining measures.
    # Consequently a lot of hardcoding is needed to extract the information.
    # https://casadocs.readthedocs.io/en/latest/notebooks/external-data.html

    # Only read data between the min and max times of the visibility data in the MSv4.
    min_max_mjd = (
        convert_casacore_time_to_mjd(time_min_max[0]),
        convert_casacore_time_to_mjd(time_min_max[1]),
    )
    taql_time_range = make_taql_where_between_min_max(
        min_max_mjd, path, table_name, "MJD"
    )
    ephemeris_xds = read_generic_table(
        path, table_name, timecols=["MJD"], taql_where=taql_time_range
    )

    assert (
        len(ephemeris_xds.ephemeris_id) == 1
    ), "Non standard ephemeris table. Only a single ephemeris is allowed per MSv4."
    ephemeris_xds = ephemeris_xds.isel(
        ephemeris_id=0
    )  # Collapse the ephemeris_id dimension.

    # Get meta data.
    ephemeris_meta = ephemeris_xds.attrs["other"]["msv2"]["ctds_attrs"]
    ephemris_column_description = ephemeris_xds.attrs["other"]["msv2"]["ctds_attrs"][
        "column_descriptions"
    ]

    assert (
        ephemeris_meta["obsloc"] == "GEOCENTRIC"
    ), "Only geocentric observer ephemeris are supported."

    if "posrefsys" in ephemeris_meta:
        sky_coord_frame = ephemeris_meta["posrefsys"].replace("ICRF/", "")
    else:
        sky_coord_frame = "ICRS"  # We will have to just assume this.

    # Find out witch keyword is used for units (UNIT/QuantumUnits)
    if "UNIT" in ephemris_column_description["RA"]["keywords"]:
        unit_keyword = "UNIT"
    else:
        unit_keyword = "QuantumUnits"

    coords = {
        "ellipsoid_pos_label": ["lon", "lat", "dist"],
        "ephem_time": ephemeris_xds[
            "time"
        ].data,  # We are using the "ephem_time" label because it might not match the optional time axis of the source and field info. If ephemeris_interpolate=True then rename it to time.
        "sky_pos_label": ["ra", "dec", "dist"],
    }

    temp_xds = xr.Dataset()

    # Add mandatory data: SOURCE_POSITION
    temp_xds["SOURCE_POSITION"] = xr.DataArray(
        np.column_stack(
            (
                ephemeris_xds["ra"].data,
                ephemeris_xds["dec"].data,
                ephemeris_xds["rho"].data,
            )
        ),
        dims=["ephem_time", "sky_pos_label"],
    )
    # Have to use cast_to_str because the ephemeris table units are not consistently in a list or a string.
    sky_coord_units = [
        cast_to_str(ephemris_column_description["RA"]["keywords"][unit_keyword]),
        cast_to_str(ephemris_column_description["DEC"]["keywords"][unit_keyword]),
        cast_to_str(ephemris_column_description["Rho"]["keywords"][unit_keyword]),
    ]
    temp_xds["SOURCE_POSITION"].attrs.update(
        {"type": "sky_coord", "frame": sky_coord_frame, "units": sky_coord_units}
    )

    # Add mandatory data: SOURCE_RADIAL_VELOCITY
    temp_xds["SOURCE_RADIAL_VELOCITY"] = xr.DataArray(
        ephemeris_xds["radvel"].data, dims=["ephem_time"]
    )
    temp_xds["SOURCE_RADIAL_VELOCITY"].attrs.update(
        {
            "type": "quantity",
            "units": [
                cast_to_str(
                    ephemris_column_description["RadVel"]["keywords"][unit_keyword]
                )
            ],
        }
    )

    # Add mandatory data: OBSERVATION_POSITION
    observation_position = [
        ephemeris_meta["GeoLong"],
        ephemeris_meta["GeoLat"],
        ephemeris_meta["GeoDist"],
    ]
    temp_xds["OBSERVATION_POSITION"] = xr.DataArray(
        observation_position, dims=["ellipsoid_pos_label"]
    )
    temp_xds["OBSERVATION_POSITION"].attrs.update(
        {
            "type": "location",
            "units": ["deg", "deg", "m"],
            "data": observation_position,
            "ellipsoid": "WGS84",
            "origin_object_name": "Earth",
            "coordinate_system": ephemeris_meta["obsloc"].lower(),
        }
    )  # I think the units are ['deg','deg','m'] and 'WGS84'.

    # Add optional data NORTH_POLE_POSITION_ANGLE and NORTH_POLE_ANGULAR_DISTANCE
    if "np_ang" in ephemeris_xds.data_vars:
        temp_xds["NORTH_POLE_POSITION_ANGLE"] = xr.DataArray(
            ephemeris_xds["np_ang"].data, dims=["ephem_time"]
        )
        temp_xds["NORTH_POLE_POSITION_ANGLE"].attrs.update(
            {
                "type": "quantity",
                "units": [
                    cast_to_str(
                        ephemris_column_description["NP_ang"]["keywords"][unit_keyword]
                    )
                ],
            }
        )

    if "np_dist" in ephemeris_xds.data_vars:
        temp_xds["NORTH_POLE_ANGULAR_DISTANCE"] = xr.DataArray(
            ephemeris_xds["np_dist"].data, dims=["ephem_time"]
        )
        temp_xds["NORTH_POLE_ANGULAR_DISTANCE"].attrs.update(
            {
                "type": "quantity",
                "units": [
                    cast_to_str(
                        ephemris_column_description["NP_dist"]["keywords"][unit_keyword]
                    )
                ],
            }
        )

    # Add optional data: SUB_OBSERVER_POSITION and SUB_SOLAR_POSITION
    if "disklong" in ephemeris_xds.data_vars:
        temp_xds["SUB_OBSERVER_POSITION"] = xr.DataArray(
            np.column_stack(
                (
                    ephemeris_xds["disklong"].data,
                    ephemeris_xds["disklat"].data,
                    np.zeros(ephemeris_xds["disklong"].shape),
                )
            ),
            dims=["ephem_time", "ellipsoid_pos_label"],
        )

        if "DiskLong" in ephemris_column_description:
            units_key_lon = "DiskLong"
            units_key_lat = "DiskLat"
        else:
            units_key_lon = "diskLong"
            units_key_lat = "diskLat"

        temp_xds["SUB_OBSERVER_POSITION"].attrs.update(
            {
                "type": "location",
                "ellipsoid": "NA",
                "origin_object_name": ephemeris_meta["NAME"],
                "coordinate_system": "planetodetic",
                "units": [
                    cast_to_str(
                        ephemris_column_description[units_key_lon]["keywords"][
                            unit_keyword
                        ]
                    ),
                    cast_to_str(
                        ephemris_column_description[units_key_lat]["keywords"][
                            unit_keyword
                        ]
                    ),
                    "m",
                ],
            }
        )

    if "si_lon" in ephemeris_xds.data_vars:
        temp_xds["SUB_SOLAR_POSITION"] = xr.DataArray(
            np.column_stack(
                (
                    ephemeris_xds["si_lon"].data,
                    ephemeris_xds["si_lat"].data,
                    ephemeris_xds["r"].data,
                )
            ),
            dims=["ephem_time", "ellipsoid_pos_label"],
        )
        temp_xds["SUB_SOLAR_POSITION"].attrs.update(
            {
                "type": "location",
                "ellipsoid": "NA",
                "origin_object_name": "Sun",
                "coordinate_system": "planetodetic",
                "units": [
                    cast_to_str(
                        ephemris_column_description["SI_lon"]["keywords"][unit_keyword]
                    ),
                    cast_to_str(
                        ephemris_column_description["SI_lat"]["keywords"][unit_keyword]
                    ),
                    cast_to_str(
                        ephemris_column_description["r"]["keywords"][unit_keyword]
                    ),
                ],
            }
        )

    # Add optional data: HELIOCENTRIC_RADIAL_VELOCITY
    if "rdot" in ephemeris_xds.data_vars:
        temp_xds["HELIOCENTRIC_RADIAL_VELOCITY"] = xr.DataArray(
            ephemeris_xds["rdot"].data, dims=["ephem_time"]
        )
        temp_xds["HELIOCENTRIC_RADIAL_VELOCITY"].attrs.update(
            {
                "type": "quantity",
                "units": [
                    cast_to_str(
                        ephemris_column_description["rdot"]["keywords"][unit_keyword]
                    )
                ],
            }
        )

    # Add optional data: OBSERVER_PHASE_ANGLE
    if "phang" in ephemeris_xds.data_vars:
        temp_xds["OBSERVER_PHASE_ANGLE"] = xr.DataArray(
            ephemeris_xds["phang"].data, dims=["ephem_time"]
        )
        temp_xds["OBSERVER_PHASE_ANGLE"].attrs.update(
            {
                "type": "quantity",
                "units": [
                    cast_to_str(
                        ephemris_column_description["phang"]["keywords"][unit_keyword]
                    )
                ],
            }
        )

    temp_xds = temp_xds.assign_coords(coords)
    temp_xds["ephem_time"].attrs.update(
        {"type": "time", "units": ["s"], "scale": "UTC", "format": "UNIX"}
    )

    # Convert to si units and interpolate if ephemeris_interpolate=True:
    temp_xds = convert_to_si_units(temp_xds)
    temp_xds = interpolate_to_time(
        temp_xds, interp_time, "field_and_source_xds", time_name="ephem_time"
    )

    # If we interpolate rename the ephem_time axis to time.
    if interp_time is not None:
        temp_xds = temp_xds.swap_dims({"ephem_time": "time"}).drop_vars("ephem_time")

    xds = xr.merge([xds, temp_xds])

    # Add the SOURCE_POSITION to the FIELD_PHASE_CENTER or FIELD_REFERENCE_CENTER. Ephemeris obs: When loaded from the MSv2 field table the FIELD_REFERENCE_CENTER or FIELD_PHASE_CENTER only contain an offset from the SOURCE_POSITION.
    # We also need to add a distance dimension to the FIELD_PHASE_CENTER or FIELD_REFERENCE_CENTER to match the SOURCE_POSITION.
    # FIELD_PHASE_CENTER is used for interferometer data and FIELD_REFERENCE_CENTER is used for single dish data.
    if is_single_dish:
        center_dv = "FIELD_REFERENCE_CENTER"
    else:
        center_dv = "FIELD_PHASE_CENTER"

    if "time" in xds[center_dv].coords:
        assert (
            interp_time is not None
        ), 'ephemeris_interpolate must be True if there is ephemeris data and multiple fields (this will occur if "FIELD_ID" is not in partition_scheme).'

        xds[center_dv] = xr.DataArray(
            add_position_offsets(
                np.column_stack(
                    (xds[center_dv].values, np.zeros(xds[center_dv].values.shape[0]))
                ),
                xds["SOURCE_POSITION"].values,
            ),
            dims=[xds["SOURCE_POSITION"].dims[0], "sky_pos_label"],
        )
    else:
        xds[center_dv] = xr.DataArray(
            add_position_offsets(
                np.append(xds[center_dv].values, 0),
                xds["SOURCE_POSITION"].values,
            ),
            dims=[xds["SOURCE_POSITION"].dims[0], "sky_pos_label"],
        )

    xds["FIELD_PHASE_CENTER"].attrs.update(xds["SOURCE_POSITION"].attrs)

    return xds


def extract_source_info(xds, path, source_id, spectral_window_id):
    """
    Extracts source information from the given path and adds it to the xarray dataset.

    Parameters:
    ----------
    xds : xr.Dataset
        The xarray dataset to which the source information will be added.
    path : str
        The path to the input file.
    source_id : int
        The ID of the source.
    spectral_window_id : int
        The ID of the spectral window.

    Returns:
    -------
    xds : xr.Dataset
        The xarray dataset with the added source information.
    """

    is_ephemeris = xds.attrs[
        "is_ephemeris"
    ]  # If ephemeris data is present we ignore the SOURCE_DIRECTION in the source table.

    if all(source_id == -1):
        logger.warning(
            f"Source_id is -1. No source information will be included in the field_and_source_xds."
        )
        xds = xds.assign_coords(
            {"source_name": "Unknown"}
        )  # Need to add this for ps.summary() to work.
        return xds

    from xradio._utils.list_and_array import check_if_consistent, unique_1d

    unique_source_id = unique_1d(source_id)
    taql_where = f"where (SOURCE_ID IN [{','.join(map(str, unique_source_id))}]) AND (SPECTRAL_WINDOW_ID = {spectral_window_id})"

    source_xds = read_generic_table(
        path,
        "SOURCE",
        ignore=["SOURCE_MODEL"],  # Trying to read SOURCE_MODEL causes an error.
        taql_where=taql_where,
    )

    if len(source_xds.data_vars) == 0:  # The source xds is empty.
        logger.warning(
            f"SOURCE table empty for source_id {source_id} and spectral_window_id {spectral_window_id}."
        )
        xds = xds.assign_coords(
            {"source_name": "Unknown"}
        )  # Need to add this for ps.summary() to work.
        return xds

    assert (
        len(source_xds.spectral_window_id) == 1
    ), "Can only process source table with a single spectral_window_id for a given MSv4 partition."

    # This source table time is not the same as the time in the field_and_source_xds that is derived from the main MSv4 time axis.
    # The source_id maps to the time axis in the field_and_source_xds. That is why "if len(source_id) == 1" is used to check if there should be a time axis.
    assert (
        len(source_xds.time) == 1
    ), "Can only process source table with a single time entry for a source_id and spectral_window_id."

    source_xds = source_xds.isel(time=0, spectral_window_id=0, drop=True)
    source_column_description = source_xds.attrs["other"]["msv2"]["ctds_attrs"][
        "column_descriptions"
    ]

    # Get source name (the time axis is optional and will probably be required if the partition scheme does not include 'FIELD_ID' or 'SOURCE_ID'.).
    # Note again that this optional time axis has nothing to do with the original time axis in the source table that we drop.
    coords = {}
    if len(source_id) == 1:
        source_xds = source_xds.sel(source_id=source_id[0])
        coords["source_name"] = (
            source_xds["name"].values.item() + "_" + str(source_id[0])
        )
        direction_dims = ["sky_dir_label"]
        # coords["source_id"] = source_id[0]
    else:
        source_xds = source_xds.sel(source_id=source_id)
        coords["source_name"] = (
            "time",
            np.char.add(
                source_xds["name"].data, np.char.add("_", source_id.astype(str))
            ),
        )
        direction_dims = ["time", "sky_dir_label"]
        # coords["source_id"] = ("time", source_id)

    # If ephemeris data is present we ignore the SOURCE_DIRECTION.
    if not is_ephemeris:
        msv4_measure = column_description_casacore_to_msv4_measure(
            source_column_description["DIRECTION"]
        )
        xds["SOURCE_DIRECTION"] = xr.DataArray(
            source_xds["direction"].data, dims=direction_dims
        )
        xds["SOURCE_DIRECTION"].attrs.update(msv4_measure)

    # Do we have line data:
    if source_xds["num_lines"].data.ndim == 0:
        num_lines = np.array([source_xds["num_lines"].data.item()])
    else:
        num_lines = source_xds["num_lines"].data

    if any(num_lines > 0):

        if len(source_id) == 1:
            coords_lines = {"line_name": source_xds["transition"].data}
            xds = xds.assign_coords(coords_lines)
            line_dims = ["line_label"]
        else:
            coords_lines = {
                "line_name": (("time", "line_label"), source_xds["transition"].data)
            }
            xds = xds.assign_coords(coords_lines)
            line_dims = ["time", "line_label"]

        optional_data_variables = {
            "rest_frequency": "LINE_REST_FREQUENCY",
            "sysvel": "LINE_SYSTEMIC_VELOCITY",
        }
        for generic_name, msv4_name in optional_data_variables.items():
            if generic_name in source_xds:
                msv4_measure = column_description_casacore_to_msv4_measure(
                    source_column_description[generic_name.upper()]
                )

                xds[msv4_name] = xr.DataArray(
                    source_xds[generic_name].data, dims=line_dims
                )
                xds[msv4_name].attrs.update(msv4_measure)

    # Need to add doppler info if present. Add check.
    try:
        doppler_xds = read_generic_table(
            path,
            "DOPPLER",
        )
        assert (
            False
        ), "Doppler table present. Please open an issue on https://github.com/casangi/xradio/issues so that we can add support for this."
    except:
        pass

    xds = xds.assign_coords(coords)
    return xds


def create_field_info_and_check_ephemeris(
    field_and_source_xds, in_file, field_id, field_times, is_single_dish
):
    """
    Create field information and check for ephemeris in the FIELD table folder.

    Parameters:
    ----------
    field_and_source_xds : xr.Dataset
        The xarray dataset to which the field and source information will be added.
    in_file : str
        The path to the input file.
    field_id : int
        The ID of the field.

    Returns:
    -------
    field_and_source_xds : xr.Dataset
        The xarray dataset with the added field and source information.
    ephemeris_path : str
        The path to the ephemeris table.
    ephemeris_table_name : str
        The name of the ephemeris table.
    """

    # Federico do know how to do this taql query?
    unqiue_field_id = unique_1d(
        field_id
    )  # field_ids can be repeated so that the time mapping is correct if there are multiple fields. The read_generic_table required unique field_ids.
    taql_where = f"where (ROWID() IN [{','.join(map(str, unqiue_field_id))}])"
    field_xds = read_generic_table(
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
    field_xds = field_xds.assign_coords({"field_id": unqiue_field_id})
    field_xds = field_xds.sel(field_id=field_id, drop=False)
    source_id = to_np_array(field_xds.source_id.values)

    ephemeris_table_name = None
    ephemeris_path = None
    is_ephemeris = False
    field_and_source_xds.attrs["is_ephemeris"] = (
        False  # If we find a path to the ephemeris table we will set this to True.
    )

    # Need to check if ephemeris_id is present and if ephemeris table is present.
    if "ephemeris_id" in field_xds:
        ephemeris_id = check_if_consistent(
            field_xds.ephemeris_id, "ephemeris_id"
        )  # int(field_xds["ephemeris_id"].data)

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
                is_ephemeris = True
                e_index = ephemeris_name_table_index[0]
                ephemeris_path = os.path.join(in_file, "FIELD")
                ephemeris_table_name = files[e_index]
                field_and_source_xds.attrs["is_ephemeris"] = True
            else:
                logger.warning(
                    f"Could not find ephemeris table for field_id {field_id}. Ephemeris information will not be included in the field_and_source_xds."
                )

    if is_single_dish:
        field_data_variables = {
            "reference_dir": "FIELD_REFERENCE_CENTER",
        }
    else:
        field_data_variables = {
            # "delay_dir": "FIELD_DELAY_CENTER",
            "phase_dir": "FIELD_PHASE_CENTER",
            # "reference_dir": "FIELD_REFERENCE_CENTER",
        }

    field_measures_type = "sky_coord"

    coords = {}
    coords["sky_dir_label"] = ["ra", "dec"]
    field_column_description = field_xds.attrs["other"]["msv2"]["ctds_attrs"][
        "column_descriptions"
    ]

    coords = {}
    # field_times is the same as the time axis in the main MSv4 dataset and is used if more than one field is present.
    if field_times is not None:
        coords["time"] = field_times
        dims = ["time", "sky_dir_label"]
        coords["field_name"] = (
            "time",
            np.char.add(field_xds["name"].data, np.char.add("_", field_id.astype(str))),
        )
        # coords["field_id"] = ("time", field_id)
    else:
        coords["field_name"] = field_xds["name"].values.item() + "_" + str(field_id)
        # coords["field_id"] = field_id
        dims = ["sky_dir_label"]

    for generic_name, msv4_name in field_data_variables.items():

        if field_xds.get("delaydir_ref") is None:
            delaydir_ref = None
        else:
            delaydir_ref = check_if_consistent(
                field_xds.get("delaydir_ref"), "delaydir_ref"
            )
        msv4_measure = column_description_casacore_to_msv4_measure(
            field_column_description[generic_name.upper()], ref_code=delaydir_ref
        )

        field_and_source_xds[msv4_name] = xr.DataArray.from_dict(
            {
                "dims": dims,
                "data": list(field_xds[generic_name].data),
                "attrs": msv4_measure,
            }
        )

        field_and_source_xds[msv4_name].attrs["type"] = field_measures_type

    field_and_source_xds = field_and_source_xds.assign_coords(coords)
    return field_and_source_xds, ephemeris_path, ephemeris_table_name, source_id
