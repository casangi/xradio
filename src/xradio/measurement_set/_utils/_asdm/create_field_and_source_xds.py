"""Create field and source metadata datasets from ASDM tables."""

from typing import Any

import numpy as np
import pandas as pd
import pyasdm
import xarray as xr
from astropy import units as u
from astropy.coordinates import EarthLocation

from xradio._utils.coord_math import wrap_to_pi
from xradio._utils.dict_helpers import (
    make_quantity_attrs,
    make_sky_coord_measure_attrs,
    make_spectral_coord_measure_attrs,
    make_time_measure_attrs,
)
from xradio._utils.list_and_array import to_np_array
from xradio._utils.logging import xradio_logger
from xradio.measurement_set._utils._asdm._utils.field_source import (
    get_source_direction_code,
    retrieve_field_direction_frame,
    retrieve_source_frequency_frame,
)
from xradio.measurement_set._utils._asdm._utils.metadata_tables import (
    exp_asdm_table_to_df,
)
from xradio.measurement_set._utils._asdm._utils.time import convert_time_asdm_to_unix
from xradio.measurement_set._utils._utils.interpolate import (
    interpolate_to_time,
    rename_and_interpolate_to_time,
)


def create_field_and_source_xds(
    asdm: pyasdm.ASDM,
    partition_descr: dict[str, Any],
    spectral_window_id: int,
    is_single_dish: bool,
    time_to_interpolate: xr.DataArray,  # | None = None,   # Needed for ephemeris/time coord
    ephemeris_interpolate: bool = True,
) -> xr.Dataset:
    """
    Build a field_and_source_xds dataset with field and source metadata from an ASDM.

    Extracts field and source information from the ASDM tables and creates an xarray
    dataset with coordinates and variables describing the field center, and source
    direction and spectral line information if available.
    Field rows are restricted to the IDs in ``partition_descr``. The dataset
    includes the phase center for interferometric observations or the reference
    center for single-dish observations.

    Parameters
    ----------
    asdm : pyasdm.ASDM
        The ASDM object to extract data from
    partition_descr : dict
        Partition description containing at least the ``fieldId`` values to include.
    spectral_window_id : int
        Spectral-window ID used to select source and spectral-line metadata.
    is_single_dish : bool
        If true, use the field reference direction; otherwise, use the phase
        direction.

    Returns
    -------
    xr.Dataset
        Dataset with ``field_name`` and ``sky_dir_label`` coordinates and a
        field center direction variable. Source coordinates and variables are
        included when source metadata is available; spectral-line coordinates
        and variables are included when line information is available. The
        dataset has ``type`` and ``is_ephemeris`` attributes.
    """

    xds = xr.Dataset(attrs={"type": "field_and_source"})

    sdm_field_required_attrs = ["fieldId", "fieldName", "phaseDir", "referenceDir"]
    sdm_field_optional_attrs = ["ephemerisId", "sourceId", "directionCode"]
    # sourceId is an opt attr, but so far always present in ALMA ASDMs. directionCode also seems to
    # always be present. ephemerisId is NOT always present.
    field_df = exp_asdm_table_to_df(
        asdm,
        "Field",
        sdm_field_required_attrs + sdm_field_optional_attrs,
        optional_per_row=True,
    )

    field_id = partition_descr["fieldId"]
    field_df = field_df.loc[field_df["fieldId"].isin(field_id)]

    field_name = (
        field_df["fieldName"].astype("str").str.strip()
        + "_"
        + field_df["fieldId"].astype("str")
    ).to_numpy(dtype="str")
    field_coords = {
        "sky_dir_label": ["ra", "dec"],
        "field_name": ("field_name", field_name),
    }
    xds = xds.assign_coords(field_coords)

    phase_or_reference = "REFERENCE" if is_single_dish else "PHASE"
    center_direction_dv = f"FIELD_{phase_or_reference}_CENTER_DIRECTION"

    if is_single_dish:
        # ignore the polynomial dimension
        phase_or_ref_dir = field_df["referenceDir"].to_numpy()[0]
    else:
        phase_or_ref_dir = field_df["phaseDir"].to_numpy()[0]
    xds[center_direction_dv] = (
        ["field_name", "sky_dir_label"],
        phase_or_ref_dir,
    )

    direction_frame = retrieve_field_direction_frame(field_df, field_id)
    xds.data_vars[center_direction_dv].attrs.update(
        make_sky_coord_measure_attrs("rad", direction_frame)
    )

    xds = _extract_source_info(xds, asdm, field_df, spectral_window_id)

    xds = _extract_ephemeris_info(
        xds, asdm, field_df, time_to_interpolate, ephemeris_interpolate
    )

    return xds


def _extract_source_info(
    xds: xr.Dataset, asdm: pyasdm.ASDM, field_df: pd.DataFrame, spectral_window_id: int
) -> xr.Dataset:
    """Add source and optional spectral-line metadata to a field dataset.

    Select Source-table rows matching the field rows and spectral-window ID,
    then add source names and direction to ``xds``. When line information is
    available, add line names, rest frequencies, and systemic velocities. If
    the field rows have no source IDs, assign ``"Unknown"`` source names and
    return without reading the Source table.

    Parameters
    ----------
    xds : xr.Dataset
        Dataset to augment, with a ``field_name`` dimension and a
        ``sky_dir_label`` coordinate.
    asdm : pyasdm.ASDM
        ASDM object from which to read Source-table metadata.
    field_df : pd.DataFrame
        Selected Field-table rows, including ``sourceId`` when available.
    spectral_window_id : int
        Spectral-window ID used to filter Source-table rows.

    Returns
    -------
    xr.Dataset
        The dataset augmented with source metadata when available.
    """

    if field_df.empty or field_df["sourceId"].isnull().all():
        xradio_logger().warning(
            "No sourceId found in the Field table. No source information will be included in field_and_source_xds."
        )
        unknown = to_np_array(["Unknown"] * field_df.shape[0]).astype("str")
        xds = xds.assign_coords({"source_name": ("field_name", unknown)})
        return xds

    source_df = _read_source_table(asdm)

    source_id = field_df["sourceId"].unique()
    source_df = source_df.loc[
        (source_df["spectralWindowId"] == spectral_window_id)
        & (source_df["sourceId"].isin(np.array(source_id)))
    ]

    source_name = source_df["sourceName"].to_numpy(dtype="str")
    source_coords = {
        "source_name": ("field_name", source_name),
    }
    xds = xds.assign_coords(source_coords)

    source_direction = source_df["direction"].values[0]
    source_key = (
        source_df["sourceId"].item(),
        source_df["timeInterval"].item(),
        source_df["spectralWindowId"].item(),
    )
    direction_code = get_source_direction_code(asdm, source_key)

    xds["SOURCE_DIRECTION"] = (
        ["field_name", "sky_dir_label"],
        [[source_direction[0].get(), source_direction[1].get()]],
        make_sky_coord_measure_attrs("rad", direction_code),
    )

    line_info_available = (
        pd.to_numeric(source_df["numLines"], errors="coerce") > 0
    ).any()
    if line_info_available:
        line_name = (
            source_df["transition"].explode().str.strip('"').to_numpy(dtype="str")
        )
        line_name = line_name[np.newaxis, :] if line_name.ndim < 2 else line_name
        line_label = [f"{idx}" for idx in np.arange(line_name.shape[-1])]
        line_coords = {
            "line_label": line_label,
            "line_name": (["field_name", "line_label"], line_name),
        }
        xds = xds.assign_coords(line_coords)

        rest_freq = (
            source_df["restFrequency"]
            .explode()
            .to_numpy(dtype="float64")[np.newaxis, :]
        )
        frequency_frame = retrieve_source_frequency_frame(source_df)
        xds["LINE_REST_FREQUENCY"] = (
            ["field_name", "line_label"],
            rest_freq,
            make_spectral_coord_measure_attrs("Hz", observer=frequency_frame),
        )
        sys_vel = source_df["sysVel"].explode().to_numpy(dtype="float64")[np.newaxis, :]
        xds["LINE_SYSTEMIC_VELOCITY"] = (
            ["field_name", "line_label"],
            sys_vel,
            make_quantity_attrs("m/s"),
        )

    return xds


def _read_source_table(asdm: pyasdm.ASDM) -> pd.DataFrame:
    """Read required and optional source metadata from the ASDM Source table.

    Parameters
    ----------
    asdm : pyasdm.ASDM
        ASDM object containing the Source table.

    Returns
    -------
    pd.DataFrame
        Source-table rows and the requested metadata columns.
    """

    sdm_source_required_attrs = [
        "sourceId",
        "timeInterval",
        "spectralWindowId",
        "direction",
        "sourceName",
    ]
    sdm_source_optional_attrs = [
        "directionCode",
        "numLines",
        "transition",
        "restFrequency",
        "sysVel",
        "frequencyRefCode",
    ]
    source_df = exp_asdm_table_to_df(
        asdm,
        "Source",
        sdm_source_required_attrs + sdm_source_optional_attrs,
        optional_per_row=True,
    )

    return source_df


def _extract_ephemeris_info(
    xds: xr.Dataset,
    asdm: pyasdm.ASDM,
    field_df: pd.DataFrame,
    time_to_interpolate: xr.DataArray,
    ephemeris_interpolate: bool = True,
) -> xr.Dataset:
    """Add ephemeris-based source coordinates and interpolate them if requested.

    If the selected fields do not reference an ephemeris, the dataset is
    returned unchanged. Otherwise, ephemeris time, observer position, source
    direction, distance, and radial velocity metadata are added. Source
    direction and distance are then either interpolated into the requested
    time coordinate or retained on the ephemeris time grid before being
    combined with the field center direction.

    From casadocs (https://casadocs.readthedocs.io/en/stable/notebooks/ephemeris_data.html):
    The ephemerides used by ALMA are originally for the observer location of ALMA (topocentric).
    They use the ICRS coordinate reference frame and typically have a time spacing of ten or
    twenty minutes.
    Note: in the importasdm task, the parameter convert_ephem2geo is True by default. It is
    passed to the sdm tool but never used there. The conversion takes place in the task, using
    casatassks/convertephem.py


    Parameters
    ----------
    xds : xr.Dataset
        Field and source dataset to augment.
    asdm : pyasdm.ASDM
        ASDM object containing the Ephemeris table.
    field_df : pd.DataFrame
        Selected Field-table rows, including ``ephemerisId``.
    time_to_interpolate : xr.DataArray
        Target time coordinate for ephemeris interpolation.
    ephemeris_interpolate : bool, default True
        Whether to interpolate the complete dataset to the target times.

    Returns
    -------
    xr.Dataset
        Dataset augmented with ephemeris-derived coordinates and variables.
    """

    ephemeris_id = field_df["ephemerisId"].item()
    if pd.isna(ephemeris_id):
        return xds

    xds.attrs["type"] = "field_and_source_ephemeris"

    ephemeris_df = _read_ephemeris_table(asdm)

    xds = _add_coord_ephemeris_time(xds, ephemeris_df)

    xds = _add_observer_position(xds, ephemeris_df)

    xds = _add_source_direction_and_distance(xds, ephemeris_df)

    xds = _add_source_radvel(xds, ephemeris_df)

    xds = _interpolate_and_sum_source_and_field_direction(
        xds, time_to_interpolate, ephemeris_interpolate
    )

    return xds


def _read_ephemeris_table(asdm: pyasdm.ASDM) -> pd.DataFrame:
    """Read required and optional metadata from the ASDM Ephemeris table.

    Parameters
    ----------
    asdm : pyasdm.ASDM
        ASDM object containing the Ephemeris table.

    Returns
    -------
    pd.DataFrame
        Ephemeris-table rows and the requested metadata columns.
    """

    # CASA's importasdm/sdm tool fills:
    # - MJD, (RA, DEC, Rho) (from "dir"),
    # - RadVel (if present, it is an optional parameter)
    # it also creates the columns diskLong and diskLast, leaving them unset/0
    sdm_ephemeris_required_attrs = [
        "timeInterval",
        "ephemerisId",
        "observerLocation",
        "equinoxEquator",
        "dir",
        "distance",
        # "timeOrigin",  # timeOrigin only needed to evaluate polynomials
        "numPolyDir",
    ]
    sdm_ephemeris_optional_attrs = [
        "radVel",
        "numPolyRadVel",
    ]

    ephemeris_df = exp_asdm_table_to_df(
        asdm,
        "Ephemeris",
        sdm_ephemeris_required_attrs + sdm_ephemeris_optional_attrs,
        optional_per_row=True,
    )

    return ephemeris_df


def _add_coord_ephemeris_time(
    xds: xr.Dataset, ephemeris_df: pd.DataFrame
) -> xr.Dataset:
    """Add ephemeris midpoint times as a coordinate to a dataset.

    Parameters
    ----------
    xds : xr.Dataset
        Dataset to augment.
    ephemeris_df : pd.DataFrame
        Ephemeris rows containing ASDM time intervals.

    Returns
    -------
    xr.Dataset
        Dataset with the ``time_ephemeris`` coordinate and its time metadata.
    """

    # pyasdm/TimeInterval has t_start, width (=> t_start+width=t_end)
    middle_points = (
        ephemeris_df["timeInterval"].apply(lambda x: x.getMidPoint().get()).to_numpy()
    )
    time_middle_point = convert_time_asdm_to_unix(middle_points)
    time_coord = {"time_ephemeris": time_middle_point}
    xds = xds.assign_coords(time_coord)

    # TODO: revise/settle the tai <=> utc/unix discrepancies together with other datasets (xradio/689)
    time_ephemeris_attrs = make_time_measure_attrs("s", "tai", time_format="unix")
    time_ephemeris_attrs["type"] = "time_ephemeris"
    xds["time_ephemeris"].attrs = time_ephemeris_attrs

    return xds


def _add_observer_position(xds: xr.Dataset, ephemeris_df: pd.DataFrame) -> xr.Dataset:
    """Add the ephemeris observer position as an Earth-centered Cartesian variable.

    Parameters
    ----------
    xds : xr.Dataset
        Dataset to augment.
    ephemeris_df : pd.DataFrame
        Ephemeris rows containing the observer location.

    Returns
    -------
    xr.Dataset
        Dataset with ``OBSERVER_POSITION`` and its location metadata.
    """

    observer_location = ephemeris_df["observerLocation"]
    if not (observer_location[0] == observer_location).all():
        xradio_logger().warning(
            "Observer location varies throughout the ephemeris table. This is not expected nor supported. "
            "taking the fist location anyway."
        )
    observer_location = observer_location[0]
    # ASDM docs for observerLocation: A triple with all its elements equal to 0.0 will mean that a geocentric
    # coordinate system is in use instead of a topocentric one.
    # (But importasdm compares only the third element (distance) against 0.0)
    #
    # usual/probably-always?: topocentric
    is_geocentric = (
        observer_location[0] == observer_location[1] == observer_location[2] == 0.0
    )

    location = EarthLocation.from_geodetic(
        lon=observer_location[0] * u.rad,
        lat=observer_location[1] * u.rad,
        height=observer_location[2] * u.m,
        ellipsoid="WGS84",
    )

    observer_position = location.itrs.cartesian.xyz.to_value(u.m)

    xds = xds.assign_coords(
        {"sky_dis_label": ["dist"], "cartesian_pos_label": ["x", "y", "z"]}
    )

    xds["OBSERVER_POSITION"] = (["cartesian_pos_label"], observer_position)
    coordinate_system = "topocentric" if not is_geocentric else "geocentric"
    observer_position_attrs = {
        "type": "location",
        "units": "m",
        "frame": "ITRS",
        "origin_object_name": "Earth",
        "coordinate_system": coordinate_system,
    }
    xds.data_vars["OBSERVER_POSITION"].attrs = observer_position_attrs

    return xds


def _add_source_direction_and_distance(
    xds: xr.Dataset, ephemeris_df: pd.DataFrame
) -> xr.Dataset:
    """Add ephemeris source direction and distance variables to a dataset.

    Direction and distance polynomial coefficients are reduced to their first
    coefficient, as higher-order polynomials are not supported.

    Note: importasdm takes equator 2000.0 as "ICRF/ICRS" => "icrs" for the source
    direction/distance frame

    Parameters
    ----------
    xds : xr.Dataset
        Dataset to augment.
    ephemeris_df : pd.DataFrame
        Ephemeris rows containing direction, distance, and equinox metadata.

    Returns
    -------
    xr.Dataset
        Dataset with ``SOURCE_DIRECTION`` and ``SOURCE_DISTANCE`` variables.

    """

    polynomial_direction = (
        pd.to_numeric(ephemeris_df["numPolyDir"], errors="coerce") > 1
    ).any()
    if polynomial_direction:
        xradio_logger().warning(
            "Directions given as polynomials of order > 1 (NumPolyDir). "
            "This is not supported. Ignoring coefficients other than 0"
        )
    direction = np.array([[x[0][0], x[0][1]] for x in ephemeris_df["dir"]])
    distance = ephemeris_df["distance"].str[0].to_numpy()

    sky_coord_units = "rad"  # Fixed in ASDM table
    expected_equator = 2000.0
    if (ephemeris_df["equinoxEquator"] == expected_equator).all():
        sky_coord_frame = "fk5"
    else:
        xradio_logger().warning(
            f"EquinoxEquator is not as expected: {expected_equator}. This is not expected nor supported. "
            "assuming icrs frame."
        )
        sky_coord_frame = "icrs"

    direction_attrs = {
        "type": "sky_coord",
        "frame": sky_coord_frame,
        "units": sky_coord_units,
    }
    xds["SOURCE_DIRECTION"] = (
        ["time_ephemeris", "sky_dir_label"],
        direction,
        direction_attrs,
    )

    distance_attrs = {
        "type": "sky_coord",
        "frame": sky_coord_frame,
        "units": "m",
    }
    xds["SOURCE_DISTANCE"] = (
        ["time_ephemeris", "sky_dis_label"],
        np.column_stack((distance,)),
        distance_attrs,
    )

    return xds


def _add_source_radvel(xds: xr.Dataset, ephemeris_df: pd.DataFrame) -> xr.Dataset:
    """Add ephemeris source radial velocity when it is available.

    Parameters
    ----------
    xds : xr.Dataset
        Dataset to augment.
    ephemeris_df : pd.DataFrame
        Ephemeris rows containing optional radial-velocity coefficients.

    Returns
    -------
    xr.Dataset
        Dataset with ``SOURCE_RADIAL_VELOCITY`` when radial velocity data
        exists; otherwise, the input dataset unchanged.
    """

    polynomial_radvel = (
        pd.to_numeric(ephemeris_df["numPolyRadVel"], errors="coerce") > 1
    ).any()
    if polynomial_radvel:
        xradio_logger().warning(
            "Radial velocities given as polynomials of order > 1 (NumPolyRadVel). "
            "This is not supported. Ignoring coefficients other than 0"
        )

    if not pd.isna(ephemeris_df["radVel"]).all():
        radvel = ephemeris_df["radVel"].str[0].to_numpy()
        xds["SOURCE_RADIAL_VELOCITY"] = (["time_ephemeris"], radvel)
        xds["SOURCE_RADIAL_VELOCITY"].attrs = {"type": "quantity", "units": "m/s"}

    return xds


def _interpolate_and_sum_source_and_field_direction(
    xds: xr.Dataset,
    time_to_interpolate: xr.DataArray,
    ephemeris_interpolate: bool = True,
) -> xr.Dataset:

    if ephemeris_interpolate:
        xds = rename_and_interpolate_to_time(
            xds, "time_ephemeris", time_to_interpolate, "field_and_source_xds"
        )
        source_direction_interpolated = xds["SOURCE_DIRECTION"]
        source_distance_interpolated = xds["SOURCE_DISTANCE"]

    else:
        source_direction_interpolated = interpolate_to_time(
            xds["SOURCE_DIRECTION"],
            time_to_interpolate,
            "field_and_source_xds",
            "time_ephemeris",
        )
        source_distance_interpolated = interpolate_to_time(
            xds["SOURCE_DISTANCE"],
            time_to_interpolate,
            "field_and_source_xds",
            "time_ephemeris",
        )

    phase_or_reference = (
        "PHASE" if "FIELD_PHASE_CENTER_DIRECTION" in xds.data_vars else "REFERENCE"
    )
    center_direction_name = f"FIELD_{phase_or_reference}_CENTER_DIRECTION"
    center_distance_name = f"FIELD_{phase_or_reference}_CENTER_DISTANCE"

    xds = xds.assign_coords({"time": ("field_name", time_to_interpolate)})
    # TODO: revise/settle the tai <=> utc/unix discrepancies together with other datasets
    # (xradio/689) - also check against "standard_time_coord_attrs"
    time_attrs = make_time_measure_attrs("s", "tai", time_format="unix")
    xds["time"].attrs = time_attrs
    xds = xds.swap_dims({"field_name": "time"})

    overall_field_center_direction = wrap_to_pi(
        xds[center_direction_name].data + source_direction_interpolated
    )

    xds[center_direction_name] = (
        ["time", "sky_dir_label"],
        overall_field_center_direction.data,
    )
    xds.data_vars[center_direction_name].attrs.update(xds["SOURCE_DIRECTION"].attrs)

    xds[center_distance_name] = (
        ["time", "sky_dis_label"],
        source_distance_interpolated.data,
    )
    xds.data_vars[center_distance_name].attrs.update(xds["SOURCE_DISTANCE"].attrs)

    return xds
