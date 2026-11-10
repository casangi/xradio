"""Create field and source xarray datasets from ASDM metadata."""

import time
from typing import Any

import numpy as np
import pandas as pd
import pyasdm
import xarray as xr

from xradio._utils.dict_helpers import (
    make_quantity_attrs,
    make_sky_coord_measure_attrs,
    make_spectral_coord_measure_attrs,
)
from xradio._utils.list_and_array import check_if_consistent
from xradio._utils.logging import xradio_logger
from xradio.measurement_set._utils._asdm._utils.field_source import (
    get_source_direction_code,
    retrieve_field_direction_frame,
    retrieve_source_frequency_frame,
)
from xradio.measurement_set._utils._asdm._utils.metadata_tables import (
    exp_asdm_table_to_df,
)


def create_field_and_source_xds(
    asdm: pyasdm.ASDM,
    partition_descr: dict[str, Any],
    spectral_window_id: int,
    is_single_dish: bool,
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

    start_time = time.time()

    # TODO: field_df => sourceId
    xds, field_df = extract_field_info_xds_and_df(
        asdm, partition_descr, spectral_window_id, is_single_dish
    )
    xds = extract_source_info(xds, asdm, field_df, spectral_window_id)

    xradio_logger().debug(
        f"create_field_and_source_xds(): execution time {time.time() - start_time:0.2f} s"
    )

    return xds


def extract_field_info_xds_and_df(
    asdm: pyasdm.ASDM,
    partition_descr: dict[str, Any],
    spectral_window_id: int,
    is_single_dish: bool,
) -> tuple[xr.Dataset, pd.DataFrame]:
    """Extract field metadata for the selected partition fields.

    Parameters
    ----------
    asdm : pyasdm.ASDM
        ASDM object containing the Field table.
    partition_descr : dict
        Partition description containing the field IDs to include.
    spectral_window_id : int
        Spectral-window ID associated with the partition.
    is_single_dish : bool
        Whether to use reference directions for single-dish data.

    Returns
    -------
    tuple of xr.Dataset and pd.DataFrame
        Dataset containing field coordinates and center directions, plus the
        selected Field-table rows.
    """
    xds = xr.Dataset(attrs={"type": "field_and_source"})

    sdm_field_required_attrs = [
        "fieldId",
        "numPoly",
        "fieldName",
        "phaseDir",
        "referenceDir",
    ]
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

    if (field_df["numPoly"] > 1).any():
        xradio_logger().warning(
            "Field directions as polynomials with more than one "
            "coefficient (numPoly > 1) are not supported. "
            "Keeping only the first coefficient. If this is a "
            "legitimate ASDM.Please report this as a bug/missing feature "
        )
        # keep only the first coefficient of the polynomial so that
        # we can generate something in field_and_source_xds
        for direction_var in ["phaseDir", "referenceDir"]:
            field_df[direction_var] = field_df[direction_var].apply(
                lambda x: x[0] if len(x) > 0 else None
            )

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
    # TODO: make distance variable once we have ephem data (below)
    # center_distance_dv = f"FIELD_{phase_or_reference}_CENTER_DISTANCE"

    center_direction = _make_field_center_direction(xds, field_df, is_single_dish)
    xds[center_direction_dv] = (
        ["field_name", "sky_dir_label"],
        center_direction,
    )
    direction_frame = [
        retrieve_field_direction_frame(field_df, single_id) for single_id in field_id
    ]
    direction_frame = check_if_consistent(
        np.array(direction_frame), "field direction code (frame)"
    )
    xds.data_vars[center_direction_dv].attrs.update(
        make_sky_coord_measure_attrs("rad", direction_frame)
    )

    # TODO: ephemeris. SOURCE_DISTANCE, etc.
    ephemeris_id = field_df["ephemerisId"]
    if not pd.isna(ephemeris_id).all():
        xradio_logger().info(
            f"Note: ephemeris data is not yet supported. Found an ephemeris id: {ephemeris_id} "
            f"for field id: {field_id} / name {field_name}. The source_and_field_xds will "
            "not contain ephemeris information."
        )

    is_ephemeris = False
    xds.attrs.update({"is_ephemeris": is_ephemeris})

    return xds, field_df


def _make_field_center_direction(
    xds: xr.Dataset, field_df: pd.DataFrame, is_single_dish: bool
) -> np.ndarray:
    """Return field center directions as an RA/Dec array.

    Parameters
    ----------
    xds : xr.Dataset
        Field dataset providing the expected empty-array shape.
    field_df : pd.DataFrame
        Selected Field-table rows.
    is_single_dish : bool
        Whether to use reference rather than phase directions.

    Returns
    -------
    np.ndarray
        Field center directions with shape ``(field, sky direction)``.
    """
    direction_attribute = "referenceDir" if is_single_dish else "phaseDir"
    center_direction = field_df[direction_attribute].to_numpy()
    if center_direction.size > 0:
        center_direction = np.vstack(center_direction)
    else:
        center_direction = np.empty(
            xds.coords["field_name"].shape + (2,), dtype="float64"
        )
    return center_direction


def extract_source_info(
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
        unknown = np.array(["Unknown"] * field_df.shape[0], dtype="str")
        xds = xds.assign_coords({"source_name": ("field_name", unknown)})
        return xds

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

    source_id = field_df["sourceId"].unique()
    # Intentionally take only first (and only) "timeInterval"
    # - never seen more than one interval in the ASDM/Source table
    # - multiple source intervals not considered in MSv4.
    source_df = source_df.loc[
        (source_df["spectralWindowId"] == spectral_window_id)
        & (source_df["sourceId"].isin(np.array(source_id)))
    ].drop_duplicates(subset=["sourceId", "spectralWindowId"], keep="first")
    # align by sourceId in the field_df order, for next steps
    source_df = pd.merge(field_df["sourceId"], source_df, on="sourceId")

    if len(source_df) != len(field_df):
        raise RuntimeError(
            "Unexpected inconsistency. Not all fields "
            "have a corresponding row in the Source table. "
            f"{spectral_window_id=}, {source_id=}.\n{field_df=}\n{source_df=}"
        )

    source_name = source_df["sourceName"].to_numpy(dtype="str")
    source_coords = {"source_name": ("field_name", source_name)}
    xds = xds.assign_coords(source_coords)

    source_direction = np.vstack(source_df["direction"].to_numpy())

    # Do we need all, other than to check uniqueness?
    direction_code = []
    for source_idx in range(len(source_direction)):
        source_idx_index = source_df.index[source_idx]
        source_key = (
            source_df.at[source_idx_index, "sourceId"],
            source_df.at[source_idx_index, "timeInterval"],
            source_df.at[source_idx_index, "spectralWindowId"],
        )
        direction_code.append(get_source_direction_code(asdm, source_key))

    direction_code = check_if_consistent(
        np.array(direction_code), "source direction code (frame)"
    )
    xds["SOURCE_DIRECTION"] = (
        ["field_name", "sky_dir_label"],
        source_direction,
        make_sky_coord_measure_attrs("rad", direction_code),
    )

    xds = extract_line_info(xds, source_df)

    return xds


def extract_line_info(xds: xr.Dataset, source_df: pd.DataFrame) -> xr.Dataset:
    """Add spectral-line metadata to a field dataset when available.

    The line count is taken from the maximum ``numLines`` value in
    ``source_df``. Line names, rest frequencies, and systemic velocities are
    padded to a common length and added along the ``field_name`` and
    ``line_label`` dimensions. If no source rows report lines, ``xds`` is
    returned unchanged.

    Parameters
    ----------
    xds : xr.Dataset
        Dataset to augment, with a ``field_name`` dimension.
    source_df : pd.DataFrame
        Source-table rows containing line counts and line metadata.

    Returns
    -------
    xr.Dataset
        Dataset with spectral-line coordinates and variables when available.
    """
    max_line_info_len = pd.to_numeric(source_df["numLines"], errors="coerce").max()
    line_info_available = max_line_info_len > 0
    if line_info_available:
        line_label = [f"{idx}" for idx in range(max_line_info_len)]

        line_name = _strip_and_pad_line_values(source_df["transition"])
        line_coords = {
            "line_label": line_label,
            "line_name": (["field_name", "line_label"], line_name),
        }
        xds = xds.assign_coords(line_coords)

        rest_freq = _pad_line_values(source_df["restFrequency"])
        frequency_frame = retrieve_source_frequency_frame(source_df)
        xds["LINE_REST_FREQUENCY"] = (
            ["field_name", "line_label"],
            rest_freq,
            make_spectral_coord_measure_attrs("Hz", observer=frequency_frame),
        )

        sys_vel = _pad_line_values(source_df["sysVel"])
        xds["LINE_SYSTEMIC_VELOCITY"] = (
            ["field_name", "line_label"],
            sys_vel,
            make_quantity_attrs("m/s"),
        )

    return xds


def _strip_and_pad_line_values(values: pd.Series) -> np.ndarray[str]:
    """Strip surrounding double quotes from line names and pad them by row.

    The list-valued entries in ``values`` are expanded into a two-dimensional
    array; shorter rows are padded with empty strings.

    Parameters
    ----------
    values : pd.Series
        Per-source lists of line-name strings.

    Returns
    -------
    np.ndarray
        Two-dimensional array of stripped line names.
    """
    # use .apply(pd.Series) to expand into separate columns the elements of the line values (list)
    # fill values are ""
    padded_ndarray = (
        values.apply(lambda lst: [s.strip('"') for s in lst])
        .apply(pd.Series)
        .fillna("")
        .to_numpy()
    )
    return padded_ndarray


def _pad_line_values(values: pd.Series) -> np.ndarray[np.float64]:
    """Expand per-source numeric line values into a padded two-dimensional array.

    Parameters
    ----------
    values : pd.Series
        Per-source lists of numeric line values.

    Returns
    -------
    np.ndarray
        Two-dimensional array with missing entries padded by ``NaN``.
    """
    # use .apply(pd.Series) to expand into separate columns the elements of the line values (list)
    # fill values are np.nan
    padded_ndarray = values.apply(pd.Series).to_numpy()
    return padded_ndarray
