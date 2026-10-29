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
from xradio._utils.list_and_array import check_if_consistent, to_np_array
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
    if field_df.empty:
        return xds

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

    direction_attribute = "referenceDir" if is_single_dish else "phaseDir"
    center_direction = field_df[direction_attribute].to_numpy()
    # ignoring the polynomial dimension / never seen in ALMA?
    xds[center_direction_dv] = (
        ["field_name", "sky_dir_label"],
        np.vstack(center_direction),
    )

    direction_frame = retrieve_field_direction_frame(field_df, field_id)
    xds.data_vars[center_direction_dv].attrs.update(
        make_sky_coord_measure_attrs("rad", direction_frame)
    )

    xds = extract_source_info(xds, asdm, field_df, spectral_window_id)

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

    return xds


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
        unknown = to_np_array(["Unknown"] * field_df.shape[0])
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
    source_df = source_df.loc[
        (source_df["spectralWindowId"] == spectral_window_id)
        & (source_df["sourceId"].isin(np.array(source_id)))
    ]

    source_name = source_df["sourceName"].to_numpy(dtype="str")
    # Broadcast required for n_fields:1 source
    source_coords = {
        "source_name": (
            "field_name",
            np.broadcast_to(source_name, xds.coords["field_name"].shape),
        ),
    }
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
        np.array(direction_code), "source direction code"
    )
    # Again, for n_fields:1 source
    shape_field_name_sky_dir_label = xds.coords["field_name"].shape + (2,)
    xds["SOURCE_DIRECTION"] = (
        ["field_name", "sky_dir_label"],
        np.broadcast_to(source_direction, shape_field_name_sky_dir_label),
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
        shape_field_name_line_label = xds.coords["field_name"].shape + (
            len(line_label),
        )
        line_name = np.broadcast_to(line_name, shape_field_name_line_label)
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
            np.broadcast_to(rest_freq, shape_field_name_line_label),
            make_spectral_coord_measure_attrs("Hz", observer=frequency_frame),
        )
        sys_vel = source_df["sysVel"].explode().to_numpy(dtype="float64")[np.newaxis, :]
        xds["LINE_SYSTEMIC_VELOCITY"] = (
            ["field_name", "line_label"],
            np.broadcast_to(sys_vel, shape_field_name_line_label),
            make_quantity_attrs("m/s"),
        )

    return xds
