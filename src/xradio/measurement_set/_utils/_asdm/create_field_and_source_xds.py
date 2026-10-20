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

    xds = extract_source_info(xds, asdm, field_df, spectral_window_id)

    # TODO: ephemeris. SOURCE_DISTANCE, etc.
    ephemeris_id = field_df["ephemerisId"].item()
    if not pd.isna(ephemeris_id):
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
        line_label = [f"{idx}" for idx in np.arange(len(line_name))]
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
