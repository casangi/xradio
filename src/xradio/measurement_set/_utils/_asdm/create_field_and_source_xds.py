import numpy as np
import pyasdm
import xarray as xr

from xradio._utils.dict_helpers import (
    make_quantity_attrs,
    make_sky_coord_measure_attrs,
    make_spectral_coord_measure_attrs,
)
from xradio._utils.logging import xradio_logger
from xradio.measurement_set._utils._asdm._utils import calculate_uvw as uvw_module
from xradio.measurement_set._utils._asdm._utils.field_source import (
    direction_code_name,
    direction_code_to_frame,
    frequency_code_to_observer,
    get_optional_row_attr,
    make_field_name,
)

# source_name used when a field has no source (as in the MSv2 converter)
UNKNOWN_SOURCE_NAME = "Unknown"


def create_field_and_source_xds(
    asdm: pyasdm.ASDM,
    partition_descr: dict,
    spectral_window_id: int,
    is_single_dish: bool,
) -> xr.Dataset:
    """
    Create an xarray Dataset containing field and source information from an ASDM.
    This function extracts field and source information from an ASDM and creates an xarray
    Dataset with coordinates and variables describing the field positions, source directions,
    and spectral line information if available.

    The partition can have any number of fields: the ``field_name`` dimension has one
    entry per field id of the partition, in ascending field id order.

    Parameters
    ----------
    asdm : pyasdm.ASDM
        The ASDM object to extract data from
    partition_descr : dict
        Dictionary containing partition description with at least a 'fieldId' key
        (the field ids of the partition)
    spectral_window_id : int
        ID of the spectral window to filter source information
    is_single_dish : bool
        Flag indicating if data is from single dish observations. Affects which center
        direction variable is produced.

    Returns
    -------
    xr.Dataset
        Dataset containing field and source information with the following structure:
        - Coordinates:
            - sky_dir_label: ['ra', 'dec']
            - field_name: unique field names, ``"<Field.fieldName>_<fieldId>"``
            - source_name (field_name): Source.sourceName of the source of each field
              ("Unknown" when the field has no source for this spectral window)
            - line_label, line_name (field_name, line_label): (optional) spectral line
              labels ('0', '1', ...) and transition names
        - Data variables:
            - FIELD_PHASE_CENTER_DIRECTION (interferometric, from Field.phaseDir) or
              FIELD_REFERENCE_CENTER_DIRECTION (single dish, from Field.referenceDir).
              Frame from Field.directionCode (J2000 when absent). Phase centres
              must be in a frame supported by the UVW calculation (see Raises).
            - SOURCE_DIRECTION: (optional) source direction coordinates
            - LINE_REST_FREQUENCY: (optional) rest frequencies of the spectral lines
            - LINE_SYSTEMIC_VELOCITY: (optional) systemic velocities of the spectral lines
        - Attributes:
            - type: 'field_and_source'
            - is_ephemeris: boolean flag for ephemeris sources (always False, see Notes)

    Raises
    ------
    RuntimeError
        If the partition has no field id, or a field id is not in the Field table
    ValueError
        If a Field direction code is not supported, or the fields of the partition
        have different direction frames, or (interferometric partitions,
        ``is_single_dish=False``) the phase centre frame is not supported by the
        UVW calculation (:mod:`calculate_uvw`): for example HADEC, AZELNE,
        AZELNEGEO and ITRF fields, whose directions would need an observation time
        and location. The error is raised when the partition is opened, so that
        the partition is skipped (and logged) instead of the UVW failing later,
        when the data is computed. Single-dish partitions (no UVW) accept all the
        frames of ``ASDM_DIRECTION_CODE_TO_FRAME``.

    Notes
    -----
    Field directions are given as polynomials in time (``Field.numPoly`` terms).
    Only the zeroth-order term (the direction at ``Field.time``) is used, and
    ephemeris tables (``Field.ephemerisId``) are not read: a warning is logged for
    fields with ``numPoly > 1`` or an ``ephemerisId``, which get a static
    direction.

    When several Source rows exist for the (sourceId, spectral window) of a field
    (different ``timeInterval``), the earliest one is used, as the MSv2 converter
    does. Line information (numLines, transition, restFrequency, sysVel) is
    optional and read row by row. Missing values are filled with NaN / empty
    names.
    """

    field_ids = _get_partition_field_ids(partition_descr)
    field_rows = _get_field_rows(asdm, field_ids)
    field_names = [
        make_field_name(row.getFieldName(), field_id)
        for row, field_id in zip(field_rows, field_ids, strict=True)
    ]

    xds = xr.Dataset(attrs={"type": "field_and_source"})
    xds = xds.assign_coords(
        {
            "sky_dir_label": ["ra", "dec"],
            "field_name": ("field_name", np.array(field_names, dtype=str)),
        }
    )

    xds = _add_field_center_direction(xds, field_rows, field_names, is_single_dish)

    source_rows = _get_source_rows(asdm, field_rows, field_names, spectral_window_id)
    xds = _add_source_info(xds, source_rows, spectral_window_id)
    xds = _add_line_info(xds, source_rows, field_names, spectral_window_id)

    # Ephemeris tables are not supported (see the warnings in
    # _add_field_center_direction): the field directions are static.
    xds.attrs.update({"is_ephemeris": False})

    return xds


def _get_partition_field_ids(partition_descr: dict) -> list[int]:
    """Unique field ids of a partition, in ascending order."""
    field_ids = sorted({int(field_id) for field_id in partition_descr["fieldId"]})
    if not field_ids:
        raise RuntimeError(
            "Cannot create field_and_source_xds: the partition has no fieldId"
        )
    return field_ids


def _get_field_rows(asdm: pyasdm.ASDM, field_ids: list[int]) -> list:
    """Field table rows for the given field ids (same order as field_ids)."""
    rows_by_id = {row.getFieldId().getTagValue(): row for row in asdm.getField().get()}
    missing = [field_id for field_id in field_ids if field_id not in rows_by_id]
    if missing:
        raise RuntimeError(
            f"Field ids {missing} of the partition not found in the Field table "
            f"(Field ids present: {sorted(rows_by_id)})"
        )
    return [rows_by_id[field_id] for field_id in field_ids]


def _add_field_center_direction(
    xds: xr.Dataset, field_rows: list, field_names: list[str], is_single_dish: bool
) -> xr.Dataset:
    """
    Add FIELD_PHASE_CENTER_DIRECTION (interferometric, Field.phaseDir) or
    FIELD_REFERENCE_CENTER_DIRECTION (single dish, Field.referenceDir), with
    one direction per field and the frame given by Field.directionCode.
    """
    if is_single_dish:
        center_direction_dv = "FIELD_REFERENCE_CENTER_DIRECTION"
    else:
        center_direction_dv = "FIELD_PHASE_CENTER_DIRECTION"
    # TODO: make the FIELD_*_CENTER_DISTANCE variable once we have ephemeris data

    directions = []
    frames = []
    for row, field_name in zip(field_rows, field_names, strict=True):
        poly_dir = row.getReferenceDir() if is_single_dish else row.getPhaseDir()
        num_poly = row.getNumPoly()
        if num_poly > 1:
            xradio_logger().warning(
                f"Field {field_name} has a polynomial direction (numPoly={num_poly}). "
                "Only the zeroth-order term (the direction at Field.time) is used: "
                "the direction of moving targets is not followed."
            )
        ephemeris_id = get_optional_row_attr(row, "ephemerisId")
        if ephemeris_id is not None:
            xradio_logger().warning(
                f"Field {field_name} refers to an ephemeris (ephemerisId={ephemeris_id}). "
                "Ephemeris tables are not supported by the ASDM backend: the field "
                "direction is static (zeroth-order term of the Field direction)."
            )
        # zeroth-order polynomial term: the direction at Field.time
        direction_term_0 = poly_dir[0]
        directions.append([direction_term_0[0].get(), direction_term_0[1].get()])
        frames.append(
            direction_code_to_frame(get_optional_row_attr(row, "directionCode"))
        )

    unique_frames = sorted(set(frames))
    if len(unique_frames) > 1:
        raise ValueError(
            f"The fields {field_names} of the partition have different direction "
            f"frames ({frames}), which is not supported in one field_and_source_xds. "
            "Use a partition_scheme that includes 'fieldId'."
        )

    frame = unique_frames[0]
    if not is_single_dish:
        _check_phase_center_frame_supported_for_uvw(frame, field_rows, field_names)

    xds[center_direction_dv] = xr.DataArray(
        np.array(directions, dtype=np.float64),
        dims=["field_name", "sky_dir_label"],
        attrs=make_sky_coord_measure_attrs("rad", frame),
    )
    return xds


def _check_phase_center_frame_supported_for_uvw(
    frame: str, field_rows: list, field_names: list[str]
) -> None:
    """
    Raise ValueError if the UVW calculation does not support phase centres in
    ``frame`` (the supported frames are those of
    ``calculate_uvw.phase_center_frame_to_astropy``).

    The UVW of interferometric partitions are computed lazily from
    FIELD_PHASE_CENTER_DIRECTION. Checking the frame here, when the partition is
    opened, makes an unsupported frame fail the partition (which open_asdm then
    skips, logging the error) rather than the UVW computation (``.values``,
    ``to_zarr``, a dask compute) of the whole processing set.
    """
    try:
        uvw_module.phase_center_frame_to_astropy(frame)
    except NotImplementedError as exc:
        codes = sorted(
            {
                direction_code_name(get_optional_row_attr(row, "directionCode"))
                for row in field_rows
            }
        )
        raise ValueError(
            f"The phase center of the fields {field_names} is in the frame "
            f"{frame!r} (Field directionCode {', '.join(codes)}), for which UVW "
            f"cannot be calculated: {exc}. Interferometric partitions with these "
            "fields cannot be opened."
        ) from None


def _get_source_rows(
    asdm: pyasdm.ASDM, field_rows: list, field_names: list[str], spectral_window_id: int
) -> list:
    """
    Get the Source row of every field for a spectral window (None, logging a
    warning, when there is none). When several rows exist (different
    timeInterval), the earliest is used.
    """
    source_ids = [get_optional_row_attr(row, "sourceId") for row in field_rows]
    wanted_ids = {source_id for source_id in source_ids if source_id is not None}
    # one pass over the Source table
    rows_by_source_id = {source_id: [] for source_id in wanted_ids}
    for row in asdm.getSource().get():
        source_id = row.getSourceId()
        if (
            source_id in rows_by_source_id
            and row.getSpectralWindowId().getTagValue() == spectral_window_id
        ):
            rows_by_source_id[source_id].append(row)

    return [
        _select_source_row(
            source_id,
            rows_by_source_id.get(source_id, []),
            field_name,
            spectral_window_id,
        )
        for source_id, field_name in zip(source_ids, field_names, strict=True)
    ]


def _select_source_row(
    source_id: int | None, rows: list, field_name: str, spectral_window_id: int
):
    """Earliest of the Source rows of a field, None (with a warning) if no row."""
    if source_id is None:
        xradio_logger().warning(
            f"Field {field_name} has no sourceId. Its source information will be "
            f"'{UNKNOWN_SOURCE_NAME}'."
        )
        return None

    if not rows:
        xradio_logger().warning(
            f"No Source row found for sourceId {source_id} (field {field_name}) and "
            f"spectral window {spectral_window_id}. Its source information will be "
            f"'{UNKNOWN_SOURCE_NAME}'."
        )
        return None

    rows = sorted(rows, key=lambda row: row.getTimeInterval().getStart().get())
    if len(rows) > 1:
        xradio_logger().warning(
            f"The Source table has {len(rows)} rows (time intervals) for sourceId "
            f"{source_id} (field {field_name}) and spectral window "
            f"{spectral_window_id}. This is not currently supported. Only the first "
            "time interval will be used."
        )
    return rows[0]


def _add_source_info(
    xds: xr.Dataset, source_rows: list, spectral_window_id: int
) -> xr.Dataset:
    """
    Add the source_name coordinate and the SOURCE_DIRECTION data variable (one
    entry per field). Fields without Source row get "Unknown" / NaN.
    """
    source_names = [
        (row.getSourceName().strip() if row is not None else UNKNOWN_SOURCE_NAME)
        for row in source_rows
    ]
    xds = xds.assign_coords(
        {"source_name": ("field_name", np.array(source_names, dtype=str))}
    )

    available_rows = [row for row in source_rows if row is not None]
    if not available_rows:
        return xds

    try:
        frames = {
            direction_code_to_frame(get_optional_row_attr(row, "directionCode"))
            for row in available_rows
        }
    except ValueError as exc:
        xradio_logger().warning(
            f"SOURCE_DIRECTION not included for spectral window {spectral_window_id}: "
            f"{exc}"
        )
        return xds
    if len(frames) > 1:
        xradio_logger().warning(
            f"SOURCE_DIRECTION not included for spectral window {spectral_window_id}: "
            f"the sources of the partition have different direction frames ({frames})."
        )
        return xds

    # TODO: to split in _DIRECTION/_DISTANCE
    source_direction = np.full((len(source_rows), 2), np.nan)
    for idx, row in enumerate(source_rows):
        if row is not None:
            direction = row.getDirection()
            source_direction[idx] = [direction[0].get(), direction[1].get()]

    xds["SOURCE_DIRECTION"] = xr.DataArray(
        source_direction,
        dims=["field_name", "sky_dir_label"],
        attrs=make_sky_coord_measure_attrs("rad", frames.pop()),
    )
    return xds


def _strip_quotes(name: str) -> str:
    """Transition names keep the quotes of the XML (e.g. '"CO_v_0_2_1(ID=0)"')."""
    return str(name).strip().strip('"').strip()


def _read_source_lines(source_row) -> dict:
    """
    Read the (optional) line information of one Source row.

    Returns a dict with "num_lines", "transition" (list[str]), "rest_frequency"
    and "sys_vel" (lists of float, or None when absent) and "frequency_ref_code".
    """
    lines = {
        "num_lines": 0,
        "transition": None,
        "rest_frequency": None,
        "sys_vel": None,
        "frequency_ref_code": None,
    }
    if source_row is None:
        return lines

    transition = get_optional_row_attr(source_row, "transition")
    rest_frequency = get_optional_row_attr(source_row, "restFrequency")
    sys_vel = get_optional_row_attr(source_row, "sysVel")
    if transition is not None:
        lines["transition"] = [_strip_quotes(name) for name in transition]
    if rest_frequency is not None:
        lines["rest_frequency"] = [freq.get() for freq in rest_frequency]
    if sys_vel is not None:
        lines["sys_vel"] = [vel.get() for vel in sys_vel]

    num_lines = get_optional_row_attr(source_row, "numLines")
    if num_lines is None:
        num_lines = max(
            len(values)
            for values in (
                lines["transition"] or [],
                lines["rest_frequency"] or [],
                lines["sys_vel"] or [],
            )
        )
    lines["num_lines"] = int(num_lines)
    lines["frequency_ref_code"] = get_optional_row_attr(source_row, "frequencyRefCode")
    return lines


def _pad_lines(values: list | None, num_lines: int, fill) -> list:
    """Pad (or cut) a per-line list to num_lines entries."""
    values = list(values or [])[:num_lines]
    return values + [fill] * (num_lines - len(values))


def _add_line_info(
    xds: xr.Dataset, source_rows: list, field_names: list[str], spectral_window_id: int
) -> xr.Dataset:
    """
    Add line_label, line_name (field_name, line_label), LINE_REST_FREQUENCY and
    LINE_SYSTEMIC_VELOCITY (field_name, line_label), following the MSv2
    converter conventions (labels '0', '1', ...). The line attributes are read
    for each field's own Source row, so rows of other sources / spectral windows
    without line information do not matter. Missing values are NaN / "".
    """
    lines_by_field = [_read_source_lines(row) for row in source_rows]
    max_num_lines = max(lines["num_lines"] for lines in lines_by_field)
    if max_num_lines == 0:
        xradio_logger().debug(
            f"No spectral line information in the Source table for spectral window "
            f"{spectral_window_id} (fields {field_names})."
        )
        return xds

    # Fields with fewer lines than others are simply padded. Warn about fields
    # without any line, or whose rest frequencies are absent / incomplete.
    partial = [
        field_name
        for field_name, lines in zip(field_names, lines_by_field, strict=True)
        if lines["num_lines"] == 0
        or len(lines["rest_frequency"] or []) < lines["num_lines"]
    ]
    if partial:
        xradio_logger().warning(
            f"Spectral line information is missing or incomplete for fields {partial} "
            f"(spectral window {spectral_window_id}). Missing rest frequencies and "
            "systemic velocities are set to NaN."
        )

    line_name = [
        _pad_lines(lines["transition"], max_num_lines, "") for lines in lines_by_field
    ]
    xds = xds.assign_coords(
        {
            "line_label": np.arange(max_num_lines).astype(str),
            "line_name": (
                ("field_name", "line_label"),
                np.array(line_name, dtype=str),
            ),
        }
    )

    if any(lines["rest_frequency"] is not None for lines in lines_by_field):
        rest_frequency = [
            _pad_lines(lines["rest_frequency"], max_num_lines, np.nan)
            for lines in lines_by_field
        ]
        xds["LINE_REST_FREQUENCY"] = xr.DataArray(
            np.array(rest_frequency, dtype=np.float64),
            dims=["field_name", "line_label"],
            attrs=make_spectral_coord_measure_attrs(
                "Hz", observer=_get_rest_frequency_observer(lines_by_field)
            ),
        )

    if any(lines["sys_vel"] is not None for lines in lines_by_field):
        sys_vel = [
            _pad_lines(lines["sys_vel"], max_num_lines, np.nan)
            for lines in lines_by_field
        ]
        xds["LINE_SYSTEMIC_VELOCITY"] = xr.DataArray(
            np.array(sys_vel, dtype=np.float64),
            dims=["field_name", "line_label"],
            attrs=make_quantity_attrs("m/s"),
        )

    return xds


def _get_rest_frequency_observer(lines_by_field: list[dict]) -> str:
    """
    Observer (frame) of the rest frequencies: from Source.frequencyRefCode when
    present, otherwise 'lsrk', as in datasets converted from MSv2.
    """
    default_observer = "lsrk"
    codes = {
        lines["frequency_ref_code"].getName()
        for lines in lines_by_field
        if lines["frequency_ref_code"] is not None
    }
    if not codes:
        return default_observer
    if len(codes) > 1:
        xradio_logger().warning(
            f"Different Source frequencyRefCode values in one partition ({codes}). "
            f"The rest frequencies are labelled with the default '{default_observer}'."
        )
        return default_observer
    code = codes.pop()
    try:
        return frequency_code_to_observer(code, default=default_observer)
    except ValueError as exc:
        xradio_logger().warning(
            f"{exc} The rest frequencies are labelled with the default "
            f"'{default_observer}'."
        )
        return default_observer
