import time

import numpy as np
import pandas as pd
import pyasdm

from xradio._utils.logging import xradio_logger
from xradio.measurement_set._utils._asdm._utils.metadata_tables import (
    exp_asdm_table_to_df,
)
from xradio.measurement_set._utils._asdm._utils.time import (
    convert_time_asdm_to_datetime64,
)

#: Partition axes that are always used (one MSv4 never mixes values of these).
#: "scanIntent" is the set of "INTENT#SUBINTENT" observing modes of a Main row.
MANDATORY_PARTITION_AXES = (
    "execBlockId",
    "configDescriptionId",
    "dataDescriptionId",
    "scanIntent",
)
#: Partition axes that can be requested with the ``partition_scheme`` parameter.
OPTIONAL_PARTITION_AXES = ("fieldId", "scanNumber", "subscanNumber")
#: Optional partition axes used when ``partition_scheme`` is None.
DEFAULT_PARTITION_SCHEME = ("fieldId",)
#: Placeholder for a missing (sub)intent in "INTENT#SUBINTENT" strings, as used
#: in MSv2 OBS_MODE / MSv4 scan_intents.
UNSPECIFIED_INTENT = "UNSPECIFIED"
#: Keys of ``partition_descr["per_bdf"]``: one value per Main row (BDF) of the
#: partition, aligned with ``partition_descr["BDFPath"]``.
PER_BDF_KEYS = ("BDFPath", "time", "scanNumber", "subscanNumber", "fieldId", "stateId")

# Main table columns loaded to build the partitions.
_MAIN_ATTRS = [
    "time",
    "fieldId",
    "configDescriptionId",
    "scanNumber",
    "subscanNumber",
    "stateId",
    "dataUID",
    # BDFPath triggers a call to MainRow.getBDFPath, which uses
    # getContainer().getDirectory()
    "BDFPath",
    "execBlockId",
]
# Partition description entries that keep one value per Main row / the time
# order of first appearance (instead of sorted unique values).
_TIME_ORDERED_KEYS = ("time", "dataUID")


def validate_partition_scheme(partition_scheme: list[str] | None) -> list[str]:
    """
    Validate the optional partition axes requested by the user.

    Parameters
    ----------
    partition_scheme : list[str] | None
        Optional partition axes. None means the default ``["fieldId"]``. An
        empty list means that only the mandatory axes are used.

    Returns
    -------
    list[str]
        The optional partition axes to use, without duplicates.

    Raises
    ------
    ValueError
        If partition_scheme is not a list of names, or if it includes names
        other than the allowed optional axes ("fieldId", "scanNumber",
        "subscanNumber").
    """
    allowed_msg = (
        f"The allowed partition_scheme axes are {list(OPTIONAL_PARTITION_AXES)}. "
        f"The axes {list(MANDATORY_PARTITION_AXES)} are always used."
    )
    if partition_scheme is None:
        return list(DEFAULT_PARTITION_SCHEME)

    if isinstance(partition_scheme, str) or not isinstance(
        partition_scheme, list | tuple | np.ndarray
    ):
        raise ValueError(
            f"partition_scheme must be a list of axis names or None, got "
            f"{partition_scheme!r}. {allowed_msg}"
        )

    invalid = [axis for axis in partition_scheme if axis not in OPTIONAL_PARTITION_AXES]
    if invalid:
        raise ValueError(f"Unsupported partition_scheme axes: {invalid}. {allowed_msg}")

    return list(dict.fromkeys(str(axis) for axis in partition_scheme))


def create_partitions(
    sdm: pyasdm.ASDM,
    partition_scheme: list[str] | None = None,
    include_processor_types: list[str] | None = None,
    include_spectral_resolution_types: list[str] | None = None,
) -> list[dict]:
    """
    Group the Main rows of an ASDM into partitions, one per MSv4.

    Every Main row (BDF) is expanded into its data descriptions (one per
    spectral window / polarization setup of its ConfigDescription). The
    resulting rows are grouped by the mandatory axes ``execBlockId``,
    ``configDescriptionId``, ``dataDescriptionId`` and ``scanIntent`` plus the
    optional axes given in ``partition_scheme``.

    Parameters
    ----------
    sdm : pyasdm.ASDM
        Input ASDM object.
    partition_scheme : list[str] | None
        Optional axes to partition the data on, in addition to the mandatory
        ones. Allowed axes: "fieldId", "scanNumber", "subscanNumber".
        None (default) means ``["fieldId"]``. An empty list means that only
        the mandatory axes are used (for example one MSv4 with all the fields
        of a mosaic).
    include_processor_types : list[str] | None
        Produce partitions only for these processor types (ASDM ProcessorType
        enumeration: "CORRELATOR", "SPECTROMETER", "RADIOMETER"). None (or an
        empty list) includes all types.
    include_spectral_resolution_types : list[str] | None
        Produce partitions only for these spectral resolution types (ASDM
        SpectralResolutionType enumeration: "FULL_RESOLUTION",
        "CHANNEL_AVERAGE", "BASEBAND_WIDE"). None (or an empty list) includes
        all types.

    Returns
    -------
    list[dict]
        One partition description per partition. Each is a dict with, for
        every column of the partitioning table, a 1-D ``np.ndarray`` of the
        values found in the partition:

        - "execBlockId", "configDescriptionId", "dataDescriptionId",
          "fieldId", "scanNumber", "subscanNumber", "stateId",
          "spectralWindowId", "polOrHoloId", "sourceId": sorted unique ints
          ("sourceId" is -1 for fields without the optional sourceId).
        - "processorType", "spectralType": sorted unique str.
        - "scanIntent": sorted unique "INTENT#SUBINTENT" str, from
          Scan.scanIntent x Subscan.subscanIntent ("UNSPECIFIED" is used for a
          missing subscan intent).
        - "time": unique Main row times (datetime64[ns], ascending).
        - "dataUID": unique data UIDs, in time order.
        - "BDFPath": one entry per Main row of the partition, ordered by Main
          time (ascending, stable w.r.t. the Main table order).
        - "per_bdf": dict of equal-length 1-D arrays aligned with "BDFPath":
          "BDFPath" (str), "time" (int64 ASDM ArrayTime ns of the Main row),
          "scanNumber", "subscanNumber", "fieldId", "stateId" (int64).

    Raises
    ------
    ValueError
        If partition_scheme is not valid (see :func:`validate_partition_scheme`).
    RuntimeError
        If no ConfigDescription is left after filtering by processor type or
        spectral resolution type.
    """
    partition_scheme = validate_partition_scheme(partition_scheme)
    logger = xradio_logger()
    start = time.perf_counter()

    main_df = _load_main_df(sdm)
    if main_df.empty:
        logger.warning("The ASDM Main table is empty, no partitions can be created.")
        return []

    config_description_df = _load_config_description_df(
        sdm, main_df, include_processor_types, include_spectral_resolution_types
    )
    # Starting with Main+ConfigDescription prunes the possibilities down to the
    # (configuration, data description) combinations actually used in Main.
    partitioning_df = pd.merge(main_df, config_description_df, on="configDescriptionId")

    data_description_df = exp_asdm_table_to_df(
        sdm, "DataDescription", ["dataDescriptionId", "spectralWindowId", "polOrHoloId"]
    )
    partitioning_df = _merge_required(
        partitioning_df, data_description_df, "dataDescriptionId", "DataDescription"
    )
    partitioning_df = _merge_required(
        partitioning_df, _load_field_df(sdm), "fieldId", "Field"
    )
    partitioning_df, obs_modes = _add_scan_intent_ids(sdm, partitioning_df)
    if partitioning_df.empty:
        logger.warning(
            "No Main rows left after matching them with the ConfigDescription, "
            "DataDescription, Field and Scan tables, no partitions can be created."
        )
        return []

    # The merges do not keep the Main order (pandas inner merges with
    # non-unique keys may scramble rows). Restore a deterministic time order,
    # stable w.r.t. the Main table order.
    partitioning_df = partitioning_df.sort_values(
        ["time", "_main_row", "dataDescriptionId"], kind="stable"
    ).reset_index(drop=True)

    partition_columns = list(MANDATORY_PARTITION_AXES) + partition_scheme
    partitions = finalize_partitions_groupby(
        partitioning_df, partition_columns, obs_modes
    )

    elapsed = time.perf_counter() - start
    logger.info(
        f"Found {len(partitions)} partitions in {len(main_df)} Main rows "
        f"(partition axes: {partition_columns}), in {elapsed:.3f} s"
    )

    return partitions


def finalize_partitions_groupby(
    partitioning_df: pd.DataFrame,
    partition_columns: list[str],
    unique_scan_intents: list | np.ndarray,
) -> list[dict]:
    """
    Produces the list of partition descriptions from the partitioning table.

    Parameters
    ----------
    partitioning_df : pd.DataFrame
        Table with one row per (Main row, data description), with at least the
        ``partition_columns``, the "scanIntent" column (integer index into
        ``unique_scan_intents``) and the ``PER_BDF_KEYS`` columns ("time" as
        int64 ASDM ArrayTime nanoseconds). Columns whose name starts with "_"
        are internal and not included in the partition descriptions.
    partition_columns : list[str]
        Columns that define the partitions: every unique combination of their
        values gives one partition.
    unique_scan_intents : list | np.ndarray
        ``unique_scan_intents[idx]`` gives the intent strings of the scan
        intent index ``idx`` used in the "scanIntent" column.

    Returns
    -------
    list[dict]
        One partition description per partition, see :func:`create_partitions`.
        Partitions are ordered by the values of the partition columns.

    Raises
    ------
    ValueError
        If required columns are missing from ``partitioning_df``.
    """
    required = list(dict.fromkeys([*partition_columns, "scanIntent", *PER_BDF_KEYS]))
    missing = [col for col in required if col not in partitioning_df.columns]
    if missing:
        raise ValueError(
            f"The partitioning table is missing the columns {missing}. "
            f"Available columns: {partitioning_df.columns.to_list()}"
        )

    value_columns = [
        col
        for col in partitioning_df.columns
        if not str(col).startswith("_") and col not in ("BDFPath", "scanIntent")
    ]

    partitions = []
    for _key, group in partitioning_df.groupby(list(partition_columns), sort=True):
        # groupby keeps the row order, sort anyway so that BDFPath is in time
        # order also for tables not sorted by the caller (stable sort)
        group = group.sort_values("time", kind="stable")
        partition_descr = {}
        for col in value_columns:
            partition_descr[col] = _unique_values(
                group[col].to_numpy(), keep_order=col in _TIME_ORDERED_KEYS
            )
        partition_descr["time"] = convert_time_asdm_to_datetime64(
            partition_descr["time"]
        )

        intent_idx = np.unique(group["scanIntent"].to_numpy())
        partition_descr["scanIntent"] = np.unique(
            np.concatenate(
                [
                    np.asarray(unique_scan_intents[int(idx)], dtype=str).ravel()
                    for idx in intent_idx
                ]
            )
        )
        partition_descr["BDFPath"] = group["BDFPath"].to_numpy(dtype=str)
        partition_descr["per_bdf"] = {
            key: group[key].to_numpy(dtype=str if key == "BDFPath" else np.int64)
            for key in PER_BDF_KEYS
        }
        partitions.append(partition_descr)

    return partitions


def _load_main_df(sdm: pyasdm.ASDM) -> pd.DataFrame:
    """
    Loads the Main table columns needed for partitioning, with normalized types.

    "time" is given as int64 ASDM ArrayTime nanoseconds, "stateId" as the
    state of the first antenna (one state is assumed for all antennas), and
    "_main_row" gives the position of the row in the Main table.
    """
    main_df = exp_asdm_table_to_df(sdm, "Main", _MAIN_ATTRS)
    if main_df.empty:
        return main_df

    main_df["_main_row"] = np.arange(len(main_df), dtype=np.int64)
    main_df["time"] = _asdm_times_to_ns(main_df["time"].to_numpy())
    main_df["stateId"] = [
        _first_state_id(state_ids) for state_ids in main_df["stateId"]
    ]
    int_cols = [
        "fieldId",
        "configDescriptionId",
        "scanNumber",
        "subscanNumber",
        "stateId",
        "execBlockId",
    ]
    main_df[int_cols] = main_df[int_cols].astype(np.int64)
    main_df["BDFPath"] = main_df["BDFPath"].astype(str)
    main_df["dataUID"] = main_df["dataUID"].astype(str)
    return main_df


def _load_config_description_df(
    sdm: pyasdm.ASDM,
    main_df: pd.DataFrame,
    include_processor_types: list[str] | None,
    include_spectral_resolution_types: list[str] | None,
) -> pd.DataFrame:
    """
    Loads the ConfigDescription table, filtered by processor and spectral
    resolution types, with one row per (configDescriptionId, dataDescriptionId).
    """
    logger = xradio_logger()
    config_description_df = exp_asdm_table_to_df(
        sdm,
        "ConfigDescription",
        ["configDescriptionId", "dataDescriptionId", "processorType", "spectralType"],
    )
    for col in ["processorType", "spectralType"]:
        config_description_df[col] = config_description_df[col].map(_enum_name)

    unknown = ~main_df["configDescriptionId"].isin(
        config_description_df["configDescriptionId"]
    )
    if unknown.any():
        logger.warning(
            f"{int(unknown.sum())} Main rows refer to configDescriptionIds "
            f"{_abbreviated(sorted(set(main_df.loc[unknown, 'configDescriptionId'])))} that are not in "
            "the ConfigDescription table. These rows are ignored."
        )

    filters = [
        ("processorType", include_processor_types, "processor types"),
        (
            "spectralType",
            include_spectral_resolution_types,
            "spectral resolution types",
        ),
    ]
    for col, include_values, description in filters:
        if not include_values:
            continue
        num_before = len(config_description_df)
        config_description_df = config_description_df.loc[
            config_description_df[col].isin(include_values)
        ]
        logger.info(
            f"Keeping only partitions for requested {description}. From the "
            f"ConfigDescription table, with {num_before} rows, "
            f"{len(config_description_df)} rows are kept for {description} "
            f"{list(include_values)}"
        )
        if config_description_df.empty:
            raise RuntimeError(f"No partitions left after filtering {description}")

    # One row per data description of every configuration
    config_description_df = config_description_df.explode(
        "dataDescriptionId", ignore_index=True
    ).dropna(subset=["dataDescriptionId"])
    # explode leaves an object column (of np.int64 items)
    config_description_df["dataDescriptionId"] = config_description_df[
        "dataDescriptionId"
    ].astype(np.int64)
    config_description_df["configDescriptionId"] = config_description_df[
        "configDescriptionId"
    ].astype(np.int64)
    return config_description_df


def _load_field_df(sdm: pyasdm.ASDM) -> pd.DataFrame:
    """
    Loads the Field ids with their (optional) sourceId, -1 when absent.
    """
    field_df = exp_asdm_table_to_df(
        sdm, "Field", ["fieldId", "sourceId"], allow_absent=True
    )
    return pd.DataFrame(
        {
            "fieldId": np.asarray(field_df["fieldId"], dtype=np.int64),
            "sourceId": np.array(
                [
                    -1 if source_id is None or pd.isna(source_id) else int(source_id)
                    for source_id in field_df["sourceId"]
                ],
                dtype=np.int64,
            ),
        }
    )


def _merge_required(
    partitioning_df: pd.DataFrame, table_df: pd.DataFrame, key: str, table_name: str
) -> pd.DataFrame:
    """
    Inner merge with a table where every key used must have a row. Rows with
    keys missing from the table cannot be opened and are dropped, with a
    warning.
    """
    table_df = table_df.astype({key: np.int64})
    missing = ~partitioning_df[key].isin(table_df[key])
    if missing.any():
        xradio_logger().warning(
            f"{int(missing.sum())} (Main row, data description) combinations refer to "
            f"{key} values {_abbreviated(sorted(set(partitioning_df.loc[missing, key])))} that are "
            f"not in the {table_name} table. These are ignored."
        )
        partitioning_df = partitioning_df.loc[~missing]
    return pd.merge(partitioning_df, table_df, on=key)


def _add_scan_intent_ids(
    sdm: pyasdm.ASDM, partitioning_df: pd.DataFrame
) -> tuple[pd.DataFrame, list[tuple[str, ...]]]:
    """
    Adds the "scanIntent" column: for every row, the index of its observing
    modes (sorted unique "INTENT#SUBINTENT" strings, from Scan.scanIntent and
    Subscan.subscanIntent) in the returned list of observing modes.

    Rows of scans that are not in the Scan table are dropped, with a warning.
    """
    logger = xradio_logger()
    scan_df = exp_asdm_table_to_df(
        sdm, "Scan", ["execBlockId", "scanNumber", "scanIntent"]
    )
    scan_intents = {
        (int(eb_id), int(scan)): [_enum_name(intent) for intent in intents]
        for eb_id, scan, intents in zip(
            scan_df["execBlockId"],
            scan_df["scanNumber"],
            scan_df["scanIntent"],
            strict=False,
        )
    }
    subscan_df = exp_asdm_table_to_df(
        sdm, "Subscan", ["execBlockId", "scanNumber", "subscanNumber", "subscanIntent"]
    )
    subscan_intents = {
        (int(eb_id), int(scan), int(subscan)): _enum_name(intent)
        for eb_id, scan, subscan, intent in zip(
            subscan_df["execBlockId"],
            subscan_df["scanNumber"],
            subscan_df["subscanNumber"],
            subscan_df["subscanIntent"],
            strict=False,
        )
    }

    key_cols = ["execBlockId", "scanNumber", "subscanNumber"]
    row_keys = list(partitioning_df[key_cols].itertuples(index=False, name=None))
    modes_by_key = {}
    missing_scans = set()
    missing_subscans = set()
    for key in dict.fromkeys(row_keys):
        eb_id, scan, subscan = (int(val) for val in key)
        if (eb_id, scan) not in scan_intents:
            missing_scans.add((eb_id, scan))
            modes_by_key[key] = None
            continue
        intents = scan_intents[(eb_id, scan)] or [UNSPECIFIED_INTENT]
        subscan_intent = subscan_intents.get((eb_id, scan, subscan))
        if subscan_intent is None:
            missing_subscans.add((eb_id, scan, subscan))
            subscan_intent = UNSPECIFIED_INTENT
        modes_by_key[key] = tuple(
            sorted({f"{intent}#{subscan_intent}" for intent in intents})
        )

    if missing_subscans:
        logger.warning(
            f"No Subscan rows found for {len(missing_subscans)} (execBlockId, "
            f"scanNumber, subscanNumber) combinations: "
            f"{_abbreviated(sorted(missing_subscans))}. Their subscan intent is set "
            f"to {UNSPECIFIED_INTENT}."
        )

    row_modes = [modes_by_key[key] for key in row_keys]
    if missing_scans:
        keep = np.array([modes is not None for modes in row_modes], dtype=bool)
        logger.warning(
            f"No Scan rows found for {len(missing_scans)} (execBlockId, scanNumber) "
            f"combinations: {_abbreviated(sorted(missing_scans))}. The data of "
            f"{int((~keep).sum())} (Main row, data description) combinations of "
            "these scans are ignored."
        )
        partitioning_df = partitioning_df.loc[keep]
        row_modes = [modes for modes in row_modes if modes is not None]

    obs_modes = sorted(set(row_modes))
    mode_index = {modes: idx for idx, modes in enumerate(obs_modes)}
    partitioning_df = partitioning_df.assign(
        scanIntent=np.array([mode_index[modes] for modes in row_modes], dtype=np.int64)
    )

    return partitioning_df, obs_modes


def _abbreviated(values: list, max_items: int = 5) -> str:
    """String with the first max_items values of a list (for log messages)."""
    if len(values) <= max_items:
        return str(values)
    return f"{str(values[:max_items])[:-1]}, ...]"


def _enum_name(value) -> str:
    """Name of an ASDM enumeration value (or the value itself if a str)."""
    if hasattr(value, "getName"):
        return str(value.getName())
    return str(value)


def _first_state_id(state_ids) -> int:
    """State of the first antenna (-1 when no state is given)."""
    state_ids = np.ravel(state_ids)
    if len(state_ids) == 0:
        return -1
    return int(state_ids[0])


def _asdm_times_to_ns(times: np.ndarray) -> np.ndarray:
    """ASDM ArrayTime values (objects or nanoseconds) to int64 nanoseconds."""
    return np.array(
        [int(value.get()) if hasattr(value, "get") else int(value) for value in times],
        dtype=np.int64,
    )


def _unique_values(values: np.ndarray, keep_order: bool = False) -> np.ndarray:
    """
    Unique values as a 1-D array (str arrays instead of object arrays of str),
    sorted or in order of first appearance.
    """
    values = np.asarray(values)
    if values.dtype == object and all(isinstance(val, str) for val in values):
        values = values.astype(str)
    unique, first_idx = np.unique(values, return_index=True)
    if keep_order:
        return values[np.sort(first_idx)]
    return unique
