import numpy as np
import pyasdm
import xarray as xr

from xradio._utils.logging import xradio_logger
from xradio.measurement_set._utils._asdm._utils.time import (
    convert_time_asdm_to_datetime64,
)


def create_info_dicts(
    asdm: pyasdm.ASDM, xds: xr.Dataset, partition_descr: dict
) -> dict[str, dict]:
    """
    Create information dictionaries from ASDM data.

    This function generates structured information dictionaries containing observation
    and processor details from an ASDM dataset.

    Parameters
    ----------
    asdm : pyasdm.ASDM
        The ASDM dataset object containing the raw data
    xds : xr.Dataset
        The xarray Dataset containing the processed data
    partition_descr : dict
        Dictionary describing the data partitioning. The "execBlockId" and
        "configDescriptionId" entries are used (one value each).

    Returns
    -------
    dict[str, dict]
        Dictionary containing two keys:
            - 'observation_info': Dictionary with observation information
            - 'processor_info': Dictionary with processor information

    Notes
    -----
    This function consolidates observation and processor information from the ASDM
    into structured dictionaries for easier access and processing.
    """

    observation_info = create_observation_info(asdm, partition_descr)

    processor_info = create_processor_info(asdm, partition_descr)

    info_dicts = {
        "observation_info": observation_info,
        "processor_info": processor_info,
    }

    return info_dicts


def create_processor_info(asdm: pyasdm.ASDM, partition_descr: dict) -> dict:
    """
    Creates a dictionary containing processor information from ASDM data.

    Parameters
    ----------
    asdm : pyasdm.ASDM
        The ASDM object containing the observation data
    partition_descr : dict
        Dictionary containing partition information, including the (single)
        configDescriptionId of the partition

    Returns
    -------
    dict
        Dictionary containing processor information with keys:
        - type: The processor type name
        - sub_type: The processor subtype name

    Raises
    ------
    ValueError
        If the partition does not have exactly one configDescriptionId, or if
        the ConfigDescription or Processor rows are not found.
    """

    config_description_id = _get_single_id(partition_descr, "configDescriptionId")
    config_tbl = asdm.getConfigDescription()
    config_row = config_tbl.getRowByKey(
        pyasdm.types.Tag(f"{config_tbl.getName()}_{config_description_id}")
    )
    if config_row is None:
        raise ValueError(
            f"No row with configDescriptionId={config_description_id} in the "
            "ConfigDescription table"
        )
    processor_row = config_row.getProcessorUsingProcessorId()
    if processor_row is None:
        raise ValueError(
            f"No Processor row for {config_row.getProcessorId()}, used by "
            f"configDescriptionId={config_description_id}"
        )

    processor_info = {
        "type": processor_row.getProcessorType().getName(),
        "sub_type": processor_row.getProcessorSubType().getName(),
    }

    return processor_info


def create_observation_info(asdm: pyasdm.ASDM, partition_descr: dict) -> dict:
    """
    Creates a dictionary with observation information from an ASDM dataset.

    The information is taken from the ExecBlock row of the partition (and the
    SBSummary row it refers to): observer, project, execution block and
    scheduling block identifiers.

    Parameters
    ----------
    asdm : pyasdm.ASDM
        The ASDM dataset object containing the observation data
    partition_descr : dict
        Dictionary containing the partition description. Its "execBlockId"
        entry (one value) selects the ExecBlock row.

    Returns
    -------
    dict
        A dictionary containing observation information (ObservationInfoDict)
        with the following keys:
        - observer : list[str]
            Name of the observer (ExecBlock.observerName)
        - release_date : str
            Date when the data becomes public (ExecBlock.releaseDate), ISO
            8601 string (YYYY-MM-DDThh:mm:ss.sssssssss). An empty string when
            the optional releaseDate is absent.
        - project_UID : str
            Project UID (entityId of ExecBlock.projectUID)
        - execution_block_UID : str
            Execution block UID (entityId of ExecBlock.execBlockUID)
        - session_reference_UID : str
            Session reference (entityId of ExecBlock.sessionReference)
        - observing_log : str | None
            Observing log (ExecBlock.observingLog), one line per log entry.
            None when the log is empty.
        - scheduling_block_UID : str | None
            Scheduling block UID (entityId of SBSummary.sbSummaryUID), None
            if the SBSummary row is not found.

    Raises
    ------
    ValueError
        If the partition does not have exactly one execBlockId or if the
        ExecBlock row is not found.
    """

    execblock_id = _get_single_id(partition_descr, "execBlockId")
    execblock_tbl = asdm.getExecBlock()
    execblock_row = execblock_tbl.getRowByKey(
        pyasdm.types.Tag(f"{execblock_tbl.getName()}_{execblock_id}")
    )
    if execblock_row is None:
        raise ValueError(
            f"No row with execBlockId={execblock_id} in the ExecBlock table"
        )

    observation_info = {
        "observer": [str(execblock_row.getObserverName())],
        "release_date": _get_release_date(execblock_row),
        "project_UID": execblock_row.getProjectUID().getEntityId(),
        "execution_block_UID": execblock_row.getExecBlockUID().getEntityId(),
        "session_reference_UID": execblock_row.getSessionReference().getEntityId(),
        "observing_log": _get_observing_log(execblock_row),
        "scheduling_block_UID": _get_scheduling_block_uid(execblock_row),
    }

    return observation_info


def _get_single_id(partition_descr: dict, key: str) -> int:
    """The single (integer) value of a partition description entry."""
    values = np.unique(np.ravel(partition_descr[key]))
    if len(values) != 1:
        raise ValueError(
            f"Expected exactly one {key} in the partition description, got "
            f"{values.tolist()}"
        )
    return int(values[0])


def _array_time_to_iso(array_time: pyasdm.types.ArrayTime) -> str:
    """ASDM ArrayTime as an ISO 8601 string (fixed epoch shift, ns precision)."""
    return str(convert_time_asdm_to_datetime64(int(array_time.get())))


def _get_release_date(execblock_row: pyasdm.ExecBlockRow) -> str:
    """ExecBlock.releaseDate as ISO string, or "" when absent."""
    if not execblock_row.isReleaseDateExists():
        return ""
    return _array_time_to_iso(execblock_row.getReleaseDate())


def _get_observing_log(execblock_row: pyasdm.ExecBlockRow) -> str | None:
    """ExecBlock.observingLog entries joined with newlines, or None when empty."""
    observing_log = execblock_row.getObservingLog()
    if not observing_log:
        return None
    return "\n".join(str(entry) for entry in observing_log)


def _get_scheduling_block_uid(execblock_row: pyasdm.ExecBlockRow) -> str | None:
    """SBSummary.sbSummaryUID of the ExecBlock, or None if not found."""
    sb_summary_row = execblock_row.getSBSummaryUsingSBSummaryId()
    if sb_summary_row is None:
        xradio_logger().warning(
            f"No SBSummary row for {execblock_row.getSBSummaryId()} (used by "
            f"{execblock_row.getExecBlockId()}), scheduling_block_UID is not set."
        )
        return None
    return sb_summary_row.getSbSummaryUID().getEntityId()
