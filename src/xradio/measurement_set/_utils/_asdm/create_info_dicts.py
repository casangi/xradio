import xarray as xr

import pyasdm

from xradio.measurement_set._utils._asdm._utils.metadata_tables import (
    exp_asdm_table_to_df,
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
        Dictionary describing the data partitioning

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


def create_processor_info(asdm: pyasdm.ASDM, partition: dict) -> dict:
    """
    Creates a dictionary containing processor information from ASDM data.

    Parameters
    ----------
    asdm : pyasdm.ASDM
        The ASDM object containing the observation data
    partition : dict
        Dictionary containing partition information including the configDescriptionId

    Returns
    -------
    dict
        Dictionary containing processor information with keys:
        - type: The processor type name
        - sub_type: The processor subtype name
    """

    config_description_id = partition["configDescriptionId"][0]
    config_tbl = asdm.getConfigDescription()
    table_name = config_tbl.getName()
    config_description_tag = pyasdm.types.Tag(f"{table_name}_{config_description_id}")
    config_row = config_tbl.getRowByKey(config_description_tag)
    processor_row = config_row.getProcessorUsingProcessorId()

    processor_info = {
        "type": processor_row.getProcessorType().getName(),
        "sub_type": processor_row.getProcessorSubType().getName(),
    }

    return processor_info


def create_observation_info(asdm: pyasdm.ASDM, partition_descr: dict) -> dict:
    """
    Creates a dictionary with observation information from an ASDM dataset.
    This function extracts various observation metadata from an ASDM (ALMA Science Data Model)
    dataset and returns it as a structured dictionary. The information includes observer details,
    project information, execution block data, and scheduling block information.

    Parameters
    ----------
    asdm : pyasdm.ASDM
        The ASDM dataset object containing the observation data
    partition_descr : dict
        Dictionary containing partition descriptions, must include 'scanIntent' key

    Returns
    -------
    dict
        A dictionary containing observation information with the following keys:
        - observer : list
            Name(s) of the observer(s)
        - project : str
            Project identifier
        - release_date : str
            Date when data becomes publicly available
        - execution_block_id : str
            Identifier for the execution block
        - execution_block_number : int
            Number of the execution block
        - execution_block_UID : str
            Unique identifier for the execution block
        - session_reference : str
            Reference to the observation session
        - observing_script : str
            Script used for the observation
        - observing_script_UID : str
            Unique identifier for the observing script
        - observing_log : str
            Log of the observation
        - scheduling_block_UID : str
            Unique identifier for the scheduling block
        - intents : list
            List of scan intents from the partition description
    """

    # TODO: needs clean-up, this comes from an early version
    sdm_main = asdm.getMain()
    sdm_main_attrs = [
        "execBlockId",
    ]
    main_df = exp_asdm_table_to_df(asdm, "Main", sdm_main_attrs)
    asdm_execblock = asdm.getExecBlock()
    table_name = asdm_execblock.getName()
    execblock_ids = main_df["execBlockId"].unique()
    execblock_tags = [
        pyasdm.types.Tag(f"{table_name}_{execblock_id}")
        for execblock_id in execblock_ids
    ]
    execblock_rows = [asdm_execblock.getRowByKey(tag) for tag in execblock_tags]

    # Reorganize the loop? / table iteration => use many-cols data frames
    observer = [row.getObserverName() for row in execblock_rows]
    release_date = [
        row.getReleaseDate() if row.isReleaseDateExists() else ""
        for row in execblock_rows
    ]
    project_uid = [row.getProjectUID().getEntityId() for row in execblock_rows]
    execblock_uid = [row.getExecBlockUID().getEntityId() for row in execblock_rows]
    session_reference_uid = [
        row.getSessionReference().getEntityId() for row in execblock_rows
    ]
    observing_log = [row.getObservingLog() for row in execblock_rows]
    sb_summary_id = [row.getSBSummaryId().getTagValue() for row in execblock_rows]

    # SBSummary sbSummaryUID
    sdm_sbsummary_attrs = ["sBSummaryId", "sbSummaryUID"]
    sb_summary_df = exp_asdm_table_to_df(asdm, "SBSummary", sdm_sbsummary_attrs)
    # scheduling_block_UID = sbsummary_df.loc[sbsummary_df["sBSummaryId"] == sb_summary_id[0]]

    scheduling_block_UID = sb_summary_df.loc[
        sb_summary_df["sBSummaryId"] == sb_summary_id[0]
    ]["sbSummaryUID"].values[0]

    def list_to_first(alist: list) -> object:
        return alist[0]

    observation_info = {
        "observer": observer,
        "release_date": list_to_first(release_date),
        "project_UID": list_to_first(project_uid),
        "execution_block_UID": list_to_first(execblock_uid),
        "session_reference_UID": list_to_first(session_reference_uid),
        "observing_log": str(list_to_first(observing_log)),
        "scheduling_block_UID": list_to_first(scheduling_block_UID),
    }

    return observation_info
