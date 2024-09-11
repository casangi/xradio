from .subtables import subt_rename_ids
from ._tables.read import load_generic_table, convert_casacore_time


def create_observation_info(in_file: str, observation_id: int):
    """
    Makes a dict with the observation info extracted from the PROCESSOR subtable.

    Parameters
    ----------
    in_file: str
       path to an input MSv2
    processor_id: int
        processor ID for one MSv4 dataset

    Returns:
    --------
    observation_info: dict
        observation description ready for the MSv4 observation_info attr
    """

    generic_observation_xds = load_generic_table(
        in_file,
        "OBSERVATION",
        rename_ids=subt_rename_ids["OBSERVATION"],
        taql_where=f" where ROWID() = {observation_id}",
    )

    observation_info = {
        "observer": [generic_observation_xds["OBSERVER"].values[0]],
        "release_date": str(
            convert_casacore_time(generic_observation_xds["RELEASE_DATE"].values)
        ),
    }
    # could just assume lower:upper case but keeping explicit dict for now
    mandatory_fields = {"project": "PROJECT"}
    for field_msv4, row_msv2 in mandatory_fields.items():
        observation_info[field_msv4] = generic_observation_xds[row_msv2].values[0]

    # TODO: some of these fields will need to be gathered from ASDM tables
    # (ExecBlock, etc.)
    optional_fields = {
        "execution_block_id": "EXECUTION_BLOCK_ID",
        "execution_block_number": "EXECUTION_BLOCK_NUMBER",
        "execution_block_UID": "EXECUTION_BLOCK_UID",
        "session_reference": "SESSION_REFERENCE",
        "observing_script": "OBSERVING_SCRIPT",
        "observing_script_UID": "OBSERVING_SCRIPT_UID",
        "observing_log": "OBSERVING_LOG",
    }
    for field_msv4, row_msv2 in optional_fields.items():
        if row_msv2 in generic_observation_xds.data_vars:
            observation_info[field_msv4] = generic_observation_xds[row_msv2].values[0]

    return observation_info


def create_processor_info(in_file: str, processor_id: int):
    """
    Makes a dict with the processor info extracted from the PROCESSOR subtable.

    Parameters
    ----------
    in_file: str
       path to an input MSv2
    processor_id: int
        processor ID for one MSv4 dataset

    Returns:
    --------
    processor_info: dict
        processor description ready for the MSv4 processor_info attr
    """

    generic_processor_xds = load_generic_table(
        in_file,
        "PROCESSOR",
        rename_ids=subt_rename_ids["PROCESSOR"],
        taql_where=f" where ROWID() = {processor_id}",
    )

    # Many telescopes (ASKAP, MeerKAT, SKA-Mid, VLBI, VLBA, ngEHT) seem to
    # produce an empty PROCESSOR subtable
    if len(generic_processor_xds.data_vars) <= 0:
        processor_info = {"type": "", "sub_type": ""}
    else:
        processor_info = {
            "type": generic_processor_xds["TYPE"].values[0],
            "sub_type": generic_processor_xds["SUB_TYPE"].values[0],
        }

    return processor_info
