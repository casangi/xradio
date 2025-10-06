import re

import numpy as np
import xarray as xr

try:
    from casacore import tables
except ImportError:
    import xradio._utils._casacore.casacore_from_casatools as tables

import toolviper.utils.logger as logger

from .subtables import subt_rename_ids
from ._tables.read import load_generic_table, convert_casacore_time
from xradio._utils.list_and_array import check_if_consistent


def create_info_dicts(
    in_file: str,
    xds: xr.Dataset,
    field_and_source_xds: xr.Dataset,
    partition_info_misc_fields: dict,
    tb_tool: tables.table,
) -> dict[str, dict]:
    """
    For an MSv4, produces several info dicts (partition_info, processor_info,
    observation_info). The info dicts are returned in a dictionary that
    contains them indexed by their corresponding keys, which can be used
    directly to update the attrs dict of an MSv4.

    Parameters:
    -----------
    in_file: str
       path to the input MSv2
    xds: xr.Dataset
       main xds of the MSv4 being converted
    field_and_source_xds: xr.Dataset
       field_and_source_xds subdataset
    partition_info_misc_fiels: dict
       dict with several scalar fields for the partition_info dict that are
       collected while processing the main MSv4 table. Expected: scan_id,
       obs_mode, taql_where
    tb_tool: tables.table
       table (query) on the main table with an MSv4 query

    Returns:
    --------
    info_dicts: dict
       info dicts ready to be used to update the attrs of the MSv4
    """

    info_dicts = {}

    observation_id = check_if_consistent(
        tb_tool.getcol("OBSERVATION_ID"), "OBSERVATION_ID"
    )
    info_dicts["observation_info"] = create_observation_info(in_file, observation_id)
    # info_dicts["observation_info"]["intents"] = partition_info_misc_fields[
    #     "intents"
    # ].split(",")

    processor_id = check_if_consistent(tb_tool.getcol("PROCESSOR_ID"), "PROCESSOR_ID")
    info_dicts["processor_info"] = create_processor_info(in_file, processor_id)

    return info_dicts


def create_observation_info(
    in_file: str, observation_id: int
) -> dict[str, list[str] | str]:
    """
    Makes a dict with the observation info extracted from the PROCESSOR subtable.
    When available, it also takes metadata from the ASDM tables (imported 'asis')
    ASDM_EXECBLOCK and ASDM_SBSUMMARY

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
        "observer": [str(generic_observation_xds["OBSERVER"].values[0])],
        "release_date": str(
            convert_casacore_time(generic_observation_xds["RELEASE_DATE"].values)[0]
        ),
    }
    # could just assume lower:upper case but keeping explicit dict for now
    mandatory_fields = {"project_UID": "PROJECT", "observing_log": "LOG"}
    for field_msv4, col_msv2 in mandatory_fields.items():
        observation_info[field_msv4] = str(generic_observation_xds[col_msv2].values[0])

    execblock_optional_fields = {
        "execution_block_UID": "execBlockUID",
        "session_reference_UID": "sessionReference",
        "observing_log": "observingLog",
    }
    execblock_info = try_optional_asdm_asis_table_info(
        in_file, "ASDM_EXECBLOCK", execblock_optional_fields
    )
    observation_info.update(execblock_info)

    sbsummary_optional_fields = {
        "scheduling_block_UID": "sbSummaryUID",
    }
    sbsummary_info = try_optional_asdm_asis_table_info(
        in_file, "ASDM_SBSUMMARY", sbsummary_optional_fields
    )
    observation_info.update(sbsummary_info)

    observation_info = replace_entity_ids(observation_info)

    observation_info = try_find_uids_from_observation_schedule(
        generic_observation_xds, observation_info
    )

    return observation_info


def try_optional_asdm_asis_table_info(
    in_file: str, asdm_table_name: str, optional_fields: dict[str, str]
) -> dict[str, str]:
    """
    Tries to find an optional ASDM_* subtable (ASDM_EXECBLOCK, ASDM_SBSUMMARY, etc.),
    and if available, gets the optional fields requested into a metadata dict. That
    dict can be used to populate the observation_info dict.

    Parameters
    ----------
    in_file: str
       path to an input MSv2
    asm_table_name: str
       name of the "asis" ASDM table to look for.
    optional_fields: dict[str, str]
       dictionary of field/column names (as {MSv4_name: MSv2/ASDM_name}

    Returns:
    --------
    table_info: dict
        observation description (partial, some fields) ready for the MSv4
        observation_info attr
    """
    asdm_asis_xds = None

    try:
        asdm_asis_xds = load_generic_table(in_file, asdm_table_name)
    except ValueError as exc:
        logger.debug(
            f"Did not find the {asdm_table_name} subtable, not loading optional fields in observation_info. Exception: {exc}"
        )

    if asdm_asis_xds:
        table_info = extract_optional_fields_asdm_asis_table(
            asdm_asis_xds, optional_fields
        )
    else:
        table_info = {}

    return table_info


def extract_optional_fields_asdm_asis_table(
    asdm_asis_xds: xr.Dataset, optional_fields: dict[str, str]
) -> dict[str, str]:
    """
    Get the (optional) fields of the observation_info that come from "asis" ASDM
    tables like the ASDM_EXECBLOCK and ASDM_SBSUMMARY subtables.

    Note this does not parse strings like 'session_reference':
    '<EntityRef entityId="uid://A001/X133d/X169f" partId="X00000000" entityTypeName="OUSStatus"'.
    If only the UID is required that needs to be filtered afterwards.

    Parameters
    ----------
    asdm_asis_xds: xr.Dataset
        raw xds read from subtable ASDM_*

    Returns:
    --------
    info: dict
        info dict with description from an ASDM_* subtable, ready
        for the MSv4 observation_info dict
    """

    table_info = {}
    for field_msv4, col_msv2 in optional_fields.items():
        if col_msv2 in asdm_asis_xds.data_vars:
            msv2_value = asdm_asis_xds[col_msv2].values[0]
            if isinstance(msv2_value, np.ndarray):
                table_info[field_msv4] = ",".join([log for log in msv2_value])
            else:
                table_info[field_msv4] = msv2_value

    return table_info


def try_find_uids_from_observation_schedule(
    generic_observation_xds: xr.Dataset, observation_info: dict
) -> dict[str, str]:
    """
    This function tries to parse the execution_block_UID and scheduling_block_UID
    from the SCHEDULE column of the OBSERVATION subtable. If found, and they
    could not alreadly be loaded from the ASDM_* subtables, adds them to the
    output observation_info dict.

    Sometimes, even if the ASDM_EXECBLOCK and ASDM_SBSUMMARY are not available to
    load various ASDM UIDs, we can still find a couple of them in the
    OBSERVATION/SCHEDULE column (when the MS is imported from an ASDM, by
    importasdm). The SCHEDULE column can have values like:

    '[SchedulingBlock uid://A001/X3571/X122, ExecBlock uid://A002/X1003af4/X75a3]'

    Parameters
    ----------
    generic_observation_xds: xr.Dataset
        generic observation dataset from the OBSERVATION subtable
    observation_info: dict
        an observation_info being populated

    Returns:
    --------
    info: dict
        info dict with possibly additional UIDs found in the OBSERVATION
        subtable
    """

    out_info = dict(observation_info)

    if "SCHEDULE" in generic_observation_xds.data_vars:
        schedule = generic_observation_xds["SCHEDULE"].values[0]
        if isinstance(schedule, np.ndarray) and 2 == len(schedule):
            if "scheduling_block_UID" not in observation_info:
                scheduling_uid_match = re.search(
                    "SchedulingBlock ([\\w/:]+)", schedule[0]
                )
                if scheduling_uid_match:
                    out_info["scheduling_block_UID"] = scheduling_uid_match.group(1)
            if "execution_block_UID" not in observation_info:
                execution_uid_match = re.search("ExecBlock ([\\w/:]+)", schedule[1])
                if execution_uid_match:
                    out_info["execution_block_UID"] = execution_uid_match.group(1)

    return out_info


def replace_entity_ids(observation_info: dict) -> dict[str, list[str] | str]:
    """
    For several fields of the input dictionary, which are known to be of "UID" type,
    replace their lengthy XML string with the UID value contained in it. For example, from
    '<EntityRef entityId="uid://A001/X133d/X169f" partId="X00000000" entityTypeName="OUSStatus">'
    it takes 'uid://A001/X133d/X169f'.

    The UID values are written in the MSv2 "asis" ASDM_* subtables imported from ASDM tables
    as the full string of the EntityRef XML elements. This function takes only the entityId
    ("uid://A00...") from the EntityRef.


    Parameters
    ----------
    observation_info: dict
        info dict where some UID fields (as xml element strings) need to be replaced/simplified

    Returns:
    --------
    info: dict
        dictionary as the input where the UIDs have been replaced by their entityId (uid://A00...)

    """
    out_info = dict(observation_info)

    entity_refs = [
        "execution_block_UID",
        "session_reference_UID",
        "scheduling_block_UID",
    ]
    for ref_name in entity_refs:
        if ref_name in observation_info:
            out_info[ref_name] = search_entity_id(observation_info[ref_name])

    return out_info


def search_entity_id(entity_ref_xml: str) -> str:
    """
    Given an EntityRef XML string from an ASDM, like the following
    examples:

    - example sbSummaryID:
    '<EntityRef entityId="uid://A001/X133d/X169a" partId="X00000000" entityTypeName="SchedBlock" documentVersion="1"/>'

    - example sessionReferenceUID:
    '<EntityRef entityId="uid://A001/X133d/X169f" partId="X00000000" entityTypeName="OUSStatus"'

    this funcion takes the "uid://..." value of the entityId.

    Parameters
    ----------
    entity_ref_xml: str
        An EntityRef from an ASDM table (usually ExecBlock or
        SBSUMMARY) as found in columns like or execBlockUID,
        sessionReference or sbSummaryUID.

    Returns:
    --------
    str
        the entityId string value of the EntityRef received, or
        the same string as received if no entityId could be found.
    """
    uid_match = re.search('entityId="([\\w/:]+)"', entity_ref_xml)
    entity_id = uid_match.group(1) if uid_match else entity_ref_xml
    return entity_id


def create_processor_info(in_file: str, processor_id: int) -> dict[str, str]:
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
