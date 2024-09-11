from .subtables import subt_rename_ids
from ._tables.read import load_generic_table, convert_casacore_time


def create_info_dicts(
    in_file: str,
    xds: xr.Dataset,
    field_and_source_xds: xr.Dataset,
    partition_info_misc_fields: dict,
    tb_tool: tables.table,
) -> dict:
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
       obs_mode, num_lines, taql_where
    tb_tool: tables.table
       table (query) on the main table with an MSv4 query

    Returns:
    --------
    info_dicts: dict
       info dicts ready to be used to update the attrs of the MSv4
    """

    if "line_name" in field_and_source_xds.coords:
        line_name = to_list(unique_1d(np.ravel(field_and_source_xds.line_name.values)))
    else:
        line_name = []

    info_dicts = {}
    info_dicts["partition_info"] = {
        # "spectral_window_id": xds.frequency.attrs["spectral_window_id"],
        "spectral_window_name": xds.frequency.attrs["spectral_window_name"],
        # "field_id": to_list(unique_1d(field_id)),
        "field_name": to_list(np.unique(field_and_source_xds.field_name.values)),
        # "source_id": to_list(unique_1d(source_id)),
        "line_name": line_name,
        "scan_number": to_list(np.unique(partition_info_misc_fields["scan_id"])),
        "source_name": to_list(np.unique(field_and_source_xds.source_name.values)),
        "polarization_setup": to_list(xds.polarization.values),
        "num_lines": partition_info_misc_fields["num_lines"],
        "obs_mode": partition_info_misc_fields["obs_mode"].split(","),
        "taql": partition_info_misc_fields["taql_where"],
    }

    observation_id = check_if_consistent(
        tb_tool.getcol("OBSERVATION_ID"), "OBSERVATION_ID"
    )
    info_dicts["observation_info"] = create_observation_info(in_file, observation_id)

    processor_id = check_if_consistent(tb_tool.getcol("PROCESSOR_ID"), "PROCESSOR_ID")
    info_dicts["processor_info"] = create_processor_info(in_file, processor_id)

    return info_dicts


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
