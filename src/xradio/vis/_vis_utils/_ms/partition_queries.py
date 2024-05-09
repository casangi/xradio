import itertools
import graphviper.utils.logger as logger
import numbers
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import xarray as xr

from casacore import tables

from ._tables.table_query import open_table_ro, open_query
from ._tables.read import read_generic_table
from .subtables import subt_rename_ids


def make_partition_ids_by_ddi_scan(
    infile: str, do_subscans: bool
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Produces arrays of per-partition ddi, scan, state_id, for when
    using partiion schemes 'scan' or 'scan/subscan', that is
    partitioning by some variant of (ddi, scan, subscan(state_id))

    :param infile: Path to MS
    :param do_subscans: also partitioning by subscan, not only scan
    :return: arrays with indices that define every partition
    """
    try:
        cctable = None
        taql_distinct_states = None
        cctable = tables.table(
            infile, readonly=True, lockoptions={"option": "usernoread"}, ack=False
        )
        if do_subscans:
            taql_distinct_states = (
                "select DISTINCT SCAN_NUMBER, STATE_ID, DATA_DESC_ID from $cctable"
            )
        else:
            taql_distinct_states = (
                "select DISTINCT SCAN_NUMBER, DATA_DESC_ID from $cctable"
            )
        with open_query(cctable, taql_distinct_states) as query_states:
            logger.debug(
                f"Got query, nrows: {query_states.nrows()}, query: {query_states}"
            )
            scan_number = query_states.getcol("SCAN_NUMBER")
            logger.debug(
                f"Got col SCAN_NUMBER (len: {len(scan_number)}): {scan_number}"
            )
            if do_subscans:
                state_id = query_states.getcol("STATE_ID")
                data_desc_id = np.full(len(scan_number), None)
            else:
                state_id = [None] * len(scan_number)
                logger.debug(f"Got col STATE_ID (len: {len(state_id)}): {state_id}")
                data_desc_id = query_states.getcol("DATA_DESC_ID")

        logger.debug(f"Got col DATA_DESC_ID (len: {len(data_desc_id)}): {data_desc_id}")
        logger.debug(
            f"Len of DISTINCT SCAN_NUMBER,etc.: {len(scan_number)}. Will generate that number of partitions"
        )
    finally:
        if cctable:
            cctable.close()

    return data_desc_id, scan_number, state_id


def partition_when_empty_state(
    infile: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate fallback partition ids when trying to partition by
    'intent' but the STATE table is empty.

    Some MSs have no STATE rows and in the main table STATE_ID==-1
    (that is not a valid MSv2 but it happens).

    :param infile: Path to the MS
    :return: same as make_partition_ids_by_ddi_intent but with
    effectively only ddi indices and other indices set to None ("any
    IDs found")
    """
    try:
        main_table = None

        main_table = tables.table(
            infile, readonly=True, lockoptions={"option": "usernoread"}, ack=False
        )
        taql_ddis = "select DISTINCT DATA_DESC_ID from $main_table"
        with open_query(main_table, taql_ddis) as query_per_intent:
            # Will take whatever scans given the STATE_IDs and DDIs
            # scan_number = query_per_intent.getcol("SCAN_NUMBER")
            distinct_ddis = query_per_intent.getcol("DATA_DESC_ID")

        logger.debug(
            f"Producing {len(distinct_ddis)} partitions for ddis: {distinct_ddis}"
        )
        nparts = len(distinct_ddis)

    finally:
        if main_table:
            main_table.close()

    return distinct_ddis, [None] * nparts, [None] * nparts, [""] * nparts


def find_distinct_obs_mode(
    infile: str, state_table: tables.table
) -> Union[List[str], None]:
    """Produce a list of unique "scan/subscan" intents.

    :param infile: Path to the MS
    :param state_table: casacore table object to read from
    :return: List of unique "scan/subscan" intents as given in the
    OBS_MODE column of the STATE subtable. None if the STATE subtable
    is empty or there is a problem reading it

    """
    taql_distinct_intents = "select DISTINCT OBS_MODE from $state_table"
    with open_query(state_table, taql_distinct_intents) as query_intents:
        if query_intents.nrows() == 0:
            logger.warning(
                "STATE subtable has no data. Cannot partition by scan/subscan intent"
            )
            return None

        distinct_obs_mode = query_intents.getcol("OBS_MODE")
        logger.debug(
            f"  Query for distinct OBS_MODE len: {len(distinct_obs_mode)}, values: {distinct_obs_mode}"
        )
        return distinct_obs_mode


def filter_intents_per_ddi(
    ddis: List[int], substr: str, intents: str, spw_name_by_ddi: Dict[int, str]
) -> List[str]:
    """For a given pair of:
    - substring (say 'WVR') associated with a type of intent we want to differentiate
    - intents string (multiple comma-separated scan/subscan intents)
    => do: for every DDI passed in the list of ddis, either keep only the
    intents that have that substring (if there are any) or drop them, depending on
    whether that substring is present in the SPW name. This is to filter in only
    the intents that really apply to every DDI/SPW.

    :param ddis: list of ddis for which the intents have to be filtered
    :param substr: substring to filter by
    :param intents: string with a comma-separated list of individual
    scan/subscan intent strings (like scan/subscan intents as stored
    in the MS STATE/OBS_MODE
    :param spw_name_by_ddi: SPW names by DDI ID (row index) key
    :return: list where the intents related to 'substr' have been filtered in our out

    """
    present = substr in intents
    # Nothing to effectively filter, full cs-list of intents apply to all DDIs
    if not present:
        return [intents] * len(ddis)

    every_intent = intents.split(",")
    filtered_intents = []
    for ddi in ddis:
        spw_name = spw_name_by_ddi[ddi]
        if not spw_name:
            # we cannot say / cannot filter
            filtered_intents.append(intents)
            continue

        # A not-xor to select/deselect (or keep-only/drop) the intents that apply
        # to this DDI
        ddi_intents = [
            intnt for intnt in every_intent if (substr in intnt) == (substr in spw_name)
        ]
        ddi_intents = ",".join(ddi_intents)
        filtered_intents.append(ddi_intents)

    return filtered_intents


def make_ddi_state_intent_lists(
    main_tbl: tables.table,
    state_tbl: tables.table,
    distinct_obs_mode: np.ndarray,
    spw_name_by_ddi: Dict[int, str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Produce arrays of (ddi indices, state indices, intent string)
    for every distinct intent string, where every item represents one
    partition of the main table

    As the obs_mode strings have concatenated intent strings from all
    the scan and subscan intents, this function has started
    implementing some simple heuristics to remove the intent items
    that are not related to the respective DDIs (for example WVR
    intent is the only kept when the DDI/SPW has WVR in its name). See
    call to filter_intents_per_ddi()

    :param main_tbl: main MS table openend as a casacore.tables.table
    :state_tbl: STATE subtable openend as a casacore.tables.table
    :param distinct_obs_mode: list of unique/distinct OBS_MODE strings from the STATE table
    :return: arrays of (ddi indices, state indices, intent string)

    """
    data_desc_id, state_id_partitions, intent_names = [], [], []
    for intent in distinct_obs_mode:
        where_intent = f"where OBS_MODE = '{intent}'"
        taql_states = f"select ROWID() as ROWS from $state_tbl {where_intent}"
        with open_query(state_tbl, taql_states) as query_states:
            state_ids_for_intent = query_states.getcol("ROWS")

        state_ids = " OR STATE_ID = ".join(np.char.mod("%d", state_ids_for_intent))
        taql_partition = (
            f"select DISTINCT DATA_DESC_ID from $main_tbl where STATE_ID = {state_ids}"
        )
        with open_query(main_tbl, taql_partition) as query_ddi_intent:
            # No data for these STATE_IDs
            if query_ddi_intent.nrows() == 0:
                continue

            # Will implicitly take whatever scans given the STATE_IDs
            # and DDIs scan_number. Not needed:
            # scan_number = query_ddi_intent.getcol("SCAN_NUMBER")
            ddis = query_ddi_intent.getcol("DATA_DESC_ID")

            data_desc_id.extend(ddis)
            state_id_partitions.extend([state_ids_for_intent] * len(ddis))

            # Try to select/exclude confusing or mixed intent names such as 'WVR#*'
            intents_ddi = filter_intents_per_ddi(ddis, "WVR", intent, spw_name_by_ddi)
            intent_names.extend(intents_ddi)

    logger.debug(
        f"Produced data_desc_id: {data_desc_id},\n state_id_partitions: {state_id_partitions}"
    )
    return data_desc_id, state_id_partitions, intent_names


def make_partition_ids_by_ddi_intent(
    infile: str, spw_names: xr.DataArray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Produces arrays of per-partition ddi, scan, state_id, for when
    using the partition scheme 'intents' (ddi, scan, subscans(state_ids))

    :param infile:
    :return: arrays with indices that define every partition
    """
    # TODO: could explore other TAQL alternatives, like
    # select ... from ::STATE where OBS_MODE = ...
    #
    # This will work only if intents are already alphabetically sorted (grouped),
    # won't work for alternating intents:
    # taql_intents = "select rowid() as ROWS from $state_tbl GROUPBY OBS_MODE "

    with open_table_ro(str(Path(infile, "STATE"))) as state_tbl:
        distinct_obs_mode = find_distinct_obs_mode(infile, state_tbl)

        if distinct_obs_mode is None:
            return partition_when_empty_state(infile)

        with open_table_ro(infile) as main_tbl:
            (
                data_desc_id,
                state_id_partitions,
                intent_names,
            ) = make_ddi_state_intent_lists(
                main_tbl, state_tbl, distinct_obs_mode, spw_names
            )

    # Take whatever scans given by the STATE_IDs and DDIs
    scan_number = [None] * len(state_id_partitions)

    return data_desc_id, scan_number, state_id_partitions, intent_names


def create_taql_query_and_file_name(out_file, intent, state_ids, field_id, ddi):
    file_name = (
        out_file
        + "/"
        + out_file.replace(".vis.zarr", "").split("/")[-1]
        + "_ddi_"
        + str(ddi)
        + "_intent_"
        + intent
    )

    taql_where = f"where (DATA_DESC_ID = {ddi})"

    if isinstance(state_ids, numbers.Integral):
        taql_where += f" AND (STATE_ID = {state_ids})"
    elif state_ids is not None:
        state_ids_or = " OR STATE_ID = ".join(np.char.mod("%d", state_ids))
        taql_where += f" AND (STATE_ID = {state_ids_or})"

    if field_id is not None:
        taql_where += f" AND (FIELD_ID = {field_id})"
        file_name = file_name + "_field_id_" + str(field_id)

    return taql_where, file_name


def get_unqiue_intents(in_file):
    """_summary_

    Parameters
    ----------
    in_file : str
        _description_

    Returns
    -------
    _type_
        _description_
    """
    state_xds = read_generic_table(
        in_file,
        "STATE",
        rename_ids=subt_rename_ids["STATE"],
    )

    if len(state_xds.data_vars) > 0:
        obs_mode_dict = {}
        for i, obs_mode in enumerate(state_xds.obs_mode.values):
            if obs_mode in obs_mode_dict:
                obs_mode_dict[obs_mode].append(i)
            else:
                obs_mode_dict[obs_mode] = [i]
        return list(obs_mode_dict.keys()), list(obs_mode_dict.values())
    else:  # empty state table
        return ["None"], [None]


def enumerated_product(*args):
    yield from zip(
        itertools.product(*(range(len(x)) for x in args)), itertools.product(*args)
    )


def create_partition_enumerated_product(in_file: str, partition_scheme: str):
    """Creates an enumerated_product of the data_desc_ids, state_ids, field_ids in a MS v2 that define the partions in a processing set.

    Parameters
    ----------
    in_file : str
        _description_
    partition_scheme : str
        _description_

    Returns
    -------
    _type_
        _description_
    """
    # Unused?
    # spw_xds = read_generic_table(
    #     in_file,
    #     "SPECTRAL_WINDOW",
    #     rename_ids=subt_rename_ids["SPECTRAL_WINDOW"],
    # )

    # TODO: probably get this via query to subtable instead of read_generic_table, we just
    # need the row numbers
    ddi_xds = read_generic_table(in_file, "DATA_DESCRIPTION")
    data_desc_ids = np.arange(ddi_xds.sizes["row"])
    state_xds = read_generic_table(in_file, "STATE")

    if (partition_scheme == "ddi_intent_field") and (len(state_xds.data_vars) > 0):
        intents, state_ids = get_unqiue_intents(in_file)
        field_ids = np.arange(read_generic_table(in_file, "FIELD").sizes["row"])
    else:  # partition_scheme == "ddi_state_field"

        if len(state_xds.data_vars) > 0:
            state_ids = [np.arange(state_xds.sizes["row"])]
            intents = state_xds.obs_mode.values
        else:  # empty state table
            state_ids = [None]
            intents = ["None"]
        # print(state_xds, intents)
        # field_ids = [None]
        field_ids = np.arange(read_generic_table(in_file, "FIELD").sizes["row"])

    return enumerated_product(data_desc_ids, state_ids, field_ids), intents
