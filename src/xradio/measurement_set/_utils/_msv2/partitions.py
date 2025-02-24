import toolviper.utils.logger as logger
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import xarray as xr

from .msv2_msv3 import ignore_msv2_cols
from .partition_queries import (
    make_partition_ids_by_ddi_intent,
)
from .subtables import subt_rename_ids
from ._tables.read import load_generic_table
from ._tables.read_main_table import read_flat_main_table, read_expanded_main_table
from .._utils.partition_attrs import add_partition_attrs
from .._utils.xds_helper import expand_xds, make_coords, optimal_chunking


PartitionKey = Tuple[Any, ...]
VisSetMetaInfo = Dict[str, xr.Dataset]
VisSetPartitions = Dict[PartitionKey, xr.Dataset]


def make_spw_names_by_ddi(ddi_xds: xr.Dataset, spw_xds: xr.Dataset) -> Dict[int, str]:
    spw_ids_by_ddi = ddi_xds.SPECTRAL_WINDOW_ID[ddi_xds.row].values
    spw_names = spw_xds.NAME[spw_ids_by_ddi].values
    return {ddi: spw_names[ddi] for ddi in np.arange(0, len(spw_names))}


def split_intents(intents: str):
    """
    Make a dict with two scan / subscan levels of intents from an
    intent string from the STATE/OBS_MODE of an MS.

    Parameters
    ----------
    intents : str
        intents "OBS_MODE" string from an MS/STATE row

    Returns
    -------
    Dict[str, list]
        per scan intent list of individual subscan intent strings
    """
    sub_sep = "#"
    if sub_sep not in intents:
        sub_sep = "."
        if sub_sep not in intents:
            return intents

    indiv = [intnt for intnt in intents.split(",")]
    scan_subscan_intents = {}
    for ind in indiv:
        scan = ind.split(sub_sep)
        if len(scan) == 1:
            scan = scan[0]
            subscan = ""
        elif len(scan) == 2:
            scan, subscan = scan
        if scan in scan_subscan_intents:
            scan_subscan_intents[scan].append(subscan)
        else:
            scan_subscan_intents[scan] = [subscan]

    return scan_subscan_intents


def make_part_key(
    xds: xr.Dataset,
    partition_scheme: str,
    intent: str = "",
    scan_state: Union[Tuple, None] = None,
) -> PartitionKey:
    """
    Makes the key that a partition (sub)xds will have in the partitions dictionary of a cds.

    Parameters
    ----------
    xds : xr.Dataset
        partition xds with data and attrs
    partition_scheme : str
        one of the schemes supported in the read_ms_*_partitions() functions
    intent : str (Default value = "")
        partition intent
    scan_state : Union[Tuple, None] (Default value = None)
        scan/state ids, required when partition_scheme != 'ddi'

    Returns
    -------
    PartitionKey
        partition key
    """
    spw_id = xds.attrs["partition_ids"]["spw_id"]
    pol_setup_id = xds.attrs["partition_ids"]["pol_setup_id"]

    if partition_scheme == "ddi":
        part_key = (spw_id, pol_setup_id)
    elif partition_scheme == "intent":
        part_key = (spw_id, pol_setup_id, intent)
    elif partition_scheme == "scan":
        scan, _state = scan_state
        part_key = (spw_id, pol_setup_id, scan)
    elif partition_scheme == "scan/subscan":
        scan, state = scan_state
        part_key = (spw_id, pol_setup_id, scan, state)

    return part_key


def finalize_partitions(
    parts: Dict[str, xr.Dataset], subts: Dict[str, xr.Dataset]
) -> Dict[str, xr.Dataset]:
    """
    Once the partition datasets and the metainfo/subtable datasets
    have been read, add to the partitions:
    - pointing variables from the pointing subtable

    Parameters
    ----------
    parts : Dict[str, xr.Dataset]
        partitions as xarray datasets, as read from an MS main table
    subts : Dict[str, xr.Dataset]
        subtables of an MS read as xarray datasets

    Returns
    -------
    Dict[str, xr.Dataset]
        partitions with additions taken from subtables
    """
    final = parts

    return final
