import toolviper.utils.logger as logger
import os
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import xarray as xr


def read_part_keys(inpath: str) -> List[Tuple]:
    """
    Reads the partition keys from a Zarr-stored cds.

    Parameters
    ----------
    inpath : str
        path to read from

    Returns
    -------
    List[Tuple]
        partition keys from a cds

    """

    xds_keys = xr.open_zarr(
        os.path.join(inpath, "partition_keys"),
    )

    spw_ids = xds_keys.coords["spw_ids"]
    pol_setup_ids = xds_keys.coords["pol_setup_ids"]
    intents = xds_keys.coords["intents"]

    return list(zip(spw_ids.values, pol_setup_ids.values, intents.values))


def read_subtables(inpath: str, asdm_subtables: bool) -> Dict[str, xr.Dataset]:
    """
    Reads the metainfo subtables from a Zarr-stored cds.

    Parameters
    ----------
    inpath : str
        path to read from

    asdm_subtables : bool


    Returns
    -------
    Dict[str, xr.Dataset]
        metainfo subtables from a cds

    """

    metainfo = {}
    metadir = Path(inpath, "metainfo")
    for subt in sorted(metadir.iterdir()):
        if subt.is_dir():
            if not asdm_subtables and subt.name.startswith("ASDM_"):
                logger.debug(f"Not loading ASDM_ subtable {subt.name}...")
                continue

    return metainfo


def read_partitions(inpath: str, part_keys: List[Tuple]) -> Dict[str, xr.Dataset]:
    """
    Reads all the data partitions a Zarr-stored cds.

    Parameters
    ----------
    inpath : str
        path to read from
    part_keys : List[Tuple]


    Returns
    -------
    Dict[str, xr.Dataset]
        partitions from a cds

    """

    partitions = {}
    partdir = Path(inpath, "partitions")
    xds_cnt = 0
    for part in sorted(partdir.iterdir()):
        if part.is_dir() and part.name.startswith("xds_"):
            partitions[part_keys[xds_cnt]] = None
            xds_cnt += 1

    return partitions
