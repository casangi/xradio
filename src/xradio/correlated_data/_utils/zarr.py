import numcodecs, os, time
from pathlib import Path
from typing import Dict, Union

import zarr
import toolviper.utils.logger as logger

from ._utils.cds import CASAVisSet
from ._zarr.read import read_part_keys, read_partitions, read_subtables
from ._zarr.write import write_metainfo, write_part_keys, write_partitions


def is_zarr_cor(inpath: str) -> bool:
    """
    Check if a given path has a visibilities dataset in Zarr format

    Parameters
    ----------
    inpath : str
        path to a (possibly) Zarr vis dataset

    Returns
    -------
    bool
        whether zarr.open can open this path
    """
    try:
        with zarr.open(Path(inpath, "partition_keys"), mode="r"):
            logger.debug(f"{inpath} can be opened as Zarr format data")
            return True
    except zarr.errors.PathNotFoundError:
        return False


def read_cor(
    inpath: str,
    subtables: bool = True,
    asdm_subtables: bool = False,
) -> CASAVisSet:
    """
    Read a CASAVisSet stored in zarr format.

    Parameters
    ----------
    inpath : str
        Input Zarr path
    subtables : bool (Default value = True)
        Also read and (metainformation) subtables along with main visibilities data.
    asdm_subtables : bool (Default value = False)
        Also read extension subtables named "ASDM_*"

    Returns
    -------
    CASAVisSet
        Main xarray dataset of datasets for this visibility dataset
    """
    inpath = os.path.expanduser(inpath)
    if not os.path.isdir(inpath):
        raise ValueError(f"invalid input filename to read_vis {inpath}")

    logger.info(f"Reading {inpath} as visibilities dataset stored in Zarr format")

    all_start = time.time()

    metainfo = {}
    if subtables:
        metainfo = read_subtables(inpath, asdm_subtables)

    part_keys = read_part_keys(inpath)
    partitions = read_partitions(inpath, part_keys)

    all_time = time.time() - all_start
    logger.info(f"Time to read dataset from_zarr {inpath}: {all_time}")

    vers = "version-WIP"
    descr_add = "read_vis from zarr"
    cds = CASAVisSet(
        metainfo=metainfo,
        partitions=dict.fromkeys(part_keys, partitions),
        descr=f"CASA vis set produced by xradio {vers}/{descr_add}",
    )

    return cds


def write_cor(
    cds: CASAVisSet,
    outpath: str,
    chunks_on_disk: Union[Dict, None] = None,
    compressor: Union[numcodecs.abc.Codec, None] = None,
) -> None:
    """
    Write CASA vis dataset to zarr format on disk. When
    chunks_on_disk is not specified the chunking in the input dataset
    is used. When chunks_on_disk is specified that dataset is saved
    using that chunking.

    Parameters
    ----------
    cds : CASAVisSet
        CASA visibilities dataset to write to disk
    outpath : str
        output path, generally ends in .zarr
    chunks_on_disk : Union[Dict, None] = None (Default value = None)
        a dictionary with the chunk size that will
        be used when writing to disk. For example {'time': 20, 'chan': 6}.
        If chunks_on_disk is not specified the chunking of dataset will
        be used.
    compressor : Union[numcodecs.abc.Codec, None] (Default value = None)
        the blosc compressor to use when saving the
        converted data to disk using zarr. If None the zstd compression
        algorithm used with compression level 2.

    Returns
    -------
    """

    if compressor is None:
        compressor = numcodecs.Blosc(cname="zstd", clevel=2, shuffle=0)

    if os.path.lexists(outpath):
        raise ValueError(f"output vis.zarr path ({outpath}) already exists")

    all_start = time.time()

    write_part_keys(cds.partitions, outpath, compressor)

    write_metainfo(outpath, cds.metainfo, chunks_on_disk, compressor)

    write_partitions(outpath, cds.partitions, chunks_on_disk, compressor)

    all_time = time.time() - all_start
    logger.info(f"Time to prepare and save dataset to_zarr {outpath}: {all_time}")
