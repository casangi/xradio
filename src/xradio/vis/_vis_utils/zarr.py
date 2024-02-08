import numcodecs, os, time
from pathlib import Path
from typing import Dict, Union

import zarr
import xradio
import graphviper.utils.logger as logger

from ._utils.cds import CASAVisSet
from ._zarr.read import read_part_keys, read_partitions, read_subtables
from ._zarr.write import write_metainfo, write_part_keys, write_partitions


def is_zarr_vis(inpath) -> bool:
    """
    Check if a given path has a visibilities dataset in Zarr format

    :param inpath: path to a (possibly) Zarr vis dataset

    :return: whether zarr.open can open this path
    """
    try:
        with zarr.open(Path(inpath, "partition_keys"), mode="r"):
            logger.debug(f"{inpath} can be opened as Zarr format data")
            return True
    except zarr.errors.PathNotFoundError:
        return False


def read_vis(
    inpath: str,
    subtables: bool = True,
    asdm_subtables: bool = False,
) -> CASAVisSet:
    """
    Read a CASAVisSet stored in zarr format.

    :param inpath: Input Zarr path
    :param subtables: Also read and (metainformation) subtables along with main visibilities data.
    :param asdm_subtables: Also read extension subtables named "ASDM_*"

    :return: Main xarray dataset of datasets for this visibility dataset
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

    vers = xradio.__version__
    descr_add = "read_vis from zarr"
    cds = CASAVisSet(
        metainfo=metainfo,
        partitions=dict.fromkeys(part_keys, partitions),
        descr=f"CASA vis set produced by xradio {vers}/{descr_add}",
    )

    return cds


def write_vis(
    cds,
    outpath: str,
    chunks_on_disk: Union[Dict, None] = None,
    compressor: Union[numcodecs.abc.Codec, None] = None,
) -> None:
    """Write CASA vis dataset to zarr format on disk. When
    chunks_on_disk is not specified the chunking in the input dataset
    is used. When chunks_on_disk is specified that dataset is saved
    using that chunking.

    :param cds: CASA visibilities dataset to write to disk
    :param outpath: output path, generally ends in .zarr
    :param chunks_on_disk: a dictionary with the chunk size that will
    be used when writing to disk. For example {'time': 20, 'chan': 6}.
    If chunks_on_disk is not specified the chunking of dataset will
    be used.
    :param compressor: the blosc compressor to use when saving the
    converted data to disk using zarr. If None the zstd compression
    algorithm used with compression level 2.
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
