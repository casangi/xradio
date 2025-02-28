import numcodecs, os, time
from pathlib import Path
from typing import Dict, Union

import zarr
import toolviper.utils.logger as logger

from ._utils.cds import CASAVisSet


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

    # write_part_keys(cds.partitions, outpath, compressor)

    # write_metainfo(outpath, cds.metainfo, chunks_on_disk, compressor)

    # write_partitions(outpath, cds.partitions, chunks_on_disk, compressor)

    all_time = time.time() - all_start
    logger.info(f"Time to prepare and save dataset to_zarr {outpath}: {all_time}")
