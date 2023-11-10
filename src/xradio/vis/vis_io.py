import numcodecs, os
from typing import Dict, List, Tuple, Union

import xarray as xr

from ._vis_utils._utils.cds import CASAVisSet
from xradio.vis import _vis_utils


def read_vis(
    infile: str,
    subtables: bool = True,
    asdm_subtables: bool = False,
    partition_scheme: str = "intent",
    chunks: Union[Tuple[int], List[int]] = None,
    expand: bool = False,
) -> CASAVisSet:
    """Read a MeasurementSet (MSv2 format) into a next generation CASA
    dataset (visibilities dataset as a set of Xarray datasets).

    The MS is partitioned into multiple sub- Xarray datasets (where the data variables are read as
    Dask delayed arrays).
    The MS is partitioned by DDI, which guarantees a fixed data shape per partition (in terms of channels
    and polarizations) and, subject to experimentation, by scan and subscan. This results in multiple
    partitions as xarray datasets (xds) contained within a main xds (mxds).

    :param infile: Input MS filename
    :param rowmap: (to be removed) Dictionary of DDI to tuple of (row indices, channel indices). Returned
    by ms_selection function. Default None ignores selections
    :param subtables: Also read and include subtables along with main table selection. Default False will
    omit subtables (faster)
    :param asdm_subtables: in addition to MeasurementSet subtables (if enabled), also read extension
    subtables names "ASDM_*"
    :param partition_scheme: (experimenting) Whether to partition sub-xds datasets by scan/subscan
    (in addition to DDI), or other alternative partitioning schemes. Accepted values: 'scan/subscan',
    'scan', 'ddi', 'intent'. Default: 'intent'
    :param chunks: Can be used to set a specific chunk shape (with a tuple of ints), or to control the
    optimization used for automatic chunking (with a list of ints). A tuple of ints in the form of (row,
    chan, pol) will use a fixed chunk shape. A list or numpy array of ints in the form of [idx1, etc]
    will trigger auto-chunking optimized for the given indices, with row=0, chan=1, pol=2. Default None
    uses auto-chunking with a best fit across all dimensions (probably sub-optimal for most cases).
    :param expand: (to be removed) Whether or not to return the original flat row structure of the MS (False)
    or expand the rows to time x baseline dimensions (True). Expanding the rows allows for easier indexing
    and parallelization across time and baseline dimensions, at the cost of some conversion time. Default
    False
    :param **kwargs: (to be removed?) Selection parameters from the standard way of making CASA MS
    selections. Supported keys are: spw, field, scan, baseline, time, scanintent, uvdist, polarization,
    array, observation.  Values are strings.

    :return: ngCASA visisbilities dataset, essentially made of two dictionaries of
    metainformation and data partitions
    """
    infile = os.path.expanduser(infile)
    if not os.path.isdir(infile):
        raise ValueError(f"invalid input filename to read_vis {infile}")

    if _vis_utils.zarr.is_zarr_vis(infile):
        return _vis_utils.zarr.read_vis(infile, subtables, asdm_subtables)
    else:
        return _vis_utils.ms.read_ms(
            infile, subtables, asdm_subtables, partition_scheme, chunks, expand
        )


def load_vis_block(
    infile: str,
    block_des: Dict[str, slice],
    partition_key: Tuple[int, int, str],
    subtables: List[str] = None,
) -> Dict[Tuple[int, int], xr.Dataset]:
    """Read a chunk of a visibilities dataset into an Xarray dataset, loading the
    data in memory.
    The input format support is the MeasurementSet v2 (MSv2 format)

    :param infile: Input visibilities path
    :param block_des: specification of chunk to load

    :return: CASA visibilities dataset holding a chunk of visibility data, for one
    partition
    (spw_id, pol_setup_id, intent_string triplet)
    """
    # TODO: use the input partition_key
    # the intent str of the partition_key is not yet effectively used
    # TODO: support subtables list
    return _vis_utils.ms.load_vis_chunk(infile, block_des, partition_key)


def write_vis(
    cds: CASAVisSet,
    outpath: str,
    chunks_on_disk: Union[Dict, None] = None,
    compressor: Union[numcodecs.abc.Codec, None] = None,
    out_format: str = "zarr",
) -> None:
    """Write CASA vis dataset to disk.
    The disk format supported is "zarr". When chunks_on_disk is not specified the
    chunking in the input dataset is used. When chunks_on_disk is specified that
    dataset is saved using that chunking.

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

    if out_format == "zarr":
        return _vis_utils.zarr.write_vis(cds, outpath, chunks_on_disk, compressor)
    else:
        raise ValueError(f"Unsupported output format: {out_format}")
