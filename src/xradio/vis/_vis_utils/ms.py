import os
import graphviper.utils.logger as logger
from typing import Dict, List, Tuple, Union

import xarray as xr

from ._utils.cds import CASAVisSet
from ._ms.chunks import load_main_chunk
from ._ms.partitions import (
    finalize_partitions,
    read_ms_ddi_partitions,
    read_ms_scan_subscan_partitions,
)
from ._ms.subtables import read_ms_subtables
from ._utils.xds_helper import vis_xds_packager_cds


def read_ms(
    infile: str,
    subtables: bool = True,
    asdm_subtables: bool = False,
    partition_scheme: str = "intent",
    chunks: Union[Tuple[int], List[int]] = None,
    expand: bool = False,
    **kwargs: str,
) -> CASAVisSet:
    """Read a MeasurementSet (MSv2 format) into a next generation CASA
    dataset (visibilities dataset as a set of Xarray datasets).

    The MS is partitioned into multiple sub- Xarray datasets (where the data variables are read as
    Dask delayed arrays).
    The MS is partitioned by DDI, which guarantees a fixed data shape per partition (in terms of channels
    and polarizations) and, subject to experimentation, by scan and subscan. This results in multiple
    partitions as xarray datasets (xds) contained within a main xds (mxds).

    :param infile: Input MS filename
    :param subtables: Also read and include subtables along with main table selection. Default False will
    omit subtables (faster)
    :param asdm_subtables: in addition to MeasurementSet subtables (if enabled), also read extension
    subtables named "ASDM_*"
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

    :return: Main xarray dataset of datasets for this visibility dataset
    """

    infile = os.path.expanduser(infile)
    if not os.path.isdir(infile):
        raise ValueError(f"invalid input filename to read_ms {infile}")

    # Several alternatives to experiment for now
    part_descr = {
        "intent": "scan/subscan intent + DDI",
        "ddi": "DDI",
        "scan": "scan + DDI",
        "scan/subscan": "scan + subscan + DDI",
    }

    if partition_scheme not in part_descr:
        raise ValueError(f"Invalid partition_scheme: {partition_scheme}")

    logger.info(
        f"Reading {infile} as MSv2 and applying partitioning by {part_descr[partition_scheme]}"
    )

    if partition_scheme == "ddi":
        logger.info(f"Reading {infile} as MSv2 and applying DDI partitioning")
        # get the indices of the ms selection (if any)
        # rowmap = ms_selection(infile, verbose=verbose, **kwargs)
        rowmap = None
        parts, subts, done_subts = read_ms_ddi_partitions(
            infile, expand, rowmap, chunks
        )
    else:
        parts, subts, done_subts = read_ms_scan_subscan_partitions(
            infile, partition_scheme, expand, chunks
        )

    if subtables:
        subts.update(read_ms_subtables(infile, done_subts, asdm_subtables))

    parts = finalize_partitions(parts, subts)

    # build the visibilities container (metainfo + partitions) to return
    cds = vis_xds_packager_cds(subts, parts, "read_ms")
    return cds


def load_vis_chunk(
    infile: str,
    block_des: Dict[str, slice],
    partition_key: Tuple[int, int, str],
) -> Dict[Tuple[int, int], xr.Dataset]:
    """Read a chunk of a MeasurementSet (MSv2 format) into an Xarray
    dataset, loading the data in memory.

    :param infile: Input MS filename
    :param block_des: specification of chunk to load

    :return: Xarray datasets with chunk of visibility data, one per DDI
    (spw_id, pol_setup_id pair)
    """
    infile = os.path.expanduser(infile)

    logger.info(f"Loading from {infile} as MSv2 a chunk of data into memory")

    if not os.path.isdir(infile):
        raise ValueError(f"invalid input filename to read_ms {infile}")

    orig_chunk_to_improve = load_main_chunk(infile, block_des)
    res = vis_xds_packager_cds(
        subtables={},
        partitions={partition_key: orig_chunk_to_improve},
        descr_add="load_vis_block",
    )
    return res
