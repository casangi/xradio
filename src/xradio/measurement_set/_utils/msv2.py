import os
import toolviper.utils.logger as logger
from typing import List, Tuple, Union

from xradio.measurement_set._utils._utils.cds import CASAVisSet
from xradio.measurement_set._utils._msv2.partitions import (
    finalize_partitions,
    read_ms_ddi_partitions,
    read_ms_scan_subscan_partitions,
)
from xradio.measurement_set._utils._msv2.subtables import read_ms_subtables
from xradio.measurement_set._utils._utils.xds_helper import vis_xds_packager_cds


def read_ms(
    infile: str,
    subtables: bool = True,
    asdm_subtables: bool = False,
    partition_scheme: str = "intent",
    chunks: Union[Tuple[int], List[int]] = None,
    expand: bool = False,
    **kwargs: str,
) -> CASAVisSet:
    """
    Read a MeasurementSet (MSv2 format) into a next generation CASA
    dataset (visibilities dataset as a set of Xarray datasets).

    The MS is partitioned into multiple sub- Xarray datasets (where the data variables are read as
    Dask delayed arrays).
    The MS is partitioned by DDI, which guarantees a fixed data shape per partition (in terms of channels
    and polarizations) and, subject to experimentation, by scan and subscan. This results in multiple
    partitions as xarray datasets (xds) contained within a main xds (mxds).

    Parameters
    ----------
    infile : str
        Input MS filename
    subtables : bool (Default value = True)
        Also read and include subtables along with main table selection. Default False will
        omit subtables (faster)
    asdm_subtables : bool (Default value = False)
        in addition to MeasurementSet subtables (if enabled), also read extension
        subtables named "ASDM_*"
    partition_scheme : str (Default value = "intent")
        experimenting) Whether to partition sub-xds datasets by scan/subscan
        (in addition to DDI), or other alternative partitioning schemes. Accepted values: 'scan/subscan',
        'scan', 'ddi', 'intent'. Default: 'intent'
    chunks : Union[Tuple[int], List[int]] (Default value = None)
        Can be used to set a specific chunk shape (with a tuple of ints), or to control the
        optimization used for automatic chunking (with a list of ints). A tuple of ints in the form of (row,
        chan, pol) will use a fixed chunk shape. A list or numpy array of ints in the form of [idx1, etc]
        will trigger auto-chunking optimized for the given indices, with row=0, chan=1, pol=2. Default None
        uses auto-chunking with a best fit across all dimensions (probably sub-optimal for most cases).
    expand : bool (Default value = False)
        to be removed) Whether or not to return the original flat row structure of the MS (False)
        or expand the rows to time x baseline dimensions (True). Expanding the rows allows for easier indexing
        and parallelization across time and baseline dimensions, at the cost of some conversion time.
    **kwargs: str :


    Returns
    -------
    CASAVisSet
        Main xarray dataset of datasets for this visibility dataset
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
