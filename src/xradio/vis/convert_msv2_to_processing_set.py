import graphviper.utils.logger as logger
import numcodecs
from typing import Dict, Union

import dask

from xradio.vis._vis_utils._ms.msv2_msv3 import ignore_msv2_cols
from xradio.vis._vis_utils._ms.partition_queries import (
    create_partition_enumerated_product,
)
from xradio.vis._vis_utils._ms.conversion import convert_and_write_partition


def convert_msv2_to_processing_set(
    in_file: str,
    out_file: str,
    partition_scheme: {"ddi_intent_field", "ddi_state_field"} = "ddi_intent_field",
    main_chunksize: Union[Dict, str, None] = None,
    pointing_chunksize: Union[Dict, str, None] = None,
    compressor: numcodecs.abc.Codec = numcodecs.Zstd(level=2),
    storage_backend="zarr",
    parallel: bool = False,
    overwrite: bool = False,
):
    """Convert a Measurement Set v2 into a Processing Set of Measurement Set v4.

    Parameters
    ----------
    in_file : str
        Input MS name.
    out_file : str
        Output PS name.
    partition_scheme : {"ddi_intent_field", "ddi_state_field"}, optional
        A MS v4 can only contain a single spectral window, polarization setup, intent, and field. Consequently, the MS v2 is partitioned when converting to MS v4.
        The partition_scheme "ddi_intent_field" gives the largest partition that meets the MS v4 specification. The partition_scheme "ddi_state_field" gives a finer granularity where the data is also partitioned by state (the state partitioning will ensure a single intent).
        By default, "ddi_intent_field".
    main_chunksize : Union[Dict, str, None], optional
        A dictionary that defines the chunk size of the main dataset. Acceptable keys are "time", "baseline", "antenna", "frequency", "polarization". By default, None.
    pointing_chunksize : Union[Dict, str, None], optional
        A dictionary that defines the chunk size of the pointing dataset. Acceptable keys are "time", "antenna", "polarization". By default, None.
    compressor : numcodecs.abc.Codec, optional
        The Blosc compressor to use when saving the converted data to disk using Zarr, by default numcodecs.Zstd(level=2).
    storage_backend : {"zarr", "netcdf"}, optional
        The on-disk format to use. "netcdf" is not yet implemented.
    parallel : bool, optional
        Makes use of Dask to execute conversion in parallel, by default False.
    overwrite : bool, optional
        Whether to overwrite an existing processing set, by default False.
    """

    partition_enumerated_product, intents = create_partition_enumerated_product(
        in_file, partition_scheme
    )

    delayed_list = []
    for idx, pair in partition_enumerated_product:
        ddi, state_id, field_id = pair
        logger.debug(
            "DDI " + str(ddi) + ", STATE " + str(state_id) + ", FIELD " + str(field_id)
        )

        if partition_scheme == "ddi_intent_field":
            intent = intents[idx[1]]
        else:
            intent = intents[idx[1]] + "_" + str(state_id)

        if parallel:
            delayed_list.append(
                dask.delayed(convert_and_write_partition)(
                    in_file,
                    out_file,
                    intent,
                    ddi,
                    state_id,
                    field_id,
                    ignore_msv2_cols=ignore_msv2_cols,
                    main_chunksize=main_chunksize,
                    compressor=compressor,
                    overwrite=overwrite,
                )
            )
        else:
            convert_and_write_partition(
                in_file,
                out_file,
                intent,
                ddi,
                state_id,
                field_id,
                ignore_msv2_cols=ignore_msv2_cols,
                main_chunksize=main_chunksize,
                compressor=compressor,
                storage_backend=storage_backend,
                overwrite=overwrite,
            )

    if parallel:
        dask.compute(delayed_list)
