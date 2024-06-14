import graphviper.utils.logger as logger
import numcodecs
from typing import Dict, Union

import dask

from xradio.vis._vis_utils._ms.partition_queries import (
    create_partition_enumerated_product,
)
from xradio.vis._vis_utils._ms.conversion import convert_and_write_partition


def convert_msv2_to_processing_set(
    in_file: str,
    out_file: str,
    partition_scheme: {"ddi_intent_field", "ddi_state_field"} = "ddi_intent_field",
    main_chunksize: Union[Dict, float, None] = None,
    with_pointing: bool = True,
    pointing_chunksize: Union[Dict, float, None] = None,
    pointing_interpolate: bool = False,
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
    main_chunksize : Union[Dict, float, None], optional
        Defines the chunk size of the main dataset. If given as a dictionary, defines the sizes of several dimensions, and acceptable keys are "time", "baseline_id", "antenna_id", "frequency", "polarization". If given as a float, gives the size of a chunk in GiB. By default, None.
    with_pointing : bool, optional
        Whether to convert the POINTING subtable into pointing sub-datasets
    pointing_chunksize : Union[Dict, float, None], optional
        Defines the chunk size of the pointing dataset. If given as a dictionary, defines the sizes of several dimensions, acceptable keys are "time" and "antenna_id". If given as a float, defines the size of a chunk in GiB. By default, None.
    pointing_interpolate : bool, optional
        Whether to interpolate the time axis of the pointing sub-dataset to the time axis of the main dataset
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
    ms_v4_id = 0
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
                    ms_v4_id,
                    ddi,
                    state_id,
                    field_id,
                    main_chunksize=main_chunksize,
                    with_pointing=with_pointing,
                    pointing_chunksize=pointing_chunksize,
                    pointing_interpolate=pointing_interpolate,
                    compressor=compressor,
                    overwrite=overwrite,
                )
            )
        else:
            convert_and_write_partition(
                in_file,
                out_file,
                intent,
                ms_v4_id,
                ddi,
                state_id,
                field_id,
                main_chunksize=main_chunksize,
                with_pointing=with_pointing,
                pointing_chunksize=pointing_chunksize,
                pointing_interpolate=pointing_interpolate,
                compressor=compressor,
                storage_backend=storage_backend,
                overwrite=overwrite,
            )
        ms_v4_id = ms_v4_id + 1

    if parallel:
        dask.compute(delayed_list)
