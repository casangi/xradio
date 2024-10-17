import toolviper.utils.logger as logger
import numcodecs
from typing import Dict, Union

import dask

from xradio.measurement_set._utils._msv2.partition_queries import create_partitions
from xradio.measurement_set._utils._msv2.conversion import convert_and_write_partition


def convert_msv2_to_processing_set(
    in_file: str,
    out_file: str,
    partition_scheme: list = ["FIELD_ID"],
    main_chunksize: Union[Dict, float, None] = None,
    with_pointing: bool = True,
    pointing_chunksize: Union[Dict, float, None] = None,
    pointing_interpolate: bool = False,
    ephemeris_interpolate: bool = False,
    phase_cal_interpolate: bool = False,
    sys_cal_interpolate: bool = False,
    use_table_iter: bool = False,
    compressor: numcodecs.abc.Codec = numcodecs.Zstd(level=2),
    storage_backend: str = "zarr",
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
    partition_scheme : list, optional
        A MS v4 can only contain a single data description (spectral window and polarization setup), and observation mode. Consequently, the MS v2 is partitioned when converting to MS v4.
        In addition to data description and polarization setup a finer partitioning is possible by specifying a list of partitioning keys. Any combination of the following keys are possible:
        "FIELD_ID", "SCAN_NUMBER", "STATE_ID", "SOURCE_ID", "SUB_SCAN_NUMBER". For mosaics where the phase center is rapidly changing (such as VLA on the fly mosaics)
        partition_scheme should be set to an empty list []. By default, ["FIELD_ID"].
    main_chunksize : Union[Dict, float, None], optional
        Defines the chunk size of the main dataset. If given as a dictionary, defines the sizes of several dimensions, and acceptable keys are "time", "baseline_id", "antenna_id", "frequency", "polarization". If given as a float, gives the size of a chunk in GiB. By default, None.
    with_pointing : bool, optional
        Whether to convert the POINTING subtable into pointing sub-datasets
    pointing_chunksize : Union[Dict, float, None], optional
        Defines the chunk size of the pointing dataset. If given as a dictionary, defines the sizes of several dimensions, acceptable keys are "time" and "antenna_id". If given as a float, defines the size of a chunk in GiB. By default, None.
    pointing_interpolate : bool, optional
        Whether to interpolate the time axis of the pointing sub-dataset to the time axis of the main dataset
    ephemeris_interpolate : bool, optional
        Whether to interpolate the time axis of the ephemeris data variables (of the field_and_source sub-dataset) to the time axis of the main dataset
    phase_cal_interpolate : bool, optional
        Whether to interpolate the time axis of the phase calibration data variables to the time axis of the main dataset
    sys_cal_interpolate : bool, optional
        Whether to interpolate the time axis of the system calibration data variables (sys_cal_xds) to the time axis of the main dataset
    use_table_iter : bool, optional
        Whether to use the table iterator to read the main table of the MS v2. This should be set to True when reading datasets with large number of rows and few partitions, by default False.
    compressor : numcodecs.abc.Codec, optional
        The Blosc compressor to use when saving the converted data to disk using Zarr, by default numcodecs.Zstd(level=2).
    storage_backend : {"zarr", "netcdf"}, optional
        The on-disk format to use. "netcdf" is not yet implemented.
    parallel : bool, optional
        Makes use of Dask to execute conversion in parallel, by default False.
    overwrite : bool, optional
        Whether to overwrite an existing processing set, by default False.
    """

    partitions = create_partitions(in_file, partition_scheme=partition_scheme)
    logger.info("Number of partitions: " + str(len(partitions)))
    delayed_list = []

    for ms_v4_id, partition_info in enumerate(partitions):
        # print(ms_v4_id,len(partition_info['FIELD_ID']))

        logger.info(
            "OBSERVATION_ID "
            + str(partition_info["OBSERVATION_ID"])
            + ", DDI "
            + str(partition_info["DATA_DESC_ID"])
            + ", STATE "
            + str(partition_info["STATE_ID"])
            + ", FIELD "
            + str(partition_info["FIELD_ID"])
            + ", SCAN "
            + str(partition_info["SCAN_NUMBER"])
            + (
                ", ANTENNA " + str(partition_info["ANTENNA1"])
                if "ANTENNA1" in partition_info
                else ""
            )
        )

        # prepend '0' to ms_v4_id as needed
        ms_v4_id = f"{ms_v4_id:0>{len(str(len(partitions) - 1))}}"
        if parallel:
            delayed_list.append(
                dask.delayed(convert_and_write_partition)(
                    in_file,
                    out_file,
                    ms_v4_id,
                    partition_info=partition_info,
                    use_table_iter=use_table_iter,
                    partition_scheme=partition_scheme,
                    main_chunksize=main_chunksize,
                    with_pointing=with_pointing,
                    pointing_chunksize=pointing_chunksize,
                    pointing_interpolate=pointing_interpolate,
                    ephemeris_interpolate=ephemeris_interpolate,
                    phase_cal_interpolate=phase_cal_interpolate,
                    sys_cal_interpolate=sys_cal_interpolate,
                    compressor=compressor,
                    overwrite=overwrite,
                )
            )
        else:
            convert_and_write_partition(
                in_file,
                out_file,
                ms_v4_id,
                partition_info=partition_info,
                use_table_iter=use_table_iter,
                partition_scheme=partition_scheme,
                main_chunksize=main_chunksize,
                with_pointing=with_pointing,
                pointing_chunksize=pointing_chunksize,
                pointing_interpolate=pointing_interpolate,
                ephemeris_interpolate=ephemeris_interpolate,
                phase_cal_interpolate=phase_cal_interpolate,
                sys_cal_interpolate=sys_cal_interpolate,
                compressor=compressor,
                overwrite=overwrite,
            )

    if parallel:
        dask.compute(delayed_list)
