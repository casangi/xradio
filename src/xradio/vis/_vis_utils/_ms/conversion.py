import numcodecs
import time
from .._zarr.encoding import add_encoding
from typing import Dict, Union
import graphviper.utils.logger as logger

import numpy as np
import xarray as xr

from .msv4_infos import create_field_info
from .msv4_sub_xdss import create_ant_xds, create_pointing_xds, create_weather_xds
from .msv2_to_msv4_meta import (
    column_description_casacore_to_msv4_measure,
    create_attribute_metadata,
    col_to_data_variable_names,
    col_dims,
)
from .partition_queries import create_taql_query_and_file_name
from .subtables import subt_rename_ids
from ._tables.table_query import open_table_ro, open_query
from ._tables.read import (
    convert_casacore_time,
    extract_table_attributes,
    read_col_conversion,
    read_generic_table,
)
from ._tables.read_main_table import get_baselines, get_baseline_indices, get_utimes_tol
from .._utils.stokes_types import stokes_types
from xradio.vis._vis_utils._ms.optimised_functions import unique_1d


def check_if_consistent(col, col_name):
    """_summary_

    Parameters
    ----------
    col : _type_
        _description_
    col_name : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """

    col_unique = unique_1d(col)
    assert len(col_unique) == 1, col_name + " is not consistent."
    return col_unique[0]


# TODO: if the didxs are not used in read_col_conversion, remove didxs from here (and convert_and_write_partition)
def calc_indx_for_row_split(tb_tool, taql_where):
    baselines = get_baselines(tb_tool)
    col_names = tb_tool.colnames()
    cshapes = [
        np.array(tb_tool.getcell(col, 0)).shape
        for col in col_names
        if tb_tool.iscelldefined(col, 0)
    ]

    freq_cnt, pol_cnt = [(cc[0], cc[1]) for cc in cshapes if len(cc) == 2][0]
    utimes, tol = get_utimes_tol(tb_tool, taql_where)

    tidxs = np.searchsorted(utimes, tb_tool.getcol("TIME"))

    ts_ant1, ts_ant2 = (
        tb_tool.getcol("ANTENNA1"),
        tb_tool.getcol("ANTENNA2"),
    )

    ts_bases = np.column_stack((ts_ant1, ts_ant2))
    bidxs = get_baseline_indices(baselines, ts_bases)

    # some antenna 2"s will be out of bounds for this chunk, store rows that are in bounds

    didxs = np.where((bidxs >= 0) & (bidxs < len(baselines)))[0]

    baseline_ant1_id = baselines[:, 0]
    baseline_ant2_id = baselines[:, 1]

    return (
        tidxs,
        bidxs,
        didxs,
        baseline_ant1_id,
        baseline_ant2_id,
        convert_casacore_time(utimes, False),
    )


def create_coordinates(
    xds, in_file, ddi, utime, interval, baseline_ant1_id, baseline_ant2_id
):
    coords = {
        "time": utime,
        "baseline_antenna1_id": ("baseline_id", baseline_ant1_id),
        "baseline_antenna2_id": ("baseline_id", baseline_ant2_id),
        "uvw_label": ["u", "v", "w"],
        "baseline_id": np.arange(len(baseline_ant1_id)),
    }

    ddi_xds = read_generic_table(in_file, "DATA_DESCRIPTION").sel(row=ddi)
    pol_setup_id = ddi_xds.polarization_id.values
    spw_id = ddi_xds.spectral_window_id.values

    spw_xds = read_generic_table(
        in_file,
        "SPECTRAL_WINDOW",
        rename_ids=subt_rename_ids["SPECTRAL_WINDOW"],
    ).sel(spectral_window_id=spw_id)
    coords["frequency"] = spw_xds["chan_freq"].data[
        ~(np.isnan(spw_xds["chan_freq"].data))
    ]

    pol_xds = read_generic_table(
        in_file,
        "POLARIZATION",
        rename_ids=subt_rename_ids["POLARIZATION"],
    )
    num_corr = int(pol_xds["num_corr"][pol_setup_id].values)
    coords["polarization"] = np.vectorize(stokes_types.get)(
        pol_xds["corr_type"][pol_setup_id, :num_corr].values
    )

    xds = xds.assign_coords(coords)

    ###### Create Frequency Coordinate ######
    freq_column_description = spw_xds.attrs["other"]["msv2"]["ctds_attrs"][
        "column_descriptions"
    ]

    msv4_measure = column_description_casacore_to_msv4_measure(
        freq_column_description["CHAN_FREQ"], ref_code=spw_xds["meas_freq_ref"].data
    )
    xds.frequency.attrs.update(msv4_measure)

    xds.frequency.attrs["spectral_window_name"] = str(spw_xds.name.values)
    msv4_measure = column_description_casacore_to_msv4_measure(
        freq_column_description["REF_FREQUENCY"], ref_code=spw_xds["meas_freq_ref"].data
    )
    xds.frequency.attrs["reference_frequency"] = {
        "dims": "",
        "data": float(spw_xds.ref_frequency.values),
        "attrs": msv4_measure,
    }
    xds.frequency.attrs["spw_id"] = spw_id

    # xds.frequency.attrs["effective_channel_width"] = "EFFECTIVE_CHANNEL_WIDTH"
    # Add if doppler table is present
    # xds.frequency.attrs["doppler_velocity"] =
    # xds.frequency.attrs["doppler_type"] =

    unique_chan_width = unique_1d(
        spw_xds.chan_width.data[np.logical_not(np.isnan(spw_xds.chan_width.data))]
    )
    # assert len(unique_chan_width) == 1, "Channel width varies for spw."
    # xds.frequency.attrs["channel_width"] = spw_xds.chan_width.data[
    #    ~(np.isnan(spw_xds.chan_width.data))
    # ]  # unique_chan_width[0]
    msv4_measure = column_description_casacore_to_msv4_measure(
        freq_column_description["CHAN_WIDTH"], ref_code=spw_xds["meas_freq_ref"].data
    )
    if not msv4_measure:
        msv4_measure["type"] = "quantity"
        msv4_measure["units"] = ["Hz"]
    xds.frequency.attrs["channel_width"] = {
        "dims": "",
        "data": np.abs(unique_chan_width[0]),
        "attrs": msv4_measure,
    }

    ###### Create Time Coordinate ######
    main_table_attrs = extract_table_attributes(in_file)
    main_column_descriptions = main_table_attrs["column_descriptions"]
    msv4_measure = column_description_casacore_to_msv4_measure(
        main_column_descriptions["TIME"]
    )
    xds.time.attrs.update(msv4_measure)

    msv4_measure = column_description_casacore_to_msv4_measure(
        main_column_descriptions["INTERVAL"]
    )
    if not msv4_measure:
        msv4_measure["type"] = "quantity"
        msv4_measure["units"] = ["s"]
    xds.time.attrs["integration_time"] = {
        "dims": "",
        "data": interval,
        "attrs": msv4_measure,
    }
    xds.time.attrs["effective_integration_time"] = "EFFECTIVE_INTEGRATION_TIME"
    return xds


def create_data_variables(
    in_file, xds, tb_tool, time_baseline_shape, tidxs, bidxs, didxs
):
    # Create Data Variables
    col_names = tb_tool.colnames()

    main_table_attrs = extract_table_attributes(in_file)
    main_column_descriptions = main_table_attrs["column_descriptions"]
    for col in col_names:
        if col in col_to_data_variable_names:
            if (col == "WEIGHT") and ("WEIGHT_SPECTRUM" in col_names):
                continue
            try:
                start = time.time()
                if col == "WEIGHT":
                    xds[col_to_data_variable_names[col]] = xr.DataArray(
                        np.tile(
                            read_col_conversion(
                                tb_tool,
                                col,
                                time_baseline_shape,
                                tidxs,
                                bidxs,
                            )[:, :, None, :],
                            (1, 1, xds.sizes["frequency"], 1),
                        ),
                        dims=col_dims[col],
                    )

                else:
                    xds[col_to_data_variable_names[col]] = xr.DataArray(
                        read_col_conversion(
                            tb_tool,
                            col,
                            time_baseline_shape,
                            tidxs,
                            bidxs,
                        ),
                        dims=col_dims[col],
                    )
                    logger.debug(
                        "Time to read column "
                        + str(col)
                        + " : "
                        + str(time.time() - start)
                    )
            except:
                # logger.debug("Could not load column",col)
                continue

            xds[col_to_data_variable_names[col]].attrs.update(
                create_attribute_metadata(col, main_column_descriptions)
            )


def convert_and_write_partition(
    in_file: str,
    out_file: str,
    intent: str,
    ddi: int = 0,
    state_ids=None,
    field_id: int = None,
    ignore_msv2_cols: Union[list, None] = None,
    main_chunksize: Union[Dict, None] = None,
    compressor: numcodecs.abc.Codec = numcodecs.Zstd(level=2),
    storage_backend="zarr",
    overwrite: bool = False,
):
    """_summary_

    Parameters
    ----------
    in_file : str
        _description_
    out_file : str
        _description_
    intent : str
        _description_
    ddi : int, optional
        _description_, by default 0
    state_ids : _type_, optional
        _description_, by default None
    field_id : int, optional
        _description_, by default None
    ignore_msv2_cols : Union[list, None], optional
        _description_, by default None
    main_chunksize : Union[Dict, None], optional
        _description_, by default None
    compressor : numcodecs.abc.Codec, optional
        _description_, by default numcodecs.Zstd(level=2)
    storage_backend : str, optional
        _description_, by default "zarr"
    overwrite : bool, optional
        _description_, by default False

    Returns
    -------
    _type_
        _description_
    """
    if ignore_msv2_cols is None:
        ignore_msv2_cols = []

    taql_where, file_name = create_taql_query_and_file_name(
        out_file, intent, state_ids, field_id, ddi
    )

    start = time.time()
    with open_table_ro(in_file) as mtable:
        taql_main = f"select * from $mtable {taql_where}"
        with open_query(mtable, taql_main) as tb_tool:
            if tb_tool.nrows() == 0:
                tb_tool.close()
                mtable.close()
                return xr.Dataset(), {}, {}

            logger.debug("Starting a real convert_and_write_partition")
            (
                tidxs,
                bidxs,
                didxs,
                baseline_ant1_id,
                baseline_ant2_id,
                utime,
            ) = calc_indx_for_row_split(tb_tool, taql_where)
            time_baseline_shape = (len(utime), len(baseline_ant1_id))
            logger.debug("Calc indx for row split " + str(time.time() - start))

            start = time.time()
            xds = xr.Dataset()
            # interval = check_if_consistent(tb_tool.getcol("INTERVAL"), "INTERVAL")
            interval = tb_tool.getcol("INTERVAL")

            interval_unique = unique_1d(interval)
            if len(interval_unique) > 1:
                print(
                    "Integration time (interval) not consitent in partition, using median."
                )
                interval = np.median(interval)
            else:
                interval = interval_unique[0]

            xds = create_coordinates(
                xds, in_file, ddi, utime, interval, baseline_ant1_id, baseline_ant2_id
            )
            logger.debug("Time create coordinates " + str(time.time() - start))

            start = time.time()
            create_data_variables(
                in_file, xds, tb_tool, time_baseline_shape, tidxs, bidxs, didxs
            )
            logger.debug("Time create data variables " + str(time.time() - start))

            # Create field_info
            start = time.time()
            field_id = check_if_consistent(tb_tool.getcol("FIELD_ID"), "FIELD_ID")
            field_info = create_field_info(in_file, field_id)
            logger.debug("Time field info " + str(time.time() - start))

            # Create ant_xds
            start = time.time()
            ant_xds = create_ant_xds(in_file)
            logger.debug("Time ant xds  " + str(time.time() - start))

            # Create weather_xds
            start = time.time()
            weather_xds = create_weather_xds(in_file)
            logger.debug("Time weather " + str(time.time() - start))

            start = time.time()
            pointing_xds = create_pointing_xds(in_file)
            logger.debug("Time pointing " + str(time.time() - start))

            start = time.time()
            # Fix UVW frame
            # From CASA fixvis docs: clean and the im tool ignore the reference frame claimed by the UVW column (it is often mislabelled as ITRF when it is really FK5 (J2000)) and instead assume the (u, v, w)s are in the same frame as the phase tracking center. calcuvw does not yet force the UVW column and field centers to use the same reference frame! Blank = use the phase tracking frame of vis.
            xds.UVW.attrs["frame"] = field_info["phase_direction"]["attrs"]["frame"]

            xds.attrs["intent"] = intent
            xds.attrs["ddi"] = ddi

            # Time and frequency should always be increasing
            if len(xds.frequency) > 1 and xds.frequency[1] - xds.frequency[0] < 0:
                xds = xds.sel(frequency=slice(None, None, -1))

            if len(xds.time) > 1 and xds.time[1] - xds.time[0] < 0:
                xds = xds.sel(time=slice(None, None, -1))

            # Add data_groups and field_info
            xds.attrs["data_groups"] = {}
            if "VISIBILITY" in xds:
                xds.attrs["data_groups"]["base"] = {
                    "visibility": "VISIBILITY",
                    "flag": "FLAG",
                    "weight": "WEIGHT",
                    "uvw": "UVW",
                }
                xds.VISIBILITY.attrs["field_info"] = field_info

            if "VISIBILITY_CORRECTED" in xds:
                xds.attrs["data_groups"]["corrected"] = {
                    "visibility": "VISIBILITY_CORRECTED",
                    "flag": "FLAG",
                    "weight": "WEIGHT",
                    "uvw": "UVW",
                }
                xds.VISIBILITY_CORRECTED.attrs["field_info"] = field_info

            if "SPECTRUM" in xds:
                xds.attrs["data_groups"]["base"] = {
                    "spectrum": "SPECTRUM",
                    "flag": "FLAG",
                    "weight": "WEIGHT",
                    "uvw": "UVW",
                }
                xds.SPECTRUM.attrs["field_info"] = field_info

            if "SPECTRUM_CORRECTED" in xds:
                xds.attrs["data_groups"]["corrected"] = {
                    "spectrum": "SPECTRUM_CORRECTED",
                    "flag": "FLAG",
                    "weight": "WEIGHT",
                    "uvw": "UVW",
                }
                xds.SPECTRUM_CORRECTED.attrs["field_info"] = field_info

            if overwrite:
                mode = "w"
            else:
                mode = "w-"

            add_encoding(xds, compressor=compressor, chunks=main_chunksize)
            logger.debug("Time add compressor and chunk " + str(time.time() - start))

            start = time.time()
            if storage_backend == "zarr":
                xds.to_zarr(store=file_name + "/MAIN", mode=mode)
                ant_xds.to_zarr(store=file_name + "/ANTENNA", mode=mode)
                pointing_xds.to_zarr(store=file_name + "/POINTING", mode=mode)
                if weather_xds:
                    weather_xds.to_zarr(store=file_name + "/WEATHER", mode=mode)
            elif storage_backend == "netcdf":
                # xds.to_netcdf(path=file_name+"/MAIN", mode=mode) #Does not work
                raise
            logger.debug("Write data  " + str(time.time() - start))

    # logger.info("Saved ms_v4 " + file_name + " in " + str(time.time() - start_with) + "s")
