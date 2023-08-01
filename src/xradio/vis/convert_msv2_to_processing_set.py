#from numcodecs.zstd import Zstd
import numcodecs
from typing import Dict, List, Tuple, Union
#import Union
from xradio.vis._vis_utils._helpers.partitions import (
    finalize_partitions,
    read_ms_ddi_partitions,
    read_ms_scan_subscan_partitions,
    make_spw_names_by_ddi,
    make_partition_ids_by_ddi_intent,
    make_partition_ids_by_ddi_scan
)

import dask
from xradio.vis._vis_utils._helpers.msv2_msv3 import ignore_msv2_cols
from xradio.vis._vis_utils._cc_tables.read import describe_ms, read_generic_table, make_freq_attrs, convert_casacore_time
from xradio.vis._vis_utils._cc_tables.read_main_table import read_flat_main_table, read_expanded_main_table, get_baselines, get_utimes_tol, read_main_table_chunks
from xradio.vis._vis_utils._helpers.subtables import subt_rename_ids, add_pointing_to_partition
from xradio.vis._vis_utils._cc_tables.table_query import open_table_ro, open_query
import numpy as np
from casacore import tables
from itertools import cycle
import logging
import time
import xarray as xr

def add_encoding(xds,compressor,chunks=None):
    encoding = {}
    for da_name in list(xds.data_vars):
        if chunks:
            da_chunks = [chunks[dim_name] for dim_name in xds[da_name].dims]
            xds[da_name].encoding = {"compressor": compressor, "chunks": da_chunks}
            #print(xds[da_name].encoding)
        else:
            xds[da_name].encoding = {"compressor": compressor}

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
    #utimes = np.unique(tb_tool.getcol("TIME"))
    
    tvars = {}

    chunks=[len(utimes),len(baselines),freq_cnt, pol_cnt]

    #print("nrows",  len(tb_tool.getcol("TIME")))

    tidxs = np.searchsorted(utimes, tb_tool.getcol("TIME"))
        

    ts_ant1, ts_ant2 = (
            tb_tool.getcol("ANTENNA1"),
            tb_tool.getcol("ANTENNA2"),
        )

    ts_bases = [
        str(ll[0]).zfill(3) + "_" + str(ll[1]).zfill(3)
        for ll in np.hstack([ts_ant1[:, None], ts_ant2[:, None]])
    ]
    bidxs = np.searchsorted(baselines, ts_bases)

    # some antenna 2"s will be out of bounds for this chunk, store rows that are in bounds
    didxs = np.where((bidxs >= 0) & (bidxs < len(baselines)))[0]
    
    baseline_ant1_id, baseline_ant2_id = np.array([tuple(map(int, x.split("_"))) for x in baselines]).T
    return tidxs,bidxs,didxs, baseline_ant1_id,baseline_ant2_id,utimes


def read_col(tb_tool,col: str,
            cshape: Tuple[int],
            tidxs: np.ndarray,
            bidxs: np.ndarray,
            didxs: np.ndarray,):

    start = time.time()
    data = tb_tool.getcol(col)
    #logging.info("Time to get col " + col + "  " + str(time.time()-start))
    
    # full data is the maximum of the data shape and chunk shape dimensions
    start = time.time()
    fulldata = np.full(cshape+data.shape[1:], np.nan, dtype=data.dtype)
    #logging.info("Time to full " + col + "  " + str(time.time()-start))

    start = time.time()
    fulldata[tidxs, bidxs] = data
    #logging.info("Time to reorganize " + col + "  " + str(time.time()-start))
    
    return fulldata
    
    
def convert_and_write_partition(infile: str,
    outfile: str,
    ddi: int = 0,
    scan_state: Union[Tuple[int, int], None] = None,
    ignore_msv2_cols: Union[list, None] = None,
    chunks: Tuple[int, ...] = (400, 200, 100, 2),
    compressor: numcodecs.abc.Codec = numcodecs.Zstd(level=2)
):

    if ignore_msv2_cols is None:
        ignore_msv2_cols = []
        
    file_name = outfile+"/ddi_" + str(ddi)

    taql_where = f"where DATA_DESC_ID = {ddi}"
    if scan_state:
        # get partitions by scan/state
        scan, state = scan_state
        if type(state) == np.ndarray:
            state_ids_or = " OR STATE_ID = ".join(np.char.mod("%d", state))
            taql_where += f" AND (STATE_ID = {state_ids_or})"
            file_name = file_name + "_state_" + str(state).replace(" ","_")[1:-1]
        elif state:
            taql_where += f" AND SCAN_NUMBER = {scan} AND STATE_ID = {state}"
            file_name = file_name + "_scan_" + str(scan) + "_state_" + str(state)
        elif scan:
            # scan can also be None, when partition_scheme="intent"
            # but the STATE table is empty!
            taql_where += f" AND SCAN_NUMBER = {scan}"
            file_name = file_name + "_scan_" + str(scan)
    
    start_with= time.time()
    with open_table_ro(infile) as mtable:
        # one partition, select just the specified ddi (+ scan/subscan)
        taql_main = f"select * from $mtable {taql_where}"
        with open_query(mtable, taql_main) as tb_tool:
            if tb_tool.nrows() == 0:
                tb_tool.close()
                mtable.close()
                return xr.Dataset(), {}, {}

            #logging.info("Setting up table "+ str(time.time()-start_with))

            start= time.time()
            tidxs, bidxs, didxs, baseline_ant1_id, baseline_ant2_id, utime = calc_indx_for_row_split(tb_tool, taql_where)
            time_baseline_shape = (len(utime),len(baseline_ant1_id))
            #logging.info("Calc indx for row split "+ str(time.time()-start))

            start = time.time()
            xds = xr.Dataset()
            col_to_data_variable_names = {"DATA":"VIS","CORRECTED_DATA":"VIS_CORRECTED","WEIGHT_SPECTRUM":"WEIGHT","WEIGHT":"WEIGHT","FLAG":"FLAG","UVW":"UVW"}
            col_dims = {"DATA":("time","baseline","freq","pol"),"CORRECTED_DATA":("time","baseline","freq","pol"),"WEIGHT_SPECTRUM":("time","baseline","freq","pol"),"WEIGHT":("time","baseline","pol"),"FLAG":("time","baseline","freq","pol"),"UVW":("time","baseline","uvw_dim")}
            col_to_coord_names = {"TIME":"time","ANTENNA1":"baseline_ant1_id","ANTENNA2":"baseline_ant2_id"}
            coords_dim_select = {"TIME":np.s_[:,0:1],"ANTENNA1":np.s_[0:1,:],"ANTENNA2":np.s_[0:1,:]}
            check_variables = {}

            col_names = tb_tool.colnames()
            coords = {"time":convert_casacore_time(utime),"baseline_ant1_id":baseline_ant1_id, "baseline_ant2_id":baseline_ant2_id}
            #Create Data Variables
            not_a_problem = True
            #logging.info("Setup xds "+ str(time.time()-start))
            
            for col in col_names:
                if col in col_to_data_variable_names:
                    if (col == "WEIGHT") and ("WEIGHT_SPECTRUM" not in col_names):
                        continue
                    try:
                        start = time.time()
                        xds[col_to_data_variable_names[col]] = xr.DataArray(read_col(tb_tool,col,time_baseline_shape,tidxs,bidxs,didxs),dims=col_dims[col])
                        #logging.info("Time to read column " + str(col) + " : " + str(time.time()-start))

                        if col == "UVW": #Just for testing
                            xds[col_to_data_variable_names[col]].attrs["units"] = "m"
                            xds[col_to_data_variable_names[col]].attrs["measure"] = {"type": "uvw", "ref_frame": "ITRF"}
                            xds[col_to_data_variable_names[col]].attrs["long_name"] = "uvw"
                            xds[col_to_data_variable_names[col]].attrs["description"] = "uvw coordinates."
                    except:
                        continue
                        #logging.debug("Could not load column",col)
                        
            start = time.time()

            spw_xds = read_generic_table(
                infile,
                "SPECTRAL_WINDOW",
                rename_ids=subt_rename_ids["SPECTRAL_WINDOW"],
                )

            pol_xds = read_generic_table(
                infile,
                "POLARIZATION",
                rename_ids=subt_rename_ids["POLARIZATION"],
                )

            coords["freq"] = spw_xds["chan_freq"][0,:].data
            
            xds = xds.assign_coords(coords)


            field_xds = read_generic_table(
                infile,
                "FIELD",
                rename_ids=subt_rename_ids["FIELD"],
            )


            field_info = {"name": "NGC4038 - Antennae North", "code": "none",
                          "time": field_xds["time"].data[0], "num_poly": 0,
                          "delay_dir": list(field_xds["delay_dir"].data[0,0,:]),
                          "phase_dir": list(field_xds["phase_dir"].data[0,0,:]),
                          "reference_dir": list(field_xds["reference_dir"].data[0,0,:])}
            xds.attrs["field_info"] = field_info
      
            add_encoding(xds,compressor=compressor,chunks=xds.dims)
            xds.to_zarr(store=file_name, mode="w")
            #logging.info(" To disk time " + str(time.time()-start))

            ant_xds = read_generic_table(
                infile,
                "ANTENNA",
                rename_ids=subt_rename_ids["ANTENNA"],
            )

            ant_df = ant_xds.to_dataframe()
            ant_df.to_parquet(path=file_name+"_ANTENNA.pq")
            
    logging.info("write_partition " + str(time.time()-start_with) )


def convert_msv2_to_processing_set(
    infile: str,
    outfile: str,
    partition_scheme: str, # intent_field, subscan
    chunks_on_disk: Union[Dict, None] = None,
    compressor: numcodecs.abc.Codec = numcodecs.Zstd(level=2),
    parallel: bool = False
):
    """

    """
    spw_xds = read_generic_table(
        infile,
        "SPECTRAL_WINDOW",
        rename_ids=subt_rename_ids["SPECTRAL_WINDOW"],
    )
    ddi_xds = read_generic_table(infile, "DATA_DESCRIPTION")
    
    if partition_scheme == "intent":
        spw_names_by_ddi = make_spw_names_by_ddi(ddi_xds, spw_xds)
        (
            data_desc_id,
            scan_number,
            state_id,
            distinct_intents,
        ) = make_partition_ids_by_ddi_intent(infile, spw_names_by_ddi)
    else:
        do_subscans = partition_scheme == "scan/subscan"
        data_desc_id, scan_number, state_id = make_partition_ids_by_ddi_scan(
            infile, do_subscans
        )
 
    ant_xds = read_generic_table(
        infile, "ANTENNA", rename_ids=subt_rename_ids["ANTENNA"]
    )
    pol_xds = read_generic_table(
        infile, "POLARIZATION", rename_ids=subt_rename_ids["POLARIZATION"]
    )

    delayed_list = []
    partitions = {}
    cnt = 0
    
    for ddi, scan, state in zip(data_desc_id, scan_number, state_id):
        logging.info("DDI " + str(ddi) + ", SCAN" + str(scan) + ", STATE " + str(state))
        
        if partition_scheme == "intent":
            intent = distinct_intents[cnt]
            cnt += 1
            
        if partition_scheme == "scan":
            scan_state = (scan, None)
        else:
            scan_state = (scan, state)

        if parallel:
            delayed_list.append(dask.delayed(convert_and_write_partition)(infile,outfile,ddi,scan_state=scan_state,ignore_msv2_cols=ignore_msv2_cols,compressor=compressor))
        else:
            convert_and_write_partition(infile,outfile,ddi,scan_state=scan_state,ignore_msv2_cols=ignore_msv2_cols,compressor=compressor)
        
    if parallel:
        dask.compute(delayed_list)

        

