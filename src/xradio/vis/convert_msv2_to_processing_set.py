#from numcodecs.zstd import Zstd
import numcodecs
from typing import Dict, List, Tuple, Union
import itertools
import numbers
from xradio.vis._vis_utils._ms.partitions import (
    finalize_partitions,
    read_ms_ddi_partitions,
    read_ms_scan_subscan_partitions,
    make_spw_names_by_ddi,
    make_partition_ids_by_ddi_intent,
    make_partition_ids_by_ddi_scan
)

import dask
from xradio.vis._vis_utils._ms.descr import describe_ms
from xradio.vis._vis_utils._ms.msv2_msv3 import ignore_msv2_cols
from xradio.vis._vis_utils._ms._tables.read import read_generic_table, make_freq_attrs, convert_casacore_time
from xradio.vis._vis_utils._ms._tables.read_main_table import read_flat_main_table, read_expanded_main_table, get_baselines, get_utimes_tol, read_main_table_chunks
from xradio.vis._vis_utils._ms.subtables import subt_rename_ids, add_pointing_to_partition
from xradio.vis._vis_utils._ms._tables.table_query import open_table_ro, open_query
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
    return tidxs,bidxs,didxs, baseline_ant1_id,baseline_ant2_id,convert_casacore_time(utimes,False)

def _check_single_field(tb_tool):
    field_id = np.unique(tb_tool.getcol("FIELD_ID"))
    #print(np.unique(field_id))
    assert len(field_id) == 1, "More than one field present."
    return field_id[0]
    
def _check_interval_consistent(tb_tool):
    interval = np.unique(tb_tool.getcol("INTERVAL"))
    assert len(interval) == 1, "Interval is not consistent."
    return interval[0]

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
    
def create_attribute_metadata(col,tb_tool):
    
    attrs_metadata = {}
    
    if col == "UVW":
        attrs_metadata["units"] = "m"
        attrs_metadata["measure"] = {"type": "uvw", "ref_frame": "ITRF"}
        attrs_metadata["long_name"] = "uvw"
        attrs_metadata["description"] = "uvw coordinates."
    
    return attrs_metadata
    
def convert_and_write_partition(infile: str,
    outfile: str,
    intent: str,
    ddi: int = 0,
    state_ids = None,
    field_id: int = None,
    ignore_msv2_cols: Union[list, None] = None,
    chunks_on_disk: Tuple[int, ...] = (400, 200, 100, 2),
    compressor: numcodecs.abc.Codec = numcodecs.Zstd(level=2),
    storage_backend = "zarr",
    overwrite: bool = False
):

    if ignore_msv2_cols is None:
        ignore_msv2_cols = []
        
    file_name = outfile+"/ddi_" + str(ddi) + "_intent_" + intent
    taql_where = f"where (DATA_DESC_ID = {ddi})"
    
#    if isinstance(state_ids,numbers.Integral):
#        taql_where += f" AND (STATE_ID = {state_ids})"
#        file_name = file_name + "_state_id_" + str(state_ids)
#    else:
#        state_ids_or = " OR STATE_ID = ".join(np.char.mod("%d", state_ids))
#        taql_where += f" AND (STATE_ID = {state_ids_or})"
#        file_name = file_name + "_state_id_" + str(state_ids).replace(", ","_")[1:-1]

    if field_id is not None:
        taql_where += f" AND (FIELD_ID = {field_id})"
        file_name = file_name + "_field_id_" + str(field_id)
    
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
            col_dims = {"DATA":("time","bl_id","freq","pol"),"CORRECTED_DATA":("time","bl_id","freq","pol"),"WEIGHT_SPECTRUM":("time","bl_id","freq","pol"),"WEIGHT":("time","bl_id","pol"),"FLAG":("time","bl_id","freq","pol"),"UVW":("time","bl_id","uvw_label")}
            col_to_coord_names = {"TIME":"time","ANTENNA1":"baseline_ant1_id","ANTENNA2":"baseline_ant2_id"}
            coords_dim_select = {"TIME":np.s_[:,0:1],"ANTENNA1":np.s_[0:1,:],"ANTENNA2":np.s_[0:1,:]}
            check_variables = {}

            col_names = tb_tool.colnames()
            coords = {"time":utime,"bl_ant1_id":("bl_id",baseline_ant1_id), "bl_ant2_id":("bl_id",baseline_ant2_id), "uvw_label":["u","v","w"],"bl_id":np.arange(len(baseline_ant1_id))}
            
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
                    except:
                        continue
                        #logging.debug("Could not load column",col)
                        
                        #xds[col_to_data_variable_names[col]].attrs.update(create_attribute_metadata(col,tb_tool))
     
            field_id = _check_single_field(tb_tool)
            interval = _check_interval_consistent(tb_tool)
                        
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
            
            #Add time attributes
            xds.time.attrs["units"] = "s"
            xds.time.attrs["time_scale"] = "utc"
            xds.time.attrs["integration_time"] = interval

            field_xds = read_generic_table(
                infile,
                "FIELD",
                rename_ids=subt_rename_ids["FIELD"],
            )
            
            #print(field_xds)

            #field_id = 2
            field_info = {"name": field_xds["name"].data[field_id], "code": field_xds["code"].data[field_id],
                          "time": field_xds["time"].data[field_id], "num_poly": 0,
                          "delay_dir": list(field_xds["delay_dir"].data[field_id,0,:]),
                          "phase_dir": list(field_xds["phase_dir"].data[field_id,0,:]),
                          "reference_dir": list(field_xds["reference_dir"].data[field_id,0,:])}
            #xds.attrs["field_info"] = field_info
            
            logging.info(file_name)
            if overwrite:
                mode='w'
            else:
                 mode='w-'
      
            add_encoding(xds,compressor=compressor,chunks=chunks_on_disk)
            
            ant_xds = read_generic_table(
                infile,
                "ANTENNA",
                rename_ids=subt_rename_ids["ANTENNA"],
            )
            del ant_xds.attrs['other']
            

            if storage_backend=="zarr":
                xds.to_zarr(store=file_name+"/MAIN", mode=mode)
                ant_xds.to_zarr(store=file_name+"/ANTENNA", mode=mode)
            elif storage_backend=="netcdf":
                #xds.to_netcdf(path=file_name+"/MAIN", mode=mode)
                print(ant_xds)
                #ant_xds.offset.attrs["measure"] = [ant_xds.offset.attrs["measure"]]
                #ant_xds.position.attrs["measure"] = [ant_xds.position.attrs["measure"]]
                import os
                
                os.system("mkdir /Users/jsteeb/Library/CloudStorage/Dropbox/xradio/doc/Antennae_North.cal.vis.zarr")
                #os.system("mkdir " + file_name+"_ANTENNA")
                ant_xds.to_netcdf(path=file_name+"_ANTENNA", mode=mode, engine="netcdf4")
            #logging.info(" To disk time " + str(time.time()-start))


            
            


            
    #logging.info("write_partition " + str(time.time()-start_with) )

def get_unqiue_intents(infile):
    state_xds = read_generic_table(
        infile,
        "STATE",
        rename_ids=subt_rename_ids["STATE"],
    )

    obs_mode_dict = {}
    for i,obs_mode in enumerate(state_xds.obs_mode.values):
        if obs_mode in obs_mode_dict:
            obs_mode_dict[obs_mode].append(i)
        else:
            obs_mode_dict[obs_mode] = [i]
            
    return list(obs_mode_dict.keys()), obs_mode_dict.values()

def enumerated_product(*args):
    yield from zip(itertools.product(*(range(len(x)) for x in args)), itertools.product(*args))

def convert_msv2_to_processing_set(
    infile: str,
    outfile: str,
    partition_scheme: str, # intent_field, subscan
    chunks_on_disk: Union[Dict, None] = None,
    compressor: numcodecs.abc.Codec = numcodecs.Zstd(level=2),
    parallel: bool = False,
    storage_backend="zarr",
    overwrite: bool = False
):
    """

    """
    spw_xds = read_generic_table(
        infile,
        "SPECTRAL_WINDOW",
        rename_ids=subt_rename_ids["SPECTRAL_WINDOW"],
    )
    

    ddi_xds = read_generic_table(infile, "DATA_DESCRIPTION")
    data_desc_ids = np.arange(read_generic_table(infile, "DATA_DESCRIPTION").dims['row'])
    
    if partition_scheme == "ddi_intent_field":
        unique_intents, state_ids = get_unqiue_intents(infile)
        field_ids = np.arange(read_generic_table(infile, "FIELD").dims['row'])
    elif partition_scheme == "ddi_state":
        state_xds = read_generic_table(infile, "STATE")
        state_ids = np.arange(state_xds.dims['row'])
        intents = state_xds.obs_mode.values
        #print(state_xds, intents)
        #field_ids = [None]
        field_ids = np.arange(read_generic_table(infile, "FIELD").dims['row'])

    delayed_list = []
    partitions = {}
    cnt = 0
    #for ddi, state, field in zip(data_desc_id, state_id, field_ids):
    #    logging.info("DDI " + str(ddi) + ", STATE " + str(state) + ", FIELD " + str(field))
      
    #for ddi, state, field in itertools.product(data_desc_id, state_id, field_ids):
    #    logging.info("DDI " + str(ddi) + ", STATE " + str(state) + ", FIELD " + str(field))
    
    for idx, pair in enumerated_product(data_desc_ids, state_ids, field_ids):
        ddi, state_id, field_id = pair
        #logging.info("DDI " + str(ddi) + ", STATE " + str(state_id) + ", FIELD " + str(field_id))
        
        if partition_scheme == "ddi_intent_field":
            intent = unique_intents[idx[1]]
        else:
            intent = intents[idx[1]] + "_" + str(state_id)
        
        if parallel:
            delayed_list.append(dask.delayed(convert_and_write_partition)(infile,outfile,intent, ddi, state_id, field_id,ignore_msv2_cols=ignore_msv2_cols,chunks_on_disk=chunks_on_disk,compressor=compressor,overwrite=overwrite))
        else:
            convert_and_write_partition(infile,outfile,intent, ddi, state_id, field_id,ignore_msv2_cols=ignore_msv2_cols,chunks_on_disk=chunks_on_disk,compressor=compressor,storage_backend=storage_backend,overwrite=overwrite)
     
        
    if parallel:
        dask.compute(delayed_list)

        

