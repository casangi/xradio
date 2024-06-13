from xradio.vis._vis_utils._ms.msv2_to_msv4_meta import column_description_casacore_to_msv4_measure
from xradio.vis._vis_utils._ms.subtables import subt_rename_ids
from xradio.vis._vis_utils._ms._tables.read import read_generic_table
import graphviper.utils.logger as logger
import numpy as np
import xarray as xr

def create_field_and_source_xds(in_file, field_id, spectral_window_id):
    


    field_and_source_xds = xr.Dataset()
    #field_directions, field_info = create_field_info(field_and_source_xds, in_file, field_id)
    field_and_source_xds, ephemeris_path, ephemeris_table_name = create_field_info_and_check_ephemeris(field_and_source_xds, in_file, field_id)
    source_id = field_and_source_xds.attrs['source_id']

    if ephemeris_path is not None: 
        field_and_source_xds = extract_ephemeris_info(field_and_source_xds, ephemeris_path, ephemeris_table_name)
        field_and_source_xds = extract_source_info(field_and_source_xds, in_file, True, source_id, spectral_window_id)
    else:  
        field_and_source_xds = extract_source_info(field_and_source_xds, in_file, False, source_id, spectral_window_id)
        
    return field_and_source_xds
   

def extract_ephemeris_info(xds, path, table_name):

    #The JPL-Horizons ephemris table implmenation in CASA does not follow the standard way of defining measures. Consequently a lot of hardcoding is needed to extract the information.
    # https://casadocs.readthedocs.io/en/latest/notebooks/external-data.html
    
    ephemeris_xds = read_generic_table(
                            path,
                            table_name,
                            timecols=['MJD']
                        )
    
    assert len(ephemeris_xds.ephemeris_id) == 1, 'Non standard ephemeris table.'
    ephemeris_xds = ephemeris_xds.isel(ephemeris_id=0)

    ephemeris_meta = ephemeris_xds.attrs["other"]["msv2"]["ctds_attrs"]
    ephemris_column_description = ephemeris_xds.attrs["other"]["msv2"]["ctds_attrs"][
        "column_descriptions"
    ]

    assert ephemeris_meta['obsloc'] == 'GEOCENTRIC', 'Only geocentric ephemeris are supported.'
    sky_coord_frame = ephemeris_meta['posrefsys'].replace("ICRF/","")
    
    xds['SOURCE_POSITION'] = xr.DataArray(np.column_stack((ephemeris_xds['ra'].data, ephemeris_xds['dec'].data, ephemeris_xds['rho'].data)), dims=['time','sky_pos_label'])
    sky_coord_units = [ephemris_column_description["RA"]['keywords']['UNIT'],ephemris_column_description["DEC"]['keywords']['UNIT'],ephemris_column_description["Rho"]['keywords']['UNIT']]
    xds['SOURCE_POSITION'].attrs.update({'type':'sky_coord', 'frame':sky_coord_frame, 'units':sky_coord_units})
    
    ####
    # xds['SOURCE_DIRECTION'] = xr.DataArray(np.column_stack((ephemeris_xds['ra'].data, ephemeris_xds['dec'].data)), dims=['time','direction_label'])
    # sky_coord_units = [ephemris_column_description["RA"]['keywords']['UNIT'],ephemris_column_description["DEC"]['keywords']['UNIT']]
    # xds['SOURCE_DIRECTION'].attrs.update({'type':'sky_coord', 'frame':sky_coord_frame, 'units':sky_coord_units})
    
    # xds['SOURCE_DISTANCE'] = xr.DataArray(ephemeris_xds['rho'].data, dims=['time'])
    # distance_units = [ephemris_column_description["Rho"]['keywords']['UNIT']]
    # xds['SOURCE_DISTANCE'].attrs.update({'type':'quantity', 'units':distance_units})
    # #######
    
    xds['SOURCE_RADIAL_VELOCITY'] = xr.DataArray(ephemeris_xds['radvel'].data, dims=['time'])
    xds['SOURCE_RADIAL_VELOCITY'].attrs.update({'type':'quantity','units':[ephemris_column_description["RadVel"]['keywords']['UNIT']]})
    
    observation_position = [ephemeris_meta["GeoLong"],ephemeris_meta["GeoLat"],ephemeris_meta["GeoDist"]]
    xds['OBSERVATION_POSITION'] = xr.DataArray(observation_position, dims=['spherical_pos_label'])
    xds['OBSERVATION_POSITION'].attrs.update({'type':'earth_location', 'units':['deg','deg','m'], 'data': observation_position, 'ellipsoid':'WGS84','coordinate_system':ephemeris_meta['obsloc'].lower()}) #I think the units are ['deg','deg','m'] and 'WGS84'.
    

    xds = xds.assign_coords({'spherical_pos_label' : ['lon','lat','dist'],'time':ephemeris_xds['time'].data, 'sky_pos_label':['ra','dec','dist']})
    xds['time'].attrs.update({'type':'time', 'units':['s'], 'scale':'UTC', 'format':'UNIX'}) 
    
    return xds

def extract_source_info(xds, path, is_ephemeris, source_id, spectral_window_id):
    
    if source_id == -1:
        logger.warning(f"Source_id is -1. No source information will be included in the field_and_source_xds.")
        return xds
    
    source_xds = read_generic_table(
                            path,
                            'SOURCE',
                            ignore=['SOURCE_MODEL'], #Trying to read SOURCE_MODEL causes an error.
                            taql_where=f'where SOURCE_ID = {source_id} AND SPECTRAL_WINDOW_ID = {spectral_window_id}'
                        )
    
    if len(source_xds.data_vars) == 0: #The source xds is empty.
        logger.warning(f"SOURCE table empty for source_id {source_id} and spectral_window_id {spectral_window_id}.")
        return xds

    assert len(source_xds.source_id) == 1, 'Can only process source table with a single source_id and spectral_window_id for a given MSv4 partition.'
    assert len(source_xds.spectral_window_id) == 1, 'Can only process source table with a single source_id and spectral_window_id for a given MSv4 partition.' 
    assert len(source_xds.time) == 1, 'Can only process source table with a single time entry for a source_id and spectral_window_id.'
    source_xds = source_xds.isel(time=0,source_id=0,spectral_window_id=0)

    xds.attrs['source_name'] = source_xds['name'].data
    xds.attrs['code'] = source_xds['code'].data
    source_column_description = source_xds.attrs["other"]["msv2"]["ctds_attrs"]["column_descriptions"]
    
    if not is_ephemeris:
        msv4_measure = column_description_casacore_to_msv4_measure(source_column_description["DIRECTION"])
        xds['SOURCE_DIRECTION'] = xr.DataArray(source_xds['direction'].data,dims=['sky_dir_label'])
        xds['SOURCE_DIRECTION'].attrs.update(msv4_measure)
        
        msv4_measure = column_description_casacore_to_msv4_measure(source_column_description["PROPER_MOTION"])
        xds['SOURCE_PROPER_MOTION'] = xr.DataArray(source_xds['proper_motion'].data,dims=['sky_dir_label'])
        xds['SOURCE_PROPER_MOTION'].attrs.update(msv4_measure)
        
 
    #['DIRECTION', 'PROPER_MOTION', 'CALIBRATION_GROUP', 'CODE', 'INTERVAL', 'NAME', 'NUM_LINES', 'SOURCE_ID', 'SPECTRAL_WINDOW_ID', 'TIME', 'POSITION', 'TRANSITION', 'REST_FREQUENCY', 'SYSVEL']
    if source_xds['num_lines'] > 0:
        coords = {'line_name':source_xds['transition'].data}
        xds = xds.assign_coords(coords)
        
        optional_data_variables = {'rest_frequency':'LINE_REST_FREQUENCY','sysvel':'LINE_SYSTEMIC_VELOCITY'}
        for generic_name, msv4_name in optional_data_variables.items():
            if generic_name in source_xds:
                msv4_measure = column_description_casacore_to_msv4_measure(source_column_description[generic_name.upper()])
                xds[msv4_name] = xr.DataArray(source_xds[generic_name].data,dims=['line_name'])
                xds[msv4_name].attrs.update(msv4_measure)

    #Need to add doppler info if present. Add check.
    try:
        doppler_xds = read_generic_table(
                        path,
                        'DOPPLER',
                    )
        assert False, 'Doppler table present. Please open an issue on https://github.com/casangi/xradio/issues so that we can addd support for this.'
    except:
        pass

    return xds
    
    
def create_field_info_and_check_ephemeris(field_and_source_xds, in_file, field_id):
    field_xds = read_generic_table(
        in_file,
        "FIELD",
        rename_ids=subt_rename_ids["FIELD"],
    ).sel(field_id=field_id)
    # https://stackoverflow.com/questions/53195684/how-to-navigate-a-dict-by-list-of-keys
    
    assert len(field_xds.poly_id) == 1, "Polynomial field positions not supported."
    field_xds = field_xds.isel(poly_id=0)
    
    field_and_source_xds.attrs.update({"field_name": str(field_xds["name"].data),
        "field_code": str(field_xds["code"].data),
        "field_id": field_id,
        "source_id": int(field_xds["source_id"].data),})
    
    ephemeris_table_name = None
    ephemeris_path =  None
    is_ephemeris = False
    
    #Need to check if ephemeris_id is present and if epehemeris table is present.
    if "ephemeris_id" in field_xds:
        ephemeris_id = int(field_xds["ephemeris_id"].data)
        if ephemeris_id > -1:
            import os
            files = os.listdir(os.path.join(in_file,'FIELD'))
            ephemeris_table_name_start = 'EPHEM' + str(ephemeris_id)
        
            ephemeris_name_table_index = [i for i in range(len(files)) if ephemeris_table_name_start in files[i]]
            assert len(ephemeris_name_table_index) == 1, "More than one ephemeris table which starts with " + ephemeris_table_name_start

            if len(ephemeris_name_table_index) > 0: #Are there any ephemeris tables. 
                is_ephemeris = True            
                e_index = ephemeris_name_table_index[0]
                ephemeris_path = os.path.join(in_file, 'FIELD')
                ephemeris_table_name = files[e_index]
            else:
                logger.warning(f"Could not find ephemeris table for field_id {field_id}. Ephemeris information will not be included in the field_and_source_xds.")
    
    if is_ephemeris:
        field_data_variables = {'delay_dir':'FIELD_DELAY_CENTER_OFFSET','phase_dir':'FIELD_PHASE_CENTER_OFFSET','reference_dir':'FIELD_REFERENCE_CENTER_OFFSET'}
        field_measures_type = 'sky_coord_offset'
        field_and_source_xds.attrs['field_and_source_xds_type'] = 'ephemeris'    
    else:
        field_data_variables = {'delay_dir':'FIELD_DELAY_CENTER','phase_dir':'FIELD_PHASE_CENTER','reference_dir':'FIELD_REFERENCE_CENTER'}
        field_measures_type = 'sky_coord'
        field_and_source_xds.attrs['field_and_source_xds_type'] = 'standard'
    
    
    coords = {}
    coords['sky_dir_label'] = ['ra','dec'] 
    field_column_description = field_xds.attrs["other"]["msv2"]["ctds_attrs"][
        "column_descriptions"
    ] #Keys are ['DELAY_DIR', 'PHASE_DIR', 'REFERENCE_DIR', 'CODE', 'FLAG_ROW', 'NAME', 'NUM_POLY', 'SOURCE_ID', 'TIME']
    
    
    for generic_name, msv4_name in field_data_variables.items():
        msv4_measure = column_description_casacore_to_msv4_measure(
            field_column_description[generic_name.upper()],
            ref_code=getattr(field_xds.get("delaydir_ref"), "data", None),
        )
        
        field_and_source_xds[msv4_name] = xr.DataArray.from_dict({
            "dims": "sky_dir_label",
            "data": list(field_xds[generic_name].data),
            "attrs": msv4_measure,
        })
    
        field_and_source_xds[msv4_name].attrs['type'] = field_measures_type
 
    coords = {}
    coords['sky_dir_label'] = ['ra','dec'] 
    field_and_source_xds = field_and_source_xds.assign_coords(coords)   
    return field_and_source_xds, ephemeris_path, ephemeris_table_name
    