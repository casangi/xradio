import astropy
from astropy.coordinates import Angle, SkyCoord
from casacore import tables
import copy
import numpy as np
import os
import xarray as xr
from .common import (
    __active_mask, __native_types, __object_name, __pointing_center
)
from ..common import __doppler_types


# TODO move this to a common file to be shared
def __compute_ref_pix(xds:xr.Dataset, direction:dict) -> np.ndarray:
    # TODO more general coordinates
    long = xds.right_ascension
    lat = xds.declination
    ra_crval = long.attrs['wcs']['crval']
    dec_crval = lat.attrs['wcs']['crval']
    long_close = np.where(np.isclose(long, ra_crval))
    lat_close = np.where(np.isclose(lat, dec_crval))
    if long_close and lat_close:
        long_list = [ (i,j) for i,j in zip(long_close[0], long_close[1]) ]
        lat_list = [ (i,j) for i,j in zip(lat_close[0], lat_close[1]) ]
        common_indices = [ t for t in long_list if t in lat_list ]
        if len(common_indices) == 1:
            return np.array(common_indices[0])
    cdelt = max(
        abs(long.attrs['wcs']['cdelt']),
        abs(lat.attrs['wcs']['cdelt'])
    )

    # this creates an image of mostly NaNs. The few pixels with values are
    # close to the reference pixel
    ra_diff = long - ra_crval
    dec_diff = lat - dec_crval
    # this returns a 2-tuple of indices where the values in aa are not NaN
    indices_close = np.where(
        ra_diff*ra_diff + dec_diff*dec_diff < 2*cdelt*cdelt
    )
    # this determines the closest pixel to the reference pixel
    closest = 5e10
    pix = []
    for i,j in zip(indices_close[0], indices_close[1]):
        dra = long[i,j] - ra_crval
        ddec = lat[i,j] - dec_crval
        if dra*dra + ddec*ddec < closest:
            pix = [i, j]
    xds_dir = xds.attrs['direction']
    # get the actual ref pix
    proj = direction['projection']
    wcs_dict = {}
    wcs_dict[f'CTYPE1'] = f'RA---{proj}'
    wcs_dict[f'NAXIS1'] = long.shape[0]
    wcs_dict[f'CUNIT1'] = long.attrs['unit']
    # FITS arrays are 1-based
    wcs_dict[f'CRPIX1'] = pix[0] + 1
    wcs_dict[f'CRVAL1'] = long[pix[0], pix[1]].item(0)
    wcs_dict[f'CDELT1'] = long.attrs['wcs']['cdelt']
    wcs_dict[f'CTYPE2'] = f'DEC--{proj}'
    wcs_dict[f'NAXIS2'] = lat.shape[1]
    wcs_dict[f'CUNIT2'] = lat.attrs['unit']
    # FITS arrays are 1-based
    wcs_dict[f'CRPIX2'] = pix[1] + 1
    wcs_dict[f'CRVAL2'] = lat[pix[0], pix[1]].item(0)
    wcs_dict[f'CDELT2'] = lat.attrs['wcs']['cdelt']
    w = astropy.wcs.WCS(wcs_dict)
    x, y = np.indices(w.pixel_shape)
    sky = SkyCoord(
        ra_crval, dec_crval,
        frame=xds.attrs['direction']['system'].lower(),
        equinox=xds.attrs['direction']['equinox'],
        unit=long.attrs['unit']
    )
    return w.world_to_pixel(sky)


def __compute_direction_dict(xds: xr.Dataset) -> dict:
    """
    Given xds metadata, compute the direction dict that is valid
    for a CASA image coordinate system
    """
    direction = {}
    xds_dir = xds.attrs['direction']
    direction['system'] = xds_dir['equinox']
    direction['projection'] = xds_dir['projection']
    direction['projection_parameters'] = xds_dir['projection_parameters']
    long = xds.right_ascension
    lat = xds.declination
    direction['units'] = np.array(
        [long.attrs['unit'], lat.attrs['unit']], dtype='<U16'
    )
    direction['crval'] = np.array([
        long.attrs['wcs']['crval'], lat.attrs['wcs']['crval']
    ])
    direction['cdelt'] = np.array([
        long.attrs['wcs']['cdelt'], lat.attrs['wcs']['cdelt']
    ])
    """
    # get the actual ref pix
    proj = direction['projection']
    wcs_dict = {}
    wcs_dict[f'CTYPE1'] = f'RA---{proj}'
    wcs_dict[f'NAXIS1'] = long.shape[0]
    wcs_dict[f'CUNIT1'] = long.attrs['unit']
    # FITS arrays are 1-based
    wcs_dict[f'CRPIX1'] = 1
    wcs_dict[f'CRVAL1'] = long[0][0].item(0)
    wcs_dict[f'CDELT1'] = long.attrs['wcs']['cdelt']
    wcs_dict[f'CTYPE2'] = f'DEC--{proj}'
    wcs_dict[f'NAXIS2'] = lat.shape[1]
    wcs_dict[f'CUNIT2'] = lat.attrs['unit']
    # FITS arrays are 1-based
    wcs_dict[f'CRPIX2'] = 1
    wcs_dict[f'CRVAL2'] = lat[0][0].item(0)
    wcs_dict[f'CDELT2'] = lat.attrs['wcs']['cdelt']
    print('*** wcs_dict', wcs_dict)
    w = astropy.wcs.WCS(wcs_dict)
    x, y = np.indices(w.pixel_shape)
    sky = SkyCoord(direction[
        'crval'][0], direction['crval'][1],
        frame=xds.attrs['direction']['system'].lower(),
        equinox=xds.attrs['direction']['equinox'],
        unit=long.attrs['unit']
    )
    crpix = w.world_to_pixel(sky)
    """
    crpix = __compute_ref_pix(xds, direction)
    direction['crpix'] = np.array([crpix[0], crpix[1]])
    direction['pc'] = xds_dir['pc']
    direction['axes'] = ['Right Ascension', 'Declination']
    direction['conversionSystem'] = direction['system']
    for s in ['longpole', 'latpole']:
        direction[s] = Angle(
            str(xds_dir[s]['value']) + xds_dir[s]['unit']
        ).deg
    return direction


def __compute_spectral_dict(
    xds: xr.Dataset, direction: dict, obsdate: dict, tel_pos: dict,
) -> dict:
    """
    Given xds metadata, compute the spectral dict that is valid
    for a CASA image coordinate system
    """
    spec = {}
    spec_conv = copy.deepcopy(xds.freq.attrs['conversion'])
    for k in ('direction', 'epoch', 'position'):
        spec_conv[k]['type'] = k
    spec_conv['direction']['refer'] = spec_conv['direction']['system']
    del spec_conv['direction']['system']
    if (
        spec_conv['direction']['refer'] == 'FK5'
        and spec_conv['direction']['equinox'] == 'J2000'
    ):
        spec_conv['direction']['refer'] = 'J2000'
    del spec_conv['direction']['equinox']
    spec['conversion'] = spec_conv
    spec['formatUnit'] = ''
    spec['name'] = 'Frequency'
    spec['nativeType'] = __native_types.index(xds.freq.attrs['native_type'])
    spec['restfreq'] = xds.freq.attrs['restfreq']
    spec['restfreqs'] = copy.deepcopy(xds.freq.attrs['restfreqs'])
    spec['system'] = xds.freq.attrs['system']
    spec['unit'] = xds.freq.attrs['unit']
    spec['velType'] = __doppler_types.index(xds.vel.attrs['doppler_type'])
    spec['velUnit'] = xds.vel.attrs['unit']
    spec['version'] = 2
    spec['waveUnit'] = xds.freq.attrs['wave_unit']
    spec_wcs = copy.deepcopy(xds.freq.attrs['wcs'])
    spec_wcs['ctype'] = 'FREQ'
    spec_wcs['pc'] = 1.0
    spec_wcs['crpix'] = (spec_wcs['crval'] - xds.freq.values[0])/spec_wcs['cdelt']
    spec['wcs'] = spec_wcs
    return spec


def __coord_dict_from_xds(xds: xr.Dataset) -> dict:
    coord = {}
    coord['telescope'] = xds.attrs['telescope']['name']
    coord['observer'] = xds.attrs['observer']
    obsdate = {}
    obsdate['refer'] = xds.coords['time'].attrs['time_scale']
    obsdate['type'] = 'epoch'
    obsdate['m0'] = {}
    obsdate['m0']['unit'] = xds.coords['time'].attrs['unit']
    obsdate['m0']['value'] = xds.coords['time'].values[0]
    #obsdate['format'] = xds.time.attrs['format']
    coord['obsdate'] = obsdate
    coord['pointingcenter'] = xds.attrs[__pointing_center].copy()
    coord['telescopeposition'] = xds.attrs['telescope']['position'].copy()
    coord['direction0'] = __compute_direction_dict(xds)
    coord['stokes1'] = {
        'axes': np.array(['Stokes'], dtype='<U16'), 'cdelt': np.array([1.]),
        'crpix': np.array([0.]), 'crval': np.array([1.]), 'pc': np.array([[1.]]),
        'stokes': np.array(xds.pol.values, dtype='<U16')
    }
    coord['spectral2'] = __compute_spectral_dict(
        xds, coord['direction0'], coord['obsdate'], coord['telescopeposition']
    )
    coord['pixelmap0'] = np.array([0, 1])
    coord['pixelmap1'] = np.array([2])
    coord['pixelmap2'] = np.array([3])
    coord['pixelreplace0'] = np.array([0., 0.])
    coord['pixelreplace1'] = np.array([0.])
    coord['pixelreplace2'] = np.array([0.])
    coord['worldmap0'] = np.array([0, 1], dtype=np.int32)
    coord['worldmap1'] = np.array([2], dtype=np.int32)
    coord['worldmap2'] = np.array([3], dtype=np.int32)
    # coord['worldreplace0'] = coord['direction0']['crval']
    # this probbably needs some verification
    coord['worldreplace0'] = [0.0, 0.0]
    coord['worldreplace1'] = np.array(coord['stokes1']['crval'])
    coord['worldreplace2'] = np.array([xds.freq.attrs['wcs']['crval']])
    return coord


def __history_from_xds(xds: xr.Dataset, image: str) -> None:
    tb = tables.table(
        os.sep.join([image, 'logtable']), readonly=False,
        lockoptions={'option': 'permanentwait'}, ack=False
    )
    nrows = len(xds.history.row) if 'row' in xds.data_vars else 0
    if nrows > 0:
        # TODO need to implement nrows == 0 case
        tb.addrows(nrows + 1)
        for c in ['TIME', 'PRIORITY', 'MESSAGE', 'LOCATION', 'OBJECT_ID']:
            vals = xds.history[c].values
            if c == 'TIME':
                k = time.time() + 40587*86400
            elif c == 'PRIORITY':
                k = 'INFO'
            elif c == 'MESSAGE':
                k = (
                    'Wrote xds to ' + os.path.basename(image)
                    + ' using cngi_io.xds_to_casa_image_2()'
                )
            elif c == 'LOCATION':
                k = 'cngi_io.xds_to_casa_image_2'
            elif c == 'OBJECT_ID':
                k = ''
            vals = np.append(vals, k)
            tb.putcol(c, vals)
    tb.close()


def __imageinfo_dict_from_xds(xds: xr.Dataset) -> dict:
    ii = {}
    ii['image_type'] = (
        xds.sky.attrs['image_type'] if 'image_type' in xds.sky.attrs else ''
    )
    ii['objectname'] = xds.attrs[__object_name]
    if 'beam' in xds.attrs:
        if xds.attrs['beam']:
            # do nothing if xds.attrs['beam'] is None
            ii['restoringbeam'] = xds.attrs['beam']
    elif 'beam' in xds.data_vars:
        # multi beam
        pp = {}
        pp['nChannels'] = len(xds.freq)
        pp['nStokes'] = len(xds.pol)
        bu = xds.beam.attrs['unit']
        chan = 0
        pol = 0
        bv = xds.beam.values
        for i in range(pp['nChannels'] * pp['nStokes']):
            bp = bv[0][pol][chan][:]
            b = {
                'major': {'unit': bu, 'value': bp[0]},
                'minor': {'unit': bu, 'value': bp[1]},
                'positionangle': {'unit': bu, 'value': bp[2]}
            }
            pp['*' + str(pp['nChannels']*pol + chan)] = b
            chan += 1
            if chan >= pp['nChannels']:
                chan = 0
                pol += 1
        ii['perplanebeams'] = pp
    return ii


def __write_casa_data(xds: xr.Dataset, image_full_path: str, casa_image_shape: tuple) -> None:
    sky_ap = 'sky' if 'sky' in xds else 'apeature'
    active_mask = xds.active_mask if __active_mask in xds.attrs else ''
    masks = []
    masks_rec = {}
    for m in xds.data_vars:
        attrs = xds[m].attrs
        if 'image_type' in attrs and attrs['image_type'] == 'Mask':
            masks_rec[m] = {
                'box': {
                    'blc': np.array([1., 1., 1., 1.]), 'comment': '', 'isRegion': 1,
                    'name': 'LCBox', 'oneRel': True, 'shape': np.array(casa_image_shape),
                    'trc': np.array(casa_image_shape)
                },
                'comment': '', 'isRegion': 1, 'name': 'LCPagedMask',
                'mask': f'Table: {os.sep.join([image_full_path, m])}'
            }
            masks.append(m)
    myvars = [sky_ap]
    myvars.extend(masks)
    for v in myvars:
        __write_pixels(v, active_mask, image_full_path, xds)
    if masks:
        tb = tables.table(
            image_full_path,
            readonly=False, lockoptions={'option': 'permanentwait'}, ack=False
        )
        tb.putkeyword('masks', masks_rec)
        tb.putkeyword('Image_defaultmask', active_mask)
        tb.close()


def __write_image_block(xda:xr.DataArray, outfile:str, blc:tuple) -> None:
    """
    Write image xda chunk to the corresponding image table slice
    """
    # trigger the DAG for this chunk and return values while the table is
    # unlocked
    values = xda.compute().values
    tb_tool = tables.table(
        outfile, readonly=False, lockoptions={'option': 'permanentwait'},
        ack=False
    )
    tb_tool.putcellslice(
        tb_tool.colnames()[0], 0, values, blc, tuple(
            np.array(blc) + np.array(values.shape) - 1
        )
    )
    tb_tool.close()


def __write_pixels(
    v: str, active_mask: str, image_full_path: str, xds: xr.Dataset
) -> None:
    flip = False
    if v == 'sky' or v == 'apeture':
        filename = image_full_path
    else:
        # mask
        flip = True
        filename = os.sep.join([image_full_path, v])
        if not os.path.exists(filename):
            tb = tables.table(os.sep.join([image_full_path, active_mask]))
            tb.copy(filename, deep=True, valuecopy=True)
            tb.close()
    arr = xds[v].isel(time=0).transpose(*('freq', 'pol', 'm', 'l'))
    chunk_bounds = arr.chunks
    b = [0, 0, 0, 0]
    loc0, loc1, loc2, loc3 = (0, 0, 0, 0)
    for i0 in chunk_bounds[0]:
        b[0] = loc0
        s0 = slice(b[0], b[0] + i0)
        loc1 = 0
        for i1 in chunk_bounds[1]:
            b[1] = loc1
            s1 = slice(b[1], b[1] + i1)
            loc2 = 0
            for i2 in chunk_bounds[2]:
                b[2] = loc2
                s2 = slice(b[2], b[2] + i2)
                loc3 = 0
                for i3 in chunk_bounds[3]:
                    b[3] = loc3
                    blc = tuple(b)
                    s3 = slice(b[3], b[3] + i3)
                    sub_arr = arr[s0, s1, s2, s3]
                    if flip:
                        sub_arr = np.logical_not(sub_arr)
                    __write_image_block(sub_arr, filename, blc)
                    loc3 += i3
                loc2 += i2
            loc1 += i1
        loc0 += i0
