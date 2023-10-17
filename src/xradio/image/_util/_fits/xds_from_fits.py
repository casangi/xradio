import astropy as ap
from astropy import units as u
from astropy.io import fits
from astropy.time import Time
from ..common import (
    __c, __default_freq_info, __doppler_types,
    __freq_from_vel, __get_xds_dim_order, __image_type
)
import copy
import dask
import dask.array as da
import logging
import numpy as np
import re
from typing import Union
import xarray as xr


def __fits_image_to_xds(
    img_full_path:str, chunks:dict, verbose:bool=False
) -> dict:
    """
    TODO: complete documentation
    Create an xds without any pixel data from metadata from the specified FITS image
    """
    # memmap = True allows only part of data to be loaded into memory
    # may also need to pass mode='denywrite'
    # https://stackoverflow.com/questions/35759713/astropy-io-fits-read-row-from-large-fits-file-with-mutliple-hdus
    hdulist = fits.open(img_full_path, memmap=True)
    attrs, helpers, header = __fits_header_to_xds_attrs(hdulist)
    hdulist.close()
    # avoid keeping reference to mem-mapped fits file
    del hdulist
    xds = __create_coords(helpers, header)
    sphr_dims = helpers['sphr_dims']
    ary = __read_image_array(img_full_path, chunks, helpers, verbose)
    dim_order = __get_xds_dim_order(sphr_dims)
    xds = __add_sky_or_apeture(xds, ary, dim_order, helpers, sphr_dims)
    xds.attrs = attrs
    xds = __add_coord_attrs(xds, helpers)
    if helpers['has_multibeam']:
        xds = __do_multibeam(xds, img_full_path)
    return xds


def __add_coord_attrs(xds: xr.Dataset, helpers: dict) -> xr.Dataset:
     xds = __add_time_attrs(xds, helpers)
     xds = __add_freq_attrs(xds, helpers)
     xds = __add_vel_attrs(xds, helpers)
     xds = __add_dir_lin_attrs(xds, helpers)
     return xds


def __add_time_attrs(xds:xr.Dataset, helpers:dict) -> xr.Dataset:
     time_coord = xds.coords['time']
     meta = copy.deepcopy(helpers['obsdate'])
     del meta['value']
     # meta['format'] = 'MJD'
     # meta['time_scale'] = meta['refer']
     # del meta['refer']
     time_coord.attrs = meta
     xds.assign_coords(time=time_coord)
     return xds


def __add_freq_attrs(xds:xr.Dataset, helpers:dict) -> xr.Dataset:
    freq_coord = xds.coords['freq']
    meta = {}
    if helpers['has_freq']:
        conv = {}
        conv['direction'] = {
            'm0': {'unit': 'rad', 'value': 0.0},
            'm1': {'unit': 'rad', 'value': np.pi/2},
            'system': helpers['ref_sys'],
            'equinox': helpers['ref_eqx']
        }
        conv['epoch'] = {'m0': {'value': 0.0, 'unit': 'd'}, 'refer': 'LAST'}
        conv['position'] = {
            'm0': {'unit': 'rad', 'value': 0.0},
            'm1': {'unit': 'rad', 'value': 0.0},
            'm2': {'unit': 'm', 'value': 0.0}, 'refer': 'ITRF'
        }
        conv['system'] = helpers['specsys']
        meta['conversion'] = conv
        meta['native_type'] = helpers['native_type']
        meta['restfreq'] = helpers['restfreq']
        meta['restfreqs'] = [ helpers['restfreq'] ]
        meta['system'] = helpers['specsys']
        meta['unit'] = 'Hz'
        meta['wave_unit'] = 'mm'
        wcs = {}
        freq_axis = helpers['freq_axis']
        wcs['crval'] = helpers['crval'][freq_axis]
        wcs['cdelt'] = helpers['cdelt'][freq_axis]
        meta['wcs'] = wcs
    if not meta:
        # this is the default frequency information CASA creates
        meta = __default_freq_info()
    freq_coord.attrs = copy.deepcopy(meta)
    xds['freq'] = freq_coord
    return xds


def __add_vel_attrs(xds:xr.Dataset, helpers:dict) -> xr.Dataset:
    vel_coord = xds.coords['vel']
    meta = {'unit': 'm/s'}
    if helpers['has_freq']:
        meta['doppler_type'] = helpers['doppler']
    else:
        meta['doppler_type'] = __doppler_types[0]
    vel_coord.attrs = copy.deepcopy(meta)
    xds.coords['vel'] = vel_coord
    return xds


def __add_dir_lin_attrs(xds:xr.Dataset, helpers:dict) -> xr.Dataset:
    if helpers['sphr_dims']:
        for i, name in zip(helpers['dir_axes'], helpers['sphr_axis_names']):
            meta = {
                'unit': 'rad',
                'wcs': {
                    'crval': helpers['crval'][i],
                    'cdelt': helpers['cdelt'][i]
                }
            }
            xds.coords[name].attrs = meta
    else:
        for i, j in zip(helpers['dir_axes'], ('u', 'v')):
            meta = {
                'unit': 'wavelengths',
                'wcs': {
                    'crval': helpers['crval'][i],
                    'cdelt': helpers['cdelt'][i]
                }
            }
            xds.coords[j].attrs = meta
    return xds


def __is_freq_like(v:str) -> bool:
    return v.startswith('FREQ') or v == 'VOPT' or v == 'VRAD'


def __fits_header_to_xds_attrs(hdulist:fits.hdu.hdulist.HDUList) -> dict:
    primary = None
    beams = None
    for hdu in hdulist:
        if hdu.name == 'PRIMARY':
            primary = hdu
        elif hdu.name == 'BEAMS':
            beams = hdu
        else:
            raise RuntimeError(f'Unknown HDU name {hdu.name}')
    if not primary:
        raise RuntimeError(f'No PRIMARY HDU found in fits file')
    dir_axes = None
    header = primary.header
    helpers = {}
    attrs = {}
    naxes = header.get('NAXIS')
    helpers['naxes'] = naxes
    # fits indexing starts at 1, not 0
    t_axes = np.array([0,0])
    dim_map = {}
    ctypes = []
    shape = []
    crval = []
    cdelt = []
    crpix = []
    cunit = []
    helpers['has_freq'] = False
    for i in range(1, naxes+1):
        ax_type = header[f'CTYPE{i}']
        if ax_type.startswith('RA-'):
            t_axes[0] = i
        elif ax_type.startswith('DEC-'):
            t_axes[1] = i
        elif ax_type == 'STOKES':
            dim_map['pol'] = i - 1
        elif __is_freq_like(ax_type):
            dim_map['freq'] = i - 1
            helpers['has_freq'] = True
            helpers['native_type'] = ax_type
        else:
            raise RuntimeError(f'{ax_type} is an unsupported axis')
        ctypes.append(ax_type)
        shape.append(header[f'NAXIS{i}'])
        crval.append(header[f'CRVAL{i}'])
        cdelt.append(header[f'CDELT{i}'])
        # FITS 1-based to python 0-based
        crpix.append(header[f'CRPIX{i}'] - 1)
        cunit.append(header[f'CUNIT{i}'])
    helpers['ctype'] = ctypes
    helpers['shape'] = shape
    helpers['crval'] = crval
    helpers['cdelt'] = cdelt
    helpers['crpix'] = crpix
    helpers['cunit'] = cunit
    if 'RESTFRQ' in header:
        helpers['restfreq'] = header['RESTFRQ']
    if 'SPECSYS' in header:
        helpers['specsys'] = header['SPECSYS']
    if (t_axes > 0).all():
        dir_axes = t_axes[:]
        dir_axes = dir_axes - 1
        helpers['dir_axes'] = dir_axes
        dim_map['l'] = dir_axes[0]
        dim_map['m'] = dir_axes[1]
        helpers['dim_map'] = dim_map
    else:
        raise RuntimeError('Could not find both direction axes')
    if dir_axes is not None:
        p0 = header[f'CTYPE{t_axes[0]}'][-3:]
        p1 = header[f'CTYPE{t_axes[1]}'][-3:]
        if p0 != p1:
            raise RuntimeError(
                f'Projections for direction axes ({p0}, {p1}) differ, but they '
                'must be the same'
            )
        direction = {}
        direction['projection'] = p0
        helpers['projection'] = p0
        ref_sys = header['RADESYS']
        ref_eqx = header['EQUINOX']
        if ref_sys == 'FK5' and ref_eqx == 2000:
            ref_eqx = 'J2000'
        helpers['ref_sys'] = ref_sys
        helpers['ref_eqx'] = ref_eqx
        # fits does not support conversion frames
        direction['conversion_system'] = ref_sys
        direction['conversion_equinox'] = ref_eqx
        direction['system'] = ref_sys
        direction['equinox'] = ref_eqx
        deg_to_rad = np.pi/180.0
        direction['latpole'] = {
            'value': header['LATPOLE'] * deg_to_rad,
            'unit': 'rad'
        }
        direction['longpole'] = {
            'value': header['LONPOLE'] * deg_to_rad,
            'unit': 'rad'
        }
        pc = np.zeros([2,2])
        for i in (0, 1):
            for j in (0, 1):
                # dir_axes are now 0-based, but fits needs 1-based
                pc[i][j] = header[f'PC{dir_axes[i]+1}_{dir_axes[j]+1}']
        direction['pc'] = pc
        # Is there really no fits header parameter for projection_parameters?
        direction['projection_parameters'] = np.array([0.0, 0.0])
        attrs['direction'] = direction
    # FIXME read fits data in chunks in case all data too large to hold in memory
    has_mask = da.any(da.isnan(primary.data)).compute()
    attrs['active_mask'] = 'mask0' if has_mask else None
    helpers['has_mask'] = has_mask
    helpers['has_multibeam'] = False
    if 'BMAJ' in header:
        # single global beam
        attrs['beam'] = {
            'bmaj': {'unit': 'arcsec', 'value': header['BMAJ']},
            'bmin': {'unit': 'arcsec', 'value': header['BMIN']},
            'positionangle': {'unit': 'arcsec', 'value': header['BPA']}
        }
    elif 'CASAMBM' in header and header['CASAMBM']:
        # multi-beam
        helpers['has_multibeam'] = True
    else:
        # no beam
        attrs['beam'] = None
    if 'BITPIX' in header:
        v = abs(header['BITPIX'])
        if v == 32:
            helpers['dtype'] = 'float32'
        elif v == 64:
            helpers['dtype'] = 'float64'
        else:
            raise RuntimeError(f'Unhandled data type {header["BITPIX"]}')
    helpers['btype'] = header['BTYPE'] if 'BTYPE' in header else None
    helpers['bunit'] = header['BUNIT'] if 'BUNIT' in header else None
    attrs['object_name'] = header['OBJECT'] if 'OBJECT' in header else None
    obsdate = {}
    obsdate['value'] = Time(header['DATE-OBS'], format='isot').mjd
    obsdate['unit'] = 'd'
    obsdate['time_scale'] = header['TIMESYS']
    obsdate['format'] = 'MJD'
    attrs['obsdate'] = obsdate
    helpers['obsdate'] = obsdate
    attrs['observer'] = header['OBSERVER']
    long_unit = header[f'CUNIT{t_axes[0]}']
    lat_unit = header[f'CUNIT{t_axes[1]}']
    unit = []
    for uu in [long_unit, lat_unit]:
        if uu == 'deg':
            unit.append(u.deg)
        elif uu == 'rad':
            unit.append(u.rad)
        else:
            raise RuntimeError(f'Unsupported direction unit {uu}')
    pc_long = float(header.get(f'CRVAL{t_axes[0]}')) * unit[0]
    pc_lat = float(header.get(f'CRVAL{t_axes[1]}')) * unit[1]
    pc_long = pc_long.to(u.rad).value
    pc_lat = pc_lat.to(u.rad).value
    attrs['pointing_center'] = {
        'value': np.array([pc_long, pc_lat]), 'initial': True
    }
    attrs['description'] = None
    tel = {}
    tel['name'] = header.get('TELESCOP')
    x = header.get('OBSGEO-X')
    y = header.get('OBSGEO-Y')
    z = header.get('OBSGEO-Z')
    xyz = np.array([x, y, z])
    r = np.sqrt(np.sum(xyz*xyz))
    lat = np.arcsin(z/r)
    long = np.arctan2(y, x)
    tel['position'] = {
        'type': 'position', 'refer': 'ITRF',
        'm2': {'value': r, 'unit': 'm'},
        'm1': {'unit': 'rad', 'value': lat},
        'm0': {'unit': 'rad', 'value': long}
    }
    attrs['telescope'] = tel
    helpers['tel_pos'] = tel['position']
    # TODO complete __make_history_xds when spec has been finalized
    # attrs['history'] = __make_history_xds(header)
    exclude = [
        'ALTRPIX', 'ALTRVAL', 'BITPIX', 'BSCALE', 'BTYPE', 'BUNIT',
        'BZERO', 'CASAMBM', 'DATE', 'DATE-OBS', 'EQUINOX', 'EXTEND',
        'HISTORY', 'LATPOLE', 'LONPOLE', 'OBSERVER', 'ORIGIN', 'TELESCOP',
        'OBJECT', 'RADESYS', 'RESTFRQ', 'SIMPLE', 'SPECSYS', 'TIMESYS',
        'VELREF'
    ]
    regex = r'|'.join([
        '^NAXIS\d?$', '^CRVAL\d$', '^CRPIX\d$', '^CTYPE\d$', '^CDELT\d$',
        '^CUNIT\d$', '^OBSGEO-(X|Y|Z)$', '^P(C|V)\d_\d$'
    ])
    user = {}
    for (k, v) in header.items():
        if re.search(regex, k) or k in exclude:
            continue
        user[k.lower()] = v
    attrs['user'] = user
    return attrs, helpers, header


def __make_history_xds(header):
    # TODO complete writing history when we actually have a spec for what
    # the image history is supposed to be, since doing this now may
    # be a waste of time if the final spec turns out to be significantly
    # different from our current ad hoc history xds
    # in astropy, 3506803168 seconds corresponds to 1970-01-01T00:00:00
    history_list = list(header.get('HISTORY'))
    for i in range(len(history_list)-1, -1, -1):
        if (
            (
                i == len(history_list) - 1
                and history_list[i] == 'CASA END LOGTABLE'
            ) or (
                i == 0
                and history_list[i] == 'CASA START LOGTABLE'
            )
        ):
            history_list.pop(i)
        elif history_list[i].startswith('>'):
            # entry continuation line
            history_list[i-1] = history_list[i-1] + history_list[i][1:]
            history_list.pop(i)


def __create_coords(helpers, header):
    dir_axes = helpers['dir_axes']
    dim_map = helpers['dim_map']
    sphr_dims = (
        [dim_map['l'], dim_map['m']]
        if ('l' in dim_map) and ('m' in dim_map)
        else []
    )
    helpers['sphr_dims'] = sphr_dims
    coords = {}
    coords['time'] = __get_time_values(helpers)
    coords['pol'] = __get_pol_values(helpers)
    coords['freq'] = __get_freq_values(helpers)
    coords['vel'] = (['freq'], __get_velocity_values(helpers))
    if len(sphr_dims) > 0:
        l_world, m_world = __compute_world_sph_dims(
            sphr_dims, dir_axes, dim_map, helpers
        )
        coords[l_world[0]] = (['l', 'm'], l_world[1])
        coords[m_world[0]] = (['l', 'm'], m_world[1])
        helpers['sphr_axis_names'] = (l_world[0], m_world[0])
    else:
        # Fourier image
        coords['u'], coords['v'] = __get_uv_values(helpers)
    xds = xr.Dataset(coords=coords)
    # attrs['xds'] = xds
    # return attrs
    return xds


def __get_time_values(helpers):
    return [ helpers['obsdate']['value'] ]


def __get_pol_values(helpers):
    # as mapped in casacore Stokes.h
    stokes_map = [
        'Undefined', 'I', 'Q', 'U', 'V',
        'RR', 'RL', 'LR', 'LL',
        'XX', 'XY', 'YX', 'YY',
        'RX', 'RY', 'LX', 'LY',
        'XR', 'XL', 'YR', 'YL',
        'PP', 'PQ'
    ]
    idx = helpers['ctype'].index('STOKES')
    if idx >= 0:
        vals = []
        crval = int(helpers['crval'][idx])
        crpix = int(helpers['crpix'][idx])
        cdelt = int(helpers['cdelt'][idx])
        stokes_start_idx = crval - cdelt*crpix
        for i in range(helpers['shape'][idx]):
            stokes_idx = (stokes_start_idx + i)*cdelt
            vals.append(stokes_map[stokes_idx])
        return vals
    else:
        return ['I']


def __get_freq_values(helpers:dict) -> list:
    vals = []
    ctype = helpers['ctype']
    if 'FREQ' in ctype:
        freq_idx = ctype.index('FREQ')
        helpers['freq_axis'] = freq_idx
        crval = helpers['crval'][freq_idx]
        crpix = helpers['crpix'][freq_idx]
        cdelt = helpers['cdelt'][freq_idx]
        cunit = helpers['cunit'][freq_idx]
        freq_start_val = crval - cdelt*crpix
        for i in range(helpers['shape'][freq_idx]):
            vals.append(freq_start_val + i*cdelt)
        helpers['freq'] = vals * u.Unit(cunit)
        return vals
    elif 'VOPT' in ctype:
        if 'restfreq' in helpers:
            restfreq = helpers['restfreq'] * u.Hz
        else:
            raise RuntimeError(
                'Spectral axis in FITS header is velocity, but there is '
                'no rest frequency so converting to frequency is not possible'
            )
        helpers['doppler'] = 'Z'
        v_idx = ctype.index('VOPT')
        helpers['freq_idx'] = v_idx
        helpers['freq_axis'] = v_idx
        crval = helpers['crval'][v_idx]
        crpix = helpers['crpix'][v_idx]
        cdelt = helpers['cdelt'][v_idx]
        cunit = helpers['cunit'][v_idx]
        freq, vel =  __freq_from_vel(
            crval, cdelt, crpix, cunit, 'Z',
            helpers['shape'][v_idx], restfreq
        )
        helpers['vel'] = vel['value'] * u.Unit(vel['unit'])
        helpers['crval'][v_idx] = (
            freq['crval'] * u.Unit(freq['unit'])
        ).to(u.Hz).value
        helpers['cdelt'][v_idx] = (
            freq['cdelt'] * u.Unit(freq['unit'])
        ).to(u.Hz).value
        return list(freq['value'])
    else:
        return [1420e6]


def __get_velocity_values(helpers:dict) -> list:
    if 'vel' in helpers:
        return helpers['vel'].to(u.m/u.s).value
    elif 'freq' in helpers:
        if helpers['doppler'] == 'Z':
            # (-1 + f0/f) = v/c
            v = (helpers['restfreq']*u.Hz/helpers['freq'].to('Hz').value - 1) * __c
            v = v.to(u.m/u.s)
            helpers['vel'] = v
            return v.value


def __compute_world_sph_dims(
    sphr_dims:list, dir_axes:list, dim_map:dict, helpers:dict
) -> list:
    shape = helpers['shape']
    ctype = helpers['ctype']
    unit = helpers['cunit']
    delt = helpers['cdelt']
    ref_pix = helpers['crpix']
    ref_val = helpers['crval']
    wcs_dict = {}
    for i in dir_axes:
        if ctype[i].startswith('RA'):
            long_axis_name = 'right_ascension'
            fi = 1
            wcs_dict[f'CTYPE1'] = ctype[i]
            wcs_dict[f'NAXIS1'] = shape[dim_map['l']]
        if ctype[i].startswith('DEC'):
            lat_axis_name = 'declination'
            fi = 2
            wcs_dict['CTYPE2'] = ctype[i]
            wcs_dict[f'NAXIS2'] = shape[dim_map['m']]
        t_unit = unit[i]
        if t_unit == "'":
            t_unit = 'arcmin'
        elif t_unit == '"':
            t_unit = 'arcsec'
        wcs_dict[f'CUNIT{fi}'] = t_unit
        wcs_dict[f'CDELT{fi}'] = delt[i]
        # FITS arrays are 1-based
        wcs_dict[f'CRPIX{fi}'] = ref_pix[i] + 1
        wcs_dict[f'CRVAL{fi}'] = ref_val[i]

    w = ap.wcs.WCS(wcs_dict)
    x, y = np.indices(w.pixel_shape)
    long, lat = w.pixel_to_world_values(x, y)
    # long, lat will always be in degrees, so convert to rad
    f = np.pi/180
    long *= f
    lat *= f
    crpix = helpers['crpix'][dir_axes[0]], helpers['crpix'][dir_axes[1]]
    wcrvalx, wcrvaly = w.pixel_to_world_values(crpix[0], crpix[1])
    wcrvalx = wcrvalx.tolist() * f
    wcrvaly = wcrvaly.tolist() * f
    wcrval = [wcrvalx, wcrvaly]
    for i, j in zip(dir_axes, (0, 1)):
        helpers['cunit'][i] = 'rad'
        helpers['crval'][i] = wcrval[j]
        helpers['cdelt'][i] *= f
    return [[long_axis_name, long], [lat_axis_name, lat]]


def __do_multibeam(xds:xr.Dataset, imname:str) -> xr.Dataset:
    """Only run if we are sure there are multiple beams"""
    hdulist = fits.open(imname)
    for hdu in hdulist:
        header = hdu.header
        if 'EXTNAME' in header and header['EXTNAME'] == 'BEAMS':
            units = (
                 u.Unit(header['TUNIT1']), u.Unit(header['TUNIT2']),
                 u.Unit(header['TUNIT3'])
            )
            nchan = header['NCHAN']
            npol = header['NPOL']
            beam_array = np.zeros([1, npol, nchan, 3])
            data = hdu.data
            for t in data:
                beam_array[0, t[4], t[3]] = t[0:3]
            for i in (0, 1, 2):
                beam_array[:, :, :, i] = (
                    (beam_array[:, :, :, i] * units[i]).to('rad').value
                )
            xdb = xr.DataArray(
                beam_array, dims=['time', 'pol', 'freq', 'beam_param']
            )
            xdb = xdb.rename('beam')
            xdb = xdb.assign_coords(beam_param=['major', 'minor', 'pa'])
            xdb.attrs['unit'] = 'rad'
            xds['beam'] = xdb
            return xds
    raise RuntimeError(
        'It looks like there should be a BEAMS table but no '
        'such table found in FITS file'
    )


def __get_uv_values(helpers:dict) -> tuple:
    shape = helpers['shape']
    ctype = helpers['ctype']
    unit = helpers['cunit']
    delt = helpers['cdelt']
    ref_pix = helpers['crpix']
    ref_val = helpers['crval']
    for i, axis in enumerate(['UU', 'VV']):
        idx = ctype.index(axis)
        if idx >= 0:
            z = []
            crpix = ref_pix[i]
            crval = ref_val[i]
            cdelt = delt[i]
            for i in range(shape[idx]):
                f = (i - crpix) * cdelt + crval
                z.append(f)
            if axis == 'UU':
                u = z
            else:
                v = z
    return u, v


def __add_sky_or_apeture(
    xds:xr.Dataset, ary:Union[np.ndarray, da.array],
    dim_order:list, helpers:dict, has_sph_dims:bool
) -> xr.Dataset:
    xda = xr.DataArray(ary, dims=dim_order)
    image_type = helpers['btype']
    unit = helpers['bunit']
    xda.attrs[__image_type] = image_type
    xda.attrs['unit'] = unit
    name = 'sky' if has_sph_dims else 'apeture'
    xda = xda.rename(name)
    xds[xda.name] = xda
    if helpers['has_mask']:
        pp = da if type(xda[0].data) == dask.array.core.Array else np
        mask = pp.isnan(xda)
        mask.attrs = {}
        mask = mask.rename('mask0')
        xds['mask0'] = mask
    return xds


def __read_image_array(
    img_full_path:str, chunks:dict, helpers:dict, verbose:bool
) -> da.array:
    # memmap = True allows only part of data to be loaded into memory
    # may also need to pass mode='denywrite'
    # https://stackoverflow.com/questions/35759713/astropy-io-fits-read-row-from-large-fits-file-with-mutliple-hdus
    if isinstance(chunks, dict):
        mychunks = __get_chunk_list(chunks, helpers)
    else:
        raise ValueError(
            f'incorrect type {type(chunks)} for parameter chunks. Must be '
            'dict'
        )
    transpose_list, new_axes = __get_transpose_list(helpers)
    data_type = helpers['dtype']
    rshape = helpers['shape'][::-1]
    full_chunks = mychunks + tuple([1 for rr in range(5) if rr >= len(mychunks)])
    d0slices = []
    blc = tuple(5 * [0])
    trc = tuple(rshape) + tuple([1 for rr in range(5) if rr >= len(mychunks)])
    for d0 in range(blc[0], trc[0], full_chunks[0]):
        d0len = min(full_chunks[0], trc[0] - d0)
        d1slices = []
        for d1 in range(blc[1], trc[1], full_chunks[1]):
            d1len = min(full_chunks[1], trc[1] - d1)
            d2slices = []
            for d2 in range(blc[2], trc[2], full_chunks[2]):
                d2len = min(full_chunks[2], trc[2] - d2)
                d3slices = []
                for d3 in range(blc[3], trc[3], full_chunks[3]):
                    d3len = min(full_chunks[3], trc[3] - d3)
                    d4slices = []
                    for d4 in range(blc[4], trc[4], full_chunks[4]):
                        d4len = min(full_chunks[4], trc[4] - d4)
                        shapes = tuple(
                            [d0len, d1len, d2len, d3len, d4len][:len(rshape)]
                        )
                        starts = tuple([d0, d1, d2, d3, d4][:len(rshape)])
                        delayed_array = dask.delayed(__read_image_chunk)(
                            img_full_path, shapes, starts
                        )
                        d4slices += [
                            da.from_delayed(
                                delayed_array, shapes, data_type
                            )
                        ]
                    d3slices += (
                        [da.concatenate(d4slices, axis=4)]
                        if len(rshape) > 4 else d4slices
                    )
                d2slices += (
                    [da.concatenate(d3slices, axis=3)]
                    if len(rshape) > 3 else d3slices
                )
            d1slices += (
                [da.concatenate(d2slices, axis=2)]
                if len(rshape) > 2 else d2slices
            )
        d0slices += (
            [da.concatenate(d1slices, axis=1)]
            if len(rshape) > 1 else d1slices
        )
    ary = da.concatenate(d0slices, axis=0)
    ary = da.expand_dims(ary, new_axes)
    return ary.transpose(transpose_list)


def __get_chunk_list(chunks:dict, helpers:dict) -> tuple:
    ret_list = list(helpers['shape'])[::-1]
    axis = 0
    ctype = helpers['ctype']
    for c in ctype[::-1]:
        if c.startswith('RA'):
            if 'l' in chunks:
                ret_list[axis] = chunks['l']
        elif c.startswith('DEC'):
            if 'm' in chunks:
                ret_list[axis] = chunks['m']
        elif c.startswith('FREQ') or c.startswith('VOPT') or c.startswith('VRAD'):
            if 'freq' in chunks:
                ret_list[axis] = chunks['freq']
        elif c.startswith('STOKES'):
            if 'pol' in chunks:
                ret_list[axis] = chunks['pol']
        else:
            raise RuntimeError(f'Unhandled coordinate type {c}')
        axis += 1
    return tuple(ret_list)


def __get_transpose_list(helpers:dict) -> tuple:
    ctype = helpers['ctype']
    transpose_list = 5 * [-1]
    # time axis
    transpose_list[0] = 4
    new_axes = [4]
    last_axis = 3
    not_covered = ['l', 'm', 'u', 'v', 's', 'f']
    for i, c in enumerate(ctype[::-1]):
        b = c.lower()
        if b.startswith('ra') or b.startswith('uu'):
            transpose_list[3] = i
            not_covered.remove('l')
            not_covered.remove('u')
        elif b.startswith('dec') or b.startswith('vv'):
            transpose_list[4] = i
            not_covered.remove('m')
            not_covered.remove('v')
        elif b.startswith('freq') or b.startswith('vopt') or b.startswith('vrad'):
            transpose_list[2] = i
            not_covered.remove('f')
        elif b.startswith('stok'):
            transpose_list[1] = i
            not_covered.remove('s')
        else:
            raise RuntimeError(f'Unhandled axis name {c}')
    h = {'l': 3, 'm': 4, 'u': 3, 'v': 4, 'f': 2, 's': 1}
    for p in not_covered:
        transpose_list[h[p]] = last_axis
        new_axes.append(last_axis)
        last_axis -= 1
    new_axes.sort()
    if transpose_list.count(-1) > 0:
        raise RuntimeError(
            f'Logic error: axes {axes}, transpose_list {transpose_list}'
        )
    return transpose_list, new_axes


def __read_image_chunk(img_full_path, shapes:tuple, starts:tuple) -> np.ndarray:
    hdulist = fits.open(img_full_path, memmap=True)
    s = []
    for start, length in zip(starts, shapes):
        s.append(slice(start, start+length))
    t = tuple(s)
    z = hdulist[0].data[t]
    hdulist.close()
    # delete to avoid having a reference to a mem-mapped hdulist
    del hdulist
    return z

