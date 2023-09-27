from astropy import units as u
from astropy.io import fits
from astropy.time import Time
import dask.array as da
import logging
import numpy as np
import re
from ..common import __c

def __fits_image_to_xds_metadata(img_full_path:str, verbose:bool=False) -> dict:
    """
    TODO: complete documentation
    Create an xds without any pixel data from metadata from the specified FITS image
    """
    hdulist = fits.open(img_full_path)
    attrs, helpers, header = __fits_header_to_xds_attrs(hdulist)
    __create_coords(helpers, header)
    return attrs


def __fits_header_to_xds_attrs(hdulist: fits.hdu.hdulist.HDUList) -> dict:
    primary = None
    beams = None
    for hdu in hdulist:
        if hdu.name == 'PRIMARY':
            primary = hdu
        elif hdu.name == 'BEAMS':
            beams = hdu
        else:
            raise RuntimeError(f'Unknown HDU name {hdu.name}')
    logging.warn('** cb')
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
    for i in range(1, naxes+1):
        ax_type = header[f'CTYPE{i}']
        if ax_type.startswith('RA-'):
            t_axes[0] = i
        elif ax_type.startswith('DEC-'):
            t_axes[1] = i
        elif ax_type == 'STOKES':
            dim_map['pol'] = i - 1
        elif ax_type.startswith('FREQ') or ax_type == 'VOPT':
            dim_map['freq'] = i - 1
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
    if (t_axes > 0).all():
        dir_axes = t_axes[:]
        dir_axes = dir_axes - 1
        helpers['dir_axes'] = dir_axes
        dim_map['l'] = dir_axes[0]
        dim_map['m'] = dir_axes[1]
        helpers['dim_map'] = dim_map
    else:
        raise RuntimeError('Could not find both direction axes')
    logging.warn('** cd')
    if dir_axes is not None:
        p0 = header.get(f'CTYPE{t_axes[0]}')[-3:]
        p1 = header.get(f'CTYPE{t_axes[1]}')[-3:]
        if p0 != p1:
            raise RuntimeError(
                f'Projections for direction axes ({p0}, {p1}) differ, but they '
                'must be the same'
            )
        direction = {}
        direction['projection'] = p0
        ref_sys = header['RADESYS']
        ref_eqx = header['EQUINOX']
        # fits does not support conversion frames
        direction['conversion_system'] = ref_sys
        direction['conversion_equinox'] = ref_eqx
        direction['system'] = ref_sys
        logging.warn('** ce')
        direction['equinox'] = ref_eqx
        logging.warn('** da')
        deg_to_rad = np.pi/180.0
        logging.warn('** db')
        direction['latpole'] = header.get('LATPOLE') * deg_to_rad
        direction['longpole'] = header.get('LONPOLE') * deg_to_rad
        pc = np.zeros([2,2])
        for i in (0, 1):
            for j in (0, 1):
                pc[i][j] = header.get(f'PC{dir_axes[i]}_{dir_axes[j]}')
        direction['pc'] = pc
        attrs['direction'] = direction
    # FIXME read fits data in chunks in case all data too large to hold in memory
    has_mask = da.any(da.isnan(primary.data)).compute()
    attrs['active_mask'] = 'mask0' if has_mask else None
    if 'BMAJ' in header.keys():
        # single global beam
        attrs['beam'] = {
            'bmaj': {'unit': 'arcsec', 'value': header.get('BMAJ')},
            'bmin': {'unit': 'arcsec', 'value': header.get('BMIN')},
            'positionangle': {'unit': 'arcsec', 'value': header.get('BPA')}
        }
    elif 'CASAMBM' in header.keys() and header.get('CASAMBM'):
        # multi-beam
        pass
    else:
        # no beam
        attrs['beam'] = None
    attrs['object'] = header.get('OBJECT') if 'OBJECT' in header else None
    obsdate = {}
    obsdate['value'] = Time(header.get('DATE-OBS'), format='isot').mjd
    obsdate['unit'] = 'd'
    obsdate['refer'] = 'UTC1'
    attrs['obsdate'] = obsdate
    helpers['obsdate'] = obsdate
    attrs['observer'] = header.get('OBSERVER')
    long_unit = header.get(f'CUNIT{t_axes[0]}')
    lat_unit = header.get(f'CUNIT{t_axes[1]}')
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
    attrs['description'] = ''
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
    print(attrs)
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
    print('sph_dims', sphr_dims)
    coords = {}
    coords['time'] = __get_time_values(helpers)
    coords['pol'] = __get_pol_values(helpers)
    coords['freq'] = __get_freq_values(helpers)
    coords['vel'] = (['freq'], __get_velocity_values(helpers))
    print('vel', coords['vel'][1])


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
    print('** ia')
    vals = []
    ctype = helpers['ctype']
    print('** ib')
    if 'FREQ' in ctype:
        freq_idx = ctype.index('FREQ')
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
        helpers['doppler'] = 'optical'
        v_idx = ctype.index('VOPT')
        crval = helpers['crval'][v_idx]
        crpix = helpers['crpix'][v_idx]
        cdelt = helpers['cdelt'][v_idx]
        cunit = helpers['cunit'][v_idx]
        v_start_val = crval - cdelt*crpix
        vel = []
        for i in range(helpers['shape'][v_idx]):
            vel.append(v_start_val + i*cdelt)
        vel = vel * u.Unit(cunit)
        print('** ie')
        # (-1 + f0/f) = v/c
        # f0/f = v/c + 1
        # f = f0/(v/c + 1)
        print('** ja')
        freq = restfreq/(np.array(vel.value) * vel.unit/__c + 1)
        print('** if')
        freq = freq.to(u.Hz)
        helpers['vel'] = vel
        print('** ig')
        return list(freq.value)
    else:
        return [1420e6]


def __get_velocity_values(helpers:dict) -> list:
    if 'vel' in helpers:
        return helpers['vel'].to(u.m/u.s).value
    elif 'freq' in helpers:
        if helpers['doppler'] == 'optical':
            # (-1 + f0/f) = v/c
            v = (helpers['restfreq']/helpers['freq'].to('Hz').value - 1) * __c
            v = v.to(u.m)
            helpers['vel'] = v
            return v.value




