from astropy.io import fits
from astropy.time import Time
import dask.array as da
import logging
import numpy as np


def __fits_image_to_xds_metadata(img_full_path:str, verbose:bool=False) -> dict:
    """
    TODO: complete documentation
    Create an xds without any pixel data from metadata from the specified FITS image
    """
    hdulist = fits.open(img_full_path)
    logging.warn('** ab')
    attrs = __fits_header_to_xds_attrs(hdulist)
    logging.warn('** ac')
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
    attrs = {}
    naxis = header.get('NAXIS')
    # fits indexing starts at 1, not 0
    t_axes = np.array([0,0])
    logging.warn('** cc')
    for i in range(1, naxis+1):
        ax_type = header.get(f'CTYPE{i}')
        if ax_type.startswith('RA-'):
            t_axes[0] = i
        elif ax_type.startswith('DEC-'):
            t_axes[1] = i
    if (t_axes > 0).all():
        dir_axes = t_axes[:]
    logging.warn('** cd')
    if dir_axes is not None:
        p0 = header.get(f'CTYPE{dir_axes[0]}')[-3:]
        p1 = header.get(f'CTYPE{dir_axes[1]}')[-3:]
        if p0 != p1:
            raise RuntimeError(
                f'Projections for direction axes ({p0}, {p1}) differ, but they '
                'must be the same'
            )
        direction = {}
        direction['projection'] = p0
        ref_sys = header.get('RADESYS')
        ref_eqx = header.get('EQUINOX')
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
        logging.warn('** dc')
        direction['longpole'] = header.get('LONPOLE') * deg_to_rad
        logging.warn('** dd')
        pc = np.zeros([2,2])
        for i in (0, 1):
            for j in (0, 1):
                pc[i][j] = header.get(f'PC{dir_axes[i]}_{dir_axes[j]}')
        direction['pc'] = pc
        attrs['direction'] = direction
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
    return attrs




