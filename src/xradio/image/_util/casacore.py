#  CASA Next Generation Infrastructure
#  Copyright (C) 2021 AUI, Inc. Washington DC, USA
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

#################################
# Helper File
#
# Not exposed in API
#
#################################
import warnings, time, os, logging
import numpy as np
import astropy.wcs
from .common import __get_xds_dim_order, __dask_arrayize
from ._casacore.common import __active_mask
from ._casacore.xds_to_casacore import (
    __coord_dict_from_xds, __history_from_xds, __imageinfo_dict_from_xds,
    __write_casa_data
)
from ._casacore.xds_from_casacore import (
    __add_coord_attrs, __add_mask, __add_sky_or_apeture,
    __casa_image_to_xds_attrs, __casa_image_to_xds_metadata, __get_mask_names,
    __get_persistent_block, __get_starts_shapes_slices, __get_transpose_list,
    __make_coord_subset, __multibeam_array, __read_image_array
)
from astropy.coordinates import SkyCoord  # High-level coordinates
from astropy.coordinates import ICRS, Galactic, FK4, FK5  # Low-level frames
from astropy.coordinates import Angle, Latitude, Longitude  # Angles
from astropy import units as u
from casacore import images, tables
from typing import Union
import xarray as xr
from casacore.images import image

warnings.filterwarnings('ignore', category=FutureWarning)


def __load_casa_image_block(infile: str, block_des: dict) -> xr.Dataset:
    image_full_path = os.path.expanduser(infile)
    casa_image = image(image_full_path)
    coords = casa_image.coordinates()
    cshape = casa_image.shape()
    del casa_image
    ret = __casa_image_to_xds_metadata(image_full_path, False)
    xds = ret['xds']
    starts, shapes, slices = __get_starts_shapes_slices(block_des, coords, cshape)
    xds = __make_coord_subset(xds, slices)
    dimorder = __get_xds_dim_order(ret['sphr_dims'])
    transpose_list, new_axes = __get_transpose_list(coords)
    block = __get_persistent_block(image_full_path, shapes, starts, dimorder, transpose_list, new_axes)
    xds = __add_sky_or_apeture(xds, block, dimorder, image_full_path, ret['sphr_dims'])
    mymasks = __get_mask_names(image_full_path)
    for m in mymasks:
        full_path = os.sep.join([image_full_path, m])
        block = __get_persistent_block(
            full_path, shapes, starts, dimorder, transpose_list, new_axes
        )
        xds = __add_mask(xds, m, block, dimorder)
    xds.attrs = __casa_image_to_xds_attrs(image_full_path, True)
    mb = __multibeam_array(xds, image_full_path, False)
    if mb is not None:
        selectors = {}
        for k in ('time', 'pol', 'freq'):
            if k in block_des:
                selectors[k] = block_des[k]
        xds['beam'] = mb.isel(selectors)
    xds = __add_coord_attrs(xds, ret['icoords'], ret['dir_axes'])
    return xds


def __read_casa_image(
    infile:str, chunks:Union[list, dict], masks:bool=True,
    history:bool=True, verbose:bool=False
) -> xr.Dataset:
    img_full_path = os.path.expanduser(infile)
    ret = __casa_image_to_xds_metadata(img_full_path, verbose)
    xds = ret['xds']
    dimorder = __get_xds_dim_order(ret['sphr_dims'])
    xds = __add_sky_or_apeture(
        xds, __read_image_array(img_full_path, chunks, verbose=verbose),
        dimorder, img_full_path, ret['sphr_dims']
    )
    if masks:
        mymasks = __get_mask_names(img_full_path)
        for m in mymasks:
            ary = __read_image_array(img_full_path, chunks, mask=m, verbose=verbose)
            xds = __add_mask(xds, m, ary, dimorder)
    xds.attrs = __casa_image_to_xds_attrs(img_full_path, history)
    mb = __multibeam_array(xds, img_full_path, True)
    if mb is not None:
        xds['beam'] = mb
    xds = __add_coord_attrs(xds, ret['icoords'], ret['dir_axes'])
    xds = __dask_arrayize(xds)
    return xds


def __xds_to_casa_image(xds: xr.Dataset, imagename:str) -> None:
    sky_ap = 'sky' if 'sky' in xds else 'apeature'
    if xds[sky_ap].shape[0] != 1:
        raise Exception('XDS can only be converted if it has exactly one time plane')
    arr = xds[sky_ap].isel(time=0).transpose(*('freq', 'pol', 'm', 'l'))
    image_full_path = os.path.expanduser(imagename)
    maskname = ''
    if __active_mask in xds.attrs and xds.attrs[__active_mask]:
        maskname = xds.attrs[__active_mask]
    # create the image and then delete the object
    casa_image = images.image(image_full_path, maskname=maskname, shape=arr.shape)
    del casa_image
    chunk_bounds = arr.chunks
    __write_casa_data(xds, image_full_path, arr.shape[::-1])
    # create coordinates
    coord = __coord_dict_from_xds(xds)
    ii = __imageinfo_dict_from_xds(xds)
    units = xds.sky.attrs['unit'] if 'unit' in xds.sky.attrs else None
    miscinfo = (
        xds.attrs['user']
        if 'user' in xds.attrs and len(xds.attrs['user']) > 0 else None
    )
    tb = tables.table(
        image_full_path, readonly=False, lockoptions={'option': 'permanentwait'},
        ack=False
    )
    tb.putkeyword('coords', coord)
    tb.putkeyword('imageinfo', ii)
    if units:
        tb.putkeyword('units', units)
    if miscinfo:
        tb.putkeyword('miscinfo', miscinfo)
    tb.done()
    # history
    __history_from_xds(xds, image_full_path)
