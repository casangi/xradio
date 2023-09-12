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
#
# Public interface
#
#################################
from ._util.casacore import __load_casa_image_block, __xds_to_casa_image
from ._util.fits import __read_fits_image
from ._util.zarr import __xds_to_zarr, __xds_from_zarr
import warnings, time, os, logging
import numpy as np
import astropy.wcs
import xradio
from astropy import units as u
from typing import Union
import xarray as xr

warnings.filterwarnings('ignore', category=FutureWarning)


def read_image(
    infile:str, chunks:dict={}, masks:bool=True,
    history:bool=True, verbose:bool=False
) -> xr.Dataset:
    """
    Read Image (currently only supports casacore images) to ngCASA image format
    ngCASA image spec is located at
    https://docs.google.com/spreadsheets/d/1buwhdrQpeWgXe-f0HO4gSHJ_kNsRoRgUAGPlYsU3DH8/edit?pli=1#gid=1049873214

    :param infile: Path to the input CASA image
    :type infile: str, required
    :param chunks: The desired dask chunk size. Only applicable for casacore images.
                   Supported optional keys are 'l', 'm', 'freq', 'pol',
                   and 'time'. The supported values are positive integers,
                   indicating the length of a chunk on that particular axis. If
                   a key is missing, then the associated chunk length along that axis
                   is equal to the number of pixels along that axis.
                   for zarr images, this parameter is ignored and the chunk size
                   used to store the arrays in the zarr image is used.
                   'l' represents the longitude like dimension, and 'm'
                   represents the latitude like dimension. For apeature images,
                   'u' may be used in place of 'l', and 'v' in place of 'm'.
    :type chunks: list | dict, required
    :param masks: Also read mask data? Default is True.
    :type masks: bool, optional
    :param history: Read and store history? Default is True.
    :type history: bool, optional
    :param verbose: emit debugging messages? Default is False.
    :type verbose: bool, optional
    :return: xr.Dataset image that conforms to the ngCASA image spec
    :rtype: xr.Dataset
    """
    do_casa = True
    try:
        from ._util.casacore import __read_casa_image
    except:
        logging.warn(
            'python-casacore not found, will not try to read as casacore image'
        )
        do_casa = False
    if do_casa:
        try:
            return __read_casa_image(infile, chunks, masks, history, verbose)
        except Exception as e:
            logging.warning('image format appears not to be casacore')
    try:
        return __read_fits_image(infile, chunks, masks, history, verbose)
    except Exception as e:
        logging.warning(f'image format appears not to be fits {e.args}')
    try:
        return __xds_from_zarr(infile)
    except Exception as e:
        logging.warning(f'image format appears not to be zarr {e.args}')
    raise RuntimeError('Unrecognized image format')


def load_image_block(infile:str, block_des: dict={}) -> xr.Dataset:
    """Load an image block (subimage) into memory
    :param infile: Path to the input image, currently only casacore images are supported
    :type infile: str, required
    :param block_des: The description of data to return, supported keys are time,
        pol, freq, l (or u if apeture image), m (or v if apeture image) a
        missing key indicates to return the entire axis length for that
        dimension. Values can be non-negative integers or slices. Slicing
        behaves as numpy slicing does, that is the start pixel is included in
        the selection, and the end pixel is not. An empty dictionary (the
        default) indicates that the entire image should be returned.
    :type block_des: dict
    :return: dataset with all numpy arrays, dask arrays cannot be used in the
        implementation of this function since this function will be called in a
        delayed context.
    :rtype: xr.Dataset, all contained arrays must be numpy arrays, not dask arrays
    """
    return __load_casa_image_block(infile, block_des)


def write_image(xds:xr.Dataset, imagename:str, out_format:str='casa') -> None:
    """Convert xds image to CASA image
    :param xds: XDS to convert
    :type xds: xr.Dataset, required
    :param imagename: path to output CASA image
    :type imagename: str
    :param out_format: format of output image, currently only 'casa' is supported
    :type out_format: str
    """
    my_format = out_format.lower()
    if my_format == 'casa':
        __xds_to_casa_image(xds, imagename)
    elif my_format == 'zarr':
        __xds_to_zarr(xds, imagename)
    else:
        raise Exception(f'Writing to format {out_format} is not supported')

