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
from ._util.image_factory import __make_empty_sky_image
from ._util.zarr import __xds_to_zarr, __xds_from_zarr
import warnings, time, os, logging
import numpy as np
import astropy.wcs
import xradio
from astropy import units as u
from typing import Union
import xarray as xr

warnings.filterwarnings('ignore', category=FutureWarning)


def read_image(infile:str, chunks:dict={}, verbose:bool=False) -> xr.Dataset:
    """
    Read Image (currently only supports casacore images) or zarr to ngCASA image format
    ngCASA image spec is located at
    https://docs.google.com/spreadsheets/d/1WW0Gl6z85cJVPgtdgW4dxucurHFa06OKGjgoK8OREFA/edit#gid=1719181934
    :param infile: Path to the input CASA image
    :type infile: str, required
    :param chunks: The desired dask chunk size. Only applicable for casacore and fits images.
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
    :type chunks: dict, required
    :param verbose: emit debugging messages? Default is False.
    :type verbose: bool, optional
    :return: xr.Dataset image that conforms to the ngCASA image spec
    :rtype: xr.Dataset
    """
    do_casa = True
    emsgs = []
    try:
        from ._util.casacore import __read_casa_image
    except Exception as e:
        emsgs.append(
            'python-casacore could not be imported, will not try to '
            f'read as casacore image: {e.args}'
        )
        do_casa = False
    if do_casa:
        try:
            return __read_casa_image(infile, chunks, verbose=verbose)
        except Exception as e:
            emsgs.append(f'image format appears not to be casacore: {e.args}')
    try:
        return __read_fits_image(infile, chunks, verbose)
    except Exception as e:
        emsgs.append(f'image format appears not to be fits {e.args}')
    try:
        return __xds_from_zarr(infile, True)
    except Exception as e:
        emsgs.append(f'image format appears not to be zarr {e.args}')
    emsgs.insert(0, f'Unrecognized image format. Supported types are casacore and zarr.\n')
    raise RuntimeError('\n'.join(emsgs))


def load_image_block(infile:str, block_des:dict={}) -> xr.Dataset:
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
    do_casa = True
    emsgs = []
    try:
        from ._util.casacore import __read_casa_image
    except Exception as e:
        emsgs.append(
            'python-casacore could not be imported, will not try to '
            f'read as casacore image: {e.args}'
        )
        do_casa = False
    if do_casa:
        try:
            return __load_casa_image_block(infile, block_des)
        except Exception as e:
            emsgs.append(f'image format appears not to be casacore: {e.args}')
    """
    try:
        return __read_fits_image(infile, chunks, masks, history, verbose)
    except Exception as e:
        emsgs.append(f'image format appears not to be fits {e.args}')
    """
    try:
        return __xds_from_zarr(infile, False).isel(block_des)
    except Exception as e:
        emsgs.append(f'image format appears not to be zarr {e.args}')
    emsgs.insert(0, f'Unrecognized image format. Supported formats are casacore and zarr.\n')
    raise RuntimeError('\n'.join(emsgs))


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
        raise Exception(
            f'Writing to format {out_format} is not supported. '
            'out_format must be either "casa" or "zarr".'
        )


def make_empty_sky_image(
    xds:xr.Dataset, phase_center:Union[list, np.ndarray],
	image_size:Union[list, np.ndarray], cell_size:Union[list, np.ndarray],
	chan_coords:Union[list, np.ndarray],
	pol_coords:Union[list, np.ndarray], time_coords:Union[list, np.ndarray],
	direction_reference:str='FK5', projection:str='SIN',
	spectral_reference:str='lsrk'
) -> xr.Dataset:
    """
    Create an image xarray.Dataset with only coordinates (no datavariables).
    The image dimensionality is either:
        l, m, time, chan, pol

    Parameters
    ----------
    xds : xarray.Dataset
        Empty dataset (dataset = xarray.Dataset()) to be modified
    phase_center : array of float, length = 2, units = rad
        Image phase center.
    image_size : array of int, length = 2, units = rad
        Number of x and y axis pixels in image.
    cell_size : array of float, length = 2, units = rad
        Cell size of x and y axis pixels in image.
    chan_coords : list or np.ndarray
        The center frequency in Hz of each image channel.
    pol_coords : list or np.ndarray
        The polarization code for each image polarization.
    time_coords : list or np.ndarray
        The time for each temporal plane in MJD.
    direction_reference : str, default = 'FK5'
    projection : str, default = 'SIN'
    spectral_reference : str, default = 'lsrk'
    Returns
    -------
    xarray.Dataset
    """
    return __make_empty_sky_image(
        xds, phase_center, image_size, cell_size, chan_coords,
	    pol_coords, time_coords, direction_reference, projection,
	    spectral_reference
    )

