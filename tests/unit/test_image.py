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

import casacore.images, casacore.tables
from xradio.image import load_image_block, read_image, write_image
import dask.array.ma as dma
import dask.array as da
import numpy as np
import numpy.ma as ma
import os
import shutil
import unittest
import xarray
import copy


class ImageBase(unittest.TestCase):


    __imname : str = 'inp.im'
    __outname : str = 'out.im'
    __xds = None
    __zarr_store : str = 'out.zarr'

    @classmethod
    def setUpClass(cls):
        cls.__make_image()


    @classmethod
    def tearDownClass(cls):
        for f in [cls.__imname, cls.__outname, cls.__zarr_store]:
            if os.path.exists(f):
                if os.path.isdir(f):
                    shutil.rmtree(f)
                else:
                    os.remove(f)


    @classmethod
    def __make_image(cls):
        shape : list[int] = [10, 4, 20, 30]
        mask : np.ndarray = np.array([ i % 3 == 0 for i in range(np.prod(shape)) ], dtype=bool).reshape(shape)
        pix : np.ndarray = np.array([ range(np.prod(shape)) ], dtype=np.float64).reshape(shape)
        masked_array = ma.masked_array(pix, mask)
        im : casacore.images.image = casacore.images.image(cls.__imname, shape=shape)
        im.put(masked_array)
        shape = im.shape()
        del im
        t = casacore.tables.table(cls.__imname, readonly=False)
        t.putkeyword('units', 'Jy/beam')
        t.close()
        cls.__xds = read_image(cls.__imname, chunks=shape[::-1])
        write_image(cls.__xds, cls.__outname, out_format='casa')
        write_image(cls.__xds, cls.__zarr_store, out_format='zarr')


    def imname(self):
        return self.__imname


    def xds(self):
        return self.__xds


    def outname(self):
        return self.__outname


    def dict_equality(self, dict1, dict2, dict1_name, dict2_name, exclude_keys=[]):
        self.assertEqual(
            dict1.keys(), dict2.keys(),
            f'{dict1_name} has different keys than {dict2_name}'
        )
        for k in dict1.keys():
            if k not in exclude_keys:
                if isinstance(dict1[k], dict):
                    self.dict_equality(
                        dict1[k], dict2[k], f'{dict1_name}[{k}]', f'{dict2_name}[{k}]'
                    )
                elif isinstance(dict1[k], np.ndarray):
                    self.assertTrue(
                        (dict1[k] == dict2[k]).all(),
                        f'{dict1_name}[{k}] != {dict2_name}[{k}]'
                    )
                else:
                    self.assertEqual(
                        dict1[k], dict2[k], f'{dict1_name}[{k}] != {dict2_name}[{k}]'
                    )


class casa_image_to_xds_test(ImageBase):
    """
    test casa_image_to_xds
    """


    def test_xds_pixel_values(self):
        """Test xds has correct pixel values"""
        im = casacore.images.image(self.imname())
        stats = im.statistics()
        exp_data = im.getdata()
        # getmask returns the negated value of the casa image mask, so True
        # has the same meaning as it does in xds.mask0
        exp_mask = im.getmask()
        xds = self.xds()
        got_data = da.squeeze(da.transpose(xds.sky, [2, 1, 4, 3, 0]), 4)
        self.assertTrue(xds.sky.attrs['image_type'] == 'Intensity', 'Wrong image type')
        got_unit = xds.sky.attrs['unit']
        self.assertTrue(got_unit == 'Jy/beam', f'Wrong image unit {got_unit}')
        got_mask = da.squeeze(da.transpose(xds.mask0, [2, 1, 4, 3, 0]), 4)
        self.assertTrue((got_data == exp_data).all(), 'pixel values incorrect')
        self.assertTrue((got_mask == exp_mask).all(), 'mask values incorrect')
        got_ma = da.ma.masked_array(xds.sky, xds.mask0)
        self.assertEqual(da.sum(got_ma), stats['sum'][0], 'Incorrect value for sum')


    def test_xds_time_axis(self):
        """Test values and attributes on the time axis"""
        im = casacore.images.image(self.imname())
        coords = im.coordinates().dict()['obsdate']
        exp_vals = coords['m0']['value']
        exp_attrs = {}
        exp_attrs['unit'] = coords['m0']['unit']
        exp_attrs['refer'] = coords['refer']
        exp_attrs['format'] = 'MJD'
        xds = self.xds()
        got_vals = xds.time
        self.assertEqual(got_vals, exp_vals, 'Incorrect time axis values')
        got_attrs = xds.time.attrs
        self.assertEqual(got_attrs, exp_attrs, 'Incoorect time axis attributes')


    def test_xds_pol_axis(self):
        """Test xds has correct stokes values"""
        im = casacore.images.image(self.imname())
        stokes_coord = im.coordinates().dict()['stokes1']
        exp_vals = stokes_coord['stokes']
        xds = self.xds()
        got_vals = xds.pol
        self.assertTrue((got_vals == exp_vals).all(), 'Incorrect pol values')


    def test_xds_freq_axis(self):
        """Test xds has correct frequency values and metadata"""
        im = casacore.images.image(self.imname())
        sd = im.coordinates().dict()['spectral2']
        exp_freq = []
        for chan in range(10):
            exp_freq.append(im.toworld([chan,0,0,0])[0])
        native_types = ['FREQ', 'VRAD', 'VOPT', 'BETA', 'WAVE', 'AWAV']
        exp_attrs = {}
        exp_attrs['conversion'] = sd['conversion']
        exp_attrs['native_type'] = native_types[sd['nativeType']]
        exp_attrs['restfreq'] = sd['restfreq']
        exp_attrs['restfreqs'] = sd['restfreqs']
        exp_attrs['system'] = sd['system']
        exp_attrs['unit'] = sd['unit']
        exp_attrs['wave_unit'] = sd['waveUnit']
        for k in ['crpix', 'ctype', 'pc']:
            del sd['wcs'][k]
        exp_attrs['wcs'] = sd['wcs']

        xds = self.xds()
        got_freq = xds.freq
        self.assertTrue((got_freq == exp_freq).all())
        got_attrs = xds.freq.attrs
        self.assertEqual(got_attrs, exp_attrs, 'Incorrect freq axis attributes')


    def test_xds_vel_axis(self):
        """Test xds has correct velocity values and metadata"""
        im = casacore.images.image(self.imname())
        freqs = []
        for chan in range(10):
            freqs.append(im.toworld([chan,0,0,0])[0])
        freqs = np.array(freqs)
        spec_coord = casacore.images.coordinates.spectralcoordinate(
            im.coordinates().dict()['spectral2']
        )
        rest_freq = spec_coord.get_restfrequency()
        exp_vels = (1 - freqs/rest_freq) * 299792458
        exp_attrs = {}
        exp_attrs['unit'] = 'm/s'
        doppler_types = ['RADIO', 'Z', 'RATIO', 'BETA', 'GAMMA']
        exp_attrs['doppler_type'] = doppler_types[
            im.coordinates().dict()['spectral2']['velType']
        ]
        xds = self.xds()
        got_vels = xds.vel
        self.assertTrue((got_vels == exp_vels).all())
        got_attrs = xds.vel.attrs
        self.assertEqual(got_attrs, exp_attrs, 'Incoorect vel axis attributes')


    def test_xds_ra_dec_axis(self):
        """Test xds has correct RA and Dec values and attributes"""
        im = casacore.images.image(self.imname())
        shape = im.shape()
        dd = im.coordinates().dict()['direction0']
        exp_ra_vals = np.zeros([shape[3], shape[2]])
        exp_dec_vals = np.zeros([shape[3], shape[2]])
        for i in range(shape[3]):
            for j in range(shape[2]):
                w = im.toworld([0, 0, j, i])
                exp_ra_vals[i][j] = w[3]
                exp_dec_vals[i][j] = w[2]
        f = np.pi/180/60
        exp_ra_vals *= f
        exp_dec_vals *= f
        exp_ra_attrs = {}
        exp_ra_attrs['unit'] = 'rad'
        exp_ra_attrs['wcs'] = {}
        exp_ra_attrs['wcs']['crval'] = dd['crval'][0]*f
        exp_ra_attrs['wcs']['cdelt'] = dd['cdelt'][0]*f
        exp_dec_attrs = {}
        exp_dec_attrs['unit'] = 'rad'
        exp_dec_attrs['wcs'] = {}
        exp_dec_attrs['wcs']['crval'] = dd['crval'][1]*f
        exp_dec_attrs['wcs']['cdelt'] = dd['cdelt'][1]*f

        xds = self.xds()
        got_ra_vals = xds.right_ascension
        got_dec_vals = xds.declination
        z = np.abs(got_ra_vals - exp_ra_vals)
        self.assertTrue(
            np.allclose(got_ra_vals, exp_ra_vals, atol=1e-15),
            'Incorrect RA values'
        )
        self.assertTrue(
            np.allclose(got_dec_vals, exp_dec_vals, atol=1e-15),
            'Incorrect Dec values'
        )
        got_ra_attrs = xds.right_ascension.attrs
        got_dec_attrs = xds.declination.attrs
        self.assertEqual(got_ra_attrs, exp_ra_attrs, 'Incorrect RA attributes')
        self.assertEqual(got_dec_attrs, exp_dec_attrs, 'Incorrect Dec attributes')


    def test_xds_attrs(self):
        """Test xds level attributes"""
        got_attrs = self.xds().attrs
        exp_direction = {
            'system': 'FK5',
            'equinox': 'J2000',
            'conversion_system': 'FK5',
            'conversion_equinox': 'J2000',
            'latpole': {'value': 0.0, 'unit': 'rad'},
            'longpole': {'value': 3.141592653589793, 'unit': 'rad'},
            'pc': np.array([[1., 0.], [0., 1.]]),
            'projection_parameters': np.array([0., 0.]),
            'projection': 'SIN'
        }
        for k in exp_direction:
            got = got_attrs['direction'][k]
            if isinstance(got, np.ndarray):
                self.assertTrue(
                    (got == exp_direction[k]).all(),
                    f'Incorrect xds level direction attribute {k}'
                )
            else:
                print(f'k {k} got {got}')
                self.assertEqual(
                    got, exp_direction[k],
                    f'Incorrect xds level direction attribute {k}'
                )
        expec = {}
        # TODO make a more intresting beam
        expec['beam'] = None
        expec['obsdate'] = {
            'type': 'epoch',
            'refer': 'UTC',
            'm0': {
                'value': 51544.00000000116,
                'unit': 'd'
            }
        }
        expec['observer'] = 'Karl Jansky'
        expec['description'] = None
        for k in expec:
            self.assertEqual(
                got_attrs[k], expec[k],
                f'Incorrect xds level attribute {k}. Got {got_attrs[k]}. Expected {expec[k]}'
            )
        self.assertEqual(got_attrs['active_mask'], 'mask0', 'Incorrect active_mask')
        self.assertEqual(got_attrs['object_name'], '', 'Incorrect object_name')
        self.assertTrue(
            (got_attrs['pointing_center']['value'] == np.array([0,0])).all(),
            'Incorrect pointing center'
        )
        self.assertTrue(
            got_attrs['pointing_center']['initial'],
            'Incorrect initial pointing center value'
        )
        expec_tscope = {
            'name': 'ALMA',
            'position': {
                'type': 'position',
                'refer': 'ITRF',
                'm2': {
                    'value': 6379946.01326443,
                    'unit': 'm'
                },
                'm1': {
                    'unit': 'rad',
                    'value': -0.3994149869262738
                },
                'm0': {
                    'unit': 'rad',
                    'value': -1.1825465955049892
                }
            }
        }
        self.assertEqual(
            got_attrs['telescope'], expec_tscope,
            'Incorrect telescope attribute(s)'
        )
        self.assertTrue(
            isinstance(self.xds().history, xarray.core.dataset.Dataset),
            'Incorrect type for history data'
        )


    def test_get_img_ds_block(self):
        xds = load_image_block(
            self.imname(),
            {
                'l': slice(2, 10), 'm': slice(3, 15), 'pol': 0,
                'freq': slice(0,4)
            }
        )
        self.assertEqual(xds.sky.shape, (1, 1, 4, 8, 12), 'Wrong block shape')
        big_xds = self.xds()
        self.assertTrue(
            (xds.sky == big_xds.sky[:,0:1, 0:4, 2:10, 3:15]).all(),
            'Wrong block sky array'
        )
        self.assertTrue(
            (xds.mask0 == big_xds.mask0[:,0:1, 0:4, 2:10, 3:15]).all(),
            'Wrong block mask0 array'
        )
        self.dict_equality(
            xds.attrs, big_xds.attrs, 'block xds', 'main xds', ['history']
        )
        for c in ('time', 'freq', 'pol', 'vel', 'right_ascension', 'declination'):
            self.dict_equality(
                xds[c].attrs, big_xds[c].attrs, f'block xds {c}', 'main xds {c}'
            )
        self.assertEqual(
            xds.time, big_xds.time, 'Incorrect time coordinate value'
        )
        self.assertEqual(
            xds.pol, big_xds.pol[0:1], 'Incorrect pol coordinate value'
        )
        self.assertTrue(
            (xds.freq == big_xds.freq[0:4]).all(),
            'Incorrect freq coordinate values'
        )
        self.assertTrue(
            (xds.vel == big_xds.vel[0:4]).all(),
            'Incorrect vel coordinate values'
        )
        self.assertTrue(
            (xds.right_ascension == big_xds.right_ascension[2:10, 3:15]).all(),
            'Incorrect right ascension coordinate values'
        )
        self.assertTrue(
            (xds.declination == big_xds.declination[2:10, 3:15]).all(),
            'Incorrect declination coordinate values'
        )


class casa_xds_to_image_test(ImageBase):
    """
    test casa_xds_to_image_test
    """

    def test_pixels_and_mask(self):
        """Test pixel values are consistent"""
        im1 = casacore.images.image(self.imname())
        im2 = casacore.images.image(self.outname())
        self.assertTrue(
            (im1.getdata() == im2.getdata()).all(),
            'Incorrect pixel values'
        )
        self.assertTrue(
            (im1.getmask() == im2.getmask()).all(),
            'Incorrect mask values'
        )


if __name__ == '__main__':
    unittest.main()
