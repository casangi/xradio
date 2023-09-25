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
import pkg_resources
import shutil
import unittest
import xarray as xr
import copy


class ImageBase(unittest.TestCase):


    __imname : str = 'inp.im'
    __outname : str = 'out.im'
    __xds = None
    __exp_vals = {
        'dec_unit': 'rad', 'freq_cdelt': 1000, 'freq_crpix': 20,
        'freq_nativetype': 'FREQ',
        'freq_system': 'LSRK', 'freq_unit': 'Hz',
        'freq_waveunit': 'mm', 'image_type': 'Intensity',
        'ra_unit': 'rad', 'stokes': ['I', 'Q', 'U', 'V'],
        'time_format': 'MJD', 'time_refer': 'UTC', 'time_unit': 'd',
        'unit': 'Jy/beam', 'vel_type': 'RADIO', 'vel_unit': 'm/s'
    }


    __exp_attrs = {}
    __exp_attrs['direction'] = {
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
    # TODO make a more intresting beam
    __exp_attrs['beam'] = None
    __exp_attrs['obsdate'] = {
        'type': 'epoch',
        'refer': 'UTC',
        'm0': {
            'value': 51544.00000000116,
            'unit': 'd'
        }
    }
    __exp_attrs['observer'] = 'Karl Jansky'
    __exp_attrs['description'] = None
    __exp_attrs['active_mask'] = 'mask0'
    __exp_attrs['object_name'] = ''
    __exp_attrs['pointing_center'] = {
        'value': np.array([0,0]), 'initial': True,
    }
    __exp_attrs['telescope'] = {
        'name': 'ALMA', 'position': {
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
    __exp_attrs['user'] = {}
    __exp_attrs['history'] = None


    __ran_measures_code = False


    @classmethod
    def setUpClass(cls):
        raise Exception('USER ' + os.environ['USER'])
        if not cls.__ran_measures_code:
            casa_data_dir = pkg_resources.resource_filename('casadata',  '__data__')
            rc_file = open(os.path.expanduser("~/.casarc"),  "a+")   # append mode
            rc_file.write("\nmeasures.directory: "  +  casa_data_dir)
            rc_file.close()
            cls.__ran_measures_code = True
        cls.__make_image()


    @classmethod
    def tearDownClass(cls):
        for f in [cls.__imname, cls.__outname,]:
            if os.path.exists(f):
                if os.path.isdir(f):
                    shutil.rmtree(f)
                else:
                    os.remove(f)


    @classmethod
    def __make_image(cls):
        shape : list[int] = [10, 4, 20, 30]
        mask : np.ndarray = np.array(
            [ i % 3 == 0 for i in range(np.prod(shape)) ], dtype=bool
        ).reshape(shape)
        pix : np.ndarray = np.array([ range(np.prod(shape)) ], dtype=np.float64).reshape(shape)
        masked_array = ma.masked_array(pix, mask)
        im : casacore.images.image = casacore.images.image(cls.__imname, shape=shape)
        im.put(masked_array)
        shape = im.shape()
        del im
        t = casacore.tables.table(cls.__imname, readonly=False)
        t.putkeyword('units', 'Jy/beam')
        t.close()
        t = casacore.tables.table(os.sep.join([cls.__imname, 'logtable']), readonly=False)
        t.addrows()
        t.putcell('MESSAGE', 0, 'HELLO FROM EARTH again')
        t.flush()
        t.close()
        cls.__xds = read_image(cls.__imname, {'freq': 5})
        write_image(cls.__xds, cls.__outname, out_format='casa')


    def imname(self):
        return self.__imname


    @classmethod
    def xds(self):
        return self.__xds


    def outname(self):
        return self.__outname


    def exp_attrs(self):
        return self.__exp_attrs


    def dict_equality(self, dict1, dict2, dict1_name, dict2_name, exclude_keys=[]):
        self.assertEqual(
            dict1.keys(), dict2.keys(),
            f'{dict1_name} has different keys than {dict2_name}:'
            f'\n{dict1.keys()} vs\n {dict2.keys()}'
        )
        for k in dict1.keys():
            if k not in exclude_keys:
                if isinstance(dict1[k], dict) and isinstance(dict2[k], dict):
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
                        dict1[k], dict2[k],
                        f'{dict1_name}[{k}] != {dict2_name}[{k}]:\n'
                        + f'{dict1[k]} vs\n{dict2[k]}'
                    )


    def compare_sky_mask(self, xds:xr.Dataset):
        """Compare got sky and mask values to expected values"""
        ev = self.__exp_vals
        self.assertTrue(
            xds.sky.attrs['image_type'] == ev['image_type'],
            'Wrong image type'
        )
        self.assertTrue(
            xds.sky.attrs['unit'] == ev['unit'], 'Wrong unit'
        )
        got_data = da.squeeze(da.transpose(xds.sky, [2, 1, 4, 3, 0]), 4)
        got_mask = da.squeeze(da.transpose(xds.mask0, [2, 1, 4, 3, 0]), 4)
        if 'sky_array' not in ev:
            im = casacore.images.image(self.imname())
            ev['sky'] = im.getdata()
            # getmask returns the negated value of the casa image mask, so True
            # has the same meaning as it does in xds.mask0
            ev['mask0'] = im.getmask()
            ev['sum'] = im.statistics()['sum'][0]
        self.assertTrue(
            (got_data == ev['sky']).all(), 'pixel values incorrect'
        )
        self.assertTrue(
            (got_mask == ev['mask0']).all(), 'mask values incorrect'
        )
        got_ma = da.ma.masked_array(xds.sky, xds.mask0)
        self.assertEqual(
            da.sum(got_ma), ev['sum'], 'Incorrect value for sum'
        )


    def compare_time(self, xds: xr.Dataset):
        ev = self.__exp_vals
        if 'time' not in ev:
            im = casacore.images.image(self.imname())
            coords = im.coordinates().dict()['obsdate']
            ev['time'] = coords['m0']['value']
        got_vals = xds.time
        self.assertEqual(got_vals, ev['time'], 'Incorrect time axis values')
        self.assertEqual(
            xds.time.attrs['format'], ev['time_format'],
            'Incoorect time axis format'
        )
        self.assertEqual(
            xds.time.attrs['refer'], ev['time_refer'],
            'Incoorect time axis refer'
        )
        self.assertEqual(
            xds.time.attrs['unit'], ev['time_unit'], 'Incoorect time axis unitt'
        )


    def compare_pol(self, xds:xr.Dataset):
        self.assertTrue(
            (xds.pol == self.__exp_vals['stokes']).all(),
            'Incorrect pol values'
        )


    def compare_freq(self, xds:xr.Dataset):
        ev = self.__exp_vals
        if 'freq' not in ev:
            im = casacore.images.image(self.imname())
            sd = im.coordinates().dict()['spectral2']
            ev['freq'] = []
            for chan in range(10):
                ev['freq'].append(im.toworld([chan,0,0,0])[0])
            ev['freq_conversion'] = sd['conversion']
            ev['restfreq'] = sd['restfreq']
            ev['restfreqs'] = sd['restfreqs']
            ev['freq_crval'] = sd['wcs']['crval']
        self.assertTrue((xds.freq == ev['freq']).all(), 'Incorrect frequencies')
        self.assertEqual(
            xds.freq.attrs['conversion'], ev['freq_conversion'],
            (
                f'Incorrect frquency conversion. Got {xds.freq.attrs["conversion"]}. '
                + 'Exprected {ev["freq_conversion"'
            )
        )
        self.assertEqual(
            xds.freq.attrs['restfreq'], ev['restfreq'],
            'Incorrect rest frequency'
        )
        self.assertTrue(
            (xds.freq.attrs['restfreqs'] == ev['restfreqs']).all(),
            'Incorrect rest frequencies'
        )
        self.assertEqual(
            xds.freq.attrs['system'], ev['freq_system'],
            'Incorrect frequency system'
        )
        self.assertEqual(
            xds.freq.attrs['unit'], ev['freq_unit'],
            'Incorrect frequency unit'
        )
        self.assertEqual(
            xds.freq.attrs['wave_unit'], ev['freq_waveunit'],
            'Incorrect wavelength unit'
        )
        self.assertEqual(
            xds.freq.attrs['wcs']['cdelt'], ev['freq_cdelt'],
            'Incorrect frequency crpix'
        )
        self.assertEqual(
            xds.freq.attrs['wcs']['crval'], ev['freq_crval'],
            'Incorrect frequency crpix'
        )


    def compare_vel_axis(self, xds:xr.Dataset):
        ev = self.__exp_vals
        if 'vel' not in ev:
            im = casacore.images.image(self.imname())
            freqs = []
            for chan in range(10):
                freqs.append(im.toworld([chan,0,0,0])[0])
            freqs = np.array(freqs)
            spec_coord = casacore.images.coordinates.spectralcoordinate(
                im.coordinates().dict()['spectral2']
            )
            rest_freq = spec_coord.get_restfrequency()
            ev['vel'] = (1 - freqs/rest_freq) * 299792458
        self.assertTrue((xds.vel == ev['vel']).all(), 'Incorrect velocities')
        self.assertEqual(
            xds.vel.attrs['unit'], ev['vel_unit'], 'Incoorect velocity unit'
        )
        self.assertEqual(
            xds.vel.attrs['doppler_type'], ev['vel_type'],
            'Incoorect velocity type'
        )


    def compare_ra_dec(self, xds:xr.Dataset):
        ev = self.__exp_vals
        if 'ra' not in ev:
            im = casacore.images.image(self.imname())
            shape = im.shape()
            dd = im.coordinates().dict()['direction0']
            ev['ra'] = np.zeros([shape[3], shape[2]])
            ev['dec'] = np.zeros([shape[3], shape[2]])
            for i in range(shape[3]):
                for j in range(shape[2]):
                    w = im.toworld([0, 0, j, i])
                    ev['ra'][i][j] = w[3]
                    ev['dec'][i][j] = w[2]
            f = np.pi/180/60
            ev['ra'] *= f
            ev['dec'] *= f
            ev['ra_crval'] = dd['crval'][0]*f
            ev['ra_cdelt'] = dd['cdelt'][0]*f
            ev['dec_crval'] = dd['crval'][1]*f
            ev['dec_cdelt'] = dd['cdelt'][1]*f
        self.assertTrue(
            np.allclose(xds.right_ascension, ev['ra'], atol=1e-15),
            'Incorrect RA values'
        )
        self.assertTrue(
            np.allclose(xds.declination, ev['dec'], atol=1e-15),
            'Incorrect Dec values'
        )
        self.assertEqual(
            xds.right_ascension.attrs['unit'], ev['ra_unit'],
            'Incorrect RA unit'
        )
        self.assertEqual(
            xds.right_ascension.attrs['wcs']['crval'], ev['ra_crval'],
            'Incorrect RA crval'
        )
        self.assertEqual(
            xds.right_ascension.attrs['wcs']['cdelt'], ev['ra_cdelt'],
            'Incorrect RA cdelt'
        )
        self.assertEqual(
            xds.declination.attrs['unit'], ev['dec_unit'],
            'Incorrect Dec unit'
        )
        self.assertEqual(
            xds.declination.attrs['wcs']['crval'], ev['dec_crval'],
            'Incorrect Dec crval'
        )
        self.assertEqual(
            xds.declination.attrs['wcs']['cdelt'], ev['dec_cdelt'],
            'Incorrect Dec cdelt'
        )


    def compare_attrs(self, xds):
        self.dict_equality(
            self.xds().attrs, self.exp_attrs(), 'Got attrs', 'Expected attrs', ['history']
        )
        self.assertTrue(
            isinstance(self.xds().attrs['history'], xr.core.dataset.Dataset),
            'Incorrect type for history data'
        )


    def compare_image_block(self, imagename):
        xds = load_image_block(
            imagename,
            {
                'l': slice(2, 10), 'm': slice(3, 15), 'pol': slice(0, 1),
                'freq': slice(0,4)
            }
        )
        self.assertEqual(xds.sky.shape, (1, 1, 4, 8, 12), 'Wrong block shape')
        big_xds = self.__xds
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
        # all coordinates and data variables should be numpy arrays when loading an
        # image section
        merged_dict = {**xds.coords, **xds.data_vars}
        for k, v in merged_dict.items():
            self.assertTrue(
                isinstance(v.data, np.ndarray),
                f'Wrong type for coord or data value {k}, got {type(v)}, must be a numpy.ndarray'
            )


class casa_image_to_xds_test(ImageBase):
    """
    test casa_image_to_xds
    """


    def test_xds_pixel_values(self):
        """Test xds has correct pixel values"""
        self.compare_sky_mask(self.xds())


    def test_xds_time_axis(self):
        """Test values and attributes on the time axis"""
        self.compare_time(self.xds())


    def test_xds_pol_axis(self):
        """Test xds has correct stokes values"""
        self.compare_pol(self.xds())


    def test_xds_freq_axis(self):
        """Test xds has correct frequency values and metadata"""
        self.compare_freq(self.xds())


    def test_xds_vel_axis(self):
        """Test xds has correct velocity values and metadata"""
        self.compare_vel_axis(self.xds())


    def test_xds_ra_dec_axis(self):
        """Test xds has correct RA and Dec values and attributes"""
        self.compare_ra_dec(self.xds())


    def test_xds_attrs(self):
        """Test xds level attributes"""
        self.compare_attrs(self.xds())


    def test_get_img_ds_block(self):
        self.compare_image_block(self.imname())


class casacore_to_xds_to_casacore(ImageBase):
    """test casacore -> xds -> casacore round trip"""


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


class xds_to_zarr_to_xds_test(ImageBase):
    """
    test xds -> zarr -> xds round trip
    """

    __zarr_store:str = 'out.zarr'


    @classmethod
    def setUpClass(cls):
        # by default, subclass setUpClass() method is called before super class',
        # so we must explicitly call the super class' method here to create the
        # xds which is located in the super class
        super().setUpClass()
        write_image(cls.xds(), cls.__zarr_store, out_format='zarr')
        cls.__zds = read_image(cls.__zarr_store)


    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        for f in [cls.__zarr_store,]:
            if os.path.exists(f):
                if os.path.isdir(f):
                    shutil.rmtree(f)
                else:
                    os.remove(f)


    def setUp(self):
        pass


    def tearDown(self):
        pass


    def test_xds_pixel_values(self):
        """Test xds has correct pixel values"""
        self.compare_sky_mask(self.__zds)


    def test_xds_time_vals(self):
        """Test xds has correct time axis values"""
        self.compare_time(self.__zds)


    def test_xds_pol_axis(self):
        """Test xds has correct stokes values"""
        self.compare_pol(self.__zds)


    def test_xds_freq_axis(self):
        """Test xds has correct frequency values and metadata"""
        self.compare_freq(self.__zds)


    def test_xds_vel_axis(self):
        """Test xds has correct velocity values and metadata"""
        self.compare_vel_axis(self.__zds)


    def test_xds_ra_dec_axis(self):
        """Test xds has correct RA and Dec values and attributes"""
        self.compare_ra_dec(self.__zds)


    def test_xds_attrs(self):
        """Test xds level attributes"""
        self.compare_attrs(self.__zds)


    def test_get_img_ds_block(self):
        self.compare_image_block(self.__zarr_store)


if __name__ == '__main__':
    unittest.main()
