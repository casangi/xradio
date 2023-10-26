import casacore.images, casacore.tables
from xradio.image import (
    load_image, make_empty_sky_image, read_image, write_image
)
import dask.array.ma as dma
import dask.array as da
import numbers
import numpy as np
import numpy.ma as ma
import os
import pkg_resources
import shutil
import sys
import unittest
import xarray as xr
import copy


class ImageBase(unittest.TestCase):


    __imname: str = 'inp.im'
    __outname: str = 'out.im'
    __infits:str = 'inp.fits'
    __xds = None
    __exp_vals:dict = {
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
        # there seems to be a casacore bug here that changing either the
        # crval or pointingcenter will also change the latpole when the
        # casacore image is reopened. As long as the xds gets the latpole
        # that the casacore image has is all we care about for testing
        'latpole': {'value': -40*np.pi/180, 'unit': 'rad'},
        'longpole': {'value': np.pi, 'unit': 'rad'},
        'pc': np.array([[1., 0.], [0., 1.]]),
        'projection_parameters': np.array([0., 0.]),
        'projection': 'SIN'
    }
    # TODO make a more intresting beam
    __exp_attrs['beam'] = None
    __exp_attrs['obsdate'] = {
        'time_scale': 'UTC',
        'value': 51544.00000000116,
        'unit': 'd',
        'format': 'MJD'
    }
    __exp_attrs['observer'] = 'Karl Jansky'
    __exp_attrs['description'] = None
    __exp_attrs['active_mask'] = 'mask0'
    __exp_attrs['object_name'] = ''
    __exp_attrs['pointing_center'] = {
        'value': np.array([6300, -2400]) * np.pi/180/60,
        'initial': True,
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
        if not cls.__ran_measures_code and os.environ['USER'] == 'runner':
            casa_data_dir = pkg_resources.resource_filename('casadata',  '__data__')
            rc_file = open(os.path.expanduser("~/.casarc"),  "a+")   # append mode
            rc_file.write("\nmeasures.directory: "  +  casa_data_dir)
            rc_file.close()
            cls.__ran_measures_code = True
        cls.__make_image()


    @classmethod
    def tearDownClass(cls):
        for f in [cls.__imname, cls.__outname, cls.__infits]:
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
        im:casacore.images.image = casacore.images.image(cls.__imname, shape=shape)
        im.put(masked_array)
        shape = im.shape()
        del im
        t = casacore.tables.table(cls.__imname, readonly=False)
        t.putkeyword('units', 'Jy/beam')
        csys = t.getkeyword('coords')
        pc = np.array([6300, -2400])
        # change pointing center
        csys['direction0']['crval'] = pc
        csys['pointingcenter']['value'] = pc * np.pi/180/60
        t.putkeyword('coords', csys)
        t.close()
        t = casacore.tables.table(os.sep.join([cls.__imname, 'logtable']), readonly=False)
        t.addrows()
        t.putcell('MESSAGE', 0, 'HELLO FROM EARTH again')
        t.flush()
        t.close()
        im = casacore.images.image(cls.__imname)
        im.tofits(cls.__infits)
        del im
        cls.__xds = read_image(cls.__imname, {'frequency': 5})
        write_image(cls.__xds, cls.__outname, out_format='casa')


    def imname(self):
        return self.__imname


    @classmethod
    def infits(self):
        return self.__infits


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
                one = dict1[k]
                two = dict2[k]
                if (
                    isinstance(one, numbers.Number)
                    and isinstance(two, numbers.Number)
                ):
                    self.assertTrue(
                        np.isclose(one, two),
                        f'{dict1_name}[{k}] != {dict2_name}[{k}]:\n'
                        + f'{one} vs\n{two}'
                    )
                elif (
                    (isinstance(one, list) and isinstance(two, np.ndarray))
                    or (isinstance(one, np.ndarray) and isinstance(two, list))
                ):
                    self.assertTrue(
                        np.isclose(np.array(one), np.array(two)).all(),
                        f'{dict1_name}[{k}] != {dict2_name}[{k}]'
                    )
                else:
                    self.assertEqual(
                        type(dict1[k]), type(dict2[k]),
                        f'Types are different {dict1_name}[{k}] {type(dict1[k])} '
                        + f'vs {dict2_name}[{k}] {type(dict2[k])}'
                    )
                    if isinstance(dict1[k], dict) and isinstance(dict2[k], dict):
                        self.dict_equality(
                            dict1[k], dict2[k], f'{dict1_name}[{k}]', f'{dict2_name}[{k}]'
                        )
                    elif isinstance(one, np.ndarray):
                        if k == 'crpix':
                            self.assertTrue(
                                np.allclose(one, two, rtol=3e-5),
                                f'{dict1_name}[{k}] != {dict2_name}[{k}], {one} vs {two}'
                            )
                        else:
                            self.assertTrue(
                                np.allclose(one, two),
                                f'{dict1_name}[{k}] != {dict2_name}[{k}], {one} vs {two}'
                            )
                    else:
                        self.assertEqual(
                            dict1[k], dict2[k],
                            f'{dict1_name}[{k}] != {dict2_name}[{k}]:\n'
                            + f'{dict1[k]} vs\n{dict2[k]}'
                        )


    def compare_sky_mask(self, xds:xr.Dataset, fits=False):
        """Compare got sky and mask values to expected values"""
        ev = self.__exp_vals
        self.assertEqual(
            xds.sky.attrs['image_type'], ev['image_type'],
            'Wrong image type'
        )
        self.assertEqual(
            xds.sky.attrs['unit'], ev['unit'], 'Wrong unit'
        )
        self.assertEqual(
            xds.sky.chunksizes['frequency'], (5, 5), 'Incorrect chunksize'
        )
        self.assertEqual(
            xds.mask0.chunksizes['frequency'], (5, 5), 'Incorrect chunksize'
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
        if fits:
            self.assertTrue(
                not np.isnan(got_data == ev['sky']).all(), 'pixel values incorrect'
            )
        else:
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


    def compare_time(self, xds: xr.Dataset) -> None:
        ev = self.__exp_vals
        if 'time' not in ev:
            im = casacore.images.image(self.imname())
            coords = im.coordinates().dict()['obsdate']
            ev['time'] = coords['m0']['value']
        got_vals = xds.time
        self.assertEqual(got_vals, ev['time'], 'Incorrect time axis values')
        self.assertEqual(
            xds.coords['time'].attrs['format'], ev['time_format'],
            'Incoorect time axis format'
        )
        self.assertEqual(
            xds.coords['time'].attrs['time_scale'], ev['time_refer'],
            'Incoorect time axis refer'
        )
        self.assertEqual(
            xds.coords['time'].attrs['unit'], ev['time_unit'],
            'Incoorect time axis unitt'
        )


    def compare_polarization(self, xds:xr.Dataset) -> None:
        self.assertTrue(
            (xds.coords['polarization'] == self.__exp_vals['stokes']).all(),
            'Incorrect polarization values'
        )


    def compare_frequency(self, xds:xr.Dataset, fits=False):
        ev = self.__exp_vals
        if 'frequency' not in ev:
            im = casacore.images.image(self.imname())
            sd = im.coordinates().dict()['spectral2']
            ev['frequency'] = []
            for chan in range(10):
                ev['frequency'].append(im.toworld([chan,0,0,0])[0])
            ev['freq_conversion'] = copy.deepcopy(sd['conversion'])
            for k in ('direction', 'position', 'epoch'):
                del ev['freq_conversion'][k]['type']
            del ev['freq_conversion']['direction']['refer']
            ev['freq_conversion']['direction']['system'] = 'FK5'
            ev['freq_conversion']['direction']['equinox'] = 'J2000'
            ev['restfreq'] = sd['restfreq']
            ev['restfreqs'] = sd['restfreqs']
            ev['freq_crval'] = sd['wcs']['crval']
        if fits:
            self.assertTrue(
                np.isclose(xds.frequency, ev['frequency']).all(), 'Incorrect frequencies'
            )
            self.assertTrue(
                np.isclose(xds.frequency.attrs['restfreq'], ev['restfreq']),
                'Incorrect rest frequency'
            )
            self.assertTrue(
                np.isclose(
                    xds.frequency.attrs['wcs']['cdelt'], ev['freq_cdelt']
                ), 'Incorrect frequency cdelt'
            )
            self.assertTrue(
                np.isclose(
                    xds.frequency.attrs['wcs']['crval'],
                    ev['freq_crval']
                ), 'Incorrect frequency crpix'
            )
        else:
            self.assertTrue(
                np.isclose(
                    xds.frequency.attrs['restfreqs'][0],
                    ev['restfreqs'][0]
                ), 'Incorrect rest frequencies'
            )
            self.assertTrue(
                (xds.frequency == ev['frequency']).all(), 'Incorrect frequencies'
            )
            self.assertEqual(
                xds.frequency.attrs['restfreq'], ev['restfreq'],
                'Incorrect rest frequency'
            )
            self.assertTrue(
                (xds.frequency.attrs['restfreqs'] == ev['restfreqs']).all(),
                'Incorrect rest frequencies'
            )
            self.assertEqual(
                xds.frequency.attrs['wcs']['cdelt'], ev['freq_cdelt'],
                'Incorrect frequency cdelt'
            )
            self.assertEqual(
                xds.frequency.attrs['wcs']['crval'], ev['freq_crval'],
                'Incorrect frequency crpix'
            )
        self.assertEqual(
            xds.frequency.attrs['conversion'], ev['freq_conversion'],
            (
                f'Incorrect frquency conversion. Got {xds.frequency.attrs["conversion"]}. '
                + 'Exprected {ev["freq_conversion"'
            )
        )
        self.assertEqual(
            xds.frequency.attrs['system'], ev['freq_system'],
            'Incorrect frequency system'
        )
        self.assertEqual(
            xds.frequency.attrs['unit'], ev['freq_unit'],
            'Incorrect frequency unit'
        )
        self.assertEqual(
            xds.frequency.attrs['wave_unit'], ev['freq_waveunit'],
            'Incorrect wavelength unit'
        )


    def compare_vel_axis(self, xds:xr.Dataset, fits:bool=False):
        ev = self.__exp_vals
        if fits:
            # casacore has written optical velocities to FITS file,
            # even though the doppler type is RADIO in the casacore
            # image
            freqs = xds.coords['frequency'].values
            rest_freq = xds.coords['frequency'].attrs['restfreq']
            v_opt = (rest_freq/freqs - 1) * 299792458
            self.assertTrue(
                np.isclose(xds.vel, v_opt).all(),
                'Incorrect velocities'
            )
            self.assertEqual(
                xds.vel.attrs['doppler_type'], 'Z',
                'Incoorect velocity type'
            )
        else:
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
                xds.vel.attrs['doppler_type'], ev['vel_type'],
                'Incoorect velocity type'
            )
        self.assertEqual(
            xds.vel.attrs['unit'], ev['vel_unit'], 'Incoorect velocity unit'
        )


    def compare_ra_dec(self, xds:xr.Dataset, fits:bool=False) -> None:
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
        if fits:
            self.assertTrue(
                np.isclose(
                    xds.right_ascension.attrs['wcs']['crval'],
                    ev['ra_crval']
                ), 'Incorrect RA crval'
            )
            self.assertTrue(
                np.isclose(
                    xds.right_ascension.attrs['wcs']['cdelt'],
                    ev['ra_cdelt']
                ), 'Incorrect RA cdelt'
            )
            self.assertTrue(
                np.isclose(
                    xds.declination.attrs['wcs']['cdelt'],
                    ev['dec_cdelt']
                ), 'Incorrect Dec cdelt'
            )
        else:
            self.assertEqual(
                xds.right_ascension.attrs['wcs']['crval'], ev['ra_crval'],
                'Incorrect RA crval'
            )
            self.assertEqual(
                xds.right_ascension.attrs['wcs']['cdelt'], ev['ra_cdelt'],
                'Incorrect RA cdelt'
            )
            self.assertEqual(
                xds.declination.attrs['wcs']['cdelt'], ev['dec_cdelt'],
                'Incorrect Dec cdelt'
            )
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
            xds.declination.attrs['unit'], ev['dec_unit'],
            'Incorrect Dec unit'
        )
        self.assertEqual(
            xds.declination.attrs['wcs']['crval'], ev['dec_crval'],
            'Incorrect Dec crval'
        )


    def compare_attrs(self, xds:xr.Dataset, fits:bool=False):
        my_exp_attrs = copy.deepcopy(self.exp_attrs())
        if fits:
            # xds from fits do not have history yet
            del my_exp_attrs['history']
            my_exp_attrs['user']['comment'] = (
                'casacore non-standard usage: 4 LSD, '
                '5 GEO, 6 SOU, 7 GAL'
            )
        self.dict_equality(
            xds.attrs, my_exp_attrs, 'Got attrs',
            'Expected attrs', ['history']
        )
        if not fits:
            self.assertTrue(
                isinstance(xds.attrs['history'], xr.core.dataset.Dataset),
                'Incorrect type for history data'
            )


    def compare_image_block(self, imagename):
        xds = load_image(
            imagename,
            {
                'l': slice(2, 10), 'm': slice(3, 15), 'polarization': slice(0, 1),
                'frequency': slice(0,4)
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
        for c in ('time', 'frequency', 'polarization', 'vel', 'right_ascension', 'declination'):
            self.dict_equality(
                xds[c].attrs, big_xds[c].attrs, f'block xds {c}', 'main xds {c}'
            )
        self.assertEqual(
            xds.time, big_xds.time, 'Incorrect time coordinate value'
        )
        self.assertEqual(
            xds.polarization, big_xds.polarization[0:1], 'Incorrect polarization coordinate value'
        )
        self.assertTrue(
            (xds.frequency == big_xds.frequency[0:4]).all(),
            'Incorrect frequency coordinate values'
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


    def test_xds_polarization_axis(self):
        """Test xds has correct stokes values"""
        self.compare_polarization(self.xds())


    def test_xds_frequency_axis(self):
        """Test xds has correct frequencyuency values and metadata"""
        self.compare_frequency(self.xds())


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
    """
    test casacore -> xds -> casacore round trip, ensure
    the two casacore images are identical
    """

    __imname2: str = os.sep.join([
        os.path.dirname(sys.argv[0]),
        'data', 'demo_simulated.im'
    ])

    __outname2: str = 'check_beam.im'


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
        del im1
        del im2


    def test_metadata(self):
        """Test to verify metadata in two casacore images is the same"""
        im1 = casacore.images.image(self.imname())
        im2 = casacore.images.image(self.outname())
        c1 = im1.info()
        c2 = im2.info()
        # some quantities are expected to have different untis and values
        f = 180*60/np.pi
        c2['coordinates']['direction0']['cdelt'] *= f
        c2['coordinates']['direction0']['crval'] *= f
        c2['coordinates']['direction0']['units'] = ["'", "'"]
        # the actual velocity values aren't stored but rather computed
        # by casacore on the fly, so we cannot easily compare them,
        # and really comes down to comparing the values of c used in
        # the computations (eg, if c is in m/s or km/s)
        c2['coordinates']['spectral2']['velUnit'] = 'km/s'
        self.dict_equality(c2, c1, 'got', 'expected')
        del im1
        del im2


    def test_multibeam(self):
        """
        Verify fix to issue 45
        https://github.com/casangi/xradio/issues/45
        """
        xds = read_image(self.__imname2)
        write_image(xds, self.__outname2, out_format='casa')
        im1 = casacore.images.image(self.__imname2)
        im2 = casacore.images.image(self.__outname2)
        beams1 = im1.imageinfo()['perplanebeams']
        beams2 = im2.imageinfo()['perplanebeams']
        for i in range(200):
            beam = beams2[f'*{i}']
            beam['major']['value'] *= 180*60/np.pi
            beam['major']['unit'] = 'arcmin'
            beam['minor']['value'] *= 180*60/np.pi
            beam['minor']['unit'] = 'arcmin'
            beam['positionangle']['value'] *= 180/np.pi
            beam['positionangle']['unit'] = 'deg'
        self.dict_equality(beams1, beams2, 'got', 'expected')
        del im1
        del im2


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


    def test_xds_polarization_axis(self):
        """Test xds has correct stokes values"""
        self.compare_polarization(self.__zds)


    def test_xds_frequency_axis(self):
        """Test xds has correct frequencyuency values and metadata"""
        self.compare_frequency(self.__zds)


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


class make_empty_sky_image_test(ImageBase):
    """Test making skeleton image"""


    @classmethod
    def setUpClass(cls):
        xxds = xr.Dataset()
        cls.__skel_im = make_empty_sky_image(
            xxds, [0.2, -0.5], [10, 10], [np.pi/180/60,
            np.pi/180/60], [1.412e9, 1.413e9],
            ['I', 'Q', 'U'], [54000.1]
        )


    @classmethod
    def tearDownClass(cls):
        pass


    def skel_im(self):
        return self.__skel_im


    def test_time_coord(self):
        skel = self.skel_im()
        self.assertTrue(
            np.isclose(skel.time, [54000.1]).all(),
            'Incorrect time coordinate values'
        )
        expec = {'refer': 'UTC', 'unit': 'd', 'format': 'MJD'}
        self.dict_equality(
            skel.time.attrs, expec, 'got', 'expected'
        )


    def test_polarization_coord(self):
        skel = self.skel_im()
        self.assertTrue(
            (skel.polarization == ['I', 'Q', 'U']).all(),
            'Incorrect polarization coordinate values'
        )

    def test_frequency_coord(self):
        skel = self.skel_im()
        self.assertTrue(
            np.isclose(skel.frequency, [1.412e+09, 1.413e+09]).all(),
            'Incorrect frequency coordinate values'
        )
        expec = {
            'conversion': {
                'direction': {
                    'm0': {'unit': 'rad', 'value': 0.0},
                    'm1': {'unit': 'rad', 'value': 1.5707963267948966},
                    'refer': 'FK5', 'type': 'direction'
                },
                'epoch': {
                    'm0': {'unit': 'd', 'value': 0.0},
                    'refer': 'LAST', 'type': 'epoch'
                },
                'position': {
                    'm0': {'unit': 'rad', 'value': 0.0},
                    'm1': {'unit': 'rad', 'value': 0.0},
                    'm2': {'unit': 'm', 'value': 0.0},
                    'refer': 'ITRF', 'type': 'position'
                },
                'system': 'LSRK'
            },
            'native_type': 'FREQ', 'restfreq': 1413000000.0,
            'restfreqs': [1413000000.0], 'system': 'LSRK', 'unit': 'Hz',
            'wave_unit': 'mm',
            'wcs': {'crval': 1413000000.0, 'cdelt': 1000000.0, 'pc': 1.0}
        }
        self.dict_equality(
            skel.frequency.attrs, expec, 'got', 'expected'
        )


    def test_vel_coord(self):
        skel = self.skel_im()
        self.assertTrue(
            np.isclose(skel.vel, [212167.34465675, 0]).all(),
            'Incorrect vel coordinate values'
        )
        expec = {'doppler_type': 'RADIO', 'unit': 'm/s'}
        self.dict_equality(
            skel.vel.attrs, expec, 'got', 'expected'
        )


    def test_right_ascension_coord(self):
        skel = self.skel_im()
        expec = [
            [
                0.20165865, 0.20165838, 0.20165812, 0.20165785, 0.20165759,
                0.20165733, 0.20165706, 0.2016568 , 0.20165654, 0.20165628
            ],
            [
                0.20132692, 0.20132671, 0.20132649, 0.20132628, 0.20132607,
                0.20132586, 0.20132565, 0.20132544, 0.20132523, 0.20132502
            ],
            [
                0.20099519, 0.20099503, 0.20099487, 0.20099471, 0.20099455,
                0.2009944 , 0.20099424, 0.20099408, 0.20099392, 0.20099377
            ],
            [
                0.20066346, 0.20066335, 0.20066325, 0.20066314, 0.20066304,
                0.20066293, 0.20066283, 0.20066272, 0.20066262, 0.20066251
            ],
            [
                0.20033173, 0.20033168, 0.20033162, 0.20033157, 0.20033152,
                0.20033147, 0.20033141, 0.20033136, 0.20033131, 0.20033126
            ],
            [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
            [
                0.19966827, 0.19966832, 0.19966838, 0.19966843, 0.19966848,
                0.19966853, 0.19966859, 0.19966864, 0.19966869, 0.19966874
            ],
            [
                0.19933654, 0.19933665, 0.19933675, 0.19933686, 0.19933696,
                0.19933707, 0.19933717, 0.19933728, 0.19933738, 0.19933749
            ],
            [
                0.19900481, 0.19900497, 0.19900513, 0.19900529, 0.19900545,
                0.1990056 , 0.19900576, 0.19900592, 0.19900608, 0.19900623
            ],
            [
                0.19867308, 0.19867329, 0.19867351, 0.19867372, 0.19867393,
                0.19867414, 0.19867435, 0.19867456, 0.19867477, 0.19867498
            ]
        ]
        self.assertTrue(
            np.isclose(skel.right_ascension, expec).all(),
            'Incorrect right_ascension coordinate values'
        )
        expec = {
            'unit': 'rad', 'wcs': {'crval': 0.2,
            'cdelt': -0.0002908882086657216}
        }
        self.dict_equality(
            skel.right_ascension.attrs, expec, 'got', 'expected'
        )


    def test_declination_coord(self):
        skel = self.skel_im()
        expec = [
            [
                -0.50145386, -0.50116297, -0.50087209, -0.5005812 , -0.50029031,
                -0.49999942, -0.49970853, -0.49941765, -0.49912676, -0.49883587
            ],
            [
                -0.50145407, -0.50116318, -0.50087229, -0.50058141, -0.50029052,
                -0.49999963, -0.49970874, -0.49941785, -0.49912697, -0.49883608
            ],
            [
                -0.50145423, -0.50116334, -0.50087246, -0.50058157, -0.50029068,
                -0.49999979, -0.4997089 , -0.49941802, -0.49912713, -0.49883624
            ],
            [
                -0.50145435, -0.50116346, -0.50087257, -0.50058168, -0.5002908 ,
                -0.49999991, -0.49970902, -0.49941813, -0.49912724, -0.49883635
            ],
            [
                -0.50145442, -0.50116353, -0.50087264, -0.50058175, -0.50029087,
                -0.49999998, -0.49970909, -0.4994182 , -0.49912731, -0.49883642
            ],
            [
                -0.50145444, -0.50116355, -0.50087266, -0.50058178, -0.50029089,
                -0.5, -0.49970911, -0.49941822, -0.49912734, -0.49883645
            ],
            [
                -0.50145442, -0.50116353, -0.50087264, -0.50058175, -0.50029087,
                -0.49999998, -0.49970909, -0.4994182 , -0.49912731, -0.49883642
            ],
            [
                -0.50145435, -0.50116346, -0.50087257, -0.50058168, -0.5002908,
                -0.49999991, -0.49970902, -0.49941813, -0.49912724, -0.49883635
            ],
            [
                -0.50145423, -0.50116334, -0.50087246, -0.50058157, -0.50029068,
                -0.49999979, -0.4997089 , -0.49941802, -0.49912713, -0.49883624
            ],
            [
                -0.50145407, -0.50116318, -0.50087229, -0.50058141, -0.50029052,
                -0.49999963, -0.49970874, -0.49941785, -0.49912697, -0.49883608
            ]
        ]
        self.assertTrue(
            np.isclose(skel.declination, expec).all(),
            'Incorrect declinationion coordinate values'

        )
        expec = {
            'unit': 'rad', 'wcs': {
                'crval': -0.5, 'cdelt': 0.0002908882086657216
            }
        }
        self.dict_equality(
            skel.declination.attrs, expec, 'got', 'expected'
        )


    def test_attrs(self):
        skel = self.skel_im()
        expec = {
            'direction': {
                'conversion_system': 'FK5',
                'conversion_equinox': 'J2000',
                'long_pole': 0.0, 'lat_pole': 0.0,
                'pc': np.array([[1., 0.], [0., 1.]]),
                'projection': 'SIN',
                'projection_parameters': np.array([0., 0.]),
                'system': 'FK5',
                'equinox': 'J2000'
            },
            'active_mask': '',
            'beam': None,
            'object_name': '',
            'obsdate': {
                'refer': 'UTC', 'format': 'MJD', 'value': 54000.0,
                'unit': 'd'
            },
            'observer': 'Karl Jansky',
            'pointing_center': {
                'value': np.array([ 0.2, -0.5]), 'initial': True
            },
            'description': '',
            'telescope': {
                'name': 'ALMA',
                'position': {
                    'type': 'position', 'refer': 'ITRF',
                    'm2': {'value': 6379946.01326443, 'unit': 'm'},
                    'm1': {'unit': 'rad', 'value': -0.3994149869262738},
                    'm0': {'unit': 'rad', 'value': -1.1825465955049892}
                }
            },
            'history': None
        }
        self.dict_equality(
            skel.attrs, expec, 'got', 'expected'
        )


class fits_to_xds_test(ImageBase):
    """
    test fits_to_xds
    """


    @classmethod
    def setUpClass(cls):
        # by default, subclass setUpClass() method is called before super class',
        # so we must explicitly call the super class' method here to create the
        # xds which is located in the super class
        super().setUpClass()
        cls.__fds = read_image(cls.infits(), {'frequency': 5})


    @classmethod
    def tearDownClass(cls):
        pass


    def setUp(self):
        pass


    def tearDown(self):
        pass


    def test_xds_pixel_values(self):
        """Test xds has correct pixel values"""
        self.compare_sky_mask(self.__fds, True)


    def test_xds_time_axis(self):
        """Test values and attributes on the time axis"""
        self.compare_time(self.__fds)


    def test_xds_polarization_axis(self):
        """Test xds has correct stokes values"""
        self.compare_polarization(self.__fds)


    def test_xds_frequency_axis(self):
        """Test xds has correct frequency values and metadata"""
        self.compare_frequency(self.__fds, True)


    def test_xds_vel_axis(self):
        """Test xds has correct velocity values and metadata"""
        self.compare_vel_axis(self.__fds, True)


    def test_xds_ra_dec_axis(self):
        """Test xds has correct RA and Dec values and attributes"""
        self.compare_ra_dec(self.__fds, True)


    def test_xds_attrs(self):
        """Test xds level attributes"""
        self.compare_attrs(self.__fds, True)

    #TODO
    def test_get_img_ds_block(self):
        #self.compare_image_block(self.imname())
        pass


if __name__ == '__main__':
    unittest.main()
