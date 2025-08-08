import astropy.units as u
from astropy.io import fits
import pytest

try:
    from casacore import images, tables
except ImportError:
    import xradio._utils._casacore.casacore_from_casatools as images
    import xradio._utils._casacore.casacore_from_casatools as tables

import copy
import numbers
import os
import shutil
import unittest
from glob import glob
import re

import dask.array as da
import numpy as np
import numpy.ma as ma
import xarray as xr
from toolviper.utils.data import download

from xradio.image import (
    load_image,
    make_empty_aperture_image,
    make_empty_lmuv_image,
    make_empty_sky_image,
    read_image,
    write_image,
)
from xradio.image._util._casacore.common import _create_new_image as create_new_image
from xradio.image._util._casacore.common import _open_image_ro as open_image_ro
from xradio.image._util.common import _image_type as image_type
from xradio.image._util._casacore.common import _object_name

from xradio.image._util._casacore.common import (
    _open_image_ro as open_image_ro,
    _create_new_image as create_new_image,
)
from toolviper.dask.client import local_client

sky = "SKY"


def safe_convert(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()  # convert to list
    raise TypeError(f"Type {type(obj)} not serializable")


@pytest.fixture(scope="module")
def dask_client_module():
    """Set up and tear down a Dask client for the test module.

    This fixture starts a local Dask cluster with specified resources before
    any tests in the module are run, and ensures the client and cluster are
    properly closed after all tests complete.

    Returns
    -------
    distributed.Client
        A Dask client connected to a local cluster, shared across all tests
        in the module.
    """

    import sys

    print("\nSetting up Dask client for the test module...")
    client = local_client(
        cores=2, memory_limit="3GB"
    )  # Do not increase size otherwise GitHub MacOS runner will hang.
    try:
        yield client
    finally:
        print("\nTearing down Dask client for the test module...")
        if client is not None:
            client.close()
            # Ensure the associated cluster is also properly closed
            cluster = getattr(client, "cluster", None)
            if cluster is not None:
                cluster.close()


@pytest.mark.usefixtures("dask_client_module")
class ImageBase(unittest.TestCase):
    def dict_equality(self, dict1, dict2, dict1_name, dict2_name, exclude_keys=[]):

        # if dict1_name == "Got attrs[direction]":
        #     print("************* Comparing dictionaries:")
        #     print("dict1_name:", dict1_name, dict1)
        #     print("dict2_name:", dict2_name, dict2)
        #     print("*************")

        self.assertEqual(
            dict1.keys(),
            dict2.keys(),
            f"{dict1_name} has different keys than {dict2_name}:"
            f"\n{dict1.keys()} vs\n {dict2.keys()}",
        )
        for k in dict1.keys():
            if k not in exclude_keys:
                one = dict1[k]
                two = dict2[k]
                if isinstance(one, numbers.Number) and isinstance(two, numbers.Number):
                    self.assertTrue(
                        np.isclose(one, two),
                        f"{dict1_name}[{k}] != {dict2_name}[{k}]:\n"
                        + f"{one} vs\n{two}",
                    )
                elif (isinstance(one, list) or isinstance(one, np.ndarray)) and (
                    isinstance(two, list) or isinstance(two, np.ndarray)
                ):
                    if len(one) == 0 or len(two) == 0:
                        self.assertEqual(
                            len(one),
                            len(two),
                            f"{dict1_name}[{k}] != {dict2_name}[{k}], "
                            f"{one} != {two}",
                        )
                    elif isinstance(one[0], numbers.Number):
                        self.assertTrue(
                            np.isclose(
                                np.array(one), np.array(two), rtol=1e-3, atol=1e-7
                            ).all(),
                            f"{dict1_name}[{k}] != {dict2_name}[{k}], "
                            f"{one} != {two}",
                        )
                else:
                    self.assertEqual(
                        type(dict1[k]),
                        type(dict2[k]),
                        f"Types are different {dict1_name}[{k}] {type(dict1[k])} "
                        + f"vs {dict2_name}[{k}] {type(dict2[k])}",
                    )
                    if isinstance(dict1[k], dict) and isinstance(dict2[k], dict):
                        self.dict_equality(
                            dict1[k],
                            dict2[k],
                            f"{dict1_name}[{k}]",
                            f"{dict2_name}[{k}]",
                        )
                    elif isinstance(one, np.ndarray):
                        if k == "crpix":
                            self.assertTrue(
                                np.allclose(one, two, rtol=3e-5),
                                f"{dict1_name}[{k}] != {dict2_name}[{k}], {one} vs {two}",
                            )
                        else:
                            self.assertTrue(
                                np.allclose(one, two),
                                f"{dict1_name}[{k}] != {dict2_name}[{k}], {one} vs {two}",
                            )
                    else:
                        self.assertEqual(
                            dict1[k],
                            dict2[k],
                            f"{dict1_name}[{k}] != {dict2_name}[{k}]:\n"
                            + f"{dict1[k]} vs\n{dict2[k]}",
                        )


class xds_from_image_test(ImageBase):
    _imname: str = "inp.im"
    _outname: str = "out.im"
    _infits: str = "inp.fits"
    _uv_image: str = "complex_valued_uv.im"
    _xds = None
    _exp_sky_attrs = {
        "active_mask": "MASK0",
        "description": None,
        "image_type": "Intensity",
        "object_name": "",
        "obsdate": {
            "attrs": {
                "format": "MJD",
                "scale": "UTC",
                "type": "time",
                "units": "d",
            },
            "data": 51544.00000000116,
            "dims": [],
        },
        "observer": "Karl Jansky",
        "pointing_center": {
            "data": [1.832595714594046, -0.6981317007977318],
            "dims": ["l", "m"],
            "attrs": {
                "type": "sky_coord",
                "frame": "fk5",
                # "equinox": "j2000.0",
                "units": "rad",
            },
            # "value": np.array([6300, -2400]) * np.pi / 180 / 60,
            # "initial": True,
        },
        "telescope": {
            "name": "ALMA",
            "direction": {
                "attrs": {
                    "coordinate_system": "geocentric",
                    "frame": "ITRF",
                    "origin_object_name": "earth",
                    "type": "location",
                    "units": "rad",
                },
                "data": np.array([-1.1825465955049892, -0.3994149869262738]),
                "dims": ["ellipsoid_dir_label"],
                "coords": {
                    "ellipsoid_dir_label": {
                        "dims": ["ellipsoid_dir_label"],
                        "data": ["lon", "lat"],
                    }
                },
            },
            "distance": {
                "attrs": {
                    "coordinate_system": "geocentric",
                    "frame": "ITRF",
                    "origin_object_name": "earth",
                    "type": "location",
                    "units": "m",
                },
                "data": np.array([6379946.01326443]),
                "dims": ["ellipsoid_dis_label"],
                "coords": {
                    "ellipsoid_dis_label": {
                        "dims": ["ellipsoid_dis_label"],
                        "data": [
                            "dist",
                        ],
                    }
                },
            },
        },
        "units": "Jy/beam",
        "user": {},
        "history": None,
    }

    _exp_vals: dict = {
        "shape": xr.core.utils.Frozen(
            {
                "time": 1,
                "frequency": 10,
                "polarization": 4,
                "l": 30,
                "m": 20,
                "beam_param": 3,
            }
        ),
        "freq_waveunit": "mm",
        "stokes": ["I", "Q", "U", "V"],
        "time_format": "MJD",
        "time_refer": "UTC",
        "time_units": "d",
        "vel_mea_type": "doppler",
        "doppler_type": "radio",
        "vel_units": "m/s",
        "frequency": [
            1.414995e09,
            1.414996e09,
            1.414997e09,
            1.414998e09,
            1.414999e09,
            1.415000e09,
            1.415001e09,
            1.415002e09,
            1.415003e09,
            1.415004e09,
        ],
        "rest_frequency": {
            "attrs": {
                "type": "quantity",
                "units": "Hz",
            },
            "data": 1420405751.7860003,
            "dims": [],
        },
        "reference_frequency": {
            "attrs": {
                "observer": "lsrk",
                "type": "spectral_coord",
                "units": "Hz",
            },
            "data": 1415000000.0,
            "dims": [],
        },
        "wave_units": "mm",
        "beam_param": ["major", "minor", "pa"],
    }
    _rad_to_arcmin = np.pi / 180 / 60
    _exp_xds_attrs = {}
    _exp_xds_attrs["direction"] = {
        "reference": {
            "data": [1.832595714594046, -0.6981317007977318],
            "dims": ["l", "m"],
            "attrs": {
                "type": "sky_coord",
                "frame": "fk5",
                "equinox": "j2000.0",
                "units": "rad",
            },
        },
        # "conversion_system": "FK5",
        # "conversion_equinox": "J2000",
        # there seems to be a casacore bug here that changing either the
        # crval or pointingcenter will also change the latpole when the
        # casacore image is reopened. As long as the xds gets the latpole
        # that the casacore image has is all we care about for testing
        "latpole": {
            "attrs": {
                "type": "quantity",
                "units": "rad",
            },
            "data": -40.0 * np.pi / 180,
            "dims": ["l", "m"],
        },
        "lonpole": {
            "attrs": {
                "type": "quantity",
                "units": "rad",
            },
            "data": np.pi,
            "dims": ["l", "m"],
        },
        "pc": np.array([[1.0, 0.0], [0.0, 1.0]]),
        "projection_parameters": np.array([0.0, 0.0]),
        "projection": "SIN",
    }
    # TODO make a more interesting beam
    # _exp_xds_attrs["history"] = None

    # _ran_measures_code = False

    _expec_uv = {}

    @classmethod
    def setUpClass(cls):
        cls.maxDiff = None
        # if not cls._ran_measures_code and os.environ["USER"] == "runner":
        #     # casa_data_dir = (
        #     #     importlib.resources.files("casadata") / "__data__"
        #     # ).as_posix()
        #     # rc_file = open(os.path.expanduser("~/.casarc"), "a+")  # append mode
        #     # rc_file.write("\nmeasures.directory: " + casa_data_dir)
        #     # rc_file.close()
        #     cls._ran_measures_code = True
        cls._make_image()

    @classmethod
    def tearDownClass(cls):
        for f in [
            cls._imname,
            cls._imname + "_2",
            cls._outname,
            cls._infits,
            cls._uv_image,
        ]:
            if os.path.exists(f):
                if os.path.isdir(f):
                    shutil.rmtree(f)
                else:
                    os.remove(f)

    @classmethod
    def _make_image(cls):
        shape: list[int] = [10, 4, 20, 30]
        mask: np.ndarray = np.array(
            [i % 3 == 0 for i in range(np.prod(shape))], dtype=bool
        ).reshape(shape)
        pix: np.ndarray = np.array([range(np.prod(shape))], dtype=np.float64).reshape(
            shape
        )
        masked_array = ma.masked_array(pix, mask)
        with create_new_image(cls._imname, shape=shape) as im:
            im.put(masked_array)
            shape = im.shape()
        t = tables.table(cls._imname, readonly=False)
        t.putkeyword("units", "Jy/beam")
        csys = t.getkeyword("coords")
        pc = np.array([6300, -2400])
        # change pointing center
        csys["direction0"]["crval"] = pc
        csys["pointingcenter"]["value"] = pc * np.pi / 180 / 60
        t.putkeyword("coords", csys)
        t.close()
        t = tables.table(os.sep.join([cls._imname, "logtable"]), readonly=False)
        t.addrows()
        t.putcell("MESSAGE", 0, "HELLO FROM EARTH again")
        t.flush()
        t.close()
        with open_image_ro(cls._imname) as im:
            im.tofits(cls._infits)
            assert os.path.exists(cls._infits), f"Could not create {cls._infits}"

        cls._xds = read_image(cls._imname, {"frequency": 5})
        cls._xds_no_sky = read_image(cls._imname, {"frequency": 5}, False, False)
        cls.assertTrue(cls._xds.sizes == cls._exp_vals["shape"], "Incorrect shape")
        write_image(cls._xds, cls._outname, out_format="casa", overwrite=True)

    def imname(self):
        return self._imname

    @classmethod
    def infits(cls):
        return cls._infits

    @classmethod
    def xds(cls):
        return cls._xds

    @classmethod
    def xds_no_sky(cls):
        return cls._xds_no_sky

    @classmethod
    def outname(cls):
        return cls._outname

    @classmethod
    def uv_image(cls):
        return cls._uv_image

    def exp_sky_attrs(self):
        return self._exp_sky_attrs

    def exp_xds_attrs(self):
        return self._exp_xds_attrs

    def compare_sky_mask(self, xds: xr.Dataset, fits=False):
        """Compare got sky and mask values to expected values"""
        ev = self._exp_vals
        self.assertEqual(
            xds[sky].attrs["image_type"],
            self.exp_sky_attrs()["image_type"],
            "Wrong image type",
        )
        self.assertEqual(
            xds[sky].attrs["units"], self.exp_sky_attrs()["units"], "Wrong unit"
        )
        self.assertEqual(
            xds[sky].chunksizes["frequency"],
            (5, 5),
            "Incorrect chunksize",
        )
        self.assertEqual(
            xds.MASK0.chunksizes["frequency"], (5, 5), "Incorrect chunksize"
        )
        got_data = da.squeeze(da.transpose(xds[sky], [1, 2, 4, 3, 0]), 4)
        got_mask = da.squeeze(da.transpose(xds.MASK0, [1, 2, 4, 3, 0]), 4)
        if "sky_array" not in ev:
            im = images.image(self.imname())
            ev[sky] = im.getdata()
            # getmask returns the negated value of the casa image mask, so True
            # has the same meaning as it does in xds.MASK0
            ev["mask0"] = im.getmask()
            ev["sum"] = im.statistics()["sum"][0]
        if fits:
            self.assertTrue(
                not np.isnan(got_data == ev[sky]).all(),
                "pixel values incorrect",
            )
        else:
            self.assertTrue((got_data == ev[sky]).all(), "pixel values incorrect")
        self.assertTrue((got_mask == ev["mask0"]).all(), "mask values incorrect")
        got_ma = da.ma.masked_array(xds[sky], xds.MASK0)
        self.assertEqual(da.sum(got_ma), ev["sum"], "Incorrect value for sum")
        self.assertTrue(
            got_data.dtype == ev[sky].dtype,
            f"Incoorect data type, got {got_data.dtype}, expected {ev[sky].dtype}",
        )

    def compare_time(self, xds: xr.Dataset) -> None:
        ev = self._exp_vals
        if "time" not in ev:
            im = images.image(self.imname())
            coords = im.coordinates().dict()["obsdate"]
            ev["time"] = coords["m0"]["value"]
        got_vals = xds.time
        self.assertEqual(got_vals, ev["time"], "Incorrect time axis values")
        self.assertEqual(
            xds.coords["time"].attrs["type"],
            "time",
            'Incoorect measure type, should be "time"',
        )
        self.assertEqual(
            xds.coords["time"].attrs["format"],
            ev["time_format"],
            "Incoorect time axis format",
        )
        self.assertEqual(
            xds.coords["time"].attrs["scale"],
            ev["time_refer"],
            "Incoorect time axis refer",
        )
        self.assertEqual(
            xds.coords["time"].attrs["units"],
            ev["time_units"],
            "Incoorect time axis unitt",
        )

    def compare_polarization(self, xds: xr.Dataset) -> None:
        self.assertTrue(
            (xds.coords["polarization"] == self._exp_vals["stokes"]).all(),
            "Incorrect polarization values",
        )

    def compare_frequency(self, xds: xr.Dataset):
        ev = self._exp_vals
        self.assertTrue(
            np.isclose(xds.frequency, ev["frequency"]).all(), "Incorrect frequencies"
        )
        self.dict_equality(
            xds.frequency.attrs["rest_frequency"],
            ev["rest_frequency"],
            "got",
            "expected",
        )
        self.dict_equality(
            xds.frequency.attrs["reference_frequency"],
            ev["reference_frequency"],
            "got",
            "expected",
        )
        self.assertEqual(
            xds.frequency.attrs["reference_frequency"]["attrs"]["type"],
            "spectral_coord",
            "Wrong measure type",
        )
        self.assertEqual(
            xds.frequency.attrs["reference_frequency"]["attrs"]["units"],
            ev["reference_frequency"]["attrs"]["units"],
            "Wrong frequency unit",
        )
        self.assertEqual(
            xds.frequency.attrs["reference_frequency"]["attrs"]["observer"],
            ev["reference_frequency"]["attrs"]["observer"],
            "Incorrect frequency frame",
        )
        self.assertEqual(
            xds.frequency.attrs["wave_units"],
            ev["freq_waveunit"],
            "Incorrect wavelength unit",
        )

    def compare_vel_axis(self, xds: xr.Dataset, fits: bool = False):
        ev = self._exp_vals
        if fits:
            # casacore has written optical velocities to FITS file,
            # even though the doppler type is RADIO in the casacore
            # image
            freqs = xds.coords["frequency"].values
            rest_freq = xds.coords["frequency"].attrs["rest_frequency"]["data"]
            v_opt = (rest_freq / freqs - 1) * 299792458
            self.assertTrue(
                np.isclose(xds.velocity, v_opt).all(), "Incorrect velocities"
            )
            self.assertEqual(
                xds.velocity.attrs["doppler_type"], "Z", "Incoorect velocity type"
            )
        else:
            if "velocity" not in ev:
                im = images.image(self.imname())
                freqs = []
                for chan in range(10):
                    freqs.append(im.toworld([chan, 0, 0, 0])[0])
                freqs = np.array(freqs)
                spec_coord = images.coordinates.spectralcoordinate(
                    im.coordinates().dict()["spectral2"]
                )
                rest_freq = spec_coord.get_restfrequency()
                ev["velocity"] = (1 - freqs / rest_freq) * 299792458
            self.assertTrue(
                (xds.velocity == ev["velocity"]).all(), "Incorrect velocities"
            )
            self.assertEqual(
                xds.velocity.attrs["doppler_type"],
                ev["doppler_type"],
                "Incoorect velocity type",
            )
        self.assertEqual(
            xds.velocity.attrs["units"], ev["vel_units"], "Incorrect velocity unit"
        )
        self.assertEqual(
            xds.velocity.attrs["type"],
            ev["vel_mea_type"],
            "Incoorect doppler measure type",
        )

    def compare_l_m(self, xds: xr.Dataset) -> None:
        cdelt = np.pi / 180 / 60
        l_vals = xds.coords["l"].values
        m_vals = xds.coords["m"].values
        self.assertTrue(
            np.isclose(
                l_vals,
                np.array([(i - 15) * (-1) * cdelt for i in range(xds.sizes["l"])]),
            ).all(),
            "Wrong l values",
        )
        self.assertTrue(
            np.isclose(
                m_vals, np.array([(i - 10) * cdelt for i in range(xds.sizes["m"])])
            ).all(),
            "Wrong m values",
        )
        l_attrs = xds.coords["l"].attrs
        m_attrs = xds.coords["m"].attrs
        e_l_attrs = {
            "note": (
                "l is the angle measured from the phase center to the east. "
                "So l = x*cdelt, where x is the number of pixels from the phase center. "
                "See AIPS Memo #27, Section III."
            ),
        }
        e_m_attrs = {
            "note": (
                "m is the angle measured from the phase center to the north. "
                "So m = y*cdelt, where y is the number of pixels from the phase center. "
                "See AIPS Memo #27, Section III."
            ),
        }
        self.dict_equality(l_attrs, e_l_attrs, "got l attrs", "expec l attrs")
        self.dict_equality(m_attrs, e_m_attrs, "got m attrs", "expec m attrs")

    def compare_ra_dec(self, xds: xr.Dataset) -> None:
        ev = self._exp_vals
        if "ra" not in ev:
            im = images.image(self.imname())
            shape = im.shape()
            dd = im.coordinates().dict()["direction0"]
            ev["ra"] = np.zeros([shape[3], shape[2]])
            ev["dec"] = np.zeros([shape[3], shape[2]])
            for i in range(shape[3]):
                for j in range(shape[2]):
                    w = im.toworld([0, 0, j, i])
                    ev["ra"][i][j] = w[3]
                    ev["dec"][i][j] = w[2]
            f = np.pi / 180 / 60
            ev["ra"] *= f
            ev["dec"] *= f
            ev["ra_crval"] = dd["crval"][0] * f
            ev["ra_cdelt"] = dd["cdelt"][0] * f
            ev["dec_crval"] = dd["crval"][1] * f
            ev["dec_cdelt"] = dd["cdelt"][1] * f
        self.assertEqual(xds.right_ascension.attrs, {}, "RA has attrs but shouldn't")
        self.assertEqual(xds.declination.attrs, {}, "RA has attrs but shouldn't")
        self.assertTrue(
            np.allclose(xds.right_ascension, ev["ra"], atol=1e-15),
            "Incorrect RA values",
        )
        self.assertTrue(
            np.allclose(xds.declination, ev["dec"], atol=1e-15), "Incorrect Dec values"
        )

    def compare_beam_param(self, xds: xr.Dataset) -> None:
        ev = self._exp_vals
        self.assertTrue(
            (xds.beam_param == ev["beam_param"]).all(), "Incorrect beam param values"
        )

    def compare_sky_attrs(self, sky: xr.DataArray, fits: bool = False) -> None:
        my_exp_attrs = copy.deepcopy(self.exp_sky_attrs())

        # print("Comparing sky attrs")
        # print("Got attrs:", sky.attrs)
        # print("******************")

        if "direction" not in sky.attrs["telescope"]:
            del my_exp_attrs["telescope"]["direction"]
            del my_exp_attrs["telescope"]["distance"]
        if fits:
            # xds from fits do not have history yet
            # del my_exp_attrs["history"]
            my_exp_attrs["user"]["comment"] = (
                "casacore non-standard usage: 4 LSD, " "5 GEO, 6 SOU, 7 GAL"
            )
            # fits doesn't have history yet
            del my_exp_attrs["history"]
        else:
            self.assertTrue(
                isinstance(sky.attrs["history"], xr.core.dataset.Dataset),
                "Incorrect type for history data",
            )
        self.dict_equality(
            sky.attrs, my_exp_attrs, "Got sky attrs", "Expected sky attrs", ["history"]
        )

    def compare_xds_attrs(self, xds: xr.Dataset):
        my_exp_attrs = copy.deepcopy(self.exp_xds_attrs())

        self.dict_equality(
            xds.attrs, my_exp_attrs, "Got attrs", "Expected attrs", ["history"]
        )

    def compare_image_block(self, imagename, zarr=False):
        x = [0] if zarr else [0, 1]
        full_xds = read_image(imagename)
        shape = (
            full_xds.sizes["time"],
            full_xds.sizes["frequency"],
            full_xds.sizes["polarization"],
            3,
        )
        ary = np.ones(shape, dtype=np.float32)
        ary[0, 2, 0, :] = 2.0
        xda = xr.DataArray(
            data=ary,
            dims=["time", "frequency", "polarization", "beam_param"],
            coords={
                "time": full_xds.time,
                "frequency": full_xds.frequency,
                "polarization": full_xds.polarization,
                "beam_param": ["major", "minor", "pa"],
            },
        )
        full_xds["BEAM"] = xda
        full_xds["BEAM"].attrs["units"] = "rad"
        imag = imagename + "_2"

        write_image(
            full_xds, imag, out_format="zarr" if zarr else "casa", overwrite=True
        )

        # print("*" * 100, "Comparing image block", imag, zarr)
        for i in x:
            xds = load_image(
                imag,
                {
                    "l": slice(2, 10),
                    "m": slice(3, 15),
                    "polarization": slice(0, 1),
                    "frequency": slice(0, 4),
                },
                do_sky_coords=i == 0,
            )

            if not zarr:
                with open_image_ro(imagename) as im:
                    self.assertTrue(
                        xds[sky].dtype == im.datatype()
                        or (xds[sky].dtype == np.float32 and im.datatype() == "float"),
                        f"got wrong data type, got {xds[sky].dtype}, "
                        + f"expected {im.datatype()}",
                    )
            self.assertEqual(xds[sky].shape, (1, 4, 1, 8, 12), "Wrong block shape")
            big_xds = self._xds if i == 0 else self._xds_no_sky
            self.assertTrue(
                (xds[sky] == big_xds[sky][:, 0:1, 0:4, 2:10, 3:15]).all(),
                "Wrong block SKY array",
            )
            self.assertTrue(
                (xds.MASK0 == big_xds.MASK0[:, 0:1, 0:4, 2:10, 3:15]).all(),
                "Wrong block mask0 array",
            )
            self.dict_equality(
                xds.attrs, big_xds.attrs, "block xds", "main xds", ["history"]
            )
            coords = [
                "time",
                "frequency",
                "velocity",
                "polarization",
                "l",
                "m",
                "beam_param",
            ]
            if i == 0:
                coords.extend(["right_ascension", "declination"])
            elif i == 1:
                for c in ["right_ascension", "declination"]:
                    self.assertTrue(
                        c not in xds.coords, "{c} in coords but should not be"
                    )
            for c in coords:
                self.dict_equality(
                    xds[c].attrs, big_xds[c].attrs, f"block xds {c}", "main xds {c}"
                )
            self.assertEqual(xds.time, big_xds.time, "Incorrect time coordinate value")
            self.assertEqual(
                xds.polarization,
                big_xds.polarization[0:1],
                "Incorrect polarization coordinate value",
            )
            self.assertTrue(
                (xds.frequency == big_xds.frequency[0:4]).all(),
                "Incorrect frequency coordinate values",
            )
            self.assertTrue(
                (xds.velocity == big_xds.velocity[0:4]).all(),
                "Incorrect vel coordinate values",
            )
            if i == 0:
                self.assertTrue(
                    (xds.right_ascension == big_xds.right_ascension[2:10, 3:15]).all(),
                    "Incorrect right ascension coordinate values",
                )
                self.assertTrue(
                    (xds.declination == big_xds.declination[2:10, 3:15]).all(),
                    "Incorrect declination coordinate values",
                )
            # all coordinates and data variables should be numpy arrays when loading an
            # image section
            merged_dict = {**xds.coords, **xds.data_vars}
            for k, v in merged_dict.items():
                self.assertTrue(
                    isinstance(v.data, np.ndarray),
                    f"Wrong type for coord or data value {k}, got {type(v)}, must be a numpy.ndarray",
                )
            # test beam
            self.assertTrue(xds["BEAM"].shape == (1, 4, 1, 3), "Wrong beam shape")
            self.assertTrue(
                tuple(xds["BEAM"].dims)
                == ("time", "frequency", "polarization", "beam_param"),
                f"Wrong beam dims, got {tuple(xds['BEAM'].dims)}",
            )
            self.assertEqual(xds["BEAM"][0, 2, 0, 0], 2.0, "Wrong beam value")
            self.assertEqual(xds["BEAM"][0, 0, 0, 0], 1.0, "Wrong beam value")

    def compare_uv(self, xds: xr.Dataset, image: str) -> None:
        if not self._expec_uv:
            with open_image_ro(image) as im:
                uv_coords = im.coordinates().dict()["linear0"]
                shape = im.shape()
            uv = {}
            for i, z in enumerate(["u", "v"]):
                uv[z] = {}
                x = uv[z]
                x["attrs"] = {
                    "type": "quantity",
                    "crval": 0.0,
                    "units": uv_coords["units"][i],
                    "cdelt": uv_coords["cdelt"][i],
                }
                x["npix"] = shape[3] if z == "u" else shape[2]
            self._expec_uv = copy.deepcopy(uv)
        expec_coords = set(
            ["time", "polarization", "frequency", "velocity", "u", "v", "beam_param"]
        )
        self.assertEqual(xds.coords.keys(), expec_coords, "incorrect coordinates")
        for c in ["u", "v"]:
            attrs = self._expec_uv[c]["attrs"]
            self.dict_equality(
                xds[c].attrs, attrs, f"got attrs {c}", f"expec attrs {c}"
            )
            npix = self._expec_uv[c]["npix"]
            self.assertEqual(xds.sizes[c], npix, "Incorrect axis length")
            expec = [(i - npix // 2) * attrs["cdelt"] for i in range(npix)]
            self.assertTrue(
                (xds[c].values == np.array(expec)).all(),
                f"Incorrect values for coordinate {c}",
            )


class casa_image_to_xds_test(xds_from_image_test):
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

    def test_xds_l_m_axis(self):
        """Test xds has correct l and m values and attributes"""
        self.compare_l_m(self.xds())

    def test_xds_beam_param_axis(self):
        """Test xds has correct beam values and attributes"""
        self.compare_beam_param(self.xds())

    def test_xds_no_sky(self):
        """Test xds does not have sky coordinates"""
        xds = self.xds_no_sky()
        expec = set(self.xds().coords.keys())
        expec.remove("right_ascension")
        expec.remove("declination")
        self.assertEqual(set(xds.coords.keys()), expec, "Incorrect coords")

    def test_xds_ra_dec_axis(self):
        """Test xds has correct RA and Dec values and attributes"""
        self.compare_ra_dec(self.xds())

    def test_xds_attrs(self):
        """Test xds level attributes"""
        # import logging
        # logger = logging.getLogger(__name__)
        # logger.info("########## Comparing xds attrs")
        # logger.info("Expected attrs: %s", self.exp_xds_attrs())
        # logger.info("Got attrs: %s", self.xds().attrs)
        # logger.info("##########")
        self.compare_xds_attrs(self.xds())
        self.compare_sky_attrs(self.xds().SKY)

    def test_get_img_ds_block(self):
        self.compare_image_block(self.imname())

    def test_uv_image(self):
        image = self.uv_image()
        download(image)
        self.assertTrue(os.path.isdir(image), f"Cound not download {image}")
        xds = read_image(image)
        self.compare_uv(xds, image)


class xds_to_casacore(xds_from_image_test):
    _outname = "rabbit.im"
    _outname2 = "rabbit2.im"

    @classmethod
    def _clean(cls):
        for f in [cls._outname, cls._outname2]:
            if os.path.exists(f):
                if os.path.isdir(f):
                    shutil.rmtree(f)
                else:
                    os.remove(f)

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._clean()

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        cls._clean()

    def test_writing_numpy_array_as_data_var(self):
        """
        Test writing an xds which has a numpy array as the
        sky data var to a casa image.
        """
        xds = self.xds()
        write_image(xds, self._outname, "casa")
        with open_image_ro(self._outname) as im:
            p = im.getdata()
        exp_data = np.squeeze(np.transpose(xds[sky], [1, 2, 4, 3, 0]), 4)
        self.assertTrue((p == exp_data).all(), "Incorrect pixel values")

    def test_object_name_not_present(self):
        """
        Test writing an xds which does not have an object name
        to a casa image.
        """
        xds = self.xds()
        import pprint

        del xds["SKY"].attrs[_object_name]
        write_image(xds, self._outname2, "casa", overwrite=True)
        with open_image_ro(self._outname2) as im:
            ii = im.imageinfo()
            self.assertEqual(ii["objectname"], "", "Incorrect object name")


class casacore_to_xds_to_casacore(xds_from_image_test):
    """
    test casacore -> xds -> casacore round trip, ensure
    the two casacore images are identical
    """

    _outname_no_sky = "from_xds_no_sky.im"
    _imname2: str = "demo_simulated.im"
    _imname3: str = "no_mask.im"
    _outname2: str = "check_beam.im"
    _outname2_no_sky: str = _outname2 + "_no_sky"
    _outname3: str = "xds_2_casa_no_mask.im"
    _outname3_no_sky: str = _outname3 + "_no_sky"
    _outname4: str = "xds_2_casa_nans_and_mask.im"
    _outname4_no_sky: str = _outname4 + "_no_sky"
    _outname5: str = "xds_2_casa_nans_already_masked.im"
    _outname5_no_sky: str = _outname5 + "_no_sky"
    _outname6: str = "single_beam.im"
    _output_uv: str = "output_uv.im"

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        write_image(cls._xds_no_sky, cls._outname_no_sky, out_format="casa")

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        for f in [
            cls._outname_no_sky,
            cls._imname2,
            cls._imname3,
            cls._outname2,
            cls._outname2_no_sky,
            cls._outname3,
            cls._outname3_no_sky,
            cls._outname4,
            cls._outname4_no_sky,
            cls._outname5,
            cls._outname5_no_sky,
            cls._outname6,
            cls._output_uv,
        ]:
            if os.path.exists(f):
                if os.path.isdir(f):
                    shutil.rmtree(f)
                else:
                    os.remove(f)

    def test_pixels_and_mask(self):
        """Test pixel values are consistent"""
        with open_image_ro(self.imname()) as im1:
            for imname in [self.outname(), self._outname_no_sky]:
                with open_image_ro(imname) as im2:
                    self.assertTrue(
                        (im1.getdata() == im2.getdata()).all(), "Incorrect pixel values"
                    )
                    self.assertTrue(
                        (im1.getmask() == im2.getmask()).all(), "Incorrect mask values"
                    )
                    self.assertTrue(
                        im1.datatype() == im2.datatype(),
                        f"Incorrect round trip pixel type, input {im1.name()} {im1.datatype()}, "
                        + f"output {im2.name()} {im2.datatype()}",
                    )

    def test_metadata(self):
        """Test to verify metadata in two casacore images is the same"""
        f = 180 * 60 / np.pi
        with open_image_ro(self.imname()) as im1:
            c1 = im1.info()
            for imname in [self.outname(), self._outname_no_sky]:
                with open_image_ro(imname) as im2:
                    c2 = im2.info()
                    # some quantities are expected to have different untis and values
                    c2["coordinates"]["direction0"]["cdelt"] *= f
                    c2["coordinates"]["direction0"]["crval"] *= f
                    c2["coordinates"]["direction0"]["units"] = ["'", "'"]
                    # the actual velocity values aren't stored but rather computed
                    # by casacore on the fly, so we cannot easily compare them,
                    # and really comes down to comparing the values of c used in
                    # the computations (eg, if c is in m/s or km/s)
                    c2["coordinates"]["spectral2"]["velUnit"] = "km/s"
                    # it appears that 'worldreplace2' is not correctly recorded or retrieved
                    # by casatools, with empty np.array returned instead.
                    c2["coordinates"]["worldreplace2"] = np.array(
                        [c2["coordinates"]["spectral2"]["wcs"]["crval"]]
                    )
                    self.dict_equality(c2, c1, "got", "expected")

    def test_beam(self):
        """
            Verify fix to issue 45
            https://github.com/casangi/xradio/issues/45
        irint("*** r", r)
        """
        download(self._imname2), f"failed to download {self._imname2}"
        shutil.copytree(self._imname2, self._outname6)
        # multibeam image
        with open_image_ro(self._imname2) as im1:
            beams1 = im1.imageinfo()["perplanebeams"]
            for do_sky, outname in zip(
                [True, False], [self._outname2, self._outname2_no_sky]
            ):
                xds = read_image(self._imname2, do_sky_coords=do_sky)
                write_image(xds, outname, out_format="casa")
                with open_image_ro(outname) as im2:
                    beams2 = im2.imageinfo()["perplanebeams"]
                    for i in range(200):
                        beam = beams2[f"*{i}"]
                        beam["major"]["value"] *= 180 * 60 / np.pi
                        beam["major"]["unit"] = "arcmin"
                        beam["minor"]["value"] *= 180 * 60 / np.pi
                        beam["minor"]["unit"] = "arcmin"
                        beam["positionangle"]["value"] *= 180 / np.pi
                        beam["positionangle"]["unit"] = "deg"

                    self.dict_equality(beams1, beams2, "got", "expected")
        # convert to single beam image
        tb = tables.table(self._outname6, readonly=False)
        beam3 = {
            "major": {"unit": "arcsec", "value": 4.0},
            "minor": {"unit": "arcsec", "value": 3.0},
            "positionangle": {"unit": "deg", "value": 5.0},
        }
        tb.putkeyword(
            "imageinfo",
            {
                "imagetype": "Intensity",
                "objectname": "",
                "restoringbeam": beam3,
            },
        )
        xds = read_image(self._outname6)
        self.assertFalse("beam" in xds.attrs, "beam should not be in xds.attrs")
        expec = np.array(
            [4 / 180 / 3600 * np.pi, 3 / 180 / 3600 * np.pi, 5 * np.pi / 180]
        )
        for i, p in enumerate(["major", "minor", "pa"]):
            self.assertTrue(
                np.allclose(
                    xds.BEAM.sel(beam_param=p).values,
                    expec[i],
                ),
                f"Incorrect {p} axis",
            )

    def test_masking(self):
        """
        issue 48, proper nan masking when writing CASA images
        https://github.com/casangi/xradio/issues/48
        """
        download(self._imname3)
        for do_sky, outname, out_1, out_2 in zip(
            [True, False],
            [self._outname3, self._outname3_no_sky],
            [self._outname4, self._outname4_no_sky],
            [self._outname5, self._outname5_no_sky],
        ):
            # case 1: no mask + no nans = no mask
            xds = read_image(self._imname3, do_sky_coords=do_sky)
            first_attrs = xds.attrs
            t = copy.deepcopy(xds.attrs)
            c = copy.deepcopy(xds.coords)
            write_image(xds, outname, out_format="casa")
            subdirs = glob(f"{outname}/*/")
            subdirs = [d[d.index("/") + 1 : -1] for d in subdirs]
            self.assertEqual(
                subdirs,
                ["logtable"],
                f"Unexpected directory (mask?) found. subdirs is {subdirs}",
            )
            # case 2a: no mask + nans = nan_mask
            xds[sky][0, 1, 1, 1, 1] = float("NaN")
            shutil.rmtree(outname)
            second_attrs = xds.attrs
            second_coords = xds.coords
            self.dict_equality(t, second_attrs, "xds before", "xds after", ["history"])
            write_image(xds, outname, out_format="casa")
            subdirs = glob(f"{outname}/*/")
            subdirs = [d[d.index("/") + 1 : -1] for d in subdirs]
            subdirs.sort()
            self.assertEqual(
                subdirs,
                ["logtable", "mask_xds_nans"],
                f"Unexpected subdirectory list found. subdirs is {subdirs}",
            )
            with open_image_ro(outname) as im1:
                casa_mask = im1.getmask()
            self.assertTrue(casa_mask[1, 1, 1, 1], "Wrong pixels are masked")
            self.assertEqual(casa_mask.sum(), 1, "More pixels masked than expected")
            casa_mask[1, 1, 1, 1] = False
            # case 2b: mask + nans = nan_mask and (nan_mask or mask)
            # the first positional parameter is a dummy array, so make an
            # empty array
            data = da.zeros_like(
                np.array([]),
                shape=xds[sky].shape,
                chunks=xds[sky].chunks,
                dtype=bool,
            )
            data[0, 2, 2, 2, 2] = True
            mask0 = xr.DataArray(
                data=data,
                dims=xds[sky].sizes,
                coords=xds[sky].coords,
                attrs={image_type: "Mask"},
            )
            xds = xds.assign(mask0=mask0)
            xds["SKY"].attrs["active_mask"] = "mask0"
            write_image(xds, out_1, out_format="casa")
            self.assertEqual(
                xds["SKY"].attrs["active_mask"],
                "mask0",
                "SKY active mask was incorrectly reset",
            )
            subdirs = glob(f"{out_1}/*/")
            subdirs = [d[d.index("/") + 1 : -1] for d in subdirs]
            subdirs.sort()
            self.assertEqual(
                subdirs,
                ["logtable", "mask0", "mask_xds_nans", "mask_xds_nans_or_mask0"],
                f"Unexpected subdirectory list found. subdirs is {subdirs}",
            )
            with open_image_ro(out_1) as im1:
                # getmask() flips so True = bad, False = good
                casa_mask = im1.getmask()
                self.assertTrue(casa_mask[1, 1, 1, 1], "Wrong pixels are masked")
                self.assertTrue(casa_mask[2, 2, 2, 2], "Wrong pixels are masked")
                self.assertEqual(casa_mask.sum(), 2, "Wrong pixels are masked")
            # case 2c: all nans are already masked by default mask = no new masks are created
            xds[sky][0, 2, 2, 2, 2] = float("NaN")
            xds[sky][0, 1, 1, 1, 1] = 0
            write_image(xds, out_2, out_format="casa")
            self.assertEqual(
                xds["SKY"].attrs["active_mask"],
                "mask0",
                "SKY active mask was incorrectly reset",
            )
            subdirs = glob(f"{out_2}/*/")
            subdirs = [d[d.index("/") + 1 : -1] for d in subdirs]
            subdirs.sort()
            self.assertEqual(
                subdirs,
                ["logtable", "mask0"],
                f"Unexpected subdirectory list found. subdirs is {subdirs}",
            )
            with open_image_ro(out_2) as im1:
                casa_mask = im1.getmask()
            self.assertTrue(casa_mask[2, 2, 2, 2], "Wrong pixel masked")
            self.assertEqual(casa_mask.sum(), 1, "Wrong number of pixels masked")

    def test_output_uv_casa_image(self):
        image = self.uv_image()
        download(image)
        self.assertTrue(os.path.isdir(image), f"Cound not download {image}")
        xds = read_image(image)
        write_image(xds, self._output_uv, "casa")
        with open_image_ro(self._output_uv) as test_im:
            with open_image_ro(image) as expec_im:
                got = test_im.coordinates().dict()["linear0"]
                expec = expec_im.coordinates().dict()["linear0"]
                self.dict_equality(got, expec, "got uv", "expec uv")
                got = test_im.getdata()
                expec = test_im.getdata()
                self.assertTrue(np.isclose(got, expec).all(), "Incorrect pixel data")


class xds_to_zarr_to_xds_test(xds_from_image_test):
    """
    test xds -> zarr -> xds round trip
    """

    _zarr_store: str = "out.zarr"
    _zarr_uv_store: str = "out_uv.zarr"
    _zarr_beam_test: str = "beam_test.zarr"

    @classmethod
    def setUpClass(cls):
        # by default, subclass setUpClass() method is called before super class',
        # so we must explicitly call the super class' method here to create the
        # xds which is located in the super class
        super().setUpClass()
        write_image(cls.xds(), cls._zarr_store, out_format="zarr", overwrite=True)
        cls._zds = read_image(cls._zarr_store)

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        for f in [
            cls._zarr_store,
            cls._zarr_store + "_2",
            cls._zarr_uv_store,
            cls._zarr_beam_test,
        ]:
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
        self.compare_sky_mask(self._zds)

    def test_xds_time_vals(self):
        """Test xds has correct time axis values"""
        self.compare_time(self._zds)

    def test_xds_polarization_axis(self):
        """Test xds has correct stokes values"""
        self.compare_polarization(self._zds)

    def test_xds_frequency_axis(self):
        """Test xds has correct frequencyuency values and metadata"""
        self.compare_frequency(self._zds)

    def test_xds_vel_axis(self):
        """Test xds has correct velocity values and metadata"""
        self.compare_vel_axis(self._zds)

    def test_xds_l_m_axis(self):
        """Test xds has correct l and m values and attributes"""
        self.compare_l_m(self._zds)

    def test_xds_ra_dec_axis(self):
        """Test xds has correct RA and Dec values and attributes"""
        self.compare_ra_dec(self._zds)

    def test_xds_attrs(self):
        """Test xds level attributes"""
        print("$$########## Comparing xds attrs")
        import json

        print(
            "Expected attrs: %s",
            json.dumps(
                self.exp_xds_attrs(), indent=4, sort_keys=True, default=safe_convert
            ),
        )
        print(
            "Got attrs: %s",
            json.dumps(self._zds.attrs, indent=4, sort_keys=True, default=safe_convert),
        )
        print("******************")
        print(
            "Expected attrs: %s",
            json.dumps(
                self.exp_xds_attrs(), indent=4, sort_keys=True, default=safe_convert
            ),
        )
        print(
            "Got attrs: %s",
            json.dumps(
                self.exp_sky_attrs(), indent=4, sort_keys=True, default=safe_convert
            ),
        )
        print("$$##########")
        self.compare_xds_attrs(self._zds)
        self.compare_sky_attrs(self._zds.SKY)
        # raise Exception("stop here")

    def test_get_img_ds_block(self):
        self.compare_image_block(self._zarr_store, zarr=True)

    def test_output_uv_zarr_image(self):
        image = self.uv_image()
        download(image)
        self.assertTrue(os.path.isdir(image), f"Cound not download {image}")
        xds = read_image(image)
        write_image(xds, self._zarr_uv_store, "zarr")
        xds2 = read_image(self._zarr_uv_store)
        self.assertTrue(
            np.isclose(xds2.APERTURE.values, xds.APERTURE.values).all(),
            "Incorrect aperture pixel values",
        )

    def test_beam(self):
        mb = np.zeros(shape=[1, 10, 4, 3], dtype=float)
        mb[:, :, :, 0] = 0.00001
        mb[:, :, :, 1] = 0.00002
        mb[:, :, :, 2] = 0.00003
        xdb = xr.DataArray(mb, dims=["time", "frequency", "polarization", "beam_param"])
        xdb = xdb.rename("BEAM")
        # xdb = xdb.assign_coords(beam_param=["major", "minor", "pa"])
        xdb.attrs["units"] = "rad"
        xds = copy.deepcopy(self.xds())
        xds["BEAM"] = xdb
        write_image(xds, self._zarr_beam_test, "zarr")
        xds2 = read_image(self._zarr_beam_test)
        self.assertTrue(
            np.allclose(xds2.BEAM.values, xds.BEAM.values), "Incorrect beam values"
        )


class fits_to_xds_test(xds_from_image_test):
    """
    test fits_to_xds
    """

    _imname1: str = "demo_simulated.im"
    _outname1: str = "demo_simulated.fits"
    _compressed_fits: str = "compressed.fits"
    _bzero: str = "bzero.fits"
    _bscale: str = "bscale.fits"

    @classmethod
    def setUpClass(cls):
        # by default, subclass setUpClass() method is called before super class',
        # so we must explicitly call the super class' method here to create the
        # xds which is located in the super class
        super().setUpClass()
        assert os.path.exists(cls.infits()), f"{cls.infits()} does not exist"
        cls._fds = read_image(cls.infits(), {"frequency": 5}, do_sky_coords=True)
        cls._fds_no_sky = read_image(
            cls.infits(), {"frequency": 5}, do_sky_coords=False
        )

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        for f in [
            cls._imname1,
            cls._outname1,
            cls._compressed_fits,
            cls._bzero,
            cls._bscale,
        ]:
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
        for fds in (self._fds, self._fds_no_sky):
            self.compare_sky_mask(fds, True)

    def test_xds_time_axis(self):
        """Test values and attributes on the time axis"""
        for fds in (self._fds, self._fds_no_sky):
            self.compare_time(fds)

    def test_xds_polarization_axis(self):
        """Test xds has correct stokes values"""
        for fds in (self._fds, self._fds_no_sky):
            self.compare_polarization(fds)

    def test_xds_frequency_axis(self):
        """Test xds has correct frequency values and metadata"""
        for fds in (self._fds, self._fds_no_sky):
            self.compare_frequency(fds)

    def test_xds_vel_axis(self):
        """Test xds has correct velocity values and metadata"""
        for fds in (self._fds, self._fds_no_sky):
            self.compare_vel_axis(fds, True)

    def test_xds_l_m_axis(self):
        """Test xds has correct l and m values and attributes"""
        for fds in (self._fds, self._fds_no_sky):
            self.compare_l_m(fds)

    def test_xds_ra_dec_axis(self):
        """Test xds has correct RA and Dec values and attributes"""
        for i, fds in enumerate([self._fds, self._fds_no_sky]):
            if i == 0:
                self.compare_ra_dec(fds)
            else:
                for c in ["right_ascension", "declination"]:
                    self.assertTrue(
                        c not in fds.coords, f"{c} in coords but should not be"
                    )

    def test_xds_beam_param_axis(self):
        for fds in (self._fds, self._fds_no_sky):
            self.assertTrue(
                "beam_param" in fds.coords,
                "beam_param not in coords",
            )
            self.assertTrue(
                (fds.beam_param == ["major", "minor", "pa"]).all(),
                "Incorrect beam_param values",
            )

    def test_xds_attrs(self):
        """Test xds level attributes"""
        for fds in (self._fds, self._fds_no_sky):
            self.compare_xds_attrs(fds)
            self.compare_sky_attrs(fds.SKY, True)

    def test_multibeam(self):
        download(self._imname1)
        self.assertTrue(
            os.path.exists(self._imname1), f"{self._imname1} not downloaded"
        )
        with open_image_ro(self._imname1) as casa_image:
            casa_image.tofits(self._outname1)
            expec = casa_image.imageinfo()["perplanebeams"]
        xds = read_image(self._outname1)
        got = xds.BEAM
        for p in range(4):
            for c in range(50):
                for b in ["major", "minor", "pa"]:
                    bb = "positionangle" if b == "pa" else b
                    expec_comp = expec[f"*{p*50+c}"][bb]
                    expec_comp = (
                        (expec_comp["value"] * u.Unit(expec_comp["unit"]))
                        .to("rad")
                        .value,
                    )
                    got_comp = got.isel(dict(time=0, polarization=p, frequency=c)).sel(
                        dict(beam_param=b)
                    )
                self.assertTrue(
                    np.isclose(got_comp, expec_comp),
                    f"Incorrect {b} value for polarization {p}, channel {c}. "
                    f"{got_comp.item()} rad vs {expec_comp} rad.",
                )

    def test_compute_mask(self):
        """
        Test compute_mask parameter
        """
        for compute_mask in [True, False]:
            fds = read_image(self.infits(), {"frequency": 5}, compute_mask=compute_mask)
            if compute_mask:
                self.assertTrue(
                    "MASK0" in fds.data_vars,
                    "MASK0 should be in data_vars, but is not",
                )
            else:
                self.assertTrue(
                    "MASK0" not in fds.data_vars,
                    "MASK0 should not be in data_vars, but is",
                )

    def test_compressed_fits_guard(self):
        """
        Test reading compressed FITS files fails because not supported by memmapping
        """
        # make a compressed fits file from what we already have access to
        with fits.open(self._infits) as hdulist:
            data = hdulist[0].data
            header = hdulist[0].header

        # Wrap in compressed HDU
        primary_hdu = fits.PrimaryHDU()
        comp_hdu = fits.CompImageHDU(data=data, header=header)
        fits.HDUList([primary_hdu, comp_hdu]).writeto(
            self._compressed_fits, overwrite=True
        )
        # Ensure the file was saved as a CompImageHDU (not auto-promoted to PrimaryHDU)
        with fits.open(self._compressed_fits, do_not_scale_image_data=True) as hdulist:
            assert isinstance(
                hdulist[1], fits.CompImageHDU
            ), "Expected CompImageHDU in HDU[1]"

        with pytest.raises(RuntimeError) as exc_info:
            read_image(self._compressed_fits, {"frequency": 5})

        self.assertTrue(
            re.search(r"name=COMPRESSED_IMAGE", str(exc_info.value)),
            f"Expected error about COMPRESSED_IMAGE HDU, but got {str(exc_info.value)}",
        )

    def _create_bzero_bscale_image(
        self, outname: str, bzero: float, bscale: float
    ) -> None:
        with fits.open(self._infits) as hdulist:
            data = hdulist[0].data
            self._clean_data(data)

        hdu = fits.PrimaryHDU(data=data)
        # Apply scaling explicitly. This enables BSCALE/BZERO generation
        hdu.header["BSCALE"] = bscale
        hdu.header["BZERO"] = bzero
        hdu.writeto(outname, overwrite=True)

    def _clean_data(self, data: np.ndarray) -> None:
        """
        Clean data to the range of int16 to avoid issues with BSCALE/BZERO.
        """
        # force astropy.io.fits to scale data so BZERO and BSCALE are set,
        # will be omitted from new header otherwise on write
        data = np.nan_to_num(data, nan=0.0)  # replace NaNs with 0
        data = np.clip(data, -32768, 32767)  # clip to int16 range
        data = data.astype(np.int16)

    def test_bzero_guard(self):
        """
        Test reading FITS files with bzero != 0 fails
        """
        self._create_bzero_bscale_image(self._bzero, bzero=5.0, bscale=1.0)
        self.assertTrue(os.path.exists(self._bzero), f"{self._bzero} was not written")
        with pytest.raises(RuntimeError) as exc_info:
            fds = read_image(self._bzero)
        self.assertTrue(
            re.search(r"BSCALE/BZERO set", str(exc_info.value)),
            f"Expected error about BSCALE/BZERO, but got {str(exc_info.value)}",
        )

    def test_bscale_guard(self):
        """
        Test reading FITS files with bscale != 1 fails
        """
        self._create_bzero_bscale_image(self._bscale, bzero=0.0, bscale=2.0)
        self.assertTrue(os.path.exists(self._bscale), f"{self._bzero} was not written")
        with pytest.raises(RuntimeError) as exc_info:
            fds = read_image(self._bscale)
        self.assertTrue(
            re.search(r"BSCALE/BZERO set", str(exc_info.value)),
            f"Expected error about BSCALE/BZERO, but got {str(exc_info.value)}",
        )

    # TODO
    # def test_get_img_ds_block(self):
    #    # self.compare_image_block(self.imname())
    #    pass


class make_empty_image_tests(ImageBase):
    @classmethod
    def create_image(cls, code, do_sky_coords=None):
        args = [
            [0.2, -0.5],
            [10, 10],
            [np.pi / 180 / 60, np.pi / 180 / 60],
            [1.412e9, 1.413e9],
            ["I", "Q", "U"],
            [54000.1],
        ]
        kwargs = {} if do_sky_coords is None else {"do_sky_coords": do_sky_coords}
        return code(*args, **kwargs)

    def run_time_tests(self, skel):
        self.assertTrue(
            np.isclose(skel.time, [54000.1]).all(),
            "Incorrect time coordinate values",
        )
        expec = {"scale": "utc", "units": "d", "format": "mjd"}
        self.dict_equality(skel.time.attrs, expec, "got", "expected")

    def run_polarization_tests(self, skel):
        self.assertTrue(
            (skel.polarization == ["I", "Q", "U"]).all(),
            "Incorrect polarization coordinate values",
        )

    def run_frequency_tests(self, skel):
        expec = {
            "observer": "lsrk",
            "reference_frequency": {
                "attrs": {
                    "observer": "lsrk",
                    "type": "spectral_coord",
                    "units": "Hz",
                },
                "data": 1413000000.0,
                "dims": [],
            },
            "rest_frequencies": {
                "attrs": {
                    "type": "quantity",
                    "units": "Hz",
                },
                "data": 1413000000.0,
                "dims": [],
            },
            "rest_frequency": {
                "attrs": {
                    "type": "quantity",
                    "units": "Hz",
                },
                "data": 1413000000.0,
                "dims": [],
            },
            "type": "spectral_coord",
            "units": "Hz",
            "wave_units": "mm",
        }
        self.assertTrue(
            np.isclose(skel.frequency, [1.412e09, 1.413e09]).all(),
            "Incorrect frequency coordinate values",
        )
        self.dict_equality(skel.frequency.attrs, expec, "got", "expected")

    def run_velocity_tests(self, skel):
        expec = {"doppler_type": "radio", "units": "m/s", "type": "doppler"}
        self.assertTrue(
            np.isclose(skel.velocity, [212167.34465675, 0]).all(),
            "Incorrect velocity coordinate values",
        )
        self.dict_equality(skel.velocity.attrs, expec, "got", "expected")

    def run_l_m_tests(self, skel):
        cdelt = np.pi / 180 / 60
        expec = {"m": np.array([(i - 5) * cdelt for i in range(10)])}
        expec["l"] = -expec["m"]
        expec_attrs = {
            "l": {
                # "type": "quantity",
                # "crval": 0.0,
                # "cdelt": -cdelt,
                # "units": "rad",
                "note": "l is the angle measured from the phase center to the east. "
                "So l = x*cdelt, where x is the number of pixels from the phase center. "
                "See AIPS Memo #27, Section III.",
            },
            "m": {
                # "type": "quantity",
                # "crval": 0.0,
                # "cdelt": cdelt,
                # "units": "rad",
                "note": "m is the angle measured from the phase center to the north. "
                "So m = y*cdelt, where y is the number of pixels from the phase center. "
                "See AIPS Memo #27, Section III.",
            },
        }
        for c in ["l", "m"]:
            self.assertTrue(
                np.isclose(skel[c].values, expec[c]).all(),
                f"Incorrect {c} coord values",
            )
            self.dict_equality(
                skel[c].attrs, expec_attrs[c], f"got {c} attrs", f"expec {c} attrs"
            )

    def run_right_ascension_tests(self, skel, do_sky_coords):
        expec = [
            [
                0.20165865,
                0.20165838,
                0.20165812,
                0.20165785,
                0.20165759,
                0.20165733,
                0.20165706,
                0.2016568,
                0.20165654,
                0.20165628,
            ],
            [
                0.20132692,
                0.20132671,
                0.20132649,
                0.20132628,
                0.20132607,
                0.20132586,
                0.20132565,
                0.20132544,
                0.20132523,
                0.20132502,
            ],
            [
                0.20099519,
                0.20099503,
                0.20099487,
                0.20099471,
                0.20099455,
                0.2009944,
                0.20099424,
                0.20099408,
                0.20099392,
                0.20099377,
            ],
            [
                0.20066346,
                0.20066335,
                0.20066325,
                0.20066314,
                0.20066304,
                0.20066293,
                0.20066283,
                0.20066272,
                0.20066262,
                0.20066251,
            ],
            [
                0.20033173,
                0.20033168,
                0.20033162,
                0.20033157,
                0.20033152,
                0.20033147,
                0.20033141,
                0.20033136,
                0.20033131,
                0.20033126,
            ],
            [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
            [
                0.19966827,
                0.19966832,
                0.19966838,
                0.19966843,
                0.19966848,
                0.19966853,
                0.19966859,
                0.19966864,
                0.19966869,
                0.19966874,
            ],
            [
                0.19933654,
                0.19933665,
                0.19933675,
                0.19933686,
                0.19933696,
                0.19933707,
                0.19933717,
                0.19933728,
                0.19933738,
                0.19933749,
            ],
            [
                0.19900481,
                0.19900497,
                0.19900513,
                0.19900529,
                0.19900545,
                0.1990056,
                0.19900576,
                0.19900592,
                0.19900608,
                0.19900623,
            ],
            [
                0.19867308,
                0.19867329,
                0.19867351,
                0.19867372,
                0.19867393,
                0.19867414,
                0.19867435,
                0.19867456,
                0.19867477,
                0.19867498,
            ],
        ]
        if do_sky_coords:
            self.assertTrue(
                np.isclose(skel.right_ascension, expec).all(),
                "Incorrect right_ascension coordinate values",
            )
            self.assertEqual(
                skel.right_ascension.attrs,
                {},
                "right ascension has non-empty attrs dict but it should be empty",
            )
        else:
            self.assertTrue(
                "right_ascension" not in skel.coords,
                "right_ascension is incorrectly in coords",
            )

    def run_declination_tests(self, skel, do_sky_coords):
        expec = [
            [
                -0.50145386,
                -0.50116297,
                -0.50087209,
                -0.5005812,
                -0.50029031,
                -0.49999942,
                -0.49970853,
                -0.49941765,
                -0.49912676,
                -0.49883587,
            ],
            [
                -0.50145407,
                -0.50116318,
                -0.50087229,
                -0.50058141,
                -0.50029052,
                -0.49999963,
                -0.49970874,
                -0.49941785,
                -0.49912697,
                -0.49883608,
            ],
            [
                -0.50145423,
                -0.50116334,
                -0.50087246,
                -0.50058157,
                -0.50029068,
                -0.49999979,
                -0.4997089,
                -0.49941802,
                -0.49912713,
                -0.49883624,
            ],
            [
                -0.50145435,
                -0.50116346,
                -0.50087257,
                -0.50058168,
                -0.5002908,
                -0.49999991,
                -0.49970902,
                -0.49941813,
                -0.49912724,
                -0.49883635,
            ],
            [
                -0.50145442,
                -0.50116353,
                -0.50087264,
                -0.50058175,
                -0.50029087,
                -0.49999998,
                -0.49970909,
                -0.4994182,
                -0.49912731,
                -0.49883642,
            ],
            [
                -0.50145444,
                -0.50116355,
                -0.50087266,
                -0.50058178,
                -0.50029089,
                -0.5,
                -0.49970911,
                -0.49941822,
                -0.49912734,
                -0.49883645,
            ],
            [
                -0.50145442,
                -0.50116353,
                -0.50087264,
                -0.50058175,
                -0.50029087,
                -0.49999998,
                -0.49970909,
                -0.4994182,
                -0.49912731,
                -0.49883642,
            ],
            [
                -0.50145435,
                -0.50116346,
                -0.50087257,
                -0.50058168,
                -0.5002908,
                -0.49999991,
                -0.49970902,
                -0.49941813,
                -0.49912724,
                -0.49883635,
            ],
            [
                -0.50145423,
                -0.50116334,
                -0.50087246,
                -0.50058157,
                -0.50029068,
                -0.49999979,
                -0.4997089,
                -0.49941802,
                -0.49912713,
                -0.49883624,
            ],
            [
                -0.50145407,
                -0.50116318,
                -0.50087229,
                -0.50058141,
                -0.50029052,
                -0.49999963,
                -0.49970874,
                -0.49941785,
                -0.49912697,
                -0.49883608,
            ],
        ]
        expec2 = {
            "data": [0.2, -0.5],
            "dims": ["l", "m"],
            "attrs": {
                "type": "sky_coord",
                "frame": "fk5",
                "equinox": "j2000.0",
                "units": "rad",
            },
        }
        if do_sky_coords:
            self.assertTrue(
                np.isclose(skel.declination, expec).all(),
                "Incorrect declinationion coordinate values",
            )
            self.assertEqual(
                skel.declination.attrs,
                {},
                "declination attrs dict is not empty but it should be empty",
            )
        else:
            self.assertTrue(
                "declination" not in skel.coords,
                "declination incorrectly in coords",
            )
        self.dict_equality(
            skel.attrs["direction"]["reference"], expec2, "got", "expected"
        )

    def run_u_v_tests(self, skel):
        cdelt = 180 * 60 / np.pi / 10
        expec = np.array([(i - 5) * cdelt for i in range(10)])
        ref_val = {
            "data": 0.0,
            "dims": [],
            "attrs": {
                "type": "quantity",
                "units": "lambda",
            },
        }
        expec_attrs = {
            "u": ref_val,
            "v": ref_val,
        }
        for c in ["u", "v"]:
            self.assertTrue(
                np.isclose(skel[c].values, expec).all(),
                f"Incorrect {c} coord values, {skel[c].values} vs {expec}.",
            )
            self.dict_equality(
                skel[c].attrs, expec_attrs[c], f"got {c} attrs", "expec {c} attrs"
            )

    def run_attrs_tests(self, skel):
        direction = {
            "latpole": {
                "data": 0.0,
                "dims": ["l", "m"],
                "attrs": {
                    "type": "quantity",
                    "units": "rad",
                },
            },
            "lonpole": {
                "data": np.pi,
                "dims": ["l", "m"],
                "attrs": {
                    "type": "quantity",
                    "units": "rad",
                },
            },
            "pc": [[1.0, 0.0], [0.0, 1.0]],
            # 'primary_beam_center': {
            #     'attrs': {
            #         'initial': True,
            #         'type': 'sky_coord',
            #         'frame': 'fk5',
            #         'equinox': 'j2000.0',
            #         'units': 'rad'
            #     }
            #     'data': [0.2, -0.5],
            #     'dims': ["l", "m"],
            # },
            "projection": "SIN",
            "projection_parameters": [0.0, 0.0],
            "reference": {
                "attrs": {
                    "type": "sky_coord",
                    "frame": "fk5",
                    "equinox": "j2000.0",
                    "units": "rad",
                },
                "data": [0.2, -0.5],
                "dims": ["l", "m"],
            },
        }
        data_groups = {"base": {}}
        expec = {
            "data_groups": data_groups,
            "direction": direction,
        }
        self.dict_equality(skel.attrs, expec, "got", "expected")


class make_empty_sky_image_tests(make_empty_image_tests):
    """Test making skeleton image"""

    @classmethod
    def setUpClass(cls):
        cls._skel_im = make_empty_image_tests.create_image(make_empty_sky_image, True)
        cls._skel_im_no_sky = make_empty_image_tests.create_image(
            make_empty_sky_image, False
        )

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()

    @classmethod
    def skel_im(cls):
        return cls._skel_im

    def skel_im_no_sky(self):
        return self._skel_im_no_sky

    def test_dims_and_coords(self):
        for skel in [self.skel_im(), self.skel_im_no_sky()]:
            self.assertEqual(
                list(skel.sizes.keys()),
                ["time", "frequency", "polarization", "l", "m", "beam_param"],
                "Incorrect dims",
            )
        self.assertEqual(
            list(self.skel_im().coords.keys()),
            [
                "time",
                "frequency",
                "velocity",
                "polarization",
                "l",
                "m",
                "right_ascension",
                "declination",
                "beam_param",
            ],
            "Incorrect coords",
        )
        self.assertEqual(
            list(self.skel_im_no_sky().coords.keys()),
            ["time", "frequency", "velocity", "polarization", "l", "m", "beam_param"],
            "Incorrect coords",
        )

    def test_time_coord(self):
        for skel in [self.skel_im(), self.skel_im_no_sky()]:
            self.run_time_tests(skel)

    def test_polarization_coord(self):
        for skel in [self.skel_im(), self.skel_im_no_sky()]:
            self.run_polarization_tests(skel)

    def test_frequency_coord(self):
        for skel in [self.skel_im(), self.skel_im_no_sky()]:
            self.run_frequency_tests(skel)

    def test_vel_coord(self):
        for skel in [self.skel_im(), self.skel_im_no_sky()]:
            self.run_velocity_tests(skel)

    def test_l_m_coord(self):
        for skel in [self.skel_im(), self.skel_im_no_sky()]:
            self.run_l_m_tests(skel)

    def test_right_ascension_coord(self):
        for i, skel in enumerate([self.skel_im(), self.skel_im_no_sky()]):
            self.run_right_ascension_tests(skel, do_sky_coords=i == 0)

    def test_declination_coord(self):
        for i, skel in enumerate([self.skel_im(), self.skel_im_no_sky()]):
            self.run_declination_tests(skel, do_sky_coords=i == 0)

    def test_attrs(self):
        for skel in [self.skel_im(), self.skel_im_no_sky()]:
            self.run_attrs_tests(skel)


class make_empty_aperture_image_tests(make_empty_image_tests):
    """Test making skeleton image"""

    @classmethod
    def setUpClass(cls):
        cls._skel_im = make_empty_image_tests.create_image(
            make_empty_aperture_image, None
        )

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()

    def skel_im(self):
        return self._skel_im

    def test_dims_and_coords(self):
        self.assertEqual(
            list(self.skel_im().sizes.keys()),
            ["time", "frequency", "polarization", "u", "v", "beam_param"],
            "Incorrect dims",
        )
        self.assertEqual(
            list(self.skel_im().coords.keys()),
            ["time", "frequency", "velocity", "polarization", "u", "v", "beam_param"],
            "Incorrect coords",
        )

    def test_time_coord(self):
        skel = self.skel_im()
        self.run_time_tests(skel)

    def test_polarization_coord(self):
        skel = self.skel_im()
        self.run_polarization_tests(skel)

    def test_frequency_coord(self):
        skel = self.skel_im()
        self.run_frequency_tests(skel)

    def test_vel_coord(self):
        skel = self.skel_im()
        self.run_velocity_tests(skel)

    def test_u_v_coord(self):
        skel = self.skel_im()
        self.run_u_v_tests(skel)

    def test_attrs(self):
        skel = self.skel_im()
        self.run_attrs_tests(skel)


class make_empty_lmuv_image_tests(make_empty_image_tests):
    """Tests making image with l, m, u, v coordinates"""

    @classmethod
    def setUpClass(cls):
        cls._skel_im = make_empty_image_tests.create_image(make_empty_lmuv_image, True)
        cls._skel_im_no_sky = make_empty_image_tests.create_image(
            make_empty_lmuv_image, False
        )

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()

    def skel_im(self):
        return self._skel_im

    def skel_im_no_sky(self):
        return self._skel_im_no_sky

    def test_dims(self):
        for skel in [self.skel_im(), self.skel_im_no_sky()]:
            self.assertEqual(
                tuple(skel.sizes.keys()),
                ("time", "frequency", "polarization", "l", "m", "u", "v", "beam_param"),
                "Incorrect sizes",
            )
        self.assertEqual(
            list(self.skel_im().coords.keys()),
            [
                "time",
                "frequency",
                "velocity",
                "polarization",
                "l",
                "m",
                "right_ascension",
                "declination",
                "u",
                "v",
                "beam_param",
            ],
            "Incorrect coords",
        )
        self.assertEqual(
            list(self.skel_im_no_sky().coords.keys()),
            [
                "time",
                "frequency",
                "velocity",
                "polarization",
                "l",
                "m",
                "u",
                "v",
                "beam_param",
            ],
            "Incorrect coords",
        )

    def test_time_coord(self):
        skel = self.skel_im()
        self.run_time_tests(skel)

    def test_polarization_coord(self):
        skel = self.skel_im()
        self.run_polarization_tests(skel)

    def test_frequency_coord(self):
        skel = self.skel_im()
        self.run_frequency_tests(skel)

    def test_vel_coord(self):
        skel = self.skel_im()
        self.run_velocity_tests(skel)

    def test_l_m_coord(self):
        for skel in [self.skel_im(), self.skel_im_no_sky()]:
            self.run_l_m_tests(skel)

    def test_right_ascension_coord(self):
        for i, skel in enumerate([self.skel_im(), self.skel_im_no_sky()]):
            self.run_right_ascension_tests(skel, do_sky_coords=i == 0)

    def test_declination_coord(self):
        for i, skel in enumerate([self.skel_im(), self.skel_im_no_sky()]):
            self.run_declination_tests(skel, do_sky_coords=i == 0)

    def test_u_v_coord(self):
        skel = self.skel_im()
        self.run_u_v_tests(skel)

    def test_attrs(self):
        skel = self.skel_im()
        self.run_attrs_tests(skel)


class write_image_test(xds_from_image_test):

    _myout = "zk.im"

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        write_image(cls.xds(), cls._myout, out_format="casa")

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        for f in [cls._myout]:
            if os.path.exists(f):
                if os.path.isdir(f):
                    shutil.rmtree(f)
                else:
                    os.remove(f)

    def test_overwrite(self):
        # test overwrite=True
        write_image(self.xds(), self._myout, out_format="casa", overwrite=True)
        # test overwrite=False
        with self.assertRaises(FileExistsError):
            write_image(self.xds(), self._myout, out_format="casa", overwrite=False)
        with self.assertRaises(FileExistsError):
            # default overwrite=False
            write_image(self.xds(), self._myout, out_format="casa")


if __name__ == "__main__":
    pytest.main(["-v", "-s", __file__])
