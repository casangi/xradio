try:
    from casacore import images, tables
except ImportError:
    import xradio._utils._casacore.casacore_from_casatools as images
    import xradio._utils._casacore.casacore_from_casatools as tables

from astropy.io import fits
import astropy.units as u
from copy import deepcopy
import dask.array as da
from glob import glob
import numpy as np
import numpy.ma as ma
import numbers
import os
import pytest
import re
import shutil
import unittest
import xarray as xr

from toolviper.dask.client import local_client
from toolviper.utils.data import download

from xradio.image import (
    load_image, make_empty_aperture_image, make_empty_lmuv_image,
    make_empty_sky_image, open_image, write_image
)
from xradio.image._util._casacore.common import _create_new_image as create_new_image
from xradio.image._util._casacore.common import _open_image_ro as open_image_ro
from xradio.image._util._casacore.common import _object_name
from xradio.image._util.common import _image_type as image_type
from xradio._utils._casacore.tables import open_table_ro

from xradio.testing.assertions import assert_xarray_datasets_equal

sky = "SKY"

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

def _remove(filename):
    if os.path.exists(filename):
        if os.path.isdir(filename):
            shutil.rmtree(filename)
        else:
            os.remove(filename)

def clean_path_logic(text: str) -> str:
    """Cleans the subtable path logic string by isolating the basename."""
    prefix = "Table: "
    if text.startswith(prefix):
        raw_path = text.removeprefix(prefix).strip()
        base_name = os.path.basename(raw_path.rstrip("/"))
        return f"Table: {base_name}"
    return text

@pytest.mark.usefixtures("dask_client_module")
class ImageBase(unittest.TestCase):
    def dict_equality(
        self,
        dict1,
        dict2,
        dict1_name,
        dict2_name,
        exclude_keys=None,
        common_keys_only=False,
    ):
        exclude_keys = exclude_keys or []
        if not common_keys_only:
            self.assertEqual(
                dict1.keys(),
                dict2.keys(),
                f"{dict1_name} has different keys than {dict2_name}:"
                f"\n{dict1.keys()} vs\n {dict2.keys()}",
            )
        for k in dict1.keys():
            if k in exclude_keys or (common_keys_only and k not in dict2):
                continue
            one = dict1[k]
            two = dict2[k]
            if isinstance(one, numbers.Number) and isinstance(two, numbers.Number):
                self.assertTrue(
                    np.isclose(one, two),
                    f"{dict1_name}[{k}] != {dict2_name}[{k}]:\n" + f"{one} vs\n{two}",
                )
            elif (isinstance(one, list) or isinstance(one, np.ndarray)) and (
                isinstance(two, list) or isinstance(two, np.ndarray)
            ):
                if len(one) == 0 or len(two) == 0:
                    self.assertEqual(
                        len(one),
                        len(two),
                        f"{dict1_name}[{k}] != {dict2_name}[{k}], " f"{one} != {two}",
                    )
                elif isinstance(one[0], numbers.Number):
                    self.assertTrue(
                        np.isclose(
                            np.array(one), np.array(two), rtol=1e-3, atol=1e-7
                        ).all(),
                        f"{dict1_name}[{k}] != {dict2_name}[{k}], " f"{one} != {two}",
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
                        common_keys_only=common_keys_only,
                        exclude_keys=exclude_keys,
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
                elif isinstance(one, str) and isinstance(two, str):
                    one_cleaned = clean_path_logic(one)
                    two_cleaned = clean_path_logic(two)
                    self.assertEqual(
                        one_cleaned,
                        two_cleaned,
                        f"{dict1_name}[{k}] != {dict2_name}[{k}]:\n"
                        + f"{one_cleaned} vs\n{two_cleaned}",
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
    _infits: str = "inp.im.fits"
    _uv_image: str = "complex_valued_uv.im"

    _xds = None
    _xds_no_sky = None
    _xds_uv = None

    _xds_true = None
    _xds_no_sky_true = None
    _xds_uv_true = None

    _xds_from_casa_true: str = "casa_to_xds_true.zarr"
    _xds_from_no_sky_casa_true: str = "casa_no_sky_to_xds_true.zarr"
    _xds_from_casa_uv_true: str = "casa_uv_true.zarr"

    @classmethod
    def setUpClass(cls):
        cls.maxDiff = None
        cls._make_image()

    @classmethod
    def tearDownClass(cls):
        for f in [
            # cls._imname,
            cls._imname + "_2",
            cls._outname,
            # cls._infits,
            cls._uv_image,
        ]:
            _remove(f)

    @classmethod
    def imname(cls):
        return cls._imname

    @classmethod
    def outname(cls):
        if not os.path.exists(cls._outname):
            # make sure xds is loaded
            xds = cls.xds()
            write_image(xds, cls._outname, out_format="casa", overwrite=True)
            assert os.path.exists(cls._outname), f"Could not create {cls._outname}"
        return cls._outname

    @classmethod
    def uv_image(cls):
        return cls._uv_image

    @classmethod
    def _make_image(cls):
        if not os.path.exists(cls.imname()):
            shape: list[int] = [10, 4, 20, 30]
            mask: np.ndarray = np.array(
                [i % 3 == 0 for i in range(np.prod(shape))], dtype=bool
            ).reshape(shape)
            pix: np.ndarray = np.array([range(np.prod(shape))], dtype=np.float64).reshape(
                shape
            )
            masked_array = ma.masked_array(pix, mask)
            with create_new_image(cls._imname, shape=shape, mask="MASK_0") as im:
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
            # with open_image_ro(cls._imname) as im:
            #    im.tofits(cls._infits)
            #    assert os.path.exists(cls._infits), f"Could not create {cls._infits}"

            # cls._xds_no_sky = open_image(cls._imname, {"frequency": 5}, False, False)

    def compare_image_block(self, imagename, zarr=False):
        x = [0] if zarr else [0, 1]
        full_xds = open_image(imagename)
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
            dims=["time", "frequency", "polarization", "beam_params_label"],
            coords={
                "time": full_xds.time,
                "frequency": full_xds.frequency,
                "polarization": full_xds.polarization,
                "beam_params_label": ["major", "minor", "pa"],
            },
        )
        full_xds["BEAM_FIT_PARAMS"] = xda
        full_xds["BEAM_FIT_PARAMS"].attrs["units"] = "rad"
        imag = imagename + "_2"

        write_image(
            full_xds, imag, out_format="zarr" if zarr else "casa", overwrite=True
        )

        xds = load_image(
            imag,
            {
                "l": slice(2, 10),
                "m": slice(3, 15),
                "polarization": slice(0, 1),
                "frequency": slice(0, 4),
            },
            do_sky_coords=True,
        )
        true_xds = full_xds.isel(
            polarization=slice(0, 1), frequency=slice(0, 4), l=slice(2, 10),
            m=slice(3, 15)
        )
        assert_xarray_datasets_equal(xds, true_xds)

    @classmethod
    def xds(cls):
        if not cls._xds:
            cls._make_image()
            cls._xds = open_image(cls.imname(), {"frequency": 5})
        return cls._xds

    @classmethod
    def xds_no_sky(cls):
        if not cls._xds_no_sky:
            cls._make_image()
            cls._xds_no_sky = open_image(cls.imname(), {"frequency": 5}, False, False)
        return cls._xds_no_sky

    @classmethod
    def xds_uv(cls):
        if not cls._xds_uv:
            download(cls.uv_image())
            assert os.path.exists(cls.uv_image()), f"Cound not download {cls.uv_image()}"
            cls._xds_uv = open_image(cls.uv_image())
        return cls._xds_uv

    @classmethod
    def infits(cls):
        if not os.path.exists(cls._infits):
            with open_image_ro(cls.imname()) as im:
                im.tofits(cls._infits)
                assert os.path.exists(cls._infits), f"Could not create {cls._infits}"
        return cls._infits

    @classmethod
    def true_xds(cls):
        if not cls._xds_true:
            # download
            cls._xds_true = xr.open_zarr(cls._xds_from_casa_true)
        return cls._xds_true

    @classmethod
    def true_no_sky_xds(cls):
        if not cls._xds_no_sky_true:
            # download
            cls._xds_no_sky_true = xr.open_zarr(cls._xds_from_no_sky_casa_true)
        return cls._xds_no_sky_true

    @classmethod
    def true_uv_xds(cls):
        if not cls._xds_uv_true:
            # download
            cls._xds_uv_true = xr.open_zarr(cls._xds_from_casa_uv_true)
        return cls._xds_uv_true

class casa_image_to_xds_test(xds_from_image_test):
    """
    test casa_image_to_xds
    """

    def test_got_xds(self):
        assert_xarray_datasets_equal(
            self.xds(), self.true_xds()
        )

    def test_got_xds_no_sky(self):
        assert_xarray_datasets_equal(
            self.xds_no_sky(), self.true_no_sky_xds()
        )

    def test_uv_image(self):
        assert_xarray_datasets_equal(
            self.xds_uv(), self.true_uv_xds()
        )

class xds_to_casacore(xds_from_image_test):
    _outname = "rabbit.im"
    _outname2 = "rabbit2.im"

    @classmethod
    def _clean(cls):
        for f in [cls._outname, cls._outname2]:
            _remove(f)

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
        write_image(xds, self._outname, "casa", overwrite=True)
        with open_image_ro(self._outname) as im:
            p = im.getdata()
        sky_name = xds.attrs.get("data_groups", {}).get("base", {}).get("sky", "SKY")
        exp_data = np.squeeze(np.transpose(xds[sky_name], [1, 2, 4, 3, 0]), 4)
        self.assertTrue((p == exp_data).all(), "Incorrect pixel values")

    def test_object_name_not_present(self):
        """
        Test writing an xds which does not have an object name
        to a casa image.
        """
        xds = deepcopy(self.xds())

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
        cls.xds_no_sky()
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
        """Test to verify metadata in two casacore images is the same."""
        f = 180 * 60 / np.pi
        with open_image_ro(self.imname()) as im1:
            c1 = im1.info()
            for imname in [self.outname(), self._outname_no_sky]:
                self.assertTrue(os.path.exists(imname), f"Output image {imname} does not exist")
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

        # Also check the table keywords
        with open_table_ro(self.imname()) as tb1:
            for imname in [self.outname(), self._outname_no_sky]:
                with open_table_ro(imname) as tb2:
                    kw1 = tb1.getkeywords()
                    kw2 = tb2.getkeywords()
                    self.dict_equality(
                        kw2,
                        kw1,
                        "got",
                        "expected",
                        common_keys_only=True,
                        exclude_keys=[
                            "cdelt",
                            "crval",
                            "latpole",
                            "velUnit",
                            "worldreplace2",
                        ],
                    )

    def test_beam(self):
        """
            Verify fix to issue 45
            https://github.com/casangi/xradio/issues/45
        irint("*** r", r)
        """
        download(self._imname2), f"failed to download {self._imname2}"
        self.assertTrue(
            os.path.isdir(self._imname2), f"Could not download {self._imname2}"
        )
        shutil.copytree(self._imname2, self._outname6)
        # multibeam image
        with open_image_ro(self._imname2) as im1:
            beams1 = im1.imageinfo()["perplanebeams"]
            for do_sky, outname in zip(
                [True, False], [self._outname2, self._outname2_no_sky]
            ):
                xds = open_image(self._imname2, do_sky_coords=do_sky)
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
        xds = open_image(self._outname6)
        self.assertFalse("beam" in xds.attrs, "beam should not be in xds.attrs")
        expec = np.array(
            [4 / 180 / 3600 * np.pi, 3 / 180 / 3600 * np.pi, 5 * np.pi / 180]
        )
        for i, p in enumerate(["major", "minor", "pa"]):
            self.assertTrue(
                np.allclose(
                    xds.BEAM_FIT_PARAMS_SKY.sel(beam_params_label=p).values,
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
        self.assertTrue(
            os.path.isdir(self._imname3), f"Could not download {self._imname3}"
        )
        for do_sky, outname, out_1, out_2 in zip(
            [True, False],
            [self._outname3, self._outname3_no_sky],
            [self._outname4, self._outname4_no_sky],
            [self._outname5, self._outname5_no_sky],
        ):
            # case 1: no mask + no nans = no mask
            xds = open_image(self._imname3, do_sky_coords=do_sky)
            first_attrs = xds.attrs
            t = deepcopy(xds.attrs)
            c = deepcopy(xds.coords)
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
                attrs={image_type: "Flag"},
            )
            xds = xds.assign(FLAG_SKY=mask0)
            xds.attrs["data_groups"]["base"]["flag"] = "FLAG_SKY"
            write_image(xds, out_1, out_format="casa")
            subdirs = glob(f"{out_1}/*/")
            subdirs = [d[d.index("/") + 1 : -1] for d in subdirs]
            subdirs.sort()
            self.assertEqual(
                subdirs,
                ["MASK_0", "logtable", "mask_xds_nans", "mask_xds_nans_or_MASK_0"],
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
            subdirs = glob(f"{out_2}/*/")
            subdirs = [d[d.index("/") + 1 : -1] for d in subdirs]
            subdirs.sort()
            self.assertEqual(
                subdirs,
                ["MASK_0", "logtable"],
                f"Unexpected subdirectory list found. subdirs is {subdirs}",
            )
            with open_image_ro(out_2) as im1:
                casa_mask = im1.getmask()
            self.assertTrue(casa_mask[2, 2, 2, 2], "Wrong pixel masked")
            self.assertEqual(casa_mask.sum(), 1, "Wrong number of pixels masked")

    def test_output_uv_casa_image(self):
        image = self.uv_image()
        if not os.path.exists(image):
            download(image)
        self.assertTrue(os.path.isdir(image), f"Cound not download {image}")
        xds = open_image(image)
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
        cls._zds = open_image(cls._zarr_store)

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

    def test_got_xds(self):
        assert_xarray_datasets_equal(
            self._zds, self.true_xds()
        )

    def test_get_img_ds_block(self):
        self.compare_image_block(self._zarr_store, zarr=True)

    def test_output_uv_zarr_image(self):
        image = self.uv_image()
        download(image)
        self.assertTrue(os.path.isdir(image), f"Cound not download {image}")
        xds_true = open_image({"APERTURE": image})
        write_image(xds_true, self._zarr_uv_store, "zarr")
        xds = open_image({"APERTURE": self._zarr_uv_store})
        assert_xarray_datasets_equal(xds, xds_true)

    def test_beam(self):
        mb = np.zeros(shape=[1, 10, 4, 3], dtype=float)
        mb[:, :, :, 0] = 0.00001
        mb[:, :, :, 1] = 0.00002
        mb[:, :, :, 2] = 0.00003
        xdb = xr.DataArray(
            mb, dims=["time", "frequency", "polarization", "beam_params_label"]
        )
        xdb = xdb.rename("BEAM_FIT_PARAMS")
        # xdb = xdb.assign_coords(beam_params_label=["major", "minor", "pa"])
        xdb.attrs["units"] = "rad"
        xds = deepcopy(self.xds())
        xds["BEAM_FIT_PARAMS"] = xdb
        write_image(xds, self._zarr_beam_test, "zarr")
        xds2 = open_image(self._zarr_beam_test)
        self.assertTrue(
            np.allclose(xds2.BEAM_FIT_PARAMS.values, xds.BEAM_FIT_PARAMS.values),
            "Incorrect beam values",
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
        cls._fds = open_image(cls.infits(), {"frequency": 5}, do_sky_coords=True)
        cls._fds_no_sky = open_image(
            cls.infits(), {"frequency": 5}, do_sky_coords=False
        )

        # print("$$$$ Opened fits file", cls.infits())

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

    def test_got_xds(self):
        # casacore writes the fits image with doppler type Z even though the casacore image
        # uses doppler type RADIO. So that may be a casacore bug, so we need to conver the
        # velocities of the fits xds to RADIO
        c = 299792458  # speed of light in m/s
        radio_velocity = c * (
            1 - self._fds["frequency"].values / self._fds["frequency"].attrs["rest_frequency"]["data"]
        )
        self._fds.coords["velocity"].values = radio_velocity
        self._fds_no_sky.coords["velocity"].values = radio_velocity
        self._fds.coords["velocity"].attrs["doppler_type"] = "radio"
        self._fds_no_sky.coords["velocity"].attrs["doppler_type"] = "radio"
        # FITS SKY values are nan where the FLAG_SKY values are True, so set SKY to nan in the true xds for comparison
        true_xds = deepcopy(self.true_xds())
        true_xds_no_sky = deepcopy(self.true_no_sky_xds())
        if "FLAG_SKY" in true_xds:
            true_xds["SKY"].values = xr.where(true_xds["FLAG_SKY"].values, np.nan, true_xds["SKY"])
        if "FLAG_SKY" in true_xds_no_sky:
            true_xds_no_sky["SKY"].values = xr.where(
                true_xds_no_sky["FLAG_SKY"].values, np.nan, true_xds_no_sky["SKY"]
            )
        # FITS gets a FITS specific user attr member added that isn't in the true data set
        self._fds["SKY"].attrs["user"] = {}
        self._fds_no_sky["SKY"].attrs["user"] = {}
        assert_xarray_datasets_equal(
            self._fds, true_xds
        )
        assert_xarray_datasets_equal(
            self._fds_no_sky, true_xds_no_sky
        )

    def test_multibeam(self):
        download(self._imname1)
        self.assertTrue(
            os.path.exists(self._imname1), f"{self._imname1} not downloaded"
        )
        with open_image_ro(self._imname1) as casa_image:
            casa_image.tofits(self._outname1)
            expec = casa_image.imageinfo()["perplanebeams"]
        xds = open_image(self._outname1)
        got = xds.BEAM_FIT_PARAMS_SKY
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
                        dict(beam_params_label=b)
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
            fds = open_image(self.infits(), {"frequency": 5}, compute_mask=compute_mask)
            flag = "FLAG_SKY"
            if compute_mask:
                self.assertTrue(
                    flag in fds.data_vars,
                    f"{flag} should be in data_vars, but is not",
                )
            else:
                self.assertTrue(
                    flag not in fds.data_vars,
                    f"{flag} should not be in data_vars, but is",
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
            open_image(self._compressed_fits, {"frequency": 5})

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
            fds = open_image(self._bzero)
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
            fds = open_image(self._bscale)
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
                "note": "l is the angle measured from the reference direction to the east. "
                "So l = x*cdelt, where x is the number of pixels from the reference direction. "
                "See AIPS Memo #27, Section III.",
            },
            "m": {
                # "type": "quantity",
                # "crval": 0.0,
                # "cdelt": cdelt,
                # "units": "rad",
                "note": "m is the angle measured from the reference direction to the north. "
                "So m = y*cdelt, where y is the number of pixels from the reference direction. "
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
            "attrs": {
                "type": "sky_coord",
                "frame": "fk5",
                "equinox": "j2000.0",
                "units": "rad",
            },
            "data": [0.2, -0.5],
            "dims": "sky_dir_label",
            "coords": {
                "sky_dir_label": {"data": ["ra", "dec"], "dims": "sky_dir_label"}
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
            skel.attrs["coordinate_system_info"]["reference_direction"],
            expec2,
            "got",
            "expected",
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
        coordinate_system_info = {
            "reference_direction": {
                "attrs": {
                    "type": "sky_coord",
                    "frame": "fk5",
                    "equinox": "j2000.0",
                    "units": "rad",
                },
                "data": [0.2, -0.5],
                "dims": "sky_dir_label",
                "coords": {
                    "sky_dir_label": {"data": ["ra", "dec"], "dims": "sky_dir_label"}
                },
            },
            "native_pole_direction": {
                "attrs": {
                    "type": "location",
                    "frame": "NATIVE_PROJECTION",
                    "units": "rad",
                },
                "data": [np.pi, 0.0],
                "dims": "ellipsoid_dir_label",
                "coords": {
                    "ellipsoid_dir_label": {
                        "data": ["lon", "lat"],
                        "dims": "ellipsoid_dir_label",
                    }
                },
            },
            "pixel_coordinate_transformation_matrix": [[1.0, 0.0], [0.0, 1.0]],
            "projection": "SIN",
            "projection_parameters": [0.0, 0.0],
        }
        data_groups = {"base": {}}
        expec = {
            "data_groups": data_groups,
            "coordinate_system_info": coordinate_system_info,
            "type": "image",
        }
        self.dict_equality(skel.attrs, expec, "got", "expected")


class make_empty_sky_image_tests(make_empty_image_tests):
    """Test making skeleton image"""
    _empty_sky_image_true: str = "empty_sky_image_true.zarr"
    _empty_sky_image_no_sky_coords_true: str = "empty_sky_image_no_sky_coords_true.zarr"
    _sky_true = None
    _sky_no_coords_true = None


    @classmethod
    def setUpClass(cls):
        cls._skel_im = make_empty_image_tests.create_image(make_empty_sky_image, True)
        cls._skel_im_no_sky = make_empty_image_tests.create_image(
            make_empty_sky_image, False
        )

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()

    def empty_image_true_xds(self):
        if not self._sky_true:
            if not os.path.exists(self._empty_sky_image_true):
                download(self._empty_sky_image_true)
            self._sky_true = open_image(self._empty_sky_image_true)
        return self._sky_true

    def empty_image_no_coords_true_xds(self):
        if not self._sky_no_coords_true:
            if not os.path.exists(self._empty_sky_image_no_sky_coords_true):
                download(self._empty_sky_image_no_sky_coords_true)
            self._sky_no_coords_true = open_image(self._empty_sky_image_no_sky_coords_true)
        return self._sky_no_coords_true


    def test_empty_sky_image(self):
        assert_xarray_datasets_equal(
            self._skel_im, self.empty_image_true_xds()
        )
    def test_empty_sky_image_no_coords(self):
        assert_xarray_datasets_equal(
            self._skel_im_no_sky, self.empty_image_no_coords_true_xds()
        )

class make_empty_aperture_image_tests(make_empty_image_tests):
    """Test making skeleton image"""
    _empty_aperture_image_true: str = "empty_aperture_image_true.zarr"
    _aperture_true = None

    @classmethod
    def setUpClass(cls):
        cls._skel_im = make_empty_image_tests.create_image(
            make_empty_aperture_image, None
        )
        # write_image(cls._skel_im, cls._empty_aperture_image_true, "zarr")

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()

    def empty_image_true_xds(self):
        if not self._aperture_true:
            if not os.path.exists(self._empty_aperture_image_true):
                download(self._empty_aperture_image_true)
            self._sky_true = open_image(self._empty_aperture_image_true)
        return self._sky_true

    def skel_im(self):
        return self._skel_im

    def test_empty_sky_image(self):
        assert_xarray_datasets_equal(
            self._skel_im, self.empty_image_true_xds()
        )

class make_empty_lmuv_image_tests(make_empty_image_tests):
    """Tests making image with l, m, u, v coordinates"""

    _empty_lmuv_image_true: str = "empty_lmuv_image_true.zarr"
    _empty_lmuv_image_no_sky_coords_true: str = "empty_lmuv_image_no_sky_coords_true.zarr"
    _lmuv_true = None
    _lmuv_no_coords_true = None

    @classmethod
    def setUpClass(cls):
        cls._skel_im = make_empty_image_tests.create_image(make_empty_lmuv_image, True)
        # write_image(cls._skel_im, cls._empty_lmuv_image_true, "zarr")
        cls._skel_im_no_sky = make_empty_image_tests.create_image(
            make_empty_lmuv_image, False
        )
        # write_image(cls._skel_im_no_sky, cls._empty_lmuv_image_no_sky_coords_true, "zarr")

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()

    def empty_image_true_xds(self):
        if not self._lmuv_true:
            if not os.path.exists(self._empty_lmuv_image_true):
                download(self._empty_lmuv_image_true)
            self._lmuv_true = open_image(self._empty_lmuv_image_true)
        return self._lmuv_true

    def empty_image_no_coords_true_xds(self):
        if not self._lmuv_no_coords_true:
            if not os.path.exists(self._empty_lmuv_image_no_sky_coords_true):
                download(self._empty_lmuv_image_no_sky_coords_true)
            self._lmuv_no_coords_true = open_image(self._empty_lmuv_image_no_sky_coords_true)
        return self._lmuv_no_coords_true


    def test_empty_sky_image(self):
        assert_xarray_datasets_equal(
            self._skel_im, self.empty_image_true_xds()
        )
    def test_empty_sky_image_no_coords(self):
        assert_xarray_datasets_equal(
            self._skel_im_no_sky, self.empty_image_no_coords_true_xds()
        )
