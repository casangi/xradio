"""Unit tests for xradio.image public API.

Each test class maps to one or two public functions defined in
``src/xradio/image/image.py``:

* ``TestLoadImage``        → ``load_image``
* ``TestOpenImageCasa``    → ``open_image`` (CASA format)
* ``TestWriteImageCasa``   → ``write_image`` (CASA format)
* ``TestCasaRoundtrip``    → ``open_image`` + ``write_image`` (CASA round-trip)
* ``TestZarrRoundtrip``    → ``open_image`` + ``write_image`` (zarr round-trip)
* ``TestWriteImageZarr``   → ``write_image`` (zarr, UV/aperture image)
* ``TestOpenImageFits``    → ``open_image`` (FITS format)
* ``TestMakeEmptyImages``  → ``make_empty_sky_image``,
                             ``make_empty_aperture_image``,
                             ``make_empty_lmuv_image``

The dask cluster fixture is provided by ``conftest.py`` in this directory.
"""

try:
    from casacore import tables
except ImportError:
    import xradio._utils._casacore.casacore_from_casatools as tables

from astropy.io import fits
import astropy.units as u
from copy import deepcopy
import dask.array as da
from glob import glob
import numpy as np
import numpy.ma as ma
import os
import pytest
import re
import shutil
import xarray as xr

from xradio.image import (
    load_image,
    make_empty_aperture_image,
    make_empty_lmuv_image,
    make_empty_sky_image,
    open_image,
    write_image,
)
from xradio.image._util.casacore import _squeeze_if_needed
from xradio.image._util._casacore.common import _create_new_image as create_new_image
from xradio.image._util._casacore.common import _open_image_ro as open_image_ro
from xradio.image._util._casacore.common import _object_name
from xradio.image._util.common import _image_type as image_type
from xradio._utils._casacore.tables import open_table_ro

from xradio.testing import assert_attrs_dicts_equal, assert_xarray_datasets_equal
from xradio.testing.image import (
    assert_image_block_equal,
    create_bzero_bscale_fits,
    create_empty_test_image,
    download_and_open_image,
    download_image,
    normalize_image_coords_for_compare,
    remove_path,
)

sky = "SKY"

pytestmark = pytest.mark.usefixtures("dask_client_module")


# --------------------------------------------------------------------------- #
# TestLoadImage  –  load_image                                                 #
# --------------------------------------------------------------------------- #


class TestLoadImage:
    """Tests for ``load_image``."""

    def test_block_squeezes_spatial_axes(self, tmp_path):
        """Visibility normalisation block drops singleton l/m dimensions."""
        imagename = tmp_path / "synthetic.sumwt"
        data = np.arange(8, dtype=np.float32).reshape(4, 2, 1, 1)
        masked_data = ma.masked_array(data, np.zeros_like(data, dtype=bool))

        with create_new_image(str(imagename), shape=list(data.shape)) as im:
            im.put(masked_data)

        xds = load_image({"visibility_normalization": str(imagename)})

        assert xds.VISIBILITY_NORMALIZATION.dims == (
            "time",
            "frequency",
            "polarization",
        )
        assert xds.VISIBILITY_NORMALIZATION.shape == (1, 4, 2)
        assert "l" not in xds.dims
        assert "m" not in xds.dims
        np.testing.assert_array_equal(
            xds.VISIBILITY_NORMALIZATION.values,
            data[np.newaxis, :, :, 0, 0],
        )

    def test_rejects_non_singleton_spatial_axes(self):
        """_squeeze_if_needed raises ValueError for non-singleton l/m."""
        data = da.from_array(np.zeros((1, 4, 2, 2, 3), dtype=np.float32))

        with pytest.raises(
            ValueError,
            match=r"VISIBILITY_NORMALIZATION casa image must have l and m of length 1\. Found \(2, 3\)",
        ):
            _squeeze_if_needed(data, "VISIBILITY_NORMALIZATION")

    def test_mask_squeezes_spatial_axes(self, tmp_path):
        """Visibility normalisation mask block drops singleton l/m dimensions."""
        imagename = tmp_path / "masked.sumwt"
        data = np.arange(8, dtype=np.float32).reshape(4, 2, 1, 1)
        mask = np.zeros_like(data, dtype=bool)
        mask[1, 0, 0, 0] = True
        masked_data = ma.masked_array(data, mask)

        with create_new_image(
            str(imagename), shape=list(data.shape), mask="MASK_0"
        ) as im:
            im.put(masked_data)

        xds = load_image({"visibility_normalization": str(imagename)})

        assert xds.FLAG_VISIBILITY_NORMALIZATION.dims == (
            "time",
            "frequency",
            "polarization",
        )
        assert xds.FLAG_VISIBILITY_NORMALIZATION.shape == (1, 4, 2)
        assert "l" not in xds.FLAG_VISIBILITY_NORMALIZATION.dims
        assert "m" not in xds.FLAG_VISIBILITY_NORMALIZATION.dims
        np.testing.assert_array_equal(
            xds.FLAG_VISIBILITY_NORMALIZATION.values,
            mask[np.newaxis, :, :, 0, 0],
        )


# --------------------------------------------------------------------------- #
# TestOpenImageCasa  –  open_image (CASA format)                               #
# --------------------------------------------------------------------------- #


class TestOpenImageCasa:
    """Tests for ``open_image`` with CASA images."""

    _imname: str = "casa_test_image.im"
    _uv_image: str = "complex_valued_uv.im"
    _xds_from_casa_true: str = "casa_to_xds_true.zarr"
    _xds_from_no_sky_casa_true: str = "casa_no_sky_to_xds_true.zarr"
    _xds_from_casa_uv_true: str = "casa_uv_true.zarr"

    @classmethod
    def setup_class(cls):
        download_image(cls._imname)
        cls._xds = open_image(cls._imname, {"frequency": 5})
        cls._xds_no_sky = open_image(cls._imname, {"frequency": 5}, False, False)
        cls._xds_uv = download_and_open_image(cls._uv_image)
        cls._xds_true = download_and_open_image(cls._xds_from_casa_true)
        cls._xds_no_sky_true = download_and_open_image(cls._xds_from_no_sky_casa_true)
        cls._xds_uv_true = download_and_open_image(cls._xds_from_casa_uv_true)

    @classmethod
    def teardown_class(cls):
        for f in [
            cls._imname,
            cls._uv_image,
            cls._xds_from_casa_true,
            cls._xds_from_no_sky_casa_true,
            cls._xds_from_casa_uv_true,
        ]:
            remove_path(f)

    def test_returns_xds(self):
        assert_xarray_datasets_equal(self._xds, self._xds_true)

    def test_returns_xds_no_sky(self):
        assert_xarray_datasets_equal(self._xds_no_sky, self._xds_no_sky_true)

    def test_uv_image(self):
        print("xds_uv", self._xds_uv)
        print("xds_uv_true", self._xds_uv_true)
        assert_xarray_datasets_equal(self._xds_uv, self._xds_uv_true)


# --------------------------------------------------------------------------- #
# TestWriteImageCasa  –  write_image (CASA format)                             #
# --------------------------------------------------------------------------- #


class TestWriteImageCasa:
    """Tests for ``write_image`` with CASA output format."""

    _imname: str = "casa_test_image.im"
    _outname: str = "rabbit.im"
    _outname2: str = "rabbit2.im"
    _myout: str = "zk.im"
    _output_uv: str = "output_uv.im"
    _uv_image: str = "complex_valued_uv.im"

    @classmethod
    def setup_class(cls):
        download_image(cls._imname)
        cls._xds = open_image(cls._imname, {"frequency": 5})

    @classmethod
    def teardown_class(cls):
        for f in [
            cls._imname,
            cls._outname,
            cls._outname2,
            cls._myout,
            cls._output_uv,
            cls._uv_image,
        ]:
            remove_path(f)

    def test_numpy_data_var(self):
        """Writing an xds whose sky data var is a numpy array round-trips pixels."""
        write_image(self._xds, self._outname, "casa", overwrite=True)
        with open_image_ro(self._outname) as im:
            p = im.getdata()
        sky_name = (
            self._xds.attrs.get("data_groups", {}).get("base", {}).get("sky", "SKY")
        )
        exp_data = np.squeeze(np.transpose(self._xds[sky_name], [1, 2, 4, 3, 0]), 4)
        assert (p == exp_data).all(), "Incorrect pixel values"

    def test_object_name_not_present(self):
        """Writing an xds without an object name does not raise."""
        xds = deepcopy(self._xds)
        del xds["SKY"].attrs[_object_name]
        write_image(xds, self._outname2, "casa", overwrite=True)
        with open_image_ro(self._outname2) as im:
            ii = im.imageinfo()
            assert ii["objectname"] == "", "Incorrect object name"

    def test_uv_image(self):
        """UV (aperture) image written to CASA preserves linear coordinates and pixels."""
        xds = download_and_open_image(self._uv_image)
        write_image(xds, self._output_uv, "casa")
        with open_image_ro(self._output_uv) as test_im:
            with open_image_ro(self._uv_image) as expec_im:
                got = test_im.coordinates().dict()["linear0"]
                expec = expec_im.coordinates().dict()["linear0"]
                assert_attrs_dicts_equal(
                    got,
                    expec,
                    context="written casa uv image coordinate dict",
                    rtol=1e-7,
                    atol=1e-7,
                )
                assert np.isclose(
                    test_im.getdata(), expec_im.getdata()
                ).all(), "Incorrect pixel data"

    def test_overwrite(self):
        """write_image respects the overwrite flag."""
        write_image(self._xds, self._myout, out_format="casa", overwrite=True)
        assert os.path.exists(self._myout), f"{self._myout} was not written"

        with pytest.raises(FileExistsError):
            write_image(self._xds, self._myout, out_format="casa", overwrite=False)
        with pytest.raises(FileExistsError):
            write_image(self._xds, self._myout, out_format="casa")


# --------------------------------------------------------------------------- #
# TestCasaRoundtrip  –  open_image + write_image (CASA round-trip)             #
# --------------------------------------------------------------------------- #


class TestCasaRoundtrip:
    """Round-trip tests: open CASA image → xarray → write CASA image → verify."""

    _imname: str = "casa_test_image.im"
    _outname: str = "out.im"
    _outname_no_sky: str = "from_xds_no_sky.im"
    _imname2: str = "demo_simulated.im"
    _imname3: str = "no_mask.im"
    _outname2: str = "check_beam.im"
    _outname2_no_sky: str = "check_beam.im_no_sky"
    _outname3: str = "xds_2_casa_no_mask.im"
    _outname3_no_sky: str = "xds_2_casa_no_mask.im_no_sky"
    _outname4: str = "xds_2_casa_nans_and_mask.im"
    _outname4_no_sky: str = "xds_2_casa_nans_and_mask.im_no_sky"
    _outname5: str = "xds_2_casa_nans_already_masked.im"
    _outname5_no_sky: str = "xds_2_casa_nans_already_masked.im_no_sky"
    _outname6: str = "single_beam.im"

    @classmethod
    def setup_class(cls):
        download_image(cls._imname)
        cls._xds = open_image(cls._imname, {"frequency": 5})
        cls._xds_no_sky = open_image(cls._imname, {"frequency": 5}, False, False)
        write_image(cls._xds, cls._outname, out_format="casa", overwrite=True)
        write_image(cls._xds_no_sky, cls._outname_no_sky, out_format="casa")

    @classmethod
    def teardown_class(cls):
        for f in [
            cls._imname,
            cls._imname + "_2",
            cls._outname,
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
        ]:
            remove_path(f)

    def test_pixels_and_mask(self):
        """Pixel values and mask are preserved in a CASA round-trip."""
        with open_image_ro(self._imname) as im1:
            for imname in [self._outname, self._outname_no_sky]:
                with open_image_ro(imname) as im2:
                    assert (
                        im1.getdata() == im2.getdata()
                    ).all(), "Incorrect pixel values"
                    assert (
                        im1.getmask() == im2.getmask()
                    ).all(), "Incorrect mask values"
                    assert im1.datatype() == im2.datatype(), (
                        f"Incorrect round-trip pixel type: "
                        f"input {im1.name()} {im1.datatype()}, "
                        f"output {im2.name()} {im2.datatype()}"
                    )

    def test_metadata(self):
        """Metadata (coordinates, table keywords) is preserved in a CASA round-trip."""
        f = 180 * 60 / np.pi
        with open_image_ro(self._imname) as im1:
            c1 = im1.info()
            for imname in [self._outname, self._outname_no_sky]:
                assert os.path.exists(imname), f"Output image {imname} does not exist"
                with open_image_ro(imname) as im2:
                    c2 = im2.info()
                    normalize_image_coords_for_compare(c2["coordinates"], f)
                    assert_attrs_dicts_equal(
                        c2,
                        c1,
                        context=f"casa image metadata test, imname={imname}",
                        rtol=1e-7,
                        atol=1e-7,
                    )

        with open_table_ro(self._imname) as tb1:
            for imname in [self._outname, self._outname_no_sky]:
                with open_table_ro(imname) as tb2:
                    kw1 = tb1.getkeywords()
                    kw2 = tb2.getkeywords()
                    exclude_keys = ["worldreplace2"]
                    union_keys = set(kw1.keys()) | set(kw2.keys())
                    for k in union_keys:
                        if k in exclude_keys or k not in kw1 or k not in kw2:
                            for kw in [kw1, kw2]:
                                if k in kw:
                                    del kw[k]
                    d2 = kw2["coords"]["direction0"]
                    s2 = kw2["coords"]["spectral2"]
                    t2 = kw2["coords"]["stokes1"]
                    for k in ["_image_axes", "_axes_sizes"]:
                        for d in [d2, s2, t2]:
                            del d[k]
                    normalize_image_coords_for_compare(kw2["coords"], f)
                    d2["latpole"] = 0
                    s1 = kw1["coords"]["spectral2"]
                    del s1["conversion"]
                    del kw1["logtable"]
                    del kw2["logtable"]
                    del kw1["masks"]["MASK_0"]["mask"]
                    del kw2["masks"]["MASK_0"]["mask"]
                    assert_attrs_dicts_equal(
                        kw2,
                        kw1,
                        context=f"casa image table keyword test, imname={imname}",
                        rtol=1e-7,
                        atol=1e-7,
                    )

    def test_beam(self):
        """Beam parameters are preserved in a CASA round-trip (issue #45)."""
        download_image(self._imname2)
        shutil.copytree(self._imname2, self._outname6)

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
                    assert_attrs_dicts_equal(
                        beams2,
                        beams1,
                        context=f"casa image beam test, do_sky={do_sky}",
                        rtol=1e-7,
                        atol=1e-7,
                    )

        beam3 = {
            "major": {"unit": "arcsec", "value": 4.0},
            "minor": {"unit": "arcsec", "value": 3.0},
            "positionangle": {"unit": "deg", "value": 5.0},
        }
        with tables.table(self._outname6, readonly=False) as tb:
            tb.putkeyword(
                "imageinfo",
                {
                    "imagetype": "Intensity",
                    "objectname": "",
                    "restoringbeam": beam3,
                },
            )
        xds = open_image(self._outname6)
        assert "beam" not in xds.attrs, "beam should not be in xds.attrs"
        expec = np.array(
            [4 / 180 / 3600 * np.pi, 3 / 180 / 3600 * np.pi, 5 * np.pi / 180]
        )
        for i, p in enumerate(["major", "minor", "pa"]):
            assert np.allclose(
                xds.BEAM_FIT_PARAMS_SKY.sel(beam_params_label=p).values,
                expec[i],
            ), f"Incorrect {p} axis"

    def test_masking(self):
        """NaN masking is correct when writing CASA images (issue #48)."""
        download_image(self._imname3)
        for do_sky, outname, out_1, out_2 in zip(
            [True, False],
            [self._outname3, self._outname3_no_sky],
            [self._outname4, self._outname4_no_sky],
            [self._outname5, self._outname5_no_sky],
        ):
            # case 1: no mask + no nans = no mask
            xds = open_image(self._imname3, do_sky_coords=do_sky)
            t = deepcopy(xds.attrs)
            write_image(xds, outname, out_format="casa")
            subdirs = glob(f"{outname}/*/")
            subdirs = [d[d.index("/") + 1 : -1] for d in subdirs]
            assert subdirs == [
                "logtable"
            ], f"Unexpected directory (mask?) found. subdirs is {subdirs}"
            # case 2a: no mask + nans = nan_mask
            xds[sky][0, 1, 1, 1, 1] = float("NaN")
            shutil.rmtree(outname)
            assert_attrs_dicts_equal(
                t,
                xds.attrs,
                context="masking test, case 2a",
                rtol=1e-7,
                atol=1e-7,
            )
            write_image(xds, outname, out_format="casa")
            subdirs = glob(f"{outname}/*/")
            subdirs = [d[d.index("/") + 1 : -1] for d in subdirs]
            subdirs.sort()
            assert subdirs == [
                "logtable",
                "mask_xds_nans",
            ], f"Unexpected subdirectory list found. subdirs is {subdirs}"
            with open_image_ro(outname) as im1:
                casa_mask = im1.getmask()
            assert casa_mask[1, 1, 1, 1], "Wrong pixels are masked"
            assert casa_mask.sum() == 1, "More pixels masked than expected"
            casa_mask[1, 1, 1, 1] = False
            # case 2b: mask + nans = nan_mask and (nan_mask or mask)
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
            assert subdirs == [
                "MASK_0",
                "logtable",
                "mask_xds_nans",
                "mask_xds_nans_or_MASK_0",
            ], f"Unexpected subdirectory list found. subdirs is {subdirs}"
            with open_image_ro(out_1) as im1:
                casa_mask = im1.getmask()
                assert casa_mask[1, 1, 1, 1], "Wrong pixels are masked"
                assert casa_mask[2, 2, 2, 2], "Wrong pixels are masked"
                assert casa_mask.sum() == 2, "Wrong pixels are masked"
            # case 2c: all nans already masked = no new masks
            xds[sky][0, 2, 2, 2, 2] = float("NaN")
            xds[sky][0, 1, 1, 1, 1] = 0
            write_image(xds, out_2, out_format="casa")
            subdirs = glob(f"{out_2}/*/")
            subdirs = [d[d.index("/") + 1 : -1] for d in subdirs]
            subdirs.sort()
            assert subdirs == [
                "MASK_0",
                "logtable",
            ], f"Unexpected subdirectory list found. subdirs is {subdirs}"
            with open_image_ro(out_2) as im1:
                casa_mask = im1.getmask()
            assert casa_mask[2, 2, 2, 2], "Wrong pixel masked"
            assert casa_mask.sum() == 1, "Wrong number of pixels masked"


# --------------------------------------------------------------------------- #
# TestZarrRoundtrip  –  open_image + write_image (zarr round-trip)             #
# --------------------------------------------------------------------------- #


class TestZarrRoundtrip:
    """Round-trip tests: CASA image → xarray → zarr → xarray → verify."""

    _imname: str = "casa_test_image.im"
    _zarr_store: str = "out.zarr"
    _zarr_beam_test: str = "beam_test.zarr"
    _xds_from_casa_true: str = "casa_to_xds_true.zarr"

    @classmethod
    def setup_class(cls):
        download_image(cls._imname)
        cls._xds = open_image(cls._imname, {"frequency": 5})
        cls._xds_true = download_and_open_image(cls._xds_from_casa_true)
        write_image(cls._xds, cls._zarr_store, out_format="zarr", overwrite=True)
        cls._zds = open_image(cls._zarr_store)

    @classmethod
    def teardown_class(cls):
        for f in [
            cls._imname,
            cls._zarr_store,
            cls._zarr_store + "_2",
            cls._zarr_beam_test,
            cls._xds_from_casa_true,
        ]:
            remove_path(f)

    def test_returns_xds(self):
        assert_xarray_datasets_equal(self._zds, self._xds_true)

    def test_image_block(self):
        """Spatial block loaded from a zarr image matches the full-image slice."""
        from xradio.testing.image.generators import make_beam_fit_params

        xds_with_beam = self._zds.assign(
            BEAM_FIT_PARAMS=make_beam_fit_params(self._zds)
        )
        xds_with_beam["BEAM_FIT_PARAMS"].attrs["units"] = "rad"
        assert_image_block_equal(
            xds_with_beam,
            self._zarr_store + "_2",
            selection={
                "l": slice(2, 10),
                "m": slice(3, 15),
                "polarization": slice(0, 1),
                "frequency": slice(0, 4),
            },
            zarr=True,
        )

    def test_beam(self):
        """Beam parameters survive a zarr round-trip."""
        mb = np.zeros(shape=[1, 10, 4, 3], dtype=float)
        mb[:, :, :, 0] = 0.00001
        mb[:, :, :, 1] = 0.00002
        mb[:, :, :, 2] = 0.00003
        xdb = xr.DataArray(
            mb, dims=["time", "frequency", "polarization", "beam_params_label"]
        )
        xdb = xdb.rename("BEAM_FIT_PARAMS")
        xdb.attrs["units"] = "rad"
        xds = deepcopy(self._xds)
        xds["BEAM_FIT_PARAMS"] = xdb
        write_image(xds, self._zarr_beam_test, "zarr")
        xds2 = open_image(self._zarr_beam_test)
        assert np.allclose(
            xds2.BEAM_FIT_PARAMS.values, xds.BEAM_FIT_PARAMS.values
        ), "Incorrect beam values"


# --------------------------------------------------------------------------- #
# TestWriteImageZarr  –  write_image (zarr, UV/aperture image)                 #
# --------------------------------------------------------------------------- #


class TestWriteImageZarr:
    """Tests for ``write_image`` to zarr format with UV (aperture) images."""

    _uv_image: str = "complex_valued_uv.im"
    _zarr_uv_store: str = "out_uv.zarr"

    @classmethod
    def setup_class(cls):
        download_image(cls._uv_image)

    @classmethod
    def teardown_class(cls):
        for f in [cls._uv_image, cls._zarr_uv_store]:
            remove_path(f)

    def test_uv_image(self):
        """UV aperture image survives a zarr write–open round-trip."""
        xds_true = open_image({"APERTURE": self._uv_image})
        write_image(xds_true, self._zarr_uv_store, "zarr")
        xds = open_image({"APERTURE": self._zarr_uv_store})
        assert_xarray_datasets_equal(xds, xds_true)


# --------------------------------------------------------------------------- #
# TestOpenImageFits  –  open_image (FITS format)                               #
# --------------------------------------------------------------------------- #


class TestOpenImageFits:
    """Tests for ``open_image`` with FITS images."""

    _infits: str = "test_image.fits"
    _imname1: str = "demo_simulated.im"
    _outname1: str = "demo_simulated.fits"
    _compressed_fits: str = "compressed.fits"
    _bzero: str = "bzero.fits"
    _bscale: str = "bscale.fits"
    _xds_from_casa_true: str = "casa_to_xds_true.zarr"
    _xds_from_no_sky_casa_true: str = "casa_no_sky_to_xds_true.zarr"

    @classmethod
    def setup_class(cls):
        download_image(cls._infits)
        cls._fds = open_image(cls._infits, {"frequency": 5}, do_sky_coords=True)
        cls._fds_no_sky = open_image(cls._infits, {"frequency": 5}, do_sky_coords=False)
        cls._xds_true = download_and_open_image(cls._xds_from_casa_true)
        cls._xds_no_sky_true = download_and_open_image(cls._xds_from_no_sky_casa_true)

    @classmethod
    def teardown_class(cls):
        for f in [
            cls._infits,
            cls._imname1,
            cls._outname1,
            cls._compressed_fits,
            cls._bzero,
            cls._bscale,
            cls._xds_from_casa_true,
            cls._xds_from_no_sky_casa_true,
        ]:
            remove_path(f)

    def test_returns_xds(self):
        """FITS image opened as xds matches the reference CASA-derived dataset."""
        fds = deepcopy(self._fds)
        fds_no_sky = deepcopy(self._fds_no_sky)

        c = 299792458  # speed of light in m/s
        radio_velocity = c * (
            1
            - fds["frequency"].values / fds["frequency"].attrs["rest_frequency"]["data"]
        )
        fds.coords["velocity"].values = radio_velocity
        fds_no_sky.coords["velocity"].values = radio_velocity
        fds.coords["velocity"].attrs["doppler_type"] = "radio"
        fds_no_sky.coords["velocity"].attrs["doppler_type"] = "radio"

        true_xds = deepcopy(self._xds_true)
        true_xds_no_sky = deepcopy(self._xds_no_sky_true)
        if "FLAG_SKY" in true_xds:
            true_xds["SKY"].values = xr.where(
                true_xds["FLAG_SKY"].values, np.nan, true_xds["SKY"]
            )
        if "FLAG_SKY" in true_xds_no_sky:
            true_xds_no_sky["SKY"].values = xr.where(
                true_xds_no_sky["FLAG_SKY"].values, np.nan, true_xds_no_sky["SKY"]
            )
        fds["SKY"].attrs["user"] = {}
        fds_no_sky["SKY"].attrs["user"] = {}
        assert_xarray_datasets_equal(fds, true_xds)
        assert_xarray_datasets_equal(fds_no_sky, true_xds_no_sky)

    def test_multibeam(self):
        """Multibeam FITS image has correct per-plane beam parameters."""
        download_image(self._imname1)
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
                assert np.isclose(got_comp, expec_comp), (
                    f"Incorrect {b} value for polarization {p}, channel {c}. "
                    f"{got_comp.item()} rad vs {expec_comp} rad."
                )

    @pytest.mark.parametrize(
        "compute_mask,expected_present",
        [
            pytest.param(True, True, id="compute_mask_true"),
            pytest.param(False, False, id="compute_mask_false"),
        ],
    )
    def test_compute_mask(self, compute_mask, expected_present):
        """compute_mask parameter controls presence of FLAG_SKY in the dataset."""
        fds = open_image(self._infits, {"frequency": 5}, compute_mask=compute_mask)
        flag = "FLAG_SKY"
        if expected_present:
            assert flag in fds.data_vars, f"{flag} should be in data_vars, but is not"
        else:
            assert (
                flag not in fds.data_vars
            ), f"{flag} should not be in data_vars, but is"

    def test_compressed_guard(self):
        """Reading a compressed FITS file raises RuntimeError."""
        with fits.open(self._infits) as hdulist:
            data = hdulist[0].data
            header = hdulist[0].header

        primary_hdu = fits.PrimaryHDU()
        comp_hdu = fits.CompImageHDU(data=data, header=header)
        fits.HDUList([primary_hdu, comp_hdu]).writeto(
            self._compressed_fits, overwrite=True
        )
        with fits.open(self._compressed_fits, do_not_scale_image_data=True) as hdulist:
            assert isinstance(
                hdulist[1], fits.CompImageHDU
            ), "Expected CompImageHDU in HDU[1]"

        with pytest.raises(RuntimeError) as exc_info:
            open_image(self._compressed_fits, {"frequency": 5})

        assert re.search(r"name=COMPRESSED_IMAGE", str(exc_info.value)), (
            f"Expected error about COMPRESSED_IMAGE HDU, "
            f"but got {str(exc_info.value)}"
        )

    def test_bzero_guard(self):
        """Reading a FITS file with non-zero BZERO raises RuntimeError."""
        create_bzero_bscale_fits(self._bzero, self._infits, bzero=5.0, bscale=1.0)
        assert os.path.exists(self._bzero), f"{self._bzero} was not written"
        with pytest.raises(RuntimeError) as exc_info:
            open_image(self._bzero)
        assert re.search(
            r"BSCALE/BZERO set", str(exc_info.value)
        ), f"Expected error about BSCALE/BZERO, but got {str(exc_info.value)}"

    def test_bscale_guard(self):
        """Reading a FITS file with BSCALE != 1 raises RuntimeError."""
        create_bzero_bscale_fits(self._bscale, self._infits, bzero=0.0, bscale=2.0)
        assert os.path.exists(self._bscale), f"{self._bscale} was not written"
        with pytest.raises(RuntimeError) as exc_info:
            open_image(self._bscale)
        assert re.search(
            r"BSCALE/BZERO set", str(exc_info.value)
        ), f"Expected error about BSCALE/BZERO, but got {str(exc_info.value)}"


# --------------------------------------------------------------------------- #
# TestMakeEmptyImages  –  make_empty_sky/aperture/lmuv_image                   #
# --------------------------------------------------------------------------- #

MAKE_EMPTY_CASES = [
    {
        "name": "sky",
        "factory": make_empty_sky_image,
        "do_sky_coords": True,
        "truth_xds": "empty_sky_image_true.zarr",
    },
    {
        "name": "sky_no_coords",
        "factory": make_empty_sky_image,
        "do_sky_coords": False,
        "truth_xds": "empty_sky_image_no_sky_coords_true.zarr",
    },
    {
        "name": "aperture",
        "factory": make_empty_aperture_image,
        "do_sky_coords": None,
        "truth_xds": "empty_aperture_image_true.zarr",
    },
    {
        "name": "lmuv",
        "factory": make_empty_lmuv_image,
        "do_sky_coords": True,
        "truth_xds": "empty_lmuv_image_true.zarr",
    },
    {
        "name": "lmuv_no_coords",
        "factory": make_empty_lmuv_image,
        "do_sky_coords": False,
        "truth_xds": "empty_lmuv_image_no_sky_coords_true.zarr",
    },
]


class TestMakeEmptyImages:
    """Tests for ``make_empty_sky_image``, ``make_empty_aperture_image``,
    and ``make_empty_lmuv_image``."""

    @classmethod
    def setup_class(cls):
        cls._generated_xds = {
            case["name"]: create_empty_test_image(
                case["factory"], case["do_sky_coords"]
            )
            for case in MAKE_EMPTY_CASES
        }

    def teardown_method(self):
        for case in MAKE_EMPTY_CASES:
            remove_path(case["truth_xds"])

    @pytest.mark.parametrize("case", MAKE_EMPTY_CASES, ids=lambda c: c["name"])
    def test_make_empty_image(self, case):
        truth_xds = download_and_open_image(case["truth_xds"])
        assert_xarray_datasets_equal(
            self._generated_xds[case["name"]],
            truth_xds,
        )


# --------------------------------------------------------------------------- #
# Entry point                                                                   #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    pytest.main(["-v", "-s", __file__])
