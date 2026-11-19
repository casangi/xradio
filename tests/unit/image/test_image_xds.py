import copy
import warnings

import dask.array as da
import numpy as np
import pytest
import xarray as xr
from astropy.io import fits
from astropy.time import Time
from astropy.utils.exceptions import AstropyWarning
from astropy.wcs import WCS, FITSFixedWarning

from xradio.image import make_empty_sky_image, open_image, write_image
from xradio.image._util._write_plan import plan_image_outputs
from xradio.image._util.conventions import FITS_STOKES_LABELS
from xradio.image.image_xds import InvalidAccessorLocation
from xradio.testing.image import create_empty_test_image, download_image


def _make_valid_image_dataset():
    """Create a minimal Dataset that is valid for the xr_img accessor.

    Built with :func:`~xradio.testing.image.create_empty_test_image`
    (``make_empty_sky_image`` factory, sky coordinates enabled) and promoted
    to an ``image_dataset`` node with a minimal ``data_groups`` mapping so
    that ImageXdsAccessor methods accept it as an image node.
    """

    xds = create_empty_test_image(make_empty_sky_image, do_sky_coords=True)

    # Promote the dataset to an image_dataset node that xr_img accepts.
    xds.attrs["type"] = "image_dataset"
    # Minimal data_groups: base group with required keys from the image schema.
    xds.attrs["data_groups"] = {
        "base": {
            "sky": "SKY",
            "flag": "FLAG_SKY",
            "description": "base image data group",
            "date": "2000-01-01T00:00:00.000",
        }
    }

    return xds


@pytest.fixture
def image_xds_valid():
    """Fixture providing a fresh xr_img-compatible Dataset per test."""
    return _make_valid_image_dataset()


class TestImageXdsValid:
    """Test suite for ImageXdsAccessor (xr_img) with valid image datasets."""

    def test_xr_img_accessor_registration(self, image_xds_valid):
        """xr_img accessor should be registered and keep a reference to the dataset."""

        xds = image_xds_valid

        assert hasattr(xds, "xr_img")
        assert xds.xr_img._xds is xds
        assert xds.xr_img.meta == {"summary": {}}

    def test_test_func_returns_hallo(self, image_xds_valid):
        """test_func should return the expected marker string on valid image datasets."""

        xds = image_xds_valid

        result = xds.xr_img.test_func()
        assert result == "Hallo"

    def test_get_lm_cell_size_returns_radians(self, image_xds_valid):
        """get_lm_cell_size should return the lm cell size in radians."""

        xds = image_xds_valid

        lm_cell_size = xds.xr_img.get_lm_cell_size()

        # With the default arguments the sky cell size is pi/180/60 radians.
        cdelt = np.pi / 180 / 60
        # l decreases and m increases with pixel index.
        expected = np.array([-cdelt, cdelt])

        assert lm_cell_size.shape == (2,)
        assert np.allclose(lm_cell_size, expected)

    def test_add_uv_coordinates_to_image_dataset(self, image_xds_valid):
        """add_uv_coordinates should attach u and v coords consistent with l and m."""

        xds = image_xds_valid

        # Preserve the original l and m coordinates for computing the expectation.
        l_vals = xds.coords["l"].values.copy()
        m_vals = xds.coords["m"].values.copy()
        size_l = xds.sizes["l"]
        size_m = xds.sizes["m"]

        lm_cell_size = xds.xr_img.get_lm_cell_size()

        u_expected = l_vals / ((lm_cell_size[0] ** 2) * size_l)
        v_expected = m_vals / ((lm_cell_size[1] ** 2) * size_m)

        xds_with_uv = xds.xr_img.add_uv_coordinates()

        assert "u" in xds_with_uv.coords
        assert "v" in xds_with_uv.coords
        assert np.allclose(xds_with_uv.coords["u"].values, u_expected)
        assert np.allclose(xds_with_uv.coords["v"].values, v_expected)

    def test_add_uv_coordinates_are_in_wavelengths(self, image_xds_valid):
        """add_uv_coordinates labels u and v in wavelengths (the schema convention)."""

        xds = image_xds_valid.xr_img.add_uv_coordinates()

        assert xds.coords["u"].attrs["units"] == "lambda"
        assert xds.coords["v"].attrs["units"] == "lambda"

    def test_get_uv_in_lambda_keeps_wavelengths(self, image_xds_valid):
        """u and v already in wavelengths do not depend on frequency."""

        xds = image_xds_valid.xr_img.add_uv_coordinates()

        for frequency in (1.412e9, 2.399476e11):
            u_in_lambda, v_in_lambda = xds.xr_img.get_uv_in_lambda(frequency)
            np.testing.assert_array_equal(u_in_lambda.values, xds.coords["u"].values)
            np.testing.assert_array_equal(v_in_lambda.values, xds.coords["v"].values)

    @pytest.mark.parametrize(
        "attrs",
        [
            {"units": "wavelengths"},
            {"units": ["lambda"]},
            # make_empty_* factories store the u/v attrs as a quantity dict
            {"data": 0.0, "dims": [], "attrs": {"type": "quantity", "units": "lambda"}},
        ],
        ids=["wavelengths", "list", "quantity_dict"],
    )
    def test_get_uv_in_lambda_wavelength_unit_spellings(self, image_xds_valid, attrs):
        xds = image_xds_valid.xr_img.add_uv_coordinates()
        xds.coords["u"].attrs = attrs
        xds.coords["v"].attrs = attrs

        u_in_lambda, _ = xds.xr_img.get_uv_in_lambda(1.412e9)

        np.testing.assert_array_equal(u_in_lambda.values, xds.coords["u"].values)

    @pytest.mark.parametrize("units, to_meters", [("m", 1.0), ("km", 1e3)])
    def test_get_uv_in_lambda_converts_lengths(self, image_xds_valid, units, to_meters):
        """u and v in a length unit are divided by the wavelength c / f."""

        xds = image_xds_valid.xr_img.add_uv_coordinates()
        xds.coords["u"].attrs = {"units": units}
        xds.coords["v"].attrs = {"units": units}

        frequency = 1.412e9
        u_in_lambda, v_in_lambda = xds.xr_img.get_uv_in_lambda(frequency)

        wavelength = 299792458.0 / frequency
        np.testing.assert_allclose(
            u_in_lambda.values, xds.coords["u"].values * to_meters / wavelength
        )
        np.testing.assert_allclose(
            v_in_lambda.values, xds.coords["v"].values * to_meters / wavelength
        )
        assert u_in_lambda.attrs["units"] == "lambda"

    @pytest.mark.parametrize("attrs", [{}, {"units": "Jy"}, {"units": "not a unit"}])
    def test_get_uv_in_lambda_rejects_unknown_units(self, image_xds_valid, attrs):
        xds = image_xds_valid.xr_img.add_uv_coordinates()
        xds.coords["u"].attrs = attrs

        with pytest.raises(ValueError, match="u coordinate"):
            xds.xr_img.get_uv_in_lambda(1.412e9)

    def test_get_reference_pixel_indices_for_lm_coords(self, image_xds_valid):
        """get_reference_pixel_indices should locate the pixel where l=0 and m=0."""

        xds = image_xds_valid

        indices = xds.xr_img.get_reference_pixel_indices()

        assert indices.shape == (2,)
        l_index, m_index = indices

        assert np.isclose(xds.coords["l"].values[l_index], 0.0)
        assert np.isclose(xds.coords["m"].values[m_index], 0.0)

    def test_get_reference_pixel_indices_for_uv_coords(self, image_xds_valid):
        """When uv coordinates are present, the reference indices should still match."""

        xds = image_xds_valid
        xds = xds.xr_img.add_uv_coordinates()

        indices = xds.xr_img.get_reference_pixel_indices()
        l_index, m_index = indices

        assert np.isclose(xds.coords["l"].values[l_index], 0.0)
        assert np.isclose(xds.coords["m"].values[m_index], 0.0)
        assert np.isclose(xds.coords["u"].values[l_index], 0.0)
        assert np.isclose(xds.coords["v"].values[m_index], 0.0)

    def test_get_reference_pixel_indices_outside_the_image(self, image_xds_valid):
        """A cutout that does not contain the reference direction has its
        reference pixel outside the image (F5): it is extrapolated."""
        xds = image_xds_valid
        l_index, m_index = xds.xr_img.get_reference_pixel_indices()
        cutout = xds.isel(l=slice(l_index + 2, l_index + 5), m=slice(0, m_index - 3))

        indices = cutout.xr_img.get_reference_pixel_indices()

        assert indices.dtype.kind == "i"
        assert indices.tolist() == [-2, m_index]

    def test_get_reference_pixel_indices_between_pixels(self, image_xds_valid):
        """A reference pixel between pixels is returned as a fractional pixel."""
        xds = image_xds_valid
        cell = xds.xr_img.get_lm_cell_size()
        xds = xds.assign_coords(l=xds.l + 0.25 * cell[0], m=xds.m - 0.5 * cell[1])
        l_index, m_index = image_xds_valid.xr_img.get_reference_pixel_indices()

        indices = xds.xr_img.get_reference_pixel_indices()

        assert indices.dtype.kind == "f"
        np.testing.assert_allclose(indices, [l_index - 0.25, m_index + 0.5])

    def test_get_reference_pixel_indices_of_an_aperture_image(self, image_xds_valid):
        """An image with only u and v coordinates gives the u = v = 0 pixel."""
        xds = image_xds_valid.xr_img.add_uv_coordinates()
        expected = xds.xr_img.get_reference_pixel_indices()
        xds = xds.drop_vars(
            ["l", "m", "right_ascension", "declination"], errors="ignore"
        )

        indices = xds.xr_img.get_reference_pixel_indices()

        assert indices.tolist() == expected.tolist()

    def test_add_data_group_adds_new_group(self, image_xds_valid):
        """add_data_group should add a new data group without modifying the base group."""

        xds = image_xds_valid
        original_groups = copy.deepcopy(xds.attrs["data_groups"])

        new_group_name = "new_group"
        new_group_spec = {"sky": "SKY_NEW"}

        xds_with_group = xds.xr_img.add_data_group(new_group_name, new_group_spec)

        assert new_group_name in xds_with_group.attrs["data_groups"]
        new_group = xds_with_group.attrs["data_groups"][new_group_name]

        # New group should contain the explicitly provided variable and inherit flag.
        assert new_group["sky"] == "SKY_NEW"
        assert new_group["flag"] == original_groups["base"]["flag"]

        # Base group should remain unchanged.
        assert xds_with_group.attrs["data_groups"]["base"] == original_groups["base"]

    def test_sel_without_data_group_name_delegates_to_xarray_sel(self, image_xds_valid):
        """sel without data_group_name should behave like xarray.Dataset.sel."""

        xds = image_xds_valid

        selected_accessor = xds.xr_img.sel(polarization="I")
        selected_direct = xds.sel(polarization="I")

        xr.testing.assert_identical(selected_accessor, selected_direct)

    @pytest.mark.parametrize(
        "sel_kwargs",
        [
            pytest.param({"data_group_name": "base"}, id="kwarg"),
            pytest.param({"indexers": {"data_group_name": "base"}}, id="indexers_dict"),
        ],
    )
    def test_sel_with_data_group_name_filters_data_vars_and_attrs(
        self, image_xds_valid, sel_kwargs
    ):
        """sel with data_group_name keeps only the selected group's variables.

        Exercises both calling conventions:
        - ``sel(data_group_name=...)`` (keyword argument)
        - ``sel(indexers={"data_group_name": ...})`` (indexers dict)
        """
        xds = image_xds_valid
        shape = (
            xds.sizes["time"],
            xds.sizes["frequency"],
            xds.sizes["polarization"],
            xds.sizes["l"],
            xds.sizes["m"],
        )
        xds = xds.copy()
        xds["SKY"] = xr.DataArray(
            np.zeros(shape, dtype=float),
            dims=("time", "frequency", "polarization", "l", "m"),
        )
        xds["POINT_SPREAD_FUNCTION"] = xr.DataArray(
            np.zeros(shape, dtype=float),
            dims=("time", "frequency", "polarization", "l", "m"),
        )
        xds.attrs["data_groups"] = {
            "base": {"sky": "SKY"},
            "psf": {"point_spread_function": "POINT_SPREAD_FUNCTION"},
        }

        original_kwargs = copy.deepcopy(sel_kwargs)

        selected = xds.xr_img.sel(**sel_kwargs)

        assert "SKY" in selected.data_vars
        assert "POINT_SPREAD_FUNCTION" not in selected.data_vars
        assert selected.attrs["data_groups"] == {
            "base": xds.attrs["data_groups"]["base"]
        }
        # The selection owns its data groups and leaves the caller's alone
        assert (
            selected.attrs["data_groups"]["base"]
            is not xds.attrs["data_groups"]["base"]
        )
        assert set(xds.attrs["data_groups"]) == {"base", "psf"}
        assert sel_kwargs == original_kwargs


def _image_with_data_groups():
    """An image dataset with two data groups sharing the flag and the PSF."""
    xds = _make_valid_image_dataset()
    dims = ("time", "frequency", "polarization", "l", "m")
    shape = tuple(xds.sizes[dim] for dim in dims)
    for name, dtype in (
        ("SKY", float),
        ("SKY_RESIDUAL", float),
        ("FLAG_SKY", bool),
        ("POINT_SPREAD_FUNCTION", float),
    ):
        xds[name] = xr.DataArray(np.zeros(shape, dtype=dtype), dims=dims)
    xds.attrs["data_groups"] = {
        "base": {
            "sky": "SKY",
            "flag": "FLAG_SKY",
            "point_spread_function": "POINT_SPREAD_FUNCTION",
        },
        "residual": {
            "sky": "SKY_RESIDUAL",
            "flag": "FLAG_SKY",
            "point_spread_function": "POINT_SPREAD_FUNCTION",
        },
    }
    return xds


_DERIVED_DATASETS = [
    pytest.param(lambda xds: xds.isel(frequency=[0]), id="isel"),
    pytest.param(lambda xds: xds.sel(polarization=["I"]), id="sel"),
    pytest.param(lambda xds: xds.copy(deep=False), id="shallow_copy"),
    pytest.param(lambda xds: xds.compute(), id="compute"),
    pytest.param(lambda xds: xds.where(xds.SKY == 0), id="where"),
    pytest.param(lambda xds: xds.chunk(), id="chunk"),
    pytest.param(
        lambda xds: xds.xr_img.sel(data_group_name="residual"), id="xr_img_sel"
    ),
]


class TestDataGroupsAreNotShared:
    """Deleting variables or adding data groups changes only the dataset the
    accessor is called on: derived datasets share the attrs dict (isel, sel)
    or the data group dicts (copy, compute, where, chunk) with their parent."""

    @pytest.mark.parametrize("derive", _DERIVED_DATASETS)
    def test_delete_from_derived_keeps_parent(self, derive):
        parent = _image_with_data_groups()
        expected = copy.deepcopy(parent.attrs["data_groups"])
        derived = derive(parent)

        derived.xr_img.delete_data_variables(["FLAG_SKY"])

        assert "FLAG_SKY" not in derived.data_vars
        assert all("flag" not in g for g in derived.attrs["data_groups"].values())
        assert "FLAG_SKY" in parent.data_vars
        assert parent.attrs["data_groups"] == expected

    @pytest.mark.parametrize("derive", _DERIVED_DATASETS)
    def test_delete_from_parent_keeps_derived(self, derive):
        parent = _image_with_data_groups()
        derived = derive(parent)
        expected = copy.deepcopy(derived.attrs["data_groups"])

        parent.xr_img.delete_data_variables(["FLAG_SKY", "POINT_SPREAD_FUNCTION"])

        assert parent.attrs["data_groups"] == {
            "base": {"sky": "SKY"},
            "residual": {"sky": "SKY_RESIDUAL"},
        }
        assert "FLAG_SKY" in derived.data_vars
        assert derived.attrs["data_groups"] == expected

    def test_delete_returns_the_dataset_and_keeps_other_attrs(self):
        xds = _image_with_data_groups()
        xds.attrs["note"] = "kept"

        result = xds.xr_img.delete_data_variables("SKY_RESIDUAL")

        assert result is xds
        assert "SKY_RESIDUAL" not in xds.data_vars
        assert xds.attrs["note"] == "kept"
        assert xds.attrs["data_groups"]["residual"] == {
            "flag": "FLAG_SKY",
            "point_spread_function": "POINT_SPREAD_FUNCTION",
        }

    def test_delete_removes_attributes_naming_the_variable(self):
        """The image attributes that name a deleted variable go too, without
        changing datasets that share the variable objects (drop_vars)."""
        xds = _image_with_data_groups()
        xds["SKY"].attrs = {"flag": "FLAG_SKY", "units": "Jy/beam"}
        xds["POINT_SPREAD_FUNCTION"].attrs = {"beam_fit_params": "FLAG_SKY"}
        other = xds.drop_vars("SKY_RESIDUAL")
        assert other["SKY"].variable is xds["SKY"].variable

        xds.xr_img.delete_data_variables(["FLAG_SKY"])

        assert xds["SKY"].attrs == {"units": "Jy/beam"}
        assert xds["POINT_SPREAD_FUNCTION"].attrs == {}
        assert other["SKY"].attrs == {"flag": "FLAG_SKY", "units": "Jy/beam"}
        assert other["POINT_SPREAD_FUNCTION"].attrs == {"beam_fit_params": "FLAG_SKY"}
        np.testing.assert_array_equal(xds["SKY"].values, other["SKY"].values)

    def test_delete_unknown_variable_deletes_nothing(self):
        xds = _image_with_data_groups()
        expected = copy.deepcopy(xds.attrs["data_groups"])

        with pytest.raises(ValueError, match="NOT_THERE"):
            xds.xr_img.delete_data_variables(["FLAG_SKY", "NOT_THERE"])

        assert "FLAG_SKY" in xds.data_vars
        assert xds.attrs["data_groups"] == expected

    @pytest.mark.parametrize("derive", _DERIVED_DATASETS)
    def test_add_data_group_keeps_other_datasets(self, derive):
        parent = _image_with_data_groups()
        derived = derive(parent)
        parent_groups = set(parent.attrs["data_groups"])
        derived_groups = set(derived.attrs["data_groups"])

        derived.xr_img.add_data_group("extra", {"sky": "SKY"})

        assert set(derived.attrs["data_groups"]) == derived_groups | {"extra"}
        assert set(parent.attrs["data_groups"]) == parent_groups


# ---------------------------------------------------------------------------
# FITS header and WCS (#655)
# ---------------------------------------------------------------------------


def _sky_image(
    pols=("I", "Q", "U", "V"),
    frame="fk5",
    projection="SIN",
    times=(59000.5, 59000.75),
    image_size=(8, 6),
    dtype=np.float32,
):
    """Sky image dataset whose data groups ``base`` and ``residual`` share a
    point spread function: base has flags and per plane beams, the point
    spread function a single beam."""
    cell = np.pi / 180 / 60  # 1 arcmin
    xds = make_empty_sky_image(
        phase_center=[0.5, -0.4],
        image_size=list(image_size),
        cell_size=[cell, cell],
        frequency_coords=[1.4e9, 1.401e9],
        pol_coords=list(pols),
        time_coords=list(times),
        direction_reference=frame,
        projection=projection,
        do_sky_coords=True,
    )
    dims = ("time", "frequency", "polarization", "l", "m")
    shape = tuple(xds.sizes[dim] for dim in dims)
    rng = np.random.default_rng(0)
    for name, image_type in (
        ("SKY", "sky"),
        ("SKY_RESIDUAL", "sky"),
        ("POINT_SPREAD_FUNCTION", "point_spread_function"),
    ):
        xds[name] = (
            dims,
            rng.standard_normal(shape).astype(dtype),
            {"type": image_type, "units": "Jy/beam"},
        )
    xds["FLAG_SKY"] = (dims, rng.random(shape) < 0.1, {"type": "flag"})
    beam_dims = ("time", "frequency", "polarization", "beam_params_label")
    beams = np.empty(shape[:3] + (3,))
    beams[...] = [1e-4, 5e-5, 0.3]
    xds = xds.assign_coords(beam_params_label=["major", "minor", "pa"])
    xds["BEAM_FIT_PARAMS_PSF"] = (beam_dims, beams.copy(), {"units": "rad"})
    beams[:, 1, :, 0] *= 1.1  # per plane beams
    xds["BEAM_FIT_PARAMS_SKY"] = (beam_dims, beams, {"units": "rad"})
    xds.attrs["data_groups"] = {
        "base": {
            "sky": "SKY",
            "flag": "FLAG_SKY",
            "beam_fit_params_sky": "BEAM_FIT_PARAMS_SKY",
            "point_spread_function": "POINT_SPREAD_FUNCTION",
            "beam_fit_params_point_spread_function": "BEAM_FIT_PARAMS_PSF",
        },
        "residual": {
            "sky": "SKY_RESIDUAL",
            "point_spread_function": "POINT_SPREAD_FUNCTION",
        },
    }
    return xds


@pytest.fixture(scope="module")
def read_images(tmp_path_factory):
    """CASA and FITS images opened with open_image."""
    directory = tmp_path_factory.mktemp("wcs_images")
    names = ("casa_test_image.im", "test_image.fits", "demo_simulated.im")
    return {name: open_image(str(download_image(name, directory))) for name in names}


class TestFitsHeader:
    """get_fits_header: the header write_image writes (#655)."""

    @pytest.mark.parametrize(
        "image",
        [
            (("I", "Q", "U", "V"), "fk5", "SIN"),
            (("RR", "RL", "LR", "LL"), "fk4", "ARC"),
            (("XX", "XY", "YX", "YY"), "icrs", "TAN"),
            (("XX", "YY"), "galactic", "SIN"),
            "casa_test_image.im",
            "test_image.fits",
            "demo_simulated.im",
        ],
    )
    def test_header_is_the_header_written(self, tmp_path, request, image):
        """For every image role of every data group and every time plane, the
        header is that of the file write_image writes, byte for byte: for
        synthetic images (polarizations, frame, projection) and for images
        read by open_image."""
        if isinstance(image, str):
            xds = request.getfixturevalue("read_images")[image]
        else:
            xds = _sky_image(*image)
        roles = ("sky", "point_spread_function")
        for time in range(xds.sizes["time"]):
            plane = xds.isel(time=[time])
            name = str(tmp_path / f"t{time}.fits")
            write_image(plane, name, out_format="fits")
            path_of = {
                output.variable: output.path
                for output in plan_image_outputs(plane, name, "fits").outputs
            }
            for group, members in xds.attrs["data_groups"].items():
                for role in (role for role in roles if role in members):
                    header = xds.xr_img.get_fits_header(group, role, time)
                    assert type(header) is fits.Header
                    expected = header.tostring().encode("ascii")
                    with open(path_of[members[role]], "rb") as f:
                        assert f.read(len(expected)) == expected, (group, role, time)

    def test_default_data_group(self):
        xds = _sky_image()
        groups = xds.attrs["data_groups"]
        # base, also when it is not the first data group
        xds.attrs["data_groups"] = {
            "residual": groups["residual"],
            "base": groups["base"],
        }
        header = xds.xr_img.get_fits_header()
        assert header == xds.xr_img.get_fits_header(data_group_name="base")
        assert header != xds.xr_img.get_fits_header(data_group_name="residual")
        # without a base data group, the first one
        xds.attrs["data_groups"] = {
            "residual": groups["residual"],
            "dirty": groups["base"],
        }
        header = xds.xr_img.get_fits_header()
        assert header == xds.xr_img.get_fits_header(data_group_name="residual")
        assert header != xds.xr_img.get_fits_header(data_group_name="dirty")

    def test_shared_image_has_the_beams_written_with_it(self):
        """An image that data groups share is written once, with the beam fit
        parameters of the first data group that names them."""
        xds = _sky_image()
        beams = xds["BEAM_FIT_PARAMS_PSF"]
        xds["BEAM_FIT_PARAMS_PSF_2"] = beams * 2
        xds.attrs["data_groups"]["residual"][
            "beam_fit_params_point_spread_function"
        ] = "BEAM_FIT_PARAMS_PSF_2"
        role = "point_spread_function"
        bmaj = np.degrees(beams.values[0, 0, 0, 0])
        for group in ("base", "residual"):
            header = xds.xr_img.get_fits_header(group, role)
            np.testing.assert_allclose(header["BMAJ"], bmaj, rtol=1e-12)

    def test_no_pixels_are_read(self):
        xds = _sky_image()

        def unread(block):
            raise AssertionError("the pixels were read")

        pixels = da.from_array(xds["SKY"].values, chunks=-1)
        xds["SKY"] = xds["SKY"].copy(
            data=pixels.map_blocks(unread, meta=np.array((), dtype=np.float32))
        )
        assert xds.xr_img.get_fits_header()["NAXIS"] == 4
        assert xds.xr_img.get_wcs(celestial=False).naxis == 4

    def test_per_plane_beams(self):
        header = _sky_image().xr_img.get_fits_header()
        assert header["CASAMBM"] is True
        assert "BMAJ" not in header

    def test_polarizations_without_a_linear_stokes_axis(self):
        """FITS cannot hold I, Q, V: no header and no full WCS, but the
        celestial WCS exists."""
        xds = _sky_image(pols=("I", "Q", "V"))
        with pytest.raises(ValueError, match="linear axis"):
            xds.xr_img.get_fits_header()
        with pytest.raises(ValueError, match="linear axis"):
            xds.xr_img.get_wcs(celestial=False)
        assert xds.xr_img.get_wcs().array_shape == (6, 8)

    def test_complex_image(self):
        xds = _sky_image(dtype=np.complex64)
        with pytest.raises(ValueError, match="complex"):
            xds.xr_img.get_fits_header()
        with pytest.raises(ValueError, match="complex"):
            xds.xr_img.get_wcs(celestial=False)
        assert xds.xr_img.get_wcs().array_shape == (6, 8)

    @pytest.mark.parametrize(
        "kwargs, error, match",
        [
            ({"data_group_name": "nonexistent"}, ValueError, "no data group"),
            ({"role": "flag"}, ValueError, "not an image role"),
            ({"role": "primary_beam"}, ValueError, "has no 'primary_beam' image"),
            ({"time": 2}, IndexError, "out of bounds"),
            ({"time": -3}, IndexError, "out of bounds"),
        ],
    )
    def test_errors(self, kwargs, error, match):
        xds = _sky_image()
        with pytest.raises(error, match=match):
            xds.xr_img.get_fits_header(**kwargs)
        with pytest.raises(error, match=match):
            xds.xr_img.get_wcs(**kwargs)

    def test_another_image_fits_cannot_hold(self, tmp_path):
        """As write_image writes no image of a dataset with an image that FITS
        cannot hold, no image gets a header or a full WCS; the celestial WCS
        exists."""
        xds = _sky_image()
        psf = xds["POINT_SPREAD_FUNCTION"]
        xds["POINT_SPREAD_FUNCTION"] = psf.astype(np.complex64)
        name = str(tmp_path / "image.fits")
        with pytest.raises(ValueError, match="POINT_SPREAD_FUNCTION .*complex"):
            write_image(xds.isel(time=[0]), name, out_format="fits")
        for group in ("base", "residual"):
            with pytest.raises(ValueError, match="POINT_SPREAD_FUNCTION .*complex"):
                xds.xr_img.get_fits_header(group)
            with pytest.raises(ValueError, match="POINT_SPREAD_FUNCTION .*complex"):
                xds.xr_img.get_wcs(group, celestial=False)
            assert xds.xr_img.get_wcs(group).array_shape == (6, 8)

    def test_problem_in_another_data_group(self, tmp_path):
        """A dataset that write_image cannot plan gets neither a header nor a
        WCS, even when the problem is in another data group."""
        xds = _sky_image()
        xds.attrs["data_groups"]["residual"]["flag"] = "FLAG_MISSING"
        name = str(tmp_path / "image.fits")
        with pytest.raises(ValueError, match="FLAG_MISSING"):
            write_image(xds.isel(time=[0]), name, out_format="fits")
        with pytest.raises(ValueError, match="FLAG_MISSING"):
            xds.xr_img.get_fits_header(data_group_name="base")
        for celestial in (True, False):
            with pytest.raises(ValueError, match="FLAG_MISSING"):
                xds.xr_img.get_wcs(data_group_name="base", celestial=celestial)

    def test_aperture_image_has_no_header(self):
        xds = _sky_image()
        xds["APERTURE"] = xr.DataArray(
            np.zeros((2, 2, 4, 8, 6), dtype=np.complex64),
            dims=("time", "frequency", "polarization", "u", "v"),
            attrs={"type": "aperture"},
        )
        xds.attrs["data_groups"]["base"]["aperture"] = "APERTURE"
        with pytest.raises(ValueError, match="not written to FITS"):
            xds.xr_img.get_fits_header(role="aperture")
        with pytest.raises(ValueError, match="not written to FITS"):
            xds.xr_img.get_wcs(role="aperture")

    def test_unsupported_direction_frame(self):
        xds = _sky_image()
        reference = xds.attrs["coordinate_system_info"]["reference_direction"]
        reference["attrs"]["frame"] = "supergalactic"
        with pytest.raises(RuntimeError, match="direction reference frames"):
            xds.xr_img.get_wcs()


class TestWcs:
    """get_wcs: the astropy WCS of the FITS header (#655)."""

    def test_celestial_array_shape_is_m_l(self):
        xds = _sky_image(image_size=(8, 6))
        wcs = xds.xr_img.get_wcs()
        assert wcs.naxis == 2
        assert wcs.array_shape == (6, 8)
        assert wcs.pixel_shape == (8, 6)
        assert WCS(xds.xr_img.get_fits_header()).celestial.array_shape == (6, 8)

    @pytest.mark.parametrize(
        "frame, projection",
        [
            ("fk5", "SIN"),
            ("fk4", "ARC"),
            ("icrs", "TAN"),
            ("galactic", "SIN"),
            ("galactic", "CAR"),
        ],
    )
    def test_pixel_to_world_gives_the_sky_coordinates(self, frame, projection):
        """pixel_to_world takes the pixel indices along l and m and gives the
        dataset's sky coordinates, in the image's frame; world_to_pixel
        returns them."""
        xds = _sky_image(frame=frame, projection=projection)
        wcs = xds.xr_img.get_wcs()
        il, im = np.indices((xds.sizes["l"], xds.sizes["m"]))
        lon, lat = wcs.pixel_to_world_values(il, im)
        names = (
            ("galactic_longitude", "galactic_latitude")
            if frame == "galactic"
            else ("right_ascension", "declination")
        )
        expected_lon, expected_lat = (np.degrees(xds[name].values) for name in names)
        tolerance = 1e-9 / 3600  # degrees
        assert np.abs((lon - expected_lon + 180) % 360 - 180).max() < tolerance
        assert np.abs(lat - expected_lat).max() < tolerance
        sky = wcs.pixel_to_world(il, im)
        assert sky.frame.name == frame
        back = wcs.world_to_pixel(sky)
        np.testing.assert_allclose(back, (il, im), rtol=0, atol=1e-9)

    @pytest.mark.parametrize("frame", ["fk5", "icrs", "galactic"])
    def test_l_and_m_are_the_direction_cosines(self, frame):
        """For SIN, l and m are the direction cosines of the pixel's sky
        position relative to the reference direction (AIPS Memo 27): the WCS
        x axis runs along l and y along m, without a flip."""
        xds = _sky_image(frame=frame)
        il, im = np.indices((xds.sizes["l"], xds.sizes["m"]))
        sky = xds.xr_img.get_wcs().pixel_to_world(il, im).spherical
        lon, lat = sky.lon.rad, sky.lat.rad
        reference = xds.attrs["coordinate_system_info"]["reference_direction"]
        lon0, lat0 = reference["data"]
        l_values = np.cos(lat) * np.sin(lon - lon0)
        m_values = np.sin(lat) * np.cos(lat0) - np.cos(lat) * np.sin(lat0) * np.cos(
            lon - lon0
        )
        np.testing.assert_allclose(l_values, xds.l.values[il], rtol=0, atol=1e-14)
        np.testing.assert_allclose(m_values, xds.m.values[im], rtol=0, atol=1e-14)

    @pytest.mark.parametrize(
        "name", ["casa_test_image.im", "test_image.fits", "demo_simulated.im"]
    )
    def test_read_images(self, read_images, name):
        """The celestial WCS gives the reader's sky coordinates and the same
        positions as the celestial axes of the header."""
        xds = read_images[name]
        wcs = xds.xr_img.get_wcs()
        il, im = np.indices((xds.sizes["l"], xds.sizes["m"]))
        lon, lat = wcs.pixel_to_world_values(il, im)
        # the images are in the fk5 frame
        expected_lon = np.degrees(xds.right_ascension.values)
        expected_lat = np.degrees(xds.declination.values)
        tolerance = 1e-9 / 3600  # degrees
        assert np.abs((lon - expected_lon + 180) % 360 - 180).max() < tolerance
        assert np.abs(lat - expected_lat).max() < tolerance
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FITSFixedWarning)
            full = WCS(xds.xr_img.get_fits_header()).celestial
        np.testing.assert_array_equal(
            wcs.pixel_to_world_values(il, im), full.pixel_to_world_values(il, im)
        )

    def test_full_wcs(self):
        """celestial=False: the axes of the FITS file, without a time axis;
        the time of the plane is the observation date."""
        xds = _sky_image(pols=("RR", "RL", "LR", "LL"))
        wcs = xds.xr_img.get_wcs(time=1, celestial=False)
        assert list(wcs.wcs.ctype) == ["RA---SIN", "DEC--SIN", "STOKES", "FREQ"]
        assert wcs.array_shape == (2, 4, 6, 8)
        expected = Time(xds.time.values[1], format="mjd", scale="utc")
        assert wcs.wcs.mjdobs == pytest.approx(expected.mjd, abs=1e-9)
        assert Time(wcs.wcs.dateobs, scale="utc").mjd == pytest.approx(
            expected.mjd, abs=1e-9
        )
        assert xds.xr_img.get_wcs(time=1).wcs.mjdobs == wcs.wcs.mjdobs
        # negative indices count from the end
        last = xds.xr_img.get_wcs(time=-1, celestial=False)
        assert last.wcs.mjdobs == wcs.wcs.mjdobs
        assert xds.xr_img.get_fits_header(time=-1) == xds.xr_img.get_fits_header(time=1)
        assert xds.xr_img.get_fits_header(time=-2) == xds.xr_img.get_fits_header(time=0)
        assert wcs.wcs.specsys == "LSRK"
        np.testing.assert_allclose(
            wcs.sub([4]).pixel_to_world_values([0, 1]), xds.frequency.values
        )

    @pytest.mark.parametrize(
        "pols, file_order",
        [
            (("I", "Q", "U", "V"), ["I", "Q", "U", "V"]),
            (("RR", "RL", "LR", "LL"), ["RR", "LL", "RL", "LR"]),
            (("XX", "XY", "YX", "YY"), ["XX", "YY", "XY", "YX"]),
            (("RR", "LL"), ["RR", "LL"]),
            (("XX", "YY"), ["XX", "YY"]),
        ],
    )
    def test_stokes_axis_in_file_order(self, pols, file_order):
        wcs = _sky_image(pols=pols).xr_img.get_wcs(celestial=False)
        codes = wcs.sub([3]).pixel_to_world_values(np.arange(len(pols)))
        assert [FITS_STOKES_LABELS[int(code)] for code in codes] == file_order

    def test_no_wcslib_warnings(self, read_images):
        """wcslib's note that it derives OBSGEO-L/B/H from OBSGEO-X/Y/Z is not
        passed on."""
        xds = read_images["casa_test_image.im"]
        with warnings.catch_warnings():
            warnings.simplefilter("error", AstropyWarning)
            xds.xr_img.get_wcs(celestial=False)
            xds.xr_img.get_wcs()


# ---------------------------------------------------------------------------
# Parametrize tables for TestImageXdsInvalid
# ---------------------------------------------------------------------------
# Each entry is (call, id) where call(xds, xds_with_uv) invokes one accessor
# method.  Defined at module level so pytest can collect them without
# instantiating the class.
_INVALID_TYPE_CALLS = [
    pytest.param(lambda xds, uv: xds.xr_img.test_func(), id="test_func"),
    pytest.param(
        lambda xds, uv: xds.xr_img.add_data_group("g", {}), id="add_data_group"
    ),
    pytest.param(lambda xds, uv: xds.xr_img.get_lm_cell_size(), id="get_lm_cell_size"),
    pytest.param(
        lambda xds, uv: xds.xr_img.add_uv_coordinates(), id="add_uv_coordinates"
    ),
    pytest.param(
        lambda xds, uv: uv.xr_img.get_uv_in_lambda(1.412e9), id="get_uv_in_lambda"
    ),
    pytest.param(
        lambda xds, uv: xds.xr_img.get_reference_pixel_indices(),
        id="get_reference_pixel_indices",
    ),
    pytest.param(lambda xds, uv: xds.xr_img.sel(polarization="I"), id="sel"),
    pytest.param(lambda xds, uv: xds.xr_img.get_fits_header(), id="get_fits_header"),
    pytest.param(lambda xds, uv: xds.xr_img.get_wcs(), id="get_wcs"),
]

_INVALID_TYPE_VALUES = [
    pytest.param("image", False, id="type_image"),
    pytest.param("other", False, id="type_other"),
    pytest.param(None, True, id="no_type"),
]


class TestImageXdsInvalid:
    """Test suite for ImageXdsAccessor (xr_img) with invalid image datasets."""

    @pytest.mark.parametrize("call", _INVALID_TYPE_CALLS)
    @pytest.mark.parametrize("type_value,delete_type", _INVALID_TYPE_VALUES)
    def test_invalid_type_raises_invalid_accessor_location(
        self, image_xds_valid, call, type_value, delete_type
    ):
        """Each xr_img method rejects datasets whose type is not image_dataset.

        Produces 27 independent items (9 methods × 3 invalid-type variants) so
        a single regression is pinpointed without masking the remaining checks.
        """
        base_xds = image_xds_valid
        base_xds_with_uv = base_xds.xr_img.add_uv_coordinates()

        def make_test_xds(template):
            test_xds = template.copy()
            if delete_type:
                test_xds.attrs.pop("type", None)
            else:
                test_xds.attrs["type"] = type_value
            return test_xds

        with pytest.raises(InvalidAccessorLocation):
            call(make_test_xds(base_xds), make_test_xds(base_xds_with_uv))

    def test_invalid_type_error_message_includes_path_and_text(self, image_xds_valid):
        """Error message for invalid accessor location should include the dataset path."""

        xds = image_xds_valid.copy()
        xds.attrs["type"] = "image"

        with pytest.raises(InvalidAccessorLocation) as excinfo:
            xds.xr_img.test_func()

        message = str(excinfo.value)
        assert "In-memory xds" in message

    def test_get_reference_pixel_indices_raises_when_no_coords(self, image_xds_valid):
        """get_reference_pixel_indices should fail if neither lm nor uv coordinates exist."""

        xds = image_xds_valid.copy()
        xds = xds.drop_vars(["l", "m"])

        with pytest.raises(ValueError) as excinfo:
            xds.xr_img.get_reference_pixel_indices()

        assert "No lm or uv coordinates found" in str(excinfo.value)

    def test_get_reference_pixel_indices_raises_when_coords_mismatch(
        self, image_xds_valid
    ):
        """get_reference_pixel_indices should assert if lm and uv reference indices differ."""

        xds = image_xds_valid.xr_img.add_uv_coordinates()

        # Force the uv reference pixel (where u == 0) to be at a different index
        # than the lm reference pixel (center of the image).
        u_vals = xds.coords["u"].values.copy()
        u_vals = u_vals + 1.0  # shift away any existing zeros
        u_vals[0] = 0.0  # place the uv zero at index 0
        xds = xds.assign_coords(u=("l", u_vals))

        with pytest.raises(AssertionError) as excinfo:
            xds.xr_img.get_reference_pixel_indices()

        assert "lm and uv reference pixel indices do not match" in str(excinfo.value)

    @pytest.mark.parametrize("isel_kwargs", [{"l": 0}, {"m": 0}])
    def test_single_pixel_l_or_m_raises_index_error(self, image_xds_valid, isel_kwargs):
        """get_lm_cell_size and add_uv_coordinates should fail when l or m has size 1."""

        xds = image_xds_valid.isel(**isel_kwargs)

        with pytest.raises(IndexError):
            xds.xr_img.get_lm_cell_size()

        with pytest.raises(IndexError):
            xds.xr_img.add_uv_coordinates()

    def test_add_data_group_raises_when_data_groups_missing(self, image_xds_valid):
        """add_data_group should fail if the dataset has no data_groups attribute."""

        xds = image_xds_valid.copy()
        xds.attrs.pop("data_groups", None)

        with pytest.raises(KeyError):
            xds.xr_img.add_data_group("g", {"sky": "SKY"})

    def test_sel_raises_when_data_group_name_unknown(self, image_xds_valid):
        """sel should fail with KeyError when the requested data_group_name is unknown."""

        xds = image_xds_valid

        with pytest.raises(KeyError):
            xds.xr_img.sel(data_group_name="nonexistent")


if __name__ == "__main__":
    pytest.main(["-v", "-s", __file__])
