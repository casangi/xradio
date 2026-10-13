import copy

import numpy as np
import pytest
import xarray as xr

from xradio.image import make_empty_sky_image
from xradio.image.image_xds import InvalidAccessorLocation
from xradio.testing.image import create_empty_test_image


def _make_valid_image_dataset():
    """Create a minimal Dataset that is valid for the ImageXds accessor.

    Built with :func:`~xradio.testing.image.create_empty_test_image`
    (``make_empty_sky_image`` factory, sky coordinates enabled) and promoted
    to an ``image_dataset`` node with a minimal ``data_groups`` mapping so
    that ImageXds methods accept it as an image node.
    """

    xds = create_empty_test_image(make_empty_sky_image, do_sky_coords=True)

    # Promote the dataset to an ImageXds-compatible image_dataset node.
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
    """Fixture providing a fresh ImageXds-compatible Dataset per test."""
    return _make_valid_image_dataset()


class TestImageXdsValid:
    """Test suite for ImageXds accessor with valid image datasets."""

    def test_xr_img_accessor_registration(self, image_xds_valid):
        """ImageXds accessor should be registered and keep a reference to the dataset."""

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
]

_INVALID_TYPE_VALUES = [
    pytest.param("image", False, id="type_image"),
    pytest.param("other", False, id="type_other"),
    pytest.param(None, True, id="no_type"),
]


class TestImageXdsInvalid:
    """Test suite for ImageXds accessor with invalid image datasets."""

    @pytest.mark.parametrize("call", _INVALID_TYPE_CALLS)
    @pytest.mark.parametrize("type_value,delete_type", _INVALID_TYPE_VALUES)
    def test_invalid_type_raises_invalid_accessor_location(
        self, image_xds_valid, call, type_value, delete_type
    ):
        """Each ImageXds method rejects datasets whose type is not image_dataset.

        Produces 21 independent items (7 methods × 3 invalid-type variants) so
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
