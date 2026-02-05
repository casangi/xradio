import copy

import numpy as np
import pytest
import xarray as xr

from xradio.image import make_empty_sky_image
from xradio.image.image_xds import InvalidAccessorLocation


def _default_image_args():
    """Return the default arguments used to build empty image datasets.

    These mirror the values used in ``make_empty_image_tests.create_image`` in
    ``tests/unit/image/test_image.py`` so that ImageXds tests exercise the same
    coordinate setup.
    """

    phase_center = [0.2, -0.5]
    image_size = [10, 10]
    cell_size = [np.pi / 180 / 60, np.pi / 180 / 60]
    frequency_coords = [1.412e9, 1.413e9]
    pol_coords = ["I", "Q", "U"]
    time_coords = [54000.1]
    return (
        phase_center,
        image_size,
        cell_size,
        frequency_coords,
        pol_coords,
        time_coords,
    )


def _make_valid_image_dataset():
    """Create a minimal Dataset that is valid for the ImageXds accessor.

    The dataset is constructed with ``make_empty_sky_image`` using the same
    arguments as ``make_empty_image_tests.create_image`` and then promoted
    to an ``image_dataset`` type with a simple ``data_groups`` mapping so
    that ImageXds methods accept it as an image node.
    """

    (
        phase_center,
        image_size,
        cell_size,
        frequency_coords,
        pol_coords,
        time_coords,
    ) = _default_image_args()

    xds = make_empty_sky_image(
        phase_center,
        image_size,
        cell_size,
        frequency_coords,
        pol_coords,
        time_coords,
        do_sky_coords=True,
    )

    # Make a shallow copy so tests can mutate coordinates and attrs safely.
    xds = xds.copy()

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


class TestImageXdsValid:
    """Test suite for ImageXds accessor with valid image datasets."""

    @pytest.fixture
    def image_xds_valid(self):
        """Fixture providing a fresh ImageXds-compatible Dataset per test."""
        return _make_valid_image_dataset()

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

    def test_get_uv_in_lambda_for_specific_frequency(self, image_xds_valid):
        """get_uv_in_lambda should convert uv coordinates from meters to wavelengths."""

        xds = image_xds_valid
        xds = xds.xr_img.add_uv_coordinates()

        frequency = 1.412e9
        u_in_lambda, v_in_lambda = xds.xr_img.get_uv_in_lambda(frequency)

        c = 299792458.0
        wavelength = c / frequency

        # Converting back to meters should recover the original u and v.
        assert np.allclose(u_in_lambda.values * wavelength, xds.coords["u"].values)
        assert np.allclose(v_in_lambda.values * wavelength, xds.coords["v"].values)

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

    def test_sel_with_data_group_name_filters_data_vars_and_attrs(
        self, image_xds_valid
    ):
        """sel with data_group_name should keep only variables from the selected group."""

        xds = image_xds_valid

        # Add simple data variables that can be associated with different groups.
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

        selected = xds.xr_img.sel(data_group_name="base")

        # Only SKY (the base group's variable) should remain.
        assert "SKY" in selected.data_vars
        assert "POINT_SPREAD_FUNCTION" not in selected.data_vars

        # attrs['data_groups'] should be reduced to just the selected group.
        assert selected.attrs["data_groups"] == {
            "base": xds.attrs["data_groups"]["base"]
        }

    def test_sel_with_data_group_name_in_indexers_kwargs(self, image_xds_valid):
        """sel with data_group_name in indexers dict should behave like data_group_name= kwarg."""

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

        # Pass data_group_name via indexers dict (covers lines 227-228 in image_xds.py).
        selected = xds.xr_img.sel(indexers={"data_group_name": "base"})

        assert "SKY" in selected.data_vars
        assert "POINT_SPREAD_FUNCTION" not in selected.data_vars
        assert selected.attrs["data_groups"] == {
            "base": xds.attrs["data_groups"]["base"]
        }


class TestImageXdsInvalid:
    """Test suite for ImageXds accessor with invalid image datasets."""

    @pytest.fixture
    def image_xds_valid(self):
        """Fixture providing a fresh ImageXds-compatible Dataset per test."""
        return _make_valid_image_dataset()

    @pytest.mark.parametrize(
        "type_value, delete_type",
        [("image", False), ("other", False), (None, True)],
    )
    def test_invalid_type_raises_invalid_accessor_location(
        self, image_xds_valid, type_value, delete_type
    ):
        """All ImageXds methods that check type should reject non-image_dataset nodes."""

        base_xds = image_xds_valid

        # Prepare datasets for methods that need extra coordinates.
        base_xds_with_uv = base_xds.xr_img.add_uv_coordinates()

        def make_test_xds(template):
            test_xds = template.copy()
            if delete_type:
                test_xds.attrs.pop("type", None)
            else:
                test_xds.attrs["type"] = type_value
            return test_xds

        # Each method gets its own dataset so failures are independent.
        with pytest.raises(InvalidAccessorLocation):
            make_test_xds(base_xds).xr_img.test_func()

        with pytest.raises(InvalidAccessorLocation):
            make_test_xds(base_xds).xr_img.add_data_group("g", {})

        with pytest.raises(InvalidAccessorLocation):
            make_test_xds(base_xds).xr_img.get_lm_cell_size()

        with pytest.raises(InvalidAccessorLocation):
            make_test_xds(base_xds).xr_img.add_uv_coordinates()

        with pytest.raises(InvalidAccessorLocation):
            make_test_xds(base_xds_with_uv).xr_img.get_uv_in_lambda(1.412e9)

        with pytest.raises(InvalidAccessorLocation):
            make_test_xds(base_xds).xr_img.get_reference_pixel_indices()

        with pytest.raises(InvalidAccessorLocation):
            make_test_xds(base_xds).xr_img.sel(polarization="I")

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
