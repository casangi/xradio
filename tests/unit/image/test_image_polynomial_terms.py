"""Unit tests for the polynomial (Taylor) terms of image schema 0.0.4 (#661).

* ``TestPolynomialImageSchema`` → ``check_image`` of multi-term
  multi-frequency synthesis (MT-MFS) products: the ``frequency_polynomial``
  and ``frequency_polynomial_moment`` dimensions, ``time_polynomial``, the
  primary beam, ``ALPHA``, ``ALPHA_ERROR`` and ``BETA``, and the layouts it
  rejects
* ``TestMakeEmptyPolynomialImages`` → the ``n_polynomial_terms`` and
  ``n_time_polynomial_terms`` keywords of the ``make_empty_*`` factories
* ``TestPolynomialImageStorage`` → zarr round trip
"""

import copy

import numpy as np
import pytest
import xarray as xr

from xradio.image import (
    load_image,
    make_empty_aperture_image,
    make_empty_lmuv_image,
    make_empty_sky_image,
    open_image,
    write_image,
)
from xradio.image.schema import (
    FrequencyPolynomialCoordArray,
    FrequencyPolynomialMomentCoordArray,
    TimePolynomialCoordArray,
    check_image,
)
from xradio.schema.check import check_array

REFERENCE_FREQUENCY = 1.5e9
SHAPE_LM = (8, 6)
TERMS = ("time", "frequency_polynomial", "polarization", "l", "m")
MOMENTS = ("time", "frequency_polynomial_moment", "polarization", "l", "m")
CHANNELS = ("time", "frequency", "polarization", "l", "m")
BEAM_DIMS = ("time", "frequency", "polarization", "beam_params_label")
SKY_VERSIONS = (
    "SKY",
    "SKY_MODEL",
    "SKY_RESIDUAL",
    "SKY_PRIMARY_BEAM_CORRECTED_RESTORED",
)
SPECTRAL_PRODUCTS = {"ALPHA": "alpha", "ALPHA_ERROR": "alpha_error", "BETA": "beta"}


def _image(dims, sizes, dtype=np.float32, **attrs):
    shape = tuple(sizes[dim] for dim in dims)
    return xr.DataArray(np.ones(shape, dtype=dtype), dims=dims, attrs=attrs)


def make_mtmfs_image_xds(n_terms=3, frequencies=(REFERENCE_FREQUENCY,)):
    """The image dataset of an MT-MFS run with ``n_terms`` Taylor terms, as
    planned for image schema 0.0.4: Taylor coefficient images on
    ``frequency_polynomial``, the point spread function and the sum of
    weights on ``frequency_polynomial_moment``, the primary beam by its
    Taylor terms, and the mask, ALPHA, ALPHA_ERROR, BETA and the beam fit
    parameters on the ``frequency`` axis ``frequencies``. The polynomial
    coordinates share the reference frequency of the frequency coordinate."""
    xds = make_empty_sky_image(
        [0.2, -0.5], list(SHAPE_LM), [1e-5, 1e-5], list(frequencies), ["I"], [55000.5]
    )
    attrs = {
        "reference_frequency": copy.deepcopy(
            xds.frequency.attrs["reference_frequency"]
        ),
        "polynomial_type": "taylor",
    }
    xds = xds.assign_coords(
        frequency_polynomial=("frequency_polynomial", np.arange(n_terms), attrs),
        frequency_polynomial_moment=(
            "frequency_polynomial_moment",
            np.arange(2 * n_terms - 1),
            copy.deepcopy(attrs),
        ),
    )
    sizes = dict(xds.sizes)
    for name in SKY_VERSIONS:
        xds[name] = _image(TERMS, sizes, type="sky", units="Jy/beam")
    xds["SKY"].attrs["flag"] = "FLAG_SKY"
    xds["SKY"].attrs["beam_fit_params"] = "BEAM_FIT_PARAMS_SKY"
    xds["FLAG_SKY"] = _image(TERMS, sizes, dtype=bool, type="flag")
    xds["POINT_SPREAD_FUNCTION"] = _image(
        MOMENTS,
        sizes,
        type="point_spread_function",
        beam_fit_params="BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION",
    )
    xds["VISIBILITY_NORMALIZATION"] = _image(
        MOMENTS[:3], sizes, dtype=np.float64, type="visibility_normalization"
    )
    xds["PRIMARY_BEAM"] = _image(TERMS, sizes, type="primary_beam")
    xds["MASK"] = _image(CHANNELS, sizes, type="mask")
    if len(frequencies) == 1:
        for name, image_type in SPECTRAL_PRODUCTS.items():
            xds[name] = _image(CHANNELS, sizes, type=image_type)
    for name in ("BEAM_FIT_PARAMS_SKY", "BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION"):
        xds[name] = _image(
            BEAM_DIMS, sizes, dtype=np.float64, type=name.lower(), units="rad"
        )
    shared = {
        "point_spread_function": "POINT_SPREAD_FUNCTION",
        "beam_fit_params_point_spread_function": "BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION",
        "primary_beam": "PRIMARY_BEAM",
        "mask": "MASK",
        "visibility_normalization": "VISIBILITY_NORMALIZATION",
    }
    xds.attrs["data_groups"] = {
        "base": {
            "sky": "SKY",
            "flag": "FLAG_SKY",
            "beam_fit_params_sky": "BEAM_FIT_PARAMS_SKY",
            **shared,
        },
        "model": {"sky": "SKY_MODEL", **shared},
        "residual": {"sky": "SKY_RESIDUAL", **shared},
        "primary_beam_corrected_restored": {
            "sky": "SKY_PRIMARY_BEAM_CORRECTED_RESTORED",
            **shared,
        },
    }
    return xds


def _messages(issues):
    return [issue.message for issue in issues]


@pytest.fixture
def mtmfs_xds():
    return make_mtmfs_image_xds()


class TestPolynomialImageSchema:
    """check_image of MT-MFS products and polynomial terms."""

    def test_mtmfs_products_conform(self, mtmfs_xds):
        """Model, residual, restored and primary beam corrected Taylor terms,
        the point spread function and sum of weights moments, the primary
        beam's Taylor terms, the mask, ALPHA, ALPHA_ERROR, BETA and the beam
        fit parameters, in the existing data group roles."""
        assert mtmfs_xds.sizes["frequency_polynomial"] == 3
        assert mtmfs_xds.sizes["frequency_polynomial_moment"] == 5
        assert mtmfs_xds.sizes["frequency"] == 1
        issues = check_image(mtmfs_xds)
        assert not issues, str(issues)

    def test_two_terms(self):
        """nterms=2 has no BETA in CASA; the layout is the same."""
        xds = make_mtmfs_image_xds(n_terms=2).drop_vars("BETA")
        assert xds.sizes["frequency_polynomial_moment"] == 3
        assert not check_image(xds)

    def test_frequency_resolved_primary_beam_beside_taylor_products(self):
        """A frequency-resolved primary beam in the dataset of the Taylor
        products: every variable with a frequency axis has all its channels."""
        frequencies = np.linspace(1.0e9, 2.0e9, 4)
        xds = make_mtmfs_image_xds(frequencies=frequencies)
        xds["PRIMARY_BEAM"] = _image(CHANNELS, dict(xds.sizes), type="primary_beam")
        assert xds.sizes["frequency"] == 4
        assert xds["PRIMARY_BEAM"].dims == CHANNELS
        assert xds["SKY"].dims == TERMS
        assert not check_image(xds), str(check_image(xds))

    def test_dataset_without_frequency(self, mtmfs_xds):
        """Without products on the frequency axis a dataset needs no
        frequency coordinate."""
        xds = mtmfs_xds.drop_vars(
            ["MASK", *SPECTRAL_PRODUCTS, "BEAM_FIT_PARAMS_SKY"]
            + ["BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION"]
        ).drop_dims("frequency")
        xds.attrs["data_groups"] = {
            name: {
                role: variable
                for role, variable in group.items()
                if variable in xds.data_vars
            }
            for name, group in mtmfs_xds.attrs["data_groups"].items()
        }
        xds["SKY"].attrs.pop("beam_fit_params")
        xds["POINT_SPREAD_FUNCTION"].attrs.pop("beam_fit_params")
        assert "frequency" not in xds.dims
        assert not check_image(xds), str(check_image(xds))

    @pytest.mark.parametrize(
        "name,dims",
        [
            ("SKY", MOMENTS),
            ("PRIMARY_BEAM", MOMENTS),
            ("POINT_SPREAD_FUNCTION", TERMS),
            ("MASK", TERMS),
            ("BEAM_FIT_PARAMS_SKY", ("time", "frequency_polynomial") + BEAM_DIMS[2:]),
        ],
    )
    def test_wrong_polynomial_dimension(self, mtmfs_xds, name, dims):
        """Taylor terms and moments are not interchangeable, and the mask and
        the beam fit parameters keep the frequency axis."""
        xds = mtmfs_xds.copy()
        xds[name] = _image(dims, dict(xds.sizes), **xds[name].attrs)
        issues = check_image(xds)
        assert issues
        assert any(issue.path[0] == ("data_vars", name) for issue in issues)

    def test_moments_for_terms(self, mtmfs_xds):
        """2n - 1 moments for n terms."""
        xds = mtmfs_xds.isel(frequency_polynomial_moment=slice(0, 4))
        issues = check_image(xds)
        assert len(issues) == 1
        assert issues[0].path == [("coords", "frequency_polynomial_moment")]
        assert issues[0].found == 4 and issues[0].expected == [5]

    @pytest.mark.parametrize(
        "dim", ["frequency_polynomial", "frequency_polynomial_moment"]
    )
    def test_terms_numbered_from_zero(self, mtmfs_xds, dim):
        values = np.arange(1, mtmfs_xds.sizes[dim] + 1)
        xds = mtmfs_xds.assign_coords({dim: (dim, values, mtmfs_xds[dim].attrs)})
        issues = check_image(xds)
        assert [issue.path for issue in issues] == [[("coords", dim)]]
        assert "numbered 0, 1, ..., n - 1" in issues[0].message

    def test_one_channel_at_the_reference_frequency(self, mtmfs_xds):
        xds = mtmfs_xds.assign_coords(
            frequency=("frequency", [1.6e9], mtmfs_xds.frequency.attrs)
        )
        issues = check_image(xds)
        assert issues
        assert {tuple(issue.path) for issue in issues} == {(("coords", "frequency"),)}
        assert all("reference frequency" in message for message in _messages(issues))

    def test_polynomial_coordinate_attributes(self, mtmfs_xds):
        coordinate = mtmfs_xds.frequency_polynomial
        assert coordinate.attrs["polynomial_type"] == "taylor"
        assert not check_array(coordinate, FrequencyPolynomialCoordArray)
        assert not check_array(
            mtmfs_xds.frequency_polynomial_moment, FrequencyPolynomialMomentCoordArray
        )
        missing = coordinate.copy()
        missing.attrs.pop("reference_frequency")
        assert check_array(missing, FrequencyPolynomialCoordArray)
        other_kind = coordinate.copy()
        other_kind.attrs["polynomial_type"] = "legendre"
        assert check_array(other_kind, FrequencyPolynomialCoordArray)
        # the bandwidth is optional (one channel has none)
        assert "channel_width" not in coordinate.attrs
        coordinate.attrs["channel_width"] = {
            "data": 1.0e9,
            "dims": [],
            "attrs": {"type": "quantity", "units": "Hz"},
        }
        assert not check_array(coordinate, FrequencyPolynomialCoordArray)
        coordinate.attrs["channel_width"]["attrs"]["units"] = "s"
        assert check_array(coordinate, FrequencyPolynomialCoordArray)

    def test_coordinate_attribute_issue_is_reported(self, mtmfs_xds):
        xds = mtmfs_xds.copy()
        xds["frequency_polynomial"].attrs.pop("reference_frequency")
        issues = check_image(xds)
        assert any(
            issue.path[:2]
            == [("coords", "frequency_polynomial"), ("attrs", "reference_frequency")]
            for issue in issues
        )

    @pytest.mark.parametrize("name", list(SPECTRAL_PRODUCTS))
    def test_spectral_products_have_one_channel(self, name):
        xds = make_mtmfs_image_xds(frequencies=np.linspace(1.0e9, 2.0e9, 3))
        xds[name] = _image(CHANNELS, dict(xds.sizes), type=SPECTRAL_PRODUCTS[name])
        issues = check_image(xds)
        assert [issue.path for issue in issues] == [
            [("data_vars", name), ("dims", None)]
        ]
        assert issues[0].found == 3

    @pytest.mark.parametrize("name", list(SPECTRAL_PRODUCTS))
    def test_spectral_products_are_checked(self, mtmfs_xds, name):
        """ALPHA, ALPHA_ERROR and BETA are schema variables of their own
        (ALPHA_ERROR is not a version of ALPHA): wrong dims or types are
        reported."""
        xds = mtmfs_xds.copy()
        xds[name] = _image(TERMS, dict(xds.sizes), type=SPECTRAL_PRODUCTS[name])
        assert any(issue.path[0] == ("data_vars", name) for issue in check_image(xds))
        xds = mtmfs_xds.copy()
        xds[name].attrs["type"] = "sky"
        issues = check_image(xds)
        assert [issue.path for issue in issues] == [
            [("data_vars", name), ("attrs", "type")]
        ]

    def test_time_polynomial(self, mtmfs_xds):
        """time_polynomial in the place of time, beside or without time."""
        attrs = {
            "reference_time": {
                "data": 55000.5,
                "dims": [],
                "attrs": {
                    "type": "time",
                    "units": "d",
                    "scale": "utc",
                    "format": "mjd",
                },
            },
            "integration_time": {
                "data": 3600.0,
                "dims": [],
                "attrs": {"type": "quantity", "units": "s"},
            },
            "polynomial_type": "taylor",
        }
        xds = mtmfs_xds.assign_coords(
            time_polynomial=("time_polynomial", np.arange(2), attrs)
        )
        dims = ("time_polynomial",) + TERMS[1:]
        xds["SKY_MODEL"] = _image(dims, dict(xds.sizes), type="sky")
        assert not check_image(xds), str(check_image(xds))
        assert not check_array(xds.time_polynomial, TimePolynomialCoordArray)

        # a dataset with time_polynomial only
        alone = xr.Dataset(
            {"SKY": xds["SKY_MODEL"]},
            attrs={**xds.attrs, "data_groups": {"base": {"sky": "SKY"}}},
        )
        assert "time" not in alone.dims
        assert not check_image(alone), str(check_image(alone))

        alone["time_polynomial"].attrs.pop("reference_time")
        assert any(
            issue.path[:2]
            == [("coords", "time_polynomial"), ("attrs", "reference_time")]
            for issue in check_image(alone)
        )

    def test_missing_time_and_frequency_dimensions(self, mtmfs_xds):
        """A dataset has time or time_polynomial, and frequency or a
        frequency polynomial dimension."""
        xds = mtmfs_xds[["MASK"]].isel(time=0, drop=True)
        xds.attrs["data_groups"] = {"base": {"mask": "MASK"}}
        messages = _messages(check_image(xds))
        assert "Missing dimension 'time' (or 'time_polynomial')!" in messages
        xds = xr.Dataset(coords={"polarization": ["I"]}, attrs=mtmfs_xds.attrs)
        xds.attrs["data_groups"] = {"base": {}}
        messages = _messages(check_image(xds))
        assert "Missing dimension 'time' (or 'time_polynomial')!" in messages
        assert (
            "Missing dimension 'frequency' (or 'frequency_polynomial' or "
            "'frequency_polynomial_moment')!" in messages
        )

    @pytest.mark.parametrize("dim", ["frequency_polynomial", "frequency", "time"])
    def test_dimension_without_coordinate(self, mtmfs_xds, dim):
        """time and frequency are optional, but not their coordinates."""
        xds = mtmfs_xds.drop_vars([dim, "velocity"] if dim == "frequency" else dim)
        issues = check_image(xds)
        assert [issue.path for issue in issues].count([("coords", dim)]) == 1
        assert f"Required coordinate '{dim}' is missing!" in _messages(issues)


class TestMakeEmptyPolynomialImages:
    """The n_polynomial_terms and n_time_polynomial_terms keywords."""

    FACTORIES = [make_empty_sky_image, make_empty_aperture_image, make_empty_lmuv_image]

    @staticmethod
    def _make(factory, frequencies=(1.4e9,), times=(55000.5,), **kwargs):
        return factory(
            [0.2, -0.5],
            [8, 6],
            [1e-5, 1e-5],
            list(frequencies),
            ["I", "Q"],
            list(times),
            **kwargs,
        )

    @pytest.mark.parametrize("factory", FACTORIES)
    def test_frequency_polynomial(self, factory):
        xds = self._make(factory, n_polynomial_terms=3)
        assert xds.frequency_polynomial.values.tolist() == [0, 1, 2]
        assert xds.frequency_polynomial_moment.values.tolist() == [0, 1, 2, 3, 4]
        # the frequency coordinate is kept, at the reference frequency
        assert xds.frequency.values.tolist() == [1.4e9]
        for name in ("frequency_polynomial", "frequency_polynomial_moment"):
            attrs = xds[name].attrs
            assert attrs["polynomial_type"] == "taylor"
            assert (
                attrs["reference_frequency"]
                == xds.frequency.attrs["reference_frequency"]
            )
            # one channel: the bandwidth is not known
            assert "channel_width" not in attrs
        assert "time_polynomial" not in xds.coords
        assert not check_image(xds), str(check_image(xds))

    @pytest.mark.parametrize("factory", FACTORIES)
    def test_frequency_polynomial_of_channels(self, factory):
        xds = self._make(
            factory, frequencies=[1.0e9, 1.5e9, 2.0e9], n_polynomial_terms=2
        )
        attrs = xds.frequency_polynomial.attrs
        assert attrs["reference_frequency"]["data"] == 1.5e9
        assert attrs["channel_width"] == {
            "data": 1.5e9,
            "dims": [],
            "attrs": {"units": "Hz", "type": "quantity"},
        }
        assert xds.sizes["frequency"] == 3
        assert not check_image(xds), str(check_image(xds))

    @pytest.mark.parametrize("factory", FACTORIES)
    def test_time_polynomial(self, factory):
        xds = self._make(
            factory, times=[55000.0, 55000.25, 55000.5], n_time_polynomial_terms=2
        )
        assert xds.time_polynomial.values.tolist() == [0, 1]
        attrs = xds.time_polynomial.attrs
        assert attrs["polynomial_type"] == "taylor"
        assert attrs["reference_time"]["data"] == 55000.25
        assert attrs["reference_time"]["attrs"]["units"] == "d"
        assert attrs["integration_time"]["data"] == pytest.approx(3 * 0.25 * 86400.0)
        assert attrs["integration_time"]["attrs"]["units"] == "s"
        assert "frequency_polynomial" not in xds.coords
        assert xds.sizes["time"] == 3
        assert not check_image(xds), str(check_image(xds))
        single = self._make(factory, n_time_polynomial_terms=1)
        assert "integration_time" not in single.time_polynomial.attrs
        assert not check_image(single)

    @pytest.mark.parametrize("factory", FACTORIES)
    def test_default_has_no_polynomial_coordinates(self, factory):
        xds = self._make(factory)
        assert not {
            "time_polynomial",
            "frequency_polynomial",
            "frequency_polynomial_moment",
        } & set(xds.coords)

    @pytest.mark.parametrize("factory", FACTORIES)
    @pytest.mark.parametrize(
        "keyword", ["n_polynomial_terms", "n_time_polynomial_terms"]
    )
    @pytest.mark.parametrize(
        "value,error",
        [(0, ValueError), (-2, ValueError), (1.5, TypeError), (True, TypeError)],
    )
    def test_invalid_number_of_terms(self, factory, keyword, value, error):
        with pytest.raises(error, match=keyword):
            self._make(factory, **{keyword: value})


class TestPolynomialImageStorage:
    """Zarr holds polynomial terms."""

    def test_zarr_round_trip(self, mtmfs_xds, tmp_path):
        xds = mtmfs_xds.copy()
        xds["SKY"] = xds["SKY"].copy(
            data=np.random.default_rng(1).random(xds["SKY"].shape, dtype=np.float32)
        )
        (path,) = write_image(xds, str(tmp_path / "mtmfs"), out_format="zarr")
        for reader in (open_image, load_image):
            back = reader(path).load()
            xr.testing.assert_identical(back, xds)
            assert back.attrs["data_groups"] == xds.attrs["data_groups"]
            assert not check_image(back), str(check_image(back))
