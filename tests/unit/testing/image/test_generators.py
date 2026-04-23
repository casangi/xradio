"""Unit tests for xradio.testing.image.generators."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from xradio.image import make_empty_sky_image
from xradio.testing.image.generators import (
    create_bzero_bscale_fits,
    create_empty_test_image,
    make_beam_fit_params,
    scale_data_for_int16,
)

# --------------------------------------------------------------------------- #
# Shared helper                                                                #
# --------------------------------------------------------------------------- #


def _make_minimal_xds(time=1, frequency=5, polarization=3, l=20, m=20):
    """Synthetic xr.Dataset with standard image dimensions."""
    return xr.Dataset(
        {
            "SKY": (
                ["time", "frequency", "polarization", "l", "m"],
                np.ones((time, frequency, polarization, l, m), dtype=np.float32),
            )
        },
        coords={
            "time": np.array([54000.1]),
            "frequency": np.linspace(1.4e9, 1.41e9, frequency),
            "polarization": ["I", "Q", "U"][:polarization],
            "l": np.arange(l),
            "m": np.arange(m),
        },
    )


# --------------------------------------------------------------------------- #
# TestScaleDataForInt16                                                        #
# --------------------------------------------------------------------------- #


class TestScaleDataForInt16:
    """Tests for ``scale_data_for_int16``."""

    def test_dtype_is_int16(self):
        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = scale_data_for_int16(data)
        assert result.dtype == np.int16

    def test_nan_replaced_by_zero(self):
        data = np.array([np.nan, 1.0])
        result = scale_data_for_int16(data)
        assert result[0] == 0

    def test_clips_above_max(self):
        data = np.array([50000.0])
        result = scale_data_for_int16(data)
        assert result[0] == 32767

    def test_clips_below_min(self):
        data = np.array([-50000.0])
        result = scale_data_for_int16(data)
        assert result[0] == -32768

    def test_normal_values_unchanged(self):
        data = np.array([100.0, -200.0, 0.0])
        result = scale_data_for_int16(data)
        np.testing.assert_array_equal(result, np.array([100, -200, 0], dtype=np.int16))


# --------------------------------------------------------------------------- #
# TestMakeBeamFitParams                                                        #
# --------------------------------------------------------------------------- #


class TestMakeBeamFitParams:
    """Tests for ``make_beam_fit_params``."""

    @pytest.fixture
    def xds(self):
        return _make_minimal_xds(time=1, frequency=5, polarization=3)

    def test_shape(self, xds):
        result = make_beam_fit_params(xds)
        assert result.shape == (1, 5, 3, 3)

    def test_dims(self, xds):
        result = make_beam_fit_params(xds)
        assert list(result.dims) == [
            "time",
            "frequency",
            "polarization",
            "beam_params_label",
        ]

    def test_beam_params_label_coord(self, xds):
        result = make_beam_fit_params(xds)
        np.testing.assert_array_equal(
            result.coords["beam_params_label"].values, ["major", "minor", "pa"]
        )

    def test_coords_inherited_from_xds(self, xds):
        result = make_beam_fit_params(xds)
        np.testing.assert_array_equal(result.coords["time"].values, xds.time.values)
        np.testing.assert_array_equal(
            result.coords["frequency"].values, xds.frequency.values
        )
        np.testing.assert_array_equal(
            result.coords["polarization"].values, xds.polarization.values
        )

    def test_default_fill_value(self, xds):
        result = make_beam_fit_params(xds)
        values = result.values.copy()
        values[0, 0, 0, :] = 1.0  # neutralise the distinct entry
        assert np.all(values == 1.0)

    def test_distinct_entry(self, xds):
        result = make_beam_fit_params(xds)
        np.testing.assert_array_equal(result.values[0, 0, 0, :], 2.0)


# --------------------------------------------------------------------------- #
# TestCreateEmptyTestImage                                                     #
# --------------------------------------------------------------------------- #


class TestCreateEmptyTestImage:
    """Tests for ``create_empty_test_image``."""

    def test_returns_dataset(self):
        result = create_empty_test_image(make_empty_sky_image)
        assert isinstance(result, xr.Dataset)

    def test_expected_dims(self):
        result = create_empty_test_image(make_empty_sky_image)
        for dim in ("time", "frequency", "polarization", "l", "m"):
            assert dim in result.dims, f"Expected dim '{dim}' not found"

    def test_do_sky_coords_false(self):
        result = create_empty_test_image(make_empty_sky_image, do_sky_coords=False)
        assert "right_ascension" not in result.coords
        assert "declination" not in result.coords

    def test_do_sky_coords_true(self):
        result = create_empty_test_image(make_empty_sky_image, do_sky_coords=True)
        assert "right_ascension" in result.coords
        assert "declination" in result.coords

    def test_none_does_not_pass_kwarg(self):
        result = create_empty_test_image(make_empty_sky_image, do_sky_coords=None)
        assert isinstance(result, xr.Dataset)


# --------------------------------------------------------------------------- #
# TestCreateBzeroBscaleFits                                                    #
# --------------------------------------------------------------------------- #


class TestCreateBzeroBscaleFits:
    """Tests for ``create_bzero_bscale_fits``."""

    @pytest.fixture
    def source_fits(self, tmp_path):
        """Write a minimal valid FITS file to use as source."""
        from astropy.io import fits

        data = np.zeros((4, 4), dtype=np.float32)
        hdu = fits.PrimaryHDU(data=data)
        path = tmp_path / "source.fits"
        hdu.writeto(str(path))
        return path

    def test_file_created(self, tmp_path, source_fits):
        out = str(tmp_path / "out.fits")
        create_bzero_bscale_fits(out, str(source_fits), bzero=5.0, bscale=2.0)
        assert (tmp_path / "out.fits").exists()

    def test_bscale_header(self, tmp_path, source_fits):
        from astropy.io import fits

        out = str(tmp_path / "out.fits")
        create_bzero_bscale_fits(out, str(source_fits), bzero=0.0, bscale=3.0)
        with fits.open(out) as hdul:
            assert hdul[0].header["BSCALE"] == pytest.approx(3.0)

    def test_bzero_header(self, tmp_path, source_fits):
        from astropy.io import fits

        out = str(tmp_path / "out.fits")
        create_bzero_bscale_fits(out, str(source_fits), bzero=7.5, bscale=1.0)
        with fits.open(out) as hdul:
            assert hdul[0].header["BZERO"] == pytest.approx(7.5)

    def test_pixel_dtype_is_int16(self, tmp_path, source_fits):
        from astropy.io import fits

        out = str(tmp_path / "out.fits")
        create_bzero_bscale_fits(out, str(source_fits), bzero=0.0, bscale=1.0)
        with fits.open(out, do_not_scale_image_data=True) as hdul:
            dtype = hdul[0].data.dtype
            # FITS uses big-endian storage ('>i2'), so compare kind and itemsize
            # rather than the exact dtype object to avoid byte-order mismatch.
            assert dtype.kind == "i", f"Expected signed integer, got {dtype}"
            assert dtype.itemsize == 2, f"Expected 2-byte (16-bit) integer, got {dtype}"
