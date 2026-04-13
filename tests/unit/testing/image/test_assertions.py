"""Unit tests for xradio.testing.image.assertions."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from xradio.testing.image.assertions import (
    assert_image_block_equal,
    normalize_image_coords_for_compare,
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


def _make_coords_dict(direction_key="direction0", spectral_key="spectral2"):
    """Build a minimal CASA-style coordinate dict for normalize tests."""
    return {
        direction_key: {
            "cdelt": np.array([1.0, 1.0]),
            "crval": np.array([0.5, -0.5]),
            "units": ["rad", "rad"],
        },
        spectral_key: {
            "velUnit": "m/s",
        },
    }


# --------------------------------------------------------------------------- #
# TestNormalizeImageCoordsForCompare                                           #
# --------------------------------------------------------------------------- #


class TestNormalizeImageCoordsForCompare:
    """Tests for ``normalize_image_coords_for_compare``."""

    def test_default_direction_key(self):
        coords = _make_coords_dict()
        original_cdelt = coords["direction0"]["cdelt"].copy()
        original_crval = coords["direction0"]["crval"].copy()
        factor = 180 * 60 / np.pi
        normalize_image_coords_for_compare(coords)
        np.testing.assert_allclose(
            coords["direction0"]["cdelt"], original_cdelt * factor
        )
        np.testing.assert_allclose(
            coords["direction0"]["crval"], original_crval * factor
        )

    def test_default_spectral_key(self):
        coords = _make_coords_dict()
        normalize_image_coords_for_compare(coords)
        assert coords["spectral2"]["velUnit"] == "km/s"

    def test_default_direction_units(self):
        coords = _make_coords_dict()
        normalize_image_coords_for_compare(coords)
        assert coords["direction0"]["units"] == ["'", "'"]

    def test_custom_direction_key(self):
        coords = _make_coords_dict(direction_key="direction1")
        original_cdelt = coords["direction1"]["cdelt"].copy()
        factor = 180 * 60 / np.pi
        normalize_image_coords_for_compare(coords, direction_key="direction1")
        np.testing.assert_allclose(
            coords["direction1"]["cdelt"], original_cdelt * factor
        )

    def test_custom_spectral_key(self):
        coords = _make_coords_dict(spectral_key="spectral0")
        normalize_image_coords_for_compare(coords, spectral_key="spectral0")
        assert coords["spectral0"]["velUnit"] == "km/s"

    def test_custom_factor(self):
        coords = _make_coords_dict()
        original_cdelt = coords["direction0"]["cdelt"].copy()
        original_crval = coords["direction0"]["crval"].copy()
        normalize_image_coords_for_compare(coords, factor=1.0)
        np.testing.assert_allclose(coords["direction0"]["cdelt"], original_cdelt)
        np.testing.assert_allclose(coords["direction0"]["crval"], original_crval)

    def test_custom_direction_units(self):
        coords = _make_coords_dict()
        normalize_image_coords_for_compare(coords, direction_units=["deg", "deg"])
        assert coords["direction0"]["units"] == ["deg", "deg"]

    def test_custom_vel_unit(self):
        coords = _make_coords_dict()
        normalize_image_coords_for_compare(coords, vel_unit="m/s")
        assert coords["spectral2"]["velUnit"] == "m/s"

    def test_modifies_in_place(self):
        coords = _make_coords_dict()
        result = normalize_image_coords_for_compare(coords)
        assert result is None


# --------------------------------------------------------------------------- #
# TestAssertImageBlockEqualGuard                                               #
# --------------------------------------------------------------------------- #


class TestAssertImageBlockEqualGuard:
    """Tests for the ``ValueError`` guard in ``assert_image_block_equal``."""

    @pytest.fixture
    def small_xds(self):
        return _make_minimal_xds(time=1, frequency=2, polarization=1, l=5, m=5)

    def test_raises_when_l_overruns(self, small_xds):
        with pytest.raises(ValueError):
            assert_image_block_equal(
                small_xds,
                "unused_path",
                selection={"l": slice(0, 10)},
            )

    def test_raises_when_m_overruns(self, small_xds):
        with pytest.raises(ValueError):
            assert_image_block_equal(
                small_xds,
                "unused_path",
                selection={"m": slice(0, 10)},
            )

    def test_raises_when_frequency_overruns(self, small_xds):
        with pytest.raises(ValueError):
            assert_image_block_equal(
                small_xds,
                "unused_path",
                selection={"frequency": slice(0, 5)},
            )

    def test_error_message_names_dim(self, small_xds):
        with pytest.raises(ValueError, match="l"):
            assert_image_block_equal(
                small_xds,
                "unused_path",
                selection={"l": slice(0, 10)},
            )

    def test_valid_selection_passes_guard(self, tmp_path, monkeypatch, small_xds):
        # write_image, load_image, and assert_xarray_datasets_equal are imported
        # lazily inside assert_image_block_equal, so they must be patched at the
        # modules that define/export them, not on xradio.testing.image.assertions.
        monkeypatch.setattr("xradio.image.write_image", lambda *a, **kw: None)
        monkeypatch.setattr(
            "xradio.image.load_image",
            lambda path, selection, do_sky_coords=True: small_xds.isel(
                **{k: v for k, v in selection.items() if k in small_xds.dims}
            ),
        )
        monkeypatch.setattr(
            "xradio.testing.assert_xarray_datasets_equal", lambda a, b: None
        )
        assert_image_block_equal(
            small_xds,
            str(tmp_path / "out"),
            selection={"l": slice(0, 3), "m": slice(0, 3)},
        )
