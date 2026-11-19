"""Unit tests for the primary beam schema (``xradio.primary_beam.schema``)."""

import numpy as np
import pytest
import xarray as xr

from xradio.primary_beam import (
    ZPC_SCHEMA_VERSION,
    ZernikePolynomialCoefficientsXds,
    check_zpc,
)
from xradio.schema.check import check_dataset


def make_zpc(polarization=("XX", "XY", "YX", "YY"), coefficient=None):
    coefficient = np.arange(66) if coefficient is None else coefficient
    frequency = np.array([109.0e9, 110.0e9])
    shape = (frequency.size, len(polarization), len(coefficient))
    return xr.Dataset(
        {
            "ZPC": (
                ("frequency", "polarization", "coefficient"),
                np.ones(shape, dtype=complex),
            ),
            "ETA": (("frequency", "polarization", "coefficient"), np.ones(shape)),
        },
        coords={
            "frequency": (
                "frequency",
                frequency,
                {"type": "spectral_coord", "units": "Hz", "observer": "TOPO"},
            ),
            "polarization": list(polarization),
            "coefficient": np.asarray(coefficient),
        },
        attrs={
            "type": "zernike_polynomial_coefficients",
            "schema_version": ZPC_SCHEMA_VERSION,
            "telescope_name": "ALMA",
            "antenna_type": "DA",
            "dish_diameter": 12.0,
            "blockage_diameter": 0.75,
            "max_rad_1GHz": 0.15,
            "zernike_basis": "zernike_osa",
            "aperture_frame": "antenna",
            "origin": {"software": "astrohack"},
        },
    )


class TestZpcSchema:
    def test_valid(self):
        assert not check_zpc(make_zpc())

    def test_parallel_hands_only(self):
        assert not check_zpc(make_zpc(polarization=("RR", "LL")))

    def test_legacy_coordinate_names(self):
        zpc = make_zpc().rename(
            {"frequency": "chan", "polarization": "pol", "coefficient": "coef_indx"}
        )
        assert check_zpc(zpc)

    @pytest.mark.parametrize(
        "attr",
        [
            "telescope_name",
            "dish_diameter",
            "max_rad_1GHz",
            "aperture_frame",
            "schema_version",
        ],
    )
    def test_missing_attribute(self, attr):
        zpc = make_zpc()
        del zpc.attrs[attr]
        assert check_zpc(zpc)

    def test_polarization_order(self):
        issues = check_zpc(make_zpc().isel(polarization=[3, 1, 2, 0]))
        assert any("canonical" in issue.message for issue in issues)

    def test_sparse_coefficients(self):
        assert not check_zpc(make_zpc(coefficient=[0, 4, 11, 12, 24]))

    @pytest.mark.parametrize("coefficient", [[0, 4, 4], [0, 12, 4], [-1, 0, 4]])
    def test_coefficient_indices(self, coefficient):
        assert any(
            "increase" in issue.message
            for issue in check_zpc(make_zpc(coefficient=coefficient))
        )

    def test_index_beyond_basis(self):
        assert any(
            "0 to 65" in issue.message
            for issue in check_zpc(make_zpc(coefficient=[0, 4, 66]))
        )

    def test_aperture_frame(self):
        zpc = make_zpc()
        zpc.attrs["aperture_frame"] = "sky"
        assert check_zpc(zpc)

    def test_eta_positive(self):
        zpc = make_zpc()
        zpc["ETA"] = zpc.ETA * 0
        assert any("ETA" in issue.message for issue in check_zpc(zpc))

    def test_wrong_frequency_units(self):
        zpc = make_zpc()
        zpc.frequency.attrs["units"] = "MHz"
        assert check_dataset(zpc, ZernikePolynomialCoefficientsXds)
