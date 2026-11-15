import pytest
import xarray

from xradio.schema.polarization import check_polarization_order


def _dataset(labels):
    return xarray.Dataset(coords={"polarization": labels})


def test_check_polarization_order_without_polarization():
    assert not check_polarization_order(xarray.Dataset())


def test_check_polarization_order_scalar_polarization():
    """A scalar coordinate (one polarization selected) is left to the
    dimension checks"""
    assert not check_polarization_order(_dataset("I"))
    assert not check_polarization_order(_dataset(["RR", "LL"]).isel(polarization=1))


@pytest.mark.parametrize(
    "labels",
    [
        ["I"],
        ["I", "Q", "U", "V"],
        ["I", "V"],
        ["RR", "RL", "LR", "LL"],
        ["RR", "LL"],
        ["XX", "XY", "YX", "YY"],
        ["XX", "YY", "NOT_A_STOKES_LABEL"],
    ],
)
def test_check_polarization_order_canonical(labels):
    assert not check_polarization_order(_dataset(labels))


@pytest.mark.parametrize(
    "labels, canonical",
    [
        (["RR", "LL", "RL", "LR"], ["RR", "RL", "LR", "LL"]),
        (["YY", "XX"], ["XX", "YY"]),
        (["V", "I"], ["I", "V"]),
        (["XX", "NOT_A_STOKES_LABEL", "YY"], ["XX", "YY", "NOT_A_STOKES_LABEL"]),
    ],
)
def test_check_polarization_order_other(labels, canonical):
    (issue,) = check_polarization_order(_dataset(labels))
    assert issue.path == [("coords", "polarization")]
    assert "canonical" in issue.message
    assert issue.found == labels
    assert issue.expected == [canonical]
