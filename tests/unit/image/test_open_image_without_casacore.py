"""``open_image`` and ``load_image`` need python-casacore or casatools only
for CASA images: a FITS image is read with astropy."""

from __future__ import annotations

import sys

import pytest

from xradio.image import load_image, open_image
from xradio.testing.image.io import download_image

_CASACORE_PACKAGES = ("casacore", "casatools", "casaconfig")
_XRADIO_CASACORE_MODULES = (
    "xradio.image._util.casacore",
    "xradio.image._util._casacore",
    "xradio._utils._casacore",
)


@pytest.fixture
def without_casacore(monkeypatch):
    """python-casacore and casatools cannot be imported during the test, and
    xradio's modules built on them are imported afresh."""
    for name in list(sys.modules):
        if name.split(".")[0] in _CASACORE_PACKAGES or name.startswith(
            _XRADIO_CASACORE_MODULES
        ):
            monkeypatch.delitem(sys.modules, name)
    for name in _CASACORE_PACKAGES:
        monkeypatch.setitem(sys.modules, name, None)


def test_fits_image_opens_without_casacore(tmp_path, without_casacore):
    fits_image = download_image("test_image.fits", tmp_path)
    xds = open_image(str(fits_image))
    assert "SKY" in xds.data_vars
    assert xds.sizes["frequency"] > 0


def test_casa_image_without_casacore_names_the_missing_packages(
    tmp_path, without_casacore
):
    casa_image = tmp_path / "image.im"
    casa_image.mkdir()
    (casa_image / "table.info").write_text("")
    with pytest.raises(ModuleNotFoundError, match="python-casacore or casatools"):
        open_image(str(casa_image))
    with pytest.raises(ModuleNotFoundError, match="python-casacore or casatools"):
        load_image(str(casa_image))
