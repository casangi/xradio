"""``open_image`` and ``load_image`` need python-casacore or casatools only
for CASA images: a FITS image is read with astropy, and ``write_image``
writes FITS images with astropy only."""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

from xradio.image import load_image, open_image, write_image
from xradio.testing.image.io import download_image

_CASACORE_PACKAGES = ("casacore", "casatools", "casaconfig")
_XRADIO_CASACORE_MODULES = (
    "xradio.image._util.casacore",
    "xradio.image._util._casacore",
    "xradio._utils._casacore",
)


@pytest.fixture(scope="module")
def fits_image(tmp_path_factory):
    """test_image.fits, downloaded once for the module."""
    return download_image("test_image.fits", tmp_path_factory.mktemp("fits"))


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


def test_fits_image_opens_without_casacore(fits_image, without_casacore):
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


def test_casa_write_without_casacore_names_the_missing_packages(
    fits_image, tmp_path, without_casacore
):
    xds = open_image(str(fits_image))
    with pytest.raises(ModuleNotFoundError, match="python-casacore or casatools"):
        write_image(xds, str(tmp_path / "out.im"), out_format="casa")
    assert list(tmp_path.iterdir()) == []


# Runs in a fresh interpreter: the casacore packages are blocked before
# xradio (or anything else) is imported, as on an install without them.
_FITS_WITHOUT_CASACORE = textwrap.dedent(
    """
    import sys

    for name in ("casacore", "casatools", "casaconfig"):
        sys.modules[name] = None

    import numpy as np

    import xradio.image
    from xradio.image import open_image, write_image

    source, written = sys.argv[1], sys.argv[2]
    xds = open_image(source)
    write_image(xds, written, out_format="fits")
    reopened = open_image(written)
    assert list(reopened.polarization.values) == list(xds.polarization.values)
    np.testing.assert_array_equal(reopened["SKY"].values, xds["SKY"].values)
    loaded = sorted(
        name
        for name, module in sys.modules.items()
        if module is not None
        and name.split(".")[0] in ("casacore", "casatools", "casaconfig")
    )
    assert not loaded, f"casacore packages were imported: {loaded}"
    print("FITS without casacore: OK")
    """
)


def test_fits_open_and_write_in_a_process_without_casacore(fits_image, tmp_path):
    written = tmp_path / "written.fits"
    result = subprocess.run(
        [sys.executable, "-c", _FITS_WITHOUT_CASACORE, str(fits_image), str(written)],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        timeout=600,
    )
    assert result.returncode == 0, (
        f"stdout:\n{result.stdout[-3000:]}\nstderr:\n{result.stderr[-5000:]}"
    )
    assert "FITS without casacore: OK" in result.stdout
    assert written.is_file()
