"""The arcae image shim against python-casacore, the reference: images
created by both store the same keywords and masks and read back the same,
and tofits writes the same observation keywords."""

import numpy as np
import pytest

pytest.importorskip("arcae")
reason = "python-casacore is the reference"
casacore_images = pytest.importorskip("casacore.images", reason=reason)
casacore_tables = pytest.importorskip("casacore.tables", reason=reason)

from astropy.io import fits  # noqa: E402

from xradio._utils._casacore import images_from_arcae  # noqa: E402


def _assert_records_equal(expected, actual, path=""):
    """Compare table keyword records. casacore pads the FREQ ctype with NULs
    and rounds the HI rest frequency differently in the last bit."""
    if isinstance(expected, dict):
        assert sorted(actual) == sorted(expected), path
        for key in expected:
            _assert_records_equal(expected[key], actual[key], f"{path}/{key}")
    elif isinstance(expected, str):
        assert actual == expected.rstrip("\x00"), path
    elif np.asarray(expected).dtype.kind in "fc":
        np.testing.assert_allclose(actual, expected, rtol=1e-15, err_msg=path)
    else:
        np.testing.assert_array_equal(actual, expected, err_msg=path)


def _stored(path):
    """Keywords (with the image path replaced) and mask pixels of an image."""
    table = casacore_tables.table(str(path), ack=False)
    keywords = table.getkeywords()
    table.close()
    keywords["logtable"] = keywords["logtable"].replace(str(path), "<image>")
    masks = {}
    for name, mask in keywords.get("masks", {}).items():
        mask["mask"] = mask["mask"].replace(str(path), "<image>")
        mask_table = casacore_tables.table(str(path / name), ack=False)
        masks[name] = mask_table.getcol("PagedArray")
        mask_table.close()
    return keywords, masks


def _read(module, path):
    image = module.image(str(path))
    coords = image.info()["coordinates"]
    return {
        "datatype": image.datatype(),
        "shape": list(image.shape()),
        "unit": image.unit(),
        "imageinfo": image.imageinfo(),
        "miscinfo": image.miscinfo(),
        "obsinfo": {
            key: coords[key]
            for key in ("telescope", "observer", "obsdate", "pointingcenter")
        },
        "mask": image.getmask(),
        "data": image.getdata(),
    }


@pytest.mark.parametrize(
    "shape, values",
    [
        ([7], None),
        ([6, 8], None),
        ([2, 6, 8], None),
        ([5, 6, 8], None),
        ([3, 2, 6, 8], None),
        ([3, 5, 6, 8], None),
        ([3, 2, 6, 8], np.float64(0)),
        ([3, 2, 6, 8], np.complex64(0)),
        ([3, 2, 6, 8], np.complex128(0)),
    ],
)
@pytest.mark.parametrize("maskname", ["", "mask0"])
def test_created_image_is_stored_and_read_like_casacore(
    tmp_path, shape, values, maskname
):
    expected_path, actual_path = tmp_path / "casacore.im", tmp_path / "arcae.im"
    kwargs = {"shape": shape, "maskname": maskname}
    if values is not None:
        kwargs["values"] = values
    image = casacore_images.image(str(expected_path), **kwargs)
    del image
    images_from_arcae.image(str(actual_path), **kwargs).close()

    expected_keywords, expected_masks = _stored(expected_path)
    actual_keywords, actual_masks = _stored(actual_path)
    _assert_records_equal(expected_keywords, actual_keywords)
    _assert_records_equal(expected_masks, actual_masks)

    expected = _read(casacore_images, expected_path)
    _assert_records_equal(expected, _read(images_from_arcae, expected_path))
    _assert_records_equal(expected, _read(images_from_arcae, actual_path))


@pytest.mark.parametrize(
    "removed",
    [(), ("telescope", "observer", "obsdate", "pointingcenter"), ("obsdate",)],
)
def test_tofits_observation_keywords_are_casacore_s(tmp_path, removed):
    path = tmp_path / "image.im"
    image = casacore_images.image(str(path), shape=[3, 2, 6, 8])
    del image
    table = casacore_tables.table(str(path), readonly=False, ack=False)
    coords = table.getkeyword("coords")
    for key in removed:
        del coords[key]
    table.putkeyword("coords", coords)
    table.close()

    casacore_images.image(str(path)).tofits(str(tmp_path / "casacore.fits"))
    images_from_arcae.image(str(path)).tofits(str(tmp_path / "arcae.fits"))

    expected = fits.getheader(tmp_path / "casacore.fits")
    actual = fits.getheader(tmp_path / "arcae.fits")
    for key in ("TELESCOP", "OBSERVER", "DATE-OBS", "TIMESYS"):
        assert actual.get(key) == expected.get(key), key
