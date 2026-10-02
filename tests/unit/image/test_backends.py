"""Tests of the ``xradio_casa_image`` and ``xradio_fits_image`` xarray engines.

The engines are passed to ``xr.open_dataset`` as classes, so the tests do not
depend on the entry points of the installed distribution (an editable install
picks up pyproject.toml changes only when it is reinstalled).
"""

from __future__ import annotations

import os
import pathlib
import shutil
import subprocess
import sys
import textwrap
import tomllib
import warnings

import dask
import numpy as np
import pytest
import xarray as xr
from astropy.io import fits

from _xradio_xarray_backends import (
    CasaImageBackendEntrypoint,
    FitsImageBackendEntrypoint,
)
from xradio.image import check_image, open_image
from xradio.testing.image.io import download_image

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]

_CASA = CasaImageBackendEntrypoint
_FITS = FitsImageBackendEntrypoint

_READER_MODULES = {
    "casa": "xradio.image._util._casacore.xds_from_casacore",
    "fits": "xradio.image._util._fits.xds_from_fits",
}


@pytest.fixture(scope="module")
def image_dir(tmp_path_factory):
    """A module-wide directory with the CASA and FITS test images."""
    directory = tmp_path_factory.mktemp("backend_images")
    download_image("casa_test_image.im", directory)
    download_image("test_image.fits", directory)
    return directory


@pytest.fixture
def casa_image(image_dir, tmp_path):
    """A private copy of the CASA test image (10 channels, 4 polarizations)."""
    path = tmp_path / "casa_test_image.im"
    shutil.copytree(image_dir / "casa_test_image.im", path)
    return path


@pytest.fixture
def fits_image(image_dir, tmp_path):
    """A private copy of the FITS test image (10 channels, 4 polarizations)."""
    path = tmp_path / "test_image.fits"
    shutil.copy(image_dir / "test_image.fits", path)
    return path


@pytest.fixture
def synchronous_dask():
    with dask.config.set(scheduler="synchronous"):
        yield


@pytest.fixture
def plane_reads(monkeypatch):
    """Count the image blocks the CASA and FITS readers read from disk."""
    import importlib

    counts = {"casa": 0, "fits": 0}
    for kind, module_name in _READER_MODULES.items():
        module = importlib.import_module(module_name)
        original = module._read_image_chunk

        def counting(*args, _kind=kind, _original=original, **kwargs):
            counts[_kind] += 1
            return _original(*args, **kwargs)

        monkeypatch.setattr(module, "_read_image_chunk", counting)
    return counts


def _engine_and_kind(image_format):
    return (_CASA, "casa") if image_format == "casa" else (_FITS, "fits")


@pytest.fixture(params=["casa", "fits"])
def image(request, casa_image, fits_image):
    """(path, engine, reader kind) for both image formats."""
    path = casa_image if request.param == "casa" else fits_image
    engine, kind = _engine_and_kind(request.param)
    return path, engine, kind


# ---------------------------------------------------------------------------
# X1: lazy reads
# ---------------------------------------------------------------------------


class TestLazyReads:
    def test_open_reads_no_pixels(self, image, plane_reads, synchronous_dask):
        path, engine, kind = image
        ds = xr.open_dataset(path, engine=engine)
        assert plane_reads[kind] == 0
        assert ds.SKY.chunks is None  # lazily indexed, not dask backed

    def test_load_reads_the_image(self, image, plane_reads, synchronous_dask):
        path, engine, kind = image
        expected = open_image(str(path)).compute()
        plane_reads[kind] = 0
        ds = xr.open_dataset(path, engine=engine)
        loaded = ds.load()
        assert loaded is ds
        # 10 channels x 4 polarizations, for SKY and again for FLAG_SKY
        assert plane_reads[kind] == 2 * 40
        # The data no longer depends on the source
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)
        for name in ("SKY", "FLAG_SKY"):
            assert isinstance(ds[name].variable._data, np.ndarray)
            np.testing.assert_array_equal(ds[name].values, expected[name].values)

    def test_indexing_reads_only_the_planes_needed(
        self, image, plane_reads, synchronous_dask
    ):
        path, engine, kind = image
        ds = xr.open_dataset(path, engine=engine)
        plane = ds.SKY.isel(frequency=3, polarization=1).values
        assert plane.shape == (1, 30, 20)
        assert plane_reads[kind] == 1
        expected = open_image(str(path)).SKY.isel(frequency=3, polarization=1)
        np.testing.assert_array_equal(plane, expected.values)

    def test_dask_chunks_read_only_their_planes(
        self, image, plane_reads, synchronous_dask
    ):
        path, engine, kind = image
        ds = xr.open_dataset(path, engine=engine, chunks={"frequency": 2})
        assert plane_reads[kind] == 0
        assert ds.SKY.chunks[1] == (2, 2, 2, 2, 2)
        values = ds.SKY.isel(frequency=slice(2, 4)).values
        assert values.shape == (1, 2, 4, 30, 20)
        # One dask chunk along frequency: 2 channels x 4 polarizations
        assert plane_reads[kind] == 8
        expected = open_image(str(path)).SKY.isel(frequency=slice(2, 4))
        np.testing.assert_array_equal(values, expected.values)

    def test_preferred_chunks_are_planes(self, image):
        path, engine, _ = image
        ds = xr.open_dataset(path, engine=engine)
        plane = {"time": 1, "frequency": 1, "polarization": 1, "l": 30, "m": 20}
        assert ds.SKY.encoding["preferred_chunks"] == plane
        assert ds.FLAG_SKY.encoding["preferred_chunks"] == plane
        # chunks={} follows the preferred chunks: one dask chunk per plane
        chunked = xr.open_dataset(path, engine=engine, chunks={})
        assert chunked.SKY.data.numblocks == (1, 10, 4, 1, 1)

    def test_image_chunks_set_the_read_blocks(
        self, image, plane_reads, synchronous_dask
    ):
        path, engine, kind = image
        ds = xr.open_dataset(path, engine=engine, image_chunks={"frequency": 5})
        assert ds.SKY.encoding["preferred_chunks"]["frequency"] == 5
        assert ds.SKY.encoding["preferred_chunks"]["polarization"] == 1
        plane = ds.SKY.isel(frequency=0, polarization=0).values
        assert plane.shape == (1, 30, 20)
        # One read block: 5 channels of one polarization
        assert plane_reads[kind] == 1

    @pytest.mark.parametrize(
        "selection",
        [
            {"frequency": [0, 5, 9], "polarization": slice(None, None, -1)},
            {"l": slice(3, 20, 4), "m": [1, 1, 7]},
            {"frequency": -1, "polarization": [3, 0], "l": 0},
            {"frequency": slice(8, 2, -3), "m": slice(5, 5)},
        ],
        ids=["fancy_reversed", "strided_repeated", "negative_int", "empty"],
    )
    def test_values_match_open_image(self, image, selection, synchronous_dask):
        path, engine, _ = image
        expected = open_image(str(path))
        ds = xr.open_dataset(path, engine=engine)
        for name in ("SKY", "FLAG_SKY"):
            np.testing.assert_array_equal(
                ds[name].isel(selection).values, expected[name].isel(selection).values
            )

    def test_coordinates_and_attrs_match_open_image(self, image):
        path, engine, _ = image
        expected = open_image(str(path))
        ds = xr.open_dataset(path, engine=engine)
        assert set(ds.data_vars) == set(expected.data_vars)
        for name, coord in expected.coords.items():
            np.testing.assert_array_equal(ds[name].values, coord.values)
        assert ds.attrs["data_groups"] == expected.attrs["data_groups"]
        assert ds.attrs.keys() == expected.attrs.keys()
        assert ds.SKY.attrs.keys() == expected.SKY.attrs.keys()

    def test_beam_fit_params_are_loaded(self, fits_image, synchronous_dask):
        """Beam fit parameters (image metadata) are in memory, so chunking
        along frequency does not warn about splitting their read block."""
        for card, value in (("BMAJ", 1e-3), ("BMIN", 5e-4), ("BPA", 30.0)):
            fits.setval(fits_image, card, value=value)
        expected = open_image(str(fits_image)).BEAM_FIT_PARAMS_SKY.values
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ds = xr.open_dataset(fits_image, engine=_FITS, chunks={"frequency": 3})
        assert not [w for w in caught if "separate the stored chunks" in str(w.message)]
        assert "preferred_chunks" not in ds.BEAM_FIT_PARAMS_SKY.encoding
        np.testing.assert_array_equal(ds.BEAM_FIT_PARAMS_SKY.values, expected)
        assert ds.SKY.attrs["beam_fit_params"] == "BEAM_FIT_PARAMS_SKY"

    def test_chunked_values_under_threads(self, image):
        path, engine, _ = image
        expected = open_image(str(path)).SKY.values
        ds = xr.open_dataset(path, engine=engine, chunks={"frequency": 3})
        with dask.config.set(scheduler="threads"):
            np.testing.assert_array_equal(ds.SKY.values, expected)


# ---------------------------------------------------------------------------
# X3: drop_variables
# ---------------------------------------------------------------------------


class TestDropVariables:
    @pytest.mark.parametrize("drop", ["FLAG_SKY", ["FLAG_SKY"], ("FLAG_SKY", "nope")])
    def test_drop_flag_prunes_data_groups(self, image, drop):
        path, engine, _ = image
        ds = xr.open_dataset(path, engine=engine, drop_variables=drop)
        assert "FLAG_SKY" not in ds.data_vars
        assert "SKY" in ds.data_vars
        assert ds.attrs["data_groups"] == {"base": {"sky": "SKY"}}
        assert ds.SKY.attrs.get("flag") != "FLAG_SKY"
        assert not [i for i in check_image(ds) if "FLAG_SKY" in str(i)]

    @pytest.mark.parametrize("name", ["velocity", "declination", "time"])
    def test_string_is_one_name(self, image, name):
        """A str is not split into characters (which named l and m)."""
        path, engine, _ = image
        ds = xr.open_dataset(path, engine=engine, drop_variables=name)
        assert name not in ds.variables
        for kept in ("l", "m", "SKY", "FLAG_SKY"):
            assert kept in ds.variables

    def test_original_open_is_unchanged(self, image):
        path, engine, _ = image
        dropped = xr.open_dataset(path, engine=engine, drop_variables="FLAG_SKY")
        full = xr.open_dataset(path, engine=engine)
        assert full.attrs["data_groups"] == {"base": {"sky": "SKY", "flag": "FLAG_SKY"}}
        assert dropped.attrs["data_groups"] == {"base": {"sky": "SKY"}}


# ---------------------------------------------------------------------------
# R5: image_type
# ---------------------------------------------------------------------------


class TestImageType:
    def test_parameter_is_declared(self):
        assert "image_type" in _CASA.open_dataset_parameters
        assert "image_type" in _FITS.open_dataset_parameters

    def test_image_type_overrides_the_name(self, casa_image):
        """A name that gives another role (or none) does not matter when
        image_type is given."""
        renamed = casa_image.with_name("ngc1234_cont.psf")
        casa_image.rename(renamed)
        assert "POINT_SPREAD_FUNCTION" in xr.open_dataset(renamed, engine=_CASA)
        ds = xr.open_dataset(renamed, engine=_CASA, image_type="sky")
        assert "SKY" in ds.data_vars
        assert "POINT_SPREAD_FUNCTION" not in ds.data_vars
        assert ds.attrs["data_groups"]["base"]["sky"] == "SKY"

    @pytest.mark.parametrize("image_format", ["casa", "fits"])
    def test_image_type_names_the_variable(self, image_format, casa_image, fits_image):
        path = casa_image if image_format == "casa" else fits_image
        engine, _ = _engine_and_kind(image_format)
        ds = xr.open_dataset(path, engine=engine, image_type="point_spread_function")
        assert "POINT_SPREAD_FUNCTION" in ds.data_vars
        assert "SKY" not in ds.data_vars

    def test_fits_suffixes_open_as_sky(self, fits_image):
        renamed = fits_image.with_name("target.fts")
        fits_image.rename(renamed)
        ds = xr.open_dataset(renamed, engine=_FITS)
        assert "SKY" in ds.data_vars

    def test_image_type_must_be_a_string(self, fits_image):
        with pytest.raises(TypeError, match="image_type"):
            xr.open_dataset(fits_image, engine=_FITS, image_type=1)

    @pytest.mark.parametrize("image_type", ["", "foo", "flag", "SKY_"])
    def test_unknown_image_type_raises(self, fits_image, image_type):
        with pytest.raises(ValueError, match="image_type"):
            xr.open_dataset(fits_image, engine=_FITS, image_type=image_type)

    @pytest.mark.parametrize(
        "image_type, variable",
        [
            ("SKY", "SKY"),
            ("sky_deconvolved", "SKY_DECONVOLVED"),
            ("pb", "PRIMARY_BEAM"),
            ("mask_deconvolve", "MASK"),
        ],
    )
    def test_image_type_names(self, fits_image, image_type, variable):
        ds = xr.open_dataset(fits_image, engine=_FITS, image_type=image_type)
        assert variable in ds.data_vars


# ---------------------------------------------------------------------------
# X5: what the engines claim and accept
# ---------------------------------------------------------------------------


def _write_fits(path, hdus):
    fits.HDUList(hdus).writeto(path)
    return path


class TestGuessCanOpen:
    def test_casa_image(self, casa_image):
        assert _CASA().guess_can_open(str(casa_image))
        assert _CASA().guess_can_open(casa_image)  # os.PathLike
        assert not _FITS().guess_can_open(str(casa_image))

    def test_fits_image(self, fits_image, tmp_path):
        assert _FITS().guess_can_open(str(fits_image))
        assert _FITS().guess_can_open(fits_image)
        assert not _CASA().guess_can_open(fits_image)
        for suffix in (".fit", ".fts", ".FITS"):
            other = tmp_path / f"copy{suffix}"
            shutil.copy(fits_image, other)
            assert _FITS().guess_can_open(other)

    @pytest.mark.parametrize("hdu", ["image", "compressed"])
    def test_fits_image_in_an_extension_is_not_claimed(self, tmp_path, hdu):
        """The reader reads the primary HDU only, so files whose image is in
        an extension HDU are not claimed, and the engine names the reason."""
        data = np.zeros((2, 3), np.float32)
        extension = fits.ImageHDU(data) if hdu == "image" else fits.CompImageHDU(data)
        path = _write_fits(tmp_path / "ext.fits", [fits.PrimaryHDU(), extension])
        assert not _FITS().guess_can_open(path)
        with pytest.raises(ValueError, match="image in its primary HDU"):
            xr.open_dataset(path, engine=_FITS)

    @pytest.mark.parametrize("kind", ["table_only", "random_groups", "not_fits"])
    def test_fits_files_without_an_image(self, tmp_path, kind):
        path = tmp_path / "other.fits"
        if kind == "table_only":
            table = fits.BinTableHDU.from_columns(
                [fits.Column(name="a", format="E", array=np.zeros(3))]
            )
            _write_fits(path, [fits.PrimaryHDU(), table])
        elif kind == "random_groups":
            data = fits.GroupData(
                np.zeros((2, 1, 1, 1, 3), np.float32),
                parnames=["UU", "VV"],
                pardata=[np.zeros(2), np.zeros(2)],
                bitpix=-32,
            )
            _write_fits(path, [fits.GroupsHDU(data)])
        else:
            path.write_text("not a FITS file\n")
        assert not _FITS().guess_can_open(path)
        with pytest.raises(ValueError, match="not a FITS image"):
            xr.open_dataset(path, engine=_FITS)

    def test_missing_paths_are_not_claimed(self, tmp_path):
        for engine in (_CASA, _FITS):
            assert not engine().guess_can_open(str(tmp_path / "missing.fits"))
            assert not engine().guess_can_open(str(tmp_path / "missing.im"))
        with pytest.raises(FileNotFoundError):
            xr.open_dataset(tmp_path / "missing.fits")

    def test_explicit_engine_missing_path(self, tmp_path):
        for engine in (_CASA, _FITS):
            with pytest.raises(FileNotFoundError):
                xr.open_dataset(tmp_path / "missing.im", engine=engine)

    def test_directory_named_fits_is_not_claimed(self, tmp_path):
        directory = tmp_path / "dir.fits"
        directory.mkdir()
        assert not _FITS().guess_can_open(directory)

    def test_non_image_tables_are_not_claimed(self, tmp_path):
        table = tmp_path / "data.ms"
        table.mkdir()
        (table / "table.info").write_bytes(b"Type = Measurement Set\nSubType = \n")
        assert not _CASA().guess_can_open(table)
        with pytest.raises(ValueError, match="not a CASA image"):
            xr.open_dataset(table, engine=_CASA)
        empty = tmp_path / "empty.im"
        empty.mkdir()
        assert not _CASA().guess_can_open(empty)
        binary = tmp_path / "binary.im"
        binary.mkdir()
        (binary / "table.info").write_bytes(b"\xff\xfe\x00 Type = Image")
        assert not _CASA().guess_can_open(binary)

    @pytest.mark.parametrize(
        "obj", [b"image.fits", 3, None, bytearray(b"x.fits"), memoryview(b"x")]
    )
    def test_non_path_objects_are_not_claimed(self, obj):
        assert not _CASA().guess_can_open(obj)
        assert not _FITS().guess_can_open(obj)

    def test_file_objects_are_rejected_by_open(self, fits_image):
        with open(fits_image, "rb") as fits_file:
            with pytest.raises(TypeError, match="local path"):
                xr.open_dataset(fits_file, engine=_FITS)


class TestWrongFormat:
    def test_casa_engine_rejects_fits(self, fits_image):
        with pytest.raises(ValueError, match="xradio_fits_image"):
            xr.open_dataset(fits_image, engine=_CASA)

    def test_fits_engine_rejects_casa(self, casa_image):
        with pytest.raises(ValueError, match="xradio_casa_image"):
            xr.open_dataset(casa_image, engine=_FITS)

    def test_engines_reject_zarr(self, tmp_path):
        store = tmp_path / "image.zarr"
        xr.Dataset({"a": ("x", np.zeros(2))}).to_zarr(store, consolidated=False)
        for engine in (_CASA, _FITS):
            with pytest.raises(ValueError, match="zarr"):
                xr.open_dataset(store, engine=engine)


# ---------------------------------------------------------------------------
# X2 and X4: light entry points, no casacore needed for FITS
# ---------------------------------------------------------------------------


def _run_python(code: str, cwd, *options: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, *options, "-c", textwrap.dedent(code)],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=600,
    )


class TestEntryPoints:
    def test_pyproject_points_at_the_light_module(self):
        with open(_REPO_ROOT / "pyproject.toml", "rb") as pyproject:
            entry_points = tomllib.load(pyproject)["project"]["entry-points"][
                "xarray.backends"
            ]
        assert entry_points == {
            "xradio_casa_image": "_xradio_xarray_backends:CasaImageBackendEntrypoint",
            "xradio_fits_image": "_xradio_xarray_backends:FitsImageBackendEntrypoint",
        }

    def test_classes_are_re_exported_for_the_docs(self):
        from xradio.image import backends

        assert backends.CasaImageBackendEntrypoint is _CASA
        assert backends.FitsImageBackendEntrypoint is _FITS

    def test_loading_the_entry_points_imports_no_reader(self, tmp_path):
        """Loading the entry points and guessing on a non-image path, as an
        engine-less xr.open_dataset does, imports neither xradio (whose
        __init__ imports zarr and sets warning filters) nor the reader
        dependencies."""
        (tmp_path / "data.nc").write_bytes(b"CDF\x01")
        result = _run_python(
            """
            import sys
            from importlib.metadata import EntryPoint

            for name in ("CasaImageBackendEntrypoint", "FitsImageBackendEntrypoint"):
                entry_point = EntryPoint(
                    name=name,
                    value=f"_xradio_xarray_backends:{name}",
                    group="xarray.backends",
                )
                backend = entry_point.load()()
                assert not backend.guess_can_open("data.nc")
                assert not backend.guess_can_open("missing.fits")
            heavy = [
                name
                for name in (
                    "xradio", "zarr", "dask", "astropy", "casacore", "casatools"
                )
                if name in sys.modules
            ]
            assert not heavy, heavy
            """,
            tmp_path,
        )
        assert result.returncode == 0, result.stderr

    def test_entry_points_load_on_an_xarray_only_install(self, tmp_path):
        """With the reader dependencies missing (the xarray-only install),
        the entry points still load and guess without any warning (run with
        warnings as errors), as xarray does for every engine-less open."""
        fits_file = tmp_path / "image.fits"
        fits.PrimaryHDU(np.zeros((2, 2), np.float32)).writeto(fits_file)
        result = _run_python(
            """
            import sys

            for name in (
                "dask", "astropy", "zarr", "numcodecs", "toolviper",
                "distributed", "casacore", "casatools",
            ):
                sys.modules[name] = None

            from _xradio_xarray_backends import (
                CasaImageBackendEntrypoint,
                FitsImageBackendEntrypoint,
            )

            assert not CasaImageBackendEntrypoint().guess_can_open("image.fits")
            # Without astropy the FITS engine cannot open, so it claims nothing
            assert not FitsImageBackendEntrypoint().guess_can_open("image.fits")
            """,
            tmp_path,
            "-W",
            "error",
        )
        assert result.returncode == 0, result.stderr


def test_fits_engine_without_astropy(tmp_path):
    """An explicit FITS engine open without astropy says that astropy is
    missing (guessing claims nothing)."""
    fits_file = tmp_path / "image.fits"
    fits.PrimaryHDU(np.zeros((2, 2), np.float32)).writeto(fits_file)
    result = _run_python(
        """
        import sys

        sys.modules["astropy"] = None

        import pytest
        import xarray as xr

        from _xradio_xarray_backends import FitsImageBackendEntrypoint

        assert not FitsImageBackendEntrypoint().guess_can_open("image.fits")
        with pytest.raises(ModuleNotFoundError, match="astropy"):
            xr.open_dataset("image.fits", engine=FitsImageBackendEntrypoint)
        """,
        tmp_path,
    )
    assert result.returncode == 0, result.stderr


def test_fits_without_casacore(fits_image, tmp_path):
    """FITS images open through the engine, with and without an explicit
    engine, and are written to FITS, with python-casacore and casatools
    blocked before xradio is imported."""
    result = _run_python(
        f"""
        import sys

        for name in ("casacore", "casatools", "casaconfig"):
            sys.modules[name] = None

        import dask
        import numpy as np
        import xarray as xr

        dask.config.set(scheduler="synchronous")
        from _xradio_xarray_backends import FitsImageBackendEntrypoint
        from xradio.image import open_image, write_image

        path = {str(fits_image)!r}
        ds = xr.open_dataset(path, engine=FitsImageBackendEntrypoint)
        assert ds.SKY.shape == (1, 10, 4, 30, 20), ds.SKY.shape
        ds.load()
        guessed = xr.open_dataset(path)
        np.testing.assert_array_equal(guessed.SKY.values, ds.SKY.values)
        write_image(ds, "written.fits", out_format="fits")
        written = open_image("written.fits")
        np.testing.assert_array_equal(written.SKY.values, ds.SKY.values)
        assert not [m for m in sys.modules if m.startswith(("casacore.", "casatools."))]
        """,
        tmp_path,
    )
    assert result.returncode == 0, result.stderr
