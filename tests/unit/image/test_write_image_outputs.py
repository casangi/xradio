"""Tests for the outputs of ``write_image``: their names, the overwrite
semantics, the staging of outputs in a temporary directory, and the CASA,
FITS and zarr drivers.

* ``TestOutputPlan``         -> the output planner (names, roles, warnings)
* ``TestMultiImageWrites``   -> one output per image variable (CASA and FITS)
* ``TestZarrStoreNames``     -> the .img.zarr extension of zarr stores and the
                                paths write_image returns
* ``TestOverwrite``          -> overwrite checks, temporary directory, failures
* ``TestFitsDriver``         -> FITS validation before any file is written
* ``TestCasaDriver``         -> CASA specific roles, masks and miscinfo
* ``TestZarrWriter``         -> chunk and codec encodings of the zarr writer
* ``TestOpenImageSelection`` -> integer selections of ``open_image``
"""

import os

import dask.array as da
import numpy as np
import pytest
import xarray as xr

from xradio._utils._casacore.tables import open_table_ro
from xradio.image import image as image_api
from xradio.image import (
    load_image,
    make_empty_sky_image,
    open_image,
    write_image,
)
from xradio.image._util import casacore as casa_driver
from xradio.image._util import fits as fits_driver
from xradio.image._util._write_plan import plan_image_outputs, zarr_store_name
from xradio.image._util._zarr import xds_to_zarr
from xradio.image.schema import IMAGE_SCHEMA_VERSION
from xradio.testing.image import download_image

_DIMS = ("time", "frequency", "polarization", "l", "m")
_GROUPS = ("deconvolved", "dirty", "model", "residual")


def _empty_image(n_chan: int = 2) -> xr.Dataset:
    return make_empty_sky_image(
        phase_center=[0.2, -0.5],
        image_size=[8, 6],
        cell_size=[np.pi / 180 / 60, np.pi / 180 / 60],
        frequency_coords=list(1.412e9 + 1e6 * np.arange(n_chan)),
        pol_coords=["I"],
        time_coords=[54000.1],
        do_sky_coords=False,
    )


def _shape(xds: xr.Dataset) -> tuple:
    return tuple(xds.sizes[dim] for dim in _DIMS)


def _add_image(xds: xr.Dataset, name: str, value: float, image_type: str) -> None:
    data = value + np.arange(np.prod(_shape(xds)), dtype=np.float32).reshape(
        _shape(xds)
    )
    xds[name] = xr.DataArray(
        data, dims=_DIMS, attrs={"type": image_type, "units": "Jy/beam"}
    )


def _add_flag(xds: xr.Dataset, name: str, l_index: int) -> None:
    flag = np.zeros(_shape(xds), dtype=bool)
    flag[0, 0, 0, l_index, 0] = True
    xds[name] = xr.DataArray(flag, dims=_DIMS, attrs={"type": "flag"})


def _add_beams(xds: xr.Dataset, name: str, scale: float) -> None:
    beams = np.empty(_shape(xds)[:3] + (3,))
    beams[..., 0] = 2e-5 * scale
    beams[..., 1] = 1e-5 * scale
    beams[..., 2] = 0.3
    beams[0, 0, 0, :2] *= 1.5  # per plane beams
    xds[name] = xr.DataArray(
        beams,
        dims=("time", "frequency", "polarization", "beam_params_label"),
        attrs={"type": name.lower(), "units": "rad"},
    )


def single_image_dataset() -> xr.Dataset:
    """A sky image with flags in data group 'base'."""
    xds = _empty_image()
    _add_image(xds, "SKY", 1.0, "sky")
    _add_flag(xds, "FLAG_SKY", 2)
    xds.attrs["data_groups"] = {"base": {"sky": "SKY", "flag": "FLAG_SKY"}}
    return xds


def multi_group_dataset() -> xr.Dataset:
    """Four data groups, each with its own sky image and flags, sharing a
    point spread function (with beam fit parameters) and a primary beam."""
    xds = _empty_image()
    groups = {}
    for index, group in enumerate(_GROUPS):
        sky = f"SKY_{group.upper()}"
        _add_image(xds, sky, 100.0 * (index + 1), "sky")
        _add_flag(xds, f"FLAG_{sky}", index)
        groups[group] = {"sky": sky, "flag": f"FLAG_{sky}"}
    _add_image(xds, "POINT_SPREAD_FUNCTION", 1000.0, "point_spread_function")
    _add_beams(xds, "BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION", 1.0)
    _add_image(xds, "PRIMARY_BEAM", 2000.0, "primary_beam")
    for group in groups.values():
        group.update(
            point_spread_function="POINT_SPREAD_FUNCTION",
            beam_fit_params_point_spread_function=(
                "BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION"
            ),
            primary_beam="PRIMARY_BEAM",
        )
    xds.attrs["data_groups"] = groups
    return xds


def _expected_multi_group_names(stem: str, extension: str = "") -> set:
    shared = ".".join(_GROUPS)
    return {f"{stem}.{group}.sky{extension}" for group in _GROUPS} | {
        f"{stem}.{shared}.point_spread_function{extension}",
        f"{stem}.{shared}.primary_beam{extension}",
    }


def _listing(directory) -> list:
    return sorted(os.listdir(directory))


def _sky_with_nans(xds: xr.Dataset, sky: str, flag: str) -> np.ndarray:
    return np.where(xds[flag].values, np.nan, xds[sky].values)


@pytest.fixture(scope="module")
def uv_image(tmp_path_factory):
    directory = tmp_path_factory.mktemp("uv_image")
    return str(download_image("complex_valued_uv.im", directory))


# --------------------------------------------------------------------------- #
# TestOutputPlan                                                               #
# --------------------------------------------------------------------------- #


class TestOutputPlan:
    """The planner names one output per image variable."""

    def test_single_image_keeps_the_given_name(self):
        plan = plan_image_outputs(single_image_dataset(), "out.fits", "fits")
        assert plan.paths == ["out.fits"]
        (output,) = plan.outputs
        assert (output.variable, output.role, output.groups) == (
            "SKY",
            "sky",
            ("base",),
        )
        assert (output.flag, output.beam_fit_params) == ("FLAG_SKY", None)

    def test_groups_are_always_in_multi_image_names(self):
        xds = single_image_dataset()
        _add_image(xds, "POINT_SPREAD_FUNCTION", 5.0, "point_spread_function")
        xds.attrs["data_groups"]["base"]["point_spread_function"] = (
            "POINT_SPREAD_FUNCTION"
        )
        plan = plan_image_outputs(xds, "out", "casa")
        assert plan.paths == ["out.base.sky", "out.base.point_spread_function"]

    @pytest.mark.parametrize("extension", [".fits", ".FITS", ".Fits"])
    def test_fits_extension_stays_last(self, extension):
        plan = plan_image_outputs(multi_group_dataset(), f"img{extension}", "fits")
        assert set(plan.paths) == _expected_multi_group_names("img", extension)

    def test_casa_names_keep_any_extension(self):
        plan = plan_image_outputs(multi_group_dataset(), "img.fits", "casa")
        assert set(plan.paths) == _expected_multi_group_names("img.fits")

    def test_shared_variable_names_its_groups_in_data_group_order(self):
        xds = multi_group_dataset()
        # the point spread function is shared by the dirty and residual groups
        # only, which are listed in that order
        for group in ("deconvolved", "model"):
            del xds.attrs["data_groups"][group]["point_spread_function"]
        plan = plan_image_outputs(xds, "img.fits", "fits")
        psf = [o for o in plan.outputs if o.variable == "POINT_SPREAD_FUNCTION"]
        assert [o.path for o in psf] == [
            "img.dirty.residual.point_spread_function.fits"
        ]
        assert psf[0].beam_fit_params == "BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION"

    def test_each_variable_written_once(self):
        plan = plan_image_outputs(multi_group_dataset(), "out", "casa")
        variables = [output.variable for output in plan.outputs]
        assert sorted(variables) == sorted(set(variables))
        assert len(variables) == 6
        flags = {output.variable: output.flag for output in plan.outputs}
        for group in _GROUPS:
            assert flags[f"SKY_{group.upper()}"] == f"FLAG_SKY_{group.upper()}"

    def test_variable_with_two_roles_is_written_once(self):
        xds = single_image_dataset()
        xds.attrs["data_groups"]["other"] = {"primary_beam": "SKY"}
        plan = plan_image_outputs(xds, "out", "casa")
        assert [(o.variable, o.role) for o in plan.outputs] == [("SKY", "sky")]
        assert any("written once, as sky" in message for message in plan.warnings)

    def test_unreferenced_variables_are_reported(self):
        xds = single_image_dataset()
        _add_image(xds, "STRAY", 0.0, "sky")
        plan = plan_image_outputs(xds, "out", "fits")
        assert plan.paths == ["out"]
        assert any("STRAY" in message for message in plan.warnings)

    def test_images_fits_cannot_hold_are_skipped_with_one_warning(self):
        xds = single_image_dataset()
        for name in ("VISIBILITY_NORMALIZATION", "UV_SAMPLING_NORMALIZATION"):
            xds[name] = xr.DataArray(
                np.ones(_shape(xds)[:3]),
                dims=_DIMS[:3],
                attrs={"type": name.lower()},
            )
            xds.attrs["data_groups"]["base"][name.lower()] = name
        for out_format in ("casa", "fits"):
            plan = plan_image_outputs(xds, "out", out_format)
            assert plan.paths == ["out"]
            skipped = [m for m in plan.warnings if "NORMALIZATION" in m]
            assert len(skipped) == 1
            assert "VISIBILITY_NORMALIZATION" in skipped[0]
            assert "UV_SAMPLING_NORMALIZATION" in skipped[0]

    def test_no_writable_image_raises(self):
        xds = single_image_dataset().rename({"l": "u", "m": "v"})
        with pytest.raises(ValueError, match="No valid image types found"):
            plan_image_outputs(xds, "out", "fits")

    def test_missing_data_groups_raise(self):
        xds = single_image_dataset()
        del xds.attrs["data_groups"]
        with pytest.raises(ValueError, match="data_groups"):
            plan_image_outputs(xds, "out", "casa")

    def test_missing_flag_variable_raises(self):
        xds = single_image_dataset().drop_vars("FLAG_SKY")
        with pytest.raises(ValueError, match="FLAG_SKY"):
            plan_image_outputs(xds, "out", "casa")

    @pytest.mark.parametrize("group", ["a/b", "", "..", "."])
    def test_group_names_must_be_file_name_parts(self, group):
        xds = multi_group_dataset()
        groups = xds.attrs["data_groups"]
        groups[group] = groups.pop("dirty")
        with pytest.raises(ValueError, match="cannot be part of an output file name"):
            plan_image_outputs(xds, "out", "casa")

    def test_ambiguous_output_names_raise(self):
        xds = single_image_dataset()
        _add_image(xds, "SKY_2", 3.0, "sky")
        xds.attrs["data_groups"] = {
            "a.b": {"sky": "SKY"},
            "a": {"sky": "SKY_2"},
            "b": {"sky": "SKY_2"},
        }
        with pytest.raises(ValueError, match=r"out\.a\.b\.sky"):
            plan_image_outputs(xds, "out", "casa")

    @pytest.mark.parametrize("out_format", ["casa", "fits"])
    def test_names_that_differ_only_in_case_raise(self, tmp_path, out_format):
        """On case insensitive file systems (macOS) they are one file."""
        xds = single_image_dataset()
        _add_image(xds, "SKY_2", 3.0, "sky")
        xds.attrs["data_groups"] = {"Base": {"sky": "SKY"}, "base": {"sky": "SKY_2"}}
        with pytest.raises(ValueError, match="differ only in case"):
            plan_image_outputs(xds, "out", out_format)
        with pytest.raises(ValueError, match="differ only in case"):
            write_image(xds, str(tmp_path / "out"), out_format)
        assert _listing(tmp_path) == []


# --------------------------------------------------------------------------- #
# TestMultiImageWrites                                                         #
# --------------------------------------------------------------------------- #


class TestMultiImageWrites:
    """Every image of every data group survives a CASA or FITS write."""

    @pytest.mark.parametrize(
        "out_format, name",
        [("casa", "out.im"), ("fits", "out.fits"), ("zarr", "o.img.zarr")],
    )
    def test_single_image_is_written_to_the_given_name(
        self, tmp_path, out_format, name
    ):
        written = write_image(single_image_dataset(), str(tmp_path / name), out_format)
        assert written == [str(tmp_path / name)]
        assert _listing(tmp_path) == [name]

    def test_casa_writes_every_group(self, tmp_path):
        xds = multi_group_dataset()
        write_image(xds, str(tmp_path / "out"), "casa")
        assert set(_listing(tmp_path)) == _expected_multi_group_names("out")
        shared = ".".join(_GROUPS)
        for group in _GROUPS:
            sky = f"SKY_{group.upper()}"
            got = load_image({"sky": str(tmp_path / f"out.{group}.sky")})
            np.testing.assert_array_equal(got["SKY"].values, xds[sky].values)
            np.testing.assert_array_equal(
                got["FLAG_SKY"].values, xds[f"FLAG_{sky}"].values
            )
        psf = load_image(
            {
                "point_spread_function": str(
                    tmp_path / f"out.{shared}.point_spread_function"
                )
            }
        )
        np.testing.assert_array_equal(
            psf["POINT_SPREAD_FUNCTION"].values, xds["POINT_SPREAD_FUNCTION"].values
        )
        np.testing.assert_allclose(
            psf["BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION"].values,
            xds["BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION"].values,
            rtol=1e-6,
        )
        pb = load_image({"primary_beam": str(tmp_path / f"out.{shared}.primary_beam")})
        np.testing.assert_array_equal(
            pb["PRIMARY_BEAM"].values, xds["PRIMARY_BEAM"].values
        )

    @pytest.mark.parametrize(
        "out_format, name", [("casa", "img"), ("fits", "img.fits")]
    )
    def test_outputs_reopen_as_a_list(self, tmp_path, out_format, name):
        """The role of each output is recognized from its name, so the
        outputs of one data group reopen as one dataset."""
        xds = single_image_dataset()
        _add_image(xds, "POINT_SPREAD_FUNCTION", 10.0, "point_spread_function")
        _add_image(xds, "PRIMARY_BEAM", 20.0, "primary_beam")
        mask = np.zeros(_shape(xds), dtype=np.float32)
        mask[..., 1:3, 1:3] = 1
        xds["MASK"] = xr.DataArray(mask, dims=_DIMS, attrs={"type": "mask"})
        xds.attrs["data_groups"]["base"].update(
            point_spread_function="POINT_SPREAD_FUNCTION",
            primary_beam="PRIMARY_BEAM",
            mask="MASK",
        )
        write_image(xds, str(tmp_path / name), out_format)
        extension = ".fits" if out_format == "fits" else ""
        roles = ("sky", "point_spread_function", "primary_beam", "mask")
        assert _listing(tmp_path) == sorted(
            f"img.base.{role}{extension}" for role in roles
        )
        got = open_image(
            [str(tmp_path / f"img.base.{role}{extension}") for role in roles]
        )
        expected_sky = (
            _sky_with_nans(xds, "SKY", "FLAG_SKY")
            if out_format == "fits"
            else xds["SKY"].values
        )
        np.testing.assert_array_equal(got["SKY"].values, expected_sky)
        np.testing.assert_array_equal(got["FLAG_SKY"].values, xds["FLAG_SKY"].values)
        for variable in ("POINT_SPREAD_FUNCTION", "PRIMARY_BEAM", "MASK"):
            np.testing.assert_array_equal(got[variable].values, xds[variable].values)
        groups = got.attrs["data_groups"]["base"]
        assert {role: groups.get(role) for role in roles} == {
            "sky": "SKY",
            "point_spread_function": "POINT_SPREAD_FUNCTION",
            "primary_beam": "PRIMARY_BEAM",
            "mask": "MASK",
        }

    def test_fits_writes_every_group(self, tmp_path):
        xds = multi_group_dataset()
        write_image(xds, str(tmp_path / "out.fits"), "fits")
        assert set(_listing(tmp_path)) == _expected_multi_group_names("out", ".fits")
        shared = ".".join(_GROUPS)
        for group in _GROUPS:
            sky = f"SKY_{group.upper()}"
            got = open_image({"sky": str(tmp_path / f"out.{group}.sky.fits")})
            np.testing.assert_array_equal(
                got["SKY"].values, _sky_with_nans(xds, sky, f"FLAG_{sky}")
            )
        for role, variable in (
            ("point_spread_function", "POINT_SPREAD_FUNCTION"),
            ("primary_beam", "PRIMARY_BEAM"),
        ):
            got = open_image({role: str(tmp_path / f"out.{shared}.{role}.fits")})
            np.testing.assert_array_equal(got[variable].values, xds[variable].values)


# --------------------------------------------------------------------------- #
# TestZarrStoreNames                                                           #
# --------------------------------------------------------------------------- #


@pytest.fixture
def write_image_log(monkeypatch):
    """The info and warning messages that write_image itself logs (not its
    drivers), as lists keyed by level."""

    class _Logger:
        def __init__(self):
            self.messages = {"info": [], "warning": []}

        def info(self, message, *args, **kwargs):
            self.messages["info"].append(str(message))

        def warning(self, message, *args, **kwargs):
            self.messages["warning"].append(str(message))

        def __getattr__(self, name):
            return lambda *args, **kwargs: None

    logger = _Logger()
    monkeypatch.setattr(image_api, "xradio_logger", lambda: logger)
    return logger.messages


class TestZarrStoreNames:
    """Zarr stores get the extension .img.zarr, and write_image returns the
    paths it wrote, for every format (single images: see
    TestMultiImageWrites.test_single_image_is_written_to_the_given_name)."""

    @pytest.mark.parametrize(
        "name, store",
        [
            ("out", "out.img.zarr"),
            ("out.zarr", "out.img.zarr"),
            ("out.ZARR", "out.img.zarr"),
            ("out.Zarr", "out.img.zarr"),
            ("out.img.zarr", "out.img.zarr"),
            ("out.IMG.ZARR", "out.IMG.ZARR"),
            ("out.Img.Zarr", "out.Img.Zarr"),
            ("out.img", "out.img.img.zarr"),
            ("out.ps.zarr", "out.ps.img.zarr"),
            ("out.zarr.old", "out.zarr.old.img.zarr"),
            # only the end of the name counts, with the leading dot
            ("img.zarr", "img.img.zarr"),
            ("out.img.zarr.zarr", "out.img.zarr.img.zarr"),
            (os.path.join("dir.zarr", "out"), os.path.join("dir.zarr", "out.img.zarr")),
        ],
    )
    def test_store_name(self, name, store):
        assert zarr_store_name(name) == store

    @pytest.mark.parametrize(
        "name, store",
        [
            ("out", "out.img.zarr"),
            ("out.zarr", "out.img.zarr"),
            ("out.ZARR", "out.img.zarr"),
            ("out.img.zarr", "out.img.zarr"),
            ("out.IMG.Zarr", "out.IMG.Zarr"),
        ],
    )
    def test_write_returns_and_logs_the_store_name(
        self, tmp_path, write_image_log, name, store
    ):
        xds = single_image_dataset()
        written = write_image(xds, str(tmp_path / name), "zarr")
        assert written == [str(tmp_path / store)]
        assert _listing(tmp_path) == [store]
        got = open_image(written[0])
        np.testing.assert_array_equal(got["SKY"].values, xds["SKY"].values)
        assert got.attrs["schema_version"] == IMAGE_SCHEMA_VERSION
        if store == name:
            assert write_image_log["info"] == []
        else:
            # the store name is logged when write_image changed it
            (message,) = write_image_log["info"]
            assert ".img.zarr" in message
            assert str(tmp_path / name) in message
            assert str(tmp_path / store) in message
        # nothing has the given name, so there is nothing to warn about
        assert write_image_log["warning"] == []

    def test_trailing_separator_and_home_directory(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        written = write_image(single_image_dataset(), "~/out.zarr" + os.sep, "zarr")
        assert written == [str(tmp_path / "out.img.zarr")]
        assert _listing(tmp_path) == ["out.img.zarr"]

    def test_overwrite_applies_to_the_store_name(self, tmp_path, write_image_log):
        """The overwrite check, the staging directory and the replacement
        concern the store name with the .img.zarr extension, not the name
        given: a store at the given name (for example one written by an
        earlier xradio) is neither checked nor changed, and every write warns
        that it is left in place."""
        xds = single_image_dataset()
        given = tmp_path / "out.zarr"
        given.mkdir()
        (given / "old").write_text("keep me")
        store = str(tmp_path / "out.img.zarr")
        assert write_image(xds, str(given), "zarr") == [store]
        assert _listing(tmp_path) == ["out.img.zarr", "out.zarr"]
        (warning,) = write_image_log["warning"]
        assert f"{given} exists and is not replaced" in warning
        assert store in warning
        with pytest.raises(FileExistsError, match=r"out\.img\.zarr"):
            write_image(xds, str(given), "zarr")
        # nothing is logged when the overwrite check fails (nothing is written)
        assert len(write_image_log["info"]) == 1
        assert len(write_image_log["warning"]) == 1
        changed = xds.copy(deep=True)
        changed["SKY"] = changed["SKY"] + 1
        assert write_image(changed, str(given), "zarr", overwrite=True) == [store]
        assert len(write_image_log["warning"]) == 2
        np.testing.assert_array_equal(
            open_image(store)["SKY"].values, changed["SKY"].values
        )
        # no staging directory is left behind, and the given name is unchanged
        assert _listing(tmp_path) == ["out.img.zarr", "out.zarr"]
        assert _listing(given) == ["old"]
        assert (given / "old").read_text() == "keep me"

    def test_file_with_the_given_name_is_kept_with_a_warning(
        self, tmp_path, write_image_log
    ):
        given = tmp_path / "out"
        given.write_text("not a store")
        written = write_image(single_image_dataset(), str(given), "zarr", True)
        assert written == [str(tmp_path / "out.img.zarr")]
        assert _listing(tmp_path) == ["out", "out.img.zarr"]
        assert given.read_text() == "not a store"
        (warning,) = write_image_log["warning"]
        assert str(given) in warning and written[0] in warning

    @pytest.mark.parametrize(
        "out_format, name, extension",
        [("casa", "out", ""), ("fits", "out.fits", ".fits")],
    )
    def test_multi_image_write_returns_every_output(
        self, tmp_path, out_format, name, extension
    ):
        xds = multi_group_dataset()
        written = write_image(xds, str(tmp_path / name), out_format)
        # every output, in output order
        assert (
            written == plan_image_outputs(xds, str(tmp_path / name), out_format).paths
        )
        assert sorted(written) == sorted(
            str(tmp_path / output)
            for output in _expected_multi_group_names("out", extension)
        )
        assert sorted(os.path.basename(path) for path in written) == _listing(tmp_path)


# --------------------------------------------------------------------------- #
# TestOverwrite                                                                #
# --------------------------------------------------------------------------- #


def _single_image_written_back(path: str, out_format: str) -> None:
    """Open an image lazily and write it back onto its own source."""
    lazy = open_image(path)
    expected = lazy.compute()
    write_image(lazy, path, out_format, overwrite=True)
    got = open_image(path).compute()
    sky = [name for name in got.data_vars if name.startswith("SKY")][0]
    np.testing.assert_array_equal(got[sky].values, expected[sky].values)


class TestOverwrite:
    """Overwrite checks happen before anything is written, outputs are
    staged in a temporary directory, and a failure changes nothing."""

    @pytest.mark.parametrize("out_format", ["casa", "fits"])
    def test_existing_derived_output_raises_before_writing(self, tmp_path, out_format):
        name = "out.fits" if out_format == "fits" else "out"
        extension = ".fits" if out_format == "fits" else ""
        stem = "out"
        shared = ".".join(_GROUPS)
        existing = tmp_path / f"{stem}.{shared}.point_spread_function{extension}"
        existing.write_text("not an image")
        with pytest.raises(FileExistsError, match="point_spread_function"):
            write_image(multi_group_dataset(), str(tmp_path / name), out_format)
        assert _listing(tmp_path) == [existing.name]
        assert existing.read_text() == "not an image"

    def test_overwrite_replaces_only_the_outputs(self, tmp_path):
        unrelated = tmp_path / "out.fits.notes"
        unrelated.write_text("keep me")
        existing = tmp_path / "out.dirty.sky.fits"
        existing.write_text("old")
        write_image(multi_group_dataset(), str(tmp_path / "out.fits"), "fits", True)
        assert set(_listing(tmp_path)) == _expected_multi_group_names(
            "out", ".fits"
        ) | {unrelated.name}
        assert unrelated.read_text() == "keep me"
        got = open_image({"sky": str(existing)})
        assert got.sizes["frequency"] == 2

    @pytest.mark.parametrize(
        "out_format, writer",
        [("casa", "_write_casa_image"), ("fits", "_xds_to_fits_image")],
    )
    def test_failure_changes_nothing(self, tmp_path, monkeypatch, out_format, writer):
        """A failure while writing the second image leaves the existing
        outputs untouched and removes the temporary directory."""
        module = casa_driver if out_format == "casa" else fits_driver
        extension = ".fits" if out_format == "fits" else ""
        name = str(tmp_path / f"out{extension}")
        existing = tmp_path / f"out.deconvolved.sky{extension}"
        existing.write_text("old")
        original = getattr(module, writer)
        calls = []

        def fail_on_second(*args, **kwargs):
            calls.append(args[1])
            if len(calls) == 2:
                raise RuntimeError("simulated write failure")
            return original(*args, **kwargs)

        monkeypatch.setattr(module, writer, fail_on_second)
        with pytest.raises(RuntimeError, match="simulated write failure"):
            write_image(multi_group_dataset(), name, out_format, overwrite=True)
        assert len(calls) == 2
        assert _listing(tmp_path) == [existing.name]
        assert existing.read_text() == "old"

    @pytest.mark.parametrize("name", ["out.zarr", "out", "out.img.zarr"])
    def test_zarr_failure_changes_nothing(self, tmp_path, monkeypatch, name):
        """A zarr write that fails, under any name that gives the store
        out.img.zarr, leaves the existing out.img.zarr untouched and removes
        the temporary directory."""
        store = tmp_path / "out.img.zarr"
        xds = single_image_dataset()
        write_image(xds, str(store), "zarr")
        files = sorted(str(path.relative_to(store)) for path in store.rglob("*"))
        original = image_api._xds_to_zarr
        staged = []

        def fail_after_writing(xds, path, *args, **kwargs):
            staged.append(path)
            original(xds, path, *args, **kwargs)
            raise RuntimeError("simulated write failure")

        monkeypatch.setattr(image_api, "_xds_to_zarr", fail_after_writing)
        changed = xds.copy(deep=True)
        changed["SKY"] = changed["SKY"] + 1
        with pytest.raises(RuntimeError, match="simulated write failure"):
            write_image(changed, str(tmp_path / name), "zarr", overwrite=True)
        # the store was written into the temporary directory under its final
        # name, and that directory is removed
        (path,) = staged
        assert os.path.basename(path) == "out.img.zarr"
        assert os.path.dirname(path) != str(tmp_path)
        assert _listing(tmp_path) == ["out.img.zarr"]
        assert sorted(str(p.relative_to(store)) for p in store.rglob("*")) == files
        np.testing.assert_array_equal(
            open_image(str(store))["SKY"].values, xds["SKY"].values
        )

    def test_created_parent_directories_are_removed_on_failure(
        self, tmp_path, monkeypatch
    ):
        def fail(*args, **kwargs):
            raise RuntimeError("simulated write failure")

        monkeypatch.setattr(fits_driver, "_xds_to_fits_image", fail)
        with pytest.raises(RuntimeError):
            write_image(single_image_dataset(), str(tmp_path / "a/b/out.fits"), "fits")
        assert _listing(tmp_path) == []

    def test_missing_parent_directories_are_created(self, tmp_path):
        write_image(single_image_dataset(), str(tmp_path / "a/b/out.fits"), "fits")
        assert _listing(tmp_path / "a" / "b") == ["out.fits"]

    @pytest.mark.parametrize(
        "out_format, name",
        [("zarr", "a.img.zarr"), ("casa", "a.im"), ("fits", "a.fits")],
    )
    def test_lazy_dataset_written_back_onto_its_source(
        self, tmp_path, out_format, name
    ):
        path = str(tmp_path / name)
        write_image(single_image_dataset(), path, out_format)
        _single_image_written_back(path, out_format)
        assert _listing(tmp_path) == [name]

    def test_source_with_a_derived_name_is_kept(self, tmp_path):
        """Writing an image opened from foo.sky to foo neither removes nor
        changes foo.sky (it used to be replaced while still being read)."""
        source = str(tmp_path / "foo.sky")
        write_image(single_image_dataset(), source, "casa")
        xds = open_image(source)
        expected = xds["SKY"].values.copy()
        write_image(xds, str(tmp_path / "foo"), "casa")
        assert _listing(tmp_path) == ["foo", "foo.sky"]
        for path in (source, str(tmp_path / "foo")):
            np.testing.assert_array_equal(
                load_image({"sky": path})["SKY"].values, expected
            )

    def test_home_directory_is_expanded(self, tmp_path, monkeypatch):
        home = tmp_path / "home"
        home.mkdir()
        work = tmp_path / "work"
        work.mkdir()
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.chdir(work)
        write_image(single_image_dataset(), "~/k2.fits", "fits")
        assert _listing(home) == ["k2.fits"]
        assert _listing(work) == []
        with pytest.raises(FileExistsError):
            write_image(single_image_dataset(), "~/k2.fits", "fits")
        write_image(single_image_dataset(), "~/k2.fits", "fits", overwrite=True)
        assert _listing(home) == ["k2.fits"]
        write_image(multi_group_dataset(), "~/multi.fits", "fits")
        assert set(_listing(home)) == {"k2.fits"} | _expected_multi_group_names(
            "multi", ".fits"
        )
        assert _listing(work) == []

    def test_unsupported_format_raises_before_writing(self, tmp_path):
        with pytest.raises(ValueError, match="not supported"):
            write_image(single_image_dataset(), str(tmp_path / "out"), "hdf5")
        assert _listing(tmp_path) == []

    @pytest.mark.parametrize("scheme", ["file://", "s3://"])
    @pytest.mark.parametrize("out_format", ["zarr", "fits"])
    def test_urls_are_rejected(self, tmp_path, monkeypatch, scheme, out_format):
        """URLs are not written to a local directory named after them."""
        monkeypatch.chdir(tmp_path)
        target = f"{scheme}{tmp_path}/bucket/image.{out_format}"
        with pytest.raises(ValueError, match="local paths only"):
            write_image(single_image_dataset(), target, out_format)
        assert _listing(tmp_path) == []

    def test_open_and_load_expand_the_home_directory(self, tmp_path, monkeypatch):
        home = tmp_path / "home"
        home.mkdir()
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.chdir(tmp_path)
        xds = single_image_dataset()
        for out_format in ("fits", "zarr", "casa"):
            name = "k2.img.zarr" if out_format == "zarr" else f"k2.{out_format}"
            path = f"~/{name}"
            # the returned path has the home directory expanded
            assert write_image(xds, path, out_format) == [str(home / name)]
            reopened = open_image(path)
            np.testing.assert_array_equal(
                reopened.SKY.values,
                _sky_with_nans(xds, "SKY", "FLAG_SKY")
                if out_format == "fits"
                else xds.SKY.values,
            )
            assert open_image([path]).SKY.shape == xds.SKY.shape
            assert open_image({"sky": path}).SKY.shape == xds.SKY.shape
            if out_format != "fits":
                assert load_image(path).SKY.shape == xds.SKY.shape


# --------------------------------------------------------------------------- #
# TestFitsDriver                                                               #
# --------------------------------------------------------------------------- #


class TestFitsDriver:
    """FITS specific parts of the multi-image driver."""

    def test_every_header_is_validated_before_any_file(self, tmp_path, monkeypatch):
        original = fits_driver._fits_image_header
        written = []

        def header(image_xds):
            if image_xds["SKY"].attrs.get("type") == "primary_beam":
                raise ValueError("bad primary beam metadata")
            return original(image_xds)

        def write_fits(image_xds, path, **kwargs):
            written.append(path)

        monkeypatch.setattr(fits_driver, "_fits_image_header", header)
        monkeypatch.setattr(fits_driver, "_xds_to_fits_image", write_fits)
        # the error names the variable that cannot be written
        with pytest.raises(
            ValueError,
            match="Cannot write PRIMARY_BEAM to FITS: bad primary beam metadata",
        ):
            write_image(multi_group_dataset(), str(tmp_path / "out.fits"), "fits")
        assert written == []
        assert _listing(tmp_path) == []

    def test_images_with_several_time_planes_are_rejected_first(self, tmp_path):
        xds = multi_group_dataset()
        xds = xr.concat([xds, xds.assign_coords(time=xds.time + 1)], dim="time")
        with pytest.raises(RuntimeError, match="exactly one time plane"):
            write_image(xds, str(tmp_path / "out.fits"), "fits")
        assert _listing(tmp_path) == []


# --------------------------------------------------------------------------- #
# TestCasaDriver                                                               #
# --------------------------------------------------------------------------- #


class TestCasaDriver:
    """CASA specific parts of the multi-image driver."""

    def test_metadata_of_every_image_is_validated_before_any_pixels(
        self, tmp_path, monkeypatch
    ):
        """The point spread function comes after the first sky image, so its
        bad metadata must be found before the sky pixels are written."""
        original = casa_driver._casa_image_keywords
        written = []

        def keywords(image_xds):
            if image_xds["SKY"].attrs.get("type") == "point_spread_function":
                raise ValueError("bad point spread function metadata")
            return original(image_xds)

        def write_casa_data(image_xds, path):
            written.append(path)

        monkeypatch.setattr(casa_driver, "_casa_image_keywords", keywords)
        monkeypatch.setattr(casa_driver, "_write_casa_data", write_casa_data)
        with pytest.raises(ValueError, match="bad point spread function metadata"):
            write_image(multi_group_dataset(), str(tmp_path / "out"), "casa")
        assert written == []
        assert _listing(tmp_path) == []

    def test_deconvolution_mask_is_not_its_own_mask(self, tmp_path):
        xds = single_image_dataset()
        mask = np.zeros(_shape(xds), dtype=np.float32)
        mask[..., 2:4, 1:3] = 1
        xds["MASK"] = xr.DataArray(mask, dims=_DIMS, attrs={"type": "mask"})
        xds.attrs["data_groups"]["base"]["mask"] = "MASK"
        write_image(xds, str(tmp_path / "out"), "casa")
        mask_path = str(tmp_path / "out.base.mask")
        with open_table_ro(mask_path) as tb:
            keywords = tb.getkeywords()
        assert "SKY" not in keywords.get("masks", {})
        assert not keywords.get("Image_defaultmask")
        got = load_image({"mask": mask_path})
        np.testing.assert_array_equal(got["MASK"].values, mask)
        # the caller's dataset keeps its attributes
        assert xds["MASK"].attrs["type"] == "mask"

    def test_uv_roles_are_written_as_aperture_images(self, tmp_path, uv_image):
        xds = open_image({"aperture": uv_image})
        xds["VISIBILITY"] = xds["APERTURE"] * 2
        xds.attrs["data_groups"]["base"]["visibility"] = "VISIBILITY"
        write_image(xds, str(tmp_path / "out"), "casa")
        assert _listing(tmp_path) == ["out.base.aperture", "out.base.visibility"]
        for role, variable in (("aperture", "APERTURE"), ("visibility", "VISIBILITY")):
            path = str(tmp_path / f"out.base.{role}")
            with open_table_ro(path) as tb:
                coords = tb.getkeyword("coords")
            assert "linear0" in coords and "direction0" not in coords
            got = open_image({"aperture": path})
            np.testing.assert_array_equal(got["APERTURE"].values, xds[variable].values)

    def test_normalization_images_are_skipped(self, tmp_path):
        xds = single_image_dataset()
        xds["VISIBILITY_NORMALIZATION"] = xr.DataArray(
            np.ones(_shape(xds)[:3]),
            dims=_DIMS[:3],
            attrs={"type": "visibility_normalization"},
        )
        xds.attrs["data_groups"]["base"]["visibility_normalization"] = (
            "VISIBILITY_NORMALIZATION"
        )
        write_image(xds, str(tmp_path / "out.im"), "casa")
        assert _listing(tmp_path) == ["out.im"]

    def test_miscinfo_comes_from_the_image_variable(self, tmp_path):
        xds = single_image_dataset()
        xds.attrs["user"] = {"origin": "dataset", "instrume": "dataset level"}
        xds["SKY"].attrs["user"] = {
            "instrume": "ALMA",
            "nested": {"a": 1},
            "mixed": [1, "a"],
        }
        first = str(tmp_path / "first.im")
        write_image(xds, first, "casa")
        with open_table_ro(first) as tb:
            miscinfo = tb.getkeyword("miscinfo")
        assert miscinfo == {"origin": "dataset", "instrume": "ALMA", "nested": {"a": 1}}
        # CASA to CASA keeps the miscinfo (the reader stores it on the image)
        second = str(tmp_path / "second.im")
        write_image(open_image(first), second, "casa")
        with open_table_ro(second) as tb:
            assert tb.getkeyword("miscinfo") == miscinfo


# --------------------------------------------------------------------------- #
# TestZarrWriter                                                               #
# --------------------------------------------------------------------------- #


class TestZarrWriter:
    """The zarr writer adapts inherited encodings and dask chunks."""

    def test_irregular_dask_chunks(self, tmp_path):
        xds = single_image_dataset()
        xds = xds.isel(frequency=[0, 1, 0, 1, 1]).assign_coords(
            frequency=1.412e9 + 1e6 * np.arange(5)
        )
        xds = xds.chunk({"frequency": (2, 1, 2)})
        before = {name: dict(xds[name].encoding) for name in xds.data_vars}
        write_image(xds, str(tmp_path / "irregular.img.zarr"), "zarr")
        got = open_image(str(tmp_path / "irregular.img.zarr"))
        np.testing.assert_array_equal(got["SKY"].values, xds["SKY"].values)
        assert got["SKY"].encoding["chunks"][1] == 2
        # the caller's dataset is not changed
        assert xds["SKY"].chunks[1] == (2, 1, 2)
        assert {name: dict(xds[name].encoding) for name in xds.data_vars} == before

    def test_selection_of_a_zarr_image(self, tmp_path):
        source = str(tmp_path / "source.img.zarr")
        xds = single_image_dataset()
        xds = xds.isel(frequency=[0, 1, 0, 1]).assign_coords(
            frequency=1.412e9 + 1e6 * np.arange(4)
        )
        write_image(xds.chunk({"frequency": 2}), source, "zarr")
        selected = open_image(source, selection={"frequency": slice(1, 4)})
        assert selected["SKY"].encoding["chunks"][1] == 2
        write_image(selected, str(tmp_path / "selected.img.zarr"), "zarr")
        got = open_image(str(tmp_path / "selected.img.zarr"))
        np.testing.assert_array_equal(
            got["SKY"].values, xds["SKY"].isel(frequency=slice(1, 4)).values
        )

    def test_zarr_v2_store_is_rewritten_as_zarr_v3(self, tmp_path, monkeypatch):
        """Stores of older xradio versions are zarr v2; their inherited v2
        codecs (also of datasets stored in attrs) must not reach a v3 write."""
        old = str(tmp_path / "old.img.zarr")
        source = single_image_dataset()
        source.attrs["extra"] = xr.Dataset({"A": ("row", np.arange(4.0))})
        monkeypatch.setattr(xds_to_zarr, "ZARR_FORMAT", 2)
        write_image(source, old, "zarr")
        monkeypatch.undo()
        assert os.path.exists(os.path.join(old, ".zgroup"))
        xds = open_image(old)
        new = str(tmp_path / "new.img.zarr")
        write_image(xds, new, "zarr")
        assert os.path.exists(os.path.join(new, "zarr.json"))
        got = open_image(new)
        np.testing.assert_array_equal(got["SKY"].values, xds["SKY"].values)
        np.testing.assert_array_equal(got.attrs["extra"]["A"].values, np.arange(4.0))

    def test_caller_attrs_are_not_changed(self, tmp_path):
        """The writer encodes numpy arrays and datasets held in attrs (T3) on
        its own copy of the attrs, not on the caller's."""
        xds = single_image_dataset()
        extra = xr.Dataset({"A": ("row", np.arange(4.0))})
        xds.attrs["array"] = np.arange(3.0)
        xds.attrs["nested"] = {"array": np.ones(2)}
        xds.attrs["extra"] = extra
        xds["SKY"].attrs["array"] = np.arange(2)
        path = str(tmp_path / "attrs.img.zarr")
        write_image(xds, path, "zarr")
        assert isinstance(xds.attrs["array"], np.ndarray)
        assert isinstance(xds.attrs["nested"]["array"], np.ndarray)
        assert xds.attrs["extra"] is extra
        assert isinstance(xds["SKY"].attrs["array"], np.ndarray)
        got = open_image(path)
        np.testing.assert_array_equal(got.attrs["array"], np.arange(3.0))
        np.testing.assert_array_equal(got.attrs["nested"]["array"], np.ones(2))
        np.testing.assert_array_equal(got.attrs["extra"]["A"].values, np.arange(4.0))
        np.testing.assert_array_equal(got["SKY"].attrs["array"], np.arange(2))

    def test_chunk_size_guard(self, tmp_path):
        data = da.zeros((1, 1, 1, 16384, 16384), dtype=np.float32, chunks=-1)
        xds = xr.Dataset({"SKY": (_DIMS, data)})
        with pytest.raises(ValueError, match="too large for compression"):
            write_image(xds, str(tmp_path / "big.img.zarr"), "zarr")
        assert _listing(tmp_path) == []


# --------------------------------------------------------------------------- #
# TestOpenImageSelection                                                       #
# --------------------------------------------------------------------------- #


class TestOpenImageSelection:
    """open_image keeps a dimension selected with an integer, like
    load_image."""

    @pytest.mark.parametrize("index", [1, np.int64(1), -1])
    def test_integer_selection_keeps_the_dimension(self, tmp_path, index):
        path = str(tmp_path / "image.img.zarr")
        xds = single_image_dataset()
        write_image(xds, path, "zarr")
        for got in (
            open_image(path, selection={"frequency": index}),
            load_image(path, {"frequency": index}),
        ):
            assert got.sizes["frequency"] == 1
            np.testing.assert_array_equal(
                got["SKY"].values, xds["SKY"].isel(frequency=[1]).values
            )
