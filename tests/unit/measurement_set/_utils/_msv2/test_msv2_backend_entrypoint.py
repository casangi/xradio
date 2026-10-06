"""
Tests of the ``xradio_msv2`` engine's entry point class
(``_xradio_xarray_backends.MSv2BackendEntrypoint``): what it claims, its
errors, the forwarding of its options and the light entry module.

The engine is passed as a class, so the tests do not depend on the entry
points of the installed distribution.
"""

import os
import pathlib
import subprocess
import sys
import textwrap

import pytest
import xarray as xr

import _xradio_xarray_backends
from _xradio_xarray_backends import MSv2BackendEntrypoint

_ENGINE = MSv2BackendEntrypoint
_SUBTABLES = ("ANTENNA", "DATA_DESCRIPTION", "POLARIZATION", "SPECTRAL_WINDOW")
_SRC = str(pathlib.Path(_xradio_xarray_backends.__file__).resolve().parent)


def _table(path: pathlib.Path, first_line: str | None, subtables=()) -> str:
    """A fake casacore table directory: table.info starting with
    ``first_line`` (no table.info if None), and sub-table directories with a
    table.dat."""
    path.mkdir(parents=True)
    (path / "table.dat").write_bytes(b"")
    if first_line is not None:
        (path / "table.info").write_text(f"{first_line}\nSubType = \n\n")
    for sub in subtables:
        (path / sub).mkdir()
        (path / sub / "table.dat").write_bytes(b"")
    return str(path)


GUESS_CASES = {
    # claimed
    "ms": ("Type = Measurement Set", (), True),
    "ms_without_subtables": ("Type = Measurement Set", (), True),
    "empty_type_with_subtables": ("Type =", _SUBTABLES, True),
    "empty_info_with_subtables": ("", _SUBTABLES, True),
    # not claimed
    "empty_type_without_subtables": ("Type =", (), False),
    "empty_type_three_subtables": ("Type =", _SUBTABLES[:3], False),
    "casa_image": ("Type = Image", (), False),
    "caltable": ("Type = Calibration", (), False),
    "no_table_info": (None, _SUBTABLES, False),
}


@pytest.mark.parametrize("case", list(GUESS_CASES))
def test_guess_can_open(case, tmp_path):
    first_line, subtables, claimed = GUESS_CASES[case]
    path = _table(tmp_path / f"{case}.ms", first_line, subtables)
    assert _ENGINE().guess_can_open(path) is claimed
    assert _ENGINE().guess_can_open(pathlib.Path(path)) is claimed


def test_guess_can_open_generated_ms(backend_ms, tmp_path):
    """A generated MS is claimed, also in a directory and with ~ and a
    PathLike; its sub-tables and the directory holding it are not."""
    msname = backend_ms("dense")
    engine = _ENGINE()
    assert engine.guess_can_open(msname)
    assert engine.guess_can_open(pathlib.Path(msname))
    assert not engine.guess_can_open(os.path.join(msname, "ANTENNA"))
    assert not engine.guess_can_open(os.path.join(msname, "FIELD"))
    assert not engine.guess_can_open(os.path.dirname(msname))
    nested = tmp_path / "data" / "obs"
    nested.mkdir(parents=True)
    os.symlink(msname, nested / "linked.ms")
    assert engine.guess_can_open(str(nested / "linked.ms"))


def test_guess_can_open_rejects_other_paths(tmp_path):
    engine = _ENGINE()
    zarr_store = tmp_path / "converted.ms"  # a zarr store named .ms
    zarr_store.mkdir()
    (zarr_store / "zarr.json").write_text("{}")
    assert not engine.guess_can_open(str(zarr_store))
    asdm = tmp_path / "uid___A002_X1"
    asdm.mkdir()
    (asdm / "ASDM.xml").write_text("<ASDM/>")
    assert not engine.guess_can_open(str(asdm))
    a_file = tmp_path / "file.ms"
    a_file.write_text("x")
    assert not engine.guess_can_open(str(a_file))
    assert not engine.guess_can_open(str(tmp_path / "missing.ms"))
    for obj in (None, 3, b"bytes.ms", object()):
        assert not engine.guess_can_open(obj)


@pytest.mark.skipif(
    hasattr(os, "geteuid") and os.geteuid() == 0, reason="root reads everything"
)
def test_guess_can_open_permission_error_propagates(tmp_path):
    path = _table(tmp_path / "locked.ms", "Type = Measurement Set")
    os.chmod(os.path.join(path, "table.info"), 0)
    try:
        with pytest.raises(PermissionError):
            _ENGINE().guess_can_open(path)
    finally:
        os.chmod(os.path.join(path, "table.info"), 0o644)


def test_errors(tmp_path):
    """Errors in order: not a path, missing path, not an MS (with what it is),
    open_dataset of an MS."""
    with pytest.raises(TypeError, match="local path"):
        xr.open_datatree(object(), engine=_ENGINE)
    with pytest.raises(FileNotFoundError):
        xr.open_datatree(str(tmp_path / "missing.ms"), engine=_ENGINE)
    image = _table(tmp_path / "image.im", "Type = Image")
    with pytest.raises(ValueError, match="CASA image"):
        xr.open_datatree(image, engine=_ENGINE)
    store = tmp_path / "ps.zarr"
    store.mkdir()
    (store / "zarr.json").write_text("{}")
    with pytest.raises(ValueError, match="open_processing_set"):
        xr.open_datatree(str(store), engine=_ENGINE)
    other = _table(tmp_path / "other", "Type = Calibration")
    with pytest.raises(ValueError, match="is not a MeasurementSet v2"):
        xr.open_datatree(other, engine=_ENGINE)
    ms = _table(tmp_path / "x.ms", "Type = Measurement Set")
    with pytest.raises(NotImplementedError, match="open_datatree"):
        xr.open_dataset(ms, engine=_ENGINE)
    with pytest.raises(FileNotFoundError):
        xr.open_dataset(str(tmp_path / "missing.ms"), engine=_ENGINE)


@pytest.fixture
def driver_calls(monkeypatch):
    """Replace the driver: record its calls, return an empty tree."""
    calls = []

    def fake_driver(path, **kwargs):
        calls.append((path, kwargs))
        tree = xr.DataTree(
            xr.Dataset(coords={"y": [5]}, attrs={"type": "processing_set"})
        )
        tree["child"] = xr.DataTree(xr.Dataset({"a": ("x", [1, 2])}, {"x": [3, 4]}))
        return tree

    monkeypatch.setattr(_xradio_xarray_backends, "_msv2_driver", lambda: fake_driver)
    return calls


DEFAULTS = {
    "drop_variables": None,
    "partition_scheme": None,
    "partition_filter": None,
    "main_chunksize": None,
    "with_pointing": True,
    "pointing_chunksize": None,
    "pointing_interpolate": False,
    "ephemeris_interpolate": False,
    "phase_cal_interpolate": False,
    "sys_cal_interpolate": False,
    "partition_cache": None,
    "on_partition_error": "skip",
    "skip_columns": None,
}


def test_options_are_forwarded_unchanged(driver_calls, tmp_path):
    ms = _table(tmp_path / "x.ms", "Type = Measurement Set")
    xr.open_datatree(ms, engine=_ENGINE)
    assert driver_calls == [(ms, DEFAULTS)]
    options = {
        "drop_variables": ["FLAG"],
        "partition_scheme": ["FIELD_ID"],
        "partition_filter": len,
        "main_chunksize": {"time": 3},
        "with_pointing": False,
        "pointing_chunksize": 0.5,
        "pointing_interpolate": True,
        "ephemeris_interpolate": True,
        "phase_cal_interpolate": True,
        "sys_cal_interpolate": True,
        "partition_cache": "read",
        "on_partition_error": "raise",
        "skip_columns": ["MODEL_DATA"],
    }
    xr.open_datatree(pathlib.Path(ms), engine=_ENGINE, chunks={}, **options)
    assert driver_calls[-1] == (ms, options)
    # decode_cf=False passes no decoder (explicit open_dataset_parameters)
    xr.open_datatree(ms, engine=_ENGINE, decode_cf=False)
    assert driver_calls[-1] == (ms, DEFAULTS)
    for unknown in ({"unknown_option": 1}, {"mask_and_scale": False}):
        with pytest.raises(TypeError, match=next(iter(unknown))):
            xr.open_datatree(ms, engine=_ENGINE, **unknown)
    # ~ is expanded
    home_ms = os.path.join("~", os.path.relpath(ms, os.path.expanduser("~")))
    if not home_ms.startswith(os.path.join("~", "..")):
        xr.open_datatree(home_ms, engine=_ENGINE)
        assert driver_calls[-1][0] == ms


def test_open_groups_has_no_inherited_coordinates(driver_calls, tmp_path):
    ms = _table(tmp_path / "x.ms", "Type = Measurement Set")
    groups = xr.open_groups(ms, engine=_ENGINE)
    assert sorted(groups) == ["/", "/child"]
    assert "type" in groups["/"].attrs and list(groups["/"].coords) == ["y"]
    # the child's own coordinates only (DataTree.to_dict() would add "y")
    assert list(groups["/child"].coords) == ["x"]


def test_guess_engine_picks_the_msv2_engine(backend_ms, monkeypatch):
    """An engine-less open_datatree of an MS picks xradio_msv2 among the
    xradio engines and zarr (list_engines monkeypatched: the entry points
    may not be installed)."""
    from xarray.backends import plugins

    from _xradio_xarray_backends import (
        CasaImageBackendEntrypoint,
        FitsImageBackendEntrypoint,
    )

    engines = {
        "xradio_casa_image": CasaImageBackendEntrypoint(),
        "xradio_fits_image": FitsImageBackendEntrypoint(),
        "xradio_msv2": MSv2BackendEntrypoint(),
        "zarr": plugins.list_engines()["zarr"],
    }
    monkeypatch.setattr(plugins, "list_engines", lambda: engines)
    msname = backend_ms("dense")
    assert plugins.guess_engine(msname, must_support_groups=True) == "xradio_msv2"
    assert plugins.guess_engine(msname) == "xradio_msv2"
    assert not engines["xradio_casa_image"].guess_can_open(msname)


def _run_python(code: str, cwd, *options: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [_SRC] + [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p]
    )
    return subprocess.run(
        [sys.executable, *options, "-c", textwrap.dedent(code)],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=600,
        env=env,
    )


def test_loading_and_guessing_imports_no_reader(backend_ms, tmp_path):
    """Loading the entry point and guessing (also on an MS), as an
    engine-less open does, imports none of xradio, zarr, dask, astropy,
    casacore or casatools."""
    result = _run_python(
        f"""
        import sys
        from importlib.metadata import EntryPoint

        entry_point = EntryPoint(
            name="xradio_msv2",
            value="_xradio_xarray_backends:MSv2BackendEntrypoint",
            group="xarray.backends",
        )
        backend = entry_point.load()()
        assert backend.guess_can_open({backend_ms("dense")!r})
        assert not backend.guess_can_open("missing.ms")
        heavy = [
            name
            for name in ("xradio", "zarr", "dask", "astropy", "casacore", "casatools")
            if name in sys.modules
        ]
        assert not heavy, heavy
        """,
        tmp_path,
    )
    assert result.returncode == 0, result.stderr


def test_entry_point_loads_on_an_xarray_only_install(backend_ms, tmp_path):
    """With the reader dependencies blocked, the entry point loads and
    guesses without any warning (warnings as errors), and an open says
    which package is missing."""
    result = _run_python(
        f"""
        import sys

        for name in (
            "dask", "astropy", "zarr", "numcodecs", "toolviper", "distributed",
            "casacore", "casatools", "casaconfig",
        ):
            sys.modules[name] = None

        import xarray as xr
        from _xradio_xarray_backends import MSv2BackendEntrypoint

        msname = {backend_ms("dense")!r}
        assert MSv2BackendEntrypoint().guess_can_open(msname)
        try:
            xr.open_datatree(msname, engine=MSv2BackendEntrypoint)
        except ImportError as exc:
            assert "python-casacore or casatools" in str(exc), exc
        else:
            raise AssertionError("no ImportError")
        """,
        tmp_path,
        "-W",
        "error",
    )
    assert result.returncode == 0, result.stderr


def test_class_attributes():
    assert _ENGINE.supports_groups is True
    assert _ENGINE.open_dataset_parameters == ("filename_or_obj", "drop_variables")
    assert _ENGINE.description and _ENGINE.url.startswith("https://")
    assert "MSv2BackendEntrypoint" in _xradio_xarray_backends.__all__
    import importlib.util

    import xradio.measurement_set as ms
    from xradio.measurement_set._utils._msv2 import open_msv2 as module

    # (a private module, as the ASDM backend's: xradio.measurement_set.open_msv2
    # is only the function)
    assert importlib.util.find_spec("xradio.measurement_set.open_msv2") is None
    assert ms.open_msv2 is module.open_msv2 and callable(ms.open_msv2)
    assert module.MSv2BackendEntrypoint is _ENGINE
    for name in module.__all__:
        assert getattr(ms, name) is getattr(module, name), name


def test_pyproject_registers_the_light_class():
    import tomllib

    root = pathlib.Path(__file__).resolve().parents[5]
    with open(root / "pyproject.toml", "rb") as pyproject:
        entry_points = tomllib.load(pyproject)["project"]["entry-points"][
            "xarray.backends"
        ]
    assert (
        entry_points["xradio_msv2"] == "_xradio_xarray_backends:MSv2BackendEntrypoint"
    )
