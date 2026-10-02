import contextlib
import io
import logging
import os

import numpy as np
import pandas as pd
import pyasdm
import pytest
import xarray as xr
from astropy.coordinates import SkyCoord

from xradio.measurement_set._utils._asdm import create_pointing_xds as pointing_module
from xradio.measurement_set._utils._asdm import open_asdm as open_asdm_module
from xradio.measurement_set._utils._asdm._utils.calculate_uvw import calculate_uvw
from xradio.measurement_set._utils._asdm.open_asdm import open_asdm
from xradio.schema.check import check_datatree

MSV4_TYPES = {"visibility", "spectrum", "radiometer"}


def test_open_asdm_none():
    with pytest.raises(TypeError, match="expected"):
        open_asdm(None, ["fieldId"])


def test_open_asdm_empty(asdm_empty, monkeypatch):
    monkeypatch.setattr(
        "xradio.measurement_set._utils._asdm.open_asdm.pyasdm.ASDM.setFromFile",
        lambda self, asdm_path: None,
    )
    monkeypatch.setattr(
        "pyasdm.MainRow.getBDFPath", lambda bdf_paths: "/monkypatched_path/foo"
    )
    with pytest.raises(RuntimeError, match="No partitions"):
        open_asdm("/unused_path/foo", ["fieldId"])


def test_open_asdm_with_spw_default(mock_asdm_set_from_file, monkeypatch):
    monkeypatch.setattr(
        "xradio.measurement_set._utils._asdm.open_asdm.pyasdm.ASDM.setFromFile",
        mock_asdm_set_from_file,
    )
    monkeypatch.setattr(
        "pyasdm.MainRow.getBDFPath", lambda bdf_paths: "/monkypatched_path/foo"
    )
    with pytest.raises(RuntimeError, match="No partitions"):
        open_asdm("/unused_path/foo", [], include_processor_types=["SPECTROMETER"])


def mock_load_times_from_partition_bdfs(
    bdf_paths: list[str], scans_metadata: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return np.array([0.1]), np.array([1.0]), np.array([0.101]), np.array([1.0]), {}


def test_open_asdm_with_mocked_set_from_file(mock_asdm_set_from_file, monkeypatch):
    monkeypatch.setattr(
        "xradio.measurement_set._utils._asdm.open_asdm.pyasdm.ASDM.setFromFile",
        mock_asdm_set_from_file,
    )
    monkeypatch.setattr(
        "pyasdm.MainRow.getBDFPath", lambda bdf_paths: "/monkypatched_path/foo"
    )
    monkeypatch.setattr(
        "xradio.measurement_set._utils._asdm.open_partition.load_times_from_partition_bdfs",
        mock_load_times_from_partition_bdfs,
    )

    ps_xdt = open_asdm(
        "/unused_path/foo",
        ["fieldId"],
        include_processor_types=["CORRELATOR", "SPECTROMETER", "RADIOMETER"],
    )
    assert isinstance(ps_xdt, xr.DataTree)
    assert ps_xdt.type == "processing_set"
    assert len(ps_xdt.children) > 0
    for msv4_name, msv4_xdt in ps_xdt.children.items():
        assert msv4_name.startswith("foo_")
        assert msv4_xdt.attrs["type"] in MSV4_TYPES
        assert "antenna_xds" in msv4_xdt
        assert "field_and_source_base_xds" in msv4_xdt
    ps_issues = check_datatree(ps_xdt)
    assert not ps_issues


@pytest.fixture
def open_partition_failing_for(monkeypatch):
    """Make open_partition raise for the given call indices (other calls open the
    partition normally)."""

    def _patch(failing_calls: set[int], exception: Exception):
        original = open_asdm_module.open_partition
        calls = []

        def unreliable_open_partition(*args, **kwargs):
            calls.append(len(calls))
            if calls[-1] in failing_calls:
                raise exception
            return original(*args, **kwargs)

        monkeypatch.setattr(
            open_asdm_module, "open_partition", unreliable_open_partition
        )
        return calls

    return _patch


@pytest.mark.parametrize(
    "exception",
    [
        RuntimeError("Some error when opening the partition"),
        ValueError("bad value"),
        KeyError("missing_key"),
        NotImplementedError("unsupported"),
    ],
)
def test_open_asdm_skips_failed_partitions(
    synth_interferometric, open_partition_failing_for, exception
):
    """A partition whose open fails (any exception type) is skipped: no empty
    node, the other partitions are complete typed MSv4s and the processing-set
    accessors work (F39, F48, K13)."""
    truth = synth_interferometric
    expected = truth.expected_partitions()
    assert len(expected) == 3
    calls = open_partition_failing_for({0}, exception)

    ps_xdt = open_asdm(truth.path)

    assert len(calls) == len(expected)
    asdm_name = os.path.basename(truth.path)
    assert sorted(ps_xdt.children) == [f"{asdm_name}_1", f"{asdm_name}_2"]
    for msv4_xdt in ps_xdt.children.values():
        assert msv4_xdt.attrs["type"] == "visibility"
        assert "VISIBILITY" in msv4_xdt.ds.data_vars
        assert (
            msv4_xdt.ds.attrs["data_groups"]["base"]["correlated_data"] == "VISIBILITY"
        )
        assert "antenna_xds" in msv4_xdt
        assert "field_and_source_base_xds" in msv4_xdt
    assert not check_datatree(ps_xdt)
    summary = ps_xdt.xr_ps.summary()
    assert len(summary) == 2


def test_open_asdm_all_partitions_fail(
    synth_interferometric, open_partition_failing_for
):
    """If no partition can be opened, RuntimeError (K13)."""
    calls = open_partition_failing_for({0, 1, 2}, ValueError("always failing"))
    with pytest.raises(RuntimeError, match="None of the 3 partitions"):
        open_asdm(synth_interferometric.path)
    assert len(calls) == 3


def test_open_asdm_partitions_and_names(synth_interferometric):
    """One MSv4 node per expected partition (one per SPW), named
    {asdm_name}_{index}, each with its sub-datasets and a clean schema check."""
    truth = synth_interferometric
    ps_xdt = open_asdm(truth.path)
    asdm_name = os.path.basename(truth.path)
    expected = truth.expected_partitions()
    assert sorted(ps_xdt.children) == [
        f"{asdm_name}_{idx}" for idx in range(len(expected))
    ]
    spw_ids = sorted(
        truth.spw_id_from_frequency(node.ds.frequency.values)
        for node in ps_xdt.children.values()
    )
    assert spw_ids == sorted(part.spw_id for part in expected)
    for node in ps_xdt.children.values():
        assert node.attrs["type"] == "visibility"
        assert "UVW" in node.ds.data_vars
    assert not check_datatree(ps_xdt)


def test_open_asdm_partition_scheme_passed_through(synth_interferometric, monkeypatch):
    """partition_scheme is passed unchanged to create_partitions (None by default,
    [] stays [] = no optional axes) (F35, K7), as are the type filters."""
    received = []
    original = open_asdm_module.create_partitions

    def spy_create_partitions(asdm, partition_scheme, *args, **kwargs):
        received.append((partition_scheme, args))
        return original(asdm, partition_scheme, *args, **kwargs)

    monkeypatch.setattr(open_asdm_module, "create_partitions", spy_create_partitions)
    open_asdm(synth_interferometric.path)
    open_asdm(synth_interferometric.path, partition_scheme=[])
    open_asdm(
        synth_interferometric.path, ["scanNumber"], ["CORRELATOR"], ["FULL_RESOLUTION"]
    )
    assert received == [
        (None, (["CORRELATOR", "SPECTROMETER"], ["FULL_RESOLUTION", "BASEBAND_WIDE"])),
        ([], (["CORRELATOR", "SPECTROMETER"], ["FULL_RESOLUTION", "BASEBAND_WIDE"])),
        (["scanNumber"], (["CORRELATOR"], ["FULL_RESOLUTION"])),
    ]


def test_open_asdm_relative_path_is_made_absolute(
    synth_interferometric, monkeypatch, tmp_path
):
    """The ASDM is opened from an absolute path, so the lazily loaded data still
    loads after the working directory changes (F78)."""
    truth = synth_interferometric
    opened_paths = []
    original_set_from_file = pyasdm.ASDM.setFromFile

    def spy_set_from_file(self, path, *args, **kwargs):
        opened_paths.append(path)
        return original_set_from_file(self, path, *args, **kwargs)

    monkeypatch.setattr(pyasdm.ASDM, "setFromFile", spy_set_from_file)

    reference = {
        name: node.ds.VISIBILITY.values
        for name, node in open_asdm(truth.path).children.items()
    }
    parent, name = os.path.split(truth.path)
    monkeypatch.chdir(parent)
    ps_xdt = open_asdm(name)
    assert len(opened_paths) == 2
    assert opened_paths[0] == os.path.abspath(truth.path)
    assert os.path.isabs(opened_paths[1])
    assert os.path.samefile(opened_paths[1], truth.path)
    monkeypatch.chdir(tmp_path)
    assert sorted(ps_xdt.children) == sorted(reference)
    for msv4_name, node in ps_xdt.children.items():
        np.testing.assert_array_equal(node.ds.VISIBILITY.values, reference[msv4_name])


def test_open_asdm_single_dish(synth_single_dish_simple):
    """An all-AUTO_ONLY ASDM gives SpectrumXds nodes (K12, F48)."""
    truth = synth_single_dish_simple
    ps_xdt = open_asdm(truth.path)
    assert ps_xdt.type == "processing_set"
    assert len(ps_xdt.children) == len(truth.expected_partitions()) == 2
    for node in ps_xdt.children.values():
        assert node.attrs["type"] == "spectrum"
        assert "SPECTRUM" in node.ds.data_vars
        assert "UVW" not in node.ds.data_vars
        assert node.ds.SPECTRUM.dims == (
            "time",
            "antenna_name",
            "frequency",
            "polarization",
        )
    assert not check_datatree(ps_xdt)


def test_open_asdm_with_pointing(synth_interferometric_pointing):
    """with_pointing / pointing_for_only_spectral_resolution_types are passed on to
    the partitions: every MSv4 gets a pointing_xds only when requested for its
    spectral resolution type."""
    truth = synth_interferometric_pointing
    ps_xdt = open_asdm(truth.path, with_pointing=True)
    assert len(ps_xdt.children) == len(truth.expected_partitions())
    for node in ps_xdt.children.values():
        assert "pointing_xds" in node
        assert node["pointing_xds"].ds.sizes["antenna_name"] == len(truth.antenna_names)
    assert not check_datatree(ps_xdt)

    for kwargs in (
        {},
        {
            "with_pointing": True,
            "pointing_for_only_spectral_resolution_types": ["CHANNEL_AVERAGE"],
        },
    ):
        ps_xdt = open_asdm(truth.path, **kwargs)
        assert ps_xdt.children
        for node in ps_xdt.children.values():
            assert "pointing_xds" not in node


def test_open_asdm_single_dish_with_pointing(
    synthetic_asdm_module, make_synthetic_asdm
):
    """Single dish with with_pointing=True and an antenna (DA42) without Pointing
    rows: every partition opens, with a pointing_xds that has the antennas of the
    partition (NaN for DA42). The pointing_xds did not align with the
    antenna_name index of the SpectrumXds before, so every partition was
    skipped."""
    spec = synthetic_asdm_module.single_dish_spec(
        name="uid___A002_X1234_X56c2", with_off_and_calibration=False
    )
    spec.with_pointing = True
    spec.pointing_antennas = (0, 2)
    truth = make_synthetic_asdm(spec)
    ps_xdt = open_asdm(truth.path, with_pointing=True)
    assert len(ps_xdt.children) == len(truth.expected_partitions()) == 2
    for node in ps_xdt.children.values():
        assert node.attrs["type"] == "spectrum"
        pxds = node["pointing_xds"].ds
        assert list(map(str, pxds.antenna_name.values)) == truth.antenna_names
        all_nan = np.isnan(pxds.POINTING_BEAM.values).all(axis=(0, 2))
        np.testing.assert_array_equal(all_nan, [False, True, False])
    assert not check_datatree(ps_xdt)


def write_unconvertible_pointing_table(path: str, as_bin: bool) -> None:
    """Rewrite the Pointing table of an ASDM (XML or binary) with every row
    flagged usePolynomials=false and 3 target values (numTerm=3) for the
    numSample=4 of the last row: inconsistent, or (when usePolynomials cannot be
    read, XML with pyasdm <= 0.0.7) indistinguishable from a polynomial."""
    asdm = pyasdm.ASDM()
    with contextlib.redirect_stdout(io.StringIO()):
        asdm.setFromFile(path)
    table = asdm.getPointing()
    rows = table.get()
    for row in rows:
        row.setUsePolynomials(False)
    assert rows[-1].getNumSample() == 4
    rows[-1].setTarget(rows[-1].getTarget()[:3])
    rows[-1].setNumTerm(3)
    table._fileAsBin = as_bin
    for name in ("Pointing.xml", "Pointing.bin"):
        if os.path.exists(os.path.join(path, name)):
            os.remove(os.path.join(path, name))
    with contextlib.redirect_stdout(io.StringIO()):
        table.toFile(path)
    with open(os.path.join(path, "Pointing.xml")) as xml_file:
        assert ("<BulkStoreRef" in xml_file.read(4096)) == as_bin


@pytest.mark.parametrize("as_bin", [False, True])
def test_open_asdm_with_unconvertible_pointing_table(
    synthetic_asdm_module, make_synthetic_asdm, monkeypatch, caplog, as_bin
):
    """A Pointing table that cannot be converted only costs the pointing_xds:
    with_pointing=True opens every partition, without pointing_xds, the same way
    for XML and binary tables (the binary one made every partition fail, so the
    open failed). The table is converted, and the error logged, only once."""
    truth = make_synthetic_asdm(
        synthetic_asdm_module.interferometric_spec(
            with_pointing=True, name=f"uid___A002_X1234_X56c{3 + as_bin}"
        )
    )
    write_unconvertible_pointing_table(truth.path, as_bin)
    builds = []
    original_build = pointing_module._build_full_pointing_xds

    def counting_build(*args, **kwargs):
        builds.append(args[0])
        return original_build(*args, **kwargs)

    monkeypatch.setattr(pointing_module, "_build_full_pointing_xds", counting_build)
    with caplog.at_level(logging.INFO):
        ps_xdt = open_asdm(truth.path, with_pointing=True)
    num_partitions = len(truth.expected_partitions())
    assert len(ps_xdt.children) == num_partitions == 3
    for node in ps_xdt.children.values():
        assert "pointing_xds" not in node
        assert "VISIBILITY" in node.ds
    assert len(builds) == 1
    errors = [rec for rec in caplog.records if rec.levelno >= logging.ERROR]
    assert len(errors) == 1
    assert "Pointing table (16 rows) cannot be converted" in errors[0].message
    assert "Skipping partition" not in caplog.text
    assert caplog.text.count("will not have a pointing_xds") == num_partitions
    assert not check_datatree(ps_xdt)


@pytest.mark.parametrize("unsupported_code", ["HADEC", "AZELNE", "ITRF"])
def test_open_asdm_phase_center_frames(
    synthetic_asdm_module, make_synthetic_asdm, unsupported_code
):
    """A field whose phase center frame the UVW calculation does not support
    (Earth-fixed / topocentric directions) makes its partition fail when the ASDM
    is opened: the partition is skipped (K13) and the processing set that is
    returned can be computed entirely (UVW included). A SUPERGAL phase center is
    supported: its UVW are those of the direction converted to ICRS."""
    spec = synthetic_asdm_module.mosaic_spec()
    spec.fields[1].direction_code = unsupported_code
    spec.fields[2].direction_code = "SUPERGAL"
    truth = make_synthetic_asdm(spec)
    expected = truth.expected_partitions()
    expected_fields = sorted(
        truth.make_field_name(field_id)
        for part in expected
        for field_id in part.field_ids
        if field_id != 1
    )

    ps_xdt = open_asdm(truth.path)

    opened_fields = sorted(
        str(name)
        for node in ps_xdt.children.values()
        for name in np.unique(node.ds.field_name.values)
    )
    assert opened_fields == expected_fields
    assert truth.make_field_name(1) not in opened_fields
    assert len(ps_xdt.children) == len(expected) - 1
    assert not check_datatree(ps_xdt)

    for node in ps_xdt.children.values():
        uvw = node.ds.UVW.values
        assert np.isfinite(uvw).all()
        field_and_source = node["field_and_source_base_xds"].ds
        phase_center = field_and_source.FIELD_PHASE_CENTER_DIRECTION
        if truth.make_field_name(2) not in node.ds.field_name.values:
            assert phase_center.attrs["frame"] == "icrs"
            continue
        assert phase_center.attrs["frame"] == "supergalactic"
        lon, lat = phase_center.values[0]
        icrs = SkyCoord(lon, lat, unit="rad", frame="supergalactic").icrs
        icrs_center = phase_center.copy(data=[[icrs.ra.rad, icrs.dec.rad]])
        icrs_center.attrs["frame"] = "icrs"
        expected_uvw = calculate_uvw(
            None,
            node.ds.time,
            node.ds.baseline_antenna1_name,
            node.ds.baseline_antenna2_name,
            node["antenna_xds"].ds.ANTENNA_POSITION,
            icrs_center,
        )
        np.testing.assert_allclose(uvw, expected_uvw, rtol=0, atol=1e-6)


def test_open_asdm_unsupported_phase_center_frame_everywhere(
    synthetic_asdm_module, make_synthetic_asdm
):
    """When every interferometric partition has a phase center in an unsupported
    frame, open_asdm fails (RuntimeError, K13) instead of returning a processing
    set whose UVW cannot be computed."""
    spec = synthetic_asdm_module.interferometric_spec()
    spec.fields[0].direction_code = "AZELNE"
    truth = make_synthetic_asdm(spec)
    with pytest.raises(RuntimeError, match="None of the 3 partitions"):
        open_asdm(truth.path)
