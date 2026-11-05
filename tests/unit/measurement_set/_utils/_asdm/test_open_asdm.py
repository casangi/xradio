import numpy as np
import pandas as pd
import pytest
import xarray as xr

from xradio.measurement_set._utils._asdm.open_asdm import open_asdm
from xradio.schema.check import check_datatree


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
    with pytest.raises(RuntimeError, match="No partitions left"):
        open_asdm("/unused_path/foo", ["fieldId"])


def test_open_asdm_with_spw_default(mock_asdm_set_from_file, monkeypatch):
    monkeypatch.setattr(
        "xradio.measurement_set._utils._asdm.open_asdm.pyasdm.ASDM.setFromFile",
        mock_asdm_set_from_file,
    )
    monkeypatch.setattr(
        "pyasdm.MainRow.getBDFPath", lambda bdf_paths: "/monkypatched_path/foo"
    )
    with pytest.raises(RuntimeError, match="No partitions left"):
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
        ["dataDescriptionId", "execBlockId", "fieldId", "scanIntent"],
        include_processor_types=["CORRELATOR", "SPECTROMETER", "RADIOMETER"],
        with_pointing=True,
    )
    assert isinstance(ps_xdt, xr.DataTree)
    assert ps_xdt.type == "processing_set"
    for _msv4_idx, msv4_name in enumerate(ps_xdt):
        assert isinstance(msv4_name, str)
        assert "antenna_xds" in ps_xdt[msv4_name]
        assert "field_and_source_base_xds" in ps_xdt[msv4_name]
        assert "pointing_xds" in ps_xdt[msv4_name]
    ps_issues = check_datatree(ps_xdt)
    assert not ps_issues


def test_open_asdm_with_spw_simple_fails_some_partitions(
    mock_asdm_set_from_file, monkeypatch
):
    """
    Forces a failure when loading at least one partition, exercising partition-level error handling code in
    open_asdm
    """

    def mock_unreliable_load_times_from_partition_bdfs(
        bdf_paths: list[str], scans_metadata: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if mock_unreliable_load_times_from_partition_bdfs.partition_count == 0:
            raise RuntimeError("Some error when loading times from BDF")
        mock_unreliable_load_times_from_partition_bdfs.partition_count += 1
        return np.array([0.1]), np.array([1.0]), np.array([0.101]), np.array([1.0]), {}

    mock_unreliable_load_times_from_partition_bdfs.partition_count = 0

    monkeypatch.setattr(
        "xradio.measurement_set._utils._asdm.open_asdm.pyasdm.ASDM.setFromFile",
        mock_asdm_set_from_file,
    )
    monkeypatch.setattr(
        "pyasdm.MainRow.getBDFPath", lambda bdf_paths: "/monkypatched_path/foo"
    )
    monkeypatch.setattr(
        "xradio.measurement_set._utils._asdm.open_partition.load_times_from_partition_bdfs",
        mock_unreliable_load_times_from_partition_bdfs,
    )

    ps_xdt = open_asdm(
        "/unused_path/foo",
        ["dataDescriptionId", "execBlockId", "fieldId", "scanIntent"],
        include_processor_types=["CORRELATOR", "SPECTROMETER", "RADIOMETER"],
    )
    assert isinstance(ps_xdt, xr.DataTree)
    assert ps_xdt.type == "processing_set"
    for _msv4_name, msv4_xdt in enumerate(ps_xdt):
        assert isinstance(msv4_xdt, str)
    ps_issues = check_datatree(ps_xdt)
    assert not ps_issues
    # TODO: ensure the type of all main datasets is VisibilityXds


def test_open_asdm_sd_with_execblock_antenna_station_feed(
    asdm_sd_with_execblock_antenna_station_feed,
    mock_sd_asdm_set_from_file,
    monkeypatch,
):
    """simple test with an SD asdm input"""

    monkeypatch.setattr(
        "xradio.measurement_set._utils._asdm.open_asdm.pyasdm.ASDM.setFromFile",
        mock_sd_asdm_set_from_file,
    )
    monkeypatch.setattr(
        "pyasdm.MainRow.getBDFPath", lambda bdf_paths: "/monkypatched_path/foo"
    )
    monkeypatch.setattr(
        "xradio.measurement_set._utils._asdm.open_partition.load_times_from_partition_bdfs",
        mock_load_times_from_partition_bdfs,
    )

    ps_xdt = open_asdm(
        "/test_unused_path/foo/sd_asdm",
        ["dataDescriptionId", "execBlockId", "fieldId", "scanIntent"],
        include_processor_types=["CORRELATOR", "SPECTROMETER", "RADIOMETER"],
    )

    assert isinstance(ps_xdt, xr.DataTree)
    assert ps_xdt.type == "processing_set"
    for msv4_name in ps_xdt:
        msv4_xds = ps_xdt[msv4_name].ds
        assert msv4_xds.attrs["type"] == "spectrum"
        assert msv4_xds.attrs["data_groups"]["base"]["correlated_data"] == "SPECTRUM"
        for variable_name in ["TIME_CENTROID", "EFFECTIVE_INTEGRATION_TIME"]:
            assert msv4_xds.data_vars[variable_name].dims == ["time", "antenna_name"]

    ps_issues = check_datatree(ps_xdt)
    assert not ps_issues


def mock_asdm_with_frequency_centers(set_from_file, monkeypatch, frequency_centers):
    monkeypatch.setattr(
        "xradio.measurement_set._utils._asdm.open_asdm.pyasdm.ASDM.setFromFile",
        set_from_file,
    )
    monkeypatch.setattr(
        "pyasdm.MainRow.getBDFPath", lambda bdf_paths: "/monkypatched_path/foo"
    )
    monkeypatch.setattr(
        "xradio.measurement_set._utils._asdm.open_partition.load_times_from_partition_bdfs",
        mock_load_times_from_partition_bdfs,
    )
    monkeypatch.setattr(
        "xradio.measurement_set._utils._asdm.open_partition.get_spw_frequency_centers",
        lambda *_args: frequency_centers,
    )


def test_open_asdm_not_monotonic_axis_raises(mock_asdm_set_from_file, monkeypatch):
    """Not one of the partition failures that open_asdm logs and goes past."""
    mock_asdm_with_frequency_centers(
        mock_asdm_set_from_file, monkeypatch, np.array([1.0, 3.0, 2.0])
    )
    with pytest.raises(ValueError, match="frequency axis .* is not monotonic"):
        open_asdm("/unused_path/foo", include_processor_types=["RADIOMETER"])


@pytest.mark.parametrize(
    "chunks, frequency_chunks",
    [(None, None), ({}, (5,)), ({"frequency": 2}, (2, 2, 1))],
)
def test_open_datatree_reverses_decreasing_frequency(
    mock_asdm_set_from_file, monkeypatch, chunks, frequency_chunks
):
    mock_asdm_with_frequency_centers(
        mock_asdm_set_from_file, monkeypatch, np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    )
    stored = np.broadcast_to(
        np.arange(5, dtype="complex128")[:, np.newaxis], (1, 2, 5, 2)
    )
    monkeypatch.setattr(
        "xradio.measurement_set._utils._asdm.asdm_backend_arrays."
        "load_visibilities_from_partition_bdfs",
        lambda *args: stored[args[-1]],
    )
    ps_xdt = xr.open_datatree(
        "/unused_path/foo",
        engine="xradio_asdm",
        chunks=chunks,
        include_processor_types=["RADIOMETER"],
    )
    msv4_xdt = next(iter(ps_xdt.children.values()))
    assert msv4_xdt.VISIBILITY.chunksizes.get("frequency") == frequency_chunks
    assert msv4_xdt.frequency.values.tolist() == [1.0, 2.0, 3.0, 4.0, 5.0]
    np.testing.assert_array_equal(msv4_xdt.VISIBILITY, stored[:, :, ::-1])
