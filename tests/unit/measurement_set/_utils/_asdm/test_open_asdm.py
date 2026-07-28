import pytest

import numpy as np
import pandas as pd
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
        open_asdm("/unused_path/foo", [])


def test_open_asdm_with_mocked_set_from_file(mock_asdm_set_from_file, monkeypatch):

    def mock_load_times_from_partition_bdfs(
        bdf_paths: list[str], scans_metadata: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return np.array([0.1]), np.array([1.0]), np.array([0.101]), np.array([1.0]), {}

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
    check_datatree(ps_xdt)


def test_open_asdm_with_spw_simple_fails_some_partitions(
    mock_asdm_set_from_file, monkeypatch
):
    """
    Forces a failure when loading at least one partitoin, exercising partition-level error handling code in
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
    check_datatree(ps_xdt)
