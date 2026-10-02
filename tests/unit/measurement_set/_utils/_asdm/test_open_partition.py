"""
Tests of open_partition (correlated dataset, coordinates, data variables and the
sub-datasets of one MSv4 opened from an ASDM partition).

Most tests use the synthetic on-disk ASDMs of ``synthetic_asdm.py`` (real pyasdm
tables and MIME BDFs, see the ``synth_*`` fixtures in conftest.py). The partition
descriptions are built here from the writer's truth (independently of
create_partitions), following the shared contract of create_partitions (K7):
1-D arrays of IDs, "BDFPath" in Main time order and "per_bdf" values aligned with
"BDFPath". Expected values come from the writer's parameters.
"""

import contextlib
import importlib.util
import io
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyasdm
import pytest
import xarray as xr

from xradio.measurement_set._utils._asdm import asdm_backend_arrays
from xradio.measurement_set._utils._asdm import open_partition as op
from xradio.measurement_set._utils._asdm._utils.time import (
    ASDM_TIME_FORMAT,
    ASDM_TIME_SCALE,
)
from xradio.measurement_set._utils._asdm.create_pointing_xds import (
    PointingConversionError,
)
from xradio.measurement_set.schema import UvwArray
from xradio.schema.check import (
    check_attributes,
    check_datatree,
    check_dimensions,
    check_dtype,
    xarray_dataclass_to_array_schema,
)


def _load_synthetic_asdm():
    """Load synthetic_asdm.py (same module object as the conftest fixture)."""
    name = "xradio_tests_asdm_synthetic_asdm"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            name, Path(__file__).with_name("synthetic_asdm.py")
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


synth = _load_synthetic_asdm()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def read_asdm(path: str) -> pyasdm.ASDM:
    """A fresh pyasdm.ASDM object (own table cache) read from disk."""
    asdm = pyasdm.ASDM()
    with contextlib.redirect_stdout(io.StringIO()):
        asdm.setFromFile(path)
    return asdm


def partition_descr_for(truth, part) -> dict:
    """
    Partition description of an expected partition (synthetic_asdm
    ExpectedPartition), as create_partitions produces it (contract K7).
    """
    cfg = truth.config(part.config_idx)
    bdfs = part.bdfs
    bdf_paths = np.array([bdf.path for bdf in bdfs], dtype=str)
    return {
        "execBlockId": np.array([0]),
        "configDescriptionId": np.array([part.config_idx]),
        "dataDescriptionId": np.array([part.dd_id]),
        "scanIntent": np.array(part.obs_modes, dtype=str),
        "fieldId": np.array(part.field_ids),
        "scanNumber": np.array(part.scan_numbers),
        "subscanNumber": np.array(part.subscan_numbers),
        "stateId": np.array([0]),
        "spectralType": np.array([cfg.spectral_type]),
        "BDFPath": bdf_paths,
        "per_bdf": {
            "BDFPath": bdf_paths,
            "time": np.array([bdf.main_time_ns for bdf in bdfs], dtype=np.int64),
            "scanNumber": np.array([bdf.scan_number for bdf in bdfs], dtype=np.int64),
            "subscanNumber": np.array(
                [bdf.subscan_number for bdf in bdfs], dtype=np.int64
            ),
            "fieldId": np.array([bdf.field_id for bdf in bdfs], dtype=np.int64),
            "stateId": np.zeros(len(bdfs), dtype=np.int64),
        },
    }


def partition_for_spw(truth, spw_id: int, partition_scheme=None):
    (part,) = [
        part
        for part in truth.expected_partitions(partition_scheme)
        if part.spw_id == spw_id
    ]
    return part


#: The fake time loader gives TIME_CENTROID = time + CENTROID_STEP * time index and
#: EFFECTIVE_INTEGRATION_TIME = interval - DURATION_STEP * time index, so that the
#: broadcast of the per-integration values is checked value by value.
CENTROID_STEP = 0.01
DURATION_STEP = 0.001


class FakeTimeLoader:
    """
    Replaces load_times_from_partition_bdfs (in open_partition) with the true
    times of the synthetic BDFs (contract K1/K5), to test the correlated dataset
    independently of the BDF time loader.
    """

    def __init__(self, truth):
        self.bdf_by_path = {bdf.path: bdf for bdf in truth.bdfs}
        self.calls = []

    def integrations(self, bdf_paths):
        refs = []
        for path in bdf_paths:
            bdf = self.bdf_by_path[str(path)]
            refs.extend(synth.IntegrationRef(bdf, idx) for idx in range(bdf.num_times))
        return refs

    def __call__(self, bdf_paths, scans_metadata):
        self.calls.append((list(bdf_paths), scans_metadata))
        refs = self.integrations(bdf_paths)
        times = synth.expected_times(refs)
        durations = synth.expected_intervals(refs)
        index = np.arange(len(times))
        bdf_start = np.concatenate(
            [[0], np.cumsum([self.bdf_by_path[str(p)].num_times for p in bdf_paths])]
        )
        return (
            times,
            durations,
            times + CENTROID_STEP * index,
            durations - DURATION_STEP * index,
            {"bdf_names": list(bdf_paths), "bdf_start": bdf_start.tolist()},
        )


@pytest.fixture
def fake_time_loader(monkeypatch):
    """Factory: install a FakeTimeLoader for a truth, returns it."""

    def _install(truth):
        loader = FakeTimeLoader(truth)
        monkeypatch.setattr(op, "load_times_from_partition_bdfs", loader)
        return loader

    return _install


@pytest.fixture
def uvw_array_args(monkeypatch):
    """Records the constructor arguments of asdm_backend_arrays.UVWArray."""
    recorded = []
    real_uvw_array = asdm_backend_arrays.UVWArray

    class RecordingUVWArray(real_uvw_array):
        def __init__(self, *args):
            recorded.append(args)
            super().__init__(*args)

    monkeypatch.setattr(asdm_backend_arrays, "UVWArray", RecordingUVWArray)
    return recorded


def expected_scan_names(part) -> list[str]:
    return [str(ref.bdf.scan_number) for ref in part.integrations]


def expected_field_names(truth, part) -> list[str]:
    return [truth.make_field_name(ref.bdf.field_id) for ref in part.integrations]


def assert_base_coords_and_time_vars(ds, truth, part, second_dim, num_rows):
    """Values of time, TIME_CENTROID, EFFECTIVE_INTEGRATION_TIME, scan_name,
    field_name and frequency for the fake time loader."""
    times = synth.expected_times(part.integrations)
    intervals = synth.expected_intervals(part.integrations)
    index = np.arange(len(times))

    np.testing.assert_allclose(ds.time.values, times, rtol=0, atol=1e-6)
    assert ds.time.attrs["format"] == ASDM_TIME_FORMAT == "unix"
    assert ds.time.attrs["scale"] == ASDM_TIME_SCALE == "utc"
    assert ds.time.attrs["units"] == "s"
    assert ds.time.attrs["integration_time"]["data"] == pytest.approx(intervals[0])

    assert ds.TIME_CENTROID.dims == ("time", second_dim)
    expected_centroid = times + CENTROID_STEP * index
    np.testing.assert_allclose(
        ds.TIME_CENTROID.values,
        np.repeat(expected_centroid[:, None], num_rows, axis=1),
        rtol=0,
        atol=1e-6,
    )
    for attr in ("format", "scale", "units", "type"):
        assert ds.TIME_CENTROID.attrs[attr] == ds.time.attrs[attr]
    assert ds.EFFECTIVE_INTEGRATION_TIME.dims == ("time", second_dim)
    np.testing.assert_allclose(
        ds.EFFECTIVE_INTEGRATION_TIME.values,
        np.repeat((intervals - DURATION_STEP * index)[:, None], num_rows, axis=1),
        rtol=0,
        atol=1e-12,
    )
    assert ds.EFFECTIVE_INTEGRATION_TIME.attrs == {"units": "s", "type": "quantity"}

    assert list(map(str, ds.scan_name.values)) == expected_scan_names(part)
    assert ds.scan_name.attrs["scan_intents"] == list(part.obs_modes)
    assert list(map(str, ds.field_name.values)) == expected_field_names(truth, part)

    spw = truth.spw(part.spw_id)
    np.testing.assert_allclose(ds.frequency.values, spw.chan_freqs, rtol=0, atol=1e-3)
    assert list(map(str, ds.polarization.values)) == list(spw.polarization_products)


def time_dependent_variables(ds) -> list[str]:
    """Variables of a dataset with a time dimension, except the time index."""
    return [
        name
        for name, var in ds.variables.items()
        if "time" in var.dims and name not in ds.xindexes
    ]


def assert_preferred_time_chunks(ds, expected: int):
    """Every time-dependent variable (data variables, TIME_CENTROID,
    EFFECTIVE_INTEGRATION_TIME, UVW, scan_name, field_name) has the same
    preferred_chunks along time in its encoding (R: chunks={} consistency)."""
    names = time_dependent_variables(ds)
    assert {"FLAG", "WEIGHT", "TIME_CENTROID", "EFFECTIVE_INTEGRATION_TIME"} <= set(
        names
    )
    assert {"scan_name", "field_name"} <= set(names)
    for name in names:
        assert ds[name].encoding["preferred_chunks"] == {"time": expected}, name
        assert "encoding" not in ds[name].attrs
    assert "preferred_chunks" not in ds.time.encoding


def max_bdf_integrations(part) -> int:
    """Largest number of integrations of a BDF of an expected partition."""
    return max(bdf.num_times for bdf in part.bdfs)


# ---------------------------------------------------------------------------
# Interferometric partitions (F02, F25, F42, F43, F44, F45, F81, F88)
# ---------------------------------------------------------------------------


def test_open_partition_interferometric_multi_scan(
    synth_interferometric, fake_time_loader
):
    """
    CROSS_AND_AUTO partition with 10 integrations in 4 BDFs (2 scans x 2
    subscans): per-integration time, TIME_CENTROID and EFFECTIVE_INTEGRATION_TIME
    values (broadcast over baselines), scan_name of every integration (no
    cycling), field names, scan intents, baselines in BDF order, data groups with
    uvw, chunking hints in the encoding (F02, F25, F42, F43, F44, F45, F81, F88).
    """
    truth = synth_interferometric
    part = partition_for_spw(truth, 0)
    loader = fake_time_loader(truth)
    xdt = op.open_partition(read_asdm(truth.path), partition_descr_for(truth, part))
    ds = xdt.ds

    assert loader.calls[0][0] == [bdf.path for bdf in part.bdfs]
    assert ds.attrs["type"] == "visibility"
    nant = truth.num_antenna
    num_baselines = len(truth.cross_baselines) + nant
    assert dict(ds.sizes) == {
        "time": 10,
        "baseline_id": num_baselines,
        "frequency": 8,
        "polarization": 2,
        "uvw_label": 3,
    }
    assert_base_coords_and_time_vars(ds, truth, part, "baseline_id", num_baselines)
    # the integrations of the 2 scans are contiguous along time
    assert list(ds.scan_name.values) == ["1"] * 5 + ["2"] * 5

    pairs = truth.cross_baselines + [(ant, ant) for ant in range(nant)]
    names = truth.antenna_names
    assert list(ds.baseline_antenna1_name.values) == [names[i] for i, _ in pairs]
    assert list(ds.baseline_antenna2_name.values) == [names[j] for _, j in pairs]
    np.testing.assert_array_equal(ds.baseline_id.values, np.arange(num_baselines))
    assert list(ds.uvw_label.values) == ["u", "v", "w"]

    base = ds.attrs["data_groups"]["base"]
    assert base["correlated_data"] == "VISIBILITY"
    assert base["flag"] == "FLAG"
    assert base["weight"] == "WEIGHT"
    assert base["uvw"] == "UVW"
    assert base["field_and_source"] == "field_and_source_base_xds"
    # the atmospheric phase correction loaded is recorded (F11)
    assert "atmospheric phase correction AP_UNCORRECTED" in base["description"]

    for var in ("VISIBILITY", "FLAG", "WEIGHT"):
        assert ds[var].dims == ("time", "baseline_id", "frequency", "polarization")
    # small data: one BDF per preferred chunk, for all the variables (UVW too)
    assert [bdf.num_times for bdf in part.bdfs] == [3, 2, 2, 3]
    assert_preferred_time_chunks(ds, 3)
    assert "UVW" in time_dependent_variables(ds)
    assert "field_and_source_xds" not in ds.VISIBILITY.attrs
    assert np.iscomplexobj(ds.VISIBILITY)
    assert ds.FLAG.dtype == bool
    assert ds.UVW.dims == ("time", "baseline_id", "uvw_label")
    assert ds.UVW.attrs == {"type": "uvw", "frame": "icrs", "units": "m"}

    assert list(map(str, xdt["antenna_xds"].antenna_name.values)) == names
    assert list(map(str, xdt["field_and_source_base_xds"].field_name.values)) == [
        truth.make_field_name(0)
    ]
    assert "pointing_xds" not in xdt.children
    assert not check_datatree(xdt)


def open_chunked_ps(path: str, chunks=None, **dask_config) -> xr.DataTree:
    """open_datatree with the xradio_asdm engine, with optional dask settings
    (for example ``**{"array.chunk-size": "2560B"}``) applied while opening."""
    import dask

    with dask.config.set(dask_config):
        return xr.open_datatree(path, engine="xradio_asdm", chunks=chunks, cache=False)


def assert_consistent_time_chunks(ds, expected_time_chunks: tuple[int, ...]):
    """All the time-dependent variables are dask arrays with the same chunks
    along time, the other dimensions in one chunk; Dataset.chunks and
    Dataset-level map_blocks work without unify_chunks()."""
    assert ds.chunks["time"] == expected_time_chunks
    assert ds.chunksizes["time"] == expected_time_chunks
    for name in time_dependent_variables(ds):
        var = ds[name]
        assert var.chunks is not None, name
        for dim, dim_chunks in zip(var.dims, var.chunks, strict=True):
            if dim == "time":
                assert dim_chunks == expected_time_chunks, name
            else:
                assert dim_chunks == (ds.sizes[dim],), (name, dim)
    # uniform chunks (except the last one, smaller): what to_zarr needs
    assert len(set(expected_time_chunks[:-1])) <= 1
    assert expected_time_chunks[-1] <= expected_time_chunks[0]
    mapped = xr.map_blocks(lambda block: block, ds[["FLAG", "TIME_CENTROID"]])
    np.testing.assert_array_equal(mapped.FLAG.values, ds.FLAG.values)


def test_open_with_chunks_uses_preferred_chunks(synth_interferometric):
    """
    Opening with chunks={} gives the same dask chunks along time for all the
    time-dependent variables (VISIBILITY, WEIGHT, FLAG, UVW, TIME_CENTROID,
    EFFECTIVE_INTEGRATION_TIME, scan_name, field_name): Dataset.chunks does not
    raise. For small data the preferred chunk is the size of the largest BDF of
    the partition (3 integrations), not one integration (R: inconsistent chunks,
    F81). The BDFs have 3, 2, 2 and 3 integrations, so the uniform chunks
    (3, 3, 3, 1) do not line up with them (as documented). The chunked values
    equal the eager ones.
    """
    truth = synth_interferometric
    ps = open_chunked_ps(truth.path, chunks={})
    eager = open_chunked_ps(truth.path)
    assert len(ps.children) == len(truth.spws)
    for name, node in ps.children.items():
        ds = node.ds
        assert "UVW" in time_dependent_variables(ds)
        assert_consistent_time_chunks(ds, (3, 3, 3, 1))
        for var in time_dependent_variables(ds):
            assert "encoding" not in ds[var].attrs
            np.testing.assert_array_equal(
                ds[var].values, eager[name].ds[var].values, err_msg=var
            )


def test_open_with_chunks_memory_target(synth_interferometric):
    """The preferred time chunk is bounded by the dask "array.chunk-size"
    setting: 2560 bytes are 2 integrations of the largest variables of SPW 0
    (10 baselines x 8 channels x 2 polarizations x 8 bytes), the same for all
    the variables, and the chunked values equal the eager ones."""
    truth = synth_interferometric
    ps = open_chunked_ps(truth.path, chunks={}, **{"array.chunk-size": "2560B"})
    eager = open_chunked_ps(truth.path)
    (name,) = [
        name for name, node in ps.children.items() if node.ds.sizes["frequency"] == 8
    ]
    ds = ps[name].ds
    assert ds.VISIBILITY.dtype.itemsize * 10 * 8 * 2 == 1280
    assert_preferred_time_chunks(ds, 2)
    assert_consistent_time_chunks(ds, (2, 2, 2, 2, 2))
    for var in ("VISIBILITY", "FLAG", "UVW"):
        np.testing.assert_array_equal(ds[var].values, eager[name].ds[var].values)


def test_open_with_chunks_writes_to_zarr(synth_interferometric, tmp_path):
    """A processing set opened with chunks={} can be written to Zarr as is
    (uniform dask chunks along time) and read back with the same values."""
    truth = synth_interferometric
    ps = open_chunked_ps(truth.path, chunks={})
    store = tmp_path / "ps.zarr"
    ps.to_zarr(store, consolidated=False)
    written = xr.open_datatree(store, engine="zarr", consolidated=False)
    eager = open_chunked_ps(truth.path)
    for name in ps.children:
        for var in ("VISIBILITY", "FLAG", "UVW", "TIME_CENTROID"):
            np.testing.assert_array_equal(
                written[name].ds[var].values, eager[name].ds[var].values
            )


@pytest.fixture(scope="module")
def synth_long_subscan(make_synthetic_asdm):
    """One subscan of 40 integrations in a single BDF (3 antennas, 1 SPW of 4
    channels, dual pol)."""
    cfg = synth.ConfigSpec(
        basebands=[synth.BasebandSpec("BB_1", [synth.SpwSpec(4, "dual")])]
    )
    spec = synth.ASDMSpec(
        configs=[cfg],
        scans=[synth.ScanSpec([synth.SubscanSpec(0, "ON_SOURCE", 40)])],
        fields=[synth.FieldSpec("J0423-0120", (1.1, -0.02))],
        name="uid___A002_X1234_X56a0",
    )
    return make_synthetic_asdm(spec)


@pytest.mark.parametrize(
    "chunk_size, expected_time_chunks",
    [(None, (40,)), ("3840B", (10, 10, 10, 10)), ("2688B", (7,) * 5 + (5,))],
)
def test_open_with_chunks_opens_each_bdf_once_per_chunk(
    synth_long_subscan, monkeypatch, chunk_size, expected_time_chunks
):
    """
    With chunks={}, loading VISIBILITY and FLAG opens the BDF once per chunk:
    once in total when the BDF fits in the memory target, not once per
    integration (one-integration chunks re-read the BDF from its start for every
    integration, O(N^2) for N integrations) (R: preferred_chunks time=1). The
    target (dask "array.chunk-size") splits the BDF: 3840 bytes are 10
    integrations of 6 baselines x 4 channels x 2 polarizations x 8 bytes.
    """
    import dask

    truth = synth_long_subscan
    config = {} if chunk_size is None else {"array.chunk-size": chunk_size}
    ps = open_chunked_ps(truth.path, chunks={}, **config)
    (node,) = ps.children.values()
    ds = node.ds
    assert dict(ds.sizes)["time"] == 40
    assert_consistent_time_chunks(ds, expected_time_chunks)

    opened = []
    real_open = pyasdm.bdf.BDFReader.open

    def counting_open(self, path, *args, **kwargs):
        opened.append(str(path))
        return real_open(self, path, *args, **kwargs)

    monkeypatch.setattr(pyasdm.bdf.BDFReader, "open", counting_open)
    with dask.config.set(scheduler="synchronous"):
        visibility = ds.VISIBILITY.values
        flag = ds.FLAG.values
    num_chunks = len(expected_time_chunks)
    assert len(opened) == 2 * num_chunks
    assert set(opened) == {bdf.path for bdf in truth.bdfs}

    monkeypatch.setattr(pyasdm.bdf.BDFReader, "open", real_open)
    eager = open_chunked_ps(truth.path)
    (eager_node,) = eager.children.values()
    np.testing.assert_array_equal(visibility, eager_node.ds.VISIBILITY.values)
    np.testing.assert_array_equal(flag, eager_node.ds.FLAG.values)


def test_open_partition_real_bdf_times(synth_full_pol):
    """Without mocks: the time coordinate and TIME_CENTROID decoded with their
    own format/scale attributes are the true UTC instants of the integrations
    (F02, F25, K1)."""
    from astropy.time import Time

    truth = synth_full_pol
    part = partition_for_spw(truth, 0)
    ds = op.open_partition(read_asdm(truth.path), partition_descr_for(truth, part)).ds
    expected = synth.expected_times(part.integrations)
    np.testing.assert_allclose(ds.time.values, expected, rtol=0, atol=1e-6)
    decoded = Time(
        ds.time.values, format=ds.time.attrs["format"], scale=ds.time.attrs["scale"]
    )
    truth_utc = Time(synth.expected_datetimes(part.integrations), scale="utc")
    assert np.max(np.abs((decoded - truth_utc).to_value("s"))) < 2e-6
    np.testing.assert_allclose(
        ds.TIME_CENTROID.values,
        np.broadcast_to(expected[:, None], ds.TIME_CENTROID.shape),
        rtol=0,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        ds.EFFECTIVE_INTEGRATION_TIME.values,
        np.broadcast_to(
            synth.expected_intervals(part.integrations)[:, None],
            ds.EFFECTIVE_INTEGRATION_TIME.shape,
        ),
        rtol=1e-9,
    )


# ---------------------------------------------------------------------------
# Multi-field partitions: field_name per time, per-time phase center (F29, F31)
# ---------------------------------------------------------------------------


def test_open_partition_multi_field_mosaic(
    synth_mosaic, fake_time_loader, uvw_array_args
):
    """
    A mosaic partition with two fields sharing the fieldName "M100" (no fieldId
    partition axis): field_name follows each integration's field with unique
    "<fieldName>_<fieldId>" names, the field_and_source dataset has both fields,
    and UVW gets one phase center per integration (Field.phaseDir of its field)
    (F29, F31, F43, K8, K10).
    """
    truth = synth_mosaic
    (part,) = [
        part
        for part in truth.expected_partitions(partition_scheme=[])
        if part.field_ids == (1, 2)
    ]
    fake_time_loader(truth)
    xdt = op.open_partition(read_asdm(truth.path), partition_descr_for(truth, part))
    ds = xdt.ds

    num_baselines = len(truth.cross_baselines) + truth.num_antenna
    assert_base_coords_and_time_vars(ds, truth, part, "baseline_id", num_baselines)
    assert list(ds.field_name.values) == ["M100_1"] * 3 + ["M100_2"] * 2
    assert list(ds.scan_name.values) == ["2"] * 5

    fsx = xdt["field_and_source_base_xds"].ds
    assert list(map(str, fsx.field_name.values)) == ["M100_1", "M100_2"]

    (args,) = uvw_array_args
    shape, time, ant1, ant2, antenna_position, phase_center = args
    assert shape == (5, num_baselines, 3)
    assert phase_center.dims == ("time", "sky_dir_label")
    assert "field_name" not in phase_center.coords
    np.testing.assert_array_equal(phase_center.time.values, ds.time.values)
    expected_dirs = np.array(
        [truth.spec.fields[ref.bdf.field_id].phase_dir for ref in part.integrations]
    )
    np.testing.assert_allclose(phase_center.values, expected_dirs, rtol=0, atol=1e-12)
    assert phase_center.attrs == fsx.FIELD_PHASE_CENTER_DIRECTION.attrs
    assert time.attrs["format"] == ASDM_TIME_FORMAT
    assert list(ant1.values) == list(ds.baseline_antenna1_name.values)
    assert list(ant2.values) == list(ds.baseline_antenna2_name.values)

    # w is the projection of the baselines on the phase center of each integration
    uvw = ds.UVW.values
    baselines = synth.baseline_vectors_itrf(truth)
    unit = synth.phase_center_itrs_unit_vectors(ds.time.values, expected_dirs)
    w_expected = unit @ baselines.T
    sign = np.sign(np.sum(uvw[..., 2] * w_expected))
    assert sign != 0
    np.testing.assert_allclose(uvw[..., 2], sign * w_expected, rtol=0, atol=0.01)
    assert not check_datatree(xdt)


def test_select_phase_center_direction_by_time():
    """One direction per time from the field of every time, attrs kept, missing
    fields raise (K10)."""
    fsx = xr.Dataset(
        {
            "FIELD_PHASE_CENTER_DIRECTION": (
                ["field_name", "sky_dir_label"],
                [[0.1, 0.2], [0.3, 0.4]],
                {"type": "sky_coord", "units": "rad", "frame": "icrs"},
            )
        },
        coords={
            "field_name": ["a_1", "b_2"],
            "sky_dir_label": ["ra", "dec"],
            "source_name": ("field_name", ["a", "b"]),
        },
    )
    field_name = xr.DataArray(
        ["b_2", "a_1", "b_2"], dims="time", coords={"time": [10.0, 11.0, 12.0]}
    )
    by_time = op._select_phase_center_direction_by_time(fsx, field_name)
    assert by_time.dims == ("time", "sky_dir_label")
    np.testing.assert_array_equal(by_time.values, [[0.3, 0.4], [0.1, 0.2], [0.3, 0.4]])
    np.testing.assert_array_equal(by_time.time.values, [10.0, 11.0, 12.0])
    assert set(by_time.coords) == {"time", "sky_dir_label"}
    assert by_time.attrs["frame"] == "icrs"

    with pytest.raises(RuntimeError, match="c_3"):
        op._select_phase_center_direction_by_time(
            fsx, xr.DataArray(["c_3"], dims="time", coords={"time": [1.0]})
        )


# ---------------------------------------------------------------------------
# BDF spectral window index (F46)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("spw_id, expected_bdf_spw_id", [(0, 1), (1, 0)])
def test_bdf_spw_id_follows_config_dd_order(
    synth_full_pol, fake_time_loader, spw_id, expected_bdf_spw_id
):
    """The ConfigDescription lists its DDs as [1, 0]: the BDF SPW position is
    the position of the partition's DD in that list, not the DataDescription
    table order (F46)."""
    truth = synth_full_pol
    part = partition_for_spw(truth, spw_id)
    assert truth.spw(spw_id).bdf_index == expected_bdf_spw_id
    fake_time_loader(truth)
    result = op.create_coordinates(
        read_asdm(truth.path), partition_descr_for(truth, part)
    )
    coords, attrs, num_antenna, out_spw_id, bdf_spw_id = result[:5]
    assert out_spw_id == spw_id
    assert bdf_spw_id == expected_bdf_spw_id
    assert num_antenna == truth.num_antenna
    np.testing.assert_allclose(
        coords["frequency"][1], truth.spw(spw_id).chan_freqs, rtol=0, atol=1e-3
    )


@pytest.mark.parametrize(
    "config_dd_ids, dd_id, expected",
    [
        ([0], 0, 0),
        ([1, 0], 1, 0),
        ([1, 0], 0, 1),
        ([2, 0, 1], 1, 2),
        (np.array([5, 7, 9, 11]), 9, 2),
    ],
)
def test_find_bdf_spw_id(config_dd_ids, dd_id, expected):
    assert op._find_bdf_spw_id(config_dd_ids, dd_id) == expected


def test_find_bdf_spw_id_missing_dd_raises():
    with pytest.raises(RuntimeError, match="dataDescriptionId 3 is not in"):
        op._find_bdf_spw_id([1, 0], 3)


def test_dd_not_in_config_raises_before_reading_times(
    synth_interleaved, fake_time_loader
):
    """A partition whose DD is not in its ConfigDescription is an error (no
    fallback to another SPW), detected before the BDF times are read (F46)."""
    truth = synth_interleaved
    part = partition_for_spw(truth, 5)
    assert part.config_idx == 3
    descr = partition_descr_for(truth, part)
    descr["dataDescriptionId"] = np.array([6])
    loader = fake_time_loader(truth)
    with pytest.raises(RuntimeError, match=r"6 is not in .* \[5, 7, 9, 11\]"):
        op.create_coordinates(read_asdm(truth.path), descr)
    assert not loader.calls


def test_check_bdf_spectral_window(synth_full_pol):
    """numSpectralPoint of the BDF SPW at the computed position must match the
    SpectralWindow numChan (F46)."""
    truth = synth_full_pol
    bdf_path = truth.bdfs[0].path
    # BDF SPW 0 (BB_1) has 4 channels, BDF SPW 1 (BB_2) has 2
    op._check_bdf_spectral_window(bdf_path, 0, 1, 4)
    op._check_bdf_spectral_window(bdf_path, 1, 0, 2)
    with pytest.raises(RuntimeError, match="numSpectralPoint"):
        op._check_bdf_spectral_window(bdf_path, 1, 0, 4)
    with pytest.raises(RuntimeError, match="only 2 spectral windows"):
        op._check_bdf_spectral_window(bdf_path, 2, 0, 4)
    # unreadable BDF: the check is skipped
    op._check_bdf_spectral_window("/nonexistent/bdf", 0, 0, 4)


# ---------------------------------------------------------------------------
# Single dish and radiometer partitions (F41, F79, K12)
# ---------------------------------------------------------------------------


def test_open_partition_single_dish_spectrum(
    synth_single_dish_simple, fake_time_loader, uvw_array_args
):
    """An AUTO_ONLY (correlator) partition is a SpectrumXds: type "spectrum",
    dims (time, antenna_name, frequency, polarization), real SPECTRUM, no UVW,
    data group with correlated_data "SPECTRUM" and no uvw, reference center
    direction in field_and_source (F41, K12)."""
    truth = synth_single_dish_simple
    part = partition_for_spw(truth, 1)
    fake_time_loader(truth)
    xdt = op.open_partition(read_asdm(truth.path), partition_descr_for(truth, part))
    ds = xdt.ds

    assert ds.attrs["type"] == "spectrum"
    nant = truth.num_antenna
    assert dict(ds.sizes) == {
        "time": 5,
        "antenna_name": nant,
        "frequency": 2,
        "polarization": 2,
    }
    assert list(map(str, ds.antenna_name.values)) == truth.antenna_names
    assert_base_coords_and_time_vars(ds, truth, part, "antenna_name", nant)
    dims = ("time", "antenna_name", "frequency", "polarization")
    for var in ("SPECTRUM", "FLAG", "WEIGHT"):
        assert ds[var].dims == dims
    assert_preferred_time_chunks(ds, max_bdf_integrations(part))
    assert ds.SPECTRUM.dtype.kind == "f"
    assert ds.FLAG.dtype == bool
    for name in ("VISIBILITY", "UVW", "uvw_label", "baseline_id"):
        assert name not in ds.variables
    assert not uvw_array_args
    base = ds.attrs["data_groups"]["base"]
    assert base["correlated_data"] == "SPECTRUM"
    assert "uvw" not in base

    fsx = xdt["field_and_source_base_xds"].ds
    assert "FIELD_REFERENCE_CENTER_DIRECTION" in fsx
    assert list(map(str, xdt["antenna_xds"].antenna_name.values)) == truth.antenna_names
    assert not check_datatree(xdt)


def test_open_partition_packed_radiometer(synth_interleaved, fake_time_loader):
    """A RADIOMETER partition (packed WVR BDFs with 4 integrations each) is a
    VisibilityXds of type "radiometer" with the auto-correlations as baselines;
    scan_name/field_name follow the per-BDF time indices (F79, K5, K12)."""
    truth = synth_interleaved
    wvr_spw = truth.spws[0].spw_id
    (part,) = [
        part
        for part in truth.expected_partitions(
            include_processor_types=["RADIOMETER"],
        )
        if part.spw_id == wvr_spw
    ]
    fake_time_loader(truth)
    xdt = op.open_partition(read_asdm(truth.path), partition_descr_for(truth, part))
    ds = xdt.ds

    assert ds.attrs["type"] == "radiometer"
    assert ds.attrs["processor_info"]["type"] == "RADIOMETER"
    nant = truth.num_antenna
    num_subscans = len(truth.spec.scans[0].subscans)
    assert ds.sizes["time"] == 4 * num_subscans
    assert ds.sizes["baseline_id"] == nant
    assert list(ds.baseline_antenna1_name.values) == truth.antenna_names
    assert list(ds.baseline_antenna2_name.values) == truth.antenna_names
    assert_base_coords_and_time_vars(ds, truth, part, "baseline_id", nant)
    assert ds.attrs["data_groups"]["base"]["correlated_data"] == "VISIBILITY"
    assert ds.attrs["data_groups"]["base"]["uvw"] == "UVW"
    # AUTO_ONLY data: no crossData, so no atmospheric phase correction is recorded
    assert (
        "atmospheric phase correction"
        not in ds.attrs["data_groups"]["base"]["description"]
    )
    assert "FIELD_PHASE_CENTER_DIRECTION" in xdt["field_and_source_base_xds"].ds
    assert not check_datatree(xdt)


@pytest.mark.parametrize(
    "correlation_mode, processor_type, expected",
    [
        (
            pyasdm.enumerations.CorrelationMode.CROSS_AND_AUTO,
            "CORRELATOR",
            "visibility",
        ),
        (pyasdm.enumerations.CorrelationMode.AUTO_ONLY, "CORRELATOR", "spectrum"),
        (pyasdm.enumerations.CorrelationMode.AUTO_ONLY, "SPECTROMETER", "spectrum"),
        (pyasdm.enumerations.CorrelationMode.AUTO_ONLY, "RADIOMETER", "radiometer"),
        ("CROSS_AND_AUTO", "RADIOMETER", "radiometer"),
        ("AUTO_ONLY", pyasdm.enumerations.ProcessorType.CORRELATOR, "spectrum"),
    ],
)
def test_find_xds_type(correlation_mode, processor_type, expected):
    assert op.find_xds_type(correlation_mode, processor_type) == expected


APC = pyasdm.enumerations.AtmPhaseCorrection


@pytest.mark.parametrize(
    "apc, expected",
    [
        ([APC.AP_UNCORRECTED], "AP_UNCORRECTED"),
        ([APC.AP_CORRECTED, APC.AP_UNCORRECTED], "AP_UNCORRECTED"),
        ([APC.AP_CORRECTED], "AP_CORRECTED"),
        (["AP_UNCORRECTED", "AP_CORRECTED"], "AP_UNCORRECTED"),
        (["AP_CORRECTED", "AP_MIXED"], None),
        ([], None),
    ],
)
def test_get_loaded_atm_phase_correction(apc, expected, caplog):
    """The APC recorded in the data group description is the one the BDF
    loaders load: AP_UNCORRECTED when there are several (F11, K6). Loading an
    APC other than AP_UNCORRECTED (the only one) is reported here, once per
    partition, as the BDF loaders do not warn."""
    config = pd.Series({"atmPhaseCorrection": apc})
    with caplog.at_level(logging.WARNING):
        assert op._get_loaded_atm_phase_correction(config) == expected
    apc_warnings = [
        rec
        for rec in caplog.records
        if "only the atmospheric phase correction" in rec.getMessage()
    ]
    if expected not in (None, "AP_UNCORRECTED"):
        assert len(apc_warnings) == 1
        assert expected in apc_warnings[0].getMessage()
    else:
        assert not apc_warnings


# ---------------------------------------------------------------------------
# Baseline antennas from ConfigDescription.antennaId (F83)
# ---------------------------------------------------------------------------


def test_baseline_antennas_follow_config_antenna_id(
    synth_interferometric, fake_time_loader
):
    """The BDF antenna slots map to ConfigDescription.antennaId (here a reversed
    subset of the Antenna table): baseline names, UVW inputs and antenna_xds
    follow that order (F83)."""
    truth = synth_interferometric
    part = partition_for_spw(truth, 0)
    asdm = read_asdm(truth.path)
    config_row = asdm.getConfigDescription().get()[0]
    slot_antennas = [3, 1, 0]
    config_row.setNumAntenna(len(slot_antennas))
    config_row.setAntennaId(
        [pyasdm.types.Tag(f"Antenna_{ant}") for ant in slot_antennas]
    )
    fake_time_loader(truth)
    xdt = op.open_partition(asdm, partition_descr_for(truth, part))
    ds = xdt.ds

    names = [truth.antenna_names[ant] for ant in slot_antennas]
    pairs = synth.cross_baseline_pairs(3) + [(slot, slot) for slot in range(3)]
    assert list(ds.baseline_antenna1_name.values) == [names[i] for i, _ in pairs]
    assert list(ds.baseline_antenna2_name.values) == [names[j] for _, j in pairs]
    axds = xdt["antenna_xds"].ds
    assert list(map(str, axds.antenna_name.values)) == names
    np.testing.assert_allclose(
        axds.ANTENNA_POSITION.transpose("antenna_name", ...).values,
        truth.antenna_positions[slot_antennas],
        rtol=0,
        atol=1e-6,
    )
    assert not check_datatree(xdt)


def test_config_antenna_names_errors_and_extra_ids(synth_interferometric, caplog):
    truth = synth_interferometric
    config = pd.Series(
        {"configDescriptionId": 0, "numAntenna": 2, "antennaId": np.array([2, 0, 1])}
    )
    asdm = read_asdm(truth.path)
    with caplog.at_level(logging.WARNING):
        names = op._get_config_antenna_names(asdm, config)
    assert names == [truth.antenna_names[2], truth.antenna_names[0]]
    assert "Using the first 2 antennaId entries: [2, 0]" in caplog.text

    config["numAntenna"] = 4
    with pytest.raises(RuntimeError, match="only 3 antennaId entries"):
        op._get_config_antenna_names(asdm, config)

    config["antennaId"] = np.array([0, 1, 9, 2])
    with pytest.raises(RuntimeError, match=r"antennaId \[9\]"):
        op._get_config_antenna_names(asdm, config)


# ---------------------------------------------------------------------------
# Frequency observer from SpectralWindow.measFreqRef (F84)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "meas_freq_ref, observer",
    [
        ("TOPO", "TOPO"),
        ("LSRK", "lsrk"),
        ("LSRD", "lsrd"),
        ("BARY", "BARY"),
        ("GEO", "gcrs"),
        ("REST", "REST"),
        ("LABREST", "REST"),
    ],
)
def test_frequency_observer_from_meas_freq_ref(
    synth_interferometric, meas_freq_ref, observer
):
    """The frequency coordinate and its reference_frequency use the SPW frame
    (mapped to the MSv4 vocabulary) as observer, no extra "frame" attr (F84)."""
    truth = synth_interferometric
    asdm = read_asdm(truth.path)
    spw_row = asdm.getSpectralWindow().getRowByKey(pyasdm.types.Tag("SpectralWindow_2"))
    spw_row.setMeasFreqRef(
        pyasdm.enumerations.FrequencyReferenceCode.newFrequencyReferenceCode(
            meas_freq_ref
        )
    )
    coord, attrs, num_chan = op._create_frequency_coord_attrs(asdm, 2)
    assert num_chan == 4
    np.testing.assert_allclose(coord[1], truth.spw(2).chan_freqs, rtol=0, atol=1e-3)
    assert attrs["observer"] == observer
    assert attrs["reference_frequency"]["attrs"]["observer"] == observer
    assert "frame" not in attrs
    # LSB SPW with a negative channel step: the channel width is positive
    assert attrs["channel_width"]["data"] == pytest.approx(31.25e6)
    assert attrs["spectral_window_name"].endswith("_2")


def test_frequency_observer_without_msv4_equivalent(synth_interferometric, caplog):
    truth = synth_interferometric
    asdm = read_asdm(truth.path)
    spw_row = asdm.getSpectralWindow().getRowByKey(pyasdm.types.Tag("SpectralWindow_0"))
    spw_row.setMeasFreqRef(pyasdm.enumerations.FrequencyReferenceCode.GALACTO)
    with caplog.at_level(logging.WARNING):
        observer = op._get_frequency_observer(asdm, 0)
    assert observer == "GALACTO"
    assert "GALACTO of spectral window 0 has no MSv4 equivalent" in caplog.text


# ---------------------------------------------------------------------------
# Pointing (F40, F76, F88)
# ---------------------------------------------------------------------------


@pytest.fixture
def pointing_calls(monkeypatch):
    """Records the calls of create_pointing_xds made by open_partition."""
    calls = []
    real_create_pointing_xds = op.create_pointing_xds

    def recording_create_pointing_xds(asdm, **kwargs):
        calls.append(kwargs)
        return real_create_pointing_xds(asdm, **kwargs)

    monkeypatch.setattr(op, "create_pointing_xds", recording_create_pointing_xds)
    return calls


def test_open_partition_pointing_time_range(
    synth_interferometric_pointing, fake_time_loader, pointing_calls
):
    """create_pointing_xds is called with a time range covering the partition
    (plus margin) and the pointing samples of the partition are included (F40,
    K11)."""
    truth = synth_interferometric_pointing
    part = partition_for_spw(truth, 1)
    fake_time_loader(truth)
    xdt = op.open_partition(
        read_asdm(truth.path), partition_descr_for(truth, part), with_pointing=True
    )
    times = synth.expected_times(part.integrations)
    half_interval = 0.5 * synth.expected_intervals(part.integrations).max()
    margin = half_interval + op.POINTING_TIME_RANGE_MARGIN
    (call,) = pointing_calls
    np.testing.assert_allclose(
        call["time_range"], (times.min() - margin, times.max() + margin), atol=1e-6
    )
    pxds = xdt["pointing_xds"].ds
    values = pxds.time_pointing.values
    samples = truth.pointing_times_unix
    inside = samples[(samples >= times.min()) & (samples <= times.max())]
    assert len(inside) > 0
    assert np.isin(np.round(inside, 6), np.round(values, 6)).all()
    assert values.min() >= call["time_range"][0]
    assert values.max() <= call["time_range"][1]
    assert not check_datatree(xdt)


def test_open_partition_empty_pointing_table(
    synth_interferometric, fake_time_loader, pointing_calls
):
    """No Pointing rows: the partition opens without pointing_xds (F76)."""
    truth = synth_interferometric
    part = partition_for_spw(truth, 0)
    fake_time_loader(truth)
    xdt = op.open_partition(
        read_asdm(truth.path), partition_descr_for(truth, part), with_pointing=True
    )
    assert len(pointing_calls) == 1
    assert "pointing_xds" not in xdt.children
    assert "VISIBILITY" in xdt.ds
    assert not check_datatree(xdt)


def test_open_partition_pointing_antennas(
    synth_interferometric_pointing, fake_time_loader, pointing_calls
):
    """create_pointing_xds gets the antennas of the partition (BDF slot order)
    and the pointing_xds has those antennas, in the same order (F40)."""
    truth = synth_interferometric_pointing
    part = partition_for_spw(truth, 0)
    fake_time_loader(truth)
    xdt = op.open_partition(
        read_asdm(truth.path), partition_descr_for(truth, part), with_pointing=True
    )
    (call,) = pointing_calls
    antenna_names = list(map(str, xdt["antenna_xds"].ds.antenna_name.values))
    assert call["antenna_names"] == antenna_names
    pointing_antennas = list(map(str, xdt["pointing_xds"].ds.antenna_name.values))
    assert pointing_antennas == antenna_names


@pytest.fixture(scope="module")
def synth_single_dish_pointing(make_synthetic_asdm):
    """Single dish (one partition per SPW) with a Pointing table, by the
    antennas with Pointing rows: all (None), or not the last one, DA43 ((0, 1))."""
    truths = {}
    for name, pointing_antennas in (
        ("uid___A002_X1234_X56c0", None),
        ("uid___A002_X1234_X56c1", (0, 1)),
    ):
        spec = synth.single_dish_spec(name=name, with_off_and_calibration=False)
        spec.with_pointing = True
        spec.pointing_antennas = pointing_antennas
        truths[pointing_antennas] = make_synthetic_asdm(spec)
    return truths


def reorder_config_antennas(asdm: pyasdm.ASDM, order) -> None:
    """Change the order of the antennas of the ConfigDescription rows (in
    memory)."""
    for row in asdm.getConfigDescription().get():
        antenna_ids = row.getAntennaId()
        row.setAntennaId([antenna_ids[idx] for idx in order])


@pytest.mark.parametrize("pointing_antennas", [None, (0, 1)])
@pytest.mark.parametrize("config_order", [(0, 1, 2), (2, 0, 1)])
def test_open_partition_single_dish_pointing_antennas(
    synth_single_dish_pointing, fake_time_loader, pointing_antennas, config_order
):
    """Single dish: the pointing_xds node must align with the antenna_name index
    of the correlated dataset in the DataTree. Its antennas are those of the
    partition in the partition (ConfigDescription) order, also when that is not
    the antennaId order, and with NaN for an antenna without Pointing samples
    (both made the partition fail to open before)."""
    truth = synth_single_dish_pointing[pointing_antennas]
    fake_time_loader(truth)
    asdm = read_asdm(truth.path)
    reorder_config_antennas(asdm, config_order)
    expected_names = [truth.antenna_names[idx] for idx in config_order]
    parts = truth.expected_partitions()
    assert len(parts) == 2
    for part in parts:
        xdt = op.open_partition(
            asdm, partition_descr_for(truth, part), with_pointing=True
        )
        assert xdt.ds.attrs["type"] == "spectrum"
        assert list(map(str, xdt.ds.antenna_name.values)) == expected_names
        pxds = xdt["pointing_xds"].ds
        assert list(map(str, pxds.antenna_name.values)) == expected_names
        sample_times = truth.pointing_times_unix
        time_pointing = pxds.time_pointing.values
        idx = np.argmin(np.abs(sample_times[None, :] - time_pointing[:, None]), axis=1)
        np.testing.assert_allclose(sample_times[idx], time_pointing, rtol=0, atol=1e-5)
        for pos, ant in enumerate(config_order):
            # NaN for the antenna without Pointing rows (NaN in the truth)
            for var, expected in (
                ("POINTING_DISH_MEASURED", truth.pointing_encoder),
                ("POINTING_BEAM", truth.pointing_target),
            ):
                values = pxds[var].isel(antenna_name=pos).values
                np.testing.assert_allclose(
                    values, expected[idx, ant], rtol=0, atol=1e-9, err_msg=var
                )
        has_rows = [
            pointing_antennas is None or ant in pointing_antennas
            for ant in config_order
        ]
        all_nan = np.isnan(pxds.POINTING_DISH_MEASURED.values).all(axis=(0, 2))
        np.testing.assert_array_equal(all_nan, np.logical_not(has_rows))
        assert not check_datatree(xdt)


@pytest.mark.parametrize(
    "error",
    [
        NotImplementedError("polynomial pointing"),
        PointingConversionError("inconsistent Pointing row"),
    ],
)
def test_open_partition_pointing_conversion_error(
    synth_interferometric_pointing, fake_time_loader, monkeypatch, caplog, error
):
    """A Pointing table that cannot be converted (polynomial pointing,
    inconsistent rows) does not fail the partition: it is opened without
    pointing_xds, which is logged. Other errors are not caught."""
    truth = synth_interferometric_pointing
    part = partition_for_spw(truth, 0)
    fake_time_loader(truth)

    def raise_error(asdm, **kwargs):
        raise error

    monkeypatch.setattr(op, "create_pointing_xds", raise_error)
    with caplog.at_level(logging.INFO):
        xdt = op.open_partition(
            read_asdm(truth.path), partition_descr_for(truth, part), with_pointing=True
        )
    assert "pointing_xds" not in xdt.children
    assert "VISIBILITY" in xdt.ds
    assert str(error) in caplog.text
    assert "will not have a pointing_xds" in caplog.text
    assert not check_datatree(xdt)

    def raise_type_error(asdm, **kwargs):
        raise TypeError("bug")

    monkeypatch.setattr(op, "create_pointing_xds", raise_type_error)
    with pytest.raises(TypeError, match="bug"):
        op.open_partition(
            read_asdm(truth.path), partition_descr_for(truth, part), with_pointing=True
        )


@pytest.mark.parametrize(
    "spectral_type, only_types, expected",
    [
        (np.array(["FULL_RESOLUTION"]), None, True),
        (np.array(["FULL_RESOLUTION"]), [], True),
        (np.array(["FULL_RESOLUTION"]), ["FULL_RESOLUTION"], True),
        (np.array(["FULL_RESOLUTION"]), ["CHANNEL_AVERAGE", "FULL_RESOLUTION"], True),
        (np.array(["FULL_RESOLUTION"]), ["CHANNEL_AVERAGE"], False),
        (["BASEBAND_WIDE"], ["BASEBAND_WIDE"], True),
        ("BASEBAND_WIDE", ["BASEBAND_WIDE"], True),
        (np.array(["BASEBAND_WIDE"]), ["FULL_RESOLUTION"], False),
    ],
)
def test_is_pointing_requested(spectral_type, only_types, expected):
    """The spectral resolution type filter works for the ndarray values given by
    create_partitions (positive and negative cases, F88)."""
    descr = {"spectralType": spectral_type}
    assert op._is_pointing_requested(descr, True, only_types) is expected
    assert op._is_pointing_requested(descr, False, only_types) is False


def test_open_partition_pointing_filtered_by_spectral_type(
    synth_interferometric_pointing, fake_time_loader, pointing_calls
):
    truth = synth_interferometric_pointing
    part = partition_for_spw(truth, 0)
    fake_time_loader(truth)
    asdm = read_asdm(truth.path)
    descr = partition_descr_for(truth, part)
    xdt = op.open_partition(
        asdm,
        descr,
        with_pointing=True,
        pointing_for_only_spectral_resolution_types=["CHANNEL_AVERAGE"],
    )
    assert "pointing_xds" not in xdt.children
    assert not pointing_calls
    xdt = op.open_partition(
        asdm,
        descr,
        with_pointing=True,
        pointing_for_only_spectral_resolution_types=["FULL_RESOLUTION"],
    )
    assert "pointing_xds" in xdt.children


def test_pointing_time_range():
    ds = xr.Dataset(
        {"EFFECTIVE_INTEGRATION_TIME": (["time", "baseline_id"], [[1.0], [3.0]])},
        coords={
            "time": (
                ["time"],
                [100.0, 110.0],
                {"integration_time": {"data": 2.0, "dims": [], "attrs": {}}},
            )
        },
    )
    margin = 1.5 + op.POINTING_TIME_RANGE_MARGIN
    assert op._pointing_time_range(ds) == pytest.approx(
        (100.0 - margin, 110.0 + margin)
    )


# ---------------------------------------------------------------------------
# Per-BDF metadata expanded to the time axis (F43)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "values, bdf_start, expected",
    [
        ([1, 3], [0, 3, 6], [1, 1, 1, 3, 3, 3]),
        ([5, 6, 5], [0, 4, 8, 9], [5, 5, 5, 5, 6, 6, 6, 6, 5]),
        (["a_0", "b_1"], [0, 1, 3], ["a_0", "b_1", "b_1"]),
        ([7], [0, 2], [7, 7]),
    ],
)
def test_expand_per_bdf_to_time(values, bdf_start, expected):
    result = op._expand_per_bdf_to_time(
        np.array(values), {"bdf_names": [], "bdf_start": bdf_start}, len(expected), "x"
    )
    assert list(result) == expected


def test_expand_per_bdf_to_time_errors():
    with pytest.raises(RuntimeError, match="do not match"):
        op._expand_per_bdf_to_time(np.array([1, 2]), {"bdf_start": [0, 3]}, 3, "scan")
    with pytest.raises(RuntimeError, match="do not match"):
        op._expand_per_bdf_to_time(
            np.array([1, 2]), {"bdf_start": [0, 1, 3]}, 4, "scan"
        )
    # without time indices: only a single value can be broadcast
    result = op._expand_per_bdf_to_time(np.array([4, 4]), {}, 3, "scan")
    assert list(result) == [4, 4, 4]
    with pytest.raises(RuntimeError, match="without the time indices"):
        op._expand_per_bdf_to_time(np.array([4, 5]), {}, 3, "scan")


def test_per_bdf_values():
    descr = {
        "BDFPath": np.array(["b0", "b1"]),
        "scanNumber": np.array([1, 3]),
        "fieldId": np.array([0]),
        "per_bdf": {"BDFPath": np.array(["b0", "b1"]), "scanNumber": np.array([3, 1])},
    }
    np.testing.assert_array_equal(op._per_bdf_values(descr, "scanNumber"), [3, 1])
    # no per-BDF fieldId, single value in the partition
    np.testing.assert_array_equal(op._per_bdf_values(descr, "fieldId"), [0, 0])

    descr_no_per_bdf = {key: val for key, val in descr.items() if key != "per_bdf"}
    with pytest.raises(RuntimeError, match="several scanNumber"):
        op._per_bdf_values(descr_no_per_bdf, "scanNumber")

    misaligned = dict(
        descr, per_bdf={"BDFPath": np.array(["b1", "b0"]), "scanNumber": [3, 1]}
    )
    with pytest.raises(RuntimeError, match="not aligned"):
        op._per_bdf_values(misaligned, "scanNumber")
    wrong_len = dict(descr, per_bdf={"scanNumber": np.array([3])})
    with pytest.raises(RuntimeError, match="1 values for 2 BDFs"):
        op._per_bdf_values(wrong_len, "scanNumber")


def test_scan_name_coord_and_intents():
    """scan_name per integration and one scan_intents entry per intent (F43,
    F44)."""
    descr = {
        "BDFPath": np.array(["b0", "b1", "b2"]),
        "scanNumber": np.array([1, 3]),
        "scanIntent": np.array(
            ["CALIBRATE_PHASE#ON_SOURCE", "CALIBRATE_WVR#ON_SOURCE"]
        ),
        "per_bdf": {"scanNumber": np.array([1, 3, 1])},
    }
    coord, attrs = op._create_scan_name_coord_attrs(
        descr, {"bdf_start": [0, 2, 3, 5]}, 5
    )
    assert coord[0] == ["time"]
    assert list(coord[1]) == ["1", "1", "3", "1", "1"]
    assert attrs == {
        "scan_intents": ["CALIBRATE_PHASE#ON_SOURCE", "CALIBRATE_WVR#ON_SOURCE"]
    }
    assert all(type(intent) is str for intent in attrs["scan_intents"])

    single = dict(descr, scanIntent=np.array(["OBSERVE_TARGET#ON_SOURCE"]))
    assert op._create_scan_name_coord_attrs(single, {"bdf_start": [0, 2, 3, 5]}, 5)[
        1
    ] == {"scan_intents": ["OBSERVE_TARGET#ON_SOURCE"]}


def test_partition_without_per_bdf_multi_scan_raises(
    synth_interferometric, fake_time_loader
):
    """Several scans without per-BDF information cannot be assigned to the
    integrations: error instead of cycling the scan numbers (F43)."""
    truth = synth_interferometric
    part = partition_for_spw(truth, 0)
    descr = partition_descr_for(truth, part)
    del descr["per_bdf"]
    fake_time_loader(truth)
    with pytest.raises(RuntimeError, match="several scanNumber"):
        op.create_coordinates(read_asdm(truth.path), descr)


# ---------------------------------------------------------------------------
# Time variables, data variables, UVW (F45, F81)
# ---------------------------------------------------------------------------


def test_create_time_vars_broadcasts_per_time():
    """TIME_CENTROID[t, b] == actual_times[t] (no cyclic tiling, F45). The data
    are lazily indexed per-integration values (no memory per baseline) that
    give writeable arrays, like the other data variables."""
    time_vars = op._create_time_vars(
        np.array([1.0, 2.0, 3.0]), np.array([100.0, 101.0, 102.0]), 2
    )
    dims, values, attrs = time_vars["TIME_CENTROID"]
    assert dims == ["time", "baseline_id"]
    assert isinstance(values, xr.core.indexing.LazilyIndexedArray)
    assert isinstance(values.array, asdm_backend_arrays.PerTimeArray)
    variable = xr.Variable(dims, values)
    assert variable.shape == (3, 2)
    np.testing.assert_array_equal(variable.values, [[100, 100], [101, 101], [102, 102]])
    assert variable.values.flags.writeable
    assert attrs == {
        "units": "s",
        "scale": ASDM_TIME_SCALE,
        "format": ASDM_TIME_FORMAT,
        "type": "time",
    }
    dims, values, attrs = time_vars["EFFECTIVE_INTEGRATION_TIME"]
    np.testing.assert_array_equal(
        xr.Variable(dims, values).values, [[1, 1], [2, 2], [3, 3]]
    )
    assert attrs == {"units": "s", "type": "quantity"}

    sd_vars = op._create_time_vars(np.ones(2), np.array([5.0, 6.0]), 3, "antenna_name")
    dims, values, _ = sd_vars["TIME_CENTROID"]
    assert dims == ["time", "antenna_name"]
    np.testing.assert_array_equal(
        xr.Variable(dims, values).values, [[5, 5, 5], [6, 6, 6]]
    )

    with pytest.raises(ValueError, match="shape"):
        op._create_time_vars(np.ones(2), np.ones((2, 3)), 3)


@pytest.mark.parametrize("single_dish", [False, True])
def test_create_data_vars(single_dish):
    """VISIBILITY (or SPECTRUM for single dish), WEIGHT and FLAG with the dims of
    the dataset and no stray attrs (F41, F81). The chunking hints are set by
    create_correlated_xds for all the variables (see test_preferred_time_chunk*)."""
    second_dim = "antenna_name" if single_dish else "baseline_id"
    xds = xr.Dataset(
        coords={
            "time": np.arange(3.0),
            second_dim: np.arange(4),
            "frequency": np.arange(5.0),
            "polarization": ["XX", "YY"],
        }
    )
    time_indices = {"bdf_names": ["b0"], "bdf_start": [0, 3]}
    data_vars = op.create_data_vars(xds, ["b0"], 1, time_indices)
    correlated = "SPECTRUM" if single_dish else "VISIBILITY"
    assert set(data_vars) == {correlated, "WEIGHT", "FLAG"}
    dims = ["time", second_dim, "frequency", "polarization"]
    for var_dims, array, attrs in data_vars.values():
        assert var_dims == dims
        assert array.shape == (3, 4, 5, 2)
        assert "encoding" not in attrs
        assert "field_and_source_xds" not in attrs
    expected_class = (
        asdm_backend_arrays.SpectrumArray
        if single_dish
        else asdm_backend_arrays.VisibilityArray
    )
    assert isinstance(data_vars[correlated][1].array, expected_class)
    assert data_vars[correlated][2] == {"type": "quantity", "units": ""}
    assert data_vars["FLAG"][1].dtype == bool

    ds = xds.assign(data_vars)
    assert "encoding" not in ds[correlated].attrs


def correlated_like_xds(num_time=100, num_rows=10, num_chan=8, num_pol=2):
    """A dataset with the time-dependent variables of a correlated dataset:
    VISIBILITY (complex64), WEIGHT (float64), FLAG, TIME_CENTROID, and a time
    coordinate. The largest variables have num_rows * num_chan * num_pol * 8
    bytes per integration (1280 by default)."""
    dims = ("time", "baseline_id", "frequency", "polarization")
    shape = (num_time, num_rows, num_chan, num_pol)

    def zeros(dtype):
        return np.broadcast_to(np.zeros((), dtype=dtype), shape)

    return xr.Dataset(
        {
            "VISIBILITY": (dims, zeros(np.complex64)),
            "WEIGHT": (dims, zeros(np.float64)),
            "FLAG": (dims, zeros(bool)),
            "TIME_CENTROID": (dims[:2], np.zeros(shape[:2])),
        },
        coords={
            "time": np.arange(float(num_time)),
            "scan_name": ("time", np.full(num_time, "1")),
        },
    )


@pytest.mark.parametrize(
    "target_bytes, bdf_start, expected",
    [
        # memory target: 12800 bytes = 10 integrations of 1280 bytes
        (12800, [0, 100], 10),
        # regular BDFs of 50: 10 divides 50, chunk boundaries on BDF boundaries
        (12800, [0, 50, 100], 10),
        # regular BDFs of 48 (last one shorter): 8 is the largest divisor <= 10
        (12800, [0, 48, 96, 100], 8),
        # irregular BDFs: no alignment
        (12800, [0, 33, 100], 10),
        # small data: one chunk per BDF (the largest one)
        (2**30, [0, 50, 100], 50),
        (2**30, [0, 30, 100], 70),
        (2**30, [0, 100], 100),
        # integrations larger than the target: one integration per chunk
        (1000, [0, 50, 100], 1),
        # no (or inconsistent) BDF time indices: bounded by the time axis
        (2**30, None, 100),
        (2**30, [0, 50], 100),
        (2**30, [0, 60, 40, 100], 100),
        (12800, None, 10),
    ],
)
def test_preferred_time_chunk(target_bytes, bdf_start, expected):
    """Preferred chunk along time: within the memory target (largest variable),
    at most one BDF, aligned with regular BDFs (R: preferred_chunks time=1)."""
    xds = correlated_like_xds()
    time_indices = {} if bdf_start is None else {"bdf_start": bdf_start}
    assert op.preferred_time_chunk(xds, time_indices, target_bytes) == expected


def test_preferred_time_chunk_bytes_per_integration():
    """The largest time-dependent data variable sets the bytes per integration:
    SPECTRUM float32 with WEIGHT float64 counts 8 bytes per value."""
    xds = correlated_like_xds().drop_vars("VISIBILITY")
    assert op.preferred_time_chunk(xds, {"bdf_start": [0, 100]}, 12800) == 10
    xds = xds.drop_vars("WEIGHT")
    # FLAG (1 byte per value) and TIME_CENTROID only: 160 bytes per integration
    assert op.preferred_time_chunk(xds, {"bdf_start": [0, 100]}, 1600) == 10


def test_preferred_time_chunk_default_target_from_dask():
    """The default memory target is the dask "array.chunk-size" setting
    (DEFAULT_PREFERRED_CHUNK_BYTES when it is missing or invalid)."""
    import dask

    xds = correlated_like_xds()
    with dask.config.set({"array.chunk-size": "2560B"}):
        assert op._preferred_chunk_target_bytes() == 2560
        assert op.preferred_time_chunk(xds, {"bdf_start": [0, 100]}) == 2
    with dask.config.set({"array.chunk-size": "1MiB"}):
        assert op._preferred_chunk_target_bytes() == 2**20
    with dask.config.set({"array.chunk-size": "not a size"}):
        assert op._preferred_chunk_target_bytes() == op.DEFAULT_PREFERRED_CHUNK_BYTES
    with dask.config.set({"array.chunk-size": 0}):
        assert op._preferred_chunk_target_bytes() == op.DEFAULT_PREFERRED_CHUNK_BYTES
    assert op.DEFAULT_PREFERRED_CHUNK_BYTES == 128 * 2**20


@pytest.mark.parametrize(
    "time_chunk, num_integrations, expected",
    [
        (7, [60, 60, 60], 6),
        (7, [60, 60, 25], 6),
        (4, [200, 100], 4),
        (1, [5, 5], 1),
        # 61 is prime: no divisor in [4, 7]
        (7, [61, 61], 7),
        # not smaller than the BDFs, one BDF, irregular BDFs
        (10, [10, 10], 10),
        (5, [10], 5),
        (3, [3, 2, 2, 3], 3),
        (7, [60, 60, 61], 7),
        (9, [12, 12, 0], 9),
        (7, [0, 0], 7),
    ],
)
def test_align_time_chunk_with_bdfs(time_chunk, num_integrations, expected):
    assert (
        op._align_time_chunk_with_bdfs(time_chunk, np.array(num_integrations))
        == expected
    )


@pytest.mark.parametrize(
    "time_indices, num_time, expected",
    [
        ({"bdf_start": [0, 3, 5, 10]}, 10, [3, 2, 5]),
        ({"bdf_start": np.array([0, 10])}, 10, [10]),
        ({"bdf_start": [0, 3, 5, 9]}, 10, None),
        ({"bdf_start": [1, 3, 10]}, 10, None),
        ({"bdf_start": [0, 6, 3, 10]}, 10, None),
        ({"bdf_start": [0]}, 10, None),
        ({"bdf_names": []}, 10, None),
        ({}, 10, None),
        (None, 10, None),
    ],
)
def test_bdf_num_integrations(time_indices, num_time, expected):
    result = op._bdf_num_integrations(time_indices, num_time)
    if expected is None:
        assert result is None
    else:
        np.testing.assert_array_equal(result, expected)


def test_set_preferred_time_chunk():
    """preferred_chunks {"time": k} on every time-dependent variable except the
    time index (coordinates too), other encoding entries kept, input unchanged."""
    xds = correlated_like_xds(num_time=6)
    xds = xds.assign(
        {
            "UVW": (("time", "uvw_label"), np.zeros((6, 3))),
            "NO_TIME": (("baseline_id",), np.zeros(10)),
        }
    )
    xds["VISIBILITY"].encoding = {"dtype": "complex64", "preferred_chunks": {"x": 2}}

    result = op._set_preferred_time_chunk(xds, 4)

    for name in ("WEIGHT", "FLAG", "TIME_CENTROID", "UVW", "scan_name"):
        assert result[name].encoding == {"preferred_chunks": {"time": 4}}, name
    assert result["VISIBILITY"].encoding == {
        "dtype": "complex64",
        "preferred_chunks": {"x": 2, "time": 4},
    }
    assert result["NO_TIME"].encoding == {}
    assert result["time"].encoding == {}
    assert op._get_preferred_time_chunk(result) == 4
    # the input dataset is not modified
    assert xds["VISIBILITY"].encoding == {
        "dtype": "complex64",
        "preferred_chunks": {"x": 2},
    }
    assert xds["FLAG"].encoding == {}


def test_create_data_vars_no_time_dim():
    with pytest.raises(KeyError, match="time"):
        op.create_data_vars(xr.Dataset(), [""], 0, {"bdf_names": [], "bdf_start": []})


def test_create_uvw_data_var():
    time_len, baseline_id_len, uvw_label_len = 2, 3, 3
    dimension_sizes = {
        "time": time_len,
        "baseline_id": baseline_id_len,
        "uvw_label": uvw_label_len,
    }
    time = xr.DataArray(
        data=np.array([0, 10]) + 1.7e9,
        dims="time",
        attrs={"type": "time", "units": "s", "scale": "utc", "format": "unix"},
    )
    baseline_antenna1_name = xr.DataArray(["DA01", "DV01", "DA01"], dims="baseline_id")
    baseline_antenna2_name = xr.DataArray(["DV01", "DV01", "DA01"], dims="baseline_id")
    phase_center_direction = xr.DataArray(
        data=[[0.11, 0.15], [0.12, 0.16]],
        dims=["time", "sky_dir_label"],
        coords={"sky_dir_label": ["ra", "dec"]},
        attrs={"type": "sky_coord", "units": "rad", "frame": "icrs"},
    )
    antenna_position = xr.DataArray(
        data=[[100, 200, -300], [50, 100, 100]],
        dims=["antenna_name", "cartesian_pos_label"],
        coords={
            "antenna_name": ["DA01", "DV01"],
            "cartesian_pos_label": ["x", "y", "z"],
        },
    )

    uvw_data_var = op._create_uvw_data_var(
        dimension_sizes,
        time,
        baseline_antenna1_name,
        baseline_antenna2_name,
        antenna_position,
        phase_center_direction,
    )
    assert list(uvw_data_var) == ["UVW"]
    dims, array, attrs = uvw_data_var["UVW"]
    uvw_schema = xarray_dataclass_to_array_schema(UvwArray)
    assert not check_dimensions(dims, uvw_schema.dimensions)
    assert not check_dtype(array.dtype, uvw_schema.dtypes)
    assert not check_attributes(attrs, uvw_schema.attributes)
    assert array.shape == (time_len, baseline_id_len, uvw_label_len)
    assert isinstance(array.array, asdm_backend_arrays.UVWArray)


# ---------------------------------------------------------------------------
# Baselines, configuration and table lookups
# ---------------------------------------------------------------------------


def test_create_baseline_coords():
    names = ["A", "B", "C"]
    coords = op._create_baseline_coords(
        names, pyasdm.enumerations.CorrelationMode.CROSS_AND_AUTO
    )
    assert list(coords["baseline_antenna1_name"][1]) == ["A", "A", "B", "A", "B", "C"]
    assert list(coords["baseline_antenna2_name"][1]) == ["B", "C", "C", "A", "B", "C"]
    np.testing.assert_array_equal(coords["baseline_id"], np.arange(6))

    coords = op._create_baseline_coords(names, "AUTO_ONLY")
    assert list(coords["baseline_antenna1_name"][1]) == names
    assert list(coords["baseline_antenna2_name"][1]) == names

    with pytest.raises(RuntimeError, match="mode='CROSS_ONLY'"):
        op._create_baseline_coords(
            names, pyasdm.enumerations.CorrelationMode.CROSS_ONLY
        )


def test_partition_metadata_errors(synth_interferometric):
    truth = synth_interferometric
    asdm = read_asdm(truth.path)
    with pytest.raises(RuntimeError, match="configDescriptionId=5"):
        op._get_partition_config(asdm, {"configDescriptionId": np.array([5])})
    with pytest.raises(RuntimeError, match="exactly one configDescriptionId"):
        op._get_partition_config(asdm, {"configDescriptionId": np.array([0, 1])})
    with pytest.raises(RuntimeError, match="dataDescriptionId=9"):
        op._create_polarizations_coord(asdm, {"dataDescriptionId": np.array([9])})
    with pytest.raises(RuntimeError, match="spectralWindowId=9"):
        op._create_frequency_coord_attrs(asdm, 9)

    pol, spw_id, dd_id = op._create_polarizations_coord(
        asdm, {"dataDescriptionId": np.array([1])}
    )
    assert list(pol) == ["XX", "YY"]
    assert (spw_id, dd_id) == (1, 1)


def test_scans_metadata_filtered_by_execblock(synth_mosaic):
    truth = synth_mosaic
    asdm = read_asdm(truth.path)
    scans = op._get_scans_metadata(
        asdm, {"scanNumber": np.array([1, 3]), "execBlockId": np.array([0])}
    )
    assert sorted(scans["scanNumber"]) == [1, 3]
    scans = op._get_scans_metadata(
        asdm, {"scanNumber": np.array([1, 3]), "execBlockId": np.array([1])}
    )
    assert scans.empty


@pytest.mark.parametrize(
    "num_antenna, expected_output",
    [
        (1, ([0], [0])),
        (2, ([0, 0, 1], [1, 0, 1])),
        (3, ([0, 0, 1, 0, 1, 2], [1, 2, 2, 0, 1, 2])),
        (4, ([0, 0, 1, 0, 1, 2, 0, 1, 2, 3], [1, 2, 2, 3, 3, 3, 0, 1, 2, 3])),
    ],
)
def test_generate_baseline_antennax_id_as_in_bdf(num_antenna, expected_output):
    """Cross baselines (i, j), i < j, in BDF order, then the autos."""
    antenna1, antenna2 = op._generate_baseline_antennax_id_as_in_bdf(num_antenna)
    np.testing.assert_array_equal(antenna1, expected_output[0])
    np.testing.assert_array_equal(antenna2, expected_output[1])
    cross = synth.cross_baseline_pairs(num_antenna)
    assert (
        list(zip(antenna1[: len(cross)], antenna2[: len(cross)], strict=False)) == cross
    )


# ---------------------------------------------------------------------------
# Hand-made (conftest) ASDM, partition description without per_bdf
# ---------------------------------------------------------------------------


def mock_load_times_from_partition_bdfs(bdf_paths, scans_metadata):
    """Two integrations, 2024-08-10 (unix seconds), without BDF time indices."""
    times = np.array([1723291200.0, 1723291206.048])
    return times, np.array([6.048, 6.048]), times + 0.001, np.array([6.0, 6.0]), {}


def test_open_partition_conftest_radiometer_asdm(
    asdm_with_main_etc_data_description_polarization_field_source, monkeypatch, caplog
):
    """
    The conftest ASDM has an AUTO_ONLY RADIOMETER config with numAntenna=2 and
    12 antennaId entries (the first 2 are used, with a warning). Without per-BDF
    information and BDF time indices, the single scan/field of the partition is
    used for all the integrations (F79, F83, F43).
    """
    monkeypatch.setattr(
        op, "load_times_from_partition_bdfs", mock_load_times_from_partition_bdfs
    )
    with caplog.at_level(logging.WARNING):
        partition = op.open_partition(
            asdm_with_main_etc_data_description_polarization_field_source,
            {
                "execBlockId": np.array([0]),
                "fieldId": np.array([0]),
                "configDescriptionId": np.array([0]),
                "scanNumber": np.array([1]),
                "scanIntent": np.array(["CALIBRATE_POINTING#ON_SOURCE"]),
                "dataDescriptionId": np.array([0]),
                "spectralType": np.array(["BASEBAND_WIDE"]),
                "BDFPath": np.array(["/inexistent_test_path/foo"]),
            },
        )

    assert "numAntenna=2 but 12 antennaId entries" in caplog.text
    ds = partition.ds
    assert ds.attrs["type"] == "radiometer"
    np.testing.assert_allclose(ds.time.values, [1723291200.0, 1723291206.048])
    assert list(ds.scan_name.values) == ["1", "1"]
    assert list(ds.field_name.values) == ["J0423-0120_0", "J0423-0120_0"]
    assert ds.scan_name.attrs["scan_intents"] == ["CALIBRATE_POINTING#ON_SOURCE"]
    assert list(ds.baseline_antenna1_name.values) == ["CM01", "CM03"]
    assert list(ds.baseline_antenna2_name.values) == ["CM01", "CM03"]
    np.testing.assert_allclose(
        ds.TIME_CENTROID.values, [[1723291200.001] * 2, [1723291206.049] * 2]
    )
    assert ds.attrs["data_groups"]["base"]["uvw"] == "UVW"
    assert not check_datatree(partition)

    weight = ds.WEIGHT[0, 0, 0, 0]
    assert weight.values == 1.0
    uvw = ds.UVW[0:1, :, :]
    # auto-correlations only
    np.testing.assert_allclose(uvw.values, np.zeros((1, 2, 3)), atol=1e-9)
