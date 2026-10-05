"""
The ``xradio_msv2`` engine against the converter (fast tier): an MSv2 opened
with ``xr.open_datatree(ms, engine=MSv2BackendEntrypoint, ...)`` equals the
processing set ``convert_msv2_to_processing_set`` writes for it, opened with
``open_processing_set`` (the reference), on generated MSs (conftest.py).
Also: open_msv2, the partition memo, partition errors, open-time reads and
file handles.
"""

import contextlib
import hashlib
import json
import os
import shutil
import threading
import time
import warnings

import numpy as np
import pytest
import xarray as xr

from _xradio_xarray_backends import MSv2BackendEntrypoint
from xradio.measurement_set import (
    convert_msv2_to_processing_set,
    open_msv2,
    open_processing_set,
)
from xradio.measurement_set._utils._msv2 import backend_open, partition_cache
from xradio.measurement_set._utils._msv2._tables import read_rows
from xradio.measurement_set._utils._msv2.partition_cache import PARTITIONS_MEMO
from xradio.schema.check import check_datatree
from xradio.testing.measurement_set.equivalence import (
    assert_nodes_identical,
    assert_processing_sets_equivalent,
)

ENGINE = MSv2BackendEntrypoint


def _ddi(*ddis):
    """partition_filter: the partitions of these DDIs."""
    return lambda partition: partition["DATA_DESC_ID"][0] in ddis


def _target(partition):
    return "OBSERVE_TARGET#ON_SOURCE" in partition["OBS_MODE"][0]


# case: (variant of backend_ms, options of the converter and of the engine)
CASES = {
    "dense": ("dense", {}),
    "dense_time3_no_pointing": (
        "dense",
        {"main_chunksize": {"time": 3}, "with_pointing": False},
    ),
    "sparse_dup_field": ("sparse_dup", {"partition_scheme": ["FIELD_ID"]}),
    "baseline_major_time7": ("baseline_major", {"main_chunksize": {"time": 7}}),
    "time_descending_scan": ("time_descending", {"partition_scheme": ["SCAN_NUMBER"]}),
    "shuffled_phase_cal_interpolate": ("shuffled", {"phase_cal_interpolate": True}),
    "rich_field_scan_filtered": (
        "rich",
        {
            "partition_scheme": ["FIELD_ID", "SCAN_NUMBER"],
            "partition_filter": _ddi(3),
        },
    ),
    "rich_state_time3": (
        "rich",
        {
            "partition_scheme": ["STATE_ID"],
            "main_chunksize": {"time": 3},
            "partition_filter": _ddi(1, 2),
        },
    ),
    "rich_sub_scan_pointing_interpolate": (
        "rich",
        {
            "partition_scheme": ["SUB_SCAN_NUMBER"],
            "pointing_interpolate": True,
            "partition_filter": _ddi(0, 2),
        },
    ),
    "rich_source_ephemeris_interpolate": (
        "rich",
        {
            "partition_scheme": ["SOURCE_ID"],
            "ephemeris_interpolate": True,
            "partition_filter": _ddi(3),
        },
    ),
    "rich_sys_cal_interpolate_pointing_chunks": (
        "rich",
        {
            "sys_cal_interpolate": True,
            "pointing_chunksize": 0.00001,
            "partition_filter": _ddi(0, 3),
        },
    ),
    "rich_intent_filter": ("rich", {"partition_filter": _target}),
    "wsp_partial_field": ("wsp_partial", {"partition_scheme": ["FIELD_ID"]}),
    "no_weight_frequency_chunks": (
        "no_weight",
        {"main_chunksize": {"time": 4, "frequency": 8}},
    ),
    "single_dish_antenna1": (
        "single_dish",
        {"partition_scheme": ["ANTENNA1"], "partition_filter": _ddi(1)},
    ),
    "single_dish": ("single_dish", {}),
    "antenna1_without_autocorrelations": ("dense", {"partition_scheme": ["ANTENNA1"]}),
}


@pytest.fixture(scope="module")
def converted(backend_ms, tmp_path_factory):
    """``converted(case)``: the processing set the converter writes for a
    case of CASES (written on first use)."""
    base = tmp_path_factory.mktemp("msv2_engine_reference")
    paths: dict[str, str] = {}

    def get(case: str) -> str:
        if case not in paths:
            variant, options = CASES[case]
            out = str(base / f"{case}.ps.zarr")
            convert_msv2_to_processing_set(backend_ms(variant), out, **options)
            paths[case] = out
        return paths[case]

    yield get
    shutil.rmtree(base, ignore_errors=True)


@pytest.mark.parametrize("case", list(CASES))
def test_engine_equals_the_converted_processing_set(
    case, backend_ms, converted, ms_copy
):
    """
    The engine's processing set equals the converted one, on a copy of the
    MS: with chunks={} against array_backend="dask" (cold: the partitions
    are computed and stored in the copy), with chunks=None against
    array_backend="xarray" (warm: the partitions read from the copy), and
    with the partition cache off. Same nodes, identical datasets (dates
    aside), the converter's dask chunks of the main data variables, lazy
    selections and accessors.
    """
    variant, options = CASES[case]
    # (same name as the converted MS: the MSv4 names derive from it)
    msname = ms_copy(variant, name=f"{variant}.ms")
    reference = converted(case)
    partition_cache.clear_partition_memo()

    cold = xr.open_datatree(
        msname, engine=ENGINE, chunks={}, partition_cache="auto", **options
    )
    assert PARTITIONS_MEMO.stats["computed"] == 1
    assert os.path.isdir(os.path.join(msname, partition_cache.SUBTABLE_NAME))
    assert_processing_sets_equivalent(
        cold, open_processing_set(reference, array_backend="dask")
    )

    partition_cache.clear_partition_memo()
    warm = xr.open_datatree(
        msname, engine=ENGINE, chunks=None, partition_cache="auto", **options
    )
    assert PARTITIONS_MEMO.stats["stored hits"] == 1
    assert PARTITIONS_MEMO.stats["computed"] == 0
    assert_processing_sets_equivalent(
        warm,
        open_processing_set(reference, array_backend="xarray"),
        chunks=False,
        accessors=False,
    )

    off = xr.open_datatree(msname, engine=ENGINE, chunks={}, **options)
    assert PARTITIONS_MEMO.stats["computed"] == 0  # (off: not memoised)
    assert_nodes_identical(off, open_processing_set(reference))
    if case == "antenna1_without_autocorrelations":
        assert not cold.children and cold.attrs == {"type": "processing_set"}


def _store_contents(path: str) -> tuple[dict, dict, dict]:
    """The sha256 of every chunk file, every array's zarr.json and every
    group's attributes, without dates."""

    def strip(obj):
        if isinstance(obj, dict):
            return {
                k: strip(v)
                for k, v in obj.items()
                if k not in ("creation_date", "date")
            }
        if isinstance(obj, list):
            return [strip(v) for v in obj]
        return obj

    chunks, arrays, groups = {}, {}, {}
    for dirpath, _, filenames in os.walk(path):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            rel = os.path.relpath(file_path, path)
            with open(file_path, "rb") as f:
                content = f.read()
            if filename != "zarr.json":
                chunks[rel] = hashlib.sha256(content).hexdigest()
                continue
            metadata = strip(json.loads(content))
            if metadata["node_type"] == "array":
                arrays[rel] = metadata
            else:
                groups[rel] = metadata.get("attributes", {})
    return chunks, arrays, groups


@pytest.mark.parametrize(
    "case",
    [
        "dense",
        "rich_sub_scan_pointing_interpolate",
        "single_dish_antenna1",
        "rich_state_time3",
    ],
)
def test_to_zarr_writes_the_converted_processing_set(
    case, backend_ms, converted, tmp_path
):
    """
    The engine's tree (chunks={}) written by to_zarr has the chunk files,
    array metadata and group attributes of the converted processing set, and
    the schema check finds what it finds in the converted one (the generated
    MSs give a few issues in sub-datasets).

    With time chunks (main_chunksize), the coordinates along time
    (scan_name, field_name) are dask arrays in the data's time chunks (so
    that Dataset.chunks is defined), which to_zarr writes in those chunks,
    where the converter writes one chunk: their values are the same.
    """
    variant, options = CASES[case]
    tree = xr.open_datatree(backend_ms(variant), engine=ENGINE, chunks={}, **options)
    reference = open_processing_set(converted(case))
    assert str(check_datatree(tree)) == str(check_datatree(reference))
    out = str(tmp_path / "engine.ps.zarr")
    tree.to_zarr(out, mode="w")
    chunks_e, arrays_e, groups_e = _store_contents(out)
    chunks_r, arrays_r, groups_r = _store_contents(converted(case))
    time_chunked = "main_chunksize" in options
    if time_chunked:
        rechunked = {"scan_name", "field_name"}

        def keep(rel):
            return rel.split(os.sep)[1] not in rechunked

        chunks_e = {k: v for k, v in chunks_e.items() if keep(k)}
        chunks_r = {k: v for k, v in chunks_r.items() if keep(k)}
        for rel in [k for k in arrays_r if not keep(k)]:
            meta_e, meta_r = arrays_e.pop(rel), arrays_r.pop(rel)
            assert meta_e["chunk_grid"] != meta_r["chunk_grid"]
            meta_e.pop("chunk_grid"), meta_r.pop("chunk_grid")
            assert meta_e == meta_r, rel
        written = open_processing_set(out)
        assert_nodes_identical(written, reference)
    assert sorted(chunks_e) == sorted(chunks_r)
    assert [name for name in chunks_r if chunks_e[name] != chunks_r[name]] == []
    assert arrays_e == arrays_r
    assert groups_e == groups_r


def test_open_msv2(backend_ms, converted):
    """open_msv2 equals open_processing_set of the converted processing set,
    for both array backends and with scan_intents."""
    variant, options = CASES["rich_intent_filter"]
    msname, reference = backend_ms(variant), converted("rich_intent_filter")
    for array_backend in ("dask", "xarray"):
        engine = open_msv2(msname, array_backend=array_backend, **options)
        assert_processing_sets_equivalent(
            engine,
            open_processing_set(reference, array_backend=array_backend),
            chunks=array_backend == "dask",
            accessors=False,
        )
    intents = ["OBSERVE_TARGET#ON_SOURCE"]
    engine = open_msv2(msname, scan_intents=intents, **options)
    assert_nodes_identical(engine, open_processing_set(reference, scan_intents=intents))
    with pytest.raises(ValueError, match="array_backend"):
        open_msv2(msname, array_backend="numpy")


def test_drop_variables_equal_deleted_variables(backend_ms, converted):
    """drop_variables of main data variables equals deleting them from the
    converted processing set (variables and data group roles)."""
    variant, options = CASES["dense"]
    reference = open_processing_set(converted("dense"))
    for node in reference.children.values():
        node.xr_ms.delete_data_variables(["WEIGHT", "FLAG"])
    engine = xr.open_datatree(
        backend_ms(variant),
        engine=ENGINE,
        chunks={},
        drop_variables=["WEIGHT", "FLAG"],
        **options,
    )
    assert_nodes_identical(engine, reference)
    for node in engine.children.values():
        assert "weight" not in node.attrs["data_groups"]["base"]


def test_relative_path_and_chdir(backend_ms, tmp_path, monkeypatch):
    """The lazy arrays read the MS by its absolute path: a relative path
    opened before a chdir still reads."""
    msname = backend_ms("dense")
    monkeypatch.chdir(os.path.dirname(msname))
    tree = xr.open_datatree(os.path.basename(msname), engine=ENGINE, chunks={})
    expected = tree["dense_0"].VISIBILITY.isel(time=slice(0, 3)).values
    monkeypatch.chdir(tmp_path)
    np.testing.assert_array_equal(
        tree["dense_0"].VISIBILITY.isel(time=slice(0, 3)).values, expected
    )


@pytest.fixture
def grid_reads(monkeypatch):
    """The MAIN grid reads: (column, number of rows) of every
    read_rows_to_grid call."""
    reads = []
    read_rows_to_grid = read_rows.read_rows_to_grid

    def spy(table, col, plan, *args, **kwargs):
        reads.append((col, int(plan.rows.size)))
        return read_rows_to_grid(table, col, plan, *args, **kwargs)

    monkeypatch.setattr(read_rows, "read_rows_to_grid", spy)
    return reads


def test_open_reads_no_main_data_and_a_selection_reads_its_rows(backend_ms, grid_reads):
    """Opening reads no MAIN data column; a selection of 2 times and 1
    channel under chunks={} reads the rows of those 2 times only."""
    tree = xr.open_datatree(
        backend_ms("dense"), engine=ENGINE, chunks={}, main_chunksize={"time": 10}
    )
    assert grid_reads == []
    values = (
        tree["dense_0"].VISIBILITY.isel(time=slice(0, 2), frequency=slice(0, 1)).values
    )
    assert values.shape == (2, 10, 1, 2)
    assert grid_reads == [("DATA", 2 * 10)]


def test_large_tree_warning(backend_ms, monkeypatch):
    monkeypatch.setattr(backend_open, "LARGE_TREE_PARTITIONS", 3)
    with pytest.warns(UserWarning, match="Opening 4 partitions"):
        xr.open_datatree(backend_ms("dense"), engine=ENGINE)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        xr.open_datatree(backend_ms("dense"), engine=ENGINE, partition_filter=_ddi(0))


def _ms_fds(msname: str) -> list[str]:
    root = os.path.realpath(msname)
    found = []
    for fd in os.listdir("/proc/self/fd"):
        with contextlib.suppress(OSError):
            target = os.readlink(f"/proc/self/fd/{fd}")
            if target.startswith(root + os.sep):
                found.append(target)
    return found


@pytest.fixture
def failing_partitions(monkeypatch):
    """Make the partitions of the given DDIs fail to open."""
    open_partition = backend_open.open_partition
    failing: set[int] = set()

    def spy(path, partition_info, *args, **kwargs):
        if partition_info["DATA_DESC_ID"][0] in failing:
            raise ValueError(
                f"simulated failure of DDI {partition_info['DATA_DESC_ID']}"
            )
        return open_partition(path, partition_info, *args, **kwargs)

    monkeypatch.setattr(backend_open, "open_partition", spy)
    return failing


def test_on_partition_error(backend_ms, failing_partitions, monkeypatch):
    """skip (default): a partition that fails is left out with a warning and
    a logged traceback (the others keep their names), RuntimeError only if
    all fail; raise: a RuntimeError naming the node."""

    class Logger:
        errors: list[str] = []

        def error(self, message):
            self.errors.append(message)

        def info(self, message):
            pass

        debug = info

    logger = Logger()
    monkeypatch.setattr(backend_open, "xradio_logger", lambda: logger)
    msname = backend_ms("dense")
    failing_partitions.add(1)
    with pytest.warns(RuntimeWarning, match=r"Partition 1 \(dense_1\)") as record:
        tree = xr.open_datatree(msname, engine=ENGINE)
    assert len([w for w in record if issubclass(w.category, RuntimeWarning)]) == 1
    assert sorted(tree.children) == ["dense_0", "dense_2", "dense_3"]
    assert len(logger.errors) == 1
    assert "simulated failure" in logger.errors[0] and "Traceback" in logger.errors[0]
    with pytest.raises(RuntimeError, match=r"Partition 1 \(dense_1\) of .* could not"):
        xr.open_datatree(msname, engine=ENGINE, on_partition_error="raise")
    failing_partitions.update({0, 2, 3})
    with pytest.raises(RuntimeError, match="None of the 4 selected partitions"):
        with pytest.warns(RuntimeWarning):
            xr.open_datatree(msname, engine=ENGINE)
    with pytest.raises(ValueError, match="on_partition_error"):
        xr.open_datatree(msname, engine=ENGINE, on_partition_error="ignore")


@pytest.mark.skipif(not os.path.isdir("/proc/self/fd"), reason="needs /proc")
def test_no_ms_file_left_open(backend_ms, tmp_path, failing_partitions):
    """No file of the MS is open after an open, a read, a failed open and an
    open that skipped a partition."""
    msname = str(tmp_path / "copy.ms")
    shutil.copytree(backend_ms("rich"), msname)
    tree = xr.open_datatree(msname, engine=ENGINE, chunks={})
    assert _ms_fds(msname) == []
    tree["copy_1"].VISIBILITY.values  # noqa: B018
    assert _ms_fds(msname) == []
    failing_partitions.add(2)
    with pytest.warns(RuntimeWarning):
        xr.open_datatree(msname, engine=ENGINE)
    assert _ms_fds(msname) == []
    with pytest.raises(RuntimeError):
        xr.open_datatree(msname, engine=ENGINE, on_partition_error="raise")
    assert _ms_fds(msname) == []


def test_partition_filter(backend_ms):
    """The filter gets copies of the descriptions; the MSv4s of the selected
    partitions are numbered as by the converter; selecting none raises."""
    seen = []

    def mutating_filter(partition):
        seen.append(partition)
        keep = partition["DATA_DESC_ID"][0] in (1, 3)
        partition["DATA_DESC_ID"] = [99]
        return keep

    msname = backend_ms("dense")
    tree = xr.open_datatree(msname, engine=ENGINE, partition_filter=mutating_filter)
    assert len(seen) == 4
    assert sorted(tree.children) == ["dense_0", "dense_1"]
    assert tree["dense_1"].frequency.attrs["spectral_window_name"].endswith("_1")
    with pytest.raises(
        RuntimeError, match="No partitions selected by partition_filter"
    ):
        xr.open_datatree(msname, engine=ENGINE, partition_filter=lambda p: False)


def test_empty_main_is_an_empty_processing_set(backend_ms, tmp_path):
    """An MS without MAIN rows opens as the empty processing set that the
    converter writes for it (with any cache mode); a partition_filter
    raises, as in the converter."""
    from casacore import tables

    msname = str(tmp_path / "empty.ms")
    with tables.table(backend_ms("dense"), ack=False) as main_tb:
        none = main_tb.selectrows([])
        none.copy(msname, deep=True).close()  # (MAIN without rows, sub-tables)
        none.close()
    out = str(tmp_path / "empty.ps.zarr")
    convert_msv2_to_processing_set(msname, out)
    reference = open_processing_set(out)
    assert reference.attrs["type"] == "processing_set" and not reference.children
    for mode in ("off", "auto", "read"):
        tree = xr.open_datatree(msname, engine=ENGINE, partition_cache=mode)
        assert dict(tree.attrs) == dict(reference.attrs) and not tree.children
    assert not open_msv2(msname).children
    for opener in (convert_msv2_to_processing_set, None):
        with pytest.raises(
            RuntimeError, match="No partitions selected by partition_filter"
        ):
            if opener is None:
                xr.open_datatree(msname, engine=ENGINE, partition_filter=_target)
            else:
                opener(msname, out, partition_filter=_target, persistence_mode="w")


def test_every_table_open_holds_the_casatools_lock(ms_copy, monkeypatch):
    """With casatools, every table of the MS that an open reads is opened
    holding the process-wide casatools lock: partitions computed, read from
    the stored row and from the memo (simulated here: casatools_serialized
    gives the lock, and python-casacore's table opens are recorded)."""
    from casacore import tables

    from xradio._utils._casacore import tables as xradio_tables

    msname = os.path.abspath(ms_copy("rich"))
    partition_cache.clear_partition_memo()
    xr.open_datatree(msname, engine=ENGINE, partition_cache="auto")  # stores
    partition_cache.clear_partition_memo()

    lock = xradio_tables.CASATOOLS_LOCK
    unlocked = []
    table_init = tables.table.__init__

    def spy(self, tablename="", *args, **kwargs):
        name = os.path.abspath(str(tablename))
        if name.startswith(msname) and not getattr(lock._held, "locks", None):
            unlocked.append(os.path.relpath(name, msname))
        table_init(self, tablename, *args, **kwargs)

    monkeypatch.setattr(xradio_tables, "uses_casatools", lambda: True)
    monkeypatch.setattr(tables.table, "__init__", spy)
    sources = []
    load = backend_open.load_or_create_partitions

    def recorded(*args, **kwargs):
        result = load(*args, **kwargs)
        sources.append(result.source)
        return result

    monkeypatch.setattr(backend_open, "load_or_create_partitions", recorded)
    for mode in ("read", "read", "off"):
        tree = xr.open_datatree(
            msname, engine=ENGINE, partition_cache=mode, partition_scheme=[]
        )
        assert tree.children
    assert sources == ["stored", "memo", "fresh"]
    assert unlocked == []


def test_option_errors(backend_ms, monkeypatch):
    msname = backend_ms("dense")
    with pytest.raises(TypeError, match="partition_scheme"):
        xr.open_datatree(msname, engine=ENGINE, partition_scheme="FIELD_ID")
    with pytest.raises(ValueError, match="NOT_A_KEY"):
        xr.open_datatree(msname, engine=ENGINE, partition_scheme=["NOT_A_KEY"])
    with pytest.raises(ValueError, match="partition_cache"):
        xr.open_datatree(msname, engine=ENGINE, partition_cache="sometimes")
    monkeypatch.setenv("XRADIO_MSV2_PARTITION_CACHE", "sometimes")
    with pytest.raises(ValueError, match="XRADIO_MSV2_PARTITION_CACHE"):
        xr.open_datatree(msname, engine=ENGINE)
    with pytest.raises(TypeError, match="drop_variables"):
        xr.open_datatree(msname, engine=ENGINE, drop_variables=3, partition_cache="off")
    with pytest.raises(TypeError, match="partition_filter"):
        xr.open_datatree(
            msname, engine=ENGINE, partition_filter=[0], partition_cache="off"
        )


# --- the partition memo --------------------------------------------------------


def test_memo_modes(ms_copy, monkeypatch):
    msname = ms_copy("dense")
    partition_cache.clear_partition_memo()
    path = os.path.abspath(msname)
    first = partition_cache.load_or_create_partitions(path, [], "read")
    assert (first.source, first.status) == ("fresh", "memory:mode-read")
    first.partitions[0]["DATA_DESC_ID"] = [99]  # copies: the memo is unchanged
    for mode in ("auto", "read"):
        result = partition_cache.load_or_create_partitions(path, [], mode)
        assert (result.source, result.status) == ("memo", "hit-memory")
        assert result.partitions[0]["DATA_DESC_ID"] == [0]
    result = partition_cache.load_or_create_partitions(path, [], "rebuild")
    assert result.source == "fresh" and PARTITIONS_MEMO.stats["computed"] == 2
    result = partition_cache.load_or_create_partitions(path, [], "off")
    assert result.status == "memory:mode-off" and PARTITIONS_MEMO.stats["computed"] == 2
    # by scheme key: another scheme is computed; the same keys in another
    # order, or with mandatory keys, are the same scheme
    other = partition_cache.load_or_create_partitions(
        path, ["FIELD_ID", "SCAN_NUMBER"], "auto"
    )
    assert other.source == "fresh"
    same = partition_cache.load_or_create_partitions(
        path, ["SCAN_NUMBER", "DATA_DESC_ID", "FIELD_ID"], "auto"
    )
    assert same.source == "memo" and same.partitions == other.partitions
    # the environment variable sets the default mode
    monkeypatch.setenv("XRADIO_MSV2_PARTITION_CACHE", "auto")
    assert partition_cache.resolve_partition_cache_mode(None) == "auto"
    assert partition_cache.resolve_partition_cache_mode("off") == "off"


def test_memo_hit_computes_nothing(backend_ms, monkeypatch):
    msname = backend_ms("rich")
    partition_cache.clear_partition_memo()
    cold = xr.open_datatree(msname, engine=ENGINE, partition_cache="read")
    with monkeypatch.context() as m:
        m.setattr(
            partition_cache,
            "create_partitions_with_main_rows",
            lambda *a, **k: pytest.fail("partitions computed"),
        )
        monkeypatch.setenv("XRADIO_MSV2_PARTITION_CACHE", "read")
        warm = xr.open_datatree(msname, engine=ENGINE)
    assert sorted(warm.children) == sorted(cold.children)


def test_memo_follows_changes_of_the_ms(backend_ms, tmp_path):
    """A change of a MAIN key column makes the next open compute the
    partitions again (the tree then equals the converter's of the changed
    MS)."""
    from casacore import tables

    msname = str(tmp_path / "changed.ms")
    shutil.copytree(backend_ms("dense"), msname)
    partition_cache.clear_partition_memo()
    before = xr.open_datatree(
        msname, engine=ENGINE, partition_cache="auto", partition_scheme=["FIELD_ID"]
    )
    assert len(before.children) == 4
    with tables.table(msname, readonly=False, ack=False) as main_tb:
        field = main_tb.getcol("FIELD_ID")
        field[::2] = 1
        main_tb.putcol("FIELD_ID", field)
    after = xr.open_datatree(
        msname,
        engine=ENGINE,
        chunks={},
        partition_cache="auto",
        partition_scheme=["FIELD_ID"],
    )
    assert PARTITIONS_MEMO.stats["computed"] == 2
    convert_msv2_to_processing_set(
        msname, str(tmp_path / "changed.ps.zarr"), partition_scheme=["FIELD_ID"]
    )
    assert len(after.children) == 8
    assert_nodes_identical(
        after, open_processing_set(str(tmp_path / "changed.ps.zarr"))
    )


def test_memo_computes_once_for_concurrent_opens(backend_ms):
    msname = backend_ms("dense")
    partition_cache.clear_partition_memo()
    path = os.path.abspath(msname)
    results, errors = [], []

    def load():
        try:
            results.append(partition_cache.load_or_create_partitions(path, [], "read"))
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=load) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == [] and len(results) == 4
    assert PARTITIONS_MEMO.stats["computed"] == 1
    assert sorted(r.source for r in results) == ["fresh", "memo", "memo", "memo"]


def test_memo_bounds(backend_ms, monkeypatch):
    monkeypatch.setattr(PARTITIONS_MEMO, "max_entries", 2)
    partition_cache.clear_partition_memo()
    path = os.path.abspath(backend_ms("rich"))
    for scheme in ([], ["FIELD_ID"], ["SCAN_NUMBER"]):
        partition_cache.load_or_create_partitions(path, scheme, "read")
    assert len(PARTITIONS_MEMO) == 2
    assert partition_cache.memo_key(path, []) not in PARTITIONS_MEMO
    monkeypatch.setattr(PARTITIONS_MEMO, "max_bytes", 1)
    partition_cache.load_or_create_partitions(path, ["STATE_ID"], "read")
    assert len(PARTITIONS_MEMO) == 1  # (the most recent entry is kept)


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs os.fork")
@pytest.mark.filterwarnings("ignore:This process .* is multi-threaded")
def test_memo_is_reset_in_a_fork_child(backend_ms):
    partition_cache.clear_partition_memo()
    partition_cache.load_or_create_partitions(
        os.path.abspath(backend_ms("dense")), [], "read"
    )
    assert len(PARTITIONS_MEMO) == 1
    with PARTITIONS_MEMO._lock:
        pid = os.fork()
        if pid == 0:  # child
            ok = len(PARTITIONS_MEMO) == 0 and PARTITIONS_MEMO._lock.acquire(timeout=5)
            os._exit(0 if ok else 1)
    deadline = time.monotonic() + 60
    while not (done := os.waitpid(pid, os.WNOHANG))[0]:
        if time.monotonic() > deadline:  # (deadlocked: killed)
            os.kill(pid, 9)
            done = os.waitpid(pid, 0)
            break
        time.sleep(0.05)
    assert os.waitstatus_to_exitcode(done[1]) == 0
    assert len(PARTITIONS_MEMO) == 1
