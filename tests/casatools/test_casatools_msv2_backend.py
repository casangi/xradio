"""
The ``xradio_msv2`` xarray engine with casatools (python-casacore not
installed: the casatools test workflow), on copies of the test MSs:

- the engine's processing set, written with ``to_zarr`` (which computes every
  lazy block in dask threads, through the process-wide casatools lock), has
  the fingerprint of the python-casacore conversions of the reference
  (reference_python_casacore.json), but for the known widening of dtypes
  (see test_casatools_conversion.py);
- 8 dask threads read the values of a synchronous read (VLASS: the main data
  variables and the lazy pointing_xds);
- opening never writes into the MS (the partition cache is not stored with
  casatools: "casatools only", logged once);
- partitions stored in the MS (the sub-table built here with casatools, as
  python-casacore's writer builds it) are read and used, empty run arrays
  included, and remove_msv2_partition_cache removes them.

Skipped where python-casacore is installed (the Linux and macOS workflows).
"""

import contextlib
import hashlib
import json
import os
import shutil
import threading

import numpy as np
import pytest

from tests.casatools import reference as ref

ref.skip_unless_casatools_backend()

# The main dataset variables casatools gives wider dtypes (as in
# test_casatools_conversion.py)
WIDENED_VARIABLES = ("VISIBILITY", "SPECTRUM", "WEIGHT")


@pytest.fixture(scope="module")
def reference() -> dict:
    return ref.load_reference()


def _widened_as_known(line: str) -> bool:
    """Whether a widened variable ("<msv4>/<path>/<var>: <dtype> -> <dtype>",
    fingerprint_differences) is a main dataset variable read from a Float or
    Complex MAIN column."""
    where = line.split(":", 1)[0]
    _, path, var = where.split("/", 2)
    return path == "." and var.startswith(WIDENED_VARIABLES)


def copy_ms(ms_name: str, folder) -> str:
    """A copy of a test MS in ``folder`` (with the same name: the MSv4 names
    derive from it)."""
    target = folder / ms_name
    shutil.copytree(ref.ms_path(ms_name), target, symlinks=True)
    return str(target)


def open_engine(msname: str, **options):
    """xr.open_datatree with the engine class (chunks={})."""
    import xarray as xr

    from _xradio_xarray_backends import MSv2BackendEntrypoint

    return xr.open_datatree(msname, engine=MSv2BackendEntrypoint, chunks={}, **options)


def file_digests(path: str) -> dict[str, str]:
    """sha256 of every file under ``path`` (table.lock files included)."""
    digests = {}
    for dirpath, _, filenames in os.walk(path):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            with open(file_path, "rb") as f:
                digests[os.path.relpath(file_path, path)] = hashlib.sha256(
                    f.read()
                ).hexdigest()
    return digests


@contextlib.contextmanager
def cache_statuses(monkeypatch):
    """The status of every partition lookup of the engine (a list, filled
    while active)."""
    from xradio.measurement_set._utils._msv2 import backend_open

    statuses = []
    load = backend_open.load_or_create_partitions

    def spy(*args, **kwargs):
        result = load(*args, **kwargs)
        statuses.append(result.status)
        return result

    with monkeypatch.context() as patch:
        patch.setattr(backend_open, "load_or_create_partitions", spy)
        yield statuses


@pytest.mark.parametrize("case", list(ref.CONVERSION_CASES))
def test_engine_tree_has_the_reference_fingerprint(case, reference, tmp_path):
    """
    The engine's tree (chunks={}, the cache off) written with to_zarr has the
    fingerprint of the python-casacore conversion of the case (default
    variant), but for the known widening.
    """
    spec = ref.CONVERSION_CASES[case]
    msname = copy_ms(spec["ms"], tmp_path)
    tree = open_engine(
        msname,
        partition_cache="off",
        partition_scheme=spec["partition_scheme"],
        with_pointing=ref.CONVERSION_DEFAULTS["with_pointing"],
        **spec["options"],
    )
    out_file = tmp_path / f"{case}.ps.zarr"
    try:
        tree.to_zarr(str(out_file), mode="w")
        actual = ref.processing_set_fingerprint(out_file)
    finally:
        shutil.rmtree(out_file, ignore_errors=True)

    expected = dict(reference["conversions"][case], nodes=reference["nodes"])
    widened = []
    diffs = ref.fingerprint_differences(expected, actual, widened)
    assert not diffs, "\n".join(diffs)
    unexpected = [line for line in widened if not _widened_as_known(line)]
    assert not unexpected, "\n".join(unexpected)


def _block_digests(arr):
    """A dask array of the digests of every block of ``arr`` (one uint64
    per block): the blocks are hashed where they are computed, so that the
    values are compared without being kept."""

    def digest(block):
        block = np.ascontiguousarray(block)
        h = hashlib.blake2b(digest_size=8)
        h.update(f"{block.dtype}|{block.shape}|".encode())
        h.update(block.tobytes())
        value = np.frombuffer(h.digest(), dtype=np.uint64)[0]
        return np.full((1,) * block.ndim, value, dtype=np.uint64)

    return arr.map_blocks(
        digest, dtype=np.uint64, chunks=tuple((1,) * len(c) for c in arr.chunks)
    )


def test_dask_threads_read_the_values(tmp_path):
    """
    8 dask threads, starting with empty index memos, read the values of a
    synchronous read: every main data variable and every lazy pointing_xds
    variable of VLASS (casatools is called by one thread at a time).
    """
    import dask

    from xradio.measurement_set._utils._msv2 import backend_arrays, backend_pointing
    from xradio.testing.measurement_set.equivalence import main_data_variables

    msname = copy_ms(ref.VLASS, tmp_path)
    tree = open_engine(msname, partition_cache="off", with_pointing=True)
    arrays = {}
    for name, node in sorted(tree.children.items()):
        for var in main_data_variables(node):
            arrays[f"{name}/{var}"] = node[var].data
        for var, data in node["pointing_xds"].data_vars.items():
            arrays[f"{name}/pointing_xds/{var}"] = data.data
    assert len(arrays) > 2 * len(tree.children)
    names = list(arrays)
    blocks = [_block_digests(arrays[name]) for name in names]
    assert sum(b.npartitions for b in blocks) > 8 * len(tree.children)

    backend_arrays.clear_index_memo()
    backend_pointing.clear_pointing_memos()
    threaded = dask.compute(*blocks, scheduler="threads", num_workers=8)
    synchronous = dask.compute(*blocks, scheduler="synchronous")
    for name, got, expected in zip(names, threaded, synchronous, strict=True):
        np.testing.assert_array_equal(got, expected, err_msg=name)


class _Messages:
    """A logger that records its INFO messages."""

    def __init__(self):
        self.info_messages: list[str] = []

    def info(self, message):
        self.info_messages.append(str(message))

    def debug(self, message):
        pass

    warning = error = debug


def _ms_fds(msname: str) -> list[str]:
    """The files under the MS that this process has open (Linux)."""
    root = os.path.realpath(msname)
    found = []
    for fd in os.listdir("/proc/self/fd"):
        with contextlib.suppress(OSError):
            target = os.readlink(f"/proc/self/fd/{fd}")
            if target.startswith(root + os.sep):
                found.append(target)
    return found


def test_auto_open_writes_nothing(tmp_path, monkeypatch):
    """
    With casatools the partitions are never stored: an "auto" open computes
    them in memory (logged once at INFO: "casatools only"), the next open
    uses the memo, and no file of the MS changes (table.lock included). No
    file of the MS is left open.
    """
    from xradio.measurement_set._utils._msv2 import partition_cache

    msname = copy_ms(ref.LOFAR, tmp_path)
    before = file_digests(msname)
    messages = _Messages()
    monkeypatch.setattr(partition_cache, "xradio_logger", lambda: messages)
    partition_cache.clear_partition_memo()
    with cache_statuses(monkeypatch) as statuses:
        for _ in range(2):
            tree = open_engine(msname, partition_cache="auto")
            tree[sorted(tree.children)[0]].VISIBILITY.isel(time=slice(0, 2)).values  # noqa: B018
    assert statuses == ["memory:casatools only", "hit-memory"]
    assert sum("casatools only" in m for m in messages.info_messages) == 1
    assert file_digests(msname) == before
    assert not os.path.exists(os.path.join(msname, partition_cache.SUBTABLE_NAME))
    if os.path.isdir("/proc/self/fd"):
        assert _ms_fds(msname) == []


def _create_subtable_with_casatools(msname: str) -> None:
    """
    The XRADIO_PARTITIONS sub-table of an MS made with casatools (the shim
    cannot create tables): the description of
    partition_cache.subtable_description() (without python-casacore's
    "shape": [] and "_c_order" of the variable-shape run arrays, which
    casatools cannot convert), the keywords and info of python-casacore's
    writer, and the MAIN keyword that links it (with the shim's putkeyword).
    """
    import casatools

    from xradio._utils._casacore.casacore_from_casatools import table
    from xradio.measurement_set._utils._msv2 import partition_cache as pc

    description = {
        name: {k: v for k, v in column.items() if k not in ("shape", "_c_order")}
        for name, column in pc.subtable_description().items()
    }
    subtable = pc.subtable_path(msname)
    tb = casatools.table()
    assert tb.create(subtable, description)
    try:
        tb.putkeyword("FORMAT_VERSION", pc.FORMAT_VERSION)
        tb.putkeyword("CREATOR", "xradio")
        tb.putinfo(
            {"type": "XRADIO Partitions", "subType": "", "readme": pc.SUBTABLE_README}
        )
    finally:
        tb.close()
    with table(msname, readonly=False) as main_tb:
        main_tb.putkeyword(pc.SUBTABLE_NAME, f"Table: {subtable}")


def _store_row_with_casatools(msname: str, partition_scheme: list) -> None:
    """Store the partitions of a scheme as python-casacore's writer does: the
    fingerprint and HISTORY rows taken before the partitions are computed, a
    HISTORY row of the content, then the row (its CHECKSUM last)."""
    import uuid

    from xradio._utils._casacore.casacore_from_casatools import table
    from xradio.measurement_set._utils._msv2 import partition_cache as pc
    from xradio.measurement_set._utils._msv2._tables.table_lock_file import (
        history_nrows,
    )
    from xradio.measurement_set._utils._msv2.partition_queries import (
        create_partitions_with_main_rows,
    )

    fingerprint, n_history = pc.fingerprint_json(msname), history_nrows(msname)
    partitions, runs = create_partitions_with_main_rows(msname, partition_scheme)
    cache_id = uuid.uuid4().hex
    history_row = pc._append_history_row(
        msname, partition_scheme, partitions, runs, cache_id, "first"
    )
    values = pc.encode_row(
        partitions,
        runs,
        partition_scheme,
        fingerprint,
        history_row,
        n_history,
        cache_id=cache_id,
    )
    with table(pc.subtable_path(msname), readonly=False) as sub_tb:
        sub_tb.addrows(1)
        pc._put_row_cells(sub_tb, sub_tb.nrows() - 1, values)


def test_stored_partitions_are_read(tmp_path, monkeypatch):
    """
    Partitions stored in an MS (the sub-table built with casatools, two rows:
    VLASS has no autocorrelations, so the ANTENNA1 row's run arrays are
    empty) are read through the shim: the fingerprint is computable, the
    rows read back (empty Double arrays as empty int64 arrays), the opens
    use them ("hit") and give the partitions and tree of an open without
    the cache, and write nothing. remove_msv2_partition_cache removes them.
    """
    from xradio.measurement_set import remove_msv2_partition_cache
    from xradio.measurement_set._utils._msv2 import partition_cache as pc
    from xradio.measurement_set._utils._msv2.partition_queries import (
        canonical_scheme_key,
        create_partitions_with_main_rows,
    )

    msname = copy_ms(ref.VLASS, tmp_path)
    schemes = ([], ["ANTENNA1"])
    _create_subtable_with_casatools(msname)
    assert pc.link_state(msname) == "linked"
    for scheme in schemes:
        _store_row_with_casatools(msname, scheme)
    assert json.loads(pc.fingerprint_json(msname))["main"]["lock_ok"]

    lookup = pc.lookup_stored_row(msname, canonical_scheme_key(["ANTENNA1"]))
    assert lookup.state == "row"
    for column in ("ROW_STARTS", "ROW_LENGTHS"):
        assert lookup.row[column].dtype == np.int64
        assert lookup.row[column].size == 0
    assert lookup.row["N_PARTITIONS"] > 0

    before = file_digests(msname)
    for scheme in schemes:
        pc.clear_partition_memo()
        with cache_statuses(monkeypatch) as statuses:
            tree = open_engine(
                msname,
                partition_cache="auto",
                partition_scheme=scheme,
                with_pointing=False,
            )
        assert statuses == ["hit"]
        stored = pc.load_or_create_partitions(msname, scheme, "read")
        assert stored.status == "hit-memory"
        fresh = create_partitions_with_main_rows(msname, scheme)
        assert stored.partitions == fresh[0]
        for name in ("starts", "lengths", "bounds"):
            np.testing.assert_array_equal(
                getattr(stored.runs, name), getattr(fresh[1], name)
            )
        off = open_engine(
            msname, partition_cache="off", partition_scheme=scheme, with_pointing=False
        )
        assert sorted(tree.children) == sorted(off.children)
        assert bool(tree.children) == (scheme == [])
        for name in sorted(tree.children)[:2]:
            for var in ("VISIBILITY", "UVW", "TIME_CENTROID"):
                np.testing.assert_array_equal(
                    tree[name][var].isel(time=slice(0, 3)).values,
                    off[name][var].isel(time=slice(0, 3)).values,
                )
    assert file_digests(msname) == before

    assert remove_msv2_partition_cache(msname)
    assert pc.link_state(msname) == "absent"
    assert not os.path.exists(pc.subtable_path(msname))
    assert not remove_msv2_partition_cache(msname)
    pc.clear_partition_memo()
    with cache_statuses(monkeypatch) as statuses:
        open_engine(msname, partition_cache="auto", with_pointing=False)
    assert statuses == ["memory:casatools only"]


def test_every_table_open_holds_the_casatools_lock(tmp_path, monkeypatch):
    """
    Every table of the MS that an open uses is opened holding the
    process-wide casatools lock, also when the partitions come from the memo
    or the stored row (the check of their rows reads FIELD, SOURCE and
    STATE), and so are the lazy reads: dask threads reading another tree
    meanwhile cannot run casatools calls concurrently with the open.
    """
    import dask

    from xradio._utils._casacore import casacore_from_casatools as shim
    from xradio._utils._casacore.tables import CASATOOLS_LOCK
    from xradio.measurement_set._utils._msv2 import partition_cache as pc

    msname = copy_ms(ref.VLASS, tmp_path)
    _create_subtable_with_casatools(msname)
    _store_row_with_casatools(msname, [])
    root = os.path.realpath(msname)
    unlocked = []
    table_init = shim.table.__init__

    def spy(self, *args, **kwargs):
        tablename = args[0] if args else kwargs.get("tablename", "")
        name = os.path.realpath(str(tablename)) if tablename else ""
        if name.startswith(root) and not getattr(CASATOOLS_LOCK._held, "locks", None):
            unlocked.append(os.path.relpath(name, root))
        table_init(self, *args, **kwargs)

    monkeypatch.setattr(shim.table, "__init__", spy)
    pc.clear_partition_memo()
    other = open_engine(msname, partition_cache="off", with_pointing=True)
    blocks = [
        _block_digests(node[var].data)
        for node in other.children.values()
        for var in ("VISIBILITY", "FLAG")
    ]
    # lazy reads of another tree in 4 dask threads, while the MS is opened
    reader = threading.Thread(
        target=dask.compute,
        args=blocks,
        kwargs={"scheduler": "threads", "num_workers": 4},
    )
    with cache_statuses(monkeypatch) as statuses:
        reader.start()
        try:
            for _ in range(2):  # the stored row, then the memo
                tree = open_engine(msname, partition_cache="read", with_pointing=True)
        finally:
            reader.join()
    assert statuses == ["hit", "hit-memory"]
    name = sorted(tree.children)[0]
    tree[name].VISIBILITY.isel(time=slice(0, 2)).values  # noqa: B018
    tree[name]["pointing_xds"].POINTING_BEAM.isel(time_pointing=[0, -1]).values  # noqa: B018
    assert unlocked == []
