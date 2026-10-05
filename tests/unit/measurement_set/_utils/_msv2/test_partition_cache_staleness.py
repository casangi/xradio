"""
Staleness of the partition cache of the MSv2 xarray backend against the
converter as the oracle: the partitions of a copy of a generated MS are
stored, the copy is modified, and the next open must give the processing
set that ``convert_msv2_to_processing_set`` writes for the modified copy,
with the expected cache status and HISTORY rows. Both layouts of the MAIN
key columns: one StandardStMan shared with other columns (as generated) and
a data manager per key column (as CASA split outputs). Also: stored
partitions that pass the staleness checks but do not describe their rows
(a forged fingerprint, runs swapped between partitions), which the check of
cache-sourced partitions against their rows catches.
"""

import os
import shutil

import numpy as np
import pytest
import xarray as xr
from casacore import tables

from _xradio_xarray_backends import MSv2BackendEntrypoint
from xradio.measurement_set import convert_msv2_to_processing_set, open_processing_set
from xradio.measurement_set._utils._msv2 import (
    backend_open,
    backend_partition,
    partition_cache,
)
from xradio.measurement_set._utils._msv2.backend_errors import PartitionCacheWarning
from xradio.measurement_set._utils._msv2.partition_cache import (
    SUBTABLE_COLUMNS,
    SUBTABLE_NAME,
    normalise_row,
    row_checksum,
)
from xradio.testing.measurement_set.equivalence import assert_nodes_identical

ENGINE = MSv2BackendEntrypoint
# The scheme of the scenarios, and the partitions opened and converted: those
# of DDI 0 and field 0, which the modifications change (the stored rows
# describe all 16 partitions)
SCHEME = ["FIELD_ID"]


def ddi0_field0(partition):
    return partition["DATA_DESC_ID"][0] == 0 and partition["FIELD_ID"][0] == 0


OPTIONS = {
    "partition_scheme": SCHEME,
    "partition_filter": ddi0_field0,
    "with_pointing": False,
}
# Rows of DDI 0 in "rich" (time-major, 10 baselines per time): 0-99 scan 1
# (state 0, CALIBRATE_PHASE), 100-199 scan 2 (state 1), 200-299 scan 3
# (state -1, the last STATE row: OBSERVE_TARGET); FIELD_ID alternates every
# 5 times (50 rows): field 0 in rows 0-49, 100-149 and 200-249


@pytest.fixture(autouse=True)
def _clear_memo():
    partition_cache.clear_partition_memo()
    yield
    partition_cache.clear_partition_memo()


def history_nrows(msname: str) -> int:
    with tables.table(os.path.join(msname, "HISTORY"), ack=False) as history:
        return history.nrows()


def status_of(msname: str, scheme=SCHEME) -> str:
    """The status of an "auto" load, as in a new process (memo emptied)."""
    partition_cache.clear_partition_memo()
    return partition_cache.load_or_create_partitions(
        os.path.abspath(msname), list(scheme), "auto"
    ).status


def assert_equals_the_oracle(msname: str, out: str, **options) -> xr.DataTree:
    """The engine's tree of an MS ("auto") equals the processing set the
    converter writes for it."""
    options = options or OPTIONS
    convert_msv2_to_processing_set(msname, out, **options)
    tree = xr.open_datatree(
        msname, engine=ENGINE, chunks={}, partition_cache="auto", **options
    )
    assert_nodes_identical(tree, open_processing_set(out))
    return tree


# --- the modifications --------------------------------------------------------------


def _update(msname, column, rows, values):
    with tables.table(msname, readonly=False, ack=False) as main_tb:
        data = main_tb.getcol(column)
        data[rows] = values
        main_tb.putcol(column, data)
    return msname


def _same_values(column):
    def modify(msname):
        with tables.table(msname, readonly=False, ack=False) as main_tb:
            main_tb.putcol(column, main_tb.getcol(column))
        return msname

    return modify


def _subtable_cell(subtable, column, row, value):
    def modify(msname):
        with tables.table(
            os.path.join(msname, subtable), readonly=False, ack=False
        ) as t:
            t.putcell(column, row, value)
        return msname

    return modify


def _source_removed(msname):
    with tables.table(msname, readonly=False, ack=False) as main_tb:
        main_tb.removekeyword("SOURCE")
    tables.tabledelete(os.path.join(msname, "SOURCE"), ack=False)
    return msname


def _taql_update(msname):
    with tables.table(msname, readonly=False, ack=False) as main_tb:
        tables.taql("UPDATE $1 SET FIELD_ID = 1 WHERE ROWID() < 50", tables=[main_tb])
    return msname


def _swap_antennas(msname):
    with tables.table(msname, readonly=False, ack=False) as main_tb:
        ant1, ant2 = main_tb.getcol("ANTENNA1"), main_tb.getcol("ANTENNA2")
        ant1[:10], ant2[:10] = ant2[:10].copy(), ant1[:10].copy()
        main_tb.putcol("ANTENNA1", ant1)
        main_tb.putcol("ANTENNA2", ant2)
    return msname


def _add_rows(msname):
    with tables.table(msname, readonly=False, ack=False) as main_tb:
        nrows = main_tb.nrows()
        main_tb.copyrows(main_tb, startrowin=0, nrow=10)
        for column in ("TIME", "TIME_CENTROID"):
            values = main_tb.getcol(column, nrows, 10)
            main_tb.putcol(column, values + 100.0, nrows, 10)
    return msname


def _add_column(msname):
    with tables.table(msname, readonly=False, ack=False) as main_tb:
        main_tb.addcols(tables.maketabdesc([tables.makescacoldesc("XR_EXTRA", 0.0)]))
    return msname


def _remove_column(msname):
    with tables.table(msname, readonly=False, ack=False) as main_tb:
        main_tb.removecols(["SIGMA"])
    return msname


def _flag(msname):
    with tables.table(msname, readonly=False, ack=False) as main_tb:
        flag = main_tb.getcol("FLAG", 0, 100)
        main_tb.putcol("FLAG", ~flag, 0, 100)
    return msname


def _flag_row(msname):
    return _update(msname, "FLAG_ROW", slice(0, 30), True)


def _copy_to(msname, name, **copy_options):
    target = os.path.join(os.path.dirname(msname), "copies", name)
    shutil.copytree(msname, target, symlinks=True, **copy_options)
    return target


def _cp_a(msname):
    return _copy_to(msname, "rich.ms")


def _cp_r(msname):
    # (new modification times, as cp -r)
    return _copy_to(msname, "rich.ms", copy_function=shutil.copyfile)


def _rename(msname):
    target = os.path.join(os.path.dirname(msname), "renamed.ms")
    os.rename(msname, target)
    return target


def _table_copy(deep, sort=None):
    def modify(msname):
        target = os.path.join(os.path.dirname(msname), "copies", "rich.ms")
        os.makedirs(os.path.dirname(target))
        with tables.table(msname, ack=False) as main_tb:
            source = main_tb.sort(sort) if sort else main_tb
            source.copy(target, deep=deep).close()
            if sort:
                source.close()
        return target

    return modify


def _msconcat(msname):
    from casacore.tables import msutil

    other = _copy_to(msname, "other.ms")
    target = os.path.join(os.path.dirname(msname), "concat.ms")
    msutil.msconcat([msname, other], target, concatTime=False)
    return target


def _foreign_history_row(msname):
    with tables.table(os.path.join(msname, "HISTORY"), readonly=False, ack=False) as h:
        row = h.nrows()
        h.addrows(1)
        h.putcell("APPLICATION", row, "ms")
        h.putcell("ORIGIN", row, "flagdata")
        h.putcell("MESSAGE", row, "taskname=flagdata")
    return msname


def _history_recreated(msname):
    history = os.path.join(msname, "HISTORY")
    with tables.table(history, ack=False) as table:
        desc, dminfo = table.getdesc(), table.getdminfo()
    shutil.rmtree(history)
    tables.table(
        history, desc, dminfo=dminfo, nrow=0, readonly=False, ack=False
    ).close()
    return msname


def _corrupt_checksum(msname):
    subtable = os.path.join(msname, SUBTABLE_NAME)
    with tables.table(subtable, readonly=False, ack=False) as table:
        for row in range(table.nrows()):
            table.putcell("CHECKSUM", row, "0" * 32)
    return msname


# scenario: (modification, status by layout ("shared", "per_column"))
SCENARIOS = {
    "FIELD_ID, membership changed": (
        lambda ms: _update(ms, "FIELD_ID", slice(0, 50), 1),
        ("stored", "stored"),
    ),
    "FIELD_ID, same values": (_same_values("FIELD_ID"), ("revalidated", "hit")),
    "SCAN_NUMBER, descriptions kept": (
        lambda ms: _update(ms, "SCAN_NUMBER", slice(200, 210), 2),
        ("revalidated", "revalidated"),
    ),
    "SCAN_NUMBER, descriptions changed": (
        lambda ms: _update(ms, "SCAN_NUMBER", slice(0, 10), 4),
        ("stored", "stored"),
    ),
    "STATE_ID, membership changed": (
        lambda ms: _update(ms, "STATE_ID", slice(0, 50), 1),
        ("stored", "stored"),
    ),
    "DATA_DESC_ID, membership changed": (
        lambda ms: _update(ms, "DATA_DESC_ID", slice(0, 50), 1),
        ("stored", "stored"),
    ),
    "OBSERVATION_ID, same values": (
        _same_values("OBSERVATION_ID"),
        ("revalidated", "hit"),
    ),
    "ANTENNA1 and ANTENNA2 swapped": (_swap_antennas, ("revalidated", "revalidated")),
    "TaQL UPDATE of FIELD_ID": (_taql_update, ("stored", "stored")),
    "STATE OBS_MODE of a used row": (
        _subtable_cell("STATE", "OBS_MODE", 0, "CALIBRATE_BANDPASS#ON_SOURCE"),
        ("stored", "stored"),
    ),
    "FIELD SOURCE_ID": (
        _subtable_cell("FIELD", "SOURCE_ID", 0, 1),
        ("stored", "stored"),
    ),
    "FIELD EPHEMERIS_ID": (
        _subtable_cell("FIELD", "EPHEMERIS_ID", 0, np.array([-1], np.int32)),
        ("stored", "stored"),
    ),
    "FIELD NAME": (
        _subtable_cell("FIELD", "NAME", 0, "another field"),
        ("revalidated", "revalidated"),
    ),
    # (the partitions then have no SOURCE_ID; an emptied SOURCE cannot be
    # converted: casacore's TaQL fails on the StandardStMan without rows)
    "SOURCE removed": (_source_removed, ("stored", "stored")),
    "addrows": (_add_rows, ("stored", "stored")),
    "column added": (_add_column, ("revalidated", "revalidated")),
    "column removed": (_remove_column, ("revalidated", "revalidated")),
    "FLAG written": (_flag, ("revalidated", "hit")),
    "FLAG_ROW written": (_flag_row, ("revalidated", "hit")),
    "cp -a": (_cp_a, ("hit", "hit")),
    "cp -r": (_cp_r, ("revalidated", "revalidated")),
    "renamed": (_rename, ("hit", "hit")),
    "deep copy with rows sorted": (
        _table_copy(True, "ANTENNA2, TIME"),
        ("stored", "stored"),
    ),
    "table.copy, deep": (_table_copy(True), ("revalidated", "revalidated")),
    "table.copy, shallow": (_table_copy(False), ("revalidated", "revalidated")),
    # (its MAIN columns forward to the MSs it concatenates)
    "msutil.msconcat": (_msconcat, ("memory:MAIN uses ForwardColumnEngine",) * 2),
    "a foreign HISTORY row": (_foreign_history_row, ("revalidated", "revalidated")),
    "HISTORY re-created": (_history_recreated, ("stored", "stored")),
    "a corrupt CHECKSUM": (_corrupt_checksum, ("stored", "stored")),
}


# The scenarios run with the data manager per key column too: those that
# write MAIN (the others do not depend on its data managers)
PER_COLUMN_SCENARIOS = [
    "FIELD_ID, membership changed",
    "FIELD_ID, same values",
    "SCAN_NUMBER, descriptions kept",
    "OBSERVATION_ID, same values",
    "ANTENNA1 and ANTENNA2 swapped",
    "TaQL UPDATE of FIELD_ID",
    "addrows",
    "column added",
    "column removed",
    "FLAG written",
    "FLAG_ROW written",
    "cp -r",
    "deep copy with rows sorted",
]


@pytest.mark.parametrize(
    "scenario, layout",
    [(scenario, "shared") for scenario in SCENARIOS]
    + [(scenario, "per_column") for scenario in PER_COLUMN_SCENARIOS],
)
def test_modified_ms_equals_the_oracle(
    scenario, layout, ms_copy, tmp_path, monkeypatch
):
    """Store, modify the copy, open it again (as in a new process): (i) the
    tree equals the converter's processing set of the modified copy, (ii)
    the partitions were obtained as expected, at once, and (iii) a HISTORY
    row was added only for stored partitions."""
    modify, expected = SCENARIOS[scenario]
    msname = ms_copy("rich", layout, name="rich.ms")
    assert status_of(msname) == "stored"
    msname = modify(msname)
    n_history = history_nrows(msname)
    statuses = []
    load = backend_open.load_or_create_partitions

    def spy(*args, **kwargs):
        result = load(*args, **kwargs)
        statuses.append(result.status)
        return result

    monkeypatch.setattr(backend_open, "load_or_create_partitions", spy)
    partition_cache.clear_partition_memo()
    assert_equals_the_oracle(msname, str(tmp_path / "oracle.ps.zarr"))
    assert statuses == [expected[0 if layout == "shared" else 1]]
    assert history_nrows(msname) == n_history + (statuses[0] == "stored")
    if not statuses[0].startswith("memory:"):
        assert status_of(msname) == "hit"


def test_unused_state_row(ms_copy, tmp_path):
    """The OBS_MODE of a STATE row no MAIN row uses: the partitions are
    computed again and equal (revalidated)."""
    msname = ms_copy("dense", name="dense.ms")
    assert status_of(msname, []) == "stored"
    _subtable_cell("STATE", "OBS_MODE", 1, "OBSERVE_TARGET#ON_SOURCE")(msname)
    n_history = history_nrows(msname)
    assert status_of(msname, []) == "revalidated"
    assert history_nrows(msname) == n_history
    assert_equals_the_oracle(
        msname, str(tmp_path / "oracle.ps.zarr"), partition_scheme=[]
    )


# --- stored partitions that do not describe their rows --------------------------------


def _stored_row(msname: str) -> dict:
    with tables.table(os.path.join(msname, SUBTABLE_NAME), ack=False) as table:
        return normalise_row(
            {name: table.getcell(name, 0) for name in SUBTABLE_COLUMNS}
        )


def _put_row(msname: str, row: dict) -> None:
    row = dict(row, CHECKSUM=row_checksum(row))
    with tables.table(
        os.path.join(msname, SUBTABLE_NAME), readonly=False, ack=False
    ) as table:
        for name in SUBTABLE_COLUMNS:
            value = row[name]
            if isinstance(value, np.ndarray):
                value = value.astype(np.float64)
            table.putcell(name, 0, value)


@pytest.mark.parametrize("layout", ["shared", "per_column"])
def test_forged_fingerprint(layout, ms_copy, tmp_path, monkeypatch):
    """A fingerprint that misses a change of the rows (forged: the stored
    one): the stored partitions are served, the check against their rows
    fails, a warning, the partitions are computed again (and stored), and
    the tree equals the oracle."""
    msname = ms_copy("rich", layout, name="rich.ms")
    assert status_of(msname) == "stored"
    forged = _stored_row(msname)["FINGERPRINT"]
    monkeypatch.setattr(partition_cache, "fingerprint_json", lambda path: forged)
    _update(msname, "FIELD_ID", slice(0, 50), 1)
    partition_cache.clear_partition_memo()
    n_history = history_nrows(msname)
    with pytest.warns(PartitionCacheWarning, match="did not describe its rows"):
        assert_equals_the_oracle(msname, str(tmp_path / "oracle.ps.zarr"))
    assert history_nrows(msname) == n_history + 1
    assert "reason=rebuild" in _last_history_params(msname)
    # stored again: the next open uses the stored partitions, and they pass
    partition_cache.clear_partition_memo()
    assert status_of(msname) == "hit"


def test_forged_fingerprint_non_grouping_axis(ms_copy, tmp_path, monkeypatch):
    """A change the fingerprint misses (forged) that alters only an axis of
    a partition description that does not group the rows (SCAN_NUMBER under
    the scheme []: the rows of scan 1 of DDI 0 moved to scan 2, which only
    another partition had): the check against the rows compares every
    described axis, so it is caught and the partitions are computed again."""
    msname = ms_copy("rich", name="rich.ms")
    options = {"partition_scheme": [], "with_pointing": False}
    assert status_of(msname, []) == "stored"
    forged = _stored_row(msname)["FINGERPRINT"]
    monkeypatch.setattr(partition_cache, "fingerprint_json", lambda path: forged)
    _update(msname, "SCAN_NUMBER", slice(0, 10), 2)
    assert status_of(msname, []) == "hit"  # (the staleness checks pass)
    partition_cache.clear_partition_memo()
    with pytest.warns(PartitionCacheWarning, match="SCAN_NUMBER"):
        tree = assert_equals_the_oracle(
            msname, str(tmp_path / "oracle.ps.zarr"), **options
        )
    assert len(tree.children) == 8


def test_rows_added_while_opening_are_no_cache_defect(ms_copy, tmp_path, monkeypatch):
    """MAIN gains rows (another process) between the load of stored
    partitions and the build: logged at INFO and opened again with the
    partitions computed, without a PartitionCacheWarning (no cache
    defect); the tree has the new rows."""
    import subprocess
    import sys
    import warnings

    msname = ms_copy("dense", name="dense.ms")
    assert status_of(msname, []) == "stored"
    partition_cache.clear_partition_memo()
    script = (
        "from casacore import tables\n"
        f"with tables.table({msname!r}, readonly=False, ack=False) as t:\n"
        "    n = t.nrows()\n"
        "    t.copyrows(t, startrowin=0, nrow=10)\n"
        "    t.putcol('TIME', t.getcol('TIME', n, 10) + 100.0, n, 10)\n"
    )
    load = backend_open.load_or_create_partitions
    sources = []

    def load_then_add_rows(*args, **kwargs):
        result = load(*args, **kwargs)
        if not sources:
            subprocess.run([sys.executable, "-c", script], check=True)
        sources.append(result.source)
        return result

    infos = []

    class Logger:
        def info(self, message):
            infos.append(message)

        def __getattr__(self, name):
            return lambda *args, **kwargs: None

    monkeypatch.setattr(backend_open, "load_or_create_partitions", load_then_add_rows)
    monkeypatch.setattr(backend_open, "xradio_logger", Logger)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tree = xr.open_datatree(
            msname, engine=ENGINE, chunks={}, partition_cache="auto"
        )
    assert sources == ["stored", "fresh"]
    assert [w for w in caught if issubclass(w.category, PartitionCacheWarning)] == []
    assert any("changed while it was opened" in m for m in infos)
    assert tree["dense_0"].sizes["time"] == 31
    out = str(tmp_path / "oracle.ps.zarr")
    convert_msv2_to_processing_set(msname, out)
    assert_nodes_identical(tree, open_processing_set(out))


@pytest.mark.parametrize("mode", ["off", "auto"])
def test_keys_rewritten_after_the_partitions_were_computed(
    mode, ms_copy, tmp_path, monkeypatch
):
    """Another writer moves rows of field 0 to field 1 right after the
    partitions were computed (kept in memory, or stored), before the
    partitions are built: the open finds that the MS changed since before the
    computation (keys_token), checks the partitions against their rows and
    opens again with the partitions computed. The tree equals the
    converter's of the changed MS (no node with the rows of two fields), and
    its reads succeed (the comparison computes them)."""
    msname = ms_copy("rich", name="rich.ms")
    load = backend_open.load_or_create_partitions
    statuses = []

    def load_then_rewrite(*args, **kwargs):
        result = load(*args, **kwargs)
        if not statuses:
            _update(msname, "FIELD_ID", slice(0, 50), 1)
        statuses.append(result.status)
        return result

    monkeypatch.setattr(backend_open, "load_or_create_partitions", load_then_rewrite)
    options = {"partition_scheme": ["FIELD_ID"], "with_pointing": False}
    tree = xr.open_datatree(
        msname, engine=ENGINE, chunks={}, partition_cache=mode, **options
    )
    expected = "memory:mode-off" if mode == "off" else "stored"
    assert statuses == [expected, expected]
    out = str(tmp_path / "oracle.ps.zarr")
    convert_msv2_to_processing_set(msname, out, **options)
    assert_nodes_identical(tree, open_processing_set(out))


def test_rows_rewritten_while_opening_are_no_cache_defect(
    ms_copy, tmp_path, monkeypatch
):
    """A key column rewritten (another writer) between the load of stored
    partitions and their check against the rows: the check fails because
    the MS changed (its fingerprint), not because of the cache, so the open
    is logged at INFO and done again with the partitions computed, without
    the "please report" PartitionCacheWarning; the tree equals the
    converter's of the changed MS."""
    import warnings

    msname = ms_copy("rich", name="rich.ms")
    assert status_of(msname) == "stored"
    partition_cache.clear_partition_memo()
    load = backend_open.load_or_create_partitions
    statuses = []

    def load_then_rewrite(*args, **kwargs):
        result = load(*args, **kwargs)
        if not statuses:
            _update(msname, "FIELD_ID", slice(0, 50), 1)
        statuses.append(result.status)
        return result

    infos = []

    class Logger:
        def info(self, message):
            infos.append(message)

        def __getattr__(self, name):
            return lambda *args, **kwargs: None

    monkeypatch.setattr(backend_open, "load_or_create_partitions", load_then_rewrite)
    monkeypatch.setattr(backend_open, "xradio_logger", Logger)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tree = xr.open_datatree(
            msname, engine=ENGINE, chunks={}, partition_cache="auto", **OPTIONS
        )
    assert statuses == ["hit", "stored"]
    assert [w for w in caught if issubclass(w.category, PartitionCacheWarning)] == []
    assert any("changed while it was opened" in m for m in infos)
    out = str(tmp_path / "oracle.ps.zarr")
    convert_msv2_to_processing_set(msname, out, **OPTIONS)
    assert_nodes_identical(tree, open_processing_set(out))


def _last_history_params(msname: str) -> list[str]:
    with tables.table(os.path.join(msname, "HISTORY"), ack=False) as history:
        return list(history.getcell("APP_PARAMS", history.nrows() - 1))


@pytest.mark.parametrize("scheme", [["FIELD_ID"], ["ANTENNA1"]])
def test_runs_swapped_between_partitions(scheme, ms_copy, tmp_path):
    """The runs of two partitions of equal size swapped in the stored row
    (consistent otherwise: the staleness checks pass): the check against the
    rows catches it, and the tree equals the oracle."""
    variant = "rich" if scheme == ["FIELD_ID"] else "single_dish"
    msname = ms_copy(variant, name=f"{variant}.ms")
    options = {"partition_scheme": scheme, "partition_filter": ddi0_field0}
    assert status_of(msname, scheme) == "stored"
    row = _stored_row(msname)
    bounds, starts, lengths = (
        row["PARTITION_BOUNDS"],
        row["ROW_STARTS"].copy(),
        row["ROW_LENGTHS"],
    )
    sizes = [tuple(lengths[bounds[p] : bounds[p + 1]]) for p in range(bounds.size - 1)]
    first = next(p for p in range(len(sizes)) if sizes.count(sizes[p]) > 1 and sizes[p])
    second = next(q for q in range(first + 1, len(sizes)) if sizes[q] == sizes[first])
    a, b = (
        slice(bounds[first], bounds[first + 1]),
        slice(bounds[second], bounds[second + 1]),
    )
    starts[a], starts[b] = row["ROW_STARTS"][b], row["ROW_STARTS"][a]
    _put_row(msname, dict(row, ROW_STARTS=starts))
    assert status_of(msname, scheme) == "hit"  # (the staleness checks pass)
    partition_cache.clear_partition_memo()
    with pytest.warns(PartitionCacheWarning, match="did not describe its rows"):
        assert_equals_the_oracle(msname, str(tmp_path / "oracle.ps.zarr"), **options)


def test_verified_partitions_of_antenna1_schemes(ms_copy):
    """ANTENNA1 partitions are described by all rows of their keys but hold
    their autocorrelations: their check accepts descriptions that hold the
    rows' values, and rejects cross-correlation rows."""
    msname = ms_copy("rich", name="rich.ms")
    assert status_of(msname, ["ANTENNA1"]) == "stored"
    tree = xr.open_datatree(
        msname, engine=ENGINE, partition_cache="auto", partition_scheme=["ANTENNA1"]
    )
    assert len(tree.children) == 0  # (no autocorrelations in "rich")
    msname = ms_copy("single_dish", name="single_dish.ms")
    assert status_of(msname, ["ANTENNA1"]) == "stored"
    partition_cache.clear_partition_memo()
    tree = xr.open_datatree(
        msname, engine=ENGINE, partition_cache="auto", partition_scheme=["ANTENNA1"]
    )
    assert len(tree.children) == 20
    from xradio.measurement_set._utils._msv2.backend_errors import StalePartitionsError
    from xradio.measurement_set._utils._msv2.partition_queries import (
        create_partitions_with_main_rows,
        partition_key_maps,
    )

    partitions, runs = create_partitions_with_main_rows(msname, ["ANTENNA1"])
    check = backend_partition.RowCheck(partition_key_maps(msname), ["ANTENNA1"])

    class Rows:
        def __init__(self, columns):
            self.columns = columns

        def getcol(self, name):
            return self.columns[name]

    rows = runs[0].rows()
    with tables.table(msname, ack=False) as main_tb:
        columns = {
            name: main_tb.getcol(name)[rows]
            for name in (
                "DATA_DESC_ID",
                "OBSERVATION_ID",
                "FIELD_ID",
                "SCAN_NUMBER",
                "STATE_ID",
                "ANTENNA1",
                "ANTENNA2",
            )
        }
    backend_partition.verify_partition_rows(check, partitions[0], Rows(columns))
    # a subset of the described values: accepted (other axes than the keys)
    wider = dict(partitions[0], SCAN_NUMBER=partitions[0]["SCAN_NUMBER"] + [999])
    backend_partition.verify_partition_rows(check, wider, Rows(columns))
    # a grouping key differs, or a cross-correlation row: rejected
    with pytest.raises(StalePartitionsError, match="ANTENNA1"):
        backend_partition.verify_partition_rows(
            check, dict(partitions[0], ANTENNA1=[99]), Rows(columns)
        )
    crossed = dict(columns, ANTENNA2=columns["ANTENNA2"] + 1)
    with pytest.raises(StalePartitionsError, match="autocorrelations"):
        backend_partition.verify_partition_rows(check, partitions[0], Rows(crossed))


def test_fresh_partitions_are_not_checked(ms_copy, monkeypatch):
    """Only partitions from the cache are checked against their rows: a
    checker that rejects everything does not fail fresh opens (and makes
    cache-sourced ones computed again)."""
    msname = ms_copy("dense", name="dense.ms")
    calls = []

    def reject(check, info, rows):
        calls.append(info)
        raise backend_partition.StalePartitionsError("rejected")

    monkeypatch.setattr(backend_partition, "verify_partition_rows", reject)
    tree = xr.open_datatree(msname, engine=ENGINE, partition_cache="off")
    assert len(tree.children) == 4 and calls == []
    xr.open_datatree(msname, engine=ENGINE, partition_cache="auto")
    assert calls == []
    with pytest.warns(PartitionCacheWarning, match="did not describe its rows"):
        tree = xr.open_datatree(msname, engine=ENGINE, partition_cache="auto")
    assert len(tree.children) == 4 and len(calls) == 1


def test_main_held_open_in_this_process(ms_copy, tmp_path):
    """MAIN held open in this process while another process adds rows: the
    open re-synchronizes MAIN, and the tree has the new rows (the oracle
    converted after the handle is closed)."""
    import subprocess
    import sys

    msname = ms_copy("dense", name="dense.ms")
    held = tables.table(
        msname, readonly=True, lockoptions={"option": "usernoread"}, ack=False
    )
    try:
        nrows = held.nrows()
        assert status_of(msname, []) == "stored"
        script = (
            "from casacore import tables\n"
            f"with tables.table({msname!r}, readonly=False, ack=False) as t:\n"
            "    n = t.nrows()\n"
            "    t.copyrows(t, startrowin=0, nrow=10)\n"
            "    t.putcol('TIME', t.getcol('TIME', n, 10) + 100.0, n, 10)\n"
        )
        subprocess.run([sys.executable, "-c", script], check=True)
        assert held.nrows() == nrows
        tree = xr.open_datatree(
            msname, engine=ENGINE, chunks={}, partition_cache="auto"
        )
        assert held.nrows() == nrows + 10  # (re-synchronized)
        assert tree["dense_0"].sizes["time"] == 31
    finally:
        held.close()
    out = str(tmp_path / "oracle.ps.zarr")
    convert_msv2_to_processing_set(msname, out)
    assert_nodes_identical(tree, open_processing_set(out))


@pytest.mark.parametrize("mode", ["off", "auto"])
def test_rows_this_process_has_not_flushed(mode, ms_copy, tmp_path, monkeypatch):
    """MAIN rows that this process added and has not flushed (a writable
    handle that holds the write lock), with the partitions of the MS stored
    and memoised before: the open keeps them (a re-synchronization of MAIN
    would roll them back), computes the partitions from the rows of the
    process without the partition cache, and gives the tree of the
    converter (which reads the same rows); the rows reach the disk."""
    from xradio.measurement_set._utils._msv2._tables.table_lock_file import (
        read_table_lock,
    )

    msname = ms_copy("dense", name="dense.ms")
    if mode == "auto":
        assert status_of(msname, []) == "stored"  # (and memoised)
    statuses = []
    load = backend_open.compute_in_memory

    def spy(*args, **kwargs):
        result = load(*args, **kwargs)
        statuses.append(result.status)
        return result

    monkeypatch.setattr(backend_open, "compute_in_memory", spy)
    writer = tables.table(msname, readonly=False, ack=False)
    try:
        nrows = writer.nrows()
        writer.addrows(10)  # (copies of rows 0-9, 100 s later)
        for column in writer.colnames():
            if not writer.iscelldefined(column, 0):  # (FLAG_CATEGORY)
                continue
            values = writer.getcol(column, 0, 10)
            if column in ("TIME", "TIME_CENTROID"):
                values = values + 100.0
            writer.putcol(column, values, nrows, 10)
        assert read_table_lock(msname).nrrow == nrows  # (not flushed)
        tree = xr.open_datatree(msname, engine=ENGINE, chunks={}, partition_cache=mode)
        assert writer.nrows() == nrows + 10
        assert statuses == ["memory:MAIN rows of this process not flushed"]
        assert tree["dense_0"].sizes["time"] == 31
    finally:
        writer.close()
    with tables.table(msname, ack=False) as main_tb:
        assert main_tb.nrows() == nrows + 10
    out = str(tmp_path / "oracle.ps.zarr")
    convert_msv2_to_processing_set(msname, out, partition_scheme=[])
    assert_nodes_identical(tree, open_processing_set(out))
