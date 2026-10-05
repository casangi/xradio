"""
Tests of the partition cache of the MSv2 xarray backend (partition_cache.py):
the XRADIO_PARTITIONS layout, the staleness rules (fingerprint, HISTORY,
row checks), reading stored rows (made by a test-only writer,
store_test_row), the memo and the modes; storing (the write protocol,
revalidation, versions, links, removal, notices). On copies of the
generated MSs.
"""

import copy
import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import warnings

import numpy as np
import pytest
import xarray as xr
from casacore import tables

from _xradio_xarray_backends import MSv2BackendEntrypoint
from xradio.measurement_set import (
    convert_msv2_to_processing_set,
    remove_msv2_partition_cache,
)
from xradio.measurement_set._utils._msv2 import partition_cache
from xradio.measurement_set._utils._msv2._tables.table_lock_file import history_nrows
from xradio.measurement_set._utils._msv2.backend_errors import PartitionCacheWarning
from xradio.measurement_set._utils._msv2.partition_cache import (
    HISTORY_APPLICATION,
    HISTORY_ORIGIN,
    PARTITIONS_MEMO,
    SUBTABLE_COLUMNS,
    SUBTABLE_NAME,
    InvalidRowError,
    decode_partitions,
    decode_row,
    encode_partitions,
    encode_row,
    fingerprint_json,
    load_or_create_partitions,
    lookup_stored_row,
    normalise_row,
    row_checksum,
    subtable_description,
)
from xradio.measurement_set._utils._msv2.partition_queries import (
    PARTITION_ALGORITHM_VERSION,
    canonical_scheme_key,
    create_partitions_with_main_rows,
    partition_axis_names,
)
from xradio.testing.measurement_set.equivalence import assert_nodes_identical

ENGINE = MSv2BackendEntrypoint
# (variant, scheme) of the generated MSs whose partitions round trip
ROUND_TRIP_CASES = [
    ("dense", []),
    ("rich", []),
    ("rich", ["FIELD_ID", "SCAN_NUMBER"]),
    ("rich", ["STATE_ID", "SUB_SCAN_NUMBER"]),
    ("rich", ["SOURCE_ID"]),
    ("rich", ["ANTENNA1"]),
    ("single_dish", ["ANTENNA1"]),
    ("sparse_dup", ["FIELD_ID"]),
    ("dense", ["ANTENNA1"]),  # (no autocorrelations: empty partitions)
]


@pytest.fixture(autouse=True)
def _clear_memo():
    partition_cache.clear_partition_memo()
    yield
    partition_cache.clear_partition_memo()


def add_history_row(
    msname: str,
    application: str = HISTORY_APPLICATION,
    origin: str = HISTORY_ORIGIN,
    app_params: list[str] | None = None,
    when: float | None = None,
) -> int:
    """Append a HISTORY row; returns its row number."""
    with tables.table(os.path.join(msname, "HISTORY"), readonly=False, ack=False) as h:
        row = h.nrows()
        h.addrows(1)
        h.putcell("TIME", row, time.time() + 3506716800.0 if when is None else when)
        h.putcell("OBSERVATION_ID", row, -1)
        h.putcell("MESSAGE", row, f"test row of {application}")
        h.putcell("PRIORITY", row, "INFO")
        h.putcell("ORIGIN", row, origin)
        h.putcell("OBJECT_ID", row, 0)
        h.putcell("APPLICATION", row, application)
        h.putcell("CLI_COMMAND", row, np.array([""]))
        h.putcell("APP_PARAMS", row, np.array(app_params or [""]))
    return row


def store_test_row(
    msname: str,
    scheme: list[str],
    *,
    result=None,
    fingerprint: str | None = None,
    history: bool = True,
    link: bool = True,
    checksum: str | None = None,
    **cells,
) -> dict:
    """
    Test-only writer: store the partitions of ``scheme`` (computed unless
    ``result`` = (partitions, runs) is given) in XRADIO_PARTITIONS (created
    and linked if needed), with a HISTORY row (unless ``history`` is False).
    ``cells`` replace cell values; the CHECKSUM is recomputed unless given.
    Returns the stored cells.
    """
    partitions, runs = result or create_partitions_with_main_rows(msname, scheme)
    if fingerprint is None:
        fingerprint = fingerprint_json(msname)
    anchor = history_nrows(msname)
    cache_id = cells.pop("CACHE_ID", None) or os.urandom(16).hex()
    history_row = (
        add_history_row(msname, app_params=[f"cache_id={cache_id}"])
        if history
        else anchor
    )
    values = encode_row(
        partitions, runs, scheme, fingerprint, history_row, anchor, cache_id=cache_id
    )
    values.update(cells)
    values["CHECKSUM"] = checksum or row_checksum(values)
    subtable = os.path.join(msname, SUBTABLE_NAME)
    if not os.path.isdir(subtable):
        with tables.table(
            subtable, subtable_description(), nrow=0, readonly=False, ack=False
        ) as table:
            table.putkeyword("FORMAT_VERSION", 1)
            table.putkeyword("CREATOR", "xradio")
    with tables.table(subtable, readonly=False, ack=False) as table:
        row = table.nrows()
        table.addrows(1)
        for name in SUBTABLE_COLUMNS:
            table.putcell(name, row, values[name])
    if link:
        with tables.table(msname, readonly=False, ack=False) as main_tb:
            main_tb.putkeyword(SUBTABLE_NAME, "Table: " + subtable)
    return values


def stored_rows(msname: str) -> list[dict]:
    with tables.table(os.path.join(msname, SUBTABLE_NAME), ack=False) as table:
        return [
            {name: table.getcell(name, row) for name in SUBTABLE_COLUMNS}
            for row in range(table.nrows())
        ]


def load(msname: str, scheme: list[str], mode: str = "read"):
    return load_or_create_partitions(os.path.abspath(msname), scheme, mode)


def no_compute(monkeypatch):
    monkeypatch.setattr(
        partition_cache,
        "create_partitions_with_main_rows",
        lambda *a, **k: pytest.fail("partitions computed"),
    )


# --- the layout ------------------------------------------------------------------


def test_subtable_layout(ms_copy):
    """No Int64 column (casatools cannot read them); TIME is an epoch; the
    run arrays are 1-D Double arrays; one StandardStMan."""
    msname = ms_copy("dense")
    store_test_row(msname, [])
    with tables.table(os.path.join(msname, SUBTABLE_NAME), ack=False) as table:
        desc = table.getdesc()
        assert table.colnames() == list(SUBTABLE_COLUMNS)
        assert [dm["TYPE"] for dm in table.getdminfo().values()] == ["StandardStMan"]
        assert table.getkeyword("FORMAT_VERSION") == 1
    value_types = {name: desc[name]["valueType"] for name in SUBTABLE_COLUMNS}
    assert set(value_types.values()) == {"string", "int", "double"}
    assert desc["TIME"]["keywords"]["MEASINFO"] == {"type": "epoch", "Ref": "UTC"}
    assert desc["TIME"]["keywords"]["QuantumUnits"] == ["s"]
    for name in ("ROW_STARTS", "ROW_LENGTHS", "PARTITION_BOUNDS"):
        assert desc[name]["ndim"] == 1 and value_types[name] == "double"
    for name in ("MAIN_NROWS", "N_RUNS", "N_ROWS_COVERED"):
        assert value_types[name] == "double"


@pytest.mark.parametrize("variant, scheme", ROUND_TRIP_CASES)
def test_round_trip(variant, scheme, backend_ms):
    """The stored cells of a result give back its partitions (JSON: [None],
    OBS_MODE strings, ANTENNA1, empty partitions) and runs."""
    msname = backend_ms(variant)
    partitions, runs = create_partitions_with_main_rows(msname, scheme)
    values = encode_row(partitions, runs, scheme, fingerprint_json(msname), 0, 0)
    # as read back: Doubles for the counts and run arrays
    read_back = dict(values, MAIN_NROWS=float(values["MAIN_NROWS"]))
    row = normalise_row(read_back)
    assert row_checksum(row) == values["CHECKSUM"]
    decoded, decoded_runs = decode_row(row, scheme)
    assert decoded == partitions
    assert [[type(v) for v in p.values()] for p in decoded] == [
        [type(v) for v in p.values()] for p in partitions
    ]
    for name in ("starts", "lengths", "bounds", "digests"):
        np.testing.assert_array_equal(
            getattr(decoded_runs, name), getattr(runs, name), err_msg=name
        )
    assert decoded_runs.main_nrows == runs.main_nrows


def test_columnar_json():
    partitions = [
        {"DATA_DESC_ID": [0], "OBS_MODE": ["A#B"], "SOURCE_ID": [None]},
        {"DATA_DESC_ID": [1], "OBS_MODE": ["C"], "SOURCE_ID": [None]},
    ]
    keys = ["DATA_DESC_ID", "OBS_MODE", "SOURCE_ID"]
    text = encode_partitions(partitions, keys)
    assert json.loads(text) == {
        "keys": keys,
        "columns": {
            "DATA_DESC_ID": [[0], [1]],
            "OBS_MODE": [["A#B"], ["C"]],
            "SOURCE_ID": [[None], [None]],
        },
    }
    assert decode_partitions(text) == partitions
    assert decode_partitions(encode_partitions([], keys)) == []
    with pytest.raises(ValueError, match="keys"):
        encode_partitions([{"OBS_MODE": ["C"], "DATA_DESC_ID": [1]}], keys[:2])
    for bad in ("[]", "{}", '{"keys": ["A"], "columns": {"A": [[]]}}', "x"):
        with pytest.raises(InvalidRowError):
            decode_partitions(bad)


def test_double_integrality():
    row = encode_row(
        [{name: [0] for name in partition_axis_names([])}],
        *_one_run_runs(),
        fingerprint='{"main": {"nrows": 5}}',
        history_row=0,
        history_nrows_at_build=0,
    )
    assert normalise_row(dict(row, MAIN_NROWS=5.0))["MAIN_NROWS"] == 5
    for name, value in [
        ("MAIN_NROWS", 5.5),
        ("MAIN_NROWS", float("nan")),
        ("MAIN_NROWS", 2.0**53),
        ("ROW_STARTS", np.array([0.5])),
        ("ROW_STARTS", np.array([-1.0])),
        ("ROW_LENGTHS", np.array([np.inf])),
        ("N_PARTITIONS", True),
        ("PARTITIONS", 3),
    ]:
        with pytest.raises(InvalidRowError, match=name):
            normalise_row(dict(row, **{name: value}))


def _one_run_runs():
    """(runs, scheme) of one partition of 5 rows."""
    from xradio.measurement_set._utils._msv2.partition_queries import (
        MainRowRuns,
        _stack_digests,
    )

    runs = MainRowRuns(
        np.array([0]), np.array([5]), np.array([0, 1]), _stack_digests([None]), 5
    )
    return runs, []


def test_checksum_normalisation():
    """The CHECKSUM of a row equals that of its normalised read-back (Doubles
    for integers: 53136 and 53136.0 give the same bytes) and changes with
    every column."""
    row = encode_row(
        [{name: [0] for name in partition_axis_names([])}],
        *_one_run_runs(),
        fingerprint='{"main": {"nrows": 5}}',
        history_row=3,
        history_nrows_at_build=2,
    )
    read_back = dict(row, MAIN_NROWS=5.0, N_RUNS=1.0, N_ROWS_COVERED=5.0)
    assert row_checksum(read_back) == row_checksum(row) == row["CHECKSUM"]
    assert row_checksum(normalise_row(read_back)) == row["CHECKSUM"]
    changes = {
        "CACHE_ID": "0" * 32,
        "SCHEME_KEY": '["FIELD_ID"]',
        "SCHEME": '["FIELD_ID"]',
        "FORMAT_VERSION": 2,
        "ALGORITHM_VERSION": 2,
        "XRADIO_VERSION": "0.0.0",
        "TIME": row["TIME"] + 1e-6,
        "MAIN_NROWS": 6,
        "N_RUNS": 2,
        "N_ROWS_COVERED": 4,
        "N_PARTITIONS": 2,
        "PARTITIONS": row["PARTITIONS"].replace("0", "1", 1),
        "ROW_STARTS": np.array([1.0]),
        "ROW_LENGTHS": np.array([4.0]),
        "PARTITION_BOUNDS": np.array([0.0, 0.0, 1.0]),
        "FINGERPRINT": '{"main": {"nrows": 6}}',
        "HISTORY_ROW": 4,
        "HISTORY_NROWS_AT_BUILD": 3,
    }
    assert sorted(changes) == sorted(set(SUBTABLE_COLUMNS) - {"CHECKSUM"})
    for name, value in changes.items():
        assert row_checksum(dict(row, **{name: value})) != row["CHECKSUM"], name


# --- the row checks (L3) ---------------------------------------------------------------


# The scheme of the rows of the row-check tests: rich has partitions of
# one and of two runs
L3_SCHEME = ["FIELD_ID"]


def _rich_row(backend_ms):
    msname = backend_ms("rich")
    partitions, runs = create_partitions_with_main_rows(msname, L3_SCHEME)
    values = encode_row(partitions, runs, L3_SCHEME, fingerprint_json(msname), 2, 2)
    return normalise_row(values)


def _swap_runs_of_a_partition(row):
    """The two first runs of a partition of several runs, swapped."""
    bounds = row["PARTITION_BOUNDS"]
    first = next(
        int(bounds[p]) for p in range(bounds.size - 1) if bounds[p + 1] - bounds[p] > 1
    )
    order = np.arange(row["N_RUNS"])
    order[[first, first + 1]] = order[[first + 1, first]]
    return _set(
        row, ROW_STARTS=row["ROW_STARTS"][order], ROW_LENGTHS=row["ROW_LENGTHS"][order]
    )


def _set(row, **cells):
    row = copy.deepcopy(row)
    row.update(cells)
    return row


def _partitions(row, change):
    partitions = decode_partitions(row["PARTITIONS"])
    change(partitions)
    keys = list(partitions[0]) if partitions else []
    return _set(row, PARTITIONS=encode_partitions(partitions, keys))


L3_CASES = {
    "format version": lambda r: _set(r, FORMAT_VERSION=2),
    "algorithm version": lambda r: _set(r, ALGORITHM_VERSION=99),
    "scheme key": lambda r: _set(r, SCHEME_KEY="[]"),
    "partition count": lambda r: _set(r, N_PARTITIONS=r["N_PARTITIONS"] + 1),
    "bounds count": lambda r: _set(
        r, PARTITION_BOUNDS=r["PARTITION_BOUNDS"][:-1].copy()
    ),
    "bounds start": lambda r: _set(r, PARTITION_BOUNDS=r["PARTITION_BOUNDS"] + 1),
    "bounds decrease": lambda r: _set(
        r,
        PARTITION_BOUNDS=r["PARTITION_BOUNDS"][
            [0, 2, 1] + list(range(3, r["PARTITION_BOUNDS"].size))
        ],
    ),
    "run count": lambda r: _set(r, N_RUNS=r["N_RUNS"] + 1),
    "empty run": lambda r: _set(
        r, ROW_LENGTHS=np.concatenate([[0], r["ROW_LENGTHS"][1:]])
    ),
    "beyond MAIN": lambda r: _set(r, MAIN_NROWS=r["MAIN_NROWS"] - 1),
    "runs not ascending": _swap_runs_of_a_partition,
    "overlapping runs": lambda r: _set(
        r,
        ROW_LENGTHS=np.concatenate([[r["ROW_LENGTHS"][0] + 10], r["ROW_LENGTHS"][1:]]),
    ),
    "rows covered": lambda r: _set(r, N_ROWS_COVERED=r["N_ROWS_COVERED"] - 1),
    "fingerprint rows": lambda r: _set(
        r, FINGERPRINT=json.dumps({"main": {"nrows": r["MAIN_NROWS"] + 1}})
    ),
    "fingerprint JSON": lambda r: _set(r, FINGERPRINT="not JSON"),
    "multi-valued key": lambda r: _partitions(
        r, lambda ps: ps[0].update(FIELD_ID=[0, 1])
    ),
    "same keys": lambda r: _partitions(
        r, lambda ps: ps[1].update({k: list(v) for k, v in ps[0].items()})
    ),
    "description keys": lambda r: _partitions(
        r, lambda ps: [p.pop("EPHEMERIS_ID") for p in ps]
    ),
    "rows in no partition": lambda r: _set(
        r,
        ROW_STARTS=r["ROW_STARTS"][:-1].copy(),
        ROW_LENGTHS=r["ROW_LENGTHS"][:-1].copy(),
        N_RUNS=r["N_RUNS"] - 1,
        N_ROWS_COVERED=r["N_ROWS_COVERED"] - int(r["ROW_LENGTHS"][-1]),
        PARTITION_BOUNDS=np.minimum(r["PARTITION_BOUNDS"], r["N_RUNS"] - 1),
    ),
}


def test_valid_row_passes(backend_ms):
    row = _rich_row(backend_ms)
    partitions, runs = decode_row(row, L3_SCHEME)
    assert len(partitions) == row["N_PARTITIONS"] == 16
    assert runs.starts.size == row["N_RUNS"] > 16  # (partitions of 2 runs)


@pytest.mark.parametrize("case", list(L3_CASES))
def test_row_checks(case, backend_ms):
    """Every rule of the row checks rejects a row that breaks it."""
    row = L3_CASES[case](_rich_row(backend_ms))
    with pytest.raises(InvalidRowError):
        decode_row(row, L3_SCHEME)


def test_antenna1_rows_need_not_cover_main(backend_ms):
    msname = backend_ms("dense")
    partitions, runs = create_partitions_with_main_rows(msname, ["ANTENNA1"])
    assert runs.lengths.sum() == 0 < runs.main_nrows  # (no autocorrelations)
    row = normalise_row(
        encode_row(partitions, runs, ["ANTENNA1"], fingerprint_json(msname), 0, 0)
    )
    assert decode_row(row, ["ANTENNA1"])[0] == partitions


# --- reading stored rows ------------------------------------------------------------------


def test_stored_row_is_used(ms_copy, monkeypatch):
    """A valid stored row is a hit (no partitions computed), then a memo hit;
    the result equals the computed one and is a copy."""
    msname = ms_copy("rich")
    scheme = ["FIELD_ID", "SCAN_NUMBER"]
    expected, expected_runs = create_partitions_with_main_rows(msname, scheme)
    store_test_row(msname, scheme)
    no_compute(monkeypatch)
    for mode, source, status in [
        ("read", "stored", "hit"),
        ("auto", "memo", "hit-memory"),
        ("read", "memo", "hit-memory"),
    ]:
        result = load(msname, scheme, mode)
        assert (result.source, result.status) == (source, status)
        assert result.partitions == expected
        np.testing.assert_array_equal(result.runs.starts, expected_runs.starts)
        result.partitions[0]["FIELD_ID"] = [99]  # (a copy)
        result.runs.starts[:] = -1


def test_stored_row_tree_equals_the_computed_tree(ms_copy, monkeypatch):
    """The tree from a stored row equals the tree of computed partitions."""
    msname = ms_copy("rich")
    options = {"partition_scheme": ["STATE_ID"], "chunks": {}}
    fresh = xr.open_datatree(msname, engine=ENGINE, partition_cache="off", **options)
    store_test_row(msname, ["STATE_ID"])
    no_compute(monkeypatch)
    cached = xr.open_datatree(msname, engine=ENGINE, partition_cache="read", **options)
    assert_nodes_identical(cached, fresh)


@pytest.mark.parametrize(
    "mode, expected", [("off", "memory:mode-off"), ("rebuild", "stored")]
)
def test_modes_that_do_not_read_stored_rows(mode, expected, ms_copy):
    msname = ms_copy("dense")
    store_test_row(msname, [], **{"PARTITIONS": "corrupt but checksummed"})
    result = load(msname, [], mode)
    assert (result.source, result.status) == ("fresh", expected)


def test_read_mode_without_a_stored_row(ms_copy):
    msname = ms_copy("dense")
    result = load(msname, [], "read")
    assert (result.source, result.status) == ("fresh", "memory:mode-read")
    assert not os.path.exists(os.path.join(msname, SUBTABLE_NAME))


def test_rows_of_other_schemes_and_algorithm_versions(ms_copy):
    """Rows are looked up by (scheme key, algorithm version): other schemes
    and versions are left alone; the key does not depend on the order of the
    keys or the mandatory keys."""
    msname = ms_copy("rich")
    store_test_row(msname, ["SCAN_NUMBER"])
    other_version = store_test_row(
        msname,
        ["FIELD_ID", "SCAN_NUMBER"],
        ALGORITHM_VERSION=PARTITION_ALGORITHM_VERSION + 1,
        PARTITIONS="from another algorithm",
    )
    store_test_row(msname, ["FIELD_ID", "SCAN_NUMBER"])
    result = load(msname, ["SCAN_NUMBER", "DATA_DESC_ID", "FIELD_ID"])
    assert (result.source, result.status) == ("stored", "hit")
    assert len(stored_rows(msname)) == 3
    assert other_version["SCHEME_KEY"] == canonical_scheme_key(
        ["FIELD_ID", "SCAN_NUMBER"]
    )
    assert load(msname, ["SCAN_NUMBER"]).status == "hit"
    assert load(msname, ["FIELD_ID"]).status == "memory:mode-read"


def test_rows_of_other_schemes_do_not_make_a_row_stale(ms_copy):
    """HISTORY rows of the cache itself (other schemes) after the anchor."""
    msname = ms_copy("rich")
    store_test_row(msname, [])
    for scheme in (["FIELD_ID"], ["SCAN_NUMBER"]):
        store_test_row(msname, scheme)
    assert load(msname, []).status == "hit"


@pytest.mark.parametrize(
    "corruption, state",
    [
        ({"checksum": "0" * 32}, "torn"),
        ({"PARTITIONS": "[]"}, "corrupt"),
        ({"MAIN_NROWS": 3}, "corrupt"),
    ],
)
def test_corrupt_rows_are_not_used(corruption, state, ms_copy):
    msname = ms_copy("dense")
    store_test_row(msname, [], **corruption)
    lookup = lookup_stored_row(msname, "[]")
    assert lookup.state == ("torn" if state == "torn" else "row")
    result = load(msname, [])
    assert (result.source, result.status) == ("fresh", "memory:mode-read")


def test_torn_reads_are_read_again(ms_copy, monkeypatch):
    msname = ms_copy("dense")
    store_test_row(msname, [])
    read_row_cells = partition_cache.read_row_cells
    calls = []

    def torn_twice(table, index):
        calls.append(index)
        row = read_row_cells(table, index)
        if len(calls) == 1:
            row["N_RUNS"] += 1  # (a torn row: CHECKSUM mismatch)
        if len(calls) == 2:
            raise RuntimeError("FilebufIO::readBlock - incorrect number of bytes")
        return row

    monkeypatch.setattr(partition_cache, "read_row_cells", torn_twice)
    assert lookup_stored_row(msname, "[]").state == "row"
    assert len(calls) == 3
    calls.clear()

    def always_raise(table, index):
        calls.append(index)
        raise RuntimeError("simulated torn read")

    monkeypatch.setattr(partition_cache, "read_row_cells", always_raise)
    lookup = lookup_stored_row(msname, "[]")
    assert lookup.state == "unreadable" and "simulated" in lookup.detail
    assert len(calls) == partition_cache.READ_ATTEMPTS


@pytest.mark.parametrize(
    "setup, state",
    [
        ("unlinked", "unlinked"),
        ("dangling", "dangling"),
        ("foreign", "foreign"),
        ("newer format", "newer format"),
        ("not ours", "not an xradio cache"),
        ("duplicate", "duplicate rows"),
    ],
)
def test_links_and_tables_that_are_not_used(setup, state, ms_copy, monkeypatch):
    msname = ms_copy("dense")
    subtable = os.path.join(msname, SUBTABLE_NAME)
    store_test_row(msname, [], link=setup not in ("unlinked", "foreign"))
    if setup == "duplicate":
        store_test_row(msname, [])
    if setup in ("newer format", "not ours"):
        with tables.table(subtable, readonly=False, ack=False) as table:
            if setup == "newer format":
                table.putkeyword("FORMAT_VERSION", 2)
            else:
                table.putkeyword("CREATOR", "someone else")
    with tables.table(msname, readonly=False, ack=False) as main_tb:
        if setup == "foreign":
            main_tb.putkeyword(SUBTABLE_NAME, "a value")
    if setup == "dangling":
        os.rename(subtable, subtable + "_moved")
    assert lookup_stored_row(msname, "[]").state == state
    assert load(msname, []).status == "memory:mode-read"


# --- the HISTORY rule ----------------------------------------------------------------------


def test_history_rule(ms_copy):
    """L2: stale after a row of another application, or when HISTORY lost
    rows (truncated, re-created) or the stored content's row; by row number,
    whatever the TIME of the rows."""
    msname = ms_copy("dense")
    store_test_row(msname, [])
    assert load(msname, []).status == "hit"
    # a foreign row with a TIME before ours (HISTORY TIME is not monotonic)
    row = add_history_row(msname, "ms", "flagdata", when=1.0)
    partition_cache.clear_partition_memo()
    assert load(msname, []).status == "memory:mode-read"
    lookup = lookup_stored_row(msname, "[]")
    reason = partition_cache.history_rule(msname, lookup.row, history_nrows(msname))
    assert f"HISTORY row {row} (ms, flagdata)" in reason


def test_history_rule_cases(ms_copy):
    msname = ms_copy("dense")
    values = store_test_row(msname, [])
    row = normalise_row(values)
    n_history = history_nrows(msname)
    rule = partition_cache.history_rule
    assert rule(msname, row, n_history) is None
    # rows of xradio's cache after the anchor: not newer
    add_history_row(msname)
    assert rule(msname, row, n_history + 1) is None
    # fewer rows than the anchor
    assert "rows" in rule(
        msname, _set(row, HISTORY_NROWS_AT_BUILD=n_history + 5), n_history
    )
    # the row of the content: gone, foreign, or without its cache_id
    assert "gone" in rule(msname, _set(row, HISTORY_ROW=n_history + 7), n_history)
    assert "not that" in rule(msname, _set(row, HISTORY_ROW=0), n_history)
    assert "not that" in rule(msname, _set(row, CACHE_ID="f" * 32), n_history)
    # an xradio row of another origin after the anchor
    add_history_row(msname, HISTORY_APPLICATION, "xradio.some_other_writer")
    assert "newer" in rule(msname, row, n_history + 2)
    # no HISTORY
    assert rule(msname, row, None) == "no HISTORY table"


def test_history_recreated(ms_copy):
    msname = ms_copy("dense")
    store_test_row(msname, [])
    history = os.path.join(msname, "HISTORY")
    with tables.table(history, ack=False) as table:
        desc, dminfo = table.getdesc(), table.getdminfo()
    import shutil

    shutil.rmtree(history)
    tables.table(
        history, desc, dminfo=dminfo, nrow=0, readonly=False, ack=False
    ).close()
    assert load(msname, []).status == "memory:mode-read"


# --- the fingerprint (L1) and the memo ------------------------------------------------------


def test_changed_fingerprint(ms_copy):
    msname = ms_copy("rich")
    store_test_row(msname, ["FIELD_ID"])
    assert load(msname, ["FIELD_ID"]).status == "hit"
    with tables.table(os.path.join(msname, "FIELD"), readonly=False, ack=False) as t:
        t.putcell("NAME", 0, "another name")
    assert load(msname, ["FIELD_ID"]).status == "memory:mode-read"
    assert lookup_stored_row(msname, '["FIELD_ID"]').state == "row"


def test_memo_validity(ms_copy, monkeypatch):
    """A memo entry is valid while the fingerprint and the HISTORY rows are
    unchanged; it has no time limit."""
    msname = ms_copy("dense")
    assert load(msname, [], "auto").source == "fresh"
    assert load(msname, [], "auto").source == "memo"
    future = time.time() + 10 * 86400
    monkeypatch.setattr(time, "time", lambda: future)
    assert load(msname, [], "auto").source == "memo"
    add_history_row(msname, "ms", "flagdata")
    assert load(msname, [], "auto").source == "fresh"
    assert load(msname, [], "auto").source == "memo"
    with tables.table(msname, readonly=False, ack=False) as main_tb:
        main_tb.putcol("SCAN_NUMBER", main_tb.getcol("SCAN_NUMBER") + 1)
    assert load(msname, [], "auto").source == "fresh"


def test_no_fingerprint(ms_copy, monkeypatch):
    msname = ms_copy("dense")
    store_test_row(msname, [])

    def fail(*args, **kwargs):
        raise RuntimeError("no dminfo")

    monkeypatch.setattr(partition_cache, "ms_fingerprint", fail)
    for _ in range(2):
        result = load(msname, [], "auto")
        assert (result.source, result.status) == ("fresh", "memory:no-fingerprint")
    assert len(PARTITIONS_MEMO) == 0


def test_changed_during_build(ms_copy, monkeypatch):
    msname = ms_copy("dense")
    compute = partition_cache.create_partitions_with_main_rows

    def compute_and_change(path, scheme):
        result = compute(path, scheme)
        with tables.table(path, readonly=False, ack=False) as main_tb:
            main_tb.putcol("FIELD_ID", main_tb.getcol("FIELD_ID"))
        return result

    monkeypatch.setattr(
        partition_cache, "create_partitions_with_main_rows", compute_and_change
    )
    result = load(msname, [], "auto")
    assert result.status == "memory:changed-during-build"
    assert len(PARTITIONS_MEMO) == 0


# --- storing ------------------------------------------------------------------------------


def file_digests(path: str) -> dict[str, tuple[int, str]]:
    """(size, sha256) of every file under ``path`` (table.lock included)."""
    digests = {}
    for dirpath, _, filenames in os.walk(path):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            with open(file_path, "rb") as f:
                content = f.read()
            digests[os.path.relpath(file_path, path)] = (
                len(content),
                hashlib.sha256(content).hexdigest(),
            )
    return digests


def history_rows(msname: str) -> list[dict]:
    with tables.table(os.path.join(msname, "HISTORY"), ack=False) as h:
        return [
            {name: h.getcell(name, row) for name in h.colnames()}
            for row in range(h.nrows())
        ]


def main_keywords(msname: str) -> list[str]:
    with tables.table(msname, ack=False) as main_tb:
        return list(main_tb.keywordnames())


def statuses(msname, scheme, modes=("auto",), clear=True):
    """The status of a load in every mode (the memo emptied before each)."""
    found = []
    for mode in modes:
        if clear:
            partition_cache.clear_partition_memo()
        found.append(load(msname, scheme, mode).status)
    return found


def test_first_open_stores(ms_copy, monkeypatch):
    """The first "auto" open stores the partitions: the sub-table (table
    keywords and info), the MAIN keyword (stored relative) and a HISTORY row
    with the documented fields. The next open is a hit, then a memo hit."""
    msname = ms_copy("rich")
    before = history_rows(msname)
    tree = xr.open_datatree(
        msname,
        engine=ENGINE,
        partition_cache="auto",
        partition_scheme=["SCAN_NUMBER", "FIELD_ID"],
    )
    assert len(tree.children) == 24
    rows = stored_rows(msname)
    assert len(rows) == 1
    row = normalise_row(rows[0])
    assert row["SCHEME_KEY"] == '["FIELD_ID", "SCAN_NUMBER"]'
    assert json.loads(row["SCHEME"]) == ["SCAN_NUMBER", "FIELD_ID"]
    assert row["N_PARTITIONS"] == 24 and row["MAIN_NROWS"] == 1200
    assert re.fullmatch("[0-9a-f]{32}", row["CACHE_ID"])
    subtable = os.path.join(msname, SUBTABLE_NAME)
    with tables.table(subtable, ack=False) as table:
        assert table.getkeyword("FORMAT_VERSION") == 1
        assert table.getkeyword("CREATOR") == "xradio"
        assert table.info()["type"] == "XRADIO Partitions"
        assert "remove_msv2_partition_cache" in table.info()["readme"]
    with tables.table(msname, ack=False) as main_tb:
        assert main_tb.getkeyword(SUBTABLE_NAME) == "Table: " + subtable
    with open(os.path.join(msname, "table.dat"), "rb") as f:
        assert b"././XRADIO_PARTITIONS" in f.read()
    history = history_rows(msname)
    assert len(history) == len(before) + 1
    new = history[-1]
    assert row["HISTORY_ROW"] == len(before) == row["HISTORY_NROWS_AT_BUILD"]
    assert new["APPLICATION"] == "xradio"
    assert new["ORIGIN"] == "xradio.measurement_set.open_msv2"
    assert (new["PRIORITY"], new["OBSERVATION_ID"], new["OBJECT_ID"]) == (
        "INFO",
        -1,
        0,
    )
    assert list(new["CLI_COMMAND"]) == [""]
    version = partition_cache._xradio_version()
    assert list(new["APP_PARAMS"]) == [
        "subtable=XRADIO_PARTITIONS",
        f"cache_id={row['CACHE_ID']}",
        'scheme_key=["FIELD_ID", "SCAN_NUMBER"]',
        "format_version=1",
        f"algorithm_version={PARTITION_ALGORITHM_VERSION}",
        f"xradio_version={version}",
        "reason=first",
    ]
    assert new["MESSAGE"] == (
        f"xradio {version}: stored 24 partitions (24 MAIN row runs) of "
        'partition_scheme ["SCAN_NUMBER", "FIELD_ID"] in XRADIO_PARTITIONS (first)'
    )
    assert abs(new["TIME"] - (time.time() + 3506716800.0)) < 600
    # the next opens
    assert load(msname, ["FIELD_ID", "SCAN_NUMBER"], "auto").status == "hit-memory"
    assert statuses(msname, ["FIELD_ID", "SCAN_NUMBER"], ("auto", "read")) == [
        "hit",
        "hit",
    ]
    assert load(msname, ["FIELD_ID", "SCAN_NUMBER"], "auto").status == "hit-memory"
    assert len(history_rows(msname)) == len(before) + 1


def _store_contents(path: str) -> dict[str, str]:
    """sha256 of every file of a converted processing set, the dates of the
    zarr.json metadata masked."""
    contents = {}
    for dirpath, _, filenames in os.walk(path):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            with open(file_path, "rb") as f:
                content = f.read()
            if filename == "zarr.json":
                content = re.sub(
                    rb'"(creation_date|date)": "[^"]*"', b'"<date>"', content
                )
            contents[os.path.relpath(file_path, path)] = hashlib.sha256(
                content
            ).hexdigest()
    return contents


def test_conversion_unchanged_by_the_stored_partitions(ms_copy, tmp_path):
    """I2: the converter writes the same processing set before and after the
    partitions were stored in the MS (the MAIN keyword is no attribute)."""
    msname = ms_copy("rich")
    options = {"partition_scheme": ["FIELD_ID"], "with_pointing": True}
    convert_msv2_to_processing_set(msname, str(tmp_path / "before.ps.zarr"), **options)
    assert load(msname, ["FIELD_ID"], "auto").status == "stored"
    convert_msv2_to_processing_set(msname, str(tmp_path / "after.ps.zarr"), **options)
    before = _store_contents(str(tmp_path / "before.ps.zarr"))
    after = _store_contents(str(tmp_path / "after.ps.zarr"))
    assert len(before) > 100 and before == after


def test_revalidation_adds_no_history_row(ms_copy):
    """A stale row with the partitions computed again: a new fingerprint and
    anchor, no HISTORY row (e.g. after a flagdata HISTORY row, or a FLAG_ROW
    write into the data manager of the key columns)."""
    msname = ms_copy("rich")
    assert statuses(msname, []) == ["stored"]
    first = normalise_row(stored_rows(msname)[0])
    add_history_row(msname, "ms", "flagdata")
    n_history = len(history_rows(msname))
    assert statuses(msname, []) == ["revalidated"]
    row = normalise_row(stored_rows(msname)[0])
    assert len(history_rows(msname)) == n_history
    assert row["HISTORY_NROWS_AT_BUILD"] == n_history
    assert row["HISTORY_ROW"] == first["HISTORY_ROW"]
    assert row["CACHE_ID"] == first["CACHE_ID"]
    with tables.table(msname, readonly=False, ack=False) as main_tb:
        main_tb.putcol("FLAG_ROW", main_tb.getcol("FLAG_ROW"))  # (shared SSM)
    assert statuses(msname, []) == ["revalidated"]
    assert normalise_row(stored_rows(msname)[0])["FINGERPRINT"] != row["FINGERPRINT"]
    assert statuses(msname, [], ("auto", "read")) == ["hit", "hit"]
    assert len(history_rows(msname)) == n_history


def test_history_rows_removed_before_the_content_row(ms_copy):
    """HISTORY rows before the content's HISTORY row removed (its row number
    shifts): the partitions are computed again and are the same, so the
    stored row is revalidated with the new row number of its HISTORY row,
    and no HISTORY row is added."""
    msname = ms_copy("rich")
    assert statuses(msname, []) == ["stored"]
    first = normalise_row(stored_rows(msname)[0])
    assert first["HISTORY_ROW"] > 0
    with tables.table(os.path.join(msname, "HISTORY"), readonly=False, ack=False) as h:
        h.removerows([0])
    n_history = len(history_rows(msname))
    assert statuses(msname, []) == ["revalidated"]
    row = normalise_row(stored_rows(msname)[0])
    assert len(history_rows(msname)) == n_history
    assert row["CACHE_ID"] == first["CACHE_ID"]
    assert row["HISTORY_ROW"] == first["HISTORY_ROW"] - 1
    assert row["HISTORY_NROWS_AT_BUILD"] == n_history
    assert statuses(msname, [], ("auto", "read")) == ["hit", "hit"]
    # the content's HISTORY row itself removed: stored again, with a new one
    with tables.table(os.path.join(msname, "HISTORY"), readonly=False, ack=False) as h:
        h.removerows([row["HISTORY_ROW"]])
    assert statuses(msname, []) == ["stored"]
    assert len(history_rows(msname)) == n_history
    assert normalise_row(stored_rows(msname)[0])["CACHE_ID"] != first["CACHE_ID"]


def test_changed_partitions_replace_the_row(ms_copy):
    msname = ms_copy("rich")
    assert statuses(msname, ["FIELD_ID"]) == ["stored"]
    first = normalise_row(stored_rows(msname)[0])
    with tables.table(msname, readonly=False, ack=False) as main_tb:
        field = main_tb.getcol("FIELD_ID")
        field[:50] = 1 - field[:50]
        main_tb.putcol("FIELD_ID", field)
    assert statuses(msname, ["FIELD_ID"]) == ["stored"]
    rows = stored_rows(msname)
    assert len(rows) == 1
    row = normalise_row(rows[0])
    assert row["CACHE_ID"] != first["CACHE_ID"]
    assert row["N_RUNS"] != first["N_RUNS"]
    history = history_rows(msname)
    assert row["HISTORY_ROW"] == len(history) - 1
    assert "reason=stale:fingerprint" in list(history[-1]["APP_PARAMS"])
    partitions, runs = create_partitions_with_main_rows(msname, ["FIELD_ID"])
    assert decode_row(row, ["FIELD_ID"])[0] == partitions
    assert statuses(msname, ["FIELD_ID"]) == ["hit"]


def test_rebuild(ms_copy):
    """rebuild: a HISTORY row only if the partitions changed."""
    msname = ms_copy("dense")
    assert statuses(msname, [], ("rebuild",)) == ["stored"]
    n_history = len(history_rows(msname))
    assert statuses(msname, [], ("rebuild", "rebuild")) == ["revalidated"] * 2
    assert len(history_rows(msname)) == n_history
    with tables.table(msname, readonly=False, ack=False) as main_tb:
        main_tb.putcol("SCAN_NUMBER", main_tb.getcol("SCAN_NUMBER") + 1)
    assert statuses(msname, [], ("rebuild", "auto")) == ["stored", "hit"]
    assert "reason=rebuild" in list(history_rows(msname)[-1]["APP_PARAMS"])


@pytest.mark.parametrize(
    "change, reason",
    [("corrupt", "stale:corrupt"), ("history", "stale:history")],
)
def test_rows_stored_again(change, reason, ms_copy):
    """A corrupt row, or one whose HISTORY row is gone (HISTORY re-created),
    is stored again with a new HISTORY row (no revalidation)."""
    msname = ms_copy("dense")
    assert statuses(msname, []) == ["stored"]
    if change == "corrupt":
        with tables.table(
            os.path.join(msname, SUBTABLE_NAME), readonly=False, ack=False
        ) as table:
            table.putcell("CHECKSUM", 0, "0" * 32)
    else:
        history = os.path.join(msname, "HISTORY")
        with tables.table(history, ack=False) as table:
            desc, dminfo = table.getdesc(), table.getdminfo()
        shutil.rmtree(history)
        tables.table(
            history, desc, dminfo=dminfo, nrow=0, readonly=False, ack=False
        ).close()
    assert statuses(msname, []) == ["stored"]
    assert len(stored_rows(msname)) == 1
    assert f"reason={reason}" in list(history_rows(msname)[-1]["APP_PARAMS"])
    assert statuses(msname, []) == ["hit"]


def test_newer_format_is_neither_read_nor_written(ms_copy, caplog):
    msname = ms_copy("dense")
    assert statuses(msname, []) == ["stored"]
    with tables.table(
        os.path.join(msname, SUBTABLE_NAME), readonly=False, ack=False
    ) as table:
        table.putkeyword("FORMAT_VERSION", 2)
    before = file_digests(msname)
    assert statuses(msname, []) == ["memory:cache written by a newer xradio"]
    assert statuses(msname, ["FIELD_ID"]) == ["memory:cache written by a newer xradio"]
    assert file_digests(msname) == before


def test_rows_of_another_algorithm_version_are_kept(ms_copy):
    msname = ms_copy("dense")
    store_test_row(msname, [], ALGORITHM_VERSION=PARTITION_ALGORITHM_VERSION + 1)
    assert statuses(msname, []) == ["stored"]
    versions = sorted(int(row["ALGORITHM_VERSION"]) for row in stored_rows(msname))
    assert versions == [PARTITION_ALGORITHM_VERSION, PARTITION_ALGORITHM_VERSION + 1]


def test_at_most_8_rows(ms_copy, monkeypatch):
    """A ninth row evicts the row stored first (the oldest TIME)."""
    msname = ms_copy("rich")
    schemes = [
        [],
        ["FIELD_ID"],
        ["SCAN_NUMBER"],
        ["STATE_ID"],
        ["SOURCE_ID"],
        ["SUB_SCAN_NUMBER"],
        ["ANTENNA1"],
        ["FIELD_ID", "SCAN_NUMBER"],
        ["FIELD_ID", "STATE_ID"],
    ]
    for scheme in schemes:
        assert statuses(msname, scheme) == ["stored"]
    keys = [row["SCHEME_KEY"] for row in stored_rows(msname)]
    assert len(keys) == 8 and "[]" not in keys
    assert statuses(msname, ["FIELD_ID", "STATE_ID"]) == ["hit"]
    assert statuses(msname, []) == ["stored"]
    keys = [row["SCHEME_KEY"] for row in stored_rows(msname)]
    assert len(keys) == 8 and '["FIELD_ID"]' not in keys


def test_dangling_keyword_is_repaired(ms_copy, monkeypatch):
    """A MAIN keyword whose sub-table is gone (CASA's mstransform fails on
    it): the next "auto" open stores the sub-table again; when nothing can
    be stored, the keyword is removed."""
    msname = ms_copy("dense")
    assert statuses(msname, []) == ["stored"]
    shutil.rmtree(os.path.join(msname, SUBTABLE_NAME))
    assert partition_cache.link_state(msname) == "dangling"
    assert statuses(msname, ["FIELD_ID"]) == ["stored"]
    assert partition_cache.link_state(msname) == "linked"
    shutil.rmtree(os.path.join(msname, SUBTABLE_NAME))
    monkeypatch.setattr(partition_cache, "MAX_STORED_RUNS", 0)
    assert statuses(msname, []) == ["memory:too many runs"]
    assert partition_cache.link_state(msname) == "absent"
    assert statuses(msname, [], ("read",)) == ["memory:mode-read"]


def test_remove_msv2_partition_cache(ms_copy, monkeypatch):
    msname = ms_copy("dense")
    for scheme in ([], ["FIELD_ID"]):
        assert statuses(msname, scheme) == ["stored"]
    n_history = len(history_rows(msname))
    assert load(msname, [], "auto").source == "stored"
    assert partition_cache.memo_key(msname, []) in PARTITIONS_MEMO
    assert remove_msv2_partition_cache(msname) is True
    assert SUBTABLE_NAME not in os.listdir(msname)
    assert SUBTABLE_NAME not in main_keywords(msname)
    assert len(history_rows(msname)) == n_history
    assert load(msname, [], "read").status == "memory:mode-read"  # (memo emptied)
    assert remove_msv2_partition_cache(msname) is False
    # the keyword and the sub-table's name first: a failure to delete the
    # sub-table leaves no dangling keyword (and a temporary sub-table, removed
    # once this process is gone)
    assert statuses(msname, []) == ["stored"]

    def fail(path, *args, **kwargs):
        raise OSError("simulated")

    monkeypatch.setattr(partition_cache.shutil, "rmtree", fail)
    with pytest.raises(OSError, match="simulated"):
        remove_msv2_partition_cache(msname)
    assert partition_cache.link_state(msname) == "absent"
    monkeypatch.undo()
    (left,) = (
        name
        for name in os.listdir(msname)
        if name.startswith(partition_cache.TMP_PREFIX)
    )
    assert f"-{os.getpid()}-" in left
    assert statuses(msname, []) == ["stored"]
    assert remove_msv2_partition_cache(msname) is True
    assert partition_cache.link_state(msname) == "absent"
    with pytest.raises(FileNotFoundError):
        remove_msv2_partition_cache(os.path.join(msname, "nothing"))


def test_remove_while_another_process_stores(ms_copy):
    """Another process opens the MS ("auto") right after the removal took
    the keyword and the sub-table away and before the sub-table is deleted:
    it stores a sub-table of its own and links it; the removal deletes only
    its own, so no dangling keyword is left (the review's race: it linked
    the sub-table being deleted)."""
    msname = ms_copy("dense")
    assert statuses(msname, []) == ["stored"]
    real_rmtree = shutil.rmtree
    raced = []

    def racing_rmtree(path, *args, **kwargs):
        name = os.path.basename(str(path))
        if not raced and (
            name == SUBTABLE_NAME or name.startswith(partition_cache.TMP_PREFIX)
        ):
            code = (
                "from xradio.measurement_set._utils._msv2 import partition_cache\n"
                f"result = partition_cache.load_or_create_partitions({msname!r}, [], "
                "'auto')\n"
                "print(result.status)\n"
            )
            run = subprocess.run(
                [sys.executable, "-c", code], capture_output=True, text=True, check=True
            )
            raced.append(run.stdout.strip().splitlines()[-1])
        return real_rmtree(path, *args, **kwargs)

    partition_cache.shutil.rmtree = racing_rmtree
    try:
        assert remove_msv2_partition_cache(msname) is True
    finally:
        partition_cache.shutil.rmtree = real_rmtree
    assert raced == ["stored"]
    assert partition_cache.link_state(msname) == "linked"
    assert statuses(msname, [], ("read",)) == ["hit"]


@pytest.mark.skipif(os.geteuid() == 0, reason="root writes read-only files")
def test_remove_from_a_read_only_ms(ms_copy):
    msname = ms_copy("dense")
    assert statuses(msname, []) == ["stored"]
    os.chmod(msname, 0o555)
    try:
        with pytest.raises(PermissionError):
            remove_msv2_partition_cache(msname)
    finally:
        os.chmod(msname, 0o755)
    assert partition_cache.link_state(msname) == "linked"


def hold_table(msname: str, lockoptions: str | None = "default"):
    """A subprocess that opens a table (python-casacore; "default": auto
    locking), reads a column and waits until killed."""
    options = "" if lockoptions == "default" else f", lockoptions={lockoptions!r}"
    script = (
        "import sys, time\n"
        "from casacore import tables\n"
        f"t = tables.table({msname!r}, ack=False{options})\n"
        "t.getcol('TIME')\n"
        "print('ready', flush=True)\n"
        "time.sleep(120)\n"
    )
    holder = subprocess.Popen(
        [sys.executable, "-c", script], stdout=subprocess.PIPE, text=True
    )
    assert holder.stdout.readline().strip() == "ready"
    return holder


def test_remove_with_main_locked_elsewhere(ms_copy):
    msname = ms_copy("dense")
    assert statuses(msname, []) == ["stored"]
    holder = hold_table(msname)
    try:
        with pytest.raises(RuntimeError, match="lock"):
            remove_msv2_partition_cache(msname)
    finally:
        holder.kill()
        holder.wait()
    assert partition_cache.link_state(msname) == "linked"
    assert remove_msv2_partition_cache(msname) is True


@pytest.mark.parametrize("link", [False, True])
def test_remove_leaves_a_foreign_subtable(link, ms_copy):
    """A table or directory XRADIO_PARTITIONS that is not xradio's partition
    cache (no CREATOR keyword "xradio"): neither it nor a MAIN keyword that
    links it is removed (ValueError)."""
    msname = ms_copy("dense")
    subtable = os.path.join(msname, SUBTABLE_NAME)
    desc = tables.maketabdesc([tables.makescacoldesc("TIME", 0.0)])
    tables.table(subtable, desc, nrow=1, ack=False).close()
    if link:
        with tables.table(msname, readonly=False, ack=False) as main_tb:
            main_tb.putkeyword(SUBTABLE_NAME, "Table: " + subtable)
    before = file_digests(subtable)
    with pytest.raises(ValueError, match="not a partition cache of xradio"):
        remove_msv2_partition_cache(msname)
    assert file_digests(subtable).keys() == before.keys()
    assert (SUBTABLE_NAME in main_keywords(msname)) == link
    shutil.rmtree(subtable)
    os.makedirs(subtable)
    with pytest.raises(ValueError, match="not a readable table"):
        remove_msv2_partition_cache(msname)
    assert os.path.isdir(subtable)


def test_remove_with_the_subtable_locked_elsewhere(ms_copy):
    """A sub-table that another process holds locked (a writer storing a
    row): nothing is removed (RuntimeError)."""
    msname = ms_copy("dense")
    assert statuses(msname, []) == ["stored"]
    holder = hold_table(os.path.join(msname, SUBTABLE_NAME))
    try:
        with pytest.raises(RuntimeError, match="lock"):
            remove_msv2_partition_cache(msname)
    finally:
        holder.kill()
        holder.wait()
    assert partition_cache.link_state(msname) == "linked"
    assert remove_msv2_partition_cache(msname) is True
    assert partition_cache.link_state(msname) == "absent"


def test_stale_temporary_tables_are_removed(ms_copy):
    msname = ms_copy("dense")
    done = subprocess.run(
        [sys.executable, "-c", "import os; print(os.getpid())"],
        capture_output=True,
        text=True,
        check=True,
    )
    dead_pid = int(done.stdout)
    host = socket.gethostname()
    names = {
        "dead": f"{partition_cache.TMP_PREFIX}{host}-{dead_pid}-0000aaaa",
        "alive": f"{partition_cache.TMP_PREFIX}{host}-{os.getpid()}-0000bbbb",
        "old elsewhere": f"{partition_cache.TMP_PREFIX}other-host-1-0000cccc",
        "new elsewhere": f"{partition_cache.TMP_PREFIX}other-host-1-0000dddd",
    }
    for name in names.values():
        os.makedirs(os.path.join(msname, name, "sub"))
    old = time.time() - 2 * partition_cache.TMP_MAX_AGE
    os.utime(os.path.join(msname, names["old elsewhere"]), (old, old))
    assert statuses(msname, []) == ["stored"]
    left = sorted(
        n for n in os.listdir(msname) if n.startswith(partition_cache.TMP_PREFIX)
    )
    assert left == sorted([names["alive"], names["new elsewhere"]])


@pytest.mark.parametrize("mode", ["read", "off"])
def test_read_and_off_change_no_file(mode, ms_copy, monkeypatch):
    msname = ms_copy("rich")
    before = file_digests(msname)
    monkeypatch.setenv("XRADIO_MSV2_PARTITION_CACHE", mode)
    for scheme in ([], ["FIELD_ID"]):
        xr.open_datatree(msname, engine=ENGINE, partition_scheme=scheme)
    assert file_digests(msname) == before


def test_environment_variable_sets_the_mode(ms_copy, monkeypatch):
    msname = ms_copy("dense")
    monkeypatch.setenv("XRADIO_MSV2_PARTITION_CACHE", "rebuild")
    xr.open_datatree(msname, engine=ENGINE)
    assert len(stored_rows(msname)) == 1
    monkeypatch.setenv("XRADIO_MSV2_PARTITION_CACHE", "auto")
    partition_cache.clear_partition_memo()
    xr.open_datatree(msname, engine=ENGINE)
    assert PARTITIONS_MEMO.stats["stored hits"] == 1


def test_not_stored_while_this_process_write_locks_main(ms_copy):
    """A MAIN whose write lock this process holds (another thread writing
    it): its changes may not be flushed, so nothing is stored (INFO)."""
    msname = ms_copy("dense")
    writer = tables.table(msname, readonly=False, lockoptions="user", ack=False)
    try:
        writer.lock(True)
        assert partition_cache.why_not_writable(msname) == (
            partition_cache.MAIN_WRITE_LOCKED,
            False,
        )
    finally:
        writer.close()
    assert partition_cache.why_not_writable(msname) is None


def _notices(caplog, recwarn):
    infos = [
        r.getMessage() for r in caplog.records if "is not stored" in r.getMessage()
    ]
    warned = [str(w.message) for w in recwarn if w.category is PartitionCacheWarning]
    return infos, warned


@pytest.mark.parametrize(
    "setup, reason, warn",
    [
        ("casatools", "casatools only", False),
        ("runs", "too many runs", False),
        ("data manager", "MAIN uses StandardStMan", False),
        ("no history", "HISTORY missing", True),
        ("history columns", "HISTORY has no APP_PARAMS column", True),
    ],
)
def test_not_stored_notices(setup, reason, warn, ms_copy, monkeypatch, caplog, recwarn):
    """Why partitions are not stored: a PartitionCacheWarning (or an INFO
    log for reasons that are no user error), once per MS and reason."""
    msname = ms_copy("dense")
    if setup == "casatools":
        monkeypatch.setattr(partition_cache, "uses_casatools", lambda: True)
    elif setup == "runs":
        monkeypatch.setattr(partition_cache, "MAX_STORED_RUNS", 3)
    elif setup == "data manager":
        monkeypatch.setattr(
            partition_cache,
            "WRITABLE_DATA_MANAGERS",
            partition_cache.WRITABLE_DATA_MANAGERS - {"StandardStMan"},
        )
    elif setup == "no history":
        shutil.rmtree(os.path.join(msname, "HISTORY"))
    else:
        with tables.table(
            os.path.join(msname, "HISTORY"), readonly=False, ack=False
        ) as h:
            h.removecols(["APP_PARAMS"])
    before = file_digests(msname)
    monkeypatch.setattr(partition_cache, "xradio_logger", lambda: _ListLogger(caplog))
    for _ in range(3):
        assert statuses(msname, []) == [f"memory:{reason}"]
    assert file_digests(msname) == before
    infos, warned = _notices(caplog, recwarn)
    expected = [f"({reason})" in m for m in (warned if warn else infos)]
    assert expected == [True]
    assert (infos if warn else warned) == []


def test_reference_table_is_reported_at_info(
    ms_copy, tmp_path, monkeypatch, caplog, recwarn
):
    """A reference table (no lock file of its own, as concatenated tables):
    its partitions are not stored because it is one (an INFO log), not
    because of its lock file (a PartitionCacheWarning naming the wrong
    cause); no file of it changes."""
    msname = ms_copy("dense")
    ref = str(tmp_path / "ref.ms")
    with tables.table(msname, ack=False) as main_tb:
        main_tb.query("ANTENNA1 >= 0", name=ref).close()
    for name in os.listdir(msname):
        if os.path.isfile(os.path.join(msname, name, "table.dat")):
            shutil.copytree(os.path.join(msname, name), os.path.join(ref, name))
    assert not os.path.exists(os.path.join(ref, "table.lock"))
    before = file_digests(ref)
    monkeypatch.setattr(partition_cache, "xradio_logger", lambda: _ListLogger(caplog))
    reason = "MAIN is a reference or concatenated table"
    for _ in range(2):
        assert statuses(ref, []) == [f"memory:{reason}"]
    infos, warned = _notices(caplog, recwarn)
    assert [f"({reason})" in m for m in infos] == [True]
    assert warned == []
    assert file_digests(ref) == before


class _ListLogger:
    """A logger that records INFO messages in caplog.records."""

    def __init__(self, caplog):
        self.caplog = caplog

    def info(self, message):
        import logging

        self.caplog.records.append(
            logging.LogRecord("test", logging.INFO, __file__, 0, message, None, None)
        )

    def debug(self, message):
        pass

    error = warning = debug


def test_concurrent_threads_store_once(ms_copy):
    """4 threads of one process opening one MS (1 and then 3 schemes): one
    row per scheme, one HISTORY row per stored content (the writer mutex)."""
    msname = ms_copy("rich")
    n_history = len(history_rows(msname))
    schemes = [
        [],
        ["FIELD_ID"],
        ["FIELD_ID", "SCAN_NUMBER"],
        ["SCAN_NUMBER", "FIELD_ID"],
    ]
    results, errors = [], []

    def run(scheme):
        try:
            results.append(load(msname, scheme, "auto").status)
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    for round_schemes in ([[]] * 4, schemes):
        partition_cache.clear_partition_memo()
        threads = [threading.Thread(target=run, args=(s,)) for s in round_schemes]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    assert errors == []
    assert len(results) == 8
    keys = sorted(row["SCHEME_KEY"] for row in stored_rows(msname))
    assert keys == sorted(["[]", '["FIELD_ID"]', '["FIELD_ID", "SCAN_NUMBER"]'])
    assert len(history_rows(msname)) == n_history + 3
    assert results.count("stored") == 3


def test_concurrent_threads_without_the_mutex_would_both_lock(ms_copy):
    """casacore's write locks belong to the process: a second handle of this
    process 'holds' the lock the first one took (why the writers need the
    in-process mutex)."""
    msname = ms_copy("dense")
    first = tables.table(
        msname, readonly=False, lockoptions={"option": "usernoread"}, ack=False
    )
    second = tables.table(
        msname, readonly=False, lockoptions={"option": "usernoread"}, ack=False
    )
    try:
        first.lock(write=True, nattempts=1)
        second.lock(write=True, nattempts=1)
        assert first.haslock(write=True) and second.haslock(write=True)
    finally:
        first.unlock()
        first.close()
        second.close()


def test_closing_a_handle_releases_the_lock_of_another(ms_copy):
    """casacore shares one table object per table in a process: closing any
    handle (python-casacore's close unlocks) releases the write lock another
    handle took (why the cache's reads and writes of an MS hold its mutex)."""
    msname = ms_copy("dense")
    subtable = os.path.join(msname, SUBTABLE_NAME)
    store_test_row(msname, [])
    writer = tables.table(
        subtable, readonly=False, lockoptions={"option": "usernoread"}, ack=False
    )
    try:
        writer.lock(write=True, nattempts=1)
        assert writer.haslock(write=True)
        reader = tables.table(
            subtable, readonly=True, lockoptions={"option": "usernoread"}, ack=False
        )
        reader.close()
        assert not writer.haslock(write=True)
        with pytest.raises(RuntimeError, match="should be locked"):
            writer.putcell("TIME", 0, 1.0)
    finally:
        writer.close()


def _close_a_main_handle(msname: str) -> None:
    """What another thread of this process does when it closes its handle
    of MAIN (a lazy read, a partition build): python-casacore's close
    unlocks the table object the process shares."""
    tables.table(
        msname, readonly=True, lockoptions={"option": "usernoread"}, ack=False
    ).close()


@pytest.mark.parametrize(
    "when", ["while the sub-table is made", "before the keyword", "before the flush"]
)
def test_main_lock_released_by_another_thread_of_this_process(
    when, ms_copy, monkeypatch
):
    """Another thread of this process closes a MAIN handle while the first
    store writes the MAIN keyword (releasing the MAIN write lock): the lock is
    taken again and the partitions are stored."""
    msname = ms_copy("dense")
    if when == "while the sub-table is made":
        create = partition_cache._create_subtable

        def create_then_close(path):
            created = create(path)
            _close_a_main_handle(msname)
            return created

        monkeypatch.setattr(partition_cache, "_create_subtable", create_then_close)
    else:
        put = partition_cache._put_main_keyword
        calls = []

        def put_with_a_close(main_tb, path):
            calls.append(path)
            if when == "before the keyword" and len(calls) == 1:
                _close_a_main_handle(msname)
            written = put(main_tb, path)
            if when == "before the flush":
                _close_a_main_handle(msname)
            return written

        monkeypatch.setattr(partition_cache, "_put_main_keyword", put_with_a_close)
    with warnings.catch_warnings():
        warnings.simplefilter("error", PartitionCacheWarning)
        assert statuses(msname, []) == ["stored"]
    assert partition_cache.link_state(msname) == "linked"
    assert statuses(msname, [], ("auto", "read")) == ["hit", "hit"]


def test_main_lock_released_again_and_again(ms_copy, monkeypatch):
    """The MAIN lock released before every attempt of the keyword write: the
    write fails after KEYWORD_WRITE_ATTEMPTS (partitions in memory, a
    warning), and no sub-table is left without its keyword."""
    msname = ms_copy("dense")
    put = partition_cache._put_main_keyword
    calls = []

    def put_after_a_close(main_tb, path):
        calls.append(path)
        _close_a_main_handle(msname)
        return put(main_tb, path)

    monkeypatch.setattr(partition_cache, "_put_main_keyword", put_after_a_close)
    with pytest.warns(PartitionCacheWarning, match="should be locked"):
        assert statuses(msname, []) == ["memory:write failed"]
    assert len(calls) == partition_cache.KEYWORD_WRITE_ATTEMPTS
    assert partition_cache.link_state(msname) == "absent"
    assert SUBTABLE_NAME not in os.listdir(msname)


def test_threads_reading_and_writing_one_ms(ms_copy):
    """Threads storing and reading the partitions of one MS (several schemes
    at once, again and again): no write fails."""
    msname = ms_copy("rich")
    schemes = [[], ["FIELD_ID"], ["SCAN_NUMBER"], ["STATE_ID"]]
    results, errors = [], []

    def run(scheme):
        try:
            for mode in ("auto", "rebuild", "auto", "read"):
                results.append(load(msname, scheme, mode).status)
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    with warnings.catch_warnings():
        warnings.simplefilter("error", PartitionCacheWarning)
        threads = [threading.Thread(target=run, args=(s,)) for s in schemes * 2]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    assert errors == []
    assert not any(status.startswith("memory:write") for status in results)
    assert len(stored_rows(msname)) == 4


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs os.fork")
@pytest.mark.filterwarnings("ignore:This process .* is multi-threaded")
def test_write_mutexes_are_reset_in_a_fork_child(tmp_path):
    """A child forked while another thread holds a writer mutex (and the
    lock of their dict) gets free ones."""
    mutex = partition_cache.write_mutex(str(tmp_path))
    with mutex, partition_cache._WRITE_MUTEXES_LOCK:
        pid = os.fork()
        if pid == 0:  # child
            ok = partition_cache._WRITE_MUTEXES_LOCK.acquire(timeout=5)
            if ok:
                partition_cache._WRITE_MUTEXES_LOCK.release()
                ok = partition_cache.write_mutex(str(tmp_path)).acquire(timeout=5)
            os._exit(0 if ok else 1)
    assert wait_child(pid) == 0


def wait_child(pid: int, seconds: float = 60.0) -> int | None:
    """The exit code of a forked child; a child still running after
    ``seconds`` (e.g. deadlocked) is killed: None."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        done, status = os.waitpid(pid, os.WNOHANG)
        if done:
            return os.waitstatus_to_exitcode(status)
        time.sleep(0.05)
    os.kill(pid, 9)
    os.waitpid(pid, 0)
    return None
