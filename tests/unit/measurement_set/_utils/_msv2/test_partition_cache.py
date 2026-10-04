"""
Tests of the partition cache of the MSv2 xarray backend (partition_cache.py):
the XRADIO_PARTITIONS layout, the staleness rules (fingerprint, HISTORY,
row checks), reading stored rows, the memo and the modes. The stored rows
of these tests are made by a test-only writer (store_test_row), on copies
of the generated MSs.
"""

import copy
import json
import os
import time

import numpy as np
import pytest
import xarray as xr
from casacore import tables

from _xradio_xarray_backends import MSv2BackendEntrypoint
from xradio.measurement_set._utils._msv2 import partition_cache
from xradio.measurement_set._utils._msv2._tables.table_lock_file import history_nrows
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
    "mode, expected",
    [("off", "memory:mode-off"), ("rebuild", "memory:computed")],
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
