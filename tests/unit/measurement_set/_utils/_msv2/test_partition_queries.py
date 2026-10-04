import shutil

import numpy as np
import pandas as pd
import pytest
from casacore import tables

from xradio.measurement_set._utils._msv2._tables.read_rows import runs_to_rows
from xradio.measurement_set._utils._msv2.conversion import create_taql_query_where
from xradio.measurement_set._utils._msv2.partition_queries import (
    MainRowRuns,
    PartitionMainRows,
    _factorize_rows,
    create_partitions,
    create_partitions_with_main_rows,
    partition_main_rows,
    partition_selection_digest,
    select_main_rows,
)

expected_partition_axes = [
    "DATA_DESC_ID",
    "OBSERVATION_ID",
    "FIELD_ID",
    "SCAN_NUMBER",
    "STATE_ID",
    "SOURCE_ID",
    "OBS_MODE",
    "SUB_SCAN_NUMBER",
    "EPHEMERIS_ID",
]


def check_expected_min_partitions(partitions: list[dict], expected_len: int = 4):
    """Checks consistency of the basic partitions of the
    ms_minimal_required test MS (1 per DDI)"""

    assert isinstance(partitions, list)
    assert len(partitions) == expected_len

    assert all(
        [axis in part for axis in expected_partition_axes for part in partitions]
    )

    assert all(
        isinstance(part[axis], list) and len(part[axis]) == 1
        for axis in expected_partition_axes
        for part in partitions
    )
    ddis = {part["DATA_DESC_ID"][0] for part in partitions}
    assert ddis == {0, 1, 2, 3}


def test_create_partitions_ms_empty(ms_empty_required):
    parts = create_partitions(ms_empty_required.fname, [])
    assert parts == []


def test_create_partitions_ms_min(ms_minimal_required):
    parts = create_partitions(ms_minimal_required.fname, [])
    assert isinstance(parts, list)
    assert len(parts) == 4
    assert all([axis in part for axis in expected_partition_axes for part in parts])


def test_create_partitions_ms_min_with_field(ms_minimal_required):
    parts = create_partitions(ms_minimal_required.fname, ["FIELD_ID"])
    check_expected_min_partitions(parts, 4)


def test_create_partitions_ms_min_with_antenna1(ms_minimal_required):
    parts = create_partitions(ms_minimal_required.fname, ["ANTENNA1"])
    check_expected_min_partitions(parts, 16)


def test_create_partitions_ms_min_with_state_id(ms_minimal_required):
    parts = create_partitions(ms_minimal_required.fname, ["STATE_ID"])
    check_expected_min_partitions(parts, 4)


def test_create_partitions_ms_min_with_all(ms_minimal_required):
    parts = create_partitions(
        ms_minimal_required.fname,
        [
            "FIELD_ID",
            "SCAN_NUMBER",
            "STATE_ID",
            "SOURCE_ID",
            "SUB_SCAN_NUMBER",
            "ANTENNA1",
        ],
    )
    check_expected_min_partitions(parts, 16)


def test_create_partitions_ms_min_with_other(ms_minimal_required):
    parts = create_partitions(
        ms_minimal_required.fname,
        ["FIELD_ID", "SOURCE_ID", "DATA_DESC_ID"],
    )

    check_expected_min_partitions(parts, 4)


# --- MAIN row membership of the partitions (row runs) ---------------------------

SCHEMES = [
    [],
    ["FIELD_ID"],
    ["ANTENNA1"],
    ["STATE_ID"],
    ["SCAN_NUMBER", "SOURCE_ID"],
    ["FIELD_ID", "SCAN_NUMBER", "STATE_ID", "SOURCE_ID", "SUB_SCAN_NUMBER", "ANTENNA1"],
]


@pytest.fixture(scope="module")
def ms_edge_rows(ms_minimal_required, tmp_path_factory):
    """
    Copy of the minimal MS with MAIN key columns rewritten to exercise the
    partition row membership: STATE_ID=-1 rows (with a non-empty STATE table),
    autocorrelation rows (ANTENNA1 partitions), interleaved fields, scans and
    states.
    """
    path = str(tmp_path_factory.mktemp("partition_rows") / "edge_rows.ms")
    shutil.copytree(ms_minimal_required.fname, path)
    with tables.table(path, readonly=False, ack=False) as main_tb:
        nrows = main_tb.nrows()
        rows = np.arange(nrows)
        state = (rows // 7) % 2
        state[rows % 5 == 0] = -1
        main_tb.putcol("STATE_ID", state.astype(np.int32))
        main_tb.putcol("FIELD_ID", ((rows // 13) % 2).astype(np.int32))
        main_tb.putcol("SCAN_NUMBER", (1 + (rows // 90) % 3).astype(np.int32))
        ant1 = main_tb.getcol("ANTENNA1")
        ant2 = main_tb.getcol("ANTENNA2")
        ant2[rows % 4 == 0] = ant1[rows % 4 == 0]
        main_tb.putcol("ANTENNA2", ant2)
    yield path
    shutil.rmtree(path, ignore_errors=True)


def taql_rows(ms: str, partition: dict) -> np.ndarray:
    """Rows of the partition's TaQL selection (create_taql_query_where)."""
    with tables.table(ms, readonly=True, ack=False) as main_tb:
        query = tables.taql(
            f"select * from $1 {create_taql_query_where(partition)}", tables=[main_tb]
        )
        try:
            return np.asarray(query.rownumbers(), dtype=np.int64)
        finally:
            query.close()


@pytest.mark.parametrize("scheme", SCHEMES)
@pytest.mark.parametrize(
    "ms_fixture", ["ms_minimal_required", "ms_minimal_misbehaved", "ms_edge_rows"]
)
def test_create_partitions_row_runs_match_taql(scheme, ms_fixture, request):
    import json

    ms = request.getfixturevalue(ms_fixture)
    ms = ms if isinstance(ms, str) else ms.fname
    parts, runs = create_partitions_with_main_rows(ms, scheme)
    assert parts
    # the descriptions are those of create_partitions: plain lists, no arrays
    assert parts == create_partitions(ms, scheme)
    json.dumps(parts)
    assert isinstance(runs, MainRowRuns) and len(runs) == len(parts)
    all_rows = []
    with tables.table(ms, readonly=True, ack=False) as main_tb:
        nrows = main_tb.nrows()
        assert runs.main_nrows == nrows
        for idx, part in enumerate(parts):
            part_runs = runs[idx]
            assert isinstance(part_runs, PartitionMainRows)
            starts, lengths = part_runs.starts, part_runs.lengths
            assert starts.dtype == np.int64 and lengths.dtype == np.int64
            assert np.all(lengths > 0)
            assert np.all(starts[1:] > starts[:-1] + lengths[:-1])
            rows = runs_to_rows(starts, lengths)
            np.testing.assert_array_equal(rows, taql_rows(ms, part))
            np.testing.assert_array_equal(part_runs.rows(), rows)
            assert part_runs.matches(part, nrows)
            np.testing.assert_array_equal(
                partition_main_rows(main_tb, part, part_runs), rows
            )
            # without the runs: numpy twin of the TaQL selection
            np.testing.assert_array_equal(select_main_rows(main_tb, part), rows)
            np.testing.assert_array_equal(partition_main_rows(main_tb, part), rows)
            all_rows.append(rows)
    all_rows = np.concatenate(all_rows)
    assert np.unique(all_rows).size == all_rows.size  # disjoint
    if "ANTENNA1" not in scheme:
        assert all_rows.size == nrows
    else:
        # autocorrelations only (create_taql_query_where's ANTENNA2 rule)
        with tables.table(ms, readonly=True, ack=False) as main_tb:
            ant1, ant2 = main_tb.getcol("ANTENNA1"), main_tb.getcol("ANTENNA2")
        np.testing.assert_array_equal(np.sort(all_rows), np.flatnonzero(ant1 == ant2))


def test_create_partitions_state_id_minus_one_grouped_with_last_state(ms_edge_rows):
    with tables.table(ms_edge_rows + "/STATE", ack=False) as state_tb:
        last_mode = state_tb.getcol("OBS_MODE")[-1]
    parts = create_partitions(ms_edge_rows, [])
    with_minus_one = [p for p in parts if -1 in p["STATE_ID"]]
    assert with_minus_one
    assert all(p["OBS_MODE"] == [last_mode] for p in with_minus_one)


def test_partition_main_rows_of_a_changed_description(ms_edge_rows):
    """
    Runs computed for one description are not used for a changed one (e.g.
    a subset of its FIELD_IDs, or merged partitions): the rows follow the
    description, as the TaQL selection does.
    """
    parts, runs = create_partitions_with_main_rows(ms_edge_rows, [])
    idx = next(i for i, p in enumerate(parts) if len(p["FIELD_ID"]) > 1)
    part = parts[idx]
    changed = [
        dict(part, FIELD_ID=part["FIELD_ID"][:1]),  # subset
        dict(part, FIELD_ID=part["FIELD_ID"][::-1] * 2),  # same rows
        dict(part, STATE_ID=sorted(set(part["STATE_ID"]) | {0, 1})),  # superset
        dict(part, SCAN_NUMBER=[None]),  # key no longer selects
        dict(part, ANTENNA1=[0]),  # key added
    ]
    with tables.table(ms_edge_rows, readonly=True, ack=False) as main_tb:
        for info in changed:
            rows = partition_main_rows(main_tb, info, runs[idx])
            np.testing.assert_array_equal(rows, taql_rows(ms_edge_rows, info))
        # the same selection keeps the runs
        assert runs[idx].matches(changed[1], main_tb.nrows())
        assert not runs[idx].matches(changed[0], main_tb.nrows())
        # another MAIN table (size): the rows are selected again
        assert not runs[idx].matches(part, main_tb.nrows() + 1)


def test_partition_selection_digest():
    base = {"DATA_DESC_ID": [0], "FIELD_ID": [3, 1], "STATE_ID": [None]}
    digest = partition_selection_digest(base)
    assert isinstance(digest, bytes) and len(digest) == 16
    same = [
        {"DATA_DESC_ID": [0], "FIELD_ID": [1, 3, 3], "STATE_ID": [None]},
        {"DATA_DESC_ID": np.array([0]), "FIELD_ID": [np.int32(1), 3]},
        dict(base, OBS_MODE=["x"], SOURCE_ID=[7]),  # not selection keys
    ]
    for info in same:
        assert partition_selection_digest(info) == digest
    different = [
        {"DATA_DESC_ID": [0], "FIELD_ID": [3]},
        {"DATA_DESC_ID": [0], "FIELD_ID": [3, 1], "STATE_ID": [5]},
        {"DATA_DESC_ID": [0], "SCAN_NUMBER": [3, 1]},  # same values, other key
        {"DATA_DESC_ID": [0], "FIELD_ID": [3, 1], "ANTENNA1": [0]},
    ]
    for info in different:
        assert partition_selection_digest(info) != digest
    # values that are not integers: no digest (never matches row runs)
    assert partition_selection_digest({"FIELD_ID": [1.5]}) is None


def test_main_row_runs_subset_and_pickle(ms_minimal_required):
    import pickle

    parts, runs = create_partitions_with_main_rows(ms_minimal_required.fname, [])
    assert len(runs) == 4 and runs[-1].digest == runs[3].digest
    with pytest.raises(IndexError):
        runs[4]
    sub = runs.subset([2, 0])
    assert len(sub) == 2
    for sub_idx, idx in enumerate([2, 0]):
        np.testing.assert_array_equal(sub[sub_idx].rows(), runs[idx].rows())
        assert sub[sub_idx].digest == runs[idx].digest
    # a pickled partition carries only its own runs
    data = pickle.dumps(runs[1])
    copy = pickle.loads(data)
    np.testing.assert_array_equal(copy.rows(), runs[1].rows())
    assert copy.matches(parts[1], runs.main_nrows)
    assert len(copy._runs) == 1
    assert len(data) < len(pickle.dumps(runs)) or runs.starts.size <= 4
    assert runs.nbytes == (
        runs.starts.nbytes + runs.lengths.nbytes + runs.bounds.nbytes + 16 * 4
    )


def test_select_main_rows_casatools_like_tables(ms_edge_rows, casatools_like_tables):
    """With the casatools table API (no getcolnp) the partitions, their row
    runs and the numpy selection of their rows are the same."""
    expected = create_partitions_with_main_rows(ms_edge_rows, ["FIELD_ID"])
    with tables.table(ms_edge_rows, readonly=True, ack=False) as main_tb:
        assert isinstance(main_tb, casatools_like_tables)
        assert not hasattr(main_tb, "getcolnp")
        parts, runs = create_partitions_with_main_rows(ms_edge_rows, ["FIELD_ID"])
        assert parts == expected[0]
        for idx, part in enumerate(parts):
            np.testing.assert_array_equal(runs[idx].rows(), expected[1][idx].rows())
            rows = select_main_rows(main_tb, part)
            np.testing.assert_array_equal(rows, runs[idx].rows())


def test_select_main_rows_without_keys_selects_all(ms_minimal_required):
    with tables.table(ms_minimal_required.fname, readonly=True, ack=False) as main_tb:
        rows = select_main_rows(main_tb, {"OBS_MODE": ["x"], "FIELD_ID": [None]})
        np.testing.assert_array_equal(rows, np.arange(main_tb.nrows()))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_factorize_rows_matches_drop_duplicates(seed):
    rng = np.random.default_rng(seed)
    cols = [rng.integers(-1, 4, 500).astype(np.int32) for _ in range(4)]
    row_key, first_rows = _factorize_rows(cols)
    unique_rows = pd.DataFrame(
        {str(i): c for i, c in enumerate(cols)}
    ).drop_duplicates()
    np.testing.assert_array_equal(first_rows, unique_rows.index.to_numpy())
    assert row_key.dtype == np.int64
    assert row_key.max() + 1 == first_rows.size
    np.testing.assert_array_equal(row_key[first_rows], np.arange(first_rows.size))
    for col in cols:
        np.testing.assert_array_equal(col, col[first_rows][row_key])


def test_factorize_rows_empty():
    row_key, first_rows = _factorize_rows([np.array([], dtype=np.int32)] * 3)
    assert row_key.size == 0 and first_rows.size == 0


def test_create_partitions_ms_empty_has_no_row_runs(ms_empty_required):
    assert create_partitions(ms_empty_required.fname, ["FIELD_ID"]) == []
    parts, runs = create_partitions_with_main_rows(ms_empty_required.fname, [])
    assert parts == [] and len(runs) == 0
