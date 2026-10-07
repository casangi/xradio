import shutil

import numpy as np
import pandas as pd
import pytest
from casacore import tables

from xradio.measurement_set._utils._msv2._tables.read_rows import runs_to_rows
from xradio.measurement_set._utils._msv2.conversion import create_taql_query_where
from xradio.measurement_set._utils._msv2.partition_queries import (
    MANDATORY_PARTITION_KEYS,
    PARTITION_ALGORITHM_VERSION,
    PARTITION_MAIN_KEY_COLUMNS,
    MainRowRuns,
    PartitionMainRows,
    _add_derived_columns,
    _factorize_rows,
    canonical_scheme_key,
    create_partitions,
    create_partitions_with_main_rows,
    describe_partition_rows,
    partition_key_maps,
    partition_main_rows,
    partition_selection_digest,
    select_main_rows,
    validate_partition_scheme,
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


# --- The partitioning algorithm: tripwire digests ---------------------------------


@pytest.fixture(scope="module")
def ms_key_variants(ms_edge_rows, tmp_path_factory):
    """
    Copies of ms_edge_rows whose sub-tables take the other paths of the
    partition keys derived from FIELD, SOURCE and STATE: no STATE table, an
    empty SOURCE table (no SOURCE_ID), and a scalar FIELD EPHEMERIS_ID that
    differs between the two fields.
    """
    base = tmp_path_factory.mktemp("partition_key_variants")
    paths = {}
    for variant in ("state_absent", "source_empty", "ephemeris_ids"):
        path = str(base / f"{variant}.ms")
        shutil.copytree(ms_edge_rows, path)
        if variant == "state_absent":
            shutil.rmtree(path + "/STATE")
            with tables.table(path, readonly=False, ack=False) as main_tb:
                main_tb.removekeyword("STATE")
        elif variant == "source_empty":
            # (its storage managers cannot remove rows: an empty table instead)
            with tables.table(path + "/SOURCE", ack=False) as tb:
                desc = tb.getdesc()
            shutil.rmtree(path + "/SOURCE")
            tables.table(path + "/SOURCE", desc, nrow=0, ack=False).close()
        else:
            with tables.table(path + "/FIELD", readonly=False, ack=False) as tb:
                tb.removecols("EPHEMERIS_ID")
                tb.addcols(tables.makescacoldesc("EPHEMERIS_ID", 0))
                tb.putcol("EPHEMERIS_ID", np.array([0, 3], dtype=np.int32))
        paths[variant] = path
    yield paths
    shutil.rmtree(base, ignore_errors=True)


def _partitions_fixture_path(name: str, request) -> str:
    """MS path of a fixture name of PARTITION_FIXTURES (variants: "<fixture>:<variant>")."""
    fixture, _, variant = name.partition(":")
    ms = request.getfixturevalue(fixture)
    if variant:
        return ms[variant]
    return ms if isinstance(ms, str) else ms.fname


def _partitions_digest(ms: str, scheme: list) -> str:
    """sha256 (16 hex digits) of the complete create_partitions_with_main_rows
    output: the descriptions (key and value order included) and the runs."""
    import hashlib
    import json

    parts, runs = create_partitions_with_main_rows(ms, scheme)
    digest = hashlib.sha256(json.dumps(parts).encode())
    for array in (runs.starts, runs.lengths, runs.bounds):
        digest.update(np.ascontiguousarray(array, dtype="<i8").tobytes())
    digest.update(runs.digests.tobytes())
    digest.update(str(runs.main_nrows).encode())
    return digest.hexdigest()[:16]


PARTITION_FIXTURES = [
    "ms_minimal_required",  # SOURCE, STATE, an EPHEMERIS_ID array column
    "ms_minimal_misbehaved",  # STATE_ID=-1 with an empty STATE, no EPHEMERIS_ID
    "ms_minimal_without_opt",  # no SOURCE table
    "ms_edge_rows",  # STATE_ID=-1 with a STATE, autocorrelations, interleaving
    "ms_key_variants:state_absent",
    "ms_key_variants:source_empty",
    "ms_key_variants:ephemeris_ids",
]
# create_partitions_with_main_rows digests per fixture, for every scheme of
# SCHEMES (in that order), pinned before the partition_queries refactor.
PARTITION_DIGESTS: dict[str, list[str]] = {
    "ms_minimal_required": [
        "712ed71157a4b606",
        "712ed71157a4b606",
        "76e19b9a519ca2a7",
        "712ed71157a4b606",
        "712ed71157a4b606",
        "76e19b9a519ca2a7",
    ],
    "ms_minimal_misbehaved": [
        "65c9f85c07a8f5fa",
        "65c9f85c07a8f5fa",
        "78658c8725f7a98b",
        "65c9f85c07a8f5fa",
        "65c9f85c07a8f5fa",
        "78658c8725f7a98b",
    ],
    "ms_minimal_without_opt": [
        "ffc9d6fd5d84f105",
        "ffc9d6fd5d84f105",
        "ae3f3ad86c455133",
        "ffc9d6fd5d84f105",
        "ffc9d6fd5d84f105",
        "ae3f3ad86c455133",
    ],
    "ms_edge_rows": [
        "bc1ec96e565aefb2",
        "a412684e0516742d",
        "73cc5684604fd50f",
        "ae029f573fd937ff",
        "d676d9ad253632b1",
        "69386ccd78c8ab70",
    ],
    "ms_key_variants:state_absent": [
        "1714767882bfa26b",
        "3843df7a59868a60",
        "02c5a7960bf16fcc",
        "8a2bb48e75715163",
        "120c9ae1c4b98c05",
        "7022538d3a113bd8",
    ],
    "ms_key_variants:source_empty": [
        "d22838212d32f757",
        "a0a387abb025b5ec",
        "ddf3b99e7ebfc2e9",
        "f3f953b1ddc35ee1",
        "0cc9049e13b0e6e4",
        "45854d5134bc2ebb",
    ],
    "ms_key_variants:ephemeris_ids": [
        "804acca491074d9a",
        "804acca491074d9a",
        "e67fa0cbb84bfe62",
        "e022a07773844739",
        "b98b850969da2f86",
        "74e3d63755248c8c",
    ],
}


@pytest.mark.parametrize("fixture", PARTITION_FIXTURES)
def test_partitioning_algorithm_tripwire(fixture, request):
    """
    The output of create_partitions_with_main_rows is pinned: partitions
    stored in an MS (the XRADIO_PARTITIONS cache of the MSv2 backend) are
    only valid for the algorithm that computed them.
    """
    ms = _partitions_fixture_path(fixture, request)
    digests = [_partitions_digest(ms, scheme) for scheme in SCHEMES]
    assert digests == PARTITION_DIGESTS.get(fixture), (
        "The partitions computed by create_partitions_with_main_rows changed: bump "
        "PARTITION_ALGORITHM_VERSION (partition_queries.py) and update "
        f"PARTITION_DIGESTS[{fixture!r}] to {digests}"
    )
    assert isinstance(PARTITION_ALGORITHM_VERSION, int)
    assert PARTITION_ALGORITHM_VERSION >= 1


@pytest.mark.parametrize(
    "scheme, expected",
    [
        (None, []),
        ([], []),
        (("FIELD_ID",), ["FIELD_ID"]),
        (["SCAN_NUMBER", "FIELD_ID", "SCAN_NUMBER"], ["SCAN_NUMBER", "FIELD_ID"]),
        (["DATA_DESC_ID", "OBS_MODE", "OBSERVATION_ID", "EPHEMERIS_ID"], []),
        (["ANTENNA1", "DATA_DESC_ID", "SOURCE_ID"], ["ANTENNA1", "SOURCE_ID"]),
        (iter(["STATE_ID", "SUB_SCAN_NUMBER"]), ["STATE_ID", "SUB_SCAN_NUMBER"]),
    ],
)
def test_validate_partition_scheme(scheme, expected):
    assert validate_partition_scheme(scheme) == expected


@pytest.mark.parametrize(
    "scheme, error, match",
    [
        ("FIELD_ID", TypeError, "not the string 'FIELD_ID'"),
        (b"FIELD_ID", TypeError, "not the string"),
        (3, TypeError, "list of keys"),
        (["FIELD_ID", 3], TypeError, "must be strings"),
        (
            ["FIELD"],
            ValueError,
            "Unknown partition_scheme key 'FIELD'.*SUB_SCAN_NUMBER",
        ),
        (["field_id"], ValueError, "Unknown partition_scheme key"),
    ],
)
def test_validate_partition_scheme_errors(scheme, error, match):
    with pytest.raises(error, match=match):
        validate_partition_scheme(scheme)


def test_canonical_scheme_key(ms_edge_rows):
    """The key ignores order, duplicates and the mandatory keys: those schemes
    give the same partitions."""
    assert canonical_scheme_key(None) == canonical_scheme_key([]) == "[]"
    key = canonical_scheme_key(["SCAN_NUMBER", "FIELD_ID"])
    assert key == '["FIELD_ID", "SCAN_NUMBER"]'
    same = [
        ["FIELD_ID", "SCAN_NUMBER"],
        ["SCAN_NUMBER", "DATA_DESC_ID", "FIELD_ID", "SCAN_NUMBER"],
    ]
    assert all(canonical_scheme_key(scheme) == key for scheme in same)
    assert canonical_scheme_key(["FIELD_ID"]) != key
    reference = _partitions_digest(ms_edge_rows, ["FIELD_ID", "SCAN_NUMBER"])
    for scheme in same:
        assert _partitions_digest(ms_edge_rows, scheme) == reference


def _main_key_columns(ms: str) -> dict[str, np.ndarray]:
    with tables.table(ms, readonly=True, ack=False) as main_tb:
        names = list(PARTITION_MAIN_KEY_COLUMNS) + ["ANTENNA2"]
        return {name: main_tb.getcol(name) for name in names}


@pytest.mark.parametrize("scheme", SCHEMES)
@pytest.mark.parametrize("fixture", PARTITION_FIXTURES)
def test_describe_partition_rows_matches_create_partitions(fixture, scheme, request):
    """
    describe_partition_rows of a partition's rows is the description of
    create_partitions: the rows of its runs (schemes without ANTENNA1, whose
    runs hold every row of the partition's key values) and, for every scheme,
    the rows of its key values (with ANTENNA1 the runs keep the
    autocorrelations only: their description can have fewer values).
    """
    ms = _partitions_fixture_path(fixture, request)
    parts, runs = create_partitions_with_main_rows(ms, scheme)
    maps = partition_key_maps(ms)
    columns = _main_key_columns(ms)
    frame = pd.DataFrame({name: columns[name] for name in PARTITION_MAIN_KEY_COLUMNS})
    _add_derived_columns(frame, maps)
    group_keys = [
        key
        for key in list(MANDATORY_PARTITION_KEYS) + list(scheme)
        if key in frame.columns
    ]
    covered = np.zeros(len(frame), dtype=bool)
    for idx, part in enumerate(parts):
        in_group = np.ones(len(frame), dtype=bool)
        for key in group_keys:
            assert len(part[key]) == 1, key  # a grouping key is single-valued
            in_group &= frame[key].to_numpy() == part[key][0]
        assert not (covered & in_group).any()
        covered |= in_group
        rows = np.flatnonzero(in_group)
        group_columns = {name: col[rows] for name, col in columns.items()}
        assert describe_partition_rows(group_columns, maps, scheme) == part
        run_rows = runs[idx].rows()
        if "ANTENNA1" not in scheme:
            np.testing.assert_array_equal(run_rows, rows)
        else:
            antenna = columns["ANTENNA1"][run_rows]
            assert np.all(antenna == columns["ANTENNA2"][run_rows])
            assert np.all(antenna == part["ANTENNA1"][0])
    assert covered.all()


def _open_files_under(path: str) -> list[str]:
    """Files under path that this process holds open (Linux /proc)."""
    import os

    prefix = os.path.realpath(path) + os.sep
    found = []
    for fd in os.listdir("/proc/self/fd"):
        try:
            target = os.readlink(f"/proc/self/fd/{fd}")
        except OSError:
            continue
        if target.startswith(prefix):
            found.append(target)
    return found


@pytest.mark.skipif(
    not __import__("os").path.isdir("/proc/self/fd"), reason="needs /proc/self/fd"
)
@pytest.mark.parametrize("fail_at", ["MAIN", "SOURCE_ID", "SUB_SCAN"])
def test_create_partitions_closes_tables_on_errors(
    ms_edge_rows, tmp_path, monkeypatch, fail_at
):
    """MAIN, FIELD, SOURCE and STATE are closed when the partitioning fails
    (also while the traceback, which holds the frames, is alive)."""
    from xradio.measurement_set._utils._msv2 import partition_queries

    ms = str(tmp_path / "copy.ms")
    shutil.copytree(ms_edge_rows, ms)

    def failing_factorize(columns):
        raise RuntimeError("simulated failure")

    getcol = tables.table.getcol

    def failing_getcol(self, columnname, *args, **kwargs):
        if columnname == fail_at:
            raise RuntimeError("simulated failure")
        return getcol(self, columnname, *args, **kwargs)

    if fail_at == "MAIN":
        monkeypatch.setattr(partition_queries, "_factorize_rows", failing_factorize)
    else:
        monkeypatch.setattr(tables.table, "getcol", failing_getcol)
    with pytest.raises(RuntimeError, match="simulated failure") as raised:
        create_partitions_with_main_rows(ms, ["FIELD_ID"])
    assert raised.traceback  # kept alive while checking
    assert _open_files_under(ms) == []
    monkeypatch.undo()
    assert create_partitions_with_main_rows(ms, ["FIELD_ID"])[0]
    assert _open_files_under(ms) == []


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
