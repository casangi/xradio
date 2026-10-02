from pathlib import Path

import numpy as np
import pandas as pd
import pyasdm
import pytest

from xradio.measurement_set._utils._asdm.create_partitions import (
    MANDATORY_PARTITION_AXES,
    OPTIONAL_PARTITION_AXES,
    PER_BDF_KEYS,
    create_partitions,
    finalize_partitions_groupby,
    validate_partition_scheme,
)

# ASDM ArrayTime ns of 2024-08-10T09:57:31.2 (unix 1723283851.2)
T0 = 5230000651200000000
DT = 10_000_000_000
ASDM_TO_UNIX_NS = 3_506_716_800 * 10**9

DEFAULT_PROCESSOR_TYPES = ["CORRELATOR", "SPECTROMETER"]
DEFAULT_SPECTRAL_TYPES = ["FULL_RESOLUTION", "BASEBAND_WIDE"]


# ---------------------------------------------------------------------------
# In-memory ASDM builder (metadata tables only). With the
# bdf_path_from_data_uid fixture, every Main row gets the BDF path
# "bdf_<Main row index>" (decoded from its dataUID), so that the order of the
# BDFs in a partition can be checked.
# ---------------------------------------------------------------------------


def _data_uid(main_row):
    return f"uid://A002/X1/X{main_row + 1:x}"


def _bdf_path_from_data_uid(row):
    main_row = int(str(row.getDataUID().getEntityId()).rsplit("/X", 1)[1], 16) - 1
    return f"bdf_{main_row}"


@pytest.fixture
def bdf_path_from_data_uid(monkeypatch):
    monkeypatch.setattr("pyasdm.MainRow.getBDFPath", _bdf_path_from_data_uid)


def _add_row(table, row_cls, xml):
    row = row_cls(table)
    row.setFromXML(xml)
    table.add(row)
    return row


def _main_xml(main_row, time_ns, scan, subscan, cd_id, field_id, eb_id=0, nant=2):
    states = " ".join(["State_0"] * nant)
    return f"""<row><time> {time_ns} </time><numAntenna> {nant} </numAntenna>
<timeSampling>INTEGRATION</timeSampling><interval> {DT} </interval>
<numIntegration> 1 </numIntegration><scanNumber> {scan} </scanNumber>
<subscanNumber> {subscan} </subscanNumber><dataSize> 100 </dataSize>
<dataUID><EntityRef entityId="{_data_uid(main_row)}" partId="X00000000" entityTypeName="Main" documentVersion="1"/></dataUID>
<configDescriptionId> ConfigDescription_{cd_id} </configDescriptionId>
<execBlockId> ExecBlock_{eb_id} </execBlockId><fieldId> Field_{field_id} </fieldId>
<stateId> 1 {nant} {states} </stateId></row>"""


def _config_description_xml(cd_id, processor_type, spectral_type, dd_ids, nant=2):
    dds = " ".join(f"DataDescription_{dd}" for dd in dd_ids)
    ants = " ".join(f"Antenna_{ant}" for ant in range(nant))
    switch_cycles = " ".join(["SwitchCycle_0"] * len(dd_ids))
    return f"""<row><numAntenna> {nant} </numAntenna><numDataDescription> {len(dd_ids)} </numDataDescription>
<numFeed> 1 </numFeed><correlationMode>CROSS_AND_AUTO</correlationMode>
<configDescriptionId> ConfigDescription_{cd_id} </configDescriptionId>
<numAtmPhaseCorrection> 1 </numAtmPhaseCorrection><atmPhaseCorrection> 1 1 AP_UNCORRECTED</atmPhaseCorrection>
<processorType>{processor_type}</processorType><spectralType>{spectral_type}</spectralType>
<antennaId> 1 {nant} {ants} </antennaId><dataDescriptionId> 1 {len(dd_ids)} {dds} </dataDescriptionId>
<feedId> 1 {nant} {" ".join(["0"] * nant)} </feedId><processorId> Processor_0 </processorId>
<switchCycleId> 1 {len(dd_ids)} {switch_cycles} </switchCycleId></row>"""


def _data_description_xml(dd_id):
    return f"""<row><dataDescriptionId> DataDescription_{dd_id} </dataDescriptionId>
<polOrHoloId> Polarization_0 </polOrHoloId><spectralWindowId> SpectralWindow_{dd_id} </spectralWindowId></row>"""


def _field_xml(field_id, name, with_source_id=True):
    # distinct directions per field: pyasdm merges Field rows with equal
    # name and directions
    source_id = f"<sourceId> {10 + field_id} </sourceId>" if with_source_id else ""
    direction = f"2 1 2 {1.1 + 0.01 * field_id} -0.02"
    return f"""<row><fieldId> Field_{field_id} </fieldId><fieldName> {name} </fieldName>
<numPoly> 1 </numPoly><delayDir> {direction} </delayDir><phaseDir> {direction} </phaseDir>
<referenceDir> {direction} </referenceDir><time> {T0} </time><code> none </code>
<directionCode>ICRS</directionCode>{source_id}</row>"""


def _scan_xml(scan, intents, eb_id=0, num_subscan=3):
    nint = len(intents)
    return f"""<row><scanNumber> {scan} </scanNumber><startTime> {T0} </startTime>
<endTime> {T0 + 1000 * DT} </endTime><numIntent> {nint} </numIntent><numSubscan> {num_subscan} </numSubscan>
<scanIntent> 1 {nint} {" ".join(intents)}</scanIntent><calDataType> 1 {nint} {" ".join(["NONE"] * nint)}</calDataType>
<calibrationOnLine> 1 {nint} {" ".join(["false"] * nint)} </calibrationOnLine><numField> 1 </numField>
<sourceName> src </sourceName><execBlockId> ExecBlock_{eb_id} </execBlockId></row>"""


def _subscan_xml(scan, subscan, intent, eb_id=0):
    return f"""<row><scanNumber> {scan} </scanNumber><subscanNumber> {subscan} </subscanNumber>
<startTime> {T0} </startTime><endTime> {T0 + DT} </endTime><fieldName> src </fieldName>
<subscanIntent>{intent}</subscanIntent><subscanMode>NO_SWITCHING</subscanMode>
<numIntegration> 1 </numIntegration><numSubintegration> 1 1 0 </numSubintegration>
<correlatorCalibration>NONE</correlatorCalibration><execBlockId> ExecBlock_{eb_id} </execBlockId></row>"""


def build_asdm(
    main_rows,
    configs,
    num_dd,
    fields,
    scans,
    subscans=(),
):
    """
    Build an in-memory ASDM with only the tables used by create_partitions.

    main_rows: list of (time_index, scan, subscan, cd_id, field_id[, eb_id]),
        in Main table order. Row i has BDF path "bdf_i".
    configs: list of (processor_type, spectral_type, dd_ids[, nant]), with
        ids 0, 1, ...
    num_dd: number of DataDescription rows (ids 0..num_dd-1)
    fields: list of (name[, with_source_id]), with ids 0, 1, ...
    scans: list of (scan, intents[, eb_id])
    subscans: list of (scan, subscan, intent[, eb_id])
    """
    asdm = pyasdm.ASDM()
    for cd_id, config in enumerate(configs):
        _add_row(
            asdm.getConfigDescription(),
            pyasdm.ConfigDescriptionRow,
            _config_description_xml(cd_id, *config),
        )
    for dd_id in range(num_dd):
        _add_row(
            asdm.getDataDescription(),
            pyasdm.DataDescriptionRow,
            _data_description_xml(dd_id),
        )
    for field_id, field in enumerate(fields):
        _add_row(asdm.getField(), pyasdm.FieldRow, _field_xml(field_id, *field))
    for scan in scans:
        _add_row(asdm.getScan(), pyasdm.ScanRow, _scan_xml(*scan))
    for subscan in subscans:
        _add_row(asdm.getSubscan(), pyasdm.SubscanRow, _subscan_xml(*subscan))
    for main_row, (time_idx, *rest) in enumerate(main_rows):
        _add_row(
            asdm.getMain(),
            pyasdm.MainRow,
            _main_xml(main_row, T0 + time_idx * DT, *rest),
        )
    return asdm


def bdf_indices(partition):
    return [int(path.split("_")[-1]) for path in partition["BDFPath"]]


def find_partition(partitions, **selection):
    found = [
        part
        for part in partitions
        if all(list(part[key]) == list(val) for key, val in selection.items())
    ]
    assert len(found) == 1, f"{len(found)} partitions match {selection}"
    return found[0]


def assert_partition_structure(partition):
    """Every entry is a 1-D array, per_bdf entries are aligned with BDFPath."""
    for key, val in partition.items():
        if key == "per_bdf":
            continue
        assert isinstance(val, np.ndarray), key
        assert val.ndim == 1, key
    for key in MANDATORY_PARTITION_AXES:
        assert key in partition
    assert set(partition["per_bdf"]) == set(PER_BDF_KEYS)
    nbdf = len(partition["BDFPath"])
    for key, val in partition["per_bdf"].items():
        assert isinstance(val, np.ndarray)
        assert val.shape == (nbdf,), key
    np.testing.assert_array_equal(partition["per_bdf"]["BDFPath"], partition["BDFPath"])
    assert partition["per_bdf"]["time"].dtype == np.int64
    for key in ("scanNumber", "subscanNumber", "fieldId", "stateId"):
        assert partition["per_bdf"][key].dtype == np.int64
        np.testing.assert_array_equal(
            np.unique(partition["per_bdf"][key]), partition[key]
        )
    assert partition["scanIntent"].dtype.kind == "U"
    assert partition["BDFPath"].dtype.kind == "U"
    # Main time order
    assert np.all(np.diff(partition["per_bdf"]["time"]) >= 0)
    np.testing.assert_array_equal(
        partition["time"],
        (np.unique(partition["per_bdf"]["time"]) - ASDM_TO_UNIX_NS).astype(
            "datetime64[ns]"
        ),
    )


# ---------------------------------------------------------------------------
# Basic / error cases
# ---------------------------------------------------------------------------


def test_create_partitions_empty():
    with pytest.raises(AttributeError, match="no attribute"):
        create_partitions(None, ["fieldId"])


def test_create_partitions_asdm_empty(asdm_empty):
    partitions = create_partitions(asdm_empty, ["fieldId"])
    assert partitions == []


def test_create_partitions_asdm_with_spw_default(asdm_with_spw_default):
    partitions = create_partitions(asdm_with_spw_default, ["fieldId"])
    assert partitions == []


def test_create_partitions_asdm_with_spw_simple(asdm_with_spw_simple, monkeypatch):
    monkeypatch.setattr(
        "pyasdm.MainRow.getBDFPath", lambda bdf_paths: "/monkypatched_path/foo"
    )
    partitions = create_partitions(asdm_with_spw_simple, ["fieldId"])
    assert partitions == []


def test_create_partitions_with_includes_asdm_with_spw_simple(
    asdm_with_main_config, monkeypatch
):
    """Main and ConfigDescription but no DataDescription/Field/Scan rows: the
    Main rows cannot be used, no partitions."""
    monkeypatch.setattr(
        "pyasdm.MainRow.getBDFPath", lambda bdf_paths: "/monkypatched_path/foo"
    )
    partitions = create_partitions(
        asdm_with_main_config,
        ["scanNumber"],
        include_processor_types=["CORRELATOR", "SPECTROMETER", "RADIOMETER"],
        include_spectral_resolution_types=[
            "CHANNEL_AVERAGE",
            "BASEBAND_WIDE",
            "FULL_RESOLUTION",
        ],
    )
    assert partitions == []


def test_create_partitions_with_filter_on_processor_type_asdm_with_spw_simple(
    asdm_with_main, monkeypatch
):
    monkeypatch.setattr(
        "pyasdm.MainRow.getBDFPath", lambda bdf_paths: "/monkypatched_path/foo"
    )
    with pytest.raises(RuntimeError, match="left after filtering processor types"):
        create_partitions(
            asdm_with_main,
            ["fieldId"],
            include_processor_types=["SPECTROMETER"],
        )


def test_create_partitions_with_filter_on_spectral_resolution_type_asdm_with_spw_simple(
    asdm_with_main, monkeypatch
):
    monkeypatch.setattr(
        "pyasdm.MainRow.getBDFPath", lambda bdf_paths: "/monkypatched_path/foo"
    )
    with pytest.raises(
        RuntimeError, match="left after filtering spectral resolution types"
    ):
        create_partitions(
            asdm_with_main,
            ["fieldId"],
            include_spectral_resolution_types=["FULL_RESOLUTION"],
        )


def test_create_partitions_mock_asdm_content(mock_asdm_set_from_file, monkeypatch):
    """The full mock ASDM of the open_asdm tests gives one partition (only the
    RADIOMETER Main row has a DataDescription row), with these values."""
    monkeypatch.setattr(
        "pyasdm.MainRow.getBDFPath", lambda row: "/monkypatched_path/foo"
    )
    asdm = pyasdm.ASDM()
    mock_asdm_set_from_file(asdm, "/unused_path/foo")

    partitions = create_partitions(asdm, None)

    assert len(partitions) == 1
    part = partitions[0]
    assert_partition_structure(part)
    expected = {
        "execBlockId": [0],
        "configDescriptionId": [0],
        "dataDescriptionId": [0],
        "spectralWindowId": [0],
        "polOrHoloId": [0],
        "fieldId": [0],
        "sourceId": [0],
        "scanNumber": [1],
        "subscanNumber": [1],
        "stateId": [0],
        "processorType": ["RADIOMETER"],
        "spectralType": ["BASEBAND_WIDE"],
        "dataUID": ["uid://A002/X11b94a6/X119f"],
        "BDFPath": ["/monkypatched_path/foo"],
        # no Subscan table in the mock ASDM
        "scanIntent": ["CALIBRATE_POINTING#UNSPECIFIED", "CALIBRATE_WVR#UNSPECIFIED"],
    }
    for key, val in expected.items():
        assert list(part[key]) == val, key
    np.testing.assert_array_equal(
        part["time"], np.array(["2024-08-10T09:57:31.200"], dtype="datetime64[ns]")
    )
    np.testing.assert_array_equal(part["per_bdf"]["time"], [5230000651200000000])


# ---------------------------------------------------------------------------
# Partition scheme (F35)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "scheme",
    [
        ["antennaId"],
        ["FIELD_ID"],
        ["ExecBlockId"],
        ["fieldId", "foo"],
        ["execBlockId"],
        ["dataDescriptionId", "execBlockId", "fieldId", "scanIntent"],
        "fieldId",
        42,
    ],
)
def test_validate_partition_scheme_invalid(scheme):
    with pytest.raises(ValueError, match="subscanNumber") as exc_info:
        validate_partition_scheme(scheme)
    for axis in OPTIONAL_PARTITION_AXES:
        assert axis in str(exc_info.value)


@pytest.mark.parametrize(
    "scheme, expected",
    [
        (None, ["fieldId"]),
        ([], []),
        (["scanNumber"], ["scanNumber"]),
        (("subscanNumber", "fieldId"), ["subscanNumber", "fieldId"]),
        (["fieldId", "scanNumber", "fieldId"], ["fieldId", "scanNumber"]),
    ],
)
def test_validate_partition_scheme(scheme, expected):
    assert validate_partition_scheme(scheme) == expected


def test_create_partitions_invalid_scheme_raises_before_reading(asdm_empty):
    with pytest.raises(ValueError, match="antennaId"):
        create_partitions(asdm_empty, ["antennaId"])


def _mosaic_asdm():
    """
    One CORRELATOR config with 2 DDs. Scan 1: calibrator field 0. Scan 2:
    fields 1 and 2 (subscans 1, 2). Scan 3: calibrator field 0 again.
    """
    main_rows = [
        (0, 1, 1, 0, 0),
        (1, 2, 1, 0, 1),
        (2, 2, 2, 0, 2),
        (3, 3, 1, 0, 0),
    ]
    return build_asdm(
        main_rows,
        configs=[("CORRELATOR", "FULL_RESOLUTION", [0, 1])],
        num_dd=2,
        fields=[("cal",), ("M100",), ("M100",)],
        scans=[
            (1, ["CALIBRATE_PHASE"]),
            (2, ["OBSERVE_TARGET"]),
            (3, ["CALIBRATE_PHASE"]),
        ],
        subscans=[
            (1, 1, "ON_SOURCE"),
            (2, 1, "ON_SOURCE"),
            (2, 2, "ON_SOURCE"),
            (3, 1, "ON_SOURCE"),
        ],
    )


@pytest.mark.parametrize(
    "scheme, expected_groups",
    [
        # (dd, intent) -> list of BDF (Main row) indices per partition
        (
            None,
            {
                "CALIBRATE_PHASE#ON_SOURCE": [[0, 3]],
                "OBSERVE_TARGET#ON_SOURCE": [[1], [2]],
            },
        ),
        (
            [],
            {
                "CALIBRATE_PHASE#ON_SOURCE": [[0, 3]],
                "OBSERVE_TARGET#ON_SOURCE": [[1, 2]],
            },
        ),
        (
            ["scanNumber"],
            {
                "CALIBRATE_PHASE#ON_SOURCE": [[0], [3]],
                "OBSERVE_TARGET#ON_SOURCE": [[1, 2]],
            },
        ),
        (
            ["subscanNumber"],
            {
                "CALIBRATE_PHASE#ON_SOURCE": [[0, 3]],
                "OBSERVE_TARGET#ON_SOURCE": [[1], [2]],
            },
        ),
        (
            ["fieldId", "scanNumber"],
            {
                "CALIBRATE_PHASE#ON_SOURCE": [[0], [3]],
                "OBSERVE_TARGET#ON_SOURCE": [[1], [2]],
            },
        ),
    ],
)
def test_create_partitions_schemes(bdf_path_from_data_uid, scheme, expected_groups):
    partitions = create_partitions(_mosaic_asdm(), scheme)

    num_expected = 2 * sum(len(groups) for groups in expected_groups.values())
    assert len(partitions) == num_expected
    for part in partitions:
        assert_partition_structure(part)
        assert len(part["dataDescriptionId"]) == 1
        assert list(part["configDescriptionId"]) == [0]
    for dd_id in (0, 1):
        for intent, groups in expected_groups.items():
            found = sorted(
                bdf_indices(part)
                for part in partitions
                if part["dataDescriptionId"][0] == dd_id
                and list(part["scanIntent"]) == [intent]
            )
            assert found == sorted(groups), (dd_id, intent)


def test_create_partitions_multi_field_per_bdf(bdf_path_from_data_uid):
    """Without fieldId axis, per_bdf gives the field and scan of every BDF (F43)."""
    partitions = create_partitions(_mosaic_asdm(), [])
    part = find_partition(
        partitions, dataDescriptionId=[1], scanIntent=["OBSERVE_TARGET#ON_SOURCE"]
    )
    assert list(part["fieldId"]) == [1, 2]
    assert list(part["sourceId"]) == [11, 12]
    assert list(part["BDFPath"]) == ["bdf_1", "bdf_2"]
    assert list(part["per_bdf"]["fieldId"]) == [1, 2]
    assert list(part["per_bdf"]["scanNumber"]) == [2, 2]
    assert list(part["per_bdf"]["subscanNumber"]) == [1, 2]
    assert list(part["per_bdf"]["time"]) == [T0 + DT, T0 + 2 * DT]

    cal = find_partition(
        partitions, dataDescriptionId=[0], scanIntent=["CALIBRATE_PHASE#ON_SOURCE"]
    )
    assert list(cal["scanNumber"]) == [1, 3]
    assert list(cal["per_bdf"]["scanNumber"]) == [1, 3]
    assert list(cal["per_bdf"]["fieldId"]) == [0, 0]
    assert list(cal["per_bdf"]["time"]) == [T0, T0 + 3 * DT]


# ---------------------------------------------------------------------------
# Time order (F07)
# ---------------------------------------------------------------------------


def test_create_partitions_interleaved_configs_time_order(bdf_path_from_data_uid):
    """
    ALMA-like layout with 4 ConfigDescriptions interleaved in Main (WVR,
    SQLD, CHANNEL_AVERAGE, FULL_RESOLUTION with 4 DDs) and the open_asdm
    default filters: the BDFs of every partition are in time order (F07).
    """
    configs = [
        ("RADIOMETER", "FULL_RESOLUTION", [0]),
        ("RADIOMETER", "BASEBAND_WIDE", [1, 2, 3, 4]),
        ("CORRELATOR", "CHANNEL_AVERAGE", [5, 6, 7, 8]),
        ("CORRELATOR", "FULL_RESOLUTION", [9, 10, 11, 12]),
    ]
    num_subscans = 14
    main_rows = [
        (sub, 3, sub, cd_id, 0)
        for sub in range(1, num_subscans + 1)
        for cd_id in range(len(configs))
    ]
    asdm = build_asdm(
        main_rows,
        configs,
        num_dd=13,
        fields=[("J2258-2758",)],
        scans=[(3, ["CALIBRATE_FOCUS", "CALIBRATE_WVR"], 0, num_subscans)],
        subscans=[(3, sub, "ON_SOURCE") for sub in range(1, num_subscans + 1)],
    )

    partitions = create_partitions(
        asdm, ["fieldId"], DEFAULT_PROCESSOR_TYPES, DEFAULT_SPECTRAL_TYPES
    )

    assert sorted(part["dataDescriptionId"][0] for part in partitions) == [
        9,
        10,
        11,
        12,
    ]
    for part in partitions:
        assert_partition_structure(part)
        assert list(part["configDescriptionId"]) == [3]
        assert list(part["scanIntent"]) == [
            "CALIBRATE_FOCUS#ON_SOURCE",
            "CALIBRATE_WVR#ON_SOURCE",
        ]
        expected_rows = [4 * (sub - 1) + 3 for sub in range(1, num_subscans + 1)]
        assert bdf_indices(part) == expected_rows
        assert list(part["per_bdf"]["subscanNumber"]) == list(
            range(1, num_subscans + 1)
        )
        assert list(part["per_bdf"]["time"]) == [
            T0 + sub * DT for sub in range(1, num_subscans + 1)
        ]


def test_create_partitions_main_not_in_time_order(bdf_path_from_data_uid):
    """BDFPath follows the Main time, not the Main table order; Main rows with
    the same time keep the Main table order (stable)."""
    # (Main keys (time, configDescriptionId, fieldId) must be unique: the two
    # rows with the same time are for two fields)
    main_rows = [
        (5, 1, 3, 0, 0),
        (2, 1, 2, 0, 1),
        (2, 1, 2, 0, 0),
        (0, 1, 1, 0, 0),
    ]
    asdm = build_asdm(
        main_rows,
        configs=[("CORRELATOR", "FULL_RESOLUTION", [0])],
        num_dd=1,
        fields=[("src",), ("src2",)],
        scans=[(1, ["OBSERVE_TARGET"])],
        subscans=[(1, sub, "ON_SOURCE") for sub in (1, 2, 3)],
    )

    (part,) = create_partitions(asdm, [])

    assert_partition_structure(part)
    assert bdf_indices(part) == [3, 1, 2, 0]
    assert list(part["per_bdf"]["time"]) == [T0, T0 + 2 * DT, T0 + 2 * DT, T0 + 5 * DT]
    assert list(part["per_bdf"]["subscanNumber"]) == [1, 2, 2, 3]
    assert list(part["per_bdf"]["fieldId"]) == [0, 1, 0, 0]
    assert list(part["dataUID"]) == [_data_uid(row) for row in (3, 1, 2, 0)]
    assert len(part["time"]) == 3


# ---------------------------------------------------------------------------
# Intents (F34, F44)
# ---------------------------------------------------------------------------


def test_create_partitions_subscan_intents(bdf_path_from_data_uid):
    """ON/OFF and HOT/AMBIENT subscans go to separate partitions, labelled with
    sorted unique "INTENT#SUBINTENT" strings (F34). A subscan without Subscan
    row gets the UNSPECIFIED subintent."""
    main_rows = [
        (0, 1, 1, 0, 0),
        (1, 1, 2, 0, 0),
        (2, 1, 3, 0, 0),
        (3, 2, 1, 0, 0),
        (4, 2, 2, 0, 0),
        (5, 3, 1, 0, 0),
    ]
    asdm = build_asdm(
        main_rows,
        configs=[("CORRELATOR", "FULL_RESOLUTION", [0])],
        num_dd=1,
        fields=[("src",)],
        scans=[
            (1, ["OBSERVE_TARGET"]),
            (2, ["CALIBRATE_WVR", "CALIBRATE_ATMOSPHERE"]),
            (3, ["OBSERVE_TARGET"]),
        ],
        subscans=[
            (1, 1, "ON_SOURCE"),
            (1, 2, "OFF_SOURCE"),
            (1, 3, "ON_SOURCE"),
            (2, 1, "HOT"),
            (2, 2, "AMBIENT"),
        ],
    )

    partitions = create_partitions(asdm, None)

    assert len(partitions) == 5
    for part in partitions:
        assert_partition_structure(part)
        assert isinstance(part["scanIntent"][0], str)
    expected = {
        ("OBSERVE_TARGET#ON_SOURCE",): [0, 2],
        ("OBSERVE_TARGET#OFF_SOURCE",): [1],
        ("CALIBRATE_ATMOSPHERE#HOT", "CALIBRATE_WVR#HOT"): [3],
        ("CALIBRATE_ATMOSPHERE#AMBIENT", "CALIBRATE_WVR#AMBIENT"): [4],
        ("OBSERVE_TARGET#UNSPECIFIED",): [5],
    }
    for intents, rows in expected.items():
        part = find_partition(partitions, scanIntent=list(intents))
        assert bdf_indices(part) == rows
        assert list(part["per_bdf"]["subscanNumber"]) == [
            main_rows[row][2] for row in rows
        ]


def test_create_partitions_multiple_exec_blocks(bdf_path_from_data_uid):
    """Scans are matched by (execBlockId, scanNumber): the same scan number in
    two ExecBlocks keeps its own intents, and EBs are never mixed."""
    main_rows = [
        (0, 1, 1, 0, 0, 0),
        (1, 2, 1, 0, 0, 0),
        (10, 1, 1, 0, 0, 1),
        (11, 2, 1, 0, 0, 1),
    ]
    asdm = build_asdm(
        main_rows,
        configs=[("CORRELATOR", "FULL_RESOLUTION", [0])],
        num_dd=1,
        fields=[("src",)],
        scans=[
            (1, ["CALIBRATE_PHASE"], 0),
            (2, ["OBSERVE_TARGET"], 0),
            (1, ["OBSERVE_TARGET"], 1),
            (2, ["OBSERVE_TARGET"], 1),
        ],
        subscans=[
            (1, 1, "ON_SOURCE", 0),
            (2, 1, "ON_SOURCE", 0),
            (1, 1, "ON_SOURCE", 1),
            (2, 1, "ON_SOURCE", 1),
        ],
    )

    partitions = create_partitions(asdm, None)

    assert len(partitions) == 3
    eb0_cal = find_partition(
        partitions, execBlockId=[0], scanIntent=["CALIBRATE_PHASE#ON_SOURCE"]
    )
    assert bdf_indices(eb0_cal) == [0]
    eb0_target = find_partition(
        partitions, execBlockId=[0], scanIntent=["OBSERVE_TARGET#ON_SOURCE"]
    )
    assert bdf_indices(eb0_target) == [1]
    eb1_target = find_partition(
        partitions, execBlockId=[1], scanIntent=["OBSERVE_TARGET#ON_SOURCE"]
    )
    assert bdf_indices(eb1_target) == [2, 3]
    assert list(eb1_target["per_bdf"]["scanNumber"]) == [1, 2]


# ---------------------------------------------------------------------------
# Mandatory configDescriptionId axis (F74), filters
# ---------------------------------------------------------------------------


def test_create_partitions_configs_sharing_a_data_description(bdf_path_from_data_uid):
    """Two ConfigDescriptions (different antenna sets) using the same DD give
    two partitions with one configDescriptionId each (F74)."""
    main_rows = [
        (0, 1, 1, 0, 0),
        (1, 1, 1, 1, 0),
        (2, 1, 2, 0, 0),
        (3, 1, 2, 1, 0),
    ]
    asdm = build_asdm(
        main_rows,
        configs=[
            ("CORRELATOR", "FULL_RESOLUTION", [0], 2),
            ("CORRELATOR", "FULL_RESOLUTION", [0], 3),
        ],
        num_dd=1,
        fields=[("src",)],
        scans=[(1, ["OBSERVE_TARGET"])],
        subscans=[(1, 1, "ON_SOURCE"), (1, 2, "ON_SOURCE")],
    )

    partitions = create_partitions(asdm, None)

    assert len(partitions) == 2
    for cd_id, rows in [(0, [0, 2]), (1, [1, 3])]:
        part = find_partition(partitions, configDescriptionId=[cd_id])
        assert_partition_structure(part)
        assert list(part["dataDescriptionId"]) == [0]
        assert bdf_indices(part) == rows


def test_create_partitions_filters(bdf_path_from_data_uid):
    """Processor and spectral resolution type filters select the configs."""
    configs = [
        ("RADIOMETER", "FULL_RESOLUTION", [0]),
        ("CORRELATOR", "CHANNEL_AVERAGE", [1]),
        ("CORRELATOR", "FULL_RESOLUTION", [2, 3]),
    ]
    main_rows = [(0, 1, 1, 0, 0), (0, 1, 1, 1, 0), (0, 1, 1, 2, 0)]
    asdm = build_asdm(
        main_rows,
        configs,
        num_dd=4,
        fields=[("src",)],
        scans=[(1, ["OBSERVE_TARGET"])],
        subscans=[(1, 1, "ON_SOURCE")],
    )

    def dds(partitions):
        return sorted(int(part["dataDescriptionId"][0]) for part in partitions)

    assert dds(create_partitions(asdm, None)) == [0, 1, 2, 3]
    corr_full = create_partitions(
        asdm, None, DEFAULT_PROCESSOR_TYPES, DEFAULT_SPECTRAL_TYPES
    )
    assert dds(corr_full) == [2, 3]
    for part in corr_full:
        assert list(part["processorType"]) == ["CORRELATOR"]
        assert list(part["spectralType"]) == ["FULL_RESOLUTION"]
        assert list(part["spectralWindowId"]) == list(part["dataDescriptionId"])
        assert bdf_indices(part) == [2]
    radiometer = create_partitions(asdm, None, ["RADIOMETER"])
    assert dds(radiometer) == [0]
    assert list(radiometer[0]["processorType"]) == ["RADIOMETER"]
    assert dds(create_partitions(asdm, None, None, ["CHANNEL_AVERAGE"])) == [1]


# ---------------------------------------------------------------------------
# Robustness: optional / missing metadata (F73), no prints (F75)
# ---------------------------------------------------------------------------


def test_create_partitions_field_without_source_id(bdf_path_from_data_uid):
    """Field.sourceId is optional: no error, sourceId -1 (F73)."""
    asdm = build_asdm(
        [(0, 1, 1, 0, 0), (1, 1, 1, 0, 1)],
        configs=[("CORRELATOR", "FULL_RESOLUTION", [0])],
        num_dd=1,
        fields=[("with_source",), ("without_source", False)],
        scans=[(1, ["OBSERVE_TARGET"])],
        subscans=[(1, 1, "ON_SOURCE")],
    )
    assert not asdm.getField().get()[1].isSourceIdExists()

    partitions = create_partitions(asdm, None)

    assert len(partitions) == 2
    assert list(find_partition(partitions, fieldId=[0])["sourceId"]) == [10]
    assert list(find_partition(partitions, fieldId=[1])["sourceId"]) == [-1]


def test_create_partitions_rows_without_metadata_are_ignored(bdf_path_from_data_uid):
    """Main rows of scans without Scan row, or with data descriptions without
    DataDescription row, are left out; the others are kept."""
    asdm = build_asdm(
        [(0, 1, 1, 0, 0), (1, 2, 1, 0, 0)],
        configs=[("CORRELATOR", "FULL_RESOLUTION", [0, 1])],
        num_dd=1,
        fields=[("src",)],
        scans=[(1, ["OBSERVE_TARGET"])],
        subscans=[(1, 1, "ON_SOURCE")],
    )

    (part,) = create_partitions(asdm, None)

    assert list(part["dataDescriptionId"]) == [0]
    assert bdf_indices(part) == [0]
    assert list(part["scanNumber"]) == [1]


def test_create_partitions_does_not_print(bdf_path_from_data_uid, capsys):
    """No output on stdout/stderr, also when every row is its own partition
    (F75)."""
    partitions = create_partitions(_mosaic_asdm(), ["scanNumber", "subscanNumber"])
    assert len(partitions) == 8
    captured = capsys.readouterr()
    assert captured.out == ""


# ---------------------------------------------------------------------------
# Synthetic on-disk ASDMs: partitions match the expected (truth) partitions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fixture_name, scheme, processor_types",
    [
        ("synth_interferometric", None, None),
        ("synth_full_pol", None, None),
        ("synth_single_dish", None, None),
        ("synth_mosaic", None, None),
        ("synth_mosaic", [], None),
        ("synth_mosaic", ["subscanNumber"], None),
        ("synth_mosaic", ["scanNumber", "fieldId"], None),
        ("synth_interleaved", None, None),
        ("synth_interleaved", None, ["RADIOMETER"]),
    ],
)
def test_create_partitions_synthetic(request, fixture_name, scheme, processor_types):
    truth = request.getfixturevalue(fixture_name)
    asdm = pyasdm.ASDM()
    asdm.setFromFile(truth.path)

    partitions = create_partitions(
        asdm,
        scheme,
        processor_types or DEFAULT_PROCESSOR_TYPES,
        DEFAULT_SPECTRAL_TYPES,
    )
    expected_partitions = truth.expected_partitions(
        scheme, processor_types, DEFAULT_SPECTRAL_TYPES
    )

    assert len(partitions) == len(expected_partitions)
    for expected in expected_partitions:
        part = find_partition(
            partitions,
            dataDescriptionId=[expected.dd_id],
            scanIntent=list(expected.obs_modes),
            fieldId=list(expected.field_ids),
            scanNumber=list(expected.scan_numbers),
            subscanNumber=list(expected.subscan_numbers),
        )
        assert_partition_structure(part)
        assert [Path(path).name for path in part["BDFPath"]] == [
            Path(bdf.path).name for bdf in expected.bdfs
        ]
        assert list(part["per_bdf"]["time"]) == [
            bdf.main_time_ns for bdf in expected.bdfs
        ]
        assert list(part["per_bdf"]["scanNumber"]) == [
            bdf.scan_number for bdf in expected.bdfs
        ]
        assert list(part["per_bdf"]["subscanNumber"]) == [
            bdf.subscan_number for bdf in expected.bdfs
        ]
        assert list(part["per_bdf"]["fieldId"]) == [
            bdf.field_id for bdf in expected.bdfs
        ]
        assert list(part["spectralWindowId"]) == [expected.spw_id]


# ---------------------------------------------------------------------------
# finalize_partitions_groupby
# ---------------------------------------------------------------------------


def test_finalize_partitions_groupby_missing_columns():
    with pytest.raises(ValueError, match="missing the columns"):
        finalize_partitions_groupby(
            pd.DataFrame([[0, 0]], columns=["fieldId", "scanIntent"]),
            ["fieldId"],
            [["CALIBRATE_POINTING#ON_SOURCE"]],
        )


def _partitioning_dict(num_rows, **overrides):
    rows = {
        "time": [T0 + idx * DT for idx in range(num_rows)],
        "fieldId": [0] * num_rows,
        "configDescriptionId": [0] * num_rows,
        "scanNumber": [1] * num_rows,
        "subscanNumber": [1] * num_rows,
        "stateId": [0] * num_rows,
        "dataUID": [f"uid://A002/X11b94a6/X{idx}" for idx in range(num_rows)],
        "BDFPath": [f"/path/bdf_{idx}" for idx in range(num_rows)],
        "execBlockId": [0] * num_rows,
        "dataDescriptionId": [0] * num_rows,
        "processorType": ["RADIOMETER"] * num_rows,
        "spectralType": ["BASEBAND_WIDE"] * num_rows,
        "spectralWindowId": [0] * num_rows,
        "polOrHoloId": [0] * num_rows,
        "sourceId": [0] * num_rows,
        "scanIntent": [0] * num_rows,
    }
    rows.update(overrides)
    return rows


UNIQUE_SCAN_INTENTS = np.array(
    [
        ["CALIBRATE_POINTING#ON_SOURCE", "CALIBRATE_WVR#ON_SOURCE"],
        ["CALIBRATE_ATMOSPHERE#HOT", "CALIBRATE_WVR#HOT"],
    ]
)


@pytest.mark.parametrize(
    "partitioning_dict, expected_bdf_groups",
    [
        (_partitioning_dict(1), [[0]]),
        (_partitioning_dict(2, spectralWindowId=[0, 1]), [[0, 1]]),
        (
            _partitioning_dict(
                3,
                subscanNumber=[1, 2, 3],
                processorType=["RADIOMETER", "RADIOMETER", "CORRELATOR"],
                scanIntent=[0, 0, 1],
            ),
            [[0], [1], [2]],
        ),
        # rows not in time order
        (
            _partitioning_dict(3, time=[T0 + 2 * DT, T0, T0 + DT]),
            [[1, 2, 0]],
        ),
    ],
)
def test_finalize_partitions_groupby(partitioning_dict, expected_bdf_groups):
    partitioning_df = pd.DataFrame.from_dict(partitioning_dict)

    parts = finalize_partitions_groupby(
        partitioning_df,
        ["fieldId", "scanNumber", "subscanNumber", "scanIntent"],
        UNIQUE_SCAN_INTENTS,
    )

    assert isinstance(parts, list)
    assert len(parts) == len(expected_bdf_groups)
    for part, rows in zip(parts, expected_bdf_groups, strict=False):
        assert_partition_structure(part)
        assert bdf_indices(part) == rows
        np.testing.assert_array_equal(
            part["per_bdf"]["time"], np.asarray(partitioning_dict["time"])[rows]
        )
        intent_idx = partitioning_dict["scanIntent"][rows[0]]
        assert list(part["scanIntent"]) == list(UNIQUE_SCAN_INTENTS[intent_idx])
        assert list(part["spectralWindowId"]) == sorted(
            set(np.asarray(partitioning_dict["spectralWindowId"])[rows])
        )
        assert part["processorType"].dtype.kind == "U"


def test_finalize_partitions_groupby_merges_intent_lists():
    """Grouping without scanIntent gives the union of the intents, sorted."""
    partitioning_df = pd.DataFrame.from_dict(_partitioning_dict(2, scanIntent=[1, 0]))
    (part,) = finalize_partitions_groupby(
        partitioning_df, ["fieldId"], UNIQUE_SCAN_INTENTS
    )
    assert list(part["scanIntent"]) == [
        "CALIBRATE_ATMOSPHERE#HOT",
        "CALIBRATE_POINTING#ON_SOURCE",
        "CALIBRATE_WVR#HOT",
        "CALIBRATE_WVR#ON_SOURCE",
    ]
