import copy
import gc
import weakref

import numpy as np
import pandas as pd
import pyasdm
import pytest

from xradio.measurement_set._utils._asdm._utils import metadata_tables
from xradio.measurement_set._utils._asdm._utils.metadata_tables import (
    clear_asdm_table_cache,
    exp_asdm_table_to_df,
    load_asdm_col,
)

FIELD_TIME = 5230000639104000000


def field_row_xml(name: str, ra: float, dec: float, source_id: int | None) -> str:
    source_id_xml = "" if source_id is None else f"<sourceId> {source_id} </sourceId>"
    return f"""
  <row>
    <fieldId> Field_0 </fieldId>
    <fieldName> {name} </fieldName>
    <numPoly> 1 </numPoly>
    <delayDir> 2 1 2 {ra!r} {dec!r} </delayDir>
    <phaseDir> 2 1 2 {ra!r} {dec!r} </phaseDir>
    <referenceDir> 2 1 2 {ra!r} {dec!r} </referenceDir>
    <time> {FIELD_TIME} </time>
    <code> none </code>
    <directionCode>ICRS</directionCode>
    {source_id_xml}
  </row>"""


def add_field_row(
    asdm: pyasdm.ASDM, name: str, ra: float, dec: float, source_id: int | None = 0
) -> pyasdm.FieldRow:
    field_table = asdm.getField()
    field_row = pyasdm.FieldRow(field_table)
    field_row.setFromXML(field_row_xml(name, ra, dec, source_id))
    return field_table.add(field_row)


def make_asdm_with_fields(fields: list[tuple]) -> pyasdm.ASDM:
    """ASDM with one Field row per (name, ra, dec, source_id) tuple."""
    asdm = pyasdm.ASDM()
    for field in fields:
        add_field_row(asdm, *field)
    return asdm


def add_polarization_row(asdm: pyasdm.ASDM, products: list[str]) -> None:
    corr_product = " ".join(" ".join(product) for product in products)
    polarization_table = asdm.getPolarization()
    polarization_row = pyasdm.PolarizationRow(polarization_table)
    polarization_row.setFromXML(
        f"""
  <row>
    <polarizationId> Polarization_0 </polarizationId>
    <numCorr> {len(products)} </numCorr>
    <corrType> 1 {len(products)} {" ".join(products)}</corrType>
    <corrProduct> 2 {len(products)} 2 {corr_product}</corrProduct>
  </row>"""
    )
    polarization_table.add(polarization_row)


@pytest.fixture
def count_conversions(monkeypatch):
    """Count the columns converted by exp_asdm_table_to_df (calls to load_asdm_col)."""
    calls = []
    original = metadata_tables.load_asdm_col

    def counting_load_asdm_col(sdm_table, col_name, **kwargs):
        calls.append(col_name)
        return original(sdm_table, col_name, **kwargs)

    monkeypatch.setattr(metadata_tables, "load_asdm_col", counting_load_asdm_col)
    return calls


def test_load_asdm_col_empty():
    with pytest.raises(AttributeError, match="no attribute"):
        load_asdm_col(None, "fieldId")


def test_load_asdm_col_asdm_empty(asdm_empty):
    with pytest.raises(AttributeError, match="no attribute"):
        load_asdm_col(asdm_empty, "foo")


def test_load_asdm_col_asdm_spw_default(asdm_with_spw_default):
    with pytest.raises(AttributeError, match="has no attribute"):
        load_asdm_col(asdm_with_spw_default.getSpectralWindow(), "foo_nonexistant")

    spw_id = load_asdm_col(
        asdm_with_spw_default.getSpectralWindow(), "spectralWindowId"
    )
    assert spw_id == [0]
    num_chan = load_asdm_col(asdm_with_spw_default.getSpectralWindow(), "numChan")
    assert num_chan == [0]
    ref_freq = load_asdm_col(asdm_with_spw_default.getSpectralWindow(), "refFreq")
    assert ref_freq == [0]
    bb_name = load_asdm_col(asdm_with_spw_default.getSpectralWindow(), "basebandName")
    assert bb_name == ["NOBB"]
    with pytest.raises(ValueError, match="assocNature"):
        spw_id = load_asdm_col(asdm_with_spw_default.getSpectralWindow(), "assocNature")
    with pytest.raises(ValueError, match="assocSpectralWindowId"):
        spw_id = load_asdm_col(
            asdm_with_spw_default.getSpectralWindow(), "assocSpectralWindowId"
        )
    assert spw_id == [0]


def test_load_asdm_col_asdm_spw_simple(asdm_with_spw_simple):
    with pytest.raises(AttributeError, match="has no attribute"):
        load_asdm_col(asdm_with_spw_simple, "foo_nonexistant")

    from pyasdm.enumerations import SpectralResolutionType

    spw_id = load_asdm_col(asdm_with_spw_simple.getSpectralWindow(), "spectralWindowId")
    assert spw_id == [0, 1]
    num_chan = load_asdm_col(asdm_with_spw_simple.getSpectralWindow(), "numChan")
    assert num_chan == [1, 128]
    ref_freq = load_asdm_col(asdm_with_spw_simple.getSpectralWindow(), "refFreq")
    assert ref_freq == [8.6021e10, 9.702100190734863e10]
    bb_name = load_asdm_col(asdm_with_spw_simple.getSpectralWindow(), "basebandName")
    assert bb_name == ["BB_1", "BB_1"]
    # spw_id = load_asdm_col(asdm_with_spw_simple.getSpectralWindow(), "ChanFreqArray")
    # assert spw_id == [8.5021e10]
    assoc_nature = load_asdm_col(
        asdm_with_spw_simple.getSpectralWindow(), "assocNature"
    )
    assert assoc_nature == [
        [SpectralResolutionType("BASEBAND_WIDE")],
        [
            SpectralResolutionType("BASEBAND_WIDE"),
            SpectralResolutionType("BASEBAND_WIDE"),
            SpectralResolutionType("BASEBAND_WIDE"),
            SpectralResolutionType("CHANNEL_AVERAGE"),
        ],
    ]

    assoc_spw_id = load_asdm_col(
        asdm_with_spw_simple.getSpectralWindow(), "assocSpectralWindowId"
    )
    assert len(assoc_spw_id) == 2
    assert assoc_spw_id[0] == [0]
    assert (assoc_spw_id[1] == [5, 6, 7, 9]).all()


def test_exp_asdm_table_to_df_empty():
    with pytest.raises(AttributeError, match="no attribute"):
        exp_asdm_table_to_df(None, "Main", ["fieldId"])


def test_exp_asdm_table_to_df_asdm_empty(asdm_empty):
    cols = ["fieldId"]
    main_df = exp_asdm_table_to_df(asdm_empty, "Main", cols)
    # assert main_df == pd.DataFrame(columns=cols)
    assert main_df.empty


def test_exp_asdm_table_to_df_asdm_with_spw_default(asdm_with_spw_default):
    cols = ["fieldId"]
    main_df = exp_asdm_table_to_df(asdm_with_spw_default, "Main", cols)
    assert main_df.empty


def test_exp_asdm_table_to_df_asdm_with_spw_simple(asdm_with_spw_simple):
    cols = ["fieldId"]
    main_df = exp_asdm_table_to_df(asdm_with_spw_simple, "Main", cols)
    assert main_df.empty
    pd.testing.assert_frame_equal(
        main_df, pd.DataFrame([], columns=cols, dtype="float64")
    )

    cols = ["spectralWindowId"]
    spw_df = exp_asdm_table_to_df(asdm_with_spw_simple, "SpectralWindow", cols)
    # assert spw_df == pd.DataFrame([[1]], columns=["spectralWindowId"])
    pd.testing.assert_frame_equal(
        spw_df, pd.DataFrame([[0], [1]], columns=["spectralWindowId"])
    )


def test_exp_asdm_table_to_df_asdm_with_spw_simple_plus_feed(asdm_with_simple_feed):
    from pyasdm.enumerations import PolarizationType

    cols = ["polarizationTypes"]
    feed_df = exp_asdm_table_to_df(asdm_with_simple_feed, "Feed", cols)
    pd.testing.assert_frame_equal(
        feed_df,
        pd.DataFrame(
            [
                [
                    [
                        PolarizationType.X,
                        PolarizationType.Y,
                    ]
                ]
            ],
            columns=["polarizationTypes"],
        ),
    )


def test_exp_asdm_table_to_df_asdm_with_spw_simple_plus_polarization(
    asdm_with_polarization,
):
    cols = ["corrProduct"]
    polarization_df = exp_asdm_table_to_df(asdm_with_polarization, "Polarization", cols)
    assert polarization_df.columns == cols
    assert polarization_df.shape == (1, 1)
    # [numCorr][2] nesting is kept
    assert polarization_df["corrProduct"].values[0] == [["X", "X"], ["Y", "Y"]]


def test_exp_asdm_table_to_df_full_pol_corr_product_and_corr_type():
    asdm = pyasdm.ASDM()
    add_polarization_row(asdm, ["XX", "XY", "YX", "YY"])

    polarization_df = exp_asdm_table_to_df(
        asdm, "Polarization", ["polarizationId", "numCorr", "corrType", "corrProduct"]
    )
    assert polarization_df.shape == (1, 4)
    assert polarization_df["polarizationId"].values[0] == 0
    assert polarization_df["numCorr"].values[0] == 4
    assert polarization_df["corrType"].values[0] == ["XX", "XY", "YX", "YY"]
    assert polarization_df["corrProduct"].values[0] == [
        ["X", "X"],
        ["X", "Y"],
        ["Y", "X"],
        ["Y", "Y"],
    ]


def test_exp_asdm_table_to_df_asdm_with_field_direction(
    asdm_with_main_etc_data_description_polarization_field_source,
):
    cols = ["referenceDir"]
    field_df = exp_asdm_table_to_df(
        asdm_with_main_etc_data_description_polarization_field_source, "Field", cols
    )
    assert field_df.columns == cols
    assert field_df.shape == (1, 1)
    assert np.allclose(
        field_df["referenceDir"].values[0], [1.14870, -0.0234314], rtol=1e-5
    )


def test_load_asdm_col_converted_values():
    asdm = make_asdm_with_fields(
        [("J0423-0120", 1.25, -0.5, 3), ("M100", 0.5, 0.25, 7)]
    )
    field_table = asdm.getField()

    assert load_asdm_col(field_table, "fieldId") == [0, 1]
    assert load_asdm_col(field_table, "fieldName") == ["J0423-0120", "M100"]
    assert load_asdm_col(field_table, "sourceId") == [3, 7]
    # 2-D Angle: nested lists of floats (rad)
    assert load_asdm_col(field_table, "phaseDir") == [[[1.25, -0.5]], [[0.5, 0.25]]]
    # ArrayTime is returned as given by pyasdm
    times = load_asdm_col(field_table, "time")
    assert all(isinstance(time, pyasdm.types.ArrayTime) for time in times)
    assert [time.get() for time in times] == [FIELD_TIME, FIELD_TIME]


def test_load_asdm_col_allow_absent():
    asdm = make_asdm_with_fields(
        [("with_source", 1.0, 0.5, 4), ("without_source", 1.5, 0.25, None)]
    )
    field_table = asdm.getField()

    with pytest.raises(ValueError, match="sourceId"):
        load_asdm_col(field_table, "sourceId")
    assert load_asdm_col(field_table, "sourceId", allow_absent=True) == [4, None]
    # Mandatory attributes are not affected
    assert load_asdm_col(field_table, "fieldName", allow_absent=True) == [
        "with_source",
        "without_source",
    ]
    with pytest.raises(AttributeError, match="has no attribute"):
        load_asdm_col(field_table, "foo_nonexistent", allow_absent=True)


def test_exp_asdm_table_to_df_allow_absent():
    asdm = make_asdm_with_fields(
        [("with_source", 1.0, 0.5, 4), ("without_source", 1.5, 0.25, None)]
    )

    field_df = exp_asdm_table_to_df(
        asdm, "Field", ["fieldId", "sourceId"], allow_absent=True
    )
    assert field_df["fieldId"].tolist() == [0, 1]
    assert field_df["sourceId"].isna().tolist() == [False, True]
    assert field_df["sourceId"].values[0] == 4

    # The values converted with allow_absent=True are not reused without it
    with pytest.raises(ValueError, match="sourceId"):
        exp_asdm_table_to_df(asdm, "Field", ["fieldId", "sourceId"])


def test_exp_asdm_table_to_df_cache_avoids_reconversion(count_conversions):
    asdm = make_asdm_with_fields([("A", 1.0, 0.5, 0), ("B", 2.0, -0.5, 1)])

    cols = ["fieldId", "fieldName"]
    field_df_1 = exp_asdm_table_to_df(asdm, "Field", cols)
    assert count_conversions == ["fieldId", "fieldName"]

    field_df_2 = exp_asdm_table_to_df(asdm, "Field", cols)
    assert count_conversions == ["fieldId", "fieldName"]
    assert field_df_2 is not field_df_1
    pd.testing.assert_frame_equal(
        field_df_2, pd.DataFrame({"fieldId": [0, 1], "fieldName": ["A", "B"]})
    )

    # Only the columns not converted before are converted
    field_df_3 = exp_asdm_table_to_df(asdm, "Field", ["fieldName", "sourceId"])
    assert count_conversions == ["fieldId", "fieldName", "sourceId"]
    pd.testing.assert_frame_equal(
        field_df_3, pd.DataFrame({"fieldName": ["A", "B"], "sourceId": [0, 1]})
    )


def test_exp_asdm_table_to_df_cache_returns_copies():
    asdm = make_asdm_with_fields([("A", 1.0, 0.5, 0), ("B", 2.0, -0.5, 1)])
    cols = ["fieldId", "fieldName", "phaseDir", "time"]

    field_df_1 = exp_asdm_table_to_df(asdm, "Field", cols)
    expected = {
        "fieldId": [0, 1],
        "fieldName": ["A", "B"],
        "phaseDir": [[[1.0, 0.5]], [[2.0, -0.5]]],
        "time": [FIELD_TIME, FIELD_TIME],
    }

    # Modify everything that can be modified in the returned DataFrame
    field_df_1.loc[0, "fieldName"] = "changed"
    field_df_1["fieldId"] += 10
    field_df_1["phaseDir"].values[1][0][0] = 99.0
    # in-place operator of pyasdm Interval/ArrayTime: changes the object itself
    time_0 = field_df_1["time"].values[0]
    time_0 += 1_000_000_000
    field_df_1["new_col"] = 1
    assert field_df_1["time"].values[0].get() == FIELD_TIME + 1_000_000_000
    assert field_df_1["phaseDir"].values[1] == [[99.0, -0.5]]

    field_df_2 = exp_asdm_table_to_df(asdm, "Field", cols)
    assert list(field_df_2.columns) == cols
    assert field_df_2["fieldId"].tolist() == expected["fieldId"]
    assert field_df_2["fieldName"].tolist() == expected["fieldName"]
    assert field_df_2["phaseDir"].tolist() == expected["phaseDir"]
    assert [time.get() for time in field_df_2["time"]] == expected["time"]
    assert field_df_2["time"].values[1] is not field_df_1["time"].values[1]
    assert field_df_2["phaseDir"].values[0] is not field_df_1["phaseDir"].values[0]


def test_exp_asdm_table_to_df_cache_copies_arrays_and_enumerations(
    asdm_with_spw_simple,
):
    from pyasdm.enumerations import SpectralResolutionType

    cols = ["spectralWindowId", "assocSpectralWindowId", "assocNature"]
    spw_df_1 = exp_asdm_table_to_df(asdm_with_spw_simple, "SpectralWindow", cols)
    spw_df_1["assocSpectralWindowId"].values[1][0] = 42
    spw_df_1["assocNature"].values[0].append(SpectralResolutionType("CHANNEL_AVERAGE"))

    spw_df_2 = exp_asdm_table_to_df(asdm_with_spw_simple, "SpectralWindow", cols)
    assert spw_df_2["spectralWindowId"].tolist() == [0, 1]
    np.testing.assert_array_equal(spw_df_2["assocSpectralWindowId"].values[0], [0])
    np.testing.assert_array_equal(
        spw_df_2["assocSpectralWindowId"].values[1], [5, 6, 7, 9]
    )
    assert spw_df_2["assocNature"].values[0] == [
        SpectralResolutionType("BASEBAND_WIDE")
    ]


def test_exp_asdm_table_to_df_cache_not_shared_between_asdms():
    asdm_1 = make_asdm_with_fields([("first", 1.0, 0.5, 0)])
    asdm_2 = make_asdm_with_fields([("second", 2.0, -0.5, 0)])

    cols = ["fieldName", "phaseDir"]
    field_df_1 = exp_asdm_table_to_df(asdm_1, "Field", cols)
    field_df_2 = exp_asdm_table_to_df(asdm_2, "Field", cols)
    assert field_df_1["fieldName"].tolist() == ["first"]
    assert field_df_2["fieldName"].tolist() == ["second"]
    assert field_df_2["phaseDir"].tolist() == [[[2.0, -0.5]]]
    assert exp_asdm_table_to_df(asdm_1, "Field", cols)["fieldName"].tolist() == [
        "first"
    ]

    # A copy of an ASDM (here made after converting its tables) has its own entries
    asdm_copy = copy.deepcopy(asdm_1)
    add_field_row(asdm_copy, "added_to_copy", 0.5, 0.25)
    asdm_copy.getField().get()[0].setFieldName("renamed_in_copy")
    assert exp_asdm_table_to_df(asdm_copy, "Field", cols)["fieldName"].tolist() == [
        "renamed_in_copy",
        "added_to_copy",
    ]
    assert exp_asdm_table_to_df(asdm_1, "Field", cols)["fieldName"].tolist() == [
        "first"
    ]


def test_exp_asdm_table_to_df_cache_invalidated_by_new_rows(count_conversions):
    asdm = make_asdm_with_fields([("A", 1.0, 0.5, 0)])

    assert exp_asdm_table_to_df(asdm, "Field", ["fieldName"])["fieldName"].tolist() == [
        "A"
    ]
    add_field_row(asdm, "B", 2.0, -0.5)
    assert exp_asdm_table_to_df(asdm, "Field", ["fieldName"])["fieldName"].tolist() == [
        "A",
        "B",
    ]
    assert count_conversions == ["fieldName", "fieldName"]


def test_exp_asdm_table_to_df_cache_does_not_keep_asdm_alive():
    asdm = make_asdm_with_fields([("A", 1.0, 0.5, 0)])
    add_polarization_row(asdm, ["XX", "YY"])
    exp_asdm_table_to_df(asdm, "Field", ["fieldId", "fieldName", "phaseDir", "time"])
    exp_asdm_table_to_df(asdm, "Polarization", ["corrType", "corrProduct"])
    exp_asdm_table_to_df(asdm, "Main", ["fieldId"])
    assert asdm in metadata_tables._asdm_table_cache
    num_cached_asdms = len(metadata_tables._asdm_table_cache)

    asdm_ref = weakref.ref(asdm)
    del asdm
    gc.collect()
    # The cache does not keep the ASDM alive, and its entry goes away with it
    assert asdm_ref() is None
    assert len(metadata_tables._asdm_table_cache) <= num_cached_asdms - 1


def test_exp_asdm_table_to_df_use_cache_false(count_conversions):
    asdm = make_asdm_with_fields([("A", 1.0, 0.5, 0)])

    for _ in range(2):
        field_df = exp_asdm_table_to_df(asdm, "Field", ["fieldName"], use_cache=False)
        assert field_df["fieldName"].tolist() == ["A"]
    assert count_conversions == ["fieldName", "fieldName"]
    assert asdm not in metadata_tables._asdm_table_cache


def test_clear_asdm_table_cache(count_conversions):
    asdm = make_asdm_with_fields([("A", 1.0, 0.5, 0)])
    other_asdm = make_asdm_with_fields([("other", 1.0, 0.5, 0)])
    exp_asdm_table_to_df(asdm, "Field", ["fieldName"])
    exp_asdm_table_to_df(other_asdm, "Field", ["fieldName"])

    # Changes to attributes of rows already converted are not detected...
    asdm.getField().get()[0].setFieldName("renamed")
    assert exp_asdm_table_to_df(asdm, "Field", ["fieldName"])["fieldName"].tolist() == [
        "A"
    ]

    # ... until the cache of that ASDM is cleared
    clear_asdm_table_cache(asdm)
    assert asdm not in metadata_tables._asdm_table_cache
    assert other_asdm in metadata_tables._asdm_table_cache
    assert exp_asdm_table_to_df(asdm, "Field", ["fieldName"])["fieldName"].tolist() == [
        "renamed"
    ]
    assert count_conversions == ["fieldName", "fieldName", "fieldName"]

    clear_asdm_table_cache()
    assert len(metadata_tables._asdm_table_cache) == 0


class _NoWeakrefASDM:
    """Stand-in ASDM that can be neither weakly referenced nor hashed."""

    __slots__ = ("_asdm",)
    __hash__ = None

    def __init__(self, asdm):
        self._asdm = asdm

    def getField(self):
        return self._asdm.getField()


def test_exp_asdm_table_to_df_uncacheable_sdm(count_conversions):
    asdm = make_asdm_with_fields([("A", 1.0, 0.5, 0), ("B", 2.0, -0.5, 1)])
    sdm = _NoWeakrefASDM(asdm)

    for _ in range(2):
        field_df = exp_asdm_table_to_df(sdm, "Field", ["fieldId", "fieldName"])
        pd.testing.assert_frame_equal(
            field_df, pd.DataFrame({"fieldId": [0, 1], "fieldName": ["A", "B"]})
        )
    assert count_conversions == ["fieldId", "fieldName"] * 2
    clear_asdm_table_cache(sdm)
