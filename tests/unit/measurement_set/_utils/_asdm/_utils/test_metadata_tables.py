import copy

import pandas as pd

import pytest

import pyasdm

from xradio.measurement_set._utils._asdm._utils.metadata_tables import (
    exp_asdm_table_to_df,
    load_asdm_col,
)


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


def add_feed_table(asdm: pyasdm.ASDM):
    feed_row_0_xml = """
  <row>
    <feedId> 0 </feedId>
    <timeInterval> 7226686294548387903 3993371484612775807 </timeInterval>
    <numReceptor> 2 </numReceptor>
    <beamOffset> 2 2 2 0.0 0.0 0.0 0.0  </beamOffset>
    <focusReference> 2 2 3 -99999.0 -99999.0 -99999.0 -99999.0 -99999.0 -99999.0  </focusReference>
    <polarizationTypes> 1 2 X Y</polarizationTypes>
    <polResponse> 2 2 2 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0  </polResponse>
    <receptorAngle> 1 2 -0.9346238144 0.6361725124  </receptorAngle>
    <antennaId> Antenna_0 </antennaId>
    <receiverId> 1 2 0 0  </receiverId>
    <spectralWindowId> SpectralWindow_0 </spectralWindowId>
  </row>
    """
    feed_table = asdm.getFeed()
    feed_row_0 = pyasdm.FeedRow(feed_table)
    feed_row_0.setFromXML(feed_row_0_xml)
    feed_table.add(feed_row_0)


def test_exp_asdm_table_to_df_asdm_with_spw_simple_plus_feed(asdm_with_spw_simple):
    asdm_with_feed = copy.deepcopy(asdm_with_spw_simple)
    add_feed_table(asdm_with_feed)

    cols = ["polarizationTypes"]
    feed_df = exp_asdm_table_to_df(asdm_with_feed, "Feed", cols)
    pd.testing.assert_frame_equal(
        feed_df,
        pd.DataFrame(
            [
                [
                    [
                        pyasdm.enumerations.PolarizationType.X,
                        pyasdm.enumerations.PolarizationType.Y,
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
    assert polarization_df["corrProduct"].values[0] == [["X", "X", "Y", "Y"]]
