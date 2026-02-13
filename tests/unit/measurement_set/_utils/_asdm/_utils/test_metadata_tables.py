import pandas as pd

import pytest

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
    with pytest.raises(NameError, match="is not defined"):
        spw_id = load_asdm_col(asdm_with_spw_default.getSpectralWindow(), "assocNature")
    with pytest.raises(NameError, match="is not defined"):
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
    print(f"{main_df=}")
    pd.testing.assert_frame_equal(
        main_df, pd.DataFrame([], columns=cols, dtype="float64")
    )

    cols = ["spectralWindowId"]
    spw_df = exp_asdm_table_to_df(asdm_with_spw_simple, "SpectralWindow", cols)
    # assert spw_df == pd.DataFrame([[1]], columns=["spectralWindowId"])
    pd.testing.assert_frame_equal(
        spw_df, pd.DataFrame([[0], [1]], columns=["spectralWindowId"])
    )
