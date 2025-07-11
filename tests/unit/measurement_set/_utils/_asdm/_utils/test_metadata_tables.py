import pytest

from xradio.measurement_set._utils._asdm._utils.metadata_tables import (
    exp_asdm_table_to_df,
    load_asdm_col,
)


def test_load_asdm_col_empty():
    with pytest.raises(AttributeError, match="no attribute"):
        load_asdm_col(None, "fieldId")


def test_exp_asdm_col_empty():
    with pytest.raises(AttributeError, match="no attribute"):
        exp_asdm_table_to_df(None, "Main", "fieldId")
