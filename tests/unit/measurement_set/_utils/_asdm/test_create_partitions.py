import pandas as pd

import pytest

from xradio.measurement_set._utils._asdm.create_partitions import (
    create_partitions,
    finalize_partitions_groupby,
)


def test_create_partitions_empty():
    with pytest.raises(AttributeError, match="no attribute"):
        create_partitions(None, ["fieldId"])


def test_create_partitions_asdm_empty(asdm_empty):
    with pytest.raises(IndexError, match="out-of-bounds"):
        create_partitions(asdm_empty, ["fieldId"])


def test_create_partitions_asdm_with_spw_default(asdm_with_spw_default):
    with pytest.raises(IndexError, match="out-of-bounds"):
        create_partitions(asdm_with_spw_default, ["fieldId"])


def test_create_partitions_asdm_with_spw_simple(asdm_with_spw_simple):
    with pytest.raises(IndexError, match="out-of-bounds"):
        create_partitions(asdm_with_spw_simple, ["fieldId"])


def test_finalize_partitions_gropuby():
    with pytest.raises(TypeError, match="scalar index"):
        finalize_partitions_groupby(
            pd.DataFrame([[0, 0]], columns=["fieldId", "scanIntent"]),
            ["fieldId"],
            [0],
        )
