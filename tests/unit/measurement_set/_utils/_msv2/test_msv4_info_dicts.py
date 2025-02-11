import pytest

import xarray as xr

from xradio.measurement_set.schema import (
    ObservationInfoDict,
    PartitionInfoDict,
    ProcessorInfoDict,
)
from xradio.schema.check import check_dict


def test_create_info_dicts_empty(ms_empty_required):
    from xradio.measurement_set._utils._msv2.msv4_info_dicts import create_info_dicts
    from xradio.measurement_set._utils._msv2._tables.table_query import open_query

    msv4_xds = xr.Dataset(
        data_vars={"FLAG": ("frequency", [True, False])},
        coords={
            "frequency": (
                "frequency",
                [0, 1],
                {"spectral_window_name": "test_SPW_name"},
            ),
            "polarization": ("polarization", ["XX", "YY"]),
        },
    )
    field_and_source_xds = xr.Dataset(
        # data_vars={"FLAG": ("frequency", [True, False])},
        coords={
            "field_name": ("field_name", ["test_field", "test_field"]),
            "source_name": ("source_name", ["test_source", "test_source"]),
            "line_name": ("line_label", ["test_line_name"]),
        }
    )
    partition_info_misc_fields = {
        "scan_id": 3,
        "intents": "intent_str#subintent_str",
        "taql_where": "test_TAQL_str",
    }

    ms_main = ms_empty_required.fname
    taql_main = f"select * from $ms_main WHERE (OBSERVATION_ID = 0) and (DATA_DESC_ID = 0) and (FIELD_ID = 0)"
    with pytest.raises(AssertionError, match="is not consistent"):
        with open_query(ms_main, taql_main) as tb_tool:

            info_dicts = create_info_dicts(
                ms_empty_required.fname,
                msv4_xds,
                field_and_source_xds,
                partition_info_misc_fields,
                tb_tool,
            )
        # assert info_dicts


def test_create_info_dicts(ms_minimal_required):
    from xradio.measurement_set._utils._msv2.msv4_info_dicts import create_info_dicts
    from xradio.measurement_set._utils._msv2._tables.table_query import open_query

    msv4_xds = xr.Dataset(
        data_vars={"FLAG": ("frequency", [True, False])},
        coords={
            "frequency": (
                "frequency",
                [0, 1],
                {"spectral_window_name": "test_SPW_name"},
            ),
            "polarization": ("polarization", ["XX", "YY"]),
        },
    )
    field_and_source_xds = xr.Dataset(
        # data_vars={"FLAG": ("frequency", [True, False])},
        coords={
            "field_name": ("field_name", ["test_field", "test_field"]),
            "source_name": ("source_name", ["test_source", "test_source"]),
            "line_name": ("line_label", ["test_line_name"]),
        }
    )
    partition_info_misc_fields = {
        "scan_id": 3,
        "intents": "intent_str#subintent_str",
        "taql_where": "test_TAQL_str",
    }

    ms_main = ms_minimal_required.fname
    taql_main = f"select * from $ms_main WHERE (OBSERVATION_ID = 0) and (DATA_DESC_ID = 0) and (FIELD_ID = 0)"

    with open_query(ms_main, taql_main) as tb_tool:

        info_dicts = create_info_dicts(
            ms_minimal_required.fname,
            msv4_xds,
            field_and_source_xds,
            partition_info_misc_fields,
            tb_tool,
        )

        assert isinstance(info_dicts, dict)
        check_dict(info_dicts["partition_info"], PartitionInfoDict)
        check_dict(info_dicts["observation_info"], ObservationInfoDict)
        check_dict(info_dicts["processor_info"], ProcessorInfoDict)
