"""The data group dictionary and the main datasets of the measurement set v4
schema do not derive from the bases of ``xradio.schema.info``
(``GenericDataGroupDict`` and ``SchemaVersionAttrs``): base fields come
first, which would reorder their constructor parameters, the keys of new
data groups and the attributes of the JSON exports. These tests pin their
field order.
"""

import dataclasses

import pytest

import xradio.measurement_set.schema as ms_schema
from xradio._utils.schema import get_data_group_keys

FIELD_ORDER = {
    "DataGroupDict": [
        "correlated_data",
        "flag",
        "weight",
        "uvw",
        "field_and_source",
        "description",
        "date",
    ],
    "VisibilityXds": [
        "time",
        "baseline_id",
        "frequency",
        "polarization",
        "field_name",
        "scan_name",
        "VISIBILITY",
        "baseline_antenna1_name",
        "baseline_antenna2_name",
        "observation_info",
        "processor_info",
        "data_groups",
        "schema_version",
        "creator",
        "creation_date",
        "type",
        "polarization_mixed",
        "uvw_label",
        "FLAG",
        "WEIGHT",
        "UVW",
        "EFFECTIVE_INTEGRATION_TIME",
        "TIME_CENTROID",
        "TIME_CENTROID_EXTRA_PRECISION",
        "EFFECTIVE_CHANNEL_WIDTH",
        "FREQUENCY_CENTROID",
    ],
    "SpectrumXds": [
        "time",
        "antenna_name",
        "frequency",
        "polarization",
        "field_name",
        "scan_name",
        "SPECTRUM",
        "observation_info",
        "processor_info",
        "data_groups",
        "schema_version",
        "creator",
        "creation_date",
        "type",
        "polarization_mixed",
        "FLAG",
        "WEIGHT",
        "EFFECTIVE_INTEGRATION_TIME",
        "TIME_CENTROID",
        "TIME_CENTROID_EXTRA_PRECISION",
        "EFFECTIVE_CHANNEL_WIDTH",
        "FREQUENCY_CENTROID",
    ],
}


@pytest.mark.parametrize("name", sorted(FIELD_ORDER))
def test_measurement_set_field_order(name):
    fields = dataclasses.fields(getattr(ms_schema, name))
    assert [field.name for field in fields] == FIELD_ORDER[name]
    # All fields stay positional constructor parameters
    assert not any(field.kw_only for field in fields)


def test_measurement_set_data_group_key_order():
    # xr_ms.add_data_group lists the keys of a new data group in this order
    assert list(get_data_group_keys("msv4")) == FIELD_ORDER["DataGroupDict"]
