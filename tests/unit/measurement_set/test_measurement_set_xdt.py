import copy

import numpy as np
import pytest
import xarray as xr

from xradio.measurement_set.measurement_set_xdt import (
    InvalidAccessorLocation,
    MeasurementSetXdt,
)
from xradio.measurement_set.schema import FieldSourceXds, VisibilityXds
from xradio.schema.check import check_dataset, check_datatree

# starting point for measurement_set_xdt unit tests. Some additional test cases added for clear
# coverage gaps, but several more still missing for better systematic coverage w/o relying on stk tests


def test_sel_invalid():
    ms_xdt = MeasurementSetXdt(xr.DataTree())

    with pytest.raises(InvalidAccessorLocation, match="not a MSv4 node"):
        assert ms_xdt.sel()


def test_get_field_and_source_xds_invalid():
    ms_xdt = MeasurementSetXdt(xr.DataTree())

    with pytest.raises(InvalidAccessorLocation, match="not a MSv4 node"):
        assert ms_xdt.get_field_and_source_xds()


def test_get_partition_info():
    ms_xdt = MeasurementSetXdt(xr.DataTree())

    with pytest.raises(InvalidAccessorLocation, match="not a MSv4 node"):
        partition_info = ms_xdt.get_partition_info()
        assert isinstance(partition_info, dict)
        assert all(
            [
                key in partition_info
                for key in [
                    "spectral_window_name",
                    "spectral_window_intents",
                    "field_name",
                    "polarization_setup",
                    "scan_name",
                    "source_name",
                    "scan_intents",
                    "line_name",
                ]
            ]
        )


def test_add_data_group_with_defaults(msv4_xdt_min):
    ms_xdt = msv4_xdt_min.xr_ms
    result_xdt = ms_xdt.add_data_group("test_added_data_group_with_defaults")

    check_dataset(result_xdt.ds, VisibilityXds)
    check_datatree(result_xdt)


def test_add_data_group_with_values(msv4_xdt_min):
    ms_xdt = msv4_xdt_min.xr_ms
    result_xdt = ms_xdt.add_data_group(
        "test_added_data_group_with_param_values",
        {
            "correlated_data": "VISIBILITY",
            "weight": "EFFECTIVE_INTEGRATION_TIME",  # no check for this (coords mismatch, etc.)
            "flag": "FLAG",
            "uvw": "UVW",
            "field_and_source_xds": "field_and_source_base_xds",
            "date_time": "today, now",
            "description": "a test data group",
        },
        data_group_dv_shared_with="base",
    )

    check_dataset(result_xdt.ds, VisibilityXds)
    check_datatree(result_xdt)


def test_get_field_and_source_xds(msv4_xdt_min):
    result_xdt = msv4_xdt_min.xr_ms.get_field_and_source_xds()
    check_dataset(result_xdt, FieldSourceXds)


def test_get_field_and_source_xds_with_group(msv4_xdt_min):
    result_xdt = msv4_xdt_min.xr_ms.get_field_and_source_xds(data_group_name="base")
    check_dataset(result_xdt, FieldSourceXds)


expected_partition_info_fields = [
    "spectral_window_name",
    "field_name",
    "polarization_setup",
    "scan_name",
    "source_name",
    "scan_intents",
    "line_name",
    "data_group_name",
]


def test_get_partition_info_default(msv4_xdt_min):
    partition_info = msv4_xdt_min.xr_ms.get_partition_info()
    assert isinstance(partition_info, dict)
    for field in expected_partition_info_fields:
        assert field in partition_info


def test_get_partition_info_with_group_wrong(msv4_xdt_min):
    wrong_group_name = "missing"
    with pytest.raises(KeyError, match=wrong_group_name):
        _partition_info = msv4_xdt_min.xr_ms.get_partition_info(
            data_group_name=wrong_group_name
        )


def test_get_partition_info_with_group(msv4_xdt_min):
    partition_info = msv4_xdt_min.xr_ms.get_partition_info(data_group_name="base")
    assert isinstance(partition_info, dict)
    for field in expected_partition_info_fields:
        assert field in partition_info


def test_sel_with_data_group_missing(msv4_xdt_min):
    wrong_group_name = "corrected"
    with pytest.raises(KeyError, match=wrong_group_name):
        _result_xds = msv4_xdt_min.xr_ms.sel(data_group_name=wrong_group_name)


def test_sel_with_data_group(msv4_xdt_min):
    result_xdt = msv4_xdt_min.xr_ms.sel(data_group_name="base")
    check_dataset(result_xdt.ds, VisibilityXds)


def test_sel_polarization(msv4_xdt_min):
    result_xdt = msv4_xdt_min.xr_ms.sel(polarization="XX")
    check_dataset(result_xdt.ds, VisibilityXds)


def _msv4_tree():
    """A small in-memory MSv4-like tree with two data groups that share the
    flags, weights and uvw."""
    dims = ("time", "baseline_id", "frequency", "polarization")
    shape = (2, 3, 4, 2)
    group = {
        "flag": "FLAG",
        "weight": "WEIGHT",
        "uvw": "UVW",
        "field_and_source": "field_and_source_base_xds",
        "description": "",
        "date": "2000-01-01T00:00:00.000",
    }
    xds = xr.Dataset(
        {
            "VISIBILITY": (dims, np.zeros(shape, complex)),
            "VISIBILITY_CORRECTED": (dims, np.ones(shape, complex)),
            "FLAG": (dims, np.zeros(shape, bool)),
            "WEIGHT": (dims, np.ones(shape)),
            "UVW": (("time", "baseline_id", "uvw_label"), np.zeros((2, 3, 3))),
        },
        coords={
            "time": [0.0, 1.0],
            "baseline_id": [0, 1, 2],
            "frequency": [1.0e9, 1.1e9, 1.2e9, 1.3e9],
            "polarization": ["XX", "YY"],
            "uvw_label": ["u", "v", "w"],
        },
        attrs={
            "type": "visibility",
            "data_groups": {
                "base": {"correlated_data": "VISIBILITY", **group},
                "corrected": {"correlated_data": "VISIBILITY_CORRECTED", **group},
            },
        },
    )
    field_and_source = xr.Dataset({"FIELD_PHASE_CENTER": ("sky_dir_label", [0.0, 0.0])})
    return xr.DataTree.from_dict(
        {"/": xds, "/field_and_source_base_xds": field_and_source}, name="ms"
    )


_DERIVED_TREES = [
    pytest.param(lambda xdt: xdt.isel(time=[0]), id="isel"),
    pytest.param(lambda xdt: xdt.sel(polarization=["XX"]), id="sel"),
    pytest.param(lambda xdt: xdt.copy(deep=False), id="shallow_copy"),
    pytest.param(
        lambda xdt: xdt.xr_ms.sel(data_group_name="corrected"), id="xr_ms_sel"
    ),
]


class TestDataGroupsAreNotShared:
    """Deleting variables, selecting a data group or adding one changes only
    the DataTree node the accessor is called on."""

    def test_delete_removes_the_variable_from_the_tree(self):
        xdt = _msv4_tree()

        result = xdt.xr_ms.delete_data_variables(["VISIBILITY_CORRECTED"])

        assert result is xdt
        assert "VISIBILITY_CORRECTED" not in xdt.data_vars
        assert "VISIBILITY_CORRECTED" not in xdt.ds.data_vars
        assert "VISIBILITY_CORRECTED" not in xdt.to_dataset().data_vars
        assert "correlated_data" not in xdt.attrs["data_groups"]["corrected"]
        assert xdt.attrs["data_groups"]["base"]["correlated_data"] == "VISIBILITY"
        assert "field_and_source_base_xds" in xdt.children

    def test_delete_ignores_unknown_names_and_accepts_a_string(self):
        xdt = _msv4_tree()

        xdt.xr_ms.delete_data_variables(["NOT_THERE"])
        xdt.xr_ms.delete_data_variables("FLAG")

        assert "FLAG" not in xdt.data_vars
        assert all("flag" not in g for g in xdt.attrs["data_groups"].values())

    @pytest.mark.parametrize("derive", _DERIVED_TREES)
    def test_delete_from_derived_keeps_parent(self, derive):
        parent = _msv4_tree()
        expected = copy.deepcopy(parent.attrs["data_groups"])
        derived = derive(parent)

        derived.xr_ms.delete_data_variables(["FLAG"])

        assert "FLAG" not in derived.data_vars
        assert all("flag" not in g for g in derived.attrs["data_groups"].values())
        assert "FLAG" in parent.data_vars
        assert parent.attrs["data_groups"] == expected

    @pytest.mark.parametrize("derive", _DERIVED_TREES)
    def test_delete_from_parent_keeps_derived(self, derive):
        parent = _msv4_tree()
        derived = derive(parent)
        expected = copy.deepcopy(derived.attrs["data_groups"])

        parent.xr_ms.delete_data_variables(["FLAG", "WEIGHT"])

        assert "FLAG" in derived.data_vars
        assert derived.attrs["data_groups"] == expected

    def test_sel_data_group_leaves_the_tree_unchanged(self):
        xdt = _msv4_tree()
        expected_groups = copy.deepcopy(xdt.attrs["data_groups"])
        indexers = {"data_group_name": "base", "polarization": "XX"}

        selected = xdt.xr_ms.sel(indexers)

        assert selected is not xdt
        assert "VISIBILITY_CORRECTED" not in selected.data_vars
        assert "polarization" not in selected.dims
        assert set(selected.attrs["data_groups"]) == {"base"}
        assert "field_and_source_base_xds" in selected.children
        # The caller's tree, its data groups and its indexers are untouched
        assert "VISIBILITY_CORRECTED" in xdt.data_vars
        assert xdt.sizes["polarization"] == 2
        assert xdt.attrs["data_groups"] == expected_groups
        assert indexers == {"data_group_name": "base", "polarization": "XX"}

    def test_sel_data_group_of_a_processing_set_child(self):
        ps = xr.DataTree.from_dict({"/ms_0": _msv4_tree()})
        ps.attrs["type"] = "processing_set"

        selected = ps["ms_0"].xr_ms.sel(data_group_name="corrected")

        assert "VISIBILITY" not in selected.data_vars
        assert "VISIBILITY" in ps["ms_0"].data_vars
        assert set(ps["ms_0"].attrs["data_groups"]) == {"base", "corrected"}

    @pytest.mark.parametrize("derive", _DERIVED_TREES)
    def test_add_data_group_keeps_other_trees(self, derive):
        parent = _msv4_tree()
        derived = derive(parent)
        parent_groups = set(parent.attrs["data_groups"])
        derived_groups = set(derived.attrs["data_groups"])

        derived.xr_ms.add_data_group("extra", {"correlated_data": "VISIBILITY"})

        assert set(derived.attrs["data_groups"]) == derived_groups | {"extra"}
        assert set(parent.attrs["data_groups"]) == parent_groups


if __name__ == "__main__":
    pytest.main(["-v", "-s", __file__])
