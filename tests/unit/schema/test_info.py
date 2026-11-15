import copy
from typing import Literal

import numpy
import pytest
import xarray

from xradio.schema.bases import (
    dict_schema,
    xarray_dataarray_schema,
    xarray_dataset_schema,
)
from xradio.schema.check import check_dataset
from xradio.schema.dataclass import (
    xarray_dataclass_to_array_schema,
    xarray_dataclass_to_dataset_schema,
)
from xradio.schema.info import check_data_groups, matching_data_var_names
from xradio.schema.typing import Attr, Coord, Data, Dataof

Dim = Literal["dim"]


@xarray_dataarray_schema
class _ValuesArray:
    data: Data[Dim, float]
    allow_multiple_versions: Attr[bool] | None = True


@xarray_dataarray_schema
class _FlagsArray:
    data: Data[Dim, bool]


@dict_schema
class _GroupDict:
    values: str
    flags: str | None
    node: str | None
    """Names something other than a data variable"""
    description: str | None


@dict_schema
class _GroupsDict:
    base: _GroupDict


@xarray_dataset_schema
class _GroupedXds:
    dim: Coord[Dim, float]
    VALUES: Dataof[_ValuesArray]
    FLAGS: Dataof[_FlagsArray] | None = None
    data_groups: Attr[_GroupsDict] = None


_ROLE_SCHEMAS = {"values": _ValuesArray, "flags": _FlagsArray}


def _dataset(**groups):
    base = {"values": "VALUES", "flags": "FLAGS", "node": "elsewhere"}
    return xarray.Dataset(
        {
            "VALUES": ("dim", numpy.zeros(3)),
            "VALUES_OTHER": ("dim", numpy.ones(3)),
            "FLAGS": ("dim", numpy.zeros(3, dtype=bool)),
        },
        coords={"dim": numpy.arange(3.0)},
        attrs={"data_groups": {"base": base, **copy.deepcopy(groups)}},
    )


def _issues(xds, **kwargs):
    return [
        (issue.path, issue.message)
        for issue in check_data_groups(xds, _GroupedXds, _ROLE_SCHEMAS, **kwargs)
    ]


def test_check_data_groups_valid():
    xds = _dataset(other={"values": "VALUES_OTHER", "description": "text"})
    assert not check_dataset(xds, _GroupedXds)
    assert not check_data_groups(xds, _GroupedXds, _ROLE_SCHEMAS)
    schema = xarray_dataclass_to_dataset_schema(_GroupedXds)
    assert not check_data_groups(xds, schema, _ROLE_SCHEMAS)


def test_check_data_groups_group_schema():
    """Every group but 'base', which check_dataset checks, conforms to the
    data group schema"""
    xds = _dataset(other={"flags": "FLAGS", "description": 3})
    assert _issues(xds) == [
        (
            [("attrs", "data_groups['other']"), ("", "values")],
            "Non-optional attribute is missing!",
        ),
        ([("attrs", "data_groups['other']"), ("", "description")], "int is not a str!"),
    ]
    del xds.attrs["data_groups"]["base"]["values"]
    assert len(_issues(xds)) == 2
    (issue,) = check_dataset(xds, _GroupedXds)
    assert issue.path == [("attrs", "data_groups"), ("", "base"), ("", "values")]


def test_check_data_groups_not_a_dictionary():
    xds = _dataset(other="VALUES_OTHER")
    assert _issues(xds) == [
        ([("attrs", "data_groups")], "Data group 'other' is not a dictionary!")
    ]


@pytest.mark.parametrize("data_groups", [None, "not a dict"])
def test_check_data_groups_without_data_groups(data_groups):
    """A missing or wrong-typed data_groups is left to check_dataset"""
    xds = _dataset()
    xds.attrs["data_groups"] = data_groups
    assert not check_data_groups(xds, _GroupedXds, _ROLE_SCHEMAS)


def test_check_data_groups_missing_variable():
    xds = _dataset(other={"values": "VALUES_NOT_THERE"})
    xds.attrs["data_groups"]["base"]["flags"] = "FLAGS_NOT_THERE"
    assert _issues(xds) == [
        (
            [("data_vars", "FLAGS_NOT_THERE")],
            "Data group 'base' role 'flags' references a missing data variable!",
        ),
        (
            [("data_vars", "VALUES_NOT_THERE")],
            "Data group 'other' role 'values' references a missing data variable!",
        ),
    ]


def test_check_data_groups_missing_required_variable():
    """A variable that check_dataset reports as a missing required data
    variable is not reported again for the groups that name it"""
    xds = _dataset(other={"values": "VALUES"}).drop_vars(["VALUES", "VALUES_OTHER"])
    (issue,) = check_dataset(xds, _GroupedXds)
    assert issue.path == [("data_vars", "VALUES")]
    assert not check_data_groups(xds, _GroupedXds, _ROLE_SCHEMAS)

    # A version of VALUES satisfies check_dataset, so the groups that name
    # the missing VALUES are reported
    xds = _dataset(other={"values": "VALUES"}).drop_vars("VALUES")
    assert not check_dataset(xds, _GroupedXds)
    assert _issues(xds) == [
        (
            [("data_vars", "VALUES")],
            f"Data group '{group}' role 'values' references a missing data variable!",
        )
        for group in ("base", "other")
    ]


def test_check_data_groups_variable_checked_once():
    """A variable that check_dataset does not match by name is checked
    against the array schema of its role, once"""
    xds = _dataset(
        one={"values": "VALUES", "flags": "MY_FLAGS"},
        two={"values": "VALUES", "flags": "MY_FLAGS"},
    )
    xds["MY_FLAGS"] = ("dim", numpy.zeros(3, dtype=numpy.int32))
    assert _issues(xds) == [
        ([("data_vars", "MY_FLAGS"), ("dtype", None)], "Wrong numpy dtype")
    ]


def test_check_data_groups_leaves_variables_of_check_dataset():
    """A variable that check_dataset checks (by its name) is not checked again"""
    xds = _dataset(other={"values": "VALUES_OTHER"})
    xds["VALUES_OTHER"] = ("dim", numpy.ones(3, dtype=numpy.int32))
    assert not check_data_groups(xds, _GroupedXds, _ROLE_SCHEMAS)
    (issue,) = check_dataset(xds, _GroupedXds)
    assert issue.path == [("data_vars", "VALUES_OTHER"), ("dtype", None)]


def test_check_data_groups_checked():
    xds = _dataset(other={"values": "VALUES", "flags": "MY_FLAGS"})
    xds["MY_FLAGS"] = ("dim", numpy.zeros(3, dtype=numpy.int32))
    flags_schema_name = xarray_dataclass_to_array_schema(_FlagsArray).schema_name
    checked = {flags_schema_name: {"MY_FLAGS"}}
    assert not check_data_groups(xds, _GroupedXds, _ROLE_SCHEMAS, checked=checked)
    checked = {}
    assert _issues(xds, checked=checked)
    assert checked == {
        flags_schema_name: {"FLAGS", "MY_FLAGS"},
        xarray_dataclass_to_array_schema(_ValuesArray).schema_name: {"VALUES"},
    }


def test_check_data_groups_unknown_roles():
    xds = _dataset(other={"values": "VALUES", "valuez": "VALUES"})
    assert _issues(xds) == [
        ([("attrs", "data_groups['other']")], "Unknown data group role 'valuez'!")
    ]
    assert not check_data_groups(
        xds, _GroupedXds, _ROLE_SCHEMAS, report_unknown_roles=False
    )


def test_check_data_groups_keys_without_role_schema():
    """Keys that name something other than a data variable are not resolved"""
    xds = _dataset(other={"values": "VALUES", "node": "not a variable"})
    assert not _issues(xds)


def test_check_data_groups_schema_without_data_groups():
    @xarray_dataset_schema
    class _UngroupedXds:
        dim: Coord[Dim, float]

    with pytest.raises(TypeError, match="no 'data_groups' attribute"):
        check_data_groups(_dataset(), _UngroupedXds, _ROLE_SCHEMAS)


def test_check_data_groups_not_a_dataset_schema():
    """As for check_dataset, with the name of check_data_groups"""
    for schema in (_ValuesArray, _GroupDict, "_GroupedXds"):
        with pytest.raises(TypeError, match="^check_data_groups: Expected Dataset"):
            check_data_groups(_dataset(), schema, _ROLE_SCHEMAS)


def test_matching_data_var_names():
    schema = xarray_dataclass_to_dataset_schema(_GroupedXds)
    values, flags = schema.data_vars
    names = ["VALUES", "VALUES_OTHER", "MY_VALUES", "FLAGS", "FLAGS_OTHER"]
    assert matching_data_var_names(names, values) == ["VALUES", "VALUES_OTHER"]
    assert matching_data_var_names(names, flags) == ["FLAGS"]
    assert matching_data_var_names(["VALUES"], flags) == []
