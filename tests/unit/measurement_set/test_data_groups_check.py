"""check_dataset and check_datatree check every data group of a measurement
set (not only 'base') and the data variables that the groups name."""

import copy

import numpy as np
import pytest
import xarray as xr

from xradio.measurement_set.schema import (
    SpectrumArray,
    SpectrumXds,
    VisibilityXds,
    WeightArray,
)
from xradio.schema.check import (
    _DATASET_CHECKS,
    check_array,
    check_dataset,
    check_datatree,
)
from xradio.schema.dataclass import xarray_dataclass_to_dataset_schema
from xradio.schema.export import export_schema_json_file, import_schema_json_file
from xradio.testing.measurement_set.msv2_io import build_minimal_msv4_xdt


@pytest.fixture(scope="module")
def msv4_tree(ms_minimal_required, tmp_path_factory):
    """A minimal MSv4 of this module's own, whose main dataset checks clean
    (other tests change the attributes of the session-wide msv4_xdt_min)"""
    path = build_minimal_msv4_xdt(
        ms_minimal_required.fname,
        out_root=tmp_path_factory.mktemp("msv4_data_groups_check"),
        msv4_id="msv4id",
        partition_kwargs={
            "DATA_DESC_ID": [0],
            "OBS_MODE": ["CAL_ATMOSPHERE#ON_SOURCE"],
        },
    )
    tree = xr.open_datatree(path, engine="zarr")
    assert not check_dataset(tree.ds, VisibilityXds)
    return tree


def _group(**roles):
    """A complete data group, as the converter writes them, with roles
    replaced (or removed, when given as None)"""
    group = {
        "correlated_data": "VISIBILITY",
        "flag": "FLAG",
        "weight": "WEIGHT",
        "uvw": "UVW",
        "field_and_source": "field_and_source_base_xds",
        "description": "a test data group",
        "date": "2026-11-15T00:00:00.000000+00:00",
    }
    group.update(roles)
    return {role: name for role, name in group.items() if name is not None}


def _with_groups(msv4_tree, **groups):
    """The main dataset of the minimal MSv4 with data groups added"""
    xds = msv4_tree.ds.copy()
    xds.attrs = copy.deepcopy(xds.attrs)
    xds.attrs["data_groups"].update(copy.deepcopy(groups))
    return xds


def _issues(xds, schema=VisibilityXds):
    return [(issue.path, issue.message) for issue in check_dataset(xds, schema)]


def test_complete_data_groups(msv4_tree):
    xds = _with_groups(
        msv4_tree,
        corrected=_group(correlated_data="VISIBILITY_CORRECTED"),
        model=_group(correlated_data="VISIBILITY_MODEL", uvw=None),
    )
    xds["VISIBILITY_CORRECTED"] = xds["VISIBILITY"]
    xds["VISIBILITY_MODEL"] = xds["VISIBILITY"]
    assert not check_dataset(xds, VisibilityXds)


def test_broken_non_base_group(msv4_tree):
    """A group other than 'base' that lacks keys of DataGroupDict passed the
    check before, and fails it now"""
    xds = _with_groups(
        msv4_tree,
        imaging={"correlated_data": "VISIBILITY", "flag": "FLAG", "weight": "WEIGHT"},
    )
    issues = check_dataset(xds, VisibilityXds)
    assert {issue.message for issue in issues} == {"Non-optional attribute is missing!"}
    # One issue per missing key
    assert sorted(issue.path for issue in issues) == sorted(
        [("attrs", "data_groups['imaging']"), ("", key)]
        for key in ("field_and_source", "description", "date")
    )


def test_group_names_a_missing_variable(msv4_tree):
    """In any group, also 'base', whose names were not resolved before"""
    xds = _with_groups(msv4_tree, corrected=_group(weight="WEIGHT_NOT_THERE"))
    xds.attrs["data_groups"]["base"]["uvw"] = "UVW_NOT_THERE"
    assert _issues(xds) == [
        (
            [("data_vars", "UVW_NOT_THERE")],
            "Data group 'base' role 'uvw' references a missing data variable!",
        ),
        (
            [("data_vars", "WEIGHT_NOT_THERE")],
            "Data group 'corrected' role 'weight' references a missing data variable!",
        ),
    ]


def test_missing_required_variable_reported_once(msv4_tree):
    """A required variable that is missing is reported by the structural
    check only, not again for each group that names it"""
    xds = _with_groups(msv4_tree, corrected=_group())
    del xds["FLAG"]
    assert _issues(xds) == [
        (
            [("data_vars", "FLAG")],
            "Required data variable 'FLAG' is missing "
            f"(have {','.join(xds.data_vars)})!",
        )
    ]


def test_group_names_a_variable_of_the_wrong_kind(msv4_tree):
    """A variable whose name check_dataset does not match is checked against
    the schema of the role, once, also when several groups name it"""
    xds = _with_groups(
        msv4_tree,
        one=_group(weight="EFFECTIVE_INTEGRATION_TIME"),
        two=_group(weight="EFFECTIVE_INTEGRATION_TIME"),
    )
    expected = check_array(xds["EFFECTIVE_INTEGRATION_TIME"], WeightArray).at_path(
        "data_vars", "EFFECTIVE_INTEGRATION_TIME"
    )
    assert expected
    assert _issues(xds) == [(issue.path, issue.message) for issue in expected]


def test_variable_checked_once(msv4_tree):
    """A version of a variable that check_dataset matches by name (WEIGHT_*)
    is not checked again for the group that names it"""
    xds = _with_groups(msv4_tree, imaging=_group(weight="WEIGHT_IMAGING"))
    xds["WEIGHT_IMAGING"] = xds["WEIGHT"].astype(np.int32)
    assert _issues(xds) == [
        ([("data_vars", "WEIGHT_IMAGING"), ("dtype", None)], "Wrong numpy dtype")
    ]


def test_other_keys_are_allowed(msv4_tree):
    """Keys that DataGroupDict does not define are allowed in every group (as
    in 'base'), for example roles that applications add, and
    field_and_source is not resolved"""
    xds = _with_groups(
        msv4_tree,
        imaging=_group(weight_imaging="WEIGHT_IMAGING", field_and_source="elsewhere"),
    )
    xds["WEIGHT_IMAGING"] = xds["WEIGHT"]
    xds.attrs["data_groups"]["base"]["weight_imaging"] = "WEIGHT_IMAGING"
    assert not check_dataset(xds, VisibilityXds)


def test_schema_imported_from_json(msv4_tree, tmp_path):
    """The check is found by the schema name, so it also runs for the schema
    imported from JSON (xradio.measurement_set.schema is imported here)"""
    export_schema_json_file(VisibilityXds, tmp_path / "VisibilityXds.json")
    schema = import_schema_json_file(tmp_path / "VisibilityXds.json")
    xds = _with_groups(msv4_tree, corrected=_group(correlated_data="VIS_NOT_THERE"))
    assert (
        _issues(xds, schema)
        == _issues(xds)
        == [
            (
                [("data_vars", "VIS_NOT_THERE")],
                "Data group 'corrected' role 'correlated_data' references a "
                "missing data variable!",
            )
        ]
    )


def test_check_datatree(msv4_tree):
    before = {repr(issue) for issue in check_datatree(msv4_tree)}
    xdt = msv4_tree.copy(deep=False)
    xdt.attrs = copy.deepcopy(dict(xdt.attrs))
    xdt.attrs["data_groups"]["corrected"] = _group(correlated_data="VIS_NOT_THERE")
    new = [issue for issue in check_datatree(xdt) if repr(issue) not in before]
    assert [(issue.path, issue.message) for issue in new] == [
        (
            [("", "/"), ("data_vars", "VIS_NOT_THERE")],
            "Data group 'corrected' role 'correlated_data' references a missing "
            "data variable!",
        )
    ]


def test_deleting_the_correlated_data_of_a_group(msv4_tree):
    """delete_data_variables removes the roles that name a deleted variable,
    which leaves a group without its correlated data: the check reports it"""
    xdt = msv4_tree.copy(deep=False)
    xdt["VISIBILITY_CORRECTED"] = xdt["VISIBILITY"]
    xdt.attrs = copy.deepcopy(dict(xdt.attrs))
    xdt.attrs["data_groups"]["corrected"] = _group(
        correlated_data="VISIBILITY_CORRECTED"
    )
    assert not check_dataset(xdt.ds, VisibilityXds)
    xdt.xr_ms.delete_data_variables(["VISIBILITY_CORRECTED"])
    assert _issues(xdt.ds) == [
        (
            [("attrs", "data_groups['corrected']"), ("", "correlated_data")],
            "Non-optional attribute is missing!",
        )
    ]


def test_spectrum_data_groups():
    """Single-dish groups: the correlated data are spectra, and a group may
    not name UVW, which single-dish data do not have"""
    dims = ("time", "antenna_name", "frequency", "polarization")
    shape = (2, 3, 4, 2)
    group = {
        "correlated_data": "SPECTRUM",
        "flag": "FLAG",
        "weight": "WEIGHT",
        "field_and_source": "field_and_source_base_xds",
        "description": "",
        "date": "2026-11-15T00:00:00.000000+00:00",
    }
    xds = xr.Dataset(
        {
            "SPECTRUM": (dims, np.ones(shape), {"units": "Jy"}),
            "SPECTRUM_MODEL": (dims, np.ones(shape, np.complex64), {"units": "Jy"}),
            "FLAG": (dims, np.zeros(shape, bool)),
            "WEIGHT": (dims, np.ones(shape)),
        },
        attrs={
            "data_groups": {
                "base": group,
                "model": {**group, "correlated_data": "SPECTRUM_MODEL", "uvw": "UVW"},
            }
        },
    )
    check = _DATASET_CHECKS[xarray_dataclass_to_dataset_schema(SpectrumXds).schema_name]
    model_issues = check_array(xds["SPECTRUM_MODEL"], SpectrumArray).at_path(
        "data_vars", "SPECTRUM_MODEL"
    )
    assert [(issue.path, issue.message) for issue in check(xds)] == [
        (issue.path, issue.message) for issue in model_issues
    ] + [
        (
            [("data_vars", "UVW")],
            "Data group 'model' role 'uvw' references a missing data variable!",
        )
    ]
    assert any(issue.path[-1] == ("dtype", None) for issue in model_issues)
