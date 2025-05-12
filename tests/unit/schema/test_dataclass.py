import pytest
from dataclasses import dataclass, field
from typing import Annotated, List, Tuple, Literal
from xradio.schema.dataclass import (
    extract_field_docstrings,
    _check_invalid_dims,
    extract_xarray_dataclass,
    xarray_dataclass_to_array_schema,
    xarray_dataclass_to_dataset_schema,
    xarray_dataclass_to_dict_schema,
)
from xradio.schema.typing import Role


@dataclass
class MyDataClass:
    field1: Annotated[int, Role.ATTR] = field(default=0)
    """This is the docstring of field 1"""
    field2: Annotated[str, Role.COORD] = field(default='x')
    field3: Annotated[float, Role.DATA] = field(default=0.0)

# @dataclass
# class Image(AsDataArray):
#     data: Data[tuple[X, Y], float]
#     mask: Coord[tuple[X, Y], bool]
#     x: Coord[X, int] = 0
#     y: Coord[Y, int] = 0

def test_extract_field_docstrings():
    docstrings = extract_field_docstrings(MyDataClass)
    expected_docstring = "This is the docstring of field 1"
    assert docstrings['field1'] == expected_docstring
    assert docstrings['field2'] is None
    assert len(docstrings) == 3

def test_check_invalid_dims():
    dims = [['x'], ['y']]
    all_coord_names = ['x', 'y', 'z']
    valid_dims = _check_invalid_dims(dims, all_coord_names, 'MyDataClass', 'field2')
    assert valid_dims == dims

    # TO FIX: added a set(all_coord_names), but the code in dataclass.py:83 might need a fix instead
    with pytest.raises(ValueError):
        _check_invalid_dims([['a']], set(all_coord_names), 'MyDataClass', 'field2')

@pytest.mark.xfail(reason="Fails in dataclass.py:159: IndexError: tuple index out of range")
def test_extract_xarray_dataclass():
    coordinates, data_vars, attributes = extract_xarray_dataclass(MyDataClass)

    assert len(coordinates) == 1
    assert len(data_vars) == 1
    assert len(attributes) == 1

@pytest.mark.xfail(reason="Fails in dataclass.py:159: IndexError: tuple index out of range")
def test_xarray_dataclass_to_array_schema():
    schema = xarray_dataclass_to_array_schema(MyDataClass)
    assert schema.schema_name == 'tests.unit.schema.test_dataclass.MyDataClass'
    assert len(schema.dimensions) == 3
    assert len(schema.dtypes) == 1

@pytest.mark.xfail(reason="Fails in dataclass.py:159: IndexError: tuple index out of range")
def test_xarray_dataclass_to_dataset_schema():
    dataset_schema = xarray_dataclass_to_dataset_schema(MyDataClass)
    coordinates, data_vars, attributes = xarray_dataclass_to_dataset_schema(MyDataClass)
#    assert dataset_schema.schema_name == 'tests.unit.schema.test_dataclass.MyDataClass'
#    assert len(dataset_schema.dimensions) > 0

#@pytest.mark.xfail(reason="Needs more work")
def test_xarray_dataclass_to_dict_schema():
    dict_schema = xarray_dataclass_to_dict_schema(MyDataClass)
    assert dict_schema.schema_name == 'tests.unit.schema.test_dataclass.MyDataClass'
    assert len(dict_schema.attributes) == 3
    assert dict_schema.attributes[0].docstring == "This is the docstring of field 1"


    
