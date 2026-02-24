import dataclasses
from typing import Literal, Optional, Union
import numpy
import xarray
import dask.array
import pytest
import inspect
import json

from xradio.schema.typing import Attr, Coord, Coordof, Data, Dataof, Name
from xradio.schema.metamodel import (
    ArraySchema,
    ArraySchemaRef,
    AttrSchemaRef,
    DatasetSchema,
    DictSchema,
)
from xradio.schema.check import (
    check_array,
    check_dataset,
    check_dimensions,
    check_dict,
    schema_checked,
    register_dataset_type,
    check_datatree,
    SchemaIssue,
    SchemaIssues,
    _check_value,
)
from xradio.schema.dataclass import (
    xarray_dataclass_to_dict_schema,
    xarray_dataclass_to_array_schema,
    xarray_dataclass_to_dataset_schema,
)
from xradio.schema.bases import (
    xarray_dataarray_schema,
    xarray_dataset_schema,
    dict_schema,
)
from xradio.schema.export import export_schema_json_file, import_schema_json_file

Dim1 = Literal["coord"]
Dim2 = Literal["coord2"]
Dim3 = Literal["coord3"]


@xarray_dataarray_schema
class _TestArraySchema:
    """
    Docstring of array schema

    Multiple lines!
    """

    data: Data[Dim1, complex]
    """Docstring of data"""
    coord: Coord[Dim1, float]
    """Docstring of coordinate"""
    attr1: Attr[str]
    """Required attribute"""
    attr2: Attr[int] = 123
    """Required attribute with default"""
    attr3: Optional[Attr[int]] = None
    """Optional attribute with default"""


# The equivalent of the above in the meta-model
TEST_ARRAY_SCHEMA = ArraySchema(
    schema_name=__name__ + "._TestArraySchema",
    dimensions=[["coord"]],
    coordinates=[
        ArraySchemaRef(
            schema_name=None,
            name="coord",
            dtypes=[numpy.dtype(float).str],
            dimensions=[["coord"]],
            coordinates=[],
            attributes=[],
            class_docstring=None,
            data_docstring=None,
            optional=False,
            default=None,
            docstring="Docstring of coordinate",
        ),
    ],
    dtypes=[numpy.dtype(complex).str],
    class_docstring="Docstring of array schema\n\nMultiple lines!",
    data_docstring="Docstring of data",
    attributes=[
        AttrSchemaRef(
            name="attr1",
            type="str",
            optional=False,
            default=None,
            docstring="Required attribute",
        ),
        AttrSchemaRef(
            name="attr2",
            type="int",
            optional=False,
            default=123,
            docstring="Required attribute with default",
        ),
        AttrSchemaRef(
            name="attr3",
            type="int",
            optional=True,
            default=None,
            docstring="Optional attribute with default",
        ),
    ],
)


def test_xarray_dataclass_to_array_schema():
    """Ensure that extracting the model from the dataclass is consistent"""

    assert xarray_dataclass_to_array_schema(_TestArraySchema) == TEST_ARRAY_SCHEMA


def test_check_array():
    # Should succeed
    data = numpy.zeros(10, dtype=complex)
    coords = [("coord", numpy.arange(10, dtype=float))]
    attrs = {"attr1": "str", "attr2": 123, "attr3": 345}
    array = xarray.DataArray(data, coords, attrs=attrs)

    # No issues
    issues = check_array(array, TEST_ARRAY_SCHEMA)
    assert not issues
    issues.expect()

    # Check contents
    assert numpy.allclose(array.coord, numpy.arange(10, dtype=float))
    assert numpy.allclose(array.data, numpy.zeros(10, dtype=complex))
    assert set(array.attrs.keys()) == {"attr1", "attr2", "attr3"}
    assert array.attr1 == "str"
    assert array.attr2 == 123
    assert array.attr3 == 345


def test_check_array_dask():
    data = dask.array.zeros(10, dtype=complex)
    coords = [("coord", numpy.arange(10, dtype=float))]
    attrs = {"attr1": "str", "attr2": 123, "attr3": 345}
    array = xarray.DataArray(data, coords, attrs=attrs)
    assert not check_array(array, TEST_ARRAY_SCHEMA)
    assert isinstance(array.data, dask.array.Array)
    numpy.testing.assert_equal(array.compute(), numpy.zeros(10, dtype=complex))


def test_check_array_constructor_array_style():
    # Try by using constructor xarray.DataArray-style
    data = numpy.zeros(10, dtype=complex)
    coords = [("coord", numpy.arange(10, dtype=float))]
    attrs = {"attr1": "str", "attr2": 1234, "attr3": 345}
    array = _TestArraySchema(data=data, coords=coords, attrs=attrs)
    assert isinstance(array, xarray.DataArray)
    check_array(array, TEST_ARRAY_SCHEMA).expect()

    # Check contents
    assert numpy.allclose(array.coord, numpy.arange(10, dtype=float))
    assert numpy.allclose(array.data, numpy.zeros(10, dtype=complex))
    assert set(array.attrs.keys()) == {"attr1", "attr2", "attr3"}
    assert array.attr1 == "str"
    assert array.attr2 == 1234
    assert array.attr3 == 345


def test_check_array_constructor_dataclass_style():
    # Check when passing parameter (dataclass-style)
    array = _TestArraySchema(
        data=numpy.zeros(10, dtype=complex),
        coord=numpy.arange(10, dtype=float),
        attr1="str",
        attr2=1234,
        attr3=345,
    )
    assert isinstance(array, xarray.DataArray)
    check_array(array, TEST_ARRAY_SCHEMA).expect()

    # Check contents
    assert numpy.allclose(array.coord, numpy.arange(10, dtype=float))
    assert numpy.allclose(array.data, numpy.zeros(10, dtype=complex))
    assert set(array.attrs.keys()) == {"attr1", "attr2", "attr3"}
    assert array.attr1 == "str"
    assert array.attr2 == 1234
    assert array.attr3 == 345


def test_check_array_constructor_from_dataarray():
    # Create schema-conformant DataArray
    data = numpy.zeros(10, dtype=complex)
    coords = [("coord", numpy.arange(10, dtype=float))]
    attrs = {"attr1": "str", "attr2": 1234, "attr3": 345}
    array = xarray.DataArray(data, coords, attrs=attrs)

    # Check that we can create a copy from it using the constructor
    array = _TestArraySchema(array)
    assert isinstance(array, xarray.DataArray)
    check_array(array, TEST_ARRAY_SCHEMA).expect()

    # Check contents
    assert numpy.allclose(array.coord, numpy.arange(10, dtype=float))
    assert numpy.allclose(array.data, numpy.zeros(10, dtype=complex))
    assert set(array.attrs.keys()) == {"attr1", "attr2", "attr3"}
    assert array.attr1 == "str"
    assert array.attr2 == 1234
    assert array.attr3 == 345


def test_check_array_constructor_from_dataarray_override():
    # Create schema-conformant DataArray
    data = numpy.zeros(10, dtype=complex)
    coords = [("coord", numpy.arange(10, dtype=float))]
    attrs = {"attr1": "str", "attr2": 1234, "attr3": 345}
    array = xarray.DataArray(data, coords, attrs=attrs)

    # Check that we can create a copy form it using constructor,
    # but override attributes and coordinates
    array = _TestArraySchema(
        data=array,
        coords=[("coord", 1 + numpy.arange(10, dtype=float))],
        attrs={"attr1": "strstr", "attr2": 12345},
    )
    assert isinstance(array, xarray.DataArray)
    check_array(array, TEST_ARRAY_SCHEMA).expect()

    # Check contents
    assert numpy.allclose(array.coord, 1 + numpy.arange(10, dtype=float))
    assert numpy.allclose(array.data, numpy.zeros(10, dtype=complex))
    assert set(array.attrs.keys()) == {"attr1", "attr2", "attr3"}
    assert array.attr1 == "strstr"
    assert array.attr2 == 12345
    assert array.attr3 == 345


def test_check_array_constructor_auto_coords():
    # Check that we can omit "coords", which should result in them getting
    # filled in by a "numpy.arange" automatically
    array = _TestArraySchema(
        data=numpy.zeros(10, dtype=complex), attr1="str", attr2=1234, attr3=345
    )
    assert isinstance(array, xarray.DataArray)
    check_array(array, TEST_ARRAY_SCHEMA).expect()

    # Check contents
    assert numpy.allclose(array.coord, numpy.arange(10, dtype=float))
    assert numpy.allclose(array.data, numpy.zeros(10, dtype=complex))
    assert set(array.attrs.keys()) == {"attr1", "attr2", "attr3"}
    assert array.attr1 == "str"
    assert array.attr2 == 1234
    assert array.attr3 == 345


def test_check_array_constructor_list():
    # Check that we can use lists instead of numpy arrays, and they get
    # converted into numpy arrays of the schema type
    array = _TestArraySchema(
        data=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        coord=("coord", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], {"asd": "foo"}),
        attr1="str",
        attr2=1234,
        attr3=345,
    )
    assert isinstance(array, xarray.DataArray)
    check_array(array, TEST_ARRAY_SCHEMA).expect()

    # Check contents
    assert numpy.allclose(array.coord, numpy.arange(10, dtype=float))
    assert numpy.allclose(array.data, numpy.zeros(10, dtype=complex))
    assert set(array.coords.keys()) == {"coord"}
    assert array.coord.attrs.keys() == {"asd"}
    assert array.coord.attrs["asd"] == "foo"
    assert set(array.attrs.keys()) == {"attr1", "attr2", "attr3"}
    assert array.attr1 == "str"
    assert array.attr2 == 1234
    assert array.attr3 == 345


def test_check_array_constructor_defaults():
    # Check when passing parameter (dataclass-style, using positional
    # parameters and defaults)
    array = _TestArraySchema(
        numpy.zeros(10, dtype=complex),
        numpy.arange(10, dtype=float),
        "str",
    )
    assert isinstance(array, xarray.DataArray)
    check_array(array, TEST_ARRAY_SCHEMA).expect()

    # Check contents
    assert numpy.allclose(array.coord, numpy.arange(10, dtype=float))
    assert numpy.allclose(array.data, numpy.zeros(10, dtype=complex))
    assert set(array.attrs.keys()) == {"attr1", "attr2"}
    assert array.attr1 == "str"
    assert array.attr2 == 123


def test_check_array_constructor_mixed():
    # Check when passing parameter (everything)
    array = _TestArraySchema(
        numpy.zeros(10, dtype=complex),
        attr1="str",
        coords={
            "coord": numpy.arange(10, dtype=float),
        },
        attrs={
            "attr2": 123,
        },
    )
    assert isinstance(array, xarray.DataArray)
    check_array(array, TEST_ARRAY_SCHEMA).expect()

    # Check contents
    assert numpy.allclose(array.coord, numpy.arange(10, dtype=float))
    assert numpy.allclose(array.data, numpy.zeros(10, dtype=complex))
    assert set(array.attrs.keys()) == {"attr1", "attr2"}
    assert array.attr1 == "str"
    assert array.attr2 == 123


def test_check_array_dtype_mismatch():
    data_f = numpy.zeros(10, dtype=float)
    coords = [("coord", numpy.arange(10, dtype=float))]
    attrs = {"attr1": "str", "attr2": 123, "attr3": 345}
    results = check_array(
        xarray.DataArray(data_f, coords, attrs=attrs), TEST_ARRAY_SCHEMA
    )
    assert len(results) == 1
    assert results[0].path == [("dtype", None)]
    assert results[0].found == numpy.dtype(float)
    assert results[0].expected == [numpy.dtype(complex)]

    with pytest.raises(SchemaIssues):
        results.expect()


def test_check_array_dtype_mismatch_constructor():
    with pytest.raises(SchemaIssues):
        _TestArraySchema(
            numpy.zeros(10, dtype=float), numpy.arange(10, dtype=float), attr1="str"
        )


def test_check_array_dtype_mismatch_expect():
    data_f = numpy.zeros(10, dtype=float)
    coords = [("coord", numpy.arange(10, dtype=float))]
    attrs = {"attr1": "str", "attr2": 123, "attr3": 345}
    results = check_array(
        xarray.DataArray(data_f, coords, attrs=attrs), TEST_ARRAY_SCHEMA
    )
    assert len(results) == 1
    assert results[0].path == [("dtype", None)]
    assert results[0].found == numpy.dtype(float)
    assert results[0].expected == [numpy.dtype(complex)]


def test_check_array_extra_coord():
    coords2 = [
        ("coord", numpy.arange(10, dtype=float)),
        ("coord2", numpy.arange(1, dtype=float)),
    ]
    data2 = numpy.zeros(10, dtype=complex)[:, numpy.newaxis]
    attrs = {"attr1": "str", "attr2": 123, "attr3": 345}
    results = check_array(
        xarray.DataArray(data2, coords2, attrs=attrs), TEST_ARRAY_SCHEMA
    )
    assert len(results) == 1
    assert results[0].path == [("dims", None)]
    assert results[0].found == ["coord", "coord2"]
    assert results[0].expected == [["coord"]]


def test_check_array_missing_coord():
    data0 = numpy.array(None, dtype=complex)
    attrs = {"attr1": "str", "attr2": 123, "attr3": 345}
    results = check_array(xarray.DataArray(data0, {}, attrs=attrs), TEST_ARRAY_SCHEMA)
    assert len(results) == 2
    assert results[0].path == [("dims", None)]
    assert results[0].found == []
    assert results[0].expected == [["coord"]]
    assert results[1].path == [("coords", "coord")]


def test_check_array_wrong_coord():
    data = numpy.zeros(10, dtype=complex)
    coords3 = [("coord2", numpy.arange(10, dtype=int))]
    attrs = {"attr1": "str", "attr2": 123, "attr3": 345}
    results = check_array(
        xarray.DataArray(data, coords3, attrs=attrs), TEST_ARRAY_SCHEMA
    )
    assert len(results) == 2
    assert results[0].path == [("dims", None)]
    assert results[0].found == [
        "coord2",
    ]
    assert results[0].expected == [["coord"]]
    assert results[1].path == [("coords", "coord")]


def test_check_array_missing_attr():
    data = numpy.zeros(10, dtype=complex)
    coords = [("coord", numpy.arange(10, dtype=float))]
    results = check_array(xarray.DataArray(data, coords), TEST_ARRAY_SCHEMA)
    assert len(results) == 2
    assert results[0].path == [("attrs", "attr1")]
    assert results[1].path == [("attrs", "attr2")]


def test_check_array_extra_attr():
    data = numpy.zeros(10, dtype=complex)
    coords = [("coord", numpy.arange(10, dtype=float))]
    attrs = {"attr1": "str", "attr2": 123, "attr3": 345, "attr4": "asd"}
    array = xarray.DataArray(data, coords, attrs=attrs)
    assert not check_array(array, TEST_ARRAY_SCHEMA)


def test_check_array_optional_attr():
    data = numpy.zeros(10, dtype=complex)
    coords = [("coord", numpy.arange(10, dtype=float))]
    attrs = {"attr1": "str", "attr2": 123}
    array = xarray.DataArray(data, coords, attrs=attrs)
    assert not check_array(array, TEST_ARRAY_SCHEMA)


def test_check_array_wrong_type():
    data = numpy.zeros(10, dtype=complex)
    coords = [("coord", numpy.arange(10, dtype=float))]
    wrong_attrs = {"attr1": 123, "attr2": str, "attr3": 345.0}
    results = check_array(
        xarray.DataArray(data, coords, attrs=wrong_attrs), TEST_ARRAY_SCHEMA
    )
    assert len(results) == 3
    assert results[0].path == [("attrs", "attr1")]
    assert results[0].found == int
    assert results[0].expected == [str]
    assert results[1].path == [("attrs", "attr2")]
    assert results[1].found == type
    assert results[1].expected == [int]
    assert results[2].path == [("attrs", "attr3")]
    assert results[2].found == float
    assert results[2].expected == [int, type(None)]


def test_schema_checked_wrap():
    @schema_checked
    def fn(a: int, b: _TestArraySchema) -> str:
        """Docstring"""

    # Make sure docstring and signature survives (mostly for Sphinx'
    # benefit...)
    assert fn.__doc__ == "Docstring"
    sig = inspect.signature(fn)
    assert sig.parameters["a"].annotation == int
    assert sig.parameters["b"].annotation == _TestArraySchema
    assert sig.return_annotation == str


def test_schema_checked_no_annotation():
    @schema_checked
    def fn(array):
        pass

    # Should be able to pass any parameters
    fn(0)
    fn(None)
    fn(xarray.DataArray(numpy.zeros(10)))


def test_schema_checked_annotation():
    @schema_checked
    def fn(array: _TestArraySchema):
        pass

    data = numpy.zeros(10, dtype=complex)
    coords = [("coord", numpy.arange(10, dtype=float))]
    attrs = {"attr1": "str", "attr2": 123, "attr3": 345}
    array = xarray.DataArray(data, coords, attrs=attrs)
    fn(array)

    data = numpy.zeros(10, dtype=float)
    array = xarray.DataArray(data, coords, attrs=attrs)
    with pytest.raises(SchemaIssues) as exc_info:
        fn(array)
    assert exc_info.value.issues[0].path == [("array", None), ("dtype", None)]

    with pytest.raises(SchemaIssues) as exc_info:
        fn(None)
    assert exc_info.value.issues[0].path == [("array", None)]


def test_schema_checked_annotation_optional():
    @schema_checked
    def fn(array: Optional[_TestArraySchema]):
        pass

    data = numpy.zeros(10, dtype=complex)
    coords = [("coord", numpy.arange(10, dtype=float))]
    attrs = {"attr1": "str", "attr2": 123, "attr3": 345}
    array = xarray.DataArray(data, coords, attrs=attrs)
    fn(array)

    data = numpy.zeros(10, dtype=float)
    array = xarray.DataArray(data, coords, attrs=attrs)
    with pytest.raises(SchemaIssues) as exc_info:
        fn(array)
    assert exc_info.value.issues[0].path == [("array", None), ("dtype", None)]

    # Should succeed
    fn(None)

    # Should fail
    with pytest.raises(SchemaIssues) as exc_info:
        fn(1)
    assert exc_info.value.issues[0].path == [("array", None)]
    print(exc_info.value.issues[0].expected)
    assert exc_info.value.issues[0].expected == [xarray.DataArray, type(None)]
    assert exc_info.value.issues[0].found == int


def test_schema_checked_annotation_optional():
    @schema_checked
    def fn(array: Optional[_TestArraySchema]):
        pass

    data = numpy.zeros(10, dtype=complex)
    coords = [("coord", numpy.arange(10, dtype=float))]
    attrs = {"attr1": "str", "attr2": 123, "attr3": 345}
    array = xarray.DataArray(data, coords, attrs=attrs)
    fn(array)

    data = numpy.zeros(10, dtype=float)
    array = xarray.DataArray(data, coords, attrs=attrs)
    with pytest.raises(SchemaIssues) as exc_info:
        fn(array)
    assert exc_info.value.issues[0].path == [("array", None), ("dtype", None)]

    # Should succeed
    fn(None)

    # Should fail
    with pytest.raises(SchemaIssues) as exc_info:
        fn(1)
    assert exc_info.value.issues[0].path == [("array", None)]
    assert exc_info.value.issues[0].expected == [xarray.DataArray, type(None)]
    assert exc_info.value.issues[0].found == int


@dict_schema
class _TestDictSchema:
    """
    Docstring of dict schema

    Multiple lines!
    """

    attr1: str
    """Required attribute"""
    attr2: int = 123
    """Required attribute with default"""
    attr3: Optional[int] = None
    """Optional attribute with default"""


# The equivalent of the above in the meta-model
TEST_DICT_SCHEMA = DictSchema(
    schema_name=__name__ + "._TestDictSchema",
    class_docstring="Docstring of dict schema\n\nMultiple lines!",
    attributes=[
        AttrSchemaRef(
            name="attr1",
            type="str",
            optional=False,
            default=None,
            docstring="Required attribute",
        ),
        AttrSchemaRef(
            name="attr2",
            type="int",
            optional=False,
            default=123,
            docstring="Required attribute with default",
        ),
        AttrSchemaRef(
            name="attr3",
            type="int",
            optional=True,
            default=None,
            docstring="Optional attribute with default",
        ),
    ],
)


def test_xarray_dataclass_to_dict_schema():
    """Ensure that extracting the model from the dataclass is consistent"""

    assert xarray_dataclass_to_dict_schema(_TestDictSchema) == TEST_DICT_SCHEMA


def test_check_dict():
    # Should succeed
    data = {"attr1": "asd", "attr2": 234, "attr3": 345}
    issues = check_dict(data, TEST_DICT_SCHEMA)
    assert not issues
    issues.expect()


def test_check_dict_optional():
    # Should succeed
    data = {"attr1": "asd", "attr2": 234}
    issues = check_dict(data, TEST_DICT_SCHEMA)
    assert not issues
    issues.expect()


def test_check_dict_constructor():
    # Should succeed
    data = _TestDictSchema(attr1="asd", attr2=234, attr3=345)
    assert isinstance(data, dict)
    issues = check_dict(data, TEST_DICT_SCHEMA)
    issues.expect()

    # Check that data is correct
    assert data == {"attr1": "asd", "attr2": 234, "attr3": 345}


def test_check_dict_constructor_defaults():
    # Should succeed
    data = _TestDictSchema(attr1="asd")
    assert isinstance(data, dict)
    issues = check_dict(data, TEST_DICT_SCHEMA)
    issues.expect()

    # Check that data is correct
    assert data == {"attr1": "asd", "attr2": 123, "attr3": None}


def test_check_dict_typ():
    # Should succeed
    data = {"attr1": "asd", "attr2": "foo"}
    results = check_dict(data, TEST_DICT_SCHEMA)
    assert len(results) == 1
    assert results[0].path == [("", "attr2")]
    assert results[0].found == str
    assert results[0].expected == [int]

    with pytest.raises(SchemaIssues):
        results.expect()


def test_check_dict_missing():
    # Should succeed
    data = {"attr1": "asd"}
    results = check_dict(data, TEST_DICT_SCHEMA)
    assert len(results) == 1
    assert results[0].path == [("", "attr2")]
    assert results[0].found == None
    assert results[0].expected == ["int"]

    with pytest.raises(SchemaIssues):
        results.expect()


@xarray_dataarray_schema
class _TestDatasetSchemaCoord:
    """
    Docstring of array schema for coordinate
    """

    data: Data[Dim1, float]
    """Docstring of coordinate data"""
    attr1: Attr[str]
    """Required attribute"""
    attr2: Attr[int] = 123
    """Required attribute with default"""
    attr3: Optional[Attr[int]] = None
    """Optional attribute with default"""


@xarray_dataset_schema
class _TestDatasetSchema:
    """
    Docstring of dataset schema

    Again multiple lines!
    """

    coord: Coordof[_TestDatasetSchemaCoord]
    """Docstring of coordinate"""
    coord2: Optional[Coord[Dim2, int]]
    """Docstring of second coordinate"""
    data_var: Dataof[_TestArraySchema]
    """Docstring of external data variable"""
    data_var_simple: Optional[Data[Dim2, numpy.float32]]
    """Docstring of simple optional data variable"""
    attr1: Attr[str]
    """Required attribute"""
    attr2: Attr[int] = 123
    """Required attribute with default"""
    attr3: Optional[Attr[int]] = None
    """Optional attribute with default"""


def _dataclass_to_dict(obj, ignore=[]):
    return {
        f.name: getattr(obj, f.name)
        for f in dataclasses.fields(type(obj))
        if f.name not in ignore
    }


# The equivalent of the above in the meta-model
TEST_DATASET_SCHEMA = DatasetSchema(
    schema_name=__name__ + "._TestDatasetSchema",
    dimensions=[["coord"], ["coord", "coord2"]],
    coordinates=[
        ArraySchemaRef(
            schema_name=__name__ + "._TestDatasetSchemaCoord",
            name="coord",
            dtypes=[numpy.dtype(float).str],
            dimensions=[["coord"]],
            optional=False,
            default=None,
            docstring="Docstring of coordinate",
            coordinates=[],
            attributes=_dataclass_to_dict(TEST_ARRAY_SCHEMA)["attributes"],
            class_docstring="Docstring of array schema for coordinate",
            data_docstring="Docstring of coordinate data",
        ),
        ArraySchemaRef(
            schema_name=None,
            name="coord2",
            dtypes=[numpy.dtype(int).str],
            dimensions=[["coord2"]],
            coordinates=[],
            attributes=[],
            class_docstring=None,
            data_docstring=None,
            optional=True,
            default=None,
            docstring="Docstring of second coordinate",
        ),
    ],
    data_vars=[
        ArraySchemaRef(
            name="data_var",
            optional=False,
            default=None,
            docstring="Docstring of external data variable",
            **_dataclass_to_dict(TEST_ARRAY_SCHEMA),
        ),
        ArraySchemaRef(
            schema_name=None,
            name="data_var_simple",
            dtypes=[numpy.dtype(numpy.float32).str],
            dimensions=[["coord2"]],
            coordinates=[],
            attributes=[],
            class_docstring=None,
            data_docstring=None,
            optional=True,
            default=None,
            docstring="Docstring of simple optional data variable",
        ),
    ],
    attributes=[
        AttrSchemaRef(
            name="attr1",
            type="str",
            optional=False,
            default=None,
            docstring="Required attribute",
        ),
        AttrSchemaRef(
            name="attr2",
            type="int",
            optional=False,
            default=123,
            docstring="Required attribute with default",
        ),
        AttrSchemaRef(
            name="attr3",
            type="int",
            optional=True,
            default=None,
            docstring="Optional attribute with default",
        ),
    ],
    class_docstring="Docstring of dataset schema\n\nAgain multiple lines!",
)


def test_xarray_dataclass_to_dataset_schema():
    assert xarray_dataclass_to_dataset_schema(_TestDatasetSchema) == TEST_DATASET_SCHEMA


def test_check_dataset():
    attrs = {"attr1": "str", "attr2": 123, "attr3": 345}
    coords = {
        "coord": xarray.DataArray(
            numpy.arange(10, dtype=float), dims=("coord",), attrs=attrs
        ),
        "coord2": numpy.arange(5, dtype=int),
    }
    data_vars = {
        "data_var": ("coord", numpy.zeros(10, dtype=complex), attrs),
        "data_var_simple": ("coord2", numpy.zeros(5, dtype=numpy.float32)),
    }
    dataset = xarray.Dataset(data_vars, coords, attrs)
    issues = check_dataset(dataset, TEST_DATASET_SCHEMA)
    assert not issues
    numpy.testing.assert_equal(dataset["data_var"], numpy.zeros(10, dtype=complex))
    numpy.testing.assert_equal(
        dataset["data_var_simple"], numpy.zeros(5, dtype=numpy.float32)
    )


def test_check_dataset_constructor_dataset_style():
    """Test typical way to construct xarray.Dataset - using tuples"""

    attrs = {"attr1": "str", "attr2": 12345, "attr3": 345}
    coords = {
        "coord": xarray.DataArray(
            numpy.arange(10, dtype=float), dims=("coord",), attrs=attrs
        ),
        "coord2": numpy.arange(5, dtype=int),
    }
    data_vars = {
        "data_var": ("coord", dask.array.zeros(10, dtype=complex), attrs),
        "data_var_simple": ("coord2", dask.array.zeros(5, dtype=numpy.float32)),
    }

    # Use constructor
    dataset = _TestDatasetSchema(data_vars=data_vars, coords=coords, attrs=attrs)
    issues = check_dataset(dataset, TEST_DATASET_SCHEMA)
    assert not issues
    assert isinstance(dataset["data_var"].data, dask.array.Array)
    assert isinstance(dataset["data_var_simple"].data, dask.array.Array)
    numpy.testing.assert_equal(
        dataset["data_var"].compute(), numpy.zeros(10, dtype=complex)
    )
    numpy.testing.assert_equal(
        dataset["data_var_simple"].compute(), numpy.zeros(5, dtype=numpy.float32)
    )
    assert dataset.attrs == attrs


def test_check_dataset_constructor_dataset_style_variable():
    """Same as above, but passing xarray.Variable instead of tuples"""

    attrs = {"attr1": "str", "attr2": 12345, "attr3": 345}
    coords = {
        "coord": xarray.Variable(
            ("coord",), numpy.arange(10, dtype=float), attrs=attrs
        ),
        "coord2": xarray.Variable(("coord2",), numpy.arange(5, dtype=int), attrs=attrs),
    }
    data_vars = {
        "data_var": xarray.Variable(
            "coord", dask.array.zeros(10, dtype=complex), attrs
        ),
        "data_var_simple": xarray.Variable(
            "coord2", dask.array.zeros(5, dtype=numpy.float32)
        ),
    }

    # Use constructor
    dataset = _TestDatasetSchema(data_vars=data_vars, coords=coords, attrs=attrs)
    issues = check_dataset(dataset, TEST_DATASET_SCHEMA)
    assert not issues
    assert isinstance(dataset["data_var"].data, dask.array.Array)
    assert isinstance(dataset["data_var_simple"].data, dask.array.Array)
    numpy.testing.assert_equal(
        dataset["data_var"].compute(), numpy.zeros(10, dtype=complex)
    )
    numpy.testing.assert_equal(
        dataset["data_var_simple"].compute(), numpy.zeros(5, dtype=numpy.float32)
    )
    assert dataset.attrs == attrs


def test_check_dataset_constructor_dataclass_style():
    """Test that we can pass named parameters to the constructor"""

    # Use constructor
    attrs = {"attr1": "str", "attr2": 123, "attr3": 345}
    dataset = _TestDatasetSchema(
        data_var=_TestArraySchema(
            dask.array.zeros(10, dtype=complex),
            dims=("coord",),
            coord=xarray.DataArray(
                numpy.arange(10, dtype=float), dims=("coord",), attrs=attrs
            ),
            attrs=attrs,
        ),
        data_var_simple=("coord2", dask.array.zeros(5, dtype=numpy.float32)),
        coord=xarray.DataArray(
            numpy.arange(10, dtype=float), dims=("coord",), attrs=attrs
        ),
        coord2=numpy.arange(5, dtype=int),
        attr1="str",
        attr2=123,
        attr3=345,
    )
    issues = check_dataset(dataset, TEST_DATASET_SCHEMA)
    assert not issues
    assert isinstance(dataset["data_var"].data, dask.array.Array)
    assert isinstance(dataset["data_var_simple"].data, dask.array.Array)
    numpy.testing.assert_equal(
        dataset["data_var"].compute(), numpy.zeros(10, dtype=complex)
    )
    numpy.testing.assert_equal(
        dataset["data_var_simple"].compute(), numpy.zeros(5, dtype=numpy.float32)
    )


def test_check_dataset_constructor_auto_coords():
    """Test that coordinates get automatically filled"""

    attrs = {"attr1": "str", "attr2": 123, "attr3": 345}
    dataset = _TestDatasetSchema(
        data_var=_TestArraySchema(
            dask.array.zeros(10, dtype=complex),
            dims=("coord",),
            coord=xarray.DataArray(
                numpy.arange(10, dtype=float), dims=("coord",), attrs=attrs
            ),
            attrs=attrs,
        ),
        # Can't default "coord" because of the attributes...
        coord=xarray.DataArray(
            numpy.arange(10, dtype=float), dims=("coord",), attrs=attrs
        ),
        data_var_simple=("coord2", dask.array.zeros(25, dtype=numpy.float32)),
        attr1="str",
        attr2=123,
        attr3=345,
    )
    issues = check_dataset(dataset, TEST_DATASET_SCHEMA)
    assert not issues
    assert isinstance(dataset["data_var"].data, dask.array.Array)
    assert isinstance(dataset["data_var_simple"].data, dask.array.Array)
    numpy.testing.assert_equal(dataset["coord2"], numpy.arange(25, dtype=int))


def test_check_dataset():
    attrs = {"attr1": "str", "attr2": 123, "attr3": 345}
    coords = {
        "coord": xarray.DataArray(
            numpy.arange(10, dtype=float), dims=("coord",), attrs=attrs
        ),
        "coord2": numpy.arange(5, dtype=int),
    }
    data_vars = {
        "data_var": ("coord", numpy.zeros(10, dtype=complex), attrs),
        "data_var_simple": ("coord2", numpy.zeros(5, dtype=numpy.float32)),
    }
    dataset = xarray.Dataset(data_vars, coords, attrs)
    issues = check_dataset(dataset, TEST_DATASET_SCHEMA)
    assert not issues
    numpy.testing.assert_equal(dataset["data_var"], numpy.zeros(10, dtype=complex))
    numpy.testing.assert_equal(
        dataset["data_var_simple"], numpy.zeros(5, dtype=numpy.float32)
    )


def test_check_dataset_wrong_dim_order():
    attrs = {"attr1": "str", "attr2": 123, "attr3": 345}
    coords = {
        "coord2": numpy.arange(5, dtype=int),
        "coord": xarray.DataArray(
            numpy.arange(10, dtype=float), dims=("coord",), attrs=attrs
        ),
    }
    data_vars = {
        "data_var_simple": (("coord2",), numpy.ones(5, dtype=numpy.float32)),
        "data_var": (("coord",), numpy.zeros(10, dtype=complex), attrs),
    }
    dataset = xarray.Dataset(data_vars, coords, attrs)
    issues = check_dataset(dataset, TEST_DATASET_SCHEMA)
    assert not issues


def test_check_dataset_dtype_mismatch():
    attrs = {"attr1": "str", "attr2": 123, "attr3": 345}
    coords = {
        "coord": xarray.DataArray(
            numpy.arange(10, dtype=float), dims=("coord",), attrs=attrs
        ),
        "coord2": numpy.arange(5, dtype=float),
    }
    data_vars = {
        "data_var_simple": (("coord2",), numpy.ones(5, dtype=float)),
        "data_var": (("coord",), numpy.zeros(10, dtype=complex), attrs),
    }
    dataset = xarray.Dataset(data_vars, coords, attrs)
    issues = check_dataset(dataset, TEST_DATASET_SCHEMA)
    assert len(issues) == 2
    assert issues[0].path == [("coords", "coord2"), ("dtype", None)]
    assert issues[0].expected == [numpy.dtype(int)]
    assert issues[0].found == numpy.dtype(float)
    assert issues[1].path == [("data_vars", "data_var_simple"), ("dtype", None)]
    assert issues[1].expected == [numpy.dtype(numpy.float32).str]
    assert issues[1].found == numpy.dtype(float)


def test_check_dataset_wrong_dim():
    attrs = {"attr1": "str", "attr2": 123, "attr3": 345}
    coords = {
        "coord2": numpy.arange(5, dtype=int),
        "coord": xarray.DataArray(
            numpy.arange(10, dtype=float), dims=("coord",), attrs=attrs
        ),
    }
    data_vars = {
        "data_var_simple": (("coord",), numpy.ones(10, dtype=numpy.float32)),
        "data_var": (("coord",), numpy.zeros(10, dtype=complex), attrs),
    }
    dataset = xarray.Dataset(data_vars, coords, attrs)
    issues = check_dataset(dataset, TEST_DATASET_SCHEMA)
    assert len(issues) == 1
    assert issues[0].path == [("data_vars", "data_var_simple"), ("dims", None)]
    assert issues[0].expected == [
        [
            "coord2",
        ]
    ]
    assert issues[0].found == ["coord"]


def test_check_dataset_extra_datavar():
    attrs = {"attr1": "str", "attr2": 123, "attr3": 345}
    coords = {
        "coord2": numpy.arange(5, dtype=int),
        "coord": xarray.DataArray(
            numpy.arange(10, dtype=float), dims=("coord",), attrs=attrs
        ),
    }
    data_vars = {
        "data_var_simple": (("coord2",), numpy.zeros(5, dtype=numpy.float32)),
        "data_var": (("coord",), numpy.zeros(10, dtype=complex), attrs),
        "extra_data_var": (("coord",), numpy.ones(10, dtype=int), attrs),
    }
    dataset = xarray.Dataset(data_vars, coords, attrs)
    issues = check_dataset(dataset, TEST_DATASET_SCHEMA)
    assert not issues


def test_check_dataset_optional_datavar():
    attrs = {"attr1": "str", "attr2": 123, "attr3": 345}
    coords = {
        "coord": xarray.DataArray(
            numpy.arange(10, dtype=float), dims=("coord",), attrs=attrs
        ),
        "coord2": numpy.arange(5, dtype=int),
    }
    data_vars = {
        "data_var": (("coord",), numpy.zeros(10, dtype=complex), attrs),
    }
    dataset = xarray.Dataset(data_vars, coords, attrs)
    issues = check_dataset(dataset, TEST_DATASET_SCHEMA)
    assert not issues


def test_check_dataset_optional_coordinate():
    attrs = {"attr1": "str", "attr2": 123, "attr3": 345}
    coords = {
        "coord": xarray.DataArray(
            numpy.arange(10, dtype=float), dims=("coord",), attrs=attrs
        ),
    }
    data_vars = {
        "data_var": (("coord",), numpy.zeros(10, dtype=complex), attrs),
    }
    dataset = xarray.Dataset(data_vars, coords, attrs)
    issues = check_dataset(dataset, TEST_DATASET_SCHEMA)
    assert not issues


def test_check_dict_array_attribute():
    # Make array
    data = numpy.zeros(10, dtype=complex)
    coords = [("coord", numpy.arange(10, dtype=float))]
    attrs = {"attr1": "str", "attr2": 123, "attr3": 345}
    array = xarray.DataArray(data, coords, attrs=attrs)

    # Check inside dictionary
    @dict_schema
    class _DictSchema:
        da: _TestArraySchema

    assert not check_dict({"da": array}, _DictSchema)

    array = xarray.DataArray(data, coords)
    assert check_dict({"da": array}, _DictSchema)


def test_check_dict_dict_attribute():
    # Check inside dictionary
    @dict_schema
    class _DictSchema:
        da: _TestDictSchema

    assert not check_dict(
        {"da": {"attr1": "asd", "attr2": 234, "attr3": 345}}, _DictSchema
    )
    assert check_dict({"da": {"attr2": 234, "attr3": 345}}, _DictSchema)


def test_check_dict_dict_attribute():
    # Check inside dictionary
    @dict_schema
    class _DictSchema:
        da: _TestDictSchema

    assert not check_dict(
        {"da": {"attr1": "asd", "attr2": 234, "attr3": 345}}, _DictSchema
    )
    assert check_dict({"da": {"attr2": 234, "attr3": 345}}, _DictSchema)


TEST_DATASET_SCHEMA_JSON = {
    "$class": "DatasetSchema",
    "schema_name": "tests.unit.schema.test_schema._TestDatasetSchema",
    "dimensions": [["coord"], ["coord", "coord2"]],
    "coordinates": [
        {
            "$class": "ArraySchemaRef",
            "schema_name": "tests.unit.schema.test_schema._TestDatasetSchemaCoord",
            "dimensions": [["coord"]],
            "dtypes": ["<f8"],
            "coordinates": [],
            "attributes": [
                {
                    "$class": "AttrSchemaRef",
                    "type": "str",
                    "name": "attr1",
                    "docstring": "Required attribute",
                },
                {
                    "$class": "AttrSchemaRef",
                    "type": "int",
                    "name": "attr2",
                    "default": 123,
                    "docstring": "Required attribute with default",
                },
                {
                    "$class": "AttrSchemaRef",
                    "type": "int",
                    "optional": True,
                    "name": "attr3",
                    "docstring": "Optional attribute with default",
                },
            ],
            "class_docstring": "Docstring of array schema for coordinate",
            "data_docstring": "Docstring of coordinate data",
            "name": "coord",
            "optional": False,
            "docstring": "Docstring of coordinate",
        },
        {
            "$class": "ArraySchemaRef",
            "schema_name": None,
            "dimensions": [["coord2"]],
            "dtypes": ["<i8"],
            "coordinates": [],
            "attributes": [],
            "class_docstring": None,
            "data_docstring": None,
            "name": "coord2",
            "optional": True,
            "docstring": "Docstring of second coordinate",
        },
    ],
    "data_vars": [
        {
            "$class": "ArraySchemaRef",
            "schema_name": "tests.unit.schema.test_schema._TestArraySchema",
            "dimensions": [["coord"]],
            "dtypes": ["<c16"],
            "coordinates": [
                {
                    "$class": "ArraySchemaRef",
                    "schema_name": None,
                    "dimensions": [["coord"]],
                    "dtypes": ["<f8"],
                    "coordinates": [],
                    "attributes": [],
                    "class_docstring": None,
                    "data_docstring": None,
                    "name": "coord",
                    "optional": False,
                    "docstring": "Docstring of coordinate",
                }
            ],
            "attributes": [
                {
                    "$class": "AttrSchemaRef",
                    "type": "str",
                    "name": "attr1",
                    "docstring": "Required attribute",
                },
                {
                    "$class": "AttrSchemaRef",
                    "type": "int",
                    "name": "attr2",
                    "default": 123,
                    "docstring": "Required attribute with default",
                },
                {
                    "$class": "AttrSchemaRef",
                    "type": "int",
                    "optional": True,
                    "name": "attr3",
                    "docstring": "Optional attribute with default",
                },
            ],
            "class_docstring": "Docstring of array schema\n\nMultiple lines!",
            "data_docstring": "Docstring of data",
            "name": "data_var",
            "optional": False,
            "docstring": "Docstring of external data variable",
        },
        {
            "$class": "ArraySchemaRef",
            "schema_name": None,
            "dimensions": [["coord2"]],
            "dtypes": ["<f4"],
            "coordinates": [],
            "attributes": [],
            "class_docstring": None,
            "data_docstring": None,
            "name": "data_var_simple",
            "optional": True,
            "docstring": "Docstring of simple optional data variable",
        },
    ],
    "attributes": [
        {
            "$class": "AttrSchemaRef",
            "type": "str",
            "name": "attr1",
            "docstring": "Required attribute",
        },
        {
            "$class": "AttrSchemaRef",
            "type": "int",
            "name": "attr2",
            "default": 123,
            "docstring": "Required attribute with default",
        },
        {
            "$class": "AttrSchemaRef",
            "type": "int",
            "optional": True,
            "name": "attr3",
            "docstring": "Optional attribute with default",
        },
    ],
    "class_docstring": "Docstring of dataset schema\n\nAgain multiple lines!",
}


def test_schema_export(tmp_path):

    # Export schema
    tmp_fname = tmp_path / "test_dataset_schema.json"
    export_schema_json_file(_TestDatasetSchema, tmp_fname)

    # Check against reference
    with open(tmp_fname, "r", encoding="utf8") as f:
        assert json.load(f) == TEST_DATASET_SCHEMA_JSON

    # Check import round-trip
    schema = import_schema_json_file(tmp_fname)
    assert schema == TEST_DATASET_SCHEMA


# ---------------------------------------------------------------------------
# Module-level fixtures for check.py tests
# ---------------------------------------------------------------------------


@xarray_dataarray_schema
class _TestArray2DSchema:
    """2D test array schema for dimension order testing"""

    data: Data[tuple[Dim1, Dim2], float]
    """2D data"""


@xarray_dataarray_schema
class _TestMultiVersionArray:
    """Array schema for allow_multiple_versions testing"""

    data: Data[Dim1, float]
    """Data"""
    allow_multiple_versions: Optional[Attr[bool]] = True


@xarray_dataset_schema
class _TestDatasetSchemaMultiVersion:
    """Dataset schema with allow_multiple_versions data variable"""

    coord: Coord[Dim1, float]
    """Coordinate"""
    data_var: Dataof[_TestMultiVersionArray]
    """Multi-version data variable"""
    attr1: Attr[str]
    """Required attribute"""


@xarray_dataset_schema
class _TestRegisteredDatasetSchema:
    """Schema for datatree registration testing"""

    coord: Coord[Dim1, float]
    """Coordinate"""
    type: Attr[Literal["test_registered_type"]]
    """Type identifier"""


@dict_schema
class _DictWithDataArrayAttr:
    da: _TestArraySchema


@dict_schema
class _DictWithOptionalDataArrayAttr:
    da: Optional[_TestArraySchema]


@dict_schema
class _DictWithNestedDictAttr:
    nested: _TestDictSchema


@dict_schema
class _DictWithOptionalNestedDictAttr:
    nested: Optional[_TestDictSchema]


@dict_schema
class _DictWithListStrAttr:
    tags: list[str]


@dict_schema
class _DictWithOptionalListStrAttr:
    tags: Optional[list[str]]


@dict_schema
class _DictWithLiteralAttr:
    mode: Literal["fast", "slow"]


# ---------------------------------------------------------------------------
# SchemaIssue.path_str()
# ---------------------------------------------------------------------------


def test_path_str_with_names():
    issue = SchemaIssue(
        path=[("data_vars", "foo"), ("coords", "bar"), ("attrs", "asd")],
        message="test",
    )
    assert issue.path_str() == "data_vars['foo'].coords['bar'].attrs['asd']"


def test_path_str_with_none_ix():
    issue = SchemaIssue(
        path=[("dims", None), ("dtype", None)],
        message="test",
    )
    assert issue.path_str() == "dims.dtype"


def test_path_str_with_empty_string_ix():
    issue = SchemaIssue(
        path=[("", "some_node"), ("attrs", "type")],
        message="test",
    )
    assert issue.path_str() == "['some_node'].attrs['type']"


def test_path_str_mixed():
    issue = SchemaIssue(
        path=[("data_vars", "VISIBILITY"), ("dims", None)],
        message="test",
    )
    assert issue.path_str() == "data_vars['VISIBILITY'].dims"


# ---------------------------------------------------------------------------
# SchemaIssue.__repr__()
# ---------------------------------------------------------------------------


def test_schema_issue_repr_no_expected():
    issue = SchemaIssue(
        path=[("attrs", "attr1")],
        message="Non-optional attribute is missing!",
    )
    result = repr(issue)
    assert "Schema issue with" in result
    assert "Non-optional attribute is missing!" in result
    assert "expected" not in result


def test_schema_issue_repr_with_expected():
    issue = SchemaIssue(
        path=[("dtype", None)],
        message="Wrong numpy dtype",
        found=numpy.dtype(float),
        expected=[numpy.dtype(complex)],
    )
    result = repr(issue)
    assert "expected:" in result
    assert "found:" in result


def test_schema_issue_repr_no_found():
    issue = SchemaIssue(
        path=[("attrs", "attr1")],
        message="Non-optional attribute is missing!",
        expected=["str"],
    )
    result = repr(issue)
    assert "expected:" in result
    assert "found:" not in result


def test_schema_issue_repr_multiple_expected():
    issue = SchemaIssue(
        path=[("attrs", "attr1")],
        message="Wrong type",
        found=int,
        expected=[str, type(None)],
    )
    result = repr(issue)
    assert " or " in result


# ---------------------------------------------------------------------------
# SchemaIssues.__init__ / __add__ / __str__ / __repr__
# ---------------------------------------------------------------------------


def test_schema_issues_init_copy():
    issue = SchemaIssue(path=[("attrs", "x")], message="missing")
    original = SchemaIssues([issue])
    copy = SchemaIssues(original)
    assert len(copy) == 1
    assert copy.issues is original.issues


def test_schema_issues_add_operator():
    issue1 = SchemaIssue(path=[("attrs", "a")], message="first")
    issue2 = SchemaIssue(path=[("attrs", "b")], message="second")
    combined = SchemaIssues([issue1]) + SchemaIssues([issue2])
    assert len(combined) == 2
    assert combined[0].message == "first"
    assert combined[1].message == "second"


def test_schema_issues_add_shares_list():
    # NOTE: SchemaIssues.__init__(SchemaIssues) shares the inner list by reference
    # (self.issues = issues.issues), so __add__ mutates the left operand as a
    # side-effect. This test documents the actual behavior.
    issue1 = SchemaIssue(path=[("attrs", "a")], message="first")
    issue2 = SchemaIssue(path=[("attrs", "b")], message="second")
    issues_b = SchemaIssues([issue2])
    combined = SchemaIssues([issue1]) + issues_b
    assert len(combined) == 2
    assert len(issues_b) == 1  # right operand is unaffected


def test_schema_issues_str_no_issues():
    assert str(SchemaIssues()) == "No schema issues found"


def test_schema_issues_str_with_issues():
    issues = SchemaIssues(
        [SchemaIssue(path=[("attrs", "attr1")], message="missing attribute")]
    )
    result = str(issues)
    assert "missing attribute" in result
    assert result.startswith("\n * ")


def test_schema_issues_repr_empty():
    assert repr(SchemaIssues()).startswith("SchemaIssues(")


def test_schema_issues_repr_with_issues():
    issues = SchemaIssues([SchemaIssue(path=[("attrs", "attr1")], message="missing")])
    result = repr(issues)
    assert result.startswith("SchemaIssues(")
    assert "missing" in result


# ---------------------------------------------------------------------------
# SchemaIssues.expect() with path argument
# ---------------------------------------------------------------------------


def test_schema_issues_expect_prepends_path():
    issue = SchemaIssue(path=[("dtype", None)], message="wrong type")
    issues = SchemaIssues([issue])
    with pytest.raises(SchemaIssues) as exc_info:
        issues.expect(elem="data_vars", ix="VISIBILITY")
    raised = exc_info.value
    assert raised.issues[0].path[0] == ("data_vars", "VISIBILITY")
    assert raised.issues[0].path[1] == ("dtype", None)


def test_schema_issues_expect_empty_no_raise():
    SchemaIssues().expect(elem="data_vars", ix="foo")


# ---------------------------------------------------------------------------
# check_array() type guards
# ---------------------------------------------------------------------------


def test_check_array_non_dataarray():
    with pytest.raises(TypeError, match="check_array: Expected xarray.DataArray"):
        check_array({"not": "a dataarray"}, TEST_ARRAY_SCHEMA)


def test_check_array_dataclass_schema():
    data = numpy.zeros(10, dtype=complex)
    coords = [("coord", numpy.arange(10, dtype=float))]
    attrs = {"attr1": "str", "attr2": 123}
    array = xarray.DataArray(data, coords, attrs=attrs)
    assert not check_array(array, _TestArraySchema)


def test_check_array_invalid_schema():
    array = xarray.DataArray(
        numpy.zeros(10, dtype=complex), [("coord", numpy.arange(10, dtype=float))]
    )
    with pytest.raises(TypeError, match="check_array: Expected ArraySchema"):
        check_array(array, "not_a_schema")


def test_check_array_wrong_dim_order():
    data = numpy.zeros((10, 5), dtype=float)
    coords = [
        ("coord2", numpy.arange(5, dtype=float)),
        ("coord", numpy.arange(10, dtype=float)),
    ]
    array = xarray.DataArray(data.T, dims=("coord2", "coord"), coords=dict(coords))
    issues = check_array(array, _TestArray2DSchema)
    assert len(issues) == 1
    assert issues[0].path == [("dims", None)]
    assert "wrong order" in issues[0].message


# ---------------------------------------------------------------------------
# check_dataset() type guards
# ---------------------------------------------------------------------------


def _make_valid_dataset():
    attrs = {"attr1": "str", "attr2": 123}
    coords = {
        "coord": xarray.DataArray(
            numpy.arange(10, dtype=float), dims=("coord",), attrs=attrs
        ),
    }
    data_vars = {
        "data_var": ("coord", numpy.zeros(10, dtype=complex), attrs),
    }
    return xarray.Dataset(data_vars, coords, attrs)


def test_check_dataset_non_dataset():
    with pytest.raises(TypeError, match="check_dataset: Expected xarray.Dataset"):
        check_dataset({"not": "a dataset"}, TEST_DATASET_SCHEMA)


def test_check_dataset_dataclass_schema():
    assert not check_dataset(_make_valid_dataset(), _TestDatasetSchema)


def test_check_dataset_invalid_schema():
    with pytest.raises(TypeError, match="check_dataset: Expected DatasetSchema"):
        check_dataset(_make_valid_dataset(), "not_a_schema")


# ---------------------------------------------------------------------------
# check_dataset() missing data_var message
# ---------------------------------------------------------------------------


def test_check_dataset_missing_datavar_msg():
    attrs = {"attr1": "str", "attr2": 123}
    coords = {
        "coord": xarray.DataArray(
            numpy.arange(10, dtype=float), dims=("coord",), attrs=attrs
        ),
    }
    issues = check_dataset(xarray.Dataset({}, coords, attrs), TEST_DATASET_SCHEMA)
    assert any(
        "data_var" in issue.message and "have" in issue.message
        for issue in issues.issues
    )


# ---------------------------------------------------------------------------
# check_data_vars() allow_multiple_versions
# ---------------------------------------------------------------------------


def test_check_dataset_multi_version_match():
    attrs = {"attr1": "str"}
    coords = {"coord": numpy.arange(10, dtype=float)}
    data_vars = {
        "data_var_v1": ("coord", numpy.zeros(10, dtype=float)),
        "data_var_v2": ("coord", numpy.ones(10, dtype=float)),
    }
    assert not check_dataset(
        xarray.Dataset(data_vars, coords, attrs), _TestDatasetSchemaMultiVersion
    )


def test_check_dataset_multi_version_no_match():
    attrs = {"attr1": "str"}
    coords = {"coord": numpy.arange(10, dtype=float)}
    data_vars = {"other_var": ("coord", numpy.zeros(10, dtype=float))}
    assert check_dataset(
        xarray.Dataset(data_vars, coords, attrs), _TestDatasetSchemaMultiVersion
    )


# ---------------------------------------------------------------------------
# check_dict() type guards
# ---------------------------------------------------------------------------


def test_check_dict_non_dict():
    with pytest.raises(TypeError, match="check_dict: Expected dictionary"):
        check_dict(["not", "a", "dict"], TEST_DICT_SCHEMA)


def test_check_dict_invalid_schema():
    with pytest.raises(TypeError, match="check_dict: Expected DictSchema"):
        check_dict({"attr1": "val"}, "not_a_schema")


# ---------------------------------------------------------------------------
# check_dimensions() edge cases
# ---------------------------------------------------------------------------


def test_check_dimensions_wrong_order():
    issues = check_dimensions(
        dims=("coord2", "coord"),
        expected=[["coord", "coord2"]],
        check_order=True,
    )
    assert len(issues) == 1
    assert issues[0].path == [("dims", None)]
    assert "wrong order" in issues[0].message
    assert issues[0].found == ["coord2", "coord"]
    assert issues[0].expected == [["coord", "coord2"]]


def test_check_dimensions_correct_order():
    assert not check_dimensions(
        dims=("coord", "coord2"),
        expected=[["coord", "coord2"]],
        check_order=True,
    )


def test_check_dimensions_missing_hint():
    issues = check_dimensions(dims=("coord",), expected=[["coord", "coord2"]])
    assert len(issues) == 1
    assert "Missing dimension" in issues[0].message


def test_check_dimensions_superfluous_hint():
    issues = check_dimensions(
        dims=("coord", "coord2", "extra"), expected=[["coord", "coord2"]]
    )
    assert len(issues) == 1
    assert "Superfluous" in issues[0].message


def test_check_dimensions_replace_hint():
    issues = check_dimensions(
        dims=("coord", "wrong_dim"), expected=[["coord", "coord2"]]
    )
    assert len(issues) == 1
    assert "replace" in issues[0].message


# ---------------------------------------------------------------------------
# _check_value() — dict → DataArray conversion
# ---------------------------------------------------------------------------


def test_check_value_da_from_dict_valid():
    valid_da_dict = {
        "dims": ["coord"],
        "data": list(numpy.zeros(10, dtype=complex)),
        "coords": {"coord": {"dims": ["coord"], "data": list(numpy.arange(10.0))}},
        "attrs": {"attr1": "str", "attr2": 123},
    }
    assert not check_dict({"da": valid_da_dict}, _DictWithDataArrayAttr)


def test_check_value_da_from_dict_value_error():
    assert check_dict({"da": {"dims": ["x"], "data": None}}, _DictWithDataArrayAttr)


def test_check_value_da_from_dict_type_error():
    # Non-string coordinate key triggers TypeError in DataArray.from_dict
    assert check_dict(
        {"da": {"dims": ["x"], "data": [1, 2, 3], "coords": {42: [1, 2, 3]}}},
        _DictWithDataArrayAttr,
    )


def test_check_value_da_non_dataarray():
    issues = check_dict({"da": 42}, _DictWithDataArrayAttr)
    assert issues
    assert any("not an xarray.DataArray" in repr(i) for i in issues.issues)


def test_check_value_optional_da_value_error():
    assert check_dict({"da": {"not_dims": [1, 2, 3]}}, _DictWithOptionalDataArrayAttr)


def test_check_value_optional_da_type_error():
    assert check_dict(
        {"da": {"dims": ["x"], "data": [1, 2, 3], "coords": {42: [1, 2, 3]}}},
        _DictWithOptionalDataArrayAttr,
    )


# ---------------------------------------------------------------------------
# _check_value() — wrong type for dict attribute
# ---------------------------------------------------------------------------


def test_check_value_wrong_type_for_dict():
    issues = check_dict({"nested": "not_a_dict"}, _DictWithNestedDictAttr)
    assert issues
    assert any("not a dictionary" in repr(i) for i in issues.issues)


def test_check_value_optional_wrong_type_dict():
    assert check_dict({"nested": 42}, _DictWithOptionalNestedDictAttr)


# ---------------------------------------------------------------------------
# _check_value() — list[str] type
# ---------------------------------------------------------------------------


def test_check_value_list_str_valid():
    assert not check_dict({"tags": ["alpha", "beta"]}, _DictWithListStrAttr)


def test_check_value_list_str_not_list():
    issues = check_dict({"tags": "not_a_list"}, _DictWithListStrAttr)
    assert issues
    assert any("not a list" in repr(i) for i in issues.issues)


def test_check_value_list_str_not_all_strings():
    issues = check_dict({"tags": ["alpha", 42, "gamma"]}, _DictWithListStrAttr)
    assert issues
    assert any("not a list of strings" in repr(i) for i in issues.issues)


def test_check_value_optional_list_not_list():
    assert check_dict({"tags": 999}, _DictWithOptionalListStrAttr)


def test_check_value_optional_list_not_strings():
    assert check_dict({"tags": ["ok", 123]}, _DictWithOptionalListStrAttr)


# ---------------------------------------------------------------------------
# _check_value() — invalid type name / literal values
# ---------------------------------------------------------------------------


def test_check_value_invalid_type():
    bad_schema = AttrSchemaRef(
        name="x", type="unsupported_type", optional=False, default=None, docstring=None
    )
    with pytest.raises(ValueError, match="Invalid typ_name in schema"):
        _check_value("some_value", bad_schema)


def test_check_value_literal_valid():
    assert not check_dict({"mode": "fast"}, _DictWithLiteralAttr)


def test_check_value_literal_valid_second():
    assert not check_dict({"mode": "slow"}, _DictWithLiteralAttr)


def test_check_value_literal_invalid():
    issues = check_dict({"mode": "medium"}, _DictWithLiteralAttr)
    assert issues
    assert any("Disallowed literal value" in repr(i) for i in issues.issues)
    assert any(
        i.found == "medium" for i in issues.issues if hasattr(i, "found") and i.found
    )


# ---------------------------------------------------------------------------
# register_dataset_type()
# ---------------------------------------------------------------------------


def test_register_dataset_type():
    from xradio.schema.check import _DATASET_TYPES
    from xradio.schema import xarray_dataclass_to_dataset_schema

    schema = xarray_dataclass_to_dataset_schema(_TestRegisteredDatasetSchema)
    register_dataset_type(schema)
    assert "test_registered_type" in _DATASET_TYPES
    assert _DATASET_TYPES["test_registered_type"] is schema


def test_register_dataset_type_no_literal():
    from xradio.schema.metamodel import DatasetSchema, AttrSchemaRef

    schema_no_literal = DatasetSchema(
        schema_name="test_no_literal",
        dimensions=[[]],
        coordinates=[],
        data_vars=[],
        attributes=[
            AttrSchemaRef(
                name="type",
                type="str",
                optional=False,
                default=None,
                docstring=None,
                literal=None,
            )
        ],
        class_docstring=None,
    )
    with pytest.warns(UserWarning, match='Attribute "type" should be a literal'):
        register_dataset_type(schema_no_literal)


# ---------------------------------------------------------------------------
# check_datatree()
# ---------------------------------------------------------------------------


def test_check_datatree_no_data_nodes():
    assert not check_datatree(xarray.DataTree())


def test_check_datatree_unknown_type():
    dataset = xarray.Dataset(
        {"x": ("coord", numpy.arange(5))},
        attrs={"type": "nonexistent_schema_type_xyz"},
    )
    issues = check_datatree(xarray.DataTree(dataset=dataset))
    assert issues
    assert any("Unknown dataset type" in repr(i) for i in issues.issues)


def test_check_datatree_valid_schema():
    from xradio.schema import xarray_dataclass_to_dataset_schema

    schema = xarray_dataclass_to_dataset_schema(_TestRegisteredDatasetSchema)
    register_dataset_type(schema)
    dataset = xarray.Dataset(
        attrs={"type": "test_registered_type"},
        coords={"coord": numpy.arange(5, dtype=float)},
    )
    assert not check_datatree(xarray.DataTree(dataset=dataset))


def test_check_datatree_missing_type_attr():
    dataset = xarray.Dataset({"x": ("coord", numpy.arange(5))})
    issues = check_datatree(xarray.DataTree(dataset=dataset))
    assert issues
    assert any("Unknown dataset type" in repr(i) for i in issues.issues)


def test_check_datatree_with_parent_dims():
    # Child node has has_data=True, parent is root with coords only.
    # The child's parent reference triggers line 685: parent_dims = set(node.parent.dims).
    # Root node has no 'type' attr → 1 unknown-type issue; child passes schema check.
    from xradio.schema import xarray_dataclass_to_dataset_schema

    schema = xarray_dataclass_to_dataset_schema(_TestRegisteredDatasetSchema)
    register_dataset_type(schema)

    parent_ds = xarray.Dataset(coords={"coord": numpy.arange(5, dtype=float)})
    child_ds = xarray.Dataset(
        data_vars={"values": ("coord", numpy.arange(5, dtype=float))},
        attrs={"type": "test_registered_type"},
        coords={"coord": numpy.arange(5, dtype=float)},
    )
    dt = xarray.DataTree.from_dict({"/": parent_ds, "/child": child_ds})
    issues = check_datatree(dt)
    root_issues = [i for i in issues.issues if i.path == [("", "/")]]
    child_issues = [i for i in issues.issues if i.path != [("", "/")]]
    assert len(root_issues) == 1
    assert "Unknown dataset type" in root_issues[0].message
    assert not child_issues


# ---------------------------------------------------------------------------
# schema_checked() — check_parameters forms
# ---------------------------------------------------------------------------


def test_schema_checked_params_false():
    def fn(array: _TestArraySchema) -> None:
        pass

    fn_checked = schema_checked(fn, check_parameters=False)
    fn_checked("not_a_dataarray")
    fn_checked(None)
    fn_checked(42)


def test_schema_checked_params_iterable():
    def fn(a: int, b: _TestArraySchema) -> None:
        pass

    fn_checked = schema_checked(fn, check_parameters=["b"])

    data = numpy.zeros(10, dtype=complex)
    coords = [("coord", numpy.arange(10, dtype=float))]
    attrs = {"attr1": "str", "attr2": 123}
    array = xarray.DataArray(data, coords, attrs=attrs)

    fn_checked(a="not_int", b=array)  # a unchecked, b valid → passes

    with pytest.raises(SchemaIssues):
        fn_checked(a=1, b="not_an_array")  # b invalid → raises


def test_schema_checked_params_skips_unchecked():
    # When check_parameters=['b'], 'a' is not checked — covers the
    # 'arg not in parameters_to_check → continue' branch (line 726)
    def fn(a: int, b: str) -> None:
        pass

    fn_checked = schema_checked(fn, check_parameters=["b"])
    fn_checked(a="wrong_type_but_not_checked", b="correct_str")
