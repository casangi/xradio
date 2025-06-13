import dataclasses
from typing import Literal, Optional, Union
import numpy
import xarray
import dask.array
import pytest
import inspect

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
    check_dict,
    schema_checked,
    SchemaIssues,
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
    dimensions=[("coord",)],
    coordinates=[
        ArraySchemaRef(
            schema_name=None,
            name="coord",
            dtypes=[numpy.dtype(float)],
            dimensions=[("coord",)],
            coordinates=[],
            attributes=[],
            class_docstring=None,
            data_docstring=None,
            optional=False,
            default=dataclasses.MISSING,
            docstring="Docstring of coordinate",
        ),
    ],
    dtypes=[numpy.dtype(complex)],
    class_docstring="Docstring of array schema\n\nMultiple lines!",
    data_docstring="Docstring of data",
    attributes=[
        AttrSchemaRef(
            name="attr1",
            typ=str,
            optional=False,
            default=dataclasses.MISSING,
            docstring="Required attribute",
        ),
        AttrSchemaRef(
            name="attr2",
            typ=int,
            optional=False,
            default=123,
            docstring="Required attribute with default",
        ),
        AttrSchemaRef(
            name="attr3",
            typ=int,
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
    assert results[0].expected == [("coord",)]


def test_check_array_missing_coord():
    data0 = numpy.array(None, dtype=complex)
    attrs = {"attr1": "str", "attr2": 123, "attr3": 345}
    results = check_array(xarray.DataArray(data0, {}, attrs=attrs), TEST_ARRAY_SCHEMA)
    assert len(results) == 2
    assert results[0].path == [("dims", None)]
    assert results[0].found == []
    assert results[0].expected == [("coord",)]
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
    assert results[0].expected == [("coord",)]
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
    assert results[2].expected == [int]


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
            typ=str,
            optional=False,
            default=dataclasses.MISSING,
            docstring="Required attribute",
        ),
        AttrSchemaRef(
            name="attr2",
            typ=int,
            optional=False,
            default=123,
            docstring="Required attribute with default",
        ),
        AttrSchemaRef(
            name="attr3",
            typ=int,
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
    assert results[0].expected == [int]

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
            dtypes=[numpy.dtype(float)],
            dimensions=[("coord",)],
            optional=False,
            default=dataclasses.MISSING,
            docstring="Docstring of coordinate",
            coordinates=[],
            attributes=_dataclass_to_dict(TEST_ARRAY_SCHEMA)["attributes"],
            class_docstring="Docstring of array schema for coordinate",
            data_docstring="Docstring of coordinate data",
        ),
        ArraySchemaRef(
            schema_name=None,
            name="coord2",
            dtypes=[numpy.dtype(int)],
            dimensions=[("coord2",)],
            coordinates=[],
            attributes=[],
            class_docstring=None,
            data_docstring=None,
            optional=True,
            default=dataclasses.MISSING,
            docstring="Docstring of second coordinate",
        ),
    ],
    data_vars=[
        ArraySchemaRef(
            name="data_var",
            optional=False,
            default=dataclasses.MISSING,
            docstring="Docstring of external data variable",
            **_dataclass_to_dict(TEST_ARRAY_SCHEMA),
        ),
        ArraySchemaRef(
            schema_name=None,
            name="data_var_simple",
            dtypes=[numpy.dtype(numpy.float32)],
            dimensions=[("coord2",)],
            coordinates=[],
            attributes=[],
            class_docstring=None,
            data_docstring=None,
            optional=True,
            default=dataclasses.MISSING,
            docstring="Docstring of simple optional data variable",
        ),
    ],
    attributes=[
        AttrSchemaRef(
            name="attr1",
            typ=str,
            optional=False,
            default=dataclasses.MISSING,
            docstring="Required attribute",
        ),
        AttrSchemaRef(
            name="attr2",
            typ=int,
            optional=False,
            default=123,
            docstring="Required attribute with default",
        ),
        AttrSchemaRef(
            name="attr3",
            typ=int,
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
    assert issues[1].expected == [numpy.float32]
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
    assert issues[0].expected == [("coord2",)]
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


def test_check_dict_dataset_attribute():
    # Make dataset
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

    # Check inside dictionary
    @dict_schema
    class _DictSchema:
        ds: _TestDatasetSchema

    assert not check_dict(
        {
            "ds": dataset,
        },
        _DictSchema,
    )
    assert check_dict(
        {
            "ds": xarray.Dataset(data_vars, coords),
        },
        _DictSchema,
    )


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
