import dataclasses
from typing import Literal, Optional, Union
import numpy
import xarray
import dask.array

from xradio.schema.typing import Attr, Coord, Coordof, Data, Dataof, Name
from xradio.schema.metamodel import (
    ArraySchema,
    ArraySchemaRef,
    AttrSchemaRef,
    DatasetSchema,
)
from xradio.schema.check import check_array, check_dataset
from xradio.schema.dataclass import (
    xarray_dataclass_to_array_schema,
    xarray_dataclass_to_dataset_schema,
)

Dim1 = Literal["coord"]
Dim2 = Literal["coord2"]
Dim3 = Literal["coord3"]


@dataclasses.dataclass(frozen=True)
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
            schema_name="tests.unit.test_schema._TestArraySchema.coord",
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

    data = numpy.zeros(10, dtype=complex)
    coords = [("coord", numpy.arange(10, dtype=float))]
    attrs = {"attr1": "str", "attr2": 123, "attr3": 345}
    array = xarray.DataArray(data, coords, attrs=attrs)
    assert not check_array(array, TEST_ARRAY_SCHEMA)


def test_check_array_dask():

    data = dask.array.zeros(10, dtype=complex)
    coords = [("coord", numpy.arange(10, dtype=float))]
    attrs = {"attr1": "str", "attr2": 123, "attr3": 345}
    array = xarray.DataArray(data, coords, attrs=attrs)
    assert not check_array(array, TEST_ARRAY_SCHEMA)
    assert isinstance(array.data, dask.array.Array)
    numpy.testing.assert_equal(array.compute(), numpy.zeros(10, dtype=complex))


def test_check_array_dtype_mismatch():

    data_f = numpy.zeros(10, dtype=float)
    coords = [("coord", numpy.arange(10, dtype=float))]
    attrs = {"attr1": "str", "attr2": 123, "attr3": 345}
    results = check_array(
        xarray.DataArray(data_f, coords, attrs=attrs), TEST_ARRAY_SCHEMA
    )
    assert len(results) == 1
    assert results[0].path == [("dtype", "")]
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
    assert results[0].path == [("dims", "")]
    assert results[0].found == ("coord", "coord2")
    assert results[0].expected == [("coord",)]


def test_check_array_missing_coord():

    data0 = numpy.array(None, dtype=complex)
    attrs = {"attr1": "str", "attr2": 123, "attr3": 345}
    results = check_array(xarray.DataArray(data0, {}, attrs=attrs), TEST_ARRAY_SCHEMA)
    assert len(results) == 2
    assert results[0].path == [("dims", "")]
    assert results[0].found == ()
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
    assert results[0].path == [("dims", "")]
    assert results[0].found == ("coord2",)
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


@dataclasses.dataclass(frozen=True)
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


@dataclasses.dataclass(frozen=True)
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
            schema_name=__name__ + "._TestDatasetSchema.coord2",
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
            **_dataclass_to_dict(TEST_ARRAY_SCHEMA)
        ),
        ArraySchemaRef(
            schema_name=__name__ + "._TestDatasetSchema.data_var_simple",
            name="data_var_simple",
            dtypes=[numpy.float32],
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


def test_check_dataset_dask():

    attrs = {"attr1": "str", "attr2": 123, "attr3": 345}
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
    dataset = xarray.Dataset(data_vars, coords, attrs)
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
    assert issues[0].path == [("coords", "coord2"), ("dtype", "")]
    assert issues[0].expected == [numpy.dtype(int)]
    assert issues[0].found == numpy.dtype(float)
    assert issues[1].path == [("data_vars", "data_var_simple"), ("dtype", "")]
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
    assert issues[0].path == [("data_vars", "data_var_simple"), ("dims", "")]
    assert issues[0].expected == [("coord2",)]
    assert issues[0].found == ("coord",)


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
