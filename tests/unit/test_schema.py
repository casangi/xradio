import dataclasses
from typing import Literal, Optional, Union
import numpy

from xradio.schema.typing import Attr, Coord, Coordof, Data, Dataof, Name
from xradio.schema.metamodel import (
    ArraySchema,
    ArraySchemaRef,
    AttrSchemaRef,
    DatasetSchema,
)
from xradio.schema.dataclass import (
    xarray_dataclass_to_array_schema,
    xarray_dataclass_to_dataset_schema,
)

Dim1 = Literal["Dim1"]
Dim2 = Literal["Dim2"]
Dim3 = Literal["Dim3"]


@dataclasses.dataclass(frozen=True)
class TestArraySchema:
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
    schema_name=__name__ + ".TestArraySchema",
    dimensions=[("Dim1",)],
    coordinates=[
        ArraySchemaRef(
            schema_name="tests.unit.test_schema.TestArraySchema.coord",
            name="coord",
            dtypes=[numpy.dtype(float)],
            dimensions=[("Dim1",)],
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

    assert xarray_dataclass_to_array_schema(TestArraySchema) == TEST_ARRAY_SCHEMA


@dataclasses.dataclass(frozen=True)
class TestDatasetSchemaCoord:
    """
    Docstring of array schema for coordinate
    """

    data: Data[Dim1, complex]
    """Docstring of coordinate data"""
    attr1: Attr[str]
    """Required attribute"""
    attr2: Attr[int] = 123
    """Required attribute with default"""
    attr3: Optional[Attr[int]] = None
    """Optional attribute with default"""


@dataclasses.dataclass(frozen=True)
class TestDatasetSchema:
    """
    Docstring of dataset schema

    Again multiple lines!
    """

    coord: Coordof[TestDatasetSchemaCoord]
    """Docstring of coordinate"""
    coord2: Coord[Dim2, int]
    """Docstring of second coordinate"""
    data_var: Dataof[TestArraySchema]
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
    schema_name=__name__ + ".TestDatasetSchema",
    dimensions=[["Dim1", "Dim2"]],
    coordinates=[
        ArraySchemaRef(
            schema_name=__name__ + ".TestDatasetSchemaCoord",
            name="coord",
            dtypes=[numpy.dtype(complex)],
            dimensions=[("Dim1",)],
            optional=False,
            default=dataclasses.MISSING,
            docstring="Docstring of coordinate",
            coordinates=[],
            attributes=_dataclass_to_dict(TEST_ARRAY_SCHEMA)["attributes"],
            class_docstring="Docstring of array schema for coordinate",
            data_docstring="Docstring of coordinate data",
        ),
        ArraySchemaRef(
            schema_name=__name__ + ".TestDatasetSchema.coord2",
            name="coord2",
            dtypes=[numpy.dtype(int)],
            dimensions=[("Dim2",)],
            coordinates=[],
            attributes=[],
            class_docstring=None,
            data_docstring=None,
            optional=False,
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
            schema_name=__name__ + ".TestDatasetSchema.data_var_simple",
            name="data_var_simple",
            dtypes=[numpy.float32],
            dimensions=[("Dim2",)],
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

    assert xarray_dataclass_to_dataset_schema(TestDatasetSchema) == TEST_DATASET_SCHEMA
