from __future__ import annotations

from dataclasses import dataclass
import typing

__all__ = [
    "AttrSchemaRef",
    "ArraySchema",
    "ArraySchemaRef",
    "DatasetSchema",
    "DictSchema",
]


@dataclass(frozen=True)
class AttrSchemaRef:
    """
    Schema information about an attribute as referenced from an array or
    dataset schema.

    This includes the name and docstring associated with the attribute
    in the array or dataset schema definition.
    """

    name: str
    """
    Name of attribute as given in data array / dataset.

    * ``bool``: A boolean
    * ``str``: A UTF-8 string
    * ``int``: A 64-bit signed integer
    * ``float``: A double-precision floating point number
    * ``str_list``: A list of strings
    * ``dataarray``: An xarray dataarray (encoded using to_dict)
    """
    type_name: typing.Literal[
        "bool", "str", "int", "float", "list[str]", "dict", "dataarray"
    ]
    """
    Dictionary schema, if it is an xarray DataArray
    """
    dict_schema: typing.Optional[DictSchema]
    """
    Array schema, if it is an xarray DataArray
    """
    array_schema: typing.Optional[ArraySchema]
    """
    Python name of type.

    * str = Unicode string
    * int = 64 bit integer
    * float = 64 bit floating point number (double)
    """
    literal: typing.Optional[typing.List[typing.Any]]
    """
    Allowed literal values, if specified.
    """
    optional: bool
    """Is the attribute optional?"""
    default: typing.Optional[typing.Any]
    """If optional: What is the default value?"""
    docstring: str
    """Documentation string of attribute reference"""


@dataclass(frozen=True)
class ArraySchema:
    """
    Schema for xarray data array

    A data array maps a tuple of dimensions to (numpy) values. The schema
    allows for multiple options both on dimensions as well as types to be
    used.
    """

    schema_name: str
    """(Class) name of the schema"""
    dimensions: typing.List[typing.List[str]]
    """List of possible dimensions"""
    dtypes: typing.List[typing.List["numpy.dtype"]]
    """List of possible (numpy) types"""

    coordinates: typing.List["ArraySchemaRef"]
    """Coordinates data arrays giving values to dimensions"""
    attributes: typing.List[AttrSchemaRef]
    """Attributes associated with data array"""

    class_docstring: typing.Optional[str]
    """Documentation string of class"""
    data_docstring: typing.Optional[str]
    """Documentation string of data in class"""

    def is_coord(self) -> bool:
        """
        Checks with this is a valid coordinate data array

        Such data arrays must not have coordinate references of their own,
        i.e. be defined in terms of (integer) dimensions only. This is of
        course because their very purpose is to map these integer dimensions
        to semantically meaningful values, such as frequencies.
        """
        return not self.coordinates

    def required_dimensions(self) -> [str]:
        """
        Returns set of dimensions that is always required
        """

        req_dims = set(self.dimensions[0])
        for dims in self.dimensions[1:]:
            req_dims &= set(dims)
        return req_dims


@dataclass(frozen=True)
class ArraySchemaRef(ArraySchema):
    """
    Schema for xarray data array as referenced from a dataset schema

    Includes information about name and docstring associated with
    array schema where referenced
    """

    name: str
    """Name of array schema as given in dataset."""
    optional: bool
    """Is the data array optional?"""
    default: typing.Optional[typing.Any]
    """If optional: What is the default value?"""
    docstring: typing.Optional[str] = None
    """Documentation string of array reference"""


@dataclass(frozen=True)
class DatasetSchema:
    """
    Schema for an xarray dataset
    """

    schema_name: str
    """(Class) name of the schema"""

    dimensions: [[str]]
    """List of possible dimensions (derived from data arrays)"""
    coordinates: [ArraySchemaRef]
    """List of coordinate data arrays"""
    data_vars: [ArraySchemaRef]
    """List of data arrays"""
    attributes: [AttrSchemaRef]
    """List of attributes"""

    class_docstring: typing.Optional[str]
    """Documentation string of class"""


@dataclass(frozen=True)
class DictSchema:
    """
    Schema for a simple dictionary
    """

    schema_name: str
    """(Class) name of the schema"""

    attributes: [AttrSchemaRef]
    """List of attributes"""

    class_docstring: typing.Optional[str]
    """Documentation string of class"""
