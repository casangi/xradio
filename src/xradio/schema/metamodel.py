from __future__ import annotations

from dataclasses import dataclass, MISSING
import typing

__all__ = [
    "ValueSchema",
    "AttrSchemaRef",
    "ArraySchema",
    "ArraySchemaRef",
    "DatasetSchema",
    "DictSchema",
]


@dataclass
class ValueSchema:
    """
    Schema information about a value in an attribute or dictionary.
    """

    type: typing.Literal[
        "bool", "str", "int", "float", "list[str]", "dict", "dataarray"
    ]
    """
    Type of value

    * ``bool``: A boolean
    * ``str``: A UTF-8 string
    * ``int``: A 64-bit signed integer
    * ``float``: A double-precision floating point number
    * ``list[str]``: A list of strings
    * ``dict``: Dictionary
    * ``dataarray``: An xarray dataarray (encoded using ``to_dict``)
    """
    dict_schema: typing.Optional[DictSchema] = None
    """
    Dictionary schema, if it is a dict
    """
    array_schema: typing.Optional[ArraySchema] = None
    """
    Array schema, if it is an xarray DataArray
    """
    literal: typing.Optional[typing.List[typing.Any]] = None
    """
    Allowed literal values, if specified.
    """
    optional: bool = False
    """Is the value optional?"""


@dataclass
class AttrSchemaRef(ValueSchema):
    """
    Schema information about an attribute as referenced from an array or
    dataset schema.

    This includes the name and docstring associated with the attribute
    in the array or dataset schema definition.
    """

    name: str = ""
    """Name of attribute as given in data array / dataset."""
    default: typing.Optional[typing.Any] = None
    """If optional: What is the default value?"""
    docstring: str = ""
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
    dtypes: typing.List[typing.List[str]]
    """List of possible dtype options, where each inner list contains
    (numpy) types as array interface protocol descriptors (e.g. `">f4"`).
    Each inner list corresponds to a possible configuration of dtypes
    for the data array."""

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
    default: typing.Optional[typing.Any] = None
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
