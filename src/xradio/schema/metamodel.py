from dataclasses import dataclass
import typing

__all__ = [
    "AttrSchemaRef",
    "ArraySchema",
    "ArraySchemaRef",
    "DatasetSchema",
]


@dataclass
class AttrSchemaRef:
    """
    Schema information about an attribute as referenced from an array or
    dataset schema.

    This includes the name and docstring associated with the attribute
    in the array or dataset schema definition.
    """

    name: str
    """Name of attribute as given in data array / dataset."""
    typ: type
    """
    Python type of attribute. Note that this might again be a data
    array or dataset, but we don't track that explicitly.
    """
    optional: bool
    """Is the attribute optional?"""
    default: typing.Optional[typing.Any]
    """If optional: What is the default value?"""
    docstring: str
    """Documentation string of attribute reference"""


@dataclass
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


@dataclass
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


@dataclass
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
