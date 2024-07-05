"""Typing support for xarray data classes

This has been extracted from the xarray-dataclasses package by astropenguin
(see https://github.com/astropenguin/xarray-dataclasses/). The reason we
replicate this here is because we actually ignore / redo everything but the
type annotations, especially adding xradio-specific support for multiple
options in data variable / coordinate dimensionality and dtype.
"""

from typing import (
    Any,
    List,
    Tuple,
    Hashable,
    Iterable,
    Type,
    ClassVar,
    Dict,
    TypeVar,
    Union,
    Sequence,
    Generic,
    Collection,
    Literal,
    get_type_hints,
    get_args,
    get_origin,
    Annotated,
    Protocol,
)

from typing import Union

try:
    # Python 3.10 forward: TypeAlias, ParamSpec are standard, and there is the
    # "a | b" UnionType alternative to "Union[a,b]"
    from typing import TypeAlias, ParamSpec
    from types import UnionType

    HAVE_UNIONTYPE = True
except ImportError:
    # Python 3.9: Get TypeAlias, ParamSpec from typing_extensions, no support
    # for "a | b"
    from typing_extensions import (
        TypeAlias,
        ParamSpec,
    )

    HAVE_UNIONTYPE = False
import numpy as np
from itertools import chain
from enum import Enum

PInit = ParamSpec("PInit")
T = TypeVar("T")
TDataClass = TypeVar("TDataClass", bound="DataClass[Any]")
TDims = TypeVar("TDims", covariant=True)
TDType = TypeVar("TDType", covariant=True)
THashable = TypeVar("THashable", bound=Hashable)

AnyArray: TypeAlias = "np.ndarray[Any, Any]"
AnyDType: TypeAlias = "np.dtype[Any]"
AnyField: TypeAlias = "Field[Any]"
AnyXarray: TypeAlias = "xr.DataArray | xr.Dataset"
Dims = Tuple[str, ...]
Order = Literal["C", "F"]
Shape = Union[Sequence[int], int]
Sizes = Dict[str, int]


class DataClass(Protocol[PInit]):
    """Type hint for dataclass objects."""

    def __init__(self, *args: PInit.args, **kwargs: PInit.kwargs) -> None: ...

    __dataclass_fields__: ClassVar[Dict[str, AnyField]]


class Labeled(Generic[TDims]):
    """Type hint for labeled objects."""

    pass


# type hints (public)
class Role(Enum):
    """Annotations for typing dataclass fields."""

    ATTR = "attr"
    """Annotation for attribute fields."""

    COORD = "coord"
    """Annotation for coordinate fields."""

    DATA = "data"
    """Annotation for data (variable) fields."""

    NAME = "name"
    """Annotation for name fields."""

    OTHER = "other"
    """Annotation for other fields."""

    @classmethod
    def annotates(cls, tp: Any) -> bool:
        """Check if any role annotates a type hint."""
        if get_origin(tp) is not Annotated:
            return False

        return any(isinstance(arg, cls) for arg in get_args(tp))


Attr = Annotated[T, Role.ATTR]
"""Type hint for attribute fields (``Attr[T]``).

Example:
    ::

        @dataclass
        class Image(AsDataArray):
            data: Data[tuple[X, Y], float]
            long_name: Attr[str] = "luminance"
            units: Attr[str] = "cd / m^2"

Hint:
    The following field names are specially treated when plotting.

    - ``long_name`` or ``standard_name``: Coordinate name.
    - ``units``: Coordinate units.

Reference:
    https://xarray.pydata.org/en/stable/user-guide/plotting.html

"""

Coord = Annotated[Union[Labeled[TDims], Collection[TDType], TDType], Role.COORD]
"""Type hint for coordinate fields (``Coord[TDims, TDType]``).

Example:
    ::

        @dataclass
        class Image(AsDataArray):
            data: Data[tuple[X, Y], float]
            mask: Coord[tuple[X, Y], bool]
            x: Coord[X, int] = 0
            y: Coord[Y, int] = 0

Hint:
    A coordinate field whose name is the same as ``TDims``
    (e.g. ``x: Coord[X, int]``) can define a dimension.

"""

Coordof = Annotated[Union[TDataClass, Any], Role.COORD]
"""Type hint for coordinate fields (``Coordof[TDataClass]``).

Unlike ``Coord``, it specifies a dataclass that defines a DataArray class.
This is useful when users want to add metadata to dimensions for plotting.

Example:
    ::

        @dataclass
        class XAxis:
            data: Data[X, int]
            long_name: Attr[str] = "x axis"


        @dataclass
        class YAxis:
            data: Data[Y, int]
            long_name: Attr[str] = "y axis"


        @dataclass
        class Image(AsDataArray):
            data: Data[tuple[X, Y], float]
            x: Coordof[XAxis] = 0
            y: Coordof[YAxis] = 0

Hint:
    A class used in ``Coordof`` does not need to inherit ``AsDataArray``.

"""

Data = Annotated[Union[Labeled[TDims], Collection[TDType], TDType], Role.DATA]
"""Type hint for data fields (``Coordof[TDims, TDType]``).

Example:
    Exactly one data field is allowed in a DataArray class
    (the second and subsequent data fields are just ignored)::

        @dataclass
        class Image(AsDataArray):
            data: Data[tuple[X, Y], float]

    Multiple data fields are allowed in a Dataset class::

        @dataclass
        class ColorImage(AsDataset):
            red: Data[tuple[X, Y], float]
            green: Data[tuple[X, Y], float]
            blue: Data[tuple[X, Y], float]

"""

Dataof = Annotated[Union[TDataClass, Any], Role.DATA]
"""Type hint for data fields (``Coordof[TDataClass]``).

Unlike ``Data``, it specifies a dataclass that defines a DataArray class.
This is useful when users want to reuse a dataclass in a Dataset class.

Example:
    ::

        @dataclass
        class Image:
            data: Data[tuple[X, Y], float]
            x: Coord[X, int] = 0
            y: Coord[Y, int] = 0


        @dataclass
        class ColorImage(AsDataset):
            red: Dataof[Image]
            green: Dataof[Image]
            blue: Dataof[Image]

Hint:
    A class used in ``Dataof`` does not need to inherit ``AsDataArray``.

"""

Name = Annotated[THashable, Role.NAME]
"""Type hint for name fields (``Name[THashable]``).

Example:
    ::

        @dataclass
        class Image(AsDataArray):
            data: Data[tuple[X, Y], float]
            name: Name[str] = "image"

"""


def deannotate(tp: Any) -> Any:
    """Recursively remove annotations in a type hint."""

    class Temporary:
        __annotations__ = dict(type=tp)

    return get_type_hints(Temporary)["type"]


def find_annotated(tp: Any) -> Iterable[Any]:
    """Generate all annotated types in a type hint."""
    args = get_args(tp)

    if get_origin(tp) is Annotated:
        yield tp
        yield from find_annotated(args[0])
    else:
        yield from chain(*map(find_annotated, args))


def get_annotated(tp: Any) -> Any:
    """Extract the first role-annotated type."""
    for annotated in filter(Role.annotates, find_annotated(tp)):
        return deannotate(annotated)

    raise TypeError("Could not find any role-annotated type.")


def get_annotations(tp: Any) -> Tuple[Any, ...]:
    """Extract annotations of the first role-annotated type."""
    for annotated in filter(Role.annotates, find_annotated(tp)):
        return get_args(annotated)[1:]

    raise TypeError("Could not find any role-annotated type.")


def get_dataclass(tp: Any) -> Type[DataClass[Any]]:
    """Extract a dataclass."""
    try:
        dataclass = get_args(get_annotated(tp))[0]
    except TypeError:
        raise TypeError(f"Could not find any dataclass in {tp!r}.")

    if not is_dataclass(dataclass):
        raise TypeError(f"Could not find any dataclass in {tp!r}.")

    return dataclass


def get_dims(tp: Any) -> List[Dims]:
    """Extract data dimensions (dims)."""
    try:
        dims = get_args(get_args(get_annotated(tp))[0])[0]
    except TypeError:
        raise TypeError(f"Could not find any dims in {tp!r}.")

    # List of allowed dtypes (might just be one)
    if get_origin(dims) is Union:
        dims_in = get_args(dims)
    elif HAVE_UNIONTYPE and get_origin(dims) is UnionType:
        dims_in = get_args(dims)
    else:
        dims_in = [dims]

    dims_out = []
    for dim in dims_in:
        args = get_args(dim)
        origin = get_origin(dim)

        # One-dimensional dimension
        if origin is Literal:
            dims_out.append((str(args[0]),))
            continue

        if not (origin is tuple or origin is Tuple):
            raise TypeError(f"Could not find any dims in {tp!r}.")

        # Zero-dimensions
        if args == () or args == ((),):
            dims_out.append(())
            continue

        if not all(get_origin(arg) is Literal for arg in args):
            raise TypeError(f"Could not find any dims in {tp!r}.")

        dims_out.append(tuple(str(get_args(arg)[0]) for arg in args))

    return dims_out


def get_types(tp: Any) -> List[AnyDType]:
    """Extract data types from type annotation

    E.g. Coord[..., Type1 | Type2 | ...] or Data[..., Type1 | Type2 | ...]

    """
    try:
        typ = get_args(get_args(get_annotated(tp))[1])[0]
    except TypeError:
        raise TypeError(f"Could not find any dtype in {tp!r}.")

    # List of allowed dtypes (might just be one)
    if get_origin(typ) is Union:
        types_in = get_args(typ)
    elif HAVE_UNIONTYPE and get_origin(typ) is UnionType:
        types_in = get_args(typ)
    else:
        types_in = [typ]

    types_out = []
    for dt in types_in:
        # Handle case that we want to allow "Any"
        if dt is Any or dt is type(None):
            types_out.append(None)
            continue

        # Allow specifying type as literal (e.g. string)
        elif get_origin(dt) is Literal:
            dt = get_args(dt)[0]

        # Return type
        types_out.append(dt)

    return types_out


def get_name(tp: Any, default: Hashable = None) -> Hashable:
    """Extract a name if found or return given default."""
    try:
        annotations = get_annotations(tp)[1:]
    except TypeError:
        return default

    for annotation in annotations:
        if isinstance(annotation, Hashable):
            return annotation

    return default


def get_role(tp: Any, default: Role = Role.OTHER) -> Role:
    """Extract a role if found or return given default."""
    try:
        return get_annotations(tp)[0]
    except TypeError:
        return default


def is_optional(type_ann):
    """
    Check whether a type annotation indicates that the value is optional

    Boils down to checking whether it's a union type that includes None
    """
    if get_origin(type_ann) is Union:
        return None.__class__ in get_args(type_ann)
    return False
