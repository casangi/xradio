from typing import get_type_hints, get_args
from .typing import get_dims, get_types, get_role, Role, get_annotated, is_optional

import typing
import inspect
import ast
import dataclasses
import numpy
import itertools
import textwrap

from xradio.schema.metamodel import *


def extract_field_docstrings(klass):
    """
    Extracts docstrings of all fields from a class

    :param klass: Class to extract docstrings from
    :returns: Dictionary of attributes to clean docstrings
    """

    # Parse class body
    try:
        src = inspect.getsource(klass)
    except OSError:
        return {}
    module = ast.parse(textwrap.dedent(src))

    # Expect module containing a class definition
    if not isinstance(module, ast.Module) or len(module.body) != 1:
        raise valueError(f"Expected parser to generate ast.Module, got {module}!")
    cls = module.body[0]
    if not isinstance(cls, ast.ClassDef):
        raise valueError(f"Expected a class definition, got {ast.dump(cls)}!")

    # Go through body, collect dostrings
    docstrings = {}
    for i, assign in enumerate(cls.body):
        # Handle both annotated and unannotated case
        if isinstance(assign, ast.AnnAssign):
            if not isinstance(assign.target, ast.Name):
                warnings.warn(f"Expected name in assignment {ast.dump(assign)}!")
                continue
            names = [assign.target.id]
        elif isinstance(assign, ast.Assign):
            if not all(isinstance(name, ast.Name) for name in assign.targets):
                warnings.warn(f"Expected names in assignment {ast.dump(assign)}!")
                continue
            names = [name.id for name in assign.targets]
        else:
            continue

        # Get docstring from next statement
        if (
            i + 1 < len(cls.body)
            and isinstance(cls.body[i + 1], ast.Expr)
            and isinstance(cls.body[i + 1].value, ast.Constant)
            and isinstance(cls.body[i + 1].value.value, str)
        ):
            for name in names:
                docstrings[name] = inspect.cleandoc(cls.body[i + 1].value.value)
        else:
            for name in names:
                docstrings[name] = None

    return docstrings


def _check_invalid_dims(
    dims: list[list[str]], all_coord_names: list[str], klass_name: str, field_name: str
):
    """
    Check dimension possibilities for undefined coordinates
    """

    # Filter out dimension possibilities with undefined coordinates
    valid_dims = [ds for ds in dims if set(ds).issubset(all_coord_names)]
    # print(f"{klass_name}.{field_name}", valid_dims, dims, all_coord_names)

    # Raise an exception if this makes the dimension set impossible
    if dims and not valid_dims:
        required_dims = sorted(map(lambda ds: set(ds) - all_coord_names, dims), key=len)
        raise ValueError(
            f"In '{klass_name}', field '{field_name}' has"
            f" undefined coordinates, consider defining {required_dims}!"
        )
    return valid_dims


def extract_xarray_dataclass(klass, allow_undefined_coords: bool = False):
    """
    Go through dataclass fields and interpret them according to xarray-dataclass

    Returns a tuple of coordinates, data variables and attributes

    :param allow_undefined_coords: Allow data variables with dimensions
      that do not have associated coordinates (e.g. for data arrays).
    """

    field_docstrings = extract_field_docstrings(klass)

    # Collect type hints, identify coordinates
    type_hints = get_type_hints(klass, include_extras=True)
    if allow_undefined_coords:

        def check_invalid_dims(dims, field_name):
            return dims

    else:
        all_coord_names = {
            field.name
            for field in dataclasses.fields(klass)
            if get_role(type_hints[field.name]) == Role.COORD
        }

        def check_invalid_dims(dims, field_name):
            return _check_invalid_dims(
                dims, all_coord_names, klass.__name__, field_name
            )

    # Go through attributes, collecting coordinates, data variables and
    # attributes
    coordinates = []
    data_vars = []
    attributes = []
    for field in dataclasses.fields(klass):
        # Get field "role" (coordinate / data variable / attribute) from its
        # type hint
        typ = type_hints[field.name]
        role = get_role(typ)

        # Is it an attribute?
        if role == Role.ATTR:
            attributes.append(
                AttrSchemaRef(
                    name=field.name,
                    typ=get_annotated(typ),
                    optional=is_optional(typ),
                    default=field.default,
                    docstring=field_docstrings.get(field.name),
                )
            )
            continue

        # Only other allowed option is some kind of data array
        if role == Role.COORD:
            is_coord = True
        elif role == Role.DATA:
            is_coord = False
        else:
            raise ValueError(
                f"Expected field '{field.name}' in '{klass.__name__}' "
                "to be annotated with either Coord, Data or Attr!"
            )

        # Defined using a dataclass, i.e. Coordof/Dataof?
        dataclass = typing.get_args(get_annotated(typ))[0]
        if dataclasses.is_dataclass(dataclass):
            # Recursively get array schema for data class
            arr_schema = xarray_dataclass_to_array_schema(dataclass)
            arr_schema_fields = {
                f.name: getattr(arr_schema, f.name)
                for f in dataclasses.fields(ArraySchema)
            }

            # Check for undefined coordinates
            arr_schema_fields["dimensions"] = check_invalid_dims(
                arr_schema_fields["dimensions"], field.name
            )

            # Repackage as reference
            schema_ref = ArraySchemaRef(
                name=field.name,
                optional=is_optional(typ),
                default=field.default,
                docstring=field_docstrings.get(field.name),
                **arr_schema_fields,
            )

        else:
            # Get dimensions and dtypes
            dims = get_dims(typ)
            types = get_types(typ)

            # Is types a (single) dataclass?
            if len(types) == 1 and dataclasses.is_dataclass(types[0]):
                # Recursively get array schema for data class
                arr_schema = xarray_dataclass_to_array_schema(types[0])

                # Prepend dimensions to array schema
                combined_dimensions = [
                    dims1 + dims2
                    for dims1, dims2 in itertools.product(dims, arr_schema.dimensions)
                ]

                # Repackage as reference
                arr_schema_fields = {
                    f.name: getattr(arr_schema, f.name)
                    for f in dataclasses.fields(ArraySchema)
                }

                arr_schema_fields["dimensions"] = check_invalid_dims(
                    combined_dimensions, field.name
                )
                schema_ref = ArraySchemaRef(
                    name=field.name,
                    optional=is_optional(typ),
                    default=field.default,
                    docstring=field_docstrings.get(field.name),
                    **arr_schema_fields,
                )
            else:
                # Assume that it's an "inline" declaration using "Coord"/"Data"
                schema_ref = ArraySchemaRef(
                    name=field.name,
                    optional=is_optional(typ),
                    default=field.default,
                    docstring=field_docstrings.get(field.name),
                    schema_name=None,
                    dimensions=check_invalid_dims(dims, field.name),
                    dtypes=[numpy.dtype(typ) for typ in types],
                    coordinates=[],
                    attributes=[],
                    class_docstring=None,
                    data_docstring=None,
                )

        if is_coord:
            # Make sure that it is valid to use as a coordinate - i.e. we don't
            # have "recursive" (?!) coordinate definitions
            if not schema_ref.is_coord():
                raise ValueError(
                    f"In '{klass.__name__}', field '{field.name}':"
                    f" {schema_ref.schema_name} has coordinates, and"
                    " therefore can't be used as coordinate array itself!"
                )

            coordinates.append(schema_ref)
        else:
            data_vars.append(schema_ref)

    return coordinates, data_vars, attributes


def xarray_dataclass_to_array_schema(klass):
    """
    Convert an xarray-dataclass schema dataclass to an ArraySchema

    This should work on any class that we would derive from AsDataArray, or
    refer to using CoordOf or DataOf
    """

    # Cached?
    if hasattr(klass, "__xradio_array_schema"):
        return klass.__xradio_array_schema

    # Extract from data class
    coordinates, data_vars, attributes = extract_xarray_dataclass(klass, True)

    # For a dataclass there must be exactly one data variable
    if not data_vars:
        raise ValueError(
            f"Found no data declaration in (supposed) data array class {klass.__name__}!"
        )
    if len(data_vars) > 1:
        raise ValueError(
            f"Found multiple data variables ({', '.join(v.name for v in data_vars)})"
            f" in supposed data array class {klass.__name__}!"
        )

    # Check that data variable is named "data". This is important for this to
    # match up with parameters to xarray.DataArray() later (see bases.AsArray)
    if data_vars[0].name != "data":
        raise ValueError(
            f"Data variable in data array class {klass.__name__} "
            f'must be called "data", not {data_vars[0].name}!'
        )

    # Make class
    schema = ArraySchema(
        schema_name=f"{klass.__module__}.{klass.__qualname__}",
        dimensions=data_vars[0].dimensions,
        dtypes=data_vars[0].dtypes,
        coordinates=coordinates,
        attributes=attributes,
        class_docstring=inspect.cleandoc(klass.__doc__),
        data_docstring=data_vars[0].docstring,
    )
    klass.__xradio_array_schema = schema
    return schema


def xarray_dataclass_to_dataset_schema(klass):
    """
    Convert an xarray-dataclass schema dataclass to an ArraySchema

    This should work on any class that we would derive from AsDataArray, or
    refer to using CoordOf or DataOf
    """

    # Cached?
    if hasattr(klass, "__xradio_dataset_schema"):
        return klass.__xradio_dataset_schema

    # Extract from data class
    coordinates, data_vars, attributes = extract_xarray_dataclass(klass)

    # Collect all possible dimensions (use dict instead of set so we retain
    # ordering)
    all_dvars = coordinates + data_vars
    all_dimensions = dict()
    for dvar in all_dvars:
        for dims in dvar.dimensions:
            all_dimensions.update(dict.fromkeys(dims))

    # Identify dimensions that are always required, because it is mentioned in
    # all possible dimensionalities on a non-optional data variable
    req_dimensions = set()
    for dvar in all_dvars:
        if not dvar.optional:
            req_dimensions |= dvar.required_dimensions()

    # Build powerset of all options
    # (Theoretically we could try to be fancy here and filter out options
    #  that "make no sense". E.g. if there are some dimensions A and B
    #  that are always used together, but never appear individually,
    #  we should not list A and B separately as options. The test would
    #  be to check whether for every dimensionality option there exists
    #  a mapping of data variables to allowed dimensions [possibly empty
    #  if the data variable is optional], such that the superset of all
    #  dimensions matches. Likely overkill for now.)
    opt_dimensions = all_dimensions.keys() - req_dimensions
    dimensions = [req_dimensions]
    for dim in opt_dimensions:
        dimensions += [dims | set([dim]) for dims in dimensions]

    # Reorder consistently / convert to lists
    dimensions = [[dim for dim in all_dimensions if dim in dims] for dims in dimensions]

    # Make class
    schema = DatasetSchema(
        schema_name=f"{klass.__module__}.{klass.__qualname__}",
        dimensions=dimensions,
        coordinates=coordinates,
        data_vars=data_vars,
        attributes=attributes,
        class_docstring=inspect.cleandoc(klass.__doc__),
    )
    klass.__xradio_dataset_schema = schema
    return schema


def xarray_dataclass_to_dict_schema(klass):
    """
    Convert an xarray-dataclass style schema dataclass to an DictSchema

    This should work on any class annotated with :py:func:`~xradio.schema.bases.dict_schema`
    """

    # Cached?
    if hasattr(klass, "__xradio_dict_schema"):
        return klass.__xradio_dict_schema

    # Get docstrings and type hints
    field_docstrings = extract_field_docstrings(klass)
    type_hints = get_type_hints(klass, include_extras=True)
    attributes = []
    for field in dataclasses.fields(klass):
        typ = type_hints[field.name]

        # Handle optional value: Strip "None" from the types
        optional = is_optional(typ)
        if optional:
            typs = [typ for typ in get_args(typ) if typ is not None.__class__]
            if len(typs) == 1:
                typ = typs[0]
            else:
                typ = typing.Union.__getitem__[tuple(typs)]

        attributes.append(
            AttrSchemaRef(
                name=field.name,
                typ=typ,
                optional=optional,
                default=field.default,
                docstring=field_docstrings.get(field.name),
            )
        )

    # Return
    schema = DictSchema(
        schema_name=f"{klass.__module__}.{klass.__qualname__}",
        attributes=attributes,
        class_docstring=inspect.cleandoc(klass.__doc__),
    )
    klass.__xradio_dict_schema = schema
    return schema
