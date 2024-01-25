from typing import get_type_hints, get_args
from .typing import get_dims, get_dtypes, get_role, Role, get_annotated, is_optional

import typing
import inspect
import ast
import dataclasses

from xradio.schema.metamodel import *


def extract_field_docstrings(klass):
    """
    Extracts docstrings of all fields from a class

    :param klass: Class to extract docstrings from
    :returns: Dictionary of attributes to clean docstrings
    """

    # Parse class body
    src = inspect.getsource(klass)
    module = ast.parse(src)

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


def extract_xarray_dataclass(klass):
    """
    Go through dataclass fields and interpret them according to xarray-dataclass

    Returns a tuple of coordinates, data variables and attributes
    """

    field_docstrings = extract_field_docstrings(klass)

    # Go through attributes, collecting coordinates, data variables and
    # attributes
    type_hints = get_type_hints(klass, include_extras=True)
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
                f"Unexpected role in '{klass.__name__}',"
                f" field '{field.name}': {get_role(typ)}"
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

            # Repackage as reference
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
                schema_name=f"{klass.__module__}.{klass.__qualname__}.{field.name}",
                dimensions=get_dims(typ),
                dtypes=get_dtypes(typ),
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

    # Extract from data class
    coordinates, data_vars, attributes = extract_xarray_dataclass(klass)

    # For a dataclass there must be exactly one data variable
    # (typically called "data", but we don't check that)
    if not data_vars:
        raise ValueError(
            f"Found no data declaration in (supposed) data darray class {klass.__name__}!"
        )
    if len(data_vars) > 1:
        raise ValueError(
            f"Found multiple data variables ({', '.join(v.name for v in data_vars)})"
            f" in supposed data darray class {klass.__name__}!"
        )

    # Make class
    return ArraySchema(
        schema_name=f"{klass.__module__}.{klass.__qualname__}",
        dimensions=data_vars[0].dimensions,
        dtypes=data_vars[0].dtypes,
        coordinates=coordinates,
        attributes=attributes,
        class_docstring=inspect.cleandoc(klass.__doc__),
        data_docstring=data_vars[0].docstring,
    )


def xarray_dataclass_to_dataset_schema(klass):
    """
    Convert an xarray-dataclass schema dataclass to an ArraySchema

    This should work on any class that we would derive from AsDataArray, or
    refer to using CoordOf or DataOf
    """

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
    return DatasetSchema(
        schema_name=f"{klass.__module__}.{klass.__qualname__}",
        dimensions=dimensions,
        coordinates=coordinates,
        data_vars=data_vars,
        attributes=attributes,
        class_docstring=inspect.cleandoc(klass.__doc__),
    )
