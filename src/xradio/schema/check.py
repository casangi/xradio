import builtins
import dataclasses
import functools
import inspect
import re
import typing
import warnings

import numpy
import xarray

# Import the xarray_dataclass_to_* functions from their defining module (not
# from the xradio.schema package namespace): this module is imported while
# xradio/schema/__init__.py is still initializing, so package-level re-exports
# are not necessarily bound yet.
from xradio.schema import bases, metamodel
from xradio.schema.dataclass import (
    value_schema,
    xarray_dataclass_to_array_schema,
    xarray_dataclass_to_dataset_schema,
    xarray_dataclass_to_dict_schema,
)
from xradio.schema.metamodel import AttrSchemaRef, ValueSchema


@dataclasses.dataclass
class SchemaIssue:
    """
    Representation of an issue found in a schema check

    As schemas can be quite big, ``path`` can be used to precisely locate the
    source of the issue.
    """

    path: list[tuple[str, str]]
    """Path to offending data item, using pairs of (entity type, entity name).
    Entity types can be ``data_var``, ``coord`` or ``attr``.

    Example: ``[('data_var', 'foo'), ('coord','bar'), ('attr', 'asd')]``
    refers to ``obj.data_vars['foo'].cords['bar'].attrs['asd']``
    """
    message: str
    """
    Explanation of the issue
    """
    found: typing.Any | None = None
    """
    What was found. Can be any type (type, dtype, value)
    """
    expected: list[typing.Any] | None = None
    """
    List of expected values. Can be any type (type, dtype, value)
    """

    def path_str(self):
        strs = []
        for path, ix in self.path:
            if ix is None or ix == "":
                strs.append(path)
            else:
                strs.append(f"{path}['{ix}']")
        return ".".join(strs)

    def __repr__(self):
        err = f"Schema issue with {self.path_str()}: {self.message}"
        if self.expected is not None:
            options = " or ".join(repr(option) for option in self.expected)
            if self.found is not None:
                err += f" (expected: {options} found: {repr(self.found)})"
            else:
                err += f" (expected: {options})"
        return err


class SchemaIssues(Exception):
    """
    List of issues found in a schema check

    Can be thrown as an exception, so we can report on multiple schema issues
    in one go.
    """

    issues: list[SchemaIssue]
    """List of issues found"""

    def __init__(self, issues=None):
        if issues is None:
            self.issues = []
        elif isinstance(issues, SchemaIssues):
            self.issues = list(issues.issues)
        else:
            self.issues = list(issues)

    def at_path(self, elem: str, ix: str | None = None) -> "SchemaIssues":
        for issue in self.issues:
            issue.path.insert(0, (elem, ix))
        return self

    def add(self, issue: SchemaIssue):
        self.issues.append(issue)

    def __iadd__(self, other: "SchemaIssues"):
        self.issues += other.issues
        return self

    def __add__(self, other: "SchemaIssues"):
        new_issues = SchemaIssues(self)
        new_issues += other
        return new_issues

    def __len__(self):
        return len(self.issues)

    def __getitem__(self, ix):
        return self.issues[ix]

    def __str__(self):
        if not self.issues:
            return "No schema issues found"
        else:
            issues_string = "\n * ".join(repr(issue) for issue in self.issues)
            return f"\n * {issues_string}"

    def __repr__(self):
        return f"SchemaIssues({str(self)})"

    def expect(self, elem: str | None = None, ix: str | None = None):
        """
        Raises this object if issues were found

        :param elem: If given, will be added to path
        :param ix: If given, will be added to path
        :raises: SchemaIssues
        """
        if elem is not None:
            self.at_path(elem, ix)

        # Hide this function in pytest tracebacks
        __tracebackhide__ = True
        if self.issues:
            raise self


def check_array(
    array: xarray.DataArray, schema: type | metamodel.ArraySchema
) -> SchemaIssues:
    """
    Check whether an xarray DataArray conforms to a schema

    :param array: DataArray to check
    :param schema: Schema to check against
    :returns: :py:class:`SchemaIssues` found
    """

    # Check that this is actually a DataArray
    if not isinstance(array, xarray.DataArray):
        raise TypeError(
            f"check_array: Expected xarray.DataArray, but got {type(array)}!"
        )
    if bases.is_dataarray_schema(schema):
        schema = xarray_dataclass_to_array_schema(schema)
    if not isinstance(schema, metamodel.ArraySchema):
        raise TypeError(f"check_array: Expected ArraySchema, but got {type(schema)}!")

    # Check dimensions
    issues = check_dimensions(array.dims, schema.dimensions)

    # Check type
    issues += check_dtype(array.dtype, schema.dtypes)

    # Check attributes
    issues += check_attributes(array.attrs, schema.attributes)

    # Check coordinates
    issues += check_data_vars(array.coords, schema.coordinates, "coords")

    return issues


def check_dataset(
    dataset: xarray.Dataset,
    schema: type | metamodel.DatasetSchema,
    allow_superflous_dims: set[str] = frozenset(),
) -> SchemaIssues:
    """
    Check whether an xarray Dataset conforms to a schema

    :param dataset: Dataset to check
    :param schema: Dataset schema to check against (a class decorated with
       :py:func:`~xradio.schema.bases.xarray_dataset_schema`, for example
       :py:class:`xradio.image.schema.ImageXds`, or a
       :py:class:`~xradio.schema.metamodel.DatasetSchema`)
    :param allow_superflous_dims: Dimensions that may be present although the
       schema does not mention them
    :returns: :py:class:`SchemaIssues` found
    :raises TypeError: If ``dataset`` is not a Dataset or ``schema`` is not a
       dataset schema
    """

    # Check that this is actually a Dataset
    if not isinstance(dataset, xarray.Dataset):
        raise TypeError(
            f"check_dataset: Expected xarray.Dataset, but got {type(dataset)}!"
        )
    if bases.is_dataset_schema(schema):
        schema = xarray_dataclass_to_dataset_schema(schema)
    if not isinstance(schema, metamodel.DatasetSchema):
        raise TypeError(_not_a_dataset_schema_message(schema))

    # Check dimensions. Order does not matter on datasets
    issues = check_dimensions(
        dataset.dims,
        schema.dimensions,
        check_order=False,
        allow_superflous=allow_superflous_dims,
    )

    # Check attributes
    issues += check_attributes(dataset.attrs, schema.attributes)

    # Check coordinates
    issues += check_data_vars(dataset.coords, schema.coordinates, "coords")

    # Check data variables
    issues += check_data_vars(dataset.data_vars, schema.data_vars, "data_vars")

    return issues


def _not_a_dataset_schema_message(schema: typing.Any) -> str:
    """
    Explain why ``schema`` cannot be used as a dataset schema

    Classes that are not dataset schemas are named explicitly. If a
    registered dataset schema has the same class name (for example the image
    accessor ``xradio.image.ImageXds`` and the schema
    ``xradio.image.schema.ImageXds``), it is suggested.

    :param schema: Object passed as the schema
    :returns: Error message
    """

    if not isinstance(schema, type):
        return f"check_dataset: Expected DatasetSchema, but got {type(schema)}!"

    class_name = f"{schema.__module__}.{schema.__qualname__}"
    message = (
        f"check_dataset: Expected DatasetSchema, but got the class {class_name}, "
        "which is not a dataset schema (a class decorated with "
        "xradio.schema.bases.xarray_dataset_schema)!"
    )
    if bases.is_dataarray_schema(schema):
        message += " It is a data array schema: use check_array to check arrays."
    elif bases.is_dict_schema(schema):
        message += " It is a dictionary schema: use check_dict to check dictionaries."
    suggestions = sorted(
        {
            registered.schema_name
            for registered in _DATASET_TYPES.values()
            if registered.schema_name.rsplit(".", 1)[-1] == schema.__qualname__
            and registered.schema_name != class_name
        }
    )
    if suggestions:
        message += f" Did you mean the dataset schema {' or '.join(suggestions)}?"
    return message


def check_dimensions(
    dims: [str],
    expected: [[str]],
    check_order: bool = True,
    allow_superflous: set[str] = frozenset(),
) -> SchemaIssues:
    """
    Check whether a dimension list conforms to a schema

    :param array: Dimension list to check
    :param schema: Expected possibilities for dimension list
    :param check_order: Whether to check order of dimensions
    :returns: :py:class:`SchemaIssues` found
    """

    # Find a dimension list that matches
    dims_set = set(dims)
    best = None
    best_diff = 0
    for exp_dims in expected:
        exp_dims_set = set(exp_dims)

        # No match? Continue, but take note of best match
        if exp_dims_set - allow_superflous != dims_set - allow_superflous:
            diff = len(dims_set.symmetric_difference(exp_dims_set))
            if best is None or diff < best_diff:
                best = exp_dims_set
                best_diff = diff
            continue

        # Check that the order matches
        if not check_order or tuple(exp_dims) == tuple(dims):
            return SchemaIssues()

        return SchemaIssues(
            [
                SchemaIssue(
                    path=[("dims", None)],
                    message="Dimensions are in the wrong order! Consider transpose()",
                    found=list(dims),
                    expected=[exp_dims],
                )
            ]
        )

    # Dimensionality not supported - try to give a helpful suggestion
    hint_remove = [f"'{hint}'" for hint in (dims_set - best) - allow_superflous]
    hint_add = [f"'{hint}'" for hint in best - dims_set]
    if hint_remove and hint_add:
        message = f"Unexpected coordinates, replace {','.join(hint_remove)} by {','.join(hint_add)}?"
    elif hint_remove:
        message = f"Superfluous coordinate {','.join(hint_remove)}?"
    elif hint_add:
        message = f"Missing dimension {','.join(hint_add)}!"
    else:
        message = "Unexpected dimensions/coordinates!"
    return SchemaIssues(
        [
            SchemaIssue(
                path=[("dims", None)],
                message=message,
                found=list(dims),
                expected=list(expected),
            )
        ]
    )


def check_dtype(dtype: numpy.dtype, expected: [numpy.dtype]) -> SchemaIssues:
    """
    Check whether a numpy dtype conforms to a schema

    :param dtype: Numeric type to check
    :param schema: Expected possibilities for dtype
    :returns: :py:class:`SchemaIssues` found
    """

    for exp_dtype_str in expected:
        exp_dtype = numpy.dtype(exp_dtype_str)
        # If the expected dtype has no size (e.g. "U", a.k.a. a string of
        # arbitrary length), we don't check itemsize, only kind.
        if (
            dtype.kind == exp_dtype.kind
            and exp_dtype.itemsize == 0
            or exp_dtype == dtype
        ):
            return SchemaIssues()

    # Not sure there's anything more helpful that we can do here? Any special
    # cases worth considering?
    return SchemaIssues(
        [
            SchemaIssue(
                path=[("dtype", None)],
                message="Wrong numpy dtype",
                found=dtype.str,
                expected=list(expected),
            )
        ]
    )


def check_attributes(
    attrs: dict[str, typing.Any],
    attrs_schema: list[metamodel.AttrSchemaRef],
    attr_kind: str = "attrs",
) -> SchemaIssues:
    """
    Check whether an attribute set conforms to a schema

    :param attrs: Dictionary of attributes
    :param attrs_schema: Expected schemas
    :returns: :py:class:`SchemaIssues` found
    """

    issues = SchemaIssues()
    for attr_schema in attrs_schema:
        # Attribute missing is equivalent to a value of "None" is
        # equivalent for the purpose of the check
        val = attrs.get(attr_schema.name)
        if val is None:
            if not attr_schema.optional:
                issues.add(
                    SchemaIssue(
                        path=[(attr_kind, attr_schema.name)],
                        message="Non-optional attribute is missing!",
                        found=None,
                        expected=[attr_schema.type],
                    )
                )
            continue

        # Check actual value
        issues += _check_value(val, attr_schema).at_path(attr_kind, attr_schema.name)

    # Extra attributes are always okay

    return issues


def check_data_vars(
    data_vars: dict[str, xarray.DataArray],
    data_vars_schema: list[metamodel.ArraySchemaRef],
    data_var_kind: str,
) -> SchemaIssues:
    """
    Check whether a data variable set conforms to a schema

    As data variables are data arrays, this will recurse into checking the
    array schemas

    :param data_vars: Dictionary(-like) of data_varinates
    :param data_vars_schema: Expected schemas
    :param datavar_kind: Either 'coords' or 'data_vars'
    :returns: :py:class:`SchemaIssues` found
    """

    assert data_var_kind in ["coords", "data_vars"]

    issues = SchemaIssues()
    for data_var_schema in data_vars_schema:
        if _allows_multiple_versions(data_var_schema):
            data_vars_names = _matching_data_var_names(data_vars, data_var_schema)
        else:
            data_vars_names = [data_var_schema.name]

        if (len(data_vars_names) == 0) and not data_var_schema.optional:
            data_vars_names = [data_var_schema.name]

        for data_var_name in data_vars_names:
            data_var = data_vars.get(data_var_name)

            if data_var is None:
                if not data_var_schema.optional:
                    if data_var_kind == "coords":
                        message = (
                            f"Required coordinate '{data_var_schema.name}' is missing!"
                        )
                    else:
                        message = (
                            f"Required data variable '{data_var_schema.name}' is missing "
                            f"(have {','.join(data_vars)})!"
                        )
                    issues.add(
                        SchemaIssue(
                            path=[(data_var_kind, data_var_schema.name)],
                            message=message,
                        )
                    )
                continue

            # Check array schema. Report issues under the name of the
            # variable that was checked, which differs from the schema name
            # for versions such as "SKY_RESIDUAL"
            issues += check_array(data_var, data_var_schema).at_path(
                data_var_kind, data_var_name
            )

    # Extra data_varinates / data variables are always okay

    return issues


def _allows_multiple_versions(data_var_schema: metamodel.ArraySchemaRef) -> bool:
    """
    Whether a data variable schema allows multiple versions

    :param data_var_schema: Schema of the data variable
    :returns: Value of its ``allow_multiple_versions`` attribute default
    """

    for attr in data_var_schema.attributes:
        if getattr(attr, "name", None) == "allow_multiple_versions":
            return bool(attr.default)
    return False


def _matching_data_var_names(
    data_var_names: typing.Iterable[str], data_var_schema: metamodel.ArraySchemaRef
) -> list[str]:
    """
    Names of the data variables that :py:func:`check_data_vars` checks
    against a data variable schema

    A data variable whose schema allows multiple versions (see
    ``allow_multiple_versions``) is matched by its canonical name and by the
    canonical name followed by a ``"_"`` separated suffix: ``"SKY"`` matches
    ``"SKY"`` and ``"SKY_DECONVOLVED"``, but not ``"FLAG_SKY"``. Other data
    variables are matched by their canonical name only.

    :param data_var_names: Names of the data variables present
    :param data_var_schema: Schema of the data variable
    :returns: Names of the present data variables that the schema applies to
    """

    name = data_var_schema.name
    if not _allows_multiple_versions(data_var_schema):
        return [name] if name in data_var_names else []
    return [
        data_var_name
        for data_var_name in data_var_names
        if isinstance(data_var_name, str)
        and (data_var_name == name or data_var_name.startswith(name + "_"))
    ]


def check_dict(dct: dict, schema: type | metamodel.DictSchema) -> SchemaIssues:
    """
    Check whether a dictionary conforms to a schema

    :param dct: Dictionary to check
    :param schema: Dictionary schema to check against
    :returns: :py:class:`SchemaIssues` found
    """

    # Check that this is actually a dictionary
    if not isinstance(dct, dict):
        raise TypeError(f"check_dict: Expected dictionary, but got {type(dct)}!")
    if bases.is_dict_schema(schema):
        schema = xarray_dataclass_to_dict_schema(schema)
    if not isinstance(schema, metamodel.DictSchema):
        raise TypeError(f"check_dict: Expected DictSchema, but got {type(schema)}!")

    # Check attributes
    return check_attributes(dct, schema.attributes, attr_kind="")


def _check_value(val: typing.Any, schema: metamodel.ValueSchema):
    """
    Check whether value satisfies annotation

    If the annotation is a data array or dataset schema, it will be checked.

    :param val: Value to check
    :param schema: Schema of value
    :returns: Schema issues
    """

    # Unspecified?
    if schema.type is None:
        return SchemaIssues()

    # Optional?
    if schema.optional and val is None:
        return SchemaIssues()

    # Is supposed to be a data array?
    if schema.type == "dataarray":
        # Attempt to convert dictionaries automatically
        if isinstance(val, dict):
            try:
                val = xarray.DataArray.from_dict(val)
            except ValueError as e:
                expected = [xarray.DataArray]
                if schema.optional:
                    expected.append(type(None))
                return SchemaIssues(
                    [
                        SchemaIssue(
                            path=[], message=str(e), expected=expected, found=type(val)
                        )
                    ]
                )
            except TypeError as e:
                expected = [xarray.DataArray]
                if schema.optional:
                    expected.append(type(None))
                return SchemaIssues(
                    [
                        SchemaIssue(
                            path=[], message=str(e), expected=expected, found=type(val)
                        )
                    ]
                )

        if isinstance(val, xarray.DataArray):
            return check_array(val, schema.array_schema)

        expected = [xarray.DataArray]
        if schema.optional:
            expected.append(type(None))
        return SchemaIssues(
            [
                SchemaIssue(
                    path=[],
                    message=f"{type(val).__name__} is not an xarray.DataArray!",
                    expected=expected,
                    found=type(val),
                )
            ]
        )

    # Is supposed to be a dictionary?
    elif schema.type == "dict":
        if not isinstance(val, dict):
            expected = [dict]
            if schema.optional:
                expected.append(type(None))
            return SchemaIssues(
                [
                    SchemaIssue(
                        path=[],
                        message=f"{type(val).__name__} is not a dictionary!",
                        expected=expected,
                        found=type(val),
                    )
                ]
            )
        else:
            return check_dict(val, schema.dict_schema)

    elif schema.type == "list[str]":
        if not isinstance(val, list):
            expected = [list[str]]
            if schema.optional:
                expected.append(type(None))
            return SchemaIssues(
                [
                    SchemaIssue(
                        path=[],
                        message=f"{type(val).__name__} is not a list!",
                        expected=expected,
                        found=type(val),
                    )
                ]
            )
        if any(type(v) is not str for v in val):
            expected = [list[str]]
            if schema.optional:
                expected.append(type(None))
            return SchemaIssues(
                [
                    SchemaIssue(
                        path=[],
                        message=f"{type(val).__name__} is not a list of strings!",
                        expected=expected,
                        found=type(val),
                    )
                ]
            )
    elif schema.type == "list[float]":
        if not isinstance(val, list) or any(
            not isinstance(v, int | float) or isinstance(v, bool) for v in val
        ):
            expected = [list[float]]
            if schema.optional:
                expected.append(type(None))
            return SchemaIssues(
                [
                    SchemaIssue(
                        path=[],
                        message=f"{type(val).__name__} is not a list of floats!",
                        expected=expected,
                        found=type(val),
                    )
                ]
            )
    elif schema.type == "list[list[float]]":
        if not isinstance(val, list) or any(
            not isinstance(row, list)
            or any(not isinstance(v, int | float) or isinstance(v, bool) for v in row)
            for row in val
        ):
            expected = [list[list[float]]]
            if schema.optional:
                expected.append(type(None))
            return SchemaIssues(
                [
                    SchemaIssue(
                        path=[],
                        message=f"{type(val).__name__} is not a list of lists of floats!",
                        expected=expected,
                        found=type(val),
                    )
                ]
            )
    elif schema.type in ["bool", "str", "int", "float"]:
        type_to_check = getattr(builtins, schema.type)
        if not isinstance(val, type_to_check):
            expected = [type_to_check]
            if schema.optional:
                expected.append(type(None))
            return SchemaIssues(
                [
                    SchemaIssue(
                        path=[],
                        message=f"{type(val).__name__} is not a {schema.type}!",
                        expected=expected,
                        found=type(val),
                    )
                ]
            )
    else:
        raise ValueError(f"Invalid typ_name in schema: {schema.type}")

    # List of literals given?
    if schema.literal is not None:
        for lit in schema.literal:
            if val == lit:
                return SchemaIssues()
        return SchemaIssues(
            [
                SchemaIssue(
                    path=[],
                    message="Disallowed literal value!",
                    expected=schema.literal,
                    found=val,
                )
            ]
        )

    return SchemaIssues()


_DATASET_TYPES = {}

EXTENSION_TYPE_PREFIX = "extension:"
"""Prefix of dataset ``type`` attributes that denote extension datasets"""

_EXTENSION_TYPE_RE = re.compile(r"^extension:[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)+$")


def is_extension_type(typ: typing.Any) -> bool:
    """
    Check whether a dataset ``type`` attribute denotes a well-formed
    extension dataset type

    Extension types allow applications to store their own datasets
    alongside xradio datasets. They have the form
    ``extension:<type>.<namespace>``, with the most specific component
    first, for example ``extension:gains.quartical``. ``<type>`` may have
    sub-kinds (``extension:delay.gains.quartical``), and ``<namespace>``
    should be a name controlled by the producer, normally its package
    name, optionally followed by a domain in normal DNS order
    (``extension:gains.quartical.sarao.ac.za``).

    Components consist of lower-case letters, digits and underscores,
    must start with a letter and are separated by ``.``. At least two
    components are required. Versions should not be part of the type, but
    given in a separate ``schema_version`` attribute.

    :param typ: Value of the ``type`` attribute
    :returns: Whether ``typ`` is a well-formed extension type
    """
    return isinstance(typ, str) and _EXTENSION_TYPE_RE.match(typ) is not None


def register_dataset_type(schema: metamodel.DatasetSchema):
    """
    Registers the given schema for usage with :py:meth:`check_datatree`

    This looks for a ``type`` attribute in the dataset schema, which
    must have a :py:class:`typing.Literal` type annotation specifying
    the type name of the dataset

    :param schema: Schema to register
    """

    # Find type attribute
    for attr in schema.attributes:
        if attr.name != "type":
            continue

        # Type should be a kind of literal
        if attr.literal is None:
            warnings.warn(
                f"In dataset schema {schema.schema_name}:"
                'Attribute "type" should be a literal!',
                stacklevel=2,
            )
            continue

        # Register type names
        for typ in attr.literal:
            assert isinstance(typ, str), (
                f"In dataset schema {schema.schema_name}:"
                'Attribute "type" should be a literal giving '
                "names of schema!"
            )
            _DATASET_TYPES[typ] = schema


def check_datatree(
    datatree: xarray.DataTree,
):
    """
    Check datatree for schema conformance

    This is the case if each contained :py:class:`~xarray.Dataset` conforms to a schema
    registed with Xradio.
    This works by looking for a ``type`` attribute in the :py:class:`~xarray.Dataset`, which
    must have a :py:class:`typing.Literal` type annotation specifying
    the name of the dataset schema.

    Datasets with an extension type (see :py:func:`is_extension_type`) are
    checked if a schema has been registered for that type, and skipped
    otherwise. Malformed extension types are reported as issues.

    :param datatree: Data to check for schema conformance
    """

    # Loop through all groups in datatree
    issues = SchemaIssues()
    for xds_name in datatree.groups:
        # Ignore any leaf without data
        node = datatree[xds_name]
        if not node.has_data:
            continue

        # Look up schema
        typ = node.attrs.get("type")
        schema = _DATASET_TYPES.get(typ)
        if schema is None and is_extension_type(typ):
            # Unregistered extension dataset: not ours to check
            continue
        if (
            schema is None
            and isinstance(typ, str)
            and typ.startswith(EXTENSION_TYPE_PREFIX)
        ):
            issues.add(
                SchemaIssue(
                    [("", xds_name)],
                    message="Malformed extension dataset type!",
                    found=typ,
                    expected=["extension:<type>.<namespace>"],
                )
            )
            continue
        if schema is None:
            issues.add(
                SchemaIssue(
                    [("", xds_name)],
                    message="Unknown dataset type!",
                    found=typ,
                    expected=None,
                )
            )
            continue

        # Determine dimensions inherited from parent
        # (they might show up as "superflous" for the child schema)
        parent_dims = frozenset()
        if node.parent is not None:
            parent_dims = set(node.parent.dims)

        # Check schema
        issues += check_dataset(node.dataset, schema, parent_dims).at_path("", xds_name)

    return issues


def schema_checked(fn, check_parameters: bool = True, check_return: bool = True):
    """
    Function decorator to check parameters and return value for
    schema conformance

    :param fn: Function to decorate
    :param check_parameters: Whether to check parameters. Can also
       pass an iterable with parameters to check
    :param check_return: Whether to check return value
    :returns: Decorated function
    """
    anns = inspect.getfullargspec(fn).annotations
    signature = inspect.signature(fn)

    if isinstance(check_parameters, bool):
        if check_parameters:
            parameters_to_check = set(signature.parameters)
        else:
            parameters_to_check = {}
    else:
        parameters_to_check = set(check_parameters)
        assert parameters_to_check.issubset(signature.parameters)

    @functools.wraps(fn)
    def _check_fn(*args, **kwargs):
        # Hide this function in pytest tracebacks
        # __tracebackhide__ = True

        # Bind parameters, collect (potential) issues
        bound = signature.bind(*args, **kwargs)
        issues = SchemaIssues()
        for arg, val in bound.arguments.items():
            if arg not in parameters_to_check:
                continue

            # Get annotation
            vschema = value_schema(anns.get(arg), "function", arg)
            pseudo_attr_schema = AttrSchemaRef(
                name=arg,
                **{
                    fld.name: getattr(vschema, fld.name)
                    for fld in dataclasses.fields(ValueSchema)
                },
            )
            issues += _check_value(val, pseudo_attr_schema).at_path(arg)

        # Any issues found? raise
        issues.expect()

        # Execute function
        result = fn(*args, **kwargs)

        # Check return
        if check_return:
            vschema = value_schema(anns.get("return"), "function", "return")
            pseudo_attr_schema = AttrSchemaRef(
                name="return",
                **{
                    fld.name: getattr(vschema, fld.name)
                    for fld in dataclasses.fields(ValueSchema)
                },
            )
            issues = _check_value(result, pseudo_attr_schema)
            issues.at_path("return").expect()

        return result

    return _check_fn
