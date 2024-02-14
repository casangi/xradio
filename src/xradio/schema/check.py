import dataclasses
import typing
import inspect
import functools

import xarray
import numpy

from xradio.schema import (
    metamodel,
    xarray_dataclass_to_array_schema,
    xarray_dataclass_to_dataset_schema,
    AsDataset,
    AsDataArray,
)


@dataclasses.dataclass
class SchemaIssue:
    """
    Representation of an issue found in a schema check

    As schemas can be quite big, ``path`` can be used to precisely locate the
    source of the issue.
    """

    path: typing.List[typing.Tuple[str, str]]
    """Path to offending data item, using pairs of (entity type, entity name).
    Entity types can be ``data_var``, ``coord`` or ``attr``.

    Example: ``[('data_var', 'foo'), ('coord','bar'), ('attr', 'asd')]``
    refers to ``obj.data_vars['foo'].cords['bar'].attrs['asd']``
    """
    message: str
    """
    Explanation of the issue
    """
    found: typing.Optional[typing.Any] = None
    """
    What was found. Can be any type (type, dtype, value)
    """
    expected: typing.Optional[typing.List[typing.Any]] = None
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
            err += f" (expected: {options} found: {repr(self.found)})"
        return err


class SchemaIssues(Exception):
    """
    List of issues found in a schema check

    Can be thrown as an exception, so we can report on multiple schema issues
    in one go.
    """

    issues: [SchemaIssue]
    """List of issues found"""

    def __init__(self, issues=None):
        if issues is None:
            self.issues = []
        elif isinstance(issues, SchemaIssues):
            self.issues = issues.issues
        else:
            self.issues = list(issues)

    def at_path(self, elem: str, ix: typing.Optional[str] = None) -> "SchemaIssues":
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

    def expect(
        self, elem: typing.Optional[str] = None, ix: typing.Optional[str] = None
    ):
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


def check_array(array: xarray.DataArray, schema: metamodel.ArraySchema) -> SchemaIssues:
    """
    Check whether an xarray DataArray conforms to a schema

    :param array: DataArray to check
    :param schema: Schema to check against
    :returns: List of :py:class:`SchemaIssue`s found
    """

    # Check that this is actually a DataArray
    if not isinstance(array, xarray.DataArray):
        raise TypeError(
            f"check_array: Expected xarray.DataArray, but got {type(array)}!"
        )
    if type(schema) == type and issubclass(schema, AsDataArray):
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
    dataset: xarray.Dataset, schema: metamodel.DatasetSchema
) -> SchemaIssues:
    """
    Check whether an xarray DataArray conforms to a schema

    :param array: DataArray to check
    :param schema: Schema to check against (possibly as dataclass)
    :returns: List of :py:class:`SchemaIssue`s found
    """

    # Check that this is actually a Dataset
    if not isinstance(dataset, xarray.Dataset):
        raise TypeError(
            f"check_dataset: Expected xarray.Dataset, but got {type(dataset)}!"
        )
    if isinstance(schema, AsDataset):
        schema = xarray_dataclass_to_dataset_schema(schema)
    if not isinstance(schema, metamodel.DatasetSchema):
        raise TypeError(
            f"check_dataset: Expected DatasetSchema, but got {type(schema)}!"
        )

    # Check dimensions. Order does not matter on datasets
    issues = check_dimensions(dataset.dims, schema.dimensions, check_order=False)

    # Check attributes
    issues += check_attributes(dataset.attrs, schema.attributes)

    # Check coordinates
    issues += check_data_vars(dataset.coords, schema.coordinates, "coords")

    # Check data variables
    issues += check_data_vars(dataset.data_vars, schema.data_vars, "data_vars")

    return issues


def check_dimensions(
    dims: [str], expected: [[str]], check_order: bool = True
) -> SchemaIssues:
    """
    Check whether a dimension list conforms to a schema

    :param array: Dimension list to check
    :param schema: Expected possibilities for dimension list
    :param check_order: Whether to check order of dimensions
    :returns: List of :py:class:`SchemaIssue`s found
    """

    # Find a dimension list that matches
    dims_set = set(dims)
    best = None
    best_diff = 0
    for exp_dims in expected:
        exp_dims_set = set(exp_dims)

        # No match? Continue, but take note of best match
        if exp_dims_set != dims_set:
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
    hint_remove = [f"'{hint}'" for hint in dims_set - best]
    hint_add = [f"'{hint}'" for hint in best - dims_set]
    if hint_remove and hint_add:
        message = f"Unexpected coordinates, replace {','.join(hint_add)} by {','.join(hint_remove)}?"
    elif hint_remove:
        message = f"Superflous coordinate {','.join(hint_remove)}?"
    elif hint_add:
        message = f"Missing dimension {','.join(hint_add)}!"
    else:
        message = f"Unexpected dimensions/coordinates!"
    return SchemaIssues(
        [
            SchemaIssue(
                path=[("dims", None)],
                message=message,
                found=dims,
                expected=expected,
            )
        ]
    )


def check_dtype(dtype: numpy.dtype, expected: [numpy.dtype]) -> SchemaIssues:
    """
    Check whether a numpy dtype conforms to a schema

    :param dtype: Numeric type to check
    :param schema: Expected possibilities for dtype
    :returns: List of :py:class:`SchemaIssue`s found
    """

    for exp_dtype in expected:
        if dtype == exp_dtype:
            return SchemaIssues()

    # Not sure there's anything more helpful that we can do here? Any special
    # cases worth considering?
    return SchemaIssues(
        [
            SchemaIssue(
                path=[("dtype", None)],
                message="Wrong numpy dtype",
                found=dtype,
                expected=expected,
            )
        ]
    )


def check_attributes(
    attrs: typing.Dict[str, typing.Any],
    attrs_schema: typing.List[metamodel.AttrSchemaRef],
) -> SchemaIssues:
    """
    Check whether an attribute set conforms to a schema

    :param attrs: Dictionary of attributes
    :param attrs_schema: Expected schemas
    :returns: List of :py:class:`SchemaIssue`s found
    """

    issues = SchemaIssues()
    for attr_schema in attrs_schema:

        # Attribute missing? Note that a value of "None" is equivalent for the
        # purpose of the check
        val = attrs.get(attr_schema.name)
        if val is None:
            if not attr_schema.optional:
                issues.add(
                    SchemaIssue(
                        path=[("attrs", attr_schema.name)],
                        message=f"Required attribute {attr_schema.name} is missing!",
                    )
                )
            continue

        # Check attribute value
        issues += _check_value_union(val, attr_schema.typ).at_path(
            "attrs", attr_schema.name
        )

    # Extra attributes are always okay

    return issues


def check_data_vars(
    data_vars: typing.Dict[str, xarray.DataArray],
    data_vars_schema: typing.List[metamodel.ArraySchemaRef],
    data_var_kind: str,
) -> SchemaIssues:
    """
    Check whether an data variable set conforms to a schema

    As data variables are data arrays, this will recurse into checking the
    array schemas

    :param data_vars: Dictionary(-like) of data_varinates
    :param data_vars_schema: Expected schemas
    :param datavar_kind: Either 'coords' or 'data_vars'
    :returns: List of :py:class:`SchemaIssue`s found
    """

    assert data_var_kind in ["coords", "data_vars"]

    issues = SchemaIssues()
    for data_var_schema in data_vars_schema:

        # Data_Varinate missing?
        data_var = data_vars.get(data_var_schema.name)
        if data_var is None:
            if not data_var_schema.optional:
                if data_var_kind == "coords":
                    message = (
                        f"Required coordinate '{data_var_schema.name}' is missing!"
                    )
                else:
                    message = (
                        f"Required data variable '{data_var_schema.name}' is missing!"
                    )
                issues.add(
                    SchemaIssue(
                        path=[(data_var_kind, data_var_schema.name)], message=message
                    )
                )
            continue

        # Check array schema
        issues += check_array(data_var, data_var_schema).at_path(
            data_var_kind, data_var_schema.name
        )

    # Extra data_varinates / data variables are always okay

    return issues


def _check_value(val, ann):
    """
    Check whether value satisfies annotation

    If the annotation is a data array or dataset schema, it will be checked.

    :param val: Value to check
    :param ann: Type annotation of value
    :returns: Schema issues
    """

    # Is supposed to be a data array?
    if type(ann) == type and issubclass(ann, AsDataArray):
        if not isinstance(val, xarray.DataArray):
            return SchemaIssues(
                [
                    SchemaIssue(
                        path=[],
                        message="Unexpected type",
                        expected=[xarray.DataArray],
                        found=type(val),
                    )
                ]
            )
        else:
            return check_array(val, ann)

    # Is supposed to be a dataset?
    if type(ann) == type and issubclass(ann, AsDataset):
        if not isinstance(val, xarray.Dataset):
            return SchemaIssues(
                [
                    SchemaIssue(
                        path=[],
                        message="Unexpected type",
                        expected=[xarray.DataArray],
                        found=type(val),
                    )
                ]
            )
        else:
            return check_dataset(val, ann)

    # Otherwise straight type check (TODO - be more fancy, possibly by
    # importing from Typeguard module? Don't want to overdo it...)
    if not isinstance(val, ann):
        return SchemaIssues(
            [
                SchemaIssue(
                    path=[], message="Unexpected type", expected=[ann], found=type(val)
                )
            ]
        )

    return SchemaIssues()


def _check_value_union(val, ann):
    """
    Check whether value satisfies annotations, including union types

    If the annotation is a data array or dataset schema, it will be checked.

    :param val: Value to check
    :param ann: Type annotation of value
    :returns: Schema issues
    """

    if ann is None or ann is inspect.Signature.empty:
        return SchemaIssues()

    # Account for union types (this especially catches "Optional")
    if typing.get_origin(ann) is typing.Union:
        options = typing.get_args(ann)
    else:
        options = [ann]

    # Go through options, try to find one without issues
    args_issues = None
    okay = False
    for option in options:
        arg_issues = _check_value(val, option)
        # We can immediately return if we find no issues with
        # some schema check
        if not arg_issues:
            return SchemaIssues()
        if args_issues is None:
            args_issues = arg_issues

        # Fancy merging of expected options (for "unexpected type")
        elif (
            len(args_issues) == 1
            and len(arg_issues) == 1
            and args_issues[0].message == arg_issues[0].message
        ):

            args_issues[0].expected += arg_issues[0].expected

    # Return representative issues list
    if not args_issues:
        raise ValueError("Empty union set?")
    return args_issues


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
        __tracebackhide__ = True

        # Bind parameters, collect (potential) issues
        bound = signature.bind(*args, **kwargs)
        issues = SchemaIssues()
        for arg, val in bound.arguments.items():
            if arg not in parameters_to_check:
                continue

            # Get annotation
            issues += _check_value_union(val, anns.get(arg)).at_path(arg)

        # Any issues found? raise
        issues.expect()

        # Execute function
        result = fn(*args, **kwargs)

        # Check return
        if check_return:
            issues = _check_value_union(val, signature.return_annotation)
            issues.at_path("return").expect()

        # Check return value
        return result

    return _check_fn
