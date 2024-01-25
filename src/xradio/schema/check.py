import dataclasses
import typing

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

    def at_path(self, elem: str, ix: str = None) -> "SchemaIssues":
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
    if isinstance(schema, AsDataArray):
        schema = xarray_dataclass_to_array_schema(schema)
    if not isinstance(schema, metamodel.ArraySchema):
        raise TypeError(
            f"check_dataset: Expected ArraySchema, but got {type(schema)}!"
        )

    # Check dimensions
    issues = check_dimensions(array.dims, schema.dimensions)

    # Check type
    issues += check_dtype(array.dtype, schema.dtypes)

    # Check attributes
    issues += check_attributes(array.attrs, schema.attributes)

    # Check coordinates
    issues += check_data_vars(array.coords, schema.coordinates, "coords")

    return issues


def check_dataset(dataset: xarray.Dataset, schema: metamodel.DatasetSchema) -> SchemaIssues:
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


def check_dimensions(dims: [str], expected: [[str]], check_order: bool = True) -> SchemaIssues:
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

        return SchemaIssues([
            SchemaIssue(
                path=[("dims", "")],
                message="Dimensions are in the wrong order! Consider transpose()",
                found=list(dims),
                expected=[exp_dims],
            )
        ])

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
    return SchemaIssues([
        SchemaIssue(
            path=[("dims", "")],
            message=message,
            found=dims,
            expected=expected,
        )
    ])


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
    return SchemaIssues([
        SchemaIssue(
            path=[("dtype", "")],
            message="Wrong numpy dtype",
            found=dtype,
            expected=expected,
        )
    ])


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

        # Is data array? Check
        if issubclass(attr_schema.typ, AsDataArray):
            issues += check_array(val, attr_schema.typ).at_path("attrs", attr_schema.name)

        # Is dataset? Check
        elif issubclass(attr_schema.typ, AsDataset):
            issues += check_dataset(val, attr_schema.typ).at_path("attrs", attr_schema.name)

        # Otherwise - check type straight
        elif type(val) != attr_schema.typ:

            issues.add(
                SchemaIssue(
                    path=[("attrs", attr_schema.name)],
                    message=f"Type mismatch!",
                    expected=[attr_schema.typ],
                    found=type(val),
                )
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
        issues += check_array(data_var, data_var_schema).at_path(data_var_kind, data_var_schema.name)

    # Extra data_varinates / data variables are always okay

    return issues
