"""
Information bases and checks shared by the xradio schemas.

The classes here hold the dictionaries, dataset attributes and data variable
attributes that the dataset schemas have in common: who created a dataset,
the data groups, the schema version, and the attributes of the data
variables that can appear in several versions, flags among them. They are
bases, not schemas: a schema derives its own decorated class from a base,
keeping its class name, and adds or redeclares what is specific to it. The
field docstrings of the bases that the measurement set v4 schema derives
from are its text.

The measurement set v4 schema derives its creator dictionary and its
versioned and flag data variables from these bases. Its data group
dictionary and its main datasets (``VisibilityXds`` and ``SpectrumXds``)
declare ``description``, ``date`` and ``schema_version`` themselves: base
fields come first, so deriving them would change the order of their
constructor parameters, keys and attributes.

The attribute-only array bases come before the ``data`` field of the arrays
that derive from them, so their attributes follow a ``KW_ONLY`` marker:
``data`` stays the first positional constructor parameter.

The functions here check this information across a dataset:
:py:func:`check_data_groups` checks every data group and the data variables
that its roles name, :py:func:`matching_data_var_names` gives the data
variables that :py:func:`~xradio.schema.check.check_dataset` checks against a
data variable of a dataset schema, and :py:func:`check_schema_version` checks
the ``schema_version`` attribute against the version of the schema in this
xradio. :py:func:`~xradio.image.schema.check_image` uses the first two, and
check_dataset runs :py:func:`check_data_groups` for the main datasets of the
measurement set v4 schema; no check calls :py:func:`check_schema_version`
yet.
"""

from __future__ import annotations

import dataclasses
import re
import typing
from dataclasses import KW_ONLY

import xarray

from xradio.schema import bases, metamodel
from xradio.schema.check import (
    SchemaIssue,
    SchemaIssues,
    _matching_data_var_names,
    _not_a_dataset_schema_message,
    check_array,
    check_dict,
)
from xradio.schema.dataclass import (
    xarray_dataclass_to_array_schema,
    xarray_dataclass_to_dataset_schema,
)
from xradio.schema.typing import Attr

#: A semantic version, MAJOR.MINOR.PATCH (the form of the ``schema_version``
#: attribute).
_SEMANTIC_VERSION = re.compile(r"(\d+)\.(\d+)\.(\d+)")


@dataclasses.dataclass(init=True, repr=False, eq=False, frozen=True)
class CreatorDictBase:
    """Base of the creator dictionaries: the software that created a dataset."""

    software_name: str
    """ Software that created the Measurement Set (XRadio, etc.). """
    version: str
    """ Version of the software. """


@dataclasses.dataclass(init=True, repr=False, eq=False, frozen=True)
class GenericDataGroupDict:
    """Base for data group dictionaries: the keys of a data group besides the
    names of its data variables."""

    description: str | None
    """ More details about the data group. """
    date: str | None
    """ Creation date-time, in ISO 8601 format: 'YYYY-MM-DDTHH:mm:ss.SSS'. """


@dataclasses.dataclass(init=True, repr=False, eq=False, frozen=True)
class SchemaVersionAttrs:
    """Base for dataset schemas: the version of the schema that a dataset
    conforms to."""

    schema_version: Attr[str]
    """Semantic version (MAJOR.MINOR.PATCH) of the schema that the dataset
    conforms to."""


@dataclasses.dataclass(init=True, repr=False, eq=False, frozen=True)
class VersionedArrayBase:
    """Base of the data variables that can appear in several versions, named
    by their canonical name followed by an underscore separated suffix (for
    example ``VISIBILITY`` and ``VISIBILITY_MODEL``)."""

    _: KW_ONLY

    allow_multiple_versions: Attr[bool] | None = True


@dataclasses.dataclass(init=True, repr=False, eq=False, frozen=True)
class FlagArrayBase(VersionedArrayBase):
    """Base of the flag data variables of the measurement set v4 schema, whose
    flags may be bit masks."""

    _: KW_ONLY

    flag_bits: Attr[list[str]] | None = (
        "UNSPECIFIED_BIT",
        "STATIC_BIT",
        "CAM_BIT",
        "DATA_LOST_BIT",
        "INGEST_RFI_BIT",
        "PREDICTED_RFI_BIT",
        "CAL_RFI_BIT",
        "POSTPROC_BIT",
    )
    """
    Labels associated with bits used in ``FlagArray`` data. See
    :ref:`flag bits` for suggested semantics.
    """


def matching_data_var_names(
    data_var_names: typing.Iterable[str], data_var_schema: metamodel.ArraySchemaRef
) -> list[str]:
    """
    Names of the data variables that
    :py:func:`~xradio.schema.check.check_dataset` checks against a data
    variable of a dataset schema

    A data variable whose schema allows multiple versions (see
    ``allow_multiple_versions``) is matched by its canonical name and by the
    canonical name followed by a ``"_"`` separated suffix: ``"SKY"`` matches
    ``"SKY"`` and ``"SKY_DECONVOLVED"``, but not ``"FLAG_SKY"``. Other data
    variables are matched by their canonical name only.

    :param data_var_names: Names of the data variables present
    :param data_var_schema: Schema of the data variable, an entry of
       :py:attr:`~xradio.schema.metamodel.DatasetSchema.data_vars`
    :returns: Names of the present data variables that the schema applies to
    """

    # The implementation stays in xradio.schema.check, which uses it and
    # cannot import this module
    return _matching_data_var_names(data_var_names, data_var_schema)


def _data_group_schema(schema: metamodel.DatasetSchema) -> metamodel.DictSchema:
    """
    The schema of the data groups of a dataset schema: that of the ``base``
    entry of its ``data_groups`` attribute, which check_dataset checks

    :param schema: Dataset schema
    :returns: Dictionary schema of a data group
    :raises TypeError: If the schema has no such data group schema
    """

    for attr in schema.attributes:
        if attr.name != "data_groups" or attr.dict_schema is None:
            continue
        for group_attr in attr.dict_schema.attributes:
            if group_attr.name == "base" and group_attr.dict_schema is not None:
                return group_attr.dict_schema
    raise TypeError(
        f"check_data_groups: dataset schema {schema.schema_name} has no "
        "'data_groups' attribute with a 'base' data group schema!"
    )


def _check_referenced_variable(
    dataset: xarray.Dataset, variable_name: str, array_schema: type, checked: dict
) -> SchemaIssues:
    """
    Check an existing data variable against an array schema, unless it has
    already been checked against that schema

    :param dataset: Dataset
    :param variable_name: Name of a data variable of ``dataset``
    :param array_schema: Array schema class the variable must conform to
    :param checked: Names of the variables checked so far, by schema name
        (updated)
    :returns: :py:class:`~xradio.schema.check.SchemaIssues` found
    """

    names = checked.setdefault(
        xarray_dataclass_to_array_schema(array_schema).schema_name, set()
    )
    if variable_name in names:
        return SchemaIssues()
    names.add(variable_name)
    return check_array(dataset[variable_name], array_schema).at_path(
        "data_vars", variable_name
    )


def check_data_groups(
    dataset: xarray.Dataset,
    schema: type | metamodel.DatasetSchema,
    role_schemas: typing.Mapping[str, type],
    *,
    checked: dict[str, set[str]] | None = None,
    report_unknown_roles: bool = True,
) -> SchemaIssues:
    """
    Check every data group of a dataset, and the data variables that their
    roles name

    This complements the structural checks of
    :py:func:`~xradio.schema.check.check_dataset`, which check only the
    ``base`` data group, against the data group schema, and resolve no names
    (for the main datasets of the measurement set v4 schema, check_dataset
    runs this function too):

    * every data group is a dictionary (a ``base`` data group that is not
      one is reported both here and by check_dataset);
    * every data group other than ``base``, which check_dataset checks,
      conforms to the data group schema of ``schema`` (the schema of its
      ``base`` data group);
    * if ``report_unknown_roles`` is set, a key that the data group schema
      does not have is reported;
    * every role of ``role_schemas`` that a data group gives names an
      existing data variable that conforms to the array schema of the role.

    Other keys, such as ``description`` and ``date``, are not resolved. A
    defect of a data variable is reported once, under the name of the
    variable: each variable is checked once per array schema, and not at all
    if it is in ``checked``. A role that names a missing data variable is
    reported, unless check_dataset reports that variable as a missing
    required data variable of ``schema``.

    :param dataset: Dataset to check
    :param schema: Dataset schema of ``dataset`` (a class decorated with
       :py:func:`~xradio.schema.bases.xarray_dataset_schema` or a
       :py:class:`~xradio.schema.metamodel.DatasetSchema`), which has a
       ``data_groups`` attribute with a ``base`` data group
    :param role_schemas: Array schema classes of the data variables that
       roles name, by role
    :param checked: Names of the data variables checked so far against each
       array schema, by schema name (updated). Defaults to the variables that
       check_dataset checks against ``schema``
       (see :py:func:`matching_data_var_names`)
    :param report_unknown_roles: Whether to report keys that the data group
       schema does not have
    :returns: :py:class:`~xradio.schema.check.SchemaIssues` found
    :raises TypeError: If ``schema`` is not a dataset schema, or has no
       ``data_groups`` attribute with a ``base`` data group schema
    """

    if bases.is_dataset_schema(schema):
        schema = xarray_dataclass_to_dataset_schema(schema)
    if not isinstance(schema, metamodel.DatasetSchema):
        raise TypeError(
            "check_data_groups: "
            + _not_a_dataset_schema_message(schema).removeprefix("check_dataset: ")
        )
    group_schema = _data_group_schema(schema)
    known_keys = {attr.name for attr in group_schema.attributes}

    # Missing or wrong-typed data_groups is reported by check_dataset
    data_groups = dataset.attrs.get("data_groups")
    if not isinstance(data_groups, dict):
        return SchemaIssues()

    if checked is None:
        checked = {}
        for data_var_schema in schema.data_vars:
            checked.setdefault(data_var_schema.schema_name, set()).update(
                matching_data_var_names(dataset.data_vars, data_var_schema)
            )

    # The required data variables that check_dataset reports as missing: a
    # role that names one is not reported again
    missing_required = {
        data_var_schema.name
        for data_var_schema in schema.data_vars
        if not data_var_schema.optional
        and not matching_data_var_names(dataset.data_vars, data_var_schema)
    }

    issues = SchemaIssues()
    for group_name, group in data_groups.items():
        if not isinstance(group, dict):
            issues.add(
                SchemaIssue(
                    path=[("attrs", "data_groups")],
                    message=f"Data group '{group_name}' is not a dictionary!",
                    found=type(group),
                    expected=[dict],
                )
            )
            continue

        # check_dataset checks the "base" group (through the attribute
        # schema of data_groups): check the others here
        if group_name != "base":
            issues += check_dict(group, group_schema).at_path(
                "attrs", f"data_groups['{group_name}']"
            )

        for role, variable_name in group.items():
            if role not in known_keys:
                if report_unknown_roles:
                    issues.add(
                        SchemaIssue(
                            path=[("attrs", f"data_groups['{group_name}']")],
                            message=f"Unknown data group role '{role}'!",
                            found=role,
                            expected=sorted(known_keys),
                        )
                    )
                continue
            role_schema = role_schemas.get(role)
            # A value of the wrong type has been reported by check_dict
            if role_schema is None or not isinstance(variable_name, str):
                continue
            if variable_name not in dataset.data_vars:
                if variable_name not in missing_required:
                    issues.add(
                        SchemaIssue(
                            path=[("data_vars", variable_name)],
                            message=f"Data group '{group_name}' role '{role}' "
                            "references a missing data variable!",
                            found=None,
                            expected=[variable_name],
                        )
                    )
                continue
            issues += _check_referenced_variable(
                dataset, variable_name, role_schema, checked
            )

    return issues


def _semantic_version(version) -> tuple[int, int, int] | None:
    """
    Parse a semantic version

    :param version: A ``schema_version`` attribute value
    :returns: The (MAJOR, MINOR, PATCH) numbers, or None when ``version`` is
       not a string of the form MAJOR.MINOR.PATCH (surrounding spaces allowed)
    """
    if not isinstance(version, str):
        return None
    match = _SEMANTIC_VERSION.fullmatch(version.strip())
    if match is None:
        return None
    major, minor, patch = (int(number) for number in match.groups())
    return major, minor, patch


def check_schema_version(dataset: xarray.Dataset, current_version: str) -> SchemaIssues:
    """
    Check the ``schema_version`` attribute of a dataset against the version of
    its schema in this xradio

    Reports a version that is not a semantic version (MAJOR.MINOR.PATCH), and
    one newer than ``current_version``: this xradio cannot know what a later
    version of the schema changed. A missing ``schema_version``, or one that
    is not a string, is left to :py:func:`~xradio.schema.check.check_dataset`
    (schemas declare it as a required ``str`` attribute).

    :param dataset: Dataset to check
    :param current_version: Version of the dataset's schema in this xradio,
       for example ``xradio.image.schema.IMAGE_SCHEMA_VERSION``
    :returns: :py:class:`~xradio.schema.check.SchemaIssues` found
    :raises ValueError: If ``current_version`` is not a semantic version
    """

    current = _semantic_version(current_version)
    if current is None:
        raise ValueError(
            f"check_schema_version: {current_version!r} is not a semantic version!"
        )
    version = dataset.attrs.get("schema_version")
    if not isinstance(version, str):
        return SchemaIssues()
    found = _semantic_version(version)
    if found is None:
        message = "Schema version is not a semantic version (MAJOR.MINOR.PATCH)!"
    elif found > current:
        message = (
            f"Schema version {version} is newer than {current_version}, the "
            "version of the schema in this xradio: update xradio to check the "
            "dataset!"
        )
    else:
        return SchemaIssues()
    return SchemaIssues(
        [
            SchemaIssue(
                path=[("attrs", "schema_version")],
                message=message,
                found=version,
                expected=[current_version],
            )
        ]
    )
