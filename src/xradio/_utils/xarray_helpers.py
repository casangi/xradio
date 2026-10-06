import warnings
from collections.abc import Callable, Iterable, Mapping

import xarray as xr
from xarray.core.extensions import AccessorRegistrationWarning

from xradio._utils.schema import get_data_group_keys


def get_data_group_name(
    xdx: xr.Dataset | xr.DataTree, data_group_name: str = None
) -> str:
    if data_group_name is None:
        if "base" in xdx.attrs["data_groups"]:
            data_group_name = "base"
        else:
            data_group_name = list(xdx.attrs["data_groups"].keys())[0]

    return data_group_name


def create_new_data_group(
    xdx: xr.Dataset | xr.DataTree,
    schema_name: str,
    new_data_group_name: str,
    data_group: dict,
    data_group_dv_shared_with: str = None,
) -> tuple[str, dict]:
    """Adds a data group to Xarray Data Structure (Dataset or DataTree).

    Parameters
    ----------
    new_data_group_name : str
        The name of the new data group to add.
    data_group : dict
        A dictionary containing the data group variables and their attributes.
    data_group_dv_shared_with : str, optional
        The name of the data group to share data variables with, by default "base"

    Returns
    -------
    tuple[str, dict]
        A tuple containing the name of the new data group and the dictionary with its variables and attributes.
    """

    data_group_dv_shared_with = get_data_group_name(xdx, data_group_dv_shared_with)

    default_data_group = xdx.attrs["data_groups"][data_group_dv_shared_with]

    new_data_group = {}

    data_group_keys = get_data_group_keys(schema_name)

    for key, optional in data_group_keys.items():
        if key in data_group:
            new_data_group[key] = data_group[key]
        else:
            if key not in default_data_group and not optional:
                raise ValueError(
                    f"Data group key '{key}' is required but not provided and not present in shared data group '{data_group_dv_shared_with}'."
                )
            elif key in default_data_group:
                new_data_group[key] = default_data_group[key]

    return new_data_group_name, new_data_group


def _names_any_of(value, names: set) -> bool:
    """True if a data group value is one of ``names``; unhashable values
    (never variable names) are not."""
    try:
        return value in names
    except TypeError:
        return False


def data_groups_without_variables(data_groups: Mapping, variables: Iterable) -> dict:
    """Return a copy of ``data_groups`` without the roles that refer to
    ``variables``.

    Data groups map roles (e.g. ``"sky"``) to variable names (e.g. ``"SKY"``),
    so every role whose value is one of ``variables`` is left out. The input
    mapping and its group dicts are not modified: they are typically shared
    with other datasets (see :func:`replace_data_groups`).

    Parameters
    ----------
    data_groups : Mapping
        The ``data_groups`` attribute, a mapping of group name to group dict.
    variables : iterable of str
        Names of the variables that no longer exist.

    Returns
    -------
    dict
        New data groups with new group dicts.
    """
    names = set(variables)
    return {
        group_name: (
            {
                role: value
                for role, value in group.items()
                if not _names_any_of(value, names)
            }
            if isinstance(group, Mapping)
            else group
        )
        for group_name, group in data_groups.items()
    }


def replace_data_groups(xdx: xr.Dataset | xr.DataTree, data_groups: dict) -> None:
    """Set the ``data_groups`` attribute of ``xdx`` by replacing its attrs
    mapping.

    Derived datasets share attrs with the dataset they come from: ``isel``,
    ``sel`` and ``load`` return datasets whose attrs dict IS the parent's
    attrs dict, and ``copy(deep=False)``, ``where``, ``chunk``, ``compute``
    and ``drop_vars`` share the nested ``data_groups`` and group dicts. Writing
    ``xdx.attrs["data_groups"][...]`` (or into a group dict) would therefore
    also change those other datasets. Assigning a new attrs mapping changes
    ``xdx`` only.

    Parameters
    ----------
    xdx : xr.Dataset or xr.DataTree
        The dataset or DataTree node to update in place.
    data_groups : dict
        The new data groups (not shared with any other object).
    """
    xdx.attrs = {**xdx.attrs, "data_groups": data_groups}


def remove_variables_from_data_groups(
    xdx: xr.Dataset | xr.DataTree, variables: Iterable
) -> None:
    """Remove the data group roles that refer to ``variables`` from ``xdx``,
    without modifying data group dicts shared with other datasets.

    Parameters
    ----------
    xdx : xr.Dataset or xr.DataTree
        The dataset or DataTree node to update in place. Nothing happens when
        it has no ``data_groups`` mapping.
    variables : iterable of str
        Names of the variables that no longer exist.
    """
    data_groups = xdx.attrs.get("data_groups")
    if isinstance(data_groups, Mapping):
        replace_data_groups(xdx, data_groups_without_variables(data_groups, variables))


#: Data variable attributes that name another data variable (the image
#: schema's ``flag`` and ``beam_fit_params`` attributes).
VARIABLE_REFERENCE_ATTRS = ("flag", "beam_fit_params")


def remove_variable_references(
    xdx: xr.Dataset | xr.DataTree, variables: Iterable
) -> None:
    """Remove the data variable attributes that name one of ``variables``
    (see :data:`VARIABLE_REFERENCE_ATTRS`) from ``xdx``.

    A changed variable is replaced by a shallow copy with new attrs, because
    variable objects (and their attrs dicts) can be shared with other
    datasets: ``drop_vars``, for example, keeps the same variable objects.

    Parameters
    ----------
    xdx : xr.Dataset or xr.DataTree
        The dataset or DataTree node to update in place.
    variables : iterable of str
        Names of the variables that no longer exist.
    """
    names = set(variables)
    for name in list(xdx.data_vars):
        variable = xdx[name].variable
        stale = [
            key
            for key in VARIABLE_REFERENCE_ATTRS
            if _names_any_of(variable.attrs.get(key), names)
        ]
        if stale:
            replacement = variable.copy(deep=False)
            replacement.attrs = {
                key: value for key, value in variable.attrs.items() if key not in stale
            }
            xdx[name] = replacement


def delete_data_variables(xdx: xr.Dataset | xr.DataTree, variables: list):
    """Deletes data variables from an Xarray Dataset or DataTree.

    The variables are deleted from ``xdx`` in place, and every data group role
    and variable attribute (see :data:`VARIABLE_REFERENCE_ATTRS`) that refers
    to one of them is removed. Datasets derived from ``xdx`` (or that ``xdx``
    was derived from) keep their variables, attributes and data groups: the
    data groups are rebuilt and the attrs mappings replaced, never mutated.

    Parameters
    ----------
    xdx : Union[xr.Dataset, xr.DataTree]
        The Xarray Dataset or DataTree to delete data variables from.
    variables : list
        A list of variable names to delete (a single name may be given as a
        string).

    Raises
    ------
    ValueError
        If a variable is not a data variable of ``xdx``. Nothing is deleted
        in that case.
    """
    if isinstance(variables, str):
        variables = [variables]
    variables = list(variables)
    for var in variables:
        if var not in xdx.data_vars:
            raise ValueError(f"Variable '{var}' not found in the dataset.")

    for var in variables:
        if var in xdx.data_vars:
            del xdx[var]

    remove_variables_from_data_groups(xdx, variables)
    remove_variable_references(xdx, variables)


class _UncachedAccessor:
    """Accessor descriptor that builds a new accessor on every access.

    xarray's ``register_dataset_accessor`` and ``register_datatree_accessor``
    cache the accessor instance on the object (``obj._cache[name]``), so an
    accessor that holds its object strongly forms the reference cycle
    ``obj -> _cache -> accessor -> obj``, which keeps the object and all its
    arrays alive until a full garbage collection. This descriptor never
    stores the accessor on the object, so the accessor can hold its object
    strongly: chained calls on temporaries (``ds.isel(...).xr_img.method()``)
    work, and the object still dies by reference counting.
    """

    def __init__(self, name: str, accessor: type):
        self._name = name
        self._accessor = accessor

    def __get__(self, obj, cls):
        if obj is None:
            # Class attribute access, e.g. xr.Dataset.xr_img (documentation tools)
            return self._accessor
        try:
            return self._accessor(obj)
        except AttributeError as err:
            # The object's __getattr__ would swallow an AttributeError raised
            # while building the accessor and report a missing attribute
            # instead (same handling as xarray's own accessor descriptor).
            raise RuntimeError(f"error initializing {self._name!r} accessor.") from err


def register_uncached_accessor(name: str, cls: type) -> Callable[[type], type]:
    """Register a non-cached accessor on an xarray class.

    Use instead of ``xr.register_dataset_accessor`` /
    ``xr.register_datatree_accessor`` for accessors that hold their object
    strongly (see :class:`_UncachedAccessor`). Every attribute access returns
    a new accessor instance, so accessors should keep no state of their own
    between calls.

    Parameters
    ----------
    name : str
        Attribute name of the accessor, e.g. ``"xr_img"``.
    cls : type
        The xarray class, ``xr.Dataset`` or ``xr.DataTree``.

    Returns
    -------
    Callable
        Class decorator registering the accessor class and returning it
        unchanged.
    """

    def decorator(accessor: type) -> type:
        existing = cls.__dict__.get(name)
        reregistration = (
            isinstance(existing, _UncachedAccessor)
            and existing._accessor.__qualname__ == accessor.__qualname__
            and existing._accessor.__module__ == accessor.__module__
        )
        if hasattr(cls, name) and not reregistration:
            warnings.warn(
                f"registration of accessor {accessor!r} under name {name!r} for "
                f"type {cls!r} is overriding a preexisting attribute with the "
                "same name.",
                AccessorRegistrationWarning,
                stacklevel=2,
            )
        setattr(cls, name, _UncachedAccessor(name, accessor))
        return accessor

    return decorator
