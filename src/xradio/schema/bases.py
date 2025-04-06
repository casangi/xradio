import xarray
import inspect
from . import dataclass, check, metamodel, typing
import numpy
import dataclasses


def _guess_dtype(obj: typing.Any):
    try:
        return _guess_dtype(next(iter(obj)))
    except TypeError:
        return numpy.dtype(type(obj))


def _set_parameter(
    val: typing.Any, args: dict, schema: typing.Union["AttrSchemaRef", "ArraySchemaRef"]
):
    """
    Extract given entry from parameters - while taking care that the
    parameter value might have been set either before or after, and that
    defaults might apply.

    :param val: Value from xarray-constructor style ("data_vars"/"coords")
    :param args: Bound arguments to constructor (positional or named)
    :param schema: Schema of argument (either attribute or array)
    :returns: Updated value
    """

    # If value appears in named parameters, overwrite
    if args.get(schema.name) is not None:
        if val is not None:
            raise ValueError(
                f"Parameter {schema.name} was passed twice ({val} vs {args[schema.name]})!"
            )
        val = args[schema.name]

    # Otherwise apply defaults *if* it doesn't exist already or deactivate
    # (typically because we are construting from a dataset/data array)
    if val is None and schema.default is not dataclasses.MISSING:
        default = schema.default
        if default is not None:
            val = default

    return val


def _np_convert(val: typing.Any, schema: metamodel.ArraySchemaRef):
    """
    Convert value to numpy, if appropriate

    This attempts to catch "early" conversions that we can do more
    appropriately than xarray because we have more information from the schema.
    Specifically, if it's a type where the dtype to choose is somewhat
    ambiguous, we can use this chance to "bias" it towards an allowed one.

    :param val: Received value
    :param schema: Execpted array schema
    :returns: Possibly converted value
    """

    # Array schema refs that are not yet a numpy or xarray data type?
    if isinstance(val, list) or isinstance(val, tuple) and isinstance(val[1], list):
        # Check whether we can "guess" the dtype from the object
        dtype = None
        if len(schema.dtypes) > 1:
            guessed = _guess_dtype(val)
            for dt in schema.dtypes:
                # Actually look for closest in precision etc?
                if dt == guessed:
                    dtype = dt
                    break

        # Otherwise just use the first one
        if dtype is None:
            dtype = schema.dtypes[0]

        # Attempt conversation
        try:
            if isinstance(val, list):
                val = numpy.array(val, dtype=dtype)
            else:
                val = tuple([val[0], numpy.array(val[1], dtype=dtype), *val[2:]])

        except TypeError:
            pass

    return val


def _dataarray_new(
    cls,
    data=None,
    *args,
    coords=None,
    dims=None,
    name=None,
    attrs=None,
    indexes=None,
    **kwargs,
):
    # Convert parameters
    if coords is not None and isinstance(coords, list):
        coords = dict(coords)
    if coords is None:
        coords = {}
    if attrs is None:
        attrs = {}

    # Get signature of __init__, map parameters and apply defaults. This
    # will raise an exception if there are any extra parameters.
    sig = inspect.Signature.from_callable(cls.__init__)
    sig = sig.replace(parameters=[v for k, v in sig.parameters.items() if k != "self"])
    mapping = sig.bind_partial(data, *args, **kwargs)

    # Check whether we have a "data" argument now. This happens if we pass
    # it as a positional argument.
    if mapping.arguments.get("data") is not None:
        data = mapping.arguments["data"]

    # No dims specified? Select one matching the data dimensionality from
    # the schema
    schema = dataclass.xarray_dataclass_to_array_schema(cls)
    data = _np_convert(data, schema)
    for schema_dims in schema.dimensions:
        if len(schema_dims) == len(data.shape):
            dims = schema_dims
            break

    # If we are constructing from a data array / variable, take over attributes
    if isinstance(data, (xarray.DataArray, xarray.Variable)):
        for attr, attr_val in data.attrs.items():
            # Explicit parameters take precedence though
            if attr not in attrs:
                attrs[attr] = attr_val

    # Get any coordinates or attributes and add them to the appropriate lists
    for coord in schema.coordinates:
        val = _np_convert(
            _set_parameter(coords.get(coord.name), mapping.arguments, coord), coord
        )

        # Default to simple range of specified dtype if part of dimensions
        # (that's roughly the behaviour of the xarray constructor as well)
        if val is None and dims is not None:
            dim_ix = dims.index(coord.name)
            if dim_ix is not None and dim_ix < len(data.shape):
                dtype = coord.dtypes[0]
                val = numpy.arange(data.shape[dim_ix], dtype=dtype)

        if val is not None:
            coords[coord.name] = val
    for attr in schema.attributes:
        val = _set_parameter(attrs.get(attr.name), mapping.arguments, attr)
        if val is not None:
            attrs[attr.name] = val

    # Redirect to xradio.DataArray constructor
    instance = xarray.DataArray(data, coords, dims, name, attrs, indexes)

    # Perform schema check
    check.check_array(instance, schema).expect()
    return instance


def xarray_dataarray_schema(cls):
    """Decorator for classes representing :py:class:`xarray.DataArray` schemas.
    The annotated class should exactly contain:

    * one field called "``data``" annotated with :py:data:`~typing.Data`
      to indicate the array type
    * fields annotated with :py:data:`~typing.Coord` to indicate mappings of
      dimensions to coordinates (coordinates directly associated with dimensions
      should have the same name as the dimension)
    * fields annotated with :py:data:`~typing.Attr` to declare attributes

    Decorated schema classes can be used with
    :py:func:`~xradio.schema.check.check_array` for checking
    :py:class:`xarray.DataArray` objects against the schema. Furthermore, the
    class constructor will be overwritten to generate schema-confirming
    :py:class:`xarray.DataArray` objects.

    For example::

        from xradio.schema import xarray_dataarray_schema
        from xradio.schema.typing import Data, Coord, Attr
        from typing import Optional, Literal
        import dataclasses

        Coo = Literal["coo"]

        @xarray_dataarray_schema
        class TestArray:
            data: Data[Coo, complex]
            coo: Coord[Coo, float]
            attr1: Attr[str]
            attr2: Attr[int] = 123
            attr3: Optional[Attr[int]] = None

    This data class represents a one-dimensional :py:class:`xarray.DataArray`
    with complex data, a ``float`` coordinate and three attributes. Instances of
    this class cannot actually be constructed, instead you will get an appropriate
    :py:class:`xarray.DataArray` object::

        >>> TestArray(data=[1,2,3], attr1="foo")
        <xarray.DataArray (coo: 3)>
        array([1.+0.j, 2.+0.j, 3.+0.j])
        Coordinates:
          * coo      (coo) float64 0.0 1.0 2.0
        Attributes:
            attr1:    foo
            attr2:    123

    Note that:

    * The constructor uses the annotations to identify the role of every parameter
    * The data was automatically converted into a :py:class:`numpy.ndarray`
    * As there was no coordinate given, it was automatically filled with an
      enumeration of the type specified in the annotation
    * Default attribute values were assigned. A value of `None` is interpreted
      as the value attribute being missing.
    * For the returned :py:class:`~xarray.DataArray` object ``data``, ``coo``,
      ``attr1`` and ``attr2`` can be accessed as if they were members. This works
      as long as the names don't collide with :py:class:`~xarray.DataArray` members.

    Positional parameters are also supported, and ``coords`` and ``attrs`` passed as
    keyword arguments can supply additional coordinates and attributes::

        >>> TestArray([1,2,3], [3,4,5], 'bar', coords={'coo_new': ('coo', [3,2,1])}, attrs={'xattr': 'baz'})
        <xarray.DataArray (coo: 3)>
        array([1.+0.j, 2.+0.j, 3.+0.j])
        Coordinates:
            coo_new  (coo) int64 3 2 1
          * coo      (coo) float64 3.0 4.0 5.0
        Attributes:
            xattr:    baz
            attr1:    bar
            attr2:    123

    """

    # Make into a dataclass (might not even be needed at some point?)
    cls = dataclasses.dataclass(cls, init=True, repr=False, eq=False, frozen=True)

    # Make schema
    cls.__xradio_array_schema = dataclass.xarray_dataclass_to_array_schema(cls)

    # Replace __new__
    cls.__new__ = _dataarray_new

    return cls


def is_dataarray_schema(val: typing.Any):
    return type(val) == type and hasattr(val, "__xradio_array_schema")


class AsDataArray:
    __new__ = _dataarray_new


def _dataset_new(cls, *args, data_vars=None, coords=None, attrs=None, **kwargs):
    # Get standard xarray parameters (data_vars, coords, attrs)
    # Note that we only support these as keyword arguments for now
    if data_vars is None:
        data_vars = {}
    if coords is None:
        coords = {}
    if attrs is None:
        attrs = {}

    # Get signature of __init__, map parameters and apply defaults. This
    # will raise an exception if there are any extra parameters.
    sig = inspect.Signature.from_callable(cls.__init__)
    sig = sig.replace(parameters=[v for k, v in sig.parameters.items() if k != "self"])
    mapping = sig.bind_partial(*args, **kwargs)

    # Now get schema for this class and identify which of the parameters
    # where meant to be data variables, coordinates and attributes
    # respectively. Note that we interpret "None" as "missing" here, so
    # setting an attribute to `None` will require passing them as
    # attrs.
    schema = dataclass.xarray_dataclass_to_dataset_schema(cls)
    for coord in schema.coordinates:
        val = _np_convert(
            _set_parameter(coords.get(coord.name), mapping.arguments, coord), coord
        )
        # Determine dimensions / convert to Variable
        if (
            val is not None
            and not isinstance(val, xarray.DataArray)
            and not isinstance(val, xarray.Variable)
            and not isinstance(val, tuple)
        ):
            default_attrs = {
                attr.name: attr.default
                for attr in coord.attributes
                if attr.default is not None
            }
            for dims in coord.dimensions:
                if len(dims) == len(val.shape):
                    val = xarray.Variable(dims, val, default_attrs)
                    break
        if val is not None:
            coords[coord.name] = val
    for data_var in schema.data_vars:
        val = _set_parameter(data_vars.get(data_var.name), mapping.arguments, data_var)

        # Determine dimensions / convert to Variable
        dims = None
        if val is None:
            dims = None
        elif isinstance(val, xarray.Variable):
            dims = val.dims
        elif isinstance(val, xarray.DataArray):
            val = val.variable
            dims = val.dims
        elif isinstance(val, tuple):
            val = xarray.Variable(*val)
            dims = val.dims
        else:
            # We are dealing with a plain value. Try to convert it to numpy first
            val = _np_convert(val, data_var)

            # Then identify dimensions by matching against dimensionality
            dims = None
            for ds in data_var.dimensions:
                if len(ds) == len(val.shape):
                    dims = ds
                    break
            if dims is None:
                options = ["[" + dims.join(",") + "]" for dims in data_var.dimensions]
                raise ValueError(
                    f"Data variable {data_var.name} shape has {len(dims)} dimensions,"
                    f" expected {' or '.join(options)}!"
                )

            # Get default attributes
            default_attrs = {
                attr.name: attr.default
                for attr in data_var.attributes
                if attr.default is not dataclasses.MISSING
            }

            # Replace by variable
            val = xarray.Variable(dims, val, default_attrs)

        # Default coordinates used by this data variable to numpy arange. We
        # can only do this now because we need an example to determine the
        # intended size of the coordinate
        if dims is not None:
            for coord in schema.coordinates:
                if coord.name in dims and coords.get(coord.name) is None:
                    dim_ix = dims.index(coord.name)
                    if dim_ix is not None and dim_ix < len(val.shape):
                        dtype = coord.dtypes[0]
                        if numpy.issubdtype(dtype, numpy.number):
                            coords[coord.name] = numpy.arange(
                                val.shape[dim_ix], dtype=dtype
                            )

        if val is not None:
            data_vars[data_var.name] = val

    for attr in schema.attributes:
        val = _set_parameter(attrs.get(attr.name), mapping.arguments, attr)
        if val is not None:
            attrs[attr.name] = val

    # Redirect to xradio.Dataset constructor
    instance = xarray.Dataset(data_vars, coords, attrs)

    # Finally check schema
    check.check_dataset(instance, schema).expect()

    return instance


def xarray_dataset_schema(cls):
    """Decorator for classes representing :py:class:`xarray.Dataset` schemas.
    The annotated class should exactly contain:

    * fields annotated with :py:data:`~typing.Coord` to indicate mappings of
      dimensions to coordinates (coordinates directly associated with dimensions
      should have the same name as the dimension)
    * fields annotated with :py:data:`~typing.Data`
      to indicate data variables
    * fields annotated with :py:data:`~typing.Attr` to declare attributes

    Decorated schema classes can be used with
    :py:func:`~xradio.schema.check.check_dataset` for checking
    :py:class:`xarray.Dataset` objects against the schema. Furthermore, the
    class constructor will be overwritten to generate schema-confirming
    :py:class:`xarray.Dataset` objects.
    """

    # Make into a dataclass (might not even be needed at some point?)
    cls = dataclasses.dataclass(cls, init=True, repr=False, eq=False, frozen=True)

    # Make schema
    schema = dataclass.xarray_dataclass_to_dataset_schema(cls)
    cls.__xradio_dataset_schema = schema

    # Replace __new__
    cls.__new__ = _dataset_new

    # Register type
    check.register_dataset_type(schema)

    return cls


def is_dataset_schema(val: typing.Any):
    return type(val) == type and hasattr(val, "__xradio_dataset_schema")


class AsDataset:
    """Mix-in class to indicate dataset data classes

    Deprecated - use decorator :py:func:`xarray_dataset_schema` instead
    """

    __new__ = _dataset_new


def _dict_new(cls, *args, **kwargs):
    # Get signature of __init__, map parameters and apply defaults. This
    # will raise an exception if there are any extra parameters.
    sig = inspect.Signature.from_callable(cls.__init__)
    sig = sig.replace(parameters=[v for k, v in sig.parameters.items() if k != "self"])
    mapping = sig.bind_partial(*args, **kwargs)
    mapping.apply_defaults()

    # The dictionary is now simply the arguments. Note that this means that
    # in contrast to the behaviour of AsDataset/AsDataarray, for
    # dictionaries we actually interpret a default of "None" as setting the
    # value in question to "None".
    instance = mapping.arguments

    # Check schema
    check.check_dict(instance, cls).expect()
    return instance


def dict_schema(cls):
    """Decorator for classes representing ``dict`` schemas, along the lines
    of :py:func:`xarray_dataarray_schema` and :py:func:`xarray_dataset_schema`.

    The annotated class can contain fields with arbitrary annotations, similar
    to a dataclass. They can be used with
    :py:func:`~xradio.schema.check.check_dict` for checking dictionieries
    against the schema. Furthermore, the class constructor will be overwritten
    to generate schema-confirming :py:class:`xarray.Dataset` objects.
    """

    # Make into a dataclass (might not even be needed at some point?)
    cls = dataclasses.dataclass(cls, init=True, repr=False, eq=False, frozen=True)

    # Make schema
    cls.__xradio_dict_schema = dataclass.xarray_dataclass_to_dict_schema(cls)

    # Replace __new__
    cls.__new__ = _dict_new

    return cls


def is_dict_schema(val: typing.Any):
    return type(val) == type and hasattr(val, "__xradio_dict_schema")


class AsDict:
    """Mix-in class to indicate dictionary data classes

    Deprecated - use decorator :py:func:`dict_schema` instead
    """

    __new__ = _dict_new
