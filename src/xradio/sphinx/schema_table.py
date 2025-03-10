import importlib
import dataclasses
import typing

from docutils import nodes, utils
from docutils.parsers.rst import Directive, DirectiveError
from docutils.parsers.rst import directives
from docutils.utils import SystemMessagePropagation
from docutils.statemachine import StringList

from sphinx.directives import ObjectDescription
from sphinx.util.docutils import switch_source_input

from xradio.schema import (
    xarray_dataclass_to_array_schema,
    xarray_dataclass_to_dataset_schema,
    xarray_dataclass_to_dict_schema,
)


class SchemaTableDirective(ObjectDescription):
    required_arguments = 1  # Xarray dataclass of schema
    has_content = False

    option_spec = {
        # "headers": directives.unchanged,
        # "widths": directives.unchanged,
        "title": directives.unchanged,
        # "columns": directives.unchanged,
        "class": directives.unchanged,
    }

    def run(self):
        # Import the referenced class
        klass_path = self.arguments[0].rsplit(".", 1)
        if len(klass_path) != 2:
            raise ValueError(
                f"Should be absolute Python name of xarray dataclass definition: {self.arguments[0]}"
            )
        klass_module = importlib.import_module(klass_path[0])
        klass = getattr(klass_module, klass_path[1])

        # Make table node
        classes = ["tbl", "colwidths-given"]
        if "class" in self.options:
            classes.append(self.options["class"])
        self._table = nodes.table(
            classes=classes, ids=[f"schema-table-{self.arguments[0]}"]
        )

        # Add title, if requested
        if "title" in self.options:
            self._table += nodes.title(text=caption)

        # Declare columns
        column_widths = [10, 10, 5, 5, 40]
        self._tgroup = nodes.tgroup(cols=len(column_widths))
        self._table += self._tgroup
        for colwidth in column_widths:
            self._tgroup += nodes.colspec(colwidth=colwidth)

        # Create head row
        header_row = nodes.row()
        header_row += nodes.entry("", nodes.paragraph(text=""))
        header_row += nodes.entry("", nodes.strong(text="Dimensions"))
        header_row += nodes.entry("", nodes.strong(text="Dtype"))
        header_row += nodes.entry("", nodes.strong(text="Model"))
        header_row += nodes.entry("", nodes.strong(text="Description"))
        self._thead = nodes.thead("", header_row)
        self._tgroup += self._thead

        # Add body
        self._tbody = nodes.tbody()
        self._tgroup += self._tbody

        # Add table contents (overridden in subclasses)
        self._add_table_contents(klass)

        # Register table
        # tbl = self.env.get_domain("tbl")
        # tbl.add_table(caption, table_id)

        return [self._table]

    def _add_section(self, name):
        # Create row
        row = nodes.row("", nodes.entry("", nodes.strong("", name), morecols=4))
        self._tbody += row

    def _add_row(
        self,
        name="",
        dimss=[],
        types=[],
        meta=None,
        descr="",
        optional=False,
        default=dataclasses.MISSING,
    ):
        # Create row
        row = nodes.row()
        self._tbody += row

        # Add name
        name_nds = [nodes.literal(text=name)]
        if optional:
            name_nds = [nodes.Text("(")] + name_nds + [nodes.Text(")")]
        row += nodes.entry("", *name_nds)

        # Add dimensions
        def mk_multi_entry(lines):
            if not lines:
                return nodes.entry()
            if len(lines) == 0:
                return nodes.entry(lines[0])
            else:
                entry = nodes.entry()
                for i, line in enumerate(lines):
                    entry += nodes.line("", "" if i == 0 else "or\xa0", line)
                return entry

        row += mk_multi_entry(
            [nodes.literal(text=f"[{','.join(dims)}]") for dims in dimss]
        )

        # Add types
        row += mk_multi_entry([nodes.literal(text=typ) for typ in types])

        # Add model link
        entry = nodes.entry()
        row += entry
        if meta is not None:
            # Preformatted? Just pass through
            if isinstance(meta, nodes.line):
                entry += meta
            else:
                vl = StringList()
                vl.append(f":py:class:`~{meta}`", "")
                with switch_source_input(self.state, vl):
                    self.state.nested_parse(vl, 0, entry)

        # Add description
        entry = nodes.entry()
        row += entry
        if descr:
            vl = StringList()
            vl.append(descr, "")
            with switch_source_input(self.state, vl):
                self.state.nested_parse(vl, 0, entry)
        if default is not dataclasses.MISSING:
            vl = StringList()
            vl.append(f"**Default:** ``{repr(default)}``", "")
            with switch_source_input(self.state, vl):
                self.state.nested_parse(vl, 0, entry)


def format_literals(typ):

    # a | b | c: Recurse and merge
    if typing.get_origin(typ) == typing.Union:
        type_args = typing.get_args(typ)
        options = []
        for arg in type_args:
            options += format_literals(arg)
        return options

    # Literal['a', 'b', ...]: Wrap into individual "literal" nodes
    if typing.get_origin(typ) == typing.Literal:
        return list(map(lambda t: nodes.literal(text=repr(t)), typing.get_args(typ)))

    # list[Literal['a'], Literal['b'], ...]: Format as one literal (compound) value
    if typing.get_origin(typ) == list:
        type_args = typing.get_args(typ)
        if any([typing.get_origin(arg) != typing.Literal for arg in type_args]):
            raise ValueError(f"List must contain only literals: {typ}")
        values = [repr(typing.get_args(val)[0]) for val in typing.get_args(typ)]
        return [nodes.literal(text=f"[{', '.join(values)}]")]

    raise ValueError(f"Must be either a type or a literal: {typ}")


def format_attr_model_text(state, attr) -> StringList:
    """
    Formats the text for the 'model' column in schema tables (arrays and datasets).
    Doesn't aim at supporting any literal types or combinations of types in general,
    but the following three ones specifically:

    - Literals (with multiple options (implicit Union of literals))
    - List of literals (e.g. ["rad","rad"]
    - Union of list of literals (e.g. ["m","m","m"]/["rad","rad","m"]

    This is meant to produce readable text listing literals as quoted text and
    their combinations, in schema attributes (particularly quantities and measures).

    Everything else than these expected literal based types would be printed as the
    type name.
    """

    type_args = typing.get_args(attr.typ)
    is_list_of_literals = typing.get_origin(attr.typ) is list and all(
        [typing.get_origin(arg) is typing.Literal for arg in type_args]
    )

    line = nodes.line()

    if not is_list_of_literals:
        # A type?
        if isinstance(attr.typ, type):
            vl = StringList()
            vl.append(f":py:class:`~{attr.typ.__module__}.{attr.typ.__name__}`", "")
            with switch_source_input(state, vl):
                state.nested_parse(vl, 0, line)
            return line

        if typing.get_origin(attr.typ) == typing.Union:
            vl = StringList()
            type_args = typing.get_args(attr.typ)
            options = []
            for i, arg in enumerate(type_args):
                vl.append(f":py:class:`~{arg.__module__}.{arg.__name__}`", "")
                if i + 1 < len(type_args):
                    vl.append(" or ", "")
            with switch_source_input(state, vl):
                state.nested_parse(vl, 0, line)
            return line

        # Derived type, e.g. list of types?
        if typing.get_origin(attr.typ) == list and all(
            [isinstance(arg, type) for arg in type_args]
        ):
            vl = StringList()
            vl.append("[", "")
            for i, arg in enumerate(typing.get_args(attr.typ)):
                if i > 0:
                    vl.append(", ", "")
                vl.append(f":py:class:`~{arg.__module__}.{arg.__name__}`", "")
            vl.append("]", "")
            with switch_source_input(state, vl):
                state.nested_parse(vl, 0, line)
            return line

    # Assume it's a literal of some kind - collect options
    literals = format_literals(attr.typ)
    for i, lit in enumerate(literals):
        if i > 0:
            if i + 1 >= len(literals):
                line += nodes.Text(" or\xa0")
            else:
                line += nodes.Text(", ")
        line += lit

    return line


class ArraySchemaTableDirective(SchemaTableDirective):
    def _add_table_contents(self, klass):
        # Extract schema
        schema = xarray_dataclass_to_array_schema(klass)

        # Add dataarray reference as first element
        self._add_row(
            "data",
            schema.dimensions,
            schema.dtypes,
            schema.schema_name,
            schema.data_docstring,
        )

        # Add coordinates
        if schema.coordinates:
            self._add_section("Coordinates:")
            for coord in schema.coordinates:
                self._add_row(
                    coord.name,
                    coord.dimensions,
                    coord.dtypes,
                    coord.schema_name,
                    coord.docstring or coord.data_docstring,
                    optional=coord.optional,
                    default=coord.default,
                )

        # Add attributes
        if schema.attributes:
            self._add_section("Attributes:")
            for attr in schema.attributes:
                model_text = format_attr_model_text(self.state, attr)
                self._add_row(
                    attr.name,
                    [],
                    [],
                    model_text,
                    attr.docstring,
                    optional=attr.optional,
                    default=attr.default,
                )


class DatasetSchemaTableDirective(SchemaTableDirective):
    def _add_table_contents(self, klass):
        # Extract schema
        schema = xarray_dataclass_to_dataset_schema(klass)

        # Add coordinates
        if schema.coordinates:
            self._add_section("Coordinates:")
            for coord in schema.coordinates:
                self._add_row(
                    coord.name,
                    coord.dimensions,
                    coord.dtypes,
                    coord.schema_name,
                    coord.docstring or coord.data_docstring,
                    optional=coord.optional,
                    default=coord.default,
                )

        # Add data variables
        if schema.data_vars:
            self._add_section("Data Variables:")
            for data_var in schema.data_vars:
                self._add_row(
                    data_var.name,
                    data_var.dimensions,
                    data_var.dtypes,
                    data_var.schema_name,
                    data_var.docstring or data_var.data_docstring,
                    optional=data_var.optional,
                    default=data_var.default,
                )

        # Add attributes
        if schema.attributes:
            self._add_section("Attributes:")
            for attr in schema.attributes:
                model_text = format_attr_model_text(self.state, attr)
                self._add_row(
                    attr.name,
                    [],
                    [],
                    model_text,
                    attr.docstring,
                    optional=attr.optional,
                    default=attr.default,
                )


class DictSchemaTableDirective(SchemaTableDirective):
    def _add_table_contents(self, klass):
        # Extract schema
        schema = xarray_dataclass_to_dict_schema(klass)

        # Add attributes
        if schema.attributes:
            self._add_section("Fields:")
            for attr in schema.attributes:
                self._add_row(
                    attr.name,
                    types=[f"{attr.typ.__name__}"],
                    optional=attr.optional,
                    descr=attr.docstring,
                    default=attr.default,
                )
