from . import schema_table


def setup(app):
    app.add_directive(
        "xradio_array_schema_table", schema_table.ArraySchemaTableDirective
    )
    app.add_directive(
        "xradio_dataset_schema_table", schema_table.DatasetSchemaTableDirective
    )
    app.add_directive("xradio_dict_schema_table", schema_table.DictSchemaTableDirective)
    return {"version": "0.1.0"}
