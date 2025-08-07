import json
import pytest
import pathlib

from xradio.measurement_set.schema import VisibilityXds, SpectrumXds
from xradio.schema.export import export_schema_json_file, import_schema_json_file


@pytest.mark.parametrize("schema", [VisibilityXds, SpectrumXds])
def test_schema_export_in_synch(tmp_path, schema):
    """
    Checks whether JSON schemas in the repository tree match
    the Python definitions.
    """

    # Export schema
    schema_fname = f"{schema.__name__}.json"
    export_schema_json_file(schema, tmp_path / schema_fname)
    with open(tmp_path / schema_fname, "r", encoding="utf8") as f:
        python_schema_json = json.load(f)

    # Load existing schema
    repository_root = pathlib.Path(__file__).parent.parent.parent.parent
    assert (
        repository_root / "schemas"
    ).is_dir(), "Schema directory doesn't exist in expected location"
    with open(repository_root / "schemas" / schema_fname, "r", encoding="utf8") as f:
        repo_schema_json = json.load(f)

    # Check that schemas are synchronised
    assert python_schema_json == repo_schema_json, (
        "Exported schemas not consistent with Python definitions! "
        "Run 'make schema-export' from repository root!"
    )
