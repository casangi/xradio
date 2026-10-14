"""The shared schema building blocks of ``xradio.schema.measures`` were moved
there from ``xradio.measurement_set.schema``, which re-exports them for
backwards compatibility. These tests pin the re-exports: every public name of
``xradio.schema.measures`` must be importable from
``xradio.measurement_set.schema`` and be the same object.
"""

import ast
import inspect

import pytest

import xradio.measurement_set.schema as ms_schema
import xradio.schema.measures as measures


def _public_names(module) -> list[str]:
    """Names that a module defines (rather than imports) at top level and
    that do not start with an underscore."""

    names = []
    for node in ast.parse(inspect.getsource(module)).body:
        if isinstance(node, ast.ClassDef | ast.FunctionDef):
            names.append(node.name)
        elif isinstance(node, ast.Assign):
            names += [
                target.id for target in node.targets if isinstance(target, ast.Name)
            ]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.append(node.target.id)
    return [name for name in names if not name.startswith("_")]


PUBLIC_MEASURES = _public_names(measures)


def test_public_names_are_found():
    # Guard the scan itself: the 41 names moved from the measurement set
    # schema include dimension literals, vocabularies and array schemas
    assert len(PUBLIC_MEASURES) >= 41
    for name in ("Time", "ZD", "AllowedSpectralCoordFrames", "SpectralCoordArray"):
        assert name in PUBLIC_MEASURES


@pytest.mark.parametrize("name", PUBLIC_MEASURES)
def test_reexported_from_measurement_set_schema(name):
    assert hasattr(ms_schema, name), (
        f"xradio.measurement_set.schema no longer re-exports {name} from "
        "xradio.schema.measures"
    )
    assert getattr(ms_schema, name) is getattr(measures, name)
