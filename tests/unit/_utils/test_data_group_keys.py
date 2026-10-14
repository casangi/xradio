import typing

import pytest

from xradio._utils.schema import get_data_group_keys


def _data_group_dict(schema_name):
    if schema_name == "msv4":
        from xradio.measurement_set.schema import DataGroupDict
    else:
        from xradio.image.schema import DataGroupDict
    return DataGroupDict


@pytest.mark.parametrize("schema_name", ["msv4", "image"])
def test_get_data_group_keys_lists_every_data_group_field(schema_name):
    """
    Verify the keys are the fields of the schema's DataGroupDict.

    Parameters
    ----------
    schema_name : str
        Schema whose data group keys are checked.

    Returns
    -------
    None
        This test validates the returned keys.
    """
    keys = get_data_group_keys(schema_name)
    assert set(keys) == set(typing.get_type_hints(_data_group_dict(schema_name)))


def test_get_data_group_keys_marks_optional_fields():
    """
    Verify optional MSv4 data group fields are marked optional.

    Parameters
    ----------
    None

    Returns
    -------
    None
        This test validates the optionality flags.
    """
    keys = get_data_group_keys("msv4")
    assert keys["uvw"] is True
    assert keys["correlated_data"] is False


@pytest.mark.parametrize("schema_name", ["msv4", "image"])
def test_get_data_group_keys_evaluates_type_hints_once(schema_name, monkeypatch):
    """
    Verify the type hints of a schema are evaluated at most once, not on
    every call.

    The DataGroupDict annotations are strings, so each evaluation compiles
    one new code object per annotation. An earlier test may already have
    evaluated them, so two calls evaluate them once or not at all.

    Parameters
    ----------
    schema_name : str
        Schema whose data group keys are requested.
    monkeypatch : pytest.MonkeyPatch
        Fixture used to count calls of ``typing.get_type_hints``.

    Returns
    -------
    None
        This test validates that later calls reuse the evaluated type hints.
    """
    evaluated = []
    get_type_hints = typing.get_type_hints

    def counting_get_type_hints(cls, *args, **kwargs):
        evaluated.append(cls)
        return get_type_hints(cls, *args, **kwargs)

    monkeypatch.setattr(typing, "get_type_hints", counting_get_type_hints)
    first = get_data_group_keys(schema_name)
    second = get_data_group_keys(schema_name)
    assert first == second
    assert evaluated in ([], [_data_group_dict(schema_name)])


def test_get_data_group_keys_rejects_unknown_schema():
    """
    Verify an unknown schema name raises ValueError.

    Parameters
    ----------
    None

    Returns
    -------
    None
        This test validates the error for an unknown schema.
    """
    with pytest.raises(ValueError, match="Unknown schema name"):
        get_data_group_keys("unknown")
