import pytest


## Tests for _utils/_ms/subtables functions. Move to its own file
def test_subtables_subt_rename_ids():
    from xradio.measurement_set._utils._msv2.subtables import subt_rename_ids

    for key, val in subt_rename_ids.items():
        assert isinstance(val, dict)
