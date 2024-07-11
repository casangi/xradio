import pytest


def test_load_main_chunk_raises():
    from xradio.vis._vis_utils._ms.chunks import load_main_chunk

    name = "any"
    with pytest.raises(ValueError, match="unknown keys"):
        res = load_main_chunk(
            name, {"unk": range(0, 3), "baseline": range(0, 7), "time": range(0, 100)}
        )
